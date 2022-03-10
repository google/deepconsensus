# Copyright (c) 2021, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Utility functions being used for data processing."""

import collections
import dataclasses
import itertools
from typing import Any, Dict, List, Optional, Union
from absl import logging
import numpy as np
import pysam
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.utils import dc_constants

Issue = dc_constants.Issue


class SubreadGrouper(collections.Iterator):
  """Returns all subreads belonging to a single zmw as a list."""

  def __init__(self, subreads_to_ccs, reader_threads):
    self.bam_reader = pysam.AlignmentFile(
        subreads_to_ccs, check_sq=False, threads=reader_threads)
    self.keep_iter = True
    self.subread_group = []
    # Setup subread group.
    first_read = next(self.bam_reader)
    self.zmw = first_read.get_tag('zm')
    # Only add read if it is mapped.
    if not first_read.is_unmapped:
      self.subread_group.append(first_read)

  def __next__(self) -> List[pysam.libcalignedsegment.AlignedSegment]:
    if not self.keep_iter:
      raise StopIteration
    while self.keep_iter:
      try:
        read = next(self.bam_reader)
        if read.is_unmapped:
          continue
      except StopIteration:
        self.keep_iter = False
        break
      read_zmw = read.get_tag('zm')
      if read_zmw == self.zmw:
        self.subread_group.append(read)
      elif read_zmw != self.zmw:
        subreads_set = self.subread_group
        self.subread_group = [read]
        self.zmw = read_zmw
        if subreads_set:
          return subreads_set
    if self.subread_group:
      return self.subread_group
    else:
      raise StopIteration


def right_pad(arr: np.ndarray, length: int, value: Any) -> np.ndarray:
  """Right-pad an array with a given value.

  Args:
    arr: A numpy array (1 x n)
    length: The length of arr after padding.
    value: Pad value.

  Returns:
    A padded array


  """
  # This function does not check for valid padding lengths.
  pad_amt = length - len(arr)
  return np.pad(arr, (0, pad_amt), 'constant', constant_values=value)[:length]


@dataclasses.dataclass
class Read(collections.Sequence):
  """Used to represent ccs alignments."""
  name: str
  bases: np.ndarray
  cigar: np.ndarray
  pw: np.ndarray
  ip: np.ndarray
  sn: np.ndarray
  strand: dc_constants.Strand
  ccs_idx: np.ndarray = np.empty(0)
  # truth_idx and truth_range only used with label reads.
  truth_idx: np.ndarray = np.empty(0)
  # truth range is a dict containing contig, begin, end.
  # It is not modified when slicing is performed.
  # The truth_range['contig'] and truth_idx are used calculate
  # label_coords from sliced regions.
  # truth_range bounds are [begin, end) in keeping with bed format.
  truth_range: Union[Dict[str, Any], None] = None

  # Alignment Variables
  seq_indices: np.ndarray = np.empty(0)
  is_insertion: np.ndarray = np.empty(0)
  seq_len: int = 0
  idx_seq: int = 0
  idx_spaced: int = 0
  done: bool = False

  def setup_spacing(self):
    """Set up an array for storing spaced indices."""
    self.seq_indices = np.zeros(len(self.bases), dtype=np.int)
    self.is_insertion = self.cigar == dc_constants.PYSAM_CINS
    self.seq_len = len(self.bases)

  def move(self):
    """For each position, track its spaced index.

    Example:
      Sequence -> seq_indices   -> put_spacing().
      'AAAA'   -> [0, 1, 3, 4]  -> 'AA AA'
      'MMIM'
    """
    self.seq_indices[self.idx_seq] = self.idx_spaced
    self.idx_seq += 1
    self.idx_spaced += 1

  def add_gap(self):
    self.idx_spaced += 1

  def is_out_of_bounds(self):
    return self.idx_seq >= self.seq_len

  def next_is_insertion(self):
    # When insertions are encountered in the label, add them in
    # to maintain spacing correctly.
    if self.truth_range:
      while self.is_insertion[self.idx_seq]:
        # For label insertions, insert bases.
        self.seq_indices[self.idx_seq] = self.idx_spaced
        self.idx_seq += 1
        self.idx_spaced += 1
      return False
    # pysam.CINS must be cast as an int or this block runs very slow.
    return self.is_insertion[self.idx_seq]

  def put_spacing(self, seq_len):
    """Generate spaced sequences and replace the originals."""
    spaced_seq = np.repeat(dc_constants.GAP_OR_PAD, seq_len)
    spaced_pw = np.zeros(seq_len, dtype=np.uint8)
    spaced_ip = np.zeros(seq_len, dtype=np.uint8)
    spaced_ccs_idx = np.repeat(-1, seq_len)
    spaced_seq[self.seq_indices] = self.bases
    spaced_pw[self.seq_indices] = self.pw
    spaced_ip[self.seq_indices] = self.ip
    spaced_ccs_idx[self.seq_indices] = self.ccs_idx
    if self.truth_range:
      spaced_cigar = np.repeat(dc_constants.PYSAM_CHARD_CLIP, seq_len)
      spaced_cigar[self.seq_indices] = self.cigar
      self.cigar = spaced_cigar
      truth_pos = np.repeat(-1, seq_len)
      truth_idx = np.arange(self.truth_range['begin'], self.truth_range['end'])
      truth_aln_base = np.isin(self.cigar,
                               dc_constants.PYSAM_READ_ADVANCING_OPS)
      assert len(truth_pos[truth_aln_base]) == len(truth_idx)
      truth_pos[truth_aln_base] = truth_idx
      self.truth_idx = truth_pos

    self.bases = spaced_seq
    self.pw = spaced_pw
    self.ip = spaced_ip
    self.ccs_idx = spaced_ccs_idx

  @property
  def bases_encoded(self) -> np.ndarray:
    bases_encoded = np.ndarray(
        self.bases.shape, dtype=dc_constants.NP_DATA_TYPE)
    for k, base in enumerate(dc_constants.VOCAB):
      bases_encoded[self.bases == base] = k
    return bases_encoded

  @property
  def zmw(self) -> int:
    return int(self.name.split('/')[1])

  @property
  def label_coords(self) -> str:
    # Reports reference coordinates as chr:begin-end.
    if self.is_label:
      begin = self.label_bounds.start
      end = self.label_bounds.stop
      return f'{self.truth_range["contig"]}:{begin}-{end}'
    return ''

  @property
  def is_label(self) -> bool:
    return self.truth_range is not None

  @property
  def ccs_bounds(self) -> slice:
    """Return ccs min and max for a given slice."""
    ccs_idx = np.ma.masked_array(self.ccs_idx, self.ccs_idx == -1)
    if not ccs_idx.count():
      # If no ccs coordinates are covered in this region, return an empty slice.
      return slice(0, 0)
    ccs_start = np.min(ccs_idx)
    ccs_end = np.max(ccs_idx)
    return slice(ccs_start, ccs_end)

  @property
  def label_bounds(self) -> slice:
    """Return label reference min and max positions for given slice."""
    truth_idx = np.ma.masked_array(self.truth_idx, self.truth_idx == -1)
    if not truth_idx.count():
      # If no truth coords are covered in this region, return an empty slice.
      return slice(0, 0)
    truth_start = np.min(truth_idx)
    truth_end = np.max(truth_idx)
    return slice(truth_start, truth_end)

  def ccs_slice(self, start, end) -> 'Read':
    """Perform slicing based on ccs coordinates. Coordinates are inclusive."""
    # Note that these bounds are inclusive by design.
    locs = np.where(np.logical_and(self.ccs_idx >= start,
                                   self.ccs_idx <= end))[0]
    if locs.any():
      ccs_slice = slice(np.min(locs), np.max(locs) + 1)
    else:
      ccs_slice = slice(0, 0)
    return Read(
        name=self.name,
        bases=self.bases[ccs_slice],
        cigar=self.cigar[ccs_slice],
        pw=self.pw[ccs_slice],
        ip=self.ip[ccs_slice],
        sn=self.sn,
        strand=self.strand,
        ccs_idx=self.ccs_idx[ccs_slice],
        truth_idx=self.truth_idx[ccs_slice],
        truth_range=self.truth_range)

  def pad(self, pad_width):
    return Read(
        name=self.name,
        bases=right_pad(self.bases, pad_width, dc_constants.GAP_OR_PAD),
        cigar=right_pad(self.cigar, pad_width, dc_constants.PYSAM_CHARD_CLIP),
        pw=right_pad(self.pw, pad_width, 0),
        ip=right_pad(self.ip, pad_width, 0),
        sn=self.sn,
        strand=self.strand,
        ccs_idx=right_pad(self.ccs_idx, pad_width, -1),
        truth_idx=right_pad(self.truth_idx, pad_width, -1),
        truth_range=self.truth_range)

  def remove_gaps_and_pad(self, pad_width: int) -> Union['Read', None]:
    """Removes gaps from sequence and returns padded."""
    # Useful for reducing label width.
    keep = self.bases != dc_constants.GAP_OR_PAD
    if sum(keep) > pad_width:
      return None
    return Read(
        name=self.name,
        bases=self.bases[keep],
        cigar=self.cigar[keep],
        pw=self.pw[keep],
        ip=self.ip[keep],
        sn=self.sn,
        strand=self.strand,
        ccs_idx=self.ccs_idx[keep],
        truth_idx=self.truth_idx[keep],
        truth_range=self.truth_range).pad(pad_width)

  def __str__(self):
    return ''.join(self.bases)

  def __len__(self):
    return len(self.bases)

  def __getitem__(self, r_slice: Union[slice, int]) -> 'Read':
    """Implements slicing across all attributes."""
    return Read(
        name=self.name,
        bases=self.bases[r_slice],
        cigar=self.cigar[r_slice],
        pw=self.pw[r_slice],
        ip=self.ip[r_slice],
        sn=self.sn,
        strand=self.strand,
        ccs_idx=self.ccs_idx[r_slice],
        truth_idx=self.truth_idx[r_slice])

  def __repr__(self):
    if np.any(self.ccs_idx >= 0):
      start = np.min(self.ccs_idx[self.ccs_idx >= 0])
      end = np.max(self.ccs_idx, initial=0)
    else:
      start = 0
      end = 0
    return (f'Read({self.name}) : CCS({start}-{end}) L={len(self.bases)} ' +
            self.label_coords).strip()


class DcConfig:
  """Option for controlling DcExample configuration and calculating indices."""

  _HAS_DYNAMIC_ATTRIBUTES = True

  # Features with n_rows = n_subreads.
  n_subread_features = ['bases', 'pw', 'ip', 'strand']
  fixed_height = 5  # ccs + sn

  def __init__(self, max_passes: int, example_width: int, padding: int):
    self.max_passes = max_passes
    self.example_width = example_width
    self.padding = padding
    self.feature_rows = {
        'bases': max_passes,
        'pw': max_passes,
        'ip': max_passes,
        'strand': max_passes,
        'ccs': 1,
        'sn': 4
    }
    # Sets slices indicating rows for each feature type.
    self.feature_indices = dict()
    i_rows = 0
    for k, v in self.feature_rows.items():
      self.feature_indices[k] = slice(i_rows, i_rows + self.feature_rows[k])
      setattr(self, k, i_rows)
      i_rows += v

  @classmethod
  def from_shape(cls, subreads_shape, padding=0):
    """Construct DcConfig from subreads shape."""
    height, width, _ = subreads_shape
    max_passes = (height - cls.fixed_height) // len(DcConfig.n_subread_features)
    if padding:
      width = width - padding
    return DcConfig(max_passes, width, padding)

  def indices(self, feature: str, n_subreads: int = 0) -> slice:
    """Returns rows for a given feature."""
    if n_subreads:
      assert feature in DcConfig.n_subread_features
      n_rows = min(n_subreads, self.max_passes)
      return slice(getattr(self, feature), getattr(self, feature) + n_rows)
    else:
      assert feature not in DcConfig.n_subread_features
      return slice(
          getattr(self, feature),
          getattr(self, feature) + self.feature_rows[feature])

  @property
  def tensor_height(self) -> int:
    """Returns total rows for tf.Example input."""
    return sum(self.feature_rows.values())

  @property
  def tensor_width(self) -> int:
    """Returns total rows for tf.Example input."""
    return self.example_width + self.padding

  def to_dict(self):
    """Output configuration properties as dict."""
    return {
        # Encode values as strings to prevent downstream aggregation.
        'max_passes': str(self.max_passes),
        'example_width': str(self.example_width),
        'padding': str(self.padding),
        'tensor_height': str(self.tensor_height),
        'tensor_width': str(self.tensor_width)
    }


@dataclasses.dataclass
class DcExample:
  """Python container used to generate DeepConsensus tf.Example."""
  name: str
  reads: List[Read]
  config: DcConfig
  counter: collections.Counter = dataclasses.field(
      default_factory=collections.Counter)

  # Define cached variables.
  _width: Optional[int] = None
  _ccs_width: Optional[int] = None

  @property
  def contig(self):
    if self.label:
      return self.label.truth_range['contig']
    return None

  @property
  def is_training(self) -> bool:
    # If a label is in the last position we are in training mode.
    return self.reads[-1].is_label

  @property
  def ccs(self) -> Read:
    if self.is_training:
      ccs_idx = -2
    else:
      ccs_idx = -1
    return self.reads[ccs_idx]

  @property
  def label(self) -> Union[Read, None]:
    if self.is_training:
      return self.reads[-1]
    return None

  @property
  def label_coords(self) -> str:
    if self.is_training:
      return self.label.label_coords
    return ''

  @property
  def subreads(self) -> List[Read]:
    if self.is_training:
      return self.reads[:-2]
    else:
      return self.reads[:-1]

  @property
  def n_subreads(self) -> int:
    # Returns the total number of subreads
    return len(self.subreads)

  @property
  def keep_subreads(self) -> int:
    # Returns usable number of subreads.
    return min(self.config.max_passes, self.n_subreads)

  @property
  def width(self) -> int:
    if self._width:
      return self._width
    else:
      self._width = len(self.ccs.bases)
    return self._width

  @property
  def ccs_width(self) -> int:
    # Width - gaps at end.
    if self._ccs_width:
      return self._ccs_width
    else:
      self._ccs_width = len(str(self.ccs).rstrip())
    return self._ccs_width

  @property
  def is_empty(self) -> bool:
    return not (self.ccs.ccs_idx >= 0).any()

  def iter_examples(self) -> 'DcExample':
    """Generates partitions from a given window."""
    # Initiate counter
    self.counter = collections.Counter()
    example_width = self.config.example_width
    padding = self.config.padding
    total_width = example_width + padding
    for start_pos in range(0, self.ccs_width, example_width):
      window = self[start_pos:start_pos + example_width]
      if start_pos > self.ccs_width:
        break
      if window.is_empty:
        self.counter['n_examples_no_ccs_idx'] += 1
        continue
      # If the label extends beyond width + padding,
      # remove gaps and right pad.
      # Gaps are helpful for visualizing alignments, but are
      # used during training.
      if self.is_training and len(window.label.bases) > total_width:
        adjusted_label = window.label.remove_gaps_and_pad(total_width)
        # Even with this adjustment it is still possible for the label to
        # be longer than the padded length. This is rare. Discard when training.
        if not adjusted_label:
          # Consider alternative solutions to incorporate these data.
          self.counter['n_examples_label_overflow'] += 1
          continue
        self.counter['n_examples_adjusted_label'] += 1
        window.reads[-1] = adjusted_label
      # Apply padding:
      reads = [x.pad(total_width) for x in window.reads]
      yield DcExample(self.name, reads, self.config)

  def stack_subread_feature(self, name):
    """Extract read feature and stack."""
    max_passes = self.config.max_passes
    return np.stack([getattr(x, name) for x in self.subreads[:max_passes]])

  def extract_features(self):
    """Convert features to a 2D array."""

    # Get shape (example_rows, width)
    n_subreads = self.n_subreads
    dims = (self.config.tensor_height, self.width)
    data = np.zeros(shape=dims, dtype=dc_constants.NP_DATA_TYPE)

    # Get feature indices.
    bases_idx = self.config.indices('bases', n_subreads)
    pw_idx = self.config.indices('pw', n_subreads)
    ip_idx = self.config.indices('ip', n_subreads)
    strand_idx = self.config.indices('strand', n_subreads)
    ccs_idx = self.config.indices('ccs')
    sn_idx = self.config.indices('sn')

    # Set features.
    data[bases_idx] = self.stack_subread_feature('bases_encoded')
    data[pw_idx] = self.stack_subread_feature('pw')
    data[ip_idx] = self.stack_subread_feature('ip')
    # Format strand feature.
    strand = self.stack_subread_feature('strand')
    strand = strand.astype(dc_constants.NP_DATA_TYPE)
    strand = np.repeat(np.expand_dims(strand, -1), self.width, -1)
    data[strand_idx] = strand
    data[ccs_idx] = self.ccs.bases_encoded
    # Format sn rows.
    sn = np.repeat(np.expand_dims(self.subreads[0].sn, -1), self.width, -1)
    data[sn_idx] = sn

    return np.expand_dims(data, -1)

  def to_features_dict(self):
    """Convert DcExample to a dictionary for inference."""
    data = self.extract_features()
    # Add additional dimension.
    features = {
        'subreads': data,
        'subreads/num_passes': self.keep_subreads,
        'name': self.name,
        'window_pos': self.ccs.ccs_bounds.start
    }
    return features

  def tf_example(self) -> tf.train.Example:
    """Convert DcExample to tf.Example."""
    data = self.extract_features()
    # Add additional dimension.
    example = tf.train.Example()
    features = example.features
    features.feature['subreads/encoded'].bytes_list.value.append(data.tobytes())
    features.feature['subreads/shape'].int64_list.value.extend(data.shape)
    features.feature['subreads/num_passes'].int64_list.value.append(
        self.keep_subreads)
    features.feature['name'].bytes_list.value.append(self.name.encode())
    features.feature['window_pos'].int64_list.value.append(
        self.ccs.ccs_bounds.start)

    if self.is_training:
      label = self.label.bases_encoded
      features.feature['label/encoded'].bytes_list.value.append(label.tobytes())
      features.feature['label/shape'].int64_list.value.extend(label.shape)
    return example

  def __getitem__(self, r_slice: Union[slice, int]) -> 'DcExample':
    """Implements windowed slicing of subreads and ccs_slicing of label."""
    if isinstance(r_slice, int):
      raise NotImplementedError
    reads = self.subreads + [self.ccs]
    reads = [x[r_slice] for x in reads]
    if self.label:
      ccs_slice = self.ccs[r_slice].ccs_bounds
      reads.append(self.label.ccs_slice(ccs_slice.start, ccs_slice.stop))
    return DcExample(self.name, reads, self.config)

  def __repr__(self):
    preview = self[:100]
    start = preview.ccs.ccs_bounds.start
    end = preview.ccs.ccs_bounds.stop
    output = ''
    output += (f'{self.name} CCS({start}-{end}) {self.label_coords}'.strip() +
               f'\n{"-"*(preview.width+24)}\n')
    for subread in self.subreads:
      subread_range = subread.name.split('/')[2]
      output += f'{subread_range:<20} {subread.strand} >{str(subread)}\n'
    output += f'{"CCS":<22} >{str(preview.ccs)}\n'

    if self.is_training:
      label = str(self.label)
      output += f'{"Label":<22} >{label}\n'
    return output


def decode_bases(bases_encoded: np.ndarray) -> np.ndarray:
  """Reverses DcExample encode_bases."""
  n_subreads, example_width = bases_encoded.shape
  bases = np.stack([np.repeat(dc_constants.GAP_OR_PAD, example_width)] *
                   n_subreads)
  for k, base in enumerate(dc_constants.VOCAB):
    bases[bases_encoded == k] = base
  return bases


def from_features_dict(features_dict: Dict[str, Any],
                       padding: int = 0) -> DcExample:
  """Converts features_dict partially back to a DcExample object for tests."""
  dc_config = DcConfig.from_shape(
      features_dict['subreads/shape'], padding=padding)
  data = np.squeeze(features_dict['subreads'])
  name = features_dict['name']
  n_subreads = features_dict['subreads/num_passes']
  # Note: The ccs start position is correct, but indices
  # may not be accurate beyond the first position.
  ccs_start_pos = features_dict['window_pos']

  # Get feature indices.
  bases_idx = dc_config.indices('bases', n_subreads)
  pw_idx = dc_config.indices('pw', n_subreads)
  ip_idx = dc_config.indices('ip', n_subreads)
  strand_idx = dc_config.indices('strand', n_subreads)
  ccs_idx = dc_config.indices('ccs')
  sn_idx = dc_config.indices('sn')

  # Convert 2D array back to features.
  bases = decode_bases(data[bases_idx])
  pw = data[pw_idx]
  ip = data[ip_idx]
  strand = data[strand_idx]
  ccs = decode_bases(data[ccs_idx])[0]
  sn = data[sn_idx][:, 1]

  ccs_idx = np.repeat(-1, dc_config.tensor_width)
  ccs_end_pos = ccs_start_pos + dc_config.example_width
  ccs_idx[0:dc_config.example_width] = np.arange(ccs_start_pos, ccs_end_pos)

  movie, zmw, _ = name.split('/')

  # Generate DcExample
  read_set = []
  for i in range(n_subreads):
    read = Read(
        f'{movie}/{zmw}/{i}',
        bases=bases[i],
        cigar=np.repeat(np.uint8(pysam.CMATCH), dc_config.example_width),
        pw=pw[i],
        ip=ip[i],
        sn=sn,
        strand=dc_constants.Strand(int(strand[i][0])),
        ccs_idx=ccs_idx)
    read_set.append(read)
  ccs_read = Read(
      name=name,
      bases=ccs,
      cigar=np.repeat(np.uint8(pysam.CMATCH), dc_config.example_width),
      pw=np.repeat(np.uint8(0), dc_config.example_width),
      ip=np.repeat(np.uint8(0), dc_config.example_width),
      sn=np.repeat(0, 4),
      strand=dc_constants.Strand.UNKNOWN,
      ccs_idx=ccs_idx)
  read_set.append(ccs_read)
  return DcExample(name=name, reads=read_set, config=dc_config)


def set_feature(feature, shape):
  """Read in feature and set shape."""
  feature = np.frombuffer(feature, dtype=dc_constants.NP_DATA_TYPE)
  feature = feature.reshape(shape)
  return feature


def tf_example_to_features_dict(tf_example_proto_str, inference=False):
  """Convert tf.Example to features_dict."""
  features = data_providers.parse_example(
      tf_example_proto_str, inference=inference)

  for key, val in features.items():
    features[key] = val.numpy()

  # Cast types
  features['name'] = str(features['name'][0], 'UTF-8')
  features['subreads/num_passes'] = int(features['subreads/num_passes'])

  features['subreads'] = set_feature(features['subreads/encoded'],
                                     features['subreads/shape'])
  del features['subreads/encoded']
  if not inference:
    features['label'] = set_feature(features['label/encoded'],
                                    features['label/shape'])
    del features['label/encoded']
  return features


def fetch_ccs_seq(ccs_seqname: str,
                  ccs_fasta: pysam.libcfaidx.FastaFile) -> Read:
  """Fetches a ccs sequence by name."""
  ccs_seq = ccs_fasta.fetch(ccs_seqname)
  ccs_seq = np.array(ccs_seq, 'c')
  # The ccs ref sequences are 1:len(ccs_seq).
  return Read(
      name=ccs_seqname,
      bases=ccs_seq,
      cigar=np.repeat(np.uint8(pysam.CMATCH), len(ccs_seq)),
      pw=np.repeat(np.uint8(0), len(ccs_seq)),
      ip=np.repeat(np.uint8(0), len(ccs_seq)),
      sn=np.repeat(0, 4),
      strand=dc_constants.Strand.UNKNOWN,
      ccs_idx=np.arange(len(ccs_seq)))


def fetch_label_alignment(
    ccs_seqname: str, truth_to_ccs: pysam.AlignmentFile,
    truth_range: Dict[str, Any]) -> Union[dc_constants.Issue, Read]:
  """Fetches a label aligned to ccs sequence."""
  try:
    truth_alignment = next(truth_to_ccs.fetch(ccs_seqname))
  except (ValueError, StopIteration):
    return Issue.TRUTH_ALIGNMENT_NOT_FOUND
  if truth_alignment.is_supplementary:
    return Issue.SUPP_TRUTH_ALIGNMENT
  truth_alignment = expand_clip_indent(truth_alignment, truth_range)
  return truth_alignment


def read_truth_bedfile(truth_bed: str) -> Dict[str, Dict[str, Any]]:
  """Reads in complete truth bed file and returns dict."""
  bed_coords = {}
  with open(truth_bed, 'r') as bedfile:
    for line in bedfile:
      contig, begin, end, ccs_seqname = line.strip().split('\t')[:4]
      bed_record = {'contig': contig, 'begin': int(begin), 'end': int(end)}
      bed_coords[ccs_seqname] = bed_record
  return bed_coords


def read_truth_split(split_fname: str) -> Dict[str, str]:
  """Reads in split bed file and returns dict."""
  contig_split = {}
  split_regions = {}
  for i in dc_constants.HUMAN_TRAIN_REGIONS:
    split_regions[i] = 'train'
  for i in dc_constants.HUMAN_EVAL_REGIONS:
    split_regions[i] = 'eval'
  for i in dc_constants.HUMAN_TEST_REGIONS:
    split_regions[i] = 'test'
  with open(split_fname, 'r') as f:
    for line in f:
      contig, chrom = line.split()
      if chrom in split_regions:
        contig_split[contig] = split_regions[chrom]
  return contig_split


def expand_clip_indent(read: pysam.AlignedSegment,
                       truth_range: Union[Dict[str, Any], None] = None) -> Read:
  """Adds GAP_OR_PAD tokens and clips reads.

  For both subreads and label:

  * Expand sequence by placing gaps where deletions are present in alignment.
  * Remove bases that are part of soft-clips.
  * Indent alignment if start position is > 0.
  * Reverse ip/pw values when the strand is reverse.

  Args:
      read: a pysam aligned segment representing a subread, ccs, or label aln.
      truth_range: truth genome alignment coordinates. If supplied, it is
                   assumed this is the label alignment.

  Returns:
      ExpandedRead
  """
  # Extract read and reference indices.
  aligned_pairs = read.get_aligned_pairs()
  read_idx = np.array([x[0] if x[0] is not None else -1 for x in aligned_pairs])
  ccs_idx = np.array([x[1] if x[1] is not None else -1 for x in aligned_pairs])
  aln_len = len(read_idx)

  # Create empty expanded read objects.
  new_seq = np.repeat(dc_constants.GAP_OR_PAD, aln_len)
  new_pw = np.repeat(np.uint8(0), aln_len)
  new_ip = np.repeat(np.uint8(0), aln_len)

  # Fill read objects based on aligned read idx positions.
  new_seq[read_idx >= 0] = list(read.seq)

  if read.is_reverse:
    strand = dc_constants.Strand.REVERSE
  else:
    strand = dc_constants.Strand.FORWARD

  # pw/ip values are never set for labels.
  # truth_range is used to test if we are working with a label Read.
  if not truth_range:
    # Reverse ip/pw values if the strand is reversed.
    pw_vals = read.get_tag('pw')
    ip_vals = read.get_tag('ip')
    if strand == dc_constants.Strand.REVERSE:
      pw_vals = pw_vals[::-1]
      ip_vals = ip_vals[::-1]
    new_pw[read_idx >= 0] = pw_vals
    new_ip[read_idx >= 0] = ip_vals
    sn = np.array(read.get_tag('sn'))
  else:
    sn = np.empty(0)

  # Extract additional read properties.
  cigar_seq = itertools.chain.from_iterable([[x] * y for x, y in read.cigar])
  new_cigar = np.fromiter(cigar_seq, dtype=np.uint8)
  # Filter hard_clip from cigar.
  new_cigar = new_cigar[new_cigar != dc_constants.PYSAM_CHARD_CLIP]

  # Trim sequence if it is soft-padded.
  if np.sum(new_cigar == dc_constants.PYSAM_CSOFT_CLIP) > 0:
    new_seq[new_cigar ==
            dc_constants.PYSAM_CSOFT_CLIP] = dc_constants.GAP_OR_PAD
    # TODO: binary search ignoring -1 vals here.
    qstart = np.where(read_idx == read.query_alignment_start)[0][0]
    qend = np.where(read_idx == read.query_alignment_end - 1)[0][0] + 1
    # Trim soft-padded segments from truth regions.
    if truth_range:
      op, op_len = read.cigartuples[0]
      if op == dc_constants.PYSAM_CSOFT_CLIP:
        truth_range['begin'] = truth_range['begin'] + op_len
      op, op_len = read.cigartuples[-1]
      if op == dc_constants.PYSAM_CSOFT_CLIP:
        truth_range['end'] = truth_range['end'] - op_len

    new_seq = new_seq[qstart:qend]
    new_pw = new_pw[qstart:qend]
    new_ip = new_ip[qstart:qend]
    new_cigar = new_cigar[qstart:qend]
    ccs_idx = ccs_idx[qstart:qend]

  # Indent sequence
  if read.pos:
    new_seq = np.insert(new_seq, 0, [dc_constants.GAP_OR_PAD] * read.pos)
    # Add N cigar op at position 0 to indicate indent.
    new_cigar = np.insert(new_cigar, 0,
                          np.repeat(int(pysam.CREF_SKIP), read.pos))
    new_pw = np.insert(new_pw, 0, np.repeat(0, read.pos))
    new_ip = np.insert(new_ip, 0, np.repeat(0, read.pos))
    ccs_idx = np.insert(ccs_idx, 0, np.repeat(-1, read.pos))

  return Read(
      name=read.qname,
      bases=new_seq,
      cigar=new_cigar,
      pw=new_pw,
      ip=new_ip,
      sn=sn,
      strand=strand,
      ccs_idx=ccs_idx,
      truth_range=truth_range)


def space_out_subreads(subreads: List[Read]) -> List[Read]:
  """Spaces out subreads to make room for insertions in any subset of them."""
  for r in subreads:
    r.setup_spacing()
  while not all([r.done for r in subreads]):
    # This loops over bases in all subreads at once, from left to right.
    any_insertions = False
    for r in subreads:
      if r.done:
        continue
      if r.next_is_insertion():
        any_insertions = True
        break

    for r in subreads:
      if r.done:
        continue
      if any_insertions and not r.next_is_insertion():
        # If other reads have insertions, but this one does NOT, add a gap to
        # this read to make space.
        r.add_gap()
      else:
        # In all other cases, just take the next base and move on.
        r.move()
        if r.is_out_of_bounds():
          # Finally, format reads with spacing.
          r.done = True

  # Right pad all spaced sequences so they have the same length.
  max_len = max([r.idx_spaced for r in subreads])
  for r in subreads:
    r.put_spacing(max_len)

  return subreads


def create_proc_feeder(subreads_to_ccs: str,
                       ccs_fasta: str,
                       dc_config: DcConfig,
                       truth_bed: Optional[str] = None,
                       truth_to_ccs: Optional[str] = None,
                       truth_split: Optional[str] = None,
                       limit: int = 0,
                       bam_reader_threads: int = 1):
  """Creates a generator to feed subread process jobs to a multiprocess pool."""
  main_counter = collections.Counter()

  # Initiate files
  subread_grouper = SubreadGrouper(subreads_to_ccs, bam_reader_threads)
  ccs_fasta = pysam.FastaFile(ccs_fasta)

  is_training = truth_bed and truth_to_ccs and truth_split
  if is_training:
    # Load files required for training.
    truth_to_ccs_bam = pysam.AlignmentFile(truth_to_ccs, require_index=True)
    truth_ref_coords = read_truth_bedfile(truth_bed)
    truth_split_dict = read_truth_split(truth_split)

  def proc_feeder():
    for read_set in subread_grouper:
      main_counter['n_zmw_processed'] += 1
      subreads = list(map(expand_clip_indent, read_set))
      ccs_seqname = '/'.join(subreads[0].name.split('/')[:2] + ['ccs'])
      # Fetch ccs sequence and append to subread set.
      ccs_seq = fetch_ccs_seq(ccs_seqname, ccs_fasta)
      subreads.append(ccs_seq)
      if is_training:
        # Fetch truth to ccs alignment.
        truth_range = truth_ref_coords.get(ccs_seqname, None)
        if not truth_range:
          logging.info('No truth_range defined for %s.', ccs_seqname)
          main_counter['n_zmw_missing_truth_range'] += 1
          continue
        label = fetch_label_alignment(ccs_seqname, truth_to_ccs_bam,
                                      truth_range)
        if label == Issue.TRUTH_ALIGNMENT_NOT_FOUND:
          logging.info('Unable to fetch label alignment for %s.', ccs_seqname)
          main_counter['n_zmw_no_label_alignment'] += 1
          continue
        elif label == Issue.SUPP_TRUTH_ALIGNMENT:
          main_counter['n_zmw_truth_label_supp_alignment'] += 1
          continue
        subreads.append(label)
        # pytype: disable=attribute-error
        split = truth_split_dict.get(truth_range['contig'], None)
        # pytype: enable=attribute-error
        if not split:
          logging.info('No split defined for %s.', ccs_seqname)
          main_counter['n_zmw_missing_contig_split'] += 1
          continue
      else:
        split = 'inference'
      main_counter[f'n_zmw_{split}'] += 1
      main_counter['n_zmw_pass'] += 1
      yield (subreads, ccs_seqname, dc_config, split)
      if limit and main_counter['n_zmw_pass'] >= limit:
        break

  return proc_feeder, main_counter


def subreads_to_dc_example(subreads: List[Read], ccs_seqname: str,
                           dc_config: DcConfig) -> DcExample:
  """Process subreads and return a DcExample object."""
  aln_reads = space_out_subreads(subreads)
  dc_example = DcExample(name=ccs_seqname, reads=aln_reads, config=dc_config)
  return dc_example
