# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DoFns for aligning subreads and labels and writing out deepconsensus_pb2.DeepConsensusInput protos."""

import copy
import logging
from typing import Iterable, List, Tuple, Union

import apache_beam as beam
from apache_beam import metrics
from apache_beam import typehints

from deepconsensus.preprocess import preprocess_utils
from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.utils import dc_constants
from nucleus.protos import bed_pb2
from nucleus.protos import cigar_pb2
from nucleus.protos import reads_pb2
from nucleus.util import cigar as cigar_utils
from nucleus.util import struct_utils


@typehints.with_input_types(reads_pb2.Read)
@typehints.with_output_types(Tuple[str, reads_pb2.Read])
class GetReadMoleculeNameDoFn(beam.DoFn):
  """DoFn that yields tuples of (molecule_name, reads_pb2.Read proto).

  The molecule name is derived form the read fragment name.
  """

  def process(self,
              read: reads_pb2.Read) -> Iterable[Tuple[str, reads_pb2.Read]]:
    """Yields molecule name and the given read proto."""
    pacbio_molecule_name = preprocess_utils.get_pacbio_molecule_name(
        read.fragment_name)
    if pacbio_molecule_name is not None:
      yield pacbio_molecule_name, read
    else:
      raise ValueError(str(read))


@typehints.with_input_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                             List[reads_pb2.Read]]])
@typehints.with_output_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                              List[reads_pb2.Read]]])
class ExpandFieldsRemoveSoftClipsDoFn(beam.DoFn):
  """DoFn that yields expanded subread and label reads_pb2.Read protos.

  For both subreads and label:

  * Cigar is expanded to a string (3M --> 'MMM')

  * Only positions where cigar is one of dc_constants.OPS_TO_CONSIDER kept. For
  example, bases (and corresponding cigar, pw, ip) at soft clips positions are
  not kept.

  * At any position where cigar contains a deletion, sequence is spaced out to
  include a dc_constants.GAP_OR_PAD token. sequence, cigar ops list, and cigar
  str cigar str should have the same length after this function.
  """

  def __init__(self):
    self.multiple_alignments_counter = metrics.Metrics.counter(
        self.__class__, 'labels_with_multiple_correct_alignments')
    self.no_subreads_counter = metrics.Metrics.counter(
        self.__class__, 'example_with_no_subreads')
    self.no_label_counter = metrics.Metrics.counter(self.__class__,
                                                    'example_with_no_label')

  def process(
      self, molecule_data: Tuple[str, Tuple[List[reads_pb2.Read],
                                            List[reads_pb2.Read]]]
  ) -> Iterable[Tuple[str, Tuple[List[reads_pb2.Read], List[reads_pb2.Read]]]]:
    """Yields copies of input reads with expanded sequence and clips removed."""

    (molecule_name, (subreads, label)) = molecule_data

    # Note, examples will only be included in one of the initial counters since
    # we are returning early.
    if not subreads:
      self.no_subreads_counter.inc()
      return
    if not label:
      self.no_label_counter.inc()
      return

    # Sanity checks. These are not enough though, we also need to check other
    # fields in the label, which is done below.
    assert subreads
    assert label

    # Do not error for labels that have multiple alignments to correct molecule.
    # One of the alignments may be a supplementary alignment.
    if len(label) > 1:
      logging.info('Unexpected: %d labels for %s', len(label),
                   label[0].fragment_name)
      self.multiple_alignments_counter.inc()

    subreads_and_label = subreads + label
    subreads_and_labels_copy = copy.deepcopy(subreads_and_label)

    for read in subreads_and_labels_copy:
      assert read.aligned_sequence
      base_index = 0
      expanded_sequence = ''
      expanded_cigar_str = ''
      new_cigar_ops = []
      pw = struct_utils.get_int_field(read.info, 'pw')
      ip = struct_utils.get_int_field(read.info, 'ip')
      new_pw = []
      new_ip = []

      for op in read.alignment.cigar:

        # Skip over ops we don't want, such as soft clips.
        if op.operation not in dc_constants.OPS_TO_CONSIDER:
          base_index += op.operation_length
          continue
        if op.operation in dc_constants.READ_ADVANCING_OPS:
          start = base_index
          end = start + op.operation_length
          expanded_sequence += read.aligned_sequence[start:end]
          new_pw += pw[start:end]
          new_ip += ip[start:end]
          base_index += op.operation_length
        else:
          # Add a special token in sequence where we have deletion.
          expanded_sequence += dc_constants.GAP_OR_PAD * op.operation_length

        new_cigar_ops.append(op)
        op_char = cigar_utils.CIGAR_OPS_TO_CHAR[op.operation]
        expanded_cigar_str += op_char * op.operation_length

      # Update the read sequence.
      read.aligned_sequence = expanded_sequence
      assert len(read.aligned_sequence) == len(expanded_cigar_str)

      # Update the read cigar to only include ops that were kept.
      del read.alignment.cigar[:]
      read.alignment.cigar.extend(new_cigar_ops)

      # Save pw, ip, and expanded cigar string to be used downstream.
      struct_utils.set_int_field(read.info, 'pw', new_pw)
      struct_utils.set_int_field(read.info, 'ip', new_ip)
      struct_utils.set_string_field(read.info, 'expanded_cigar',
                                    expanded_cigar_str)

    subreads_copy = subreads_and_labels_copy[:-1]
    label_copy = subreads_and_labels_copy[-1:]

    # <internal> identified an error where a subread was used as the label in
    # some cases. This led to the label having PW and IP fields, which should
    # not happen since only subreads have PW and IP.
    assert not struct_utils.get_int_field(label_copy[0].info, 'pw')

    yield molecule_name, (subreads_copy, label_copy)


@typehints.with_input_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                             List[reads_pb2.Read]]])
@typehints.with_output_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                              List[reads_pb2.Read]]])
class IndentReadsDoFn(beam.DoFn):
  """DoFn that adds padding to beginning of reads.

  Padding at beginning of reads is added based on start position of read.
  """

  def process(
      self, molecule_data: Tuple[str, Tuple[List[reads_pb2.Read],
                                            List[reads_pb2.Read]]]
  ) -> Iterable[Tuple[str, Tuple[List[reads_pb2.Read], List[reads_pb2.Read]]]]:
    """Yields copies of reads indented based on start position."""

    (molecule_name, (subreads, label)) = molecule_data
    subreads_and_label = subreads + label
    subreads_and_labels_copy = copy.deepcopy(subreads_and_label)

    # Indent sequence strings by starting position.
    for read in subreads_and_labels_copy:
      indent = dc_constants.GAP_OR_PAD * read.alignment.position.position
      read.aligned_sequence = indent + read.aligned_sequence
      indented_cigar_str = indent + struct_utils.get_string_field(
          read.info, 'expanded_cigar')[0]
      struct_utils.set_string_field(read.info, 'expanded_cigar',
                                    indented_cigar_str)

    subreads_copy = subreads_and_labels_copy[:-1]
    label_copy = subreads_and_labels_copy[-1:]
    yield molecule_name, (subreads_copy, label_copy)


def get_index_info(reads: List[Union[reads_pb2.Read,
                                     deepconsensus_pb2.Subread]],
                   index: int) -> Tuple[bool, bool]:
  """Returns whether `index` is valid and some read has an insertion at `index`.

  Args:
    reads: reads_pb2.Read protos that will be checked for insertions.
    index: position at which we will check for insertions.

  Returns:
    bool: True if `index` is larger then the length of all cigar strings in
    `reads`, else False.
    bool: True if any read has an insertion at `index`, else False.
  """

  insert_char = cigar_utils.CIGAR_OPS_TO_CHAR[cigar_pb2.CigarUnit.INSERT]
  out_of_bounds = True
  has_insert = False
  for read in reads:
    expanded_cigar = get_expanded_cigar(read)
    if index < len(expanded_cigar):
      out_of_bounds = False
      has_insert = expanded_cigar[index] == insert_char
    # Stop early as soon as we find a read with an insertion.
    if has_insert:
      break
  return out_of_bounds, has_insert


def shift(reads: List[reads_pb2.Read], op: cigar_pb2.CigarUnit.Operation,
          index) -> None:
  """Add PAD_TOKEN at `index` if cigar op at that index is not `op`."""

  op_char = cigar_utils.CIGAR_OPS_TO_CHAR[op]
  for read in reads:
    expanded_cigar = get_expanded_cigar(read)
    if index < len(read.aligned_sequence) and index >= len(expanded_cigar):
      raise ValueError(
          'Index len: %d, cigar len: %d, seq len: %d, molecule: %s' %
          (index, len(expanded_cigar), len(
              read.aligned_sequence), read.fragment_name))
    if index < len(read.aligned_sequence) and expanded_cigar[index] != op_char:
      seq = read.aligned_sequence
      read.aligned_sequence = seq[:index] + dc_constants.GAP_OR_PAD + seq[index:]
      new_expanded_cigar = expanded_cigar[:
                                          index] + dc_constants.GAP_OR_PAD + expanded_cigar[
                                              index:]
      struct_utils.set_string_field(read.info, 'expanded_cigar',
                                    new_expanded_cigar)


def get_expanded_cigar(
    read: Union[reads_pb2.Read, deepconsensus_pb2.Subread]) -> str:
  """Return the expanded cigar string for the given read from the info field."""

  if isinstance(read, reads_pb2.Read):
    # Explicitly cast to str for Python 2, since `get_string_field()` returns
    # unicode for Python 2.
    expanded_cigar = str(
        struct_utils.get_string_field(read.info, 'expanded_cigar')[0])
    read_name = read.fragment_name
  elif isinstance(read, deepconsensus_pb2.Subread):
    expanded_cigar = read.expanded_cigar
    read_name = read.molecule_name
  if not expanded_cigar:
    raise ValueError('%s read does not contain expanded cigar' % read_name)
  return expanded_cigar


@typehints.with_input_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                             List[reads_pb2.Read]]])
@typehints.with_output_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                              List[reads_pb2.Read]]])
class AlignSequenceDoFn(beam.DoFn):
  """Returns aligned version of subreads and labels.

  For each position, go through all cigar strings in `bases_cigar_tuples`. For
  each position where at least one cigar has an insertion, insert a space into
  the cigar strings without an insertion and their corresponding bases.
  """

  def process(
      self, molecule_data: Tuple[str, Tuple[List[reads_pb2.Read],
                                            List[reads_pb2.Read]]]
  ) -> Iterable[Tuple[str, Tuple[List[reads_pb2.Read], List[reads_pb2.Read]]]]:
    """Yields reads with gaps inserted so that reads and labels are aligned."""

    (molecule_name, (subreads, label)) = molecule_data
    assert len(label) == 1
    subreads_copy = copy.deepcopy(subreads)
    label_copy = copy.deepcopy(label)

    # Stop if we have reached end of all reads.
    base_index = 0
    out_of_bounds = False
    label_offset = 0
    subread_indices = []
    unsup_insertions_by_pos = {}
    while not out_of_bounds:
      out_of_bounds, has_insert = get_index_info(subreads_copy, base_index)

      # `has_insert` will only be true if we are not out of bounds, meaning
      # at least one read has a base at `base_index`.
      if has_insert:
        shift(subreads_copy, cigar_pb2.CigarUnit.INSERT, base_index)
        shift(label_copy, cigar_pb2.CigarUnit.INSERT, base_index + label_offset)

      label_out_of_bounds, label_has_insert = get_index_info(
          label_copy, base_index + label_offset)
      # Label has an insertion that isn't present in subreads.
      if not has_insert and not label_out_of_bounds and label_has_insert:
        label_offset += 1
        # Increment previous index since it should also cover this insertion.
        if subread_indices:
          subread_indices = subread_indices[:-1] + [subread_indices[-1] + 1]
        prev_base_index = max(0, base_index - 1)
        unsup_insertions_by_pos[prev_base_index] = unsup_insertions_by_pos.get(
            prev_base_index, 0) + 1
      elif not out_of_bounds:
        base_index += 1
        subread_indices.append(base_index + label_offset)

    unsup_insertions_by_pos_keys = []
    unsup_insertions_by_pos_values = []
    for k, v in unsup_insertions_by_pos.items():
      unsup_insertions_by_pos_keys.append(k)
      unsup_insertions_by_pos_values.append(v)

    for s in subreads_copy:
      struct_utils.set_int_field(s.info, 'subread_indices', subread_indices)
      if unsup_insertions_by_pos_keys and unsup_insertions_by_pos_values:
        struct_utils.set_int_field(s.info, 'unsup_insertions_by_pos_keys',
                                   unsup_insertions_by_pos_keys)
        struct_utils.set_int_field(s.info, 'unsup_insertions_by_pos_values',
                                   unsup_insertions_by_pos_values)
    yield molecule_name, (subreads_copy, label_copy)


@typehints.with_input_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                             List[reads_pb2.Read]]])
@typehints.with_output_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                              List[reads_pb2.Read]]])
class PadReadsDoFn(beam.DoFn):
  """DoFn that adds padding to ends of reads.

  Padding at end of reads is added to ensure all reads are of same length.
  """

  def process(
      self, molecule_data: Tuple[str, Tuple[List[reads_pb2.Read],
                                            List[reads_pb2.Read]]]
  ) -> Iterable[Tuple[str, Tuple[List[reads_pb2.Read], List[reads_pb2.Read]]]]:
    """Yields copies of reads, all padded to the same length."""

    (molecule_name, (subreads, label)) = molecule_data
    subreads_and_label = subreads + label
    subreads_and_labels_copy = copy.deepcopy(subreads_and_label)

    # Get maximum subread length for this molecule.
    max_length = -float('inf')
    for read in subreads_and_labels_copy:
      max_length = max(max_length, len(read.aligned_sequence))

    # Pad ends of all reads so length is equal to max_length.
    for read in subreads_and_labels_copy:
      pad_length = max_length - len(read.aligned_sequence)
      padding = dc_constants.GAP_OR_PAD * pad_length
      read.aligned_sequence += padding
      padded_cigar_str = struct_utils.get_string_field(
          read.info, 'expanded_cigar')[0] + padding
      struct_utils.set_string_field(read.info, 'expanded_cigar',
                                    padded_cigar_str)

    subreads_copy = subreads_and_labels_copy[:-1]
    label_copy = subreads_and_labels_copy[-1:]
    yield molecule_name, (subreads_copy, label_copy)


@typehints.with_input_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                             List[reads_pb2.Read]]])
@typehints.with_output_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                              List[reads_pb2.Read]]])
class AlignPwIpDoFn(beam.DoFn):
  """DoFn that aligns pulse width (pw) and interpulse distance (ip) per read.

  Modify pw and ip to be aligned in the same way as subreads and labels. At
  positions where reads contain padding or internal gaps, a corresponding token
  is added to pw and ip. After this DoFn, the sequence, expanded cigar, pw, and
  ip for a read should all be of the same length.
  """

  def process(
      self, molecule_data: Tuple[str, Tuple[List[reads_pb2.Read],
                                            List[reads_pb2.Read]]]
  ) -> Iterable[Tuple[str, Tuple[List[reads_pb2.Read], List[reads_pb2.Read]]]]:
    """Yields copies of reads, with pw and ip aligned."""

    (molecule_name, (subreads, label)) = molecule_data

    # Do not need to consider label because it does not contain pw and ip.
    subreads_copy = copy.deepcopy(subreads)
    for read in subreads_copy:
      pw = struct_utils.get_int_field(read.info, 'pw')
      ip = struct_utils.get_int_field(read.info, 'ip')
      new_pw = []
      new_ip = []
      pw_ip_index = 0

      for base in read.aligned_sequence:
        # Padding and gap tokens are strings and cannot directly be added to pw
        # and ip, which are lists of ints. Instead, integer representations of
        # each must be added.
        if base == dc_constants.GAP_OR_PAD:
          new_pw.append(dc_constants.GAP_OR_PAD_INT)
          new_ip.append(dc_constants.GAP_OR_PAD_INT)
        elif base == dc_constants.GAP_OR_PAD:
          new_pw.append(dc_constants.GAP_OR_PAD_INT)
          new_ip.append(dc_constants.GAP_OR_PAD_INT)
        # If base is neither padding nor gap, copy over the existing pw and ip.
        else:
          assert pw_ip_index < len(pw)
          assert pw_ip_index < len(ip)
          new_pw.append(pw[pw_ip_index])
          new_ip.append(ip[pw_ip_index])
          pw_ip_index += 1

      # pw, ip, and sequence should all be of the same length.
      assert len(new_pw) == len(read.aligned_sequence)
      assert len(new_ip) == len(read.aligned_sequence)
      struct_utils.set_int_field(read.info, 'pw', new_pw)
      struct_utils.set_int_field(read.info, 'ip', new_ip)

    yield molecule_name, (subreads_copy, label)


@typehints.with_input_types(bed_pb2.BedRecord)
@typehints.with_output_types(Tuple[str, bed_pb2.BedRecord])
class GetBedRecordMoleculeNameDoFn(beam.DoFn):
  """DoFn for extracting molecule name for each record in a PacBio bed file."""

  def process(
      self,
      bed_record: bed_pb2.BedRecord) -> Iterable[Tuple[str, bed_pb2.BedRecord]]:
    """Yields molecule name and the input bed record."""

    pacbio_molecule_name = preprocess_utils.get_pacbio_molecule_name(
        bed_record.name)
    if pacbio_molecule_name is not None:
      yield pacbio_molecule_name, bed_record


@typehints.with_input_types(Tuple[str, Tuple[List[Tuple[List[reads_pb2.Read],
                                                        List[reads_pb2.Read]]],
                                             List[bed_pb2.BedRecord]]])
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class CreateTrainDeepConsensusInputDoFn(beam.DoFn):
  """DoFn for writing DeepConsensusInput protos consisting of Subread protos.

  DeepConsensusInput protos contain all the information that is needed for
  downstream models and can be used to create tf.Examples.
  """

  def __init__(self):
    self.no_bed_records_counter = metrics.Metrics.counter(
        self.__class__, 'example_with_no_bed_record')
    self.no_aligned_subreads_and_label = metrics.Metrics.counter(
        self.__class__, 'example_with_no_aligned_subreads_and_label')
    self.no_subreads_counter = metrics.Metrics.counter(
        self.__class__, 'example_with_no_subreads')
    self.no_label_counter = metrics.Metrics.counter(self.__class__,
                                                    'example_with_no_label')
    self.deepconsensus_input_counter = metrics.Metrics.counter(
        self.__class__, 'num_deepconsensus_input_protos_written')

  def process(
      self, molecule_data: Tuple[str, Tuple[List[Tuple[List[reads_pb2.Read],
                                                       List[reads_pb2.Read]]],
                                            List[bed_pb2.BedRecord]]]
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Yields deepconsensus_pb2.DeepConsensusInput protos for each molecule."""

    # Format of aligned_subreads_and_label is [([subreads], [label])].
    molecule_name, (aligned_subreads_and_label, bed_records) = molecule_data

    # Note, examples will only be included in one of the initial counters since
    # we are returning early.
    if not bed_records:
      self.no_bed_records_counter.inc()
      return
    if not aligned_subreads_and_label:
      self.no_aligned_subreads_and_label.inc()
      return

    subreads, label = aligned_subreads_and_label[0]
    if not subreads:
      self.no_subreads_counter.inc()
      return
    if not label:
      self.no_label_counter.inc()
      return

    # Sanity checks. These are not enough though, we also need to check other
    # fields in the label, which is done below.
    assert subreads
    assert label

    bed_record = bed_records[0]
    subread_protos = get_subread_protos(subreads + label, molecule_name)
    sn = struct_utils.get_number_field(subreads[0].info, 'sn')
    label_proto = subread_protos[-1]

    # <internal> identified an error where a subread was used as the label in
    # some cases. This led to the label having PW and IP fields, which should
    # not happen since only subreads have PW and IP.
    assert not label_proto.pw

    self.deepconsensus_input_counter.inc()
    subread_indices = struct_utils.get_int_field(subreads[0].info,
                                                 'subread_indices')
    unsup_insertions_by_pos = {}
    # Make a copy so Beam doesn't complain about input mutation.
    one_subread = copy.deepcopy(subreads[0])
    unsup_insertions_by_pos_keys = list(
        struct_utils.get_int_field(one_subread.info,
                                   'unsup_insertions_by_pos_keys'))
    unsup_insertions_by_pos_values = list(
        struct_utils.get_int_field(one_subread.info,
                                   'unsup_insertions_by_pos_values'))
    unsup_insertions_by_pos = {
        k: v for k, v in zip(unsup_insertions_by_pos_keys,
                             unsup_insertions_by_pos_values)
    }

    yield deepconsensus_pb2.DeepConsensusInput(
        subreads=subread_protos[:-1],
        label=label_proto,
        molecule_name=molecule_name,
        molecule_start=0,
        chrom_name=bed_record.reference_name,
        chrom_start=bed_record.start,
        chrom_end=bed_record.end,
        sn=sn,
        strand=bed_record.strand,
        subread_indices=subread_indices,
        unsup_insertions_by_pos=unsup_insertions_by_pos,
    )


def get_subread_protos(subreads: List[reads_pb2.Read],
                       molecule_name: str) -> List[deepconsensus_pb2.Subread]:
  """Returns subread protos created from the provided inputs."""
  subread_protos = []
  for read in subreads:
    subread_protos.append(
        deepconsensus_pb2.Subread(
            molecule_name=molecule_name,
            subread_strand=deepconsensus_pb2.Subread.REVERSE
            if read.alignment.position.reverse_strand else
            deepconsensus_pb2.Subread.FORWARD,
            bases=read.aligned_sequence,
            expanded_cigar=get_expanded_cigar(read),
            pw=struct_utils.get_int_field(read.info, 'pw'),
            ip=struct_utils.get_int_field(read.info, 'ip')))
  return subread_protos


@typehints.with_input_types(Tuple[str, List[reads_pb2.Read]])
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class CreateInferenceDeepConsensusInputDoFn(beam.DoFn):
  """DoFn for writing DeepConsensusInput protos consisting of Subread protos.

  DeepConsensusInput protos contain all the information that is needed for
  downstream models and can be used to create tf.Examples.
  """

  def __init__(self):
    self.no_subreads_counter = metrics.Metrics.counter(
        self.__class__, 'example_with_no_subreads')
    self.deepconsensus_input_counter = metrics.Metrics.counter(
        self.__class__, 'num_deepconsensus_input_protos_written')

  def process(
      self,
      name_and_reads: Tuple[str, List[reads_pb2.Read]],
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Yields deepconsensus_pb2.DeepConsensusInput protos for each molecule."""

    # Format of aligned_subreads_and_label is [([subreads], [label])].
    molecule_name, subreads = name_and_reads
    if not subreads:
      self.no_subreads_counter.inc()
      return
    subread_protos = get_subread_protos(subreads, molecule_name)
    sn = struct_utils.get_number_field(subreads[0].info, 'sn')
    self.deepconsensus_input_counter.inc()
    yield deepconsensus_pb2.DeepConsensusInput(
        subreads=subread_protos,
        molecule_name=molecule_name,
        molecule_start=0,
        sn=sn,
    )


class GetMoleculeNameFromSequenceName(beam.DoFn):
  """DoFn for getting the PacBio molecule name from the full sequence name."""

  def process(self, name_and_sequence):
    """Yields the molecule name and input sequence."""
    name, sequence = name_and_sequence
    yield preprocess_utils.get_pacbio_molecule_name(name), sequence


class AddCcsSequenceDoFn(beam.DoFn):
  """DoFn for adding CCS sequence into the DeepConsensusInput proto."""

  def process(self, dc_proto_info):
    """Yields given DeepConsensusInput protos with given CCS sequence added."""
    _, (dc_input, ccs_sequence) = dc_proto_info
    if not dc_input:
      return
    assert len(dc_input) == 1
    assert len(ccs_sequence) == 1
    dc_input = dc_input[0]
    ccs_sequence = ccs_sequence[0]
    dc_input_copy = copy.deepcopy(dc_input)

    # Add gaps where subreads have insertions.
    base_idx = 0
    out_of_bounds = False
    while not out_of_bounds:
      out_of_bounds, has_insert = get_index_info(dc_input_copy.subreads,
                                                 base_idx)
      # `has_insert` will only be true if we are not out of bounds, meaning
      # at least one read has a base at `base_index`.
      if has_insert:
        gap = dc_constants.GAP_OR_PAD
        ccs_sequence = ccs_sequence[:base_idx] + gap + ccs_sequence[base_idx:]
      base_idx += 1

    # Add padding at the end.
    pad_length = len(dc_input_copy.subreads[0].bases) - len(ccs_sequence)
    ccs_sequence += dc_constants.GAP_OR_PAD * pad_length
    dc_input_copy.ccs_sequence = ccs_sequence
    yield dc_input_copy
