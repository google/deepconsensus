# Copyright (c) 2021, Google Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of Google Inc. nor the names of its
#    contributors may be used to endorse or promote products derived from this
#    software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""DoFns for converting DeepConsensusInput protos into tf.Example protos."""

import copy
import itertools
import os
import random
from typing import Any, Iterable, Optional
import apache_beam as beam
from apache_beam import metrics
from apache_beam import typehints
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes
import tensorflow as tf

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.tf_examples import tf_example_utils
from deepconsensus.utils import dc_constants
from nucleus.io import fasta
from nucleus.io import vcf
from nucleus.util import ranges


class ProcessAndWriteTfExamples(beam.PTransform):
  """A PTransform that processes DeepConsensusInputs and writes tf.Examples."""

  def __init__(self, reference_fasta, example_width, example_height, truth_vcf,
               species, split, output_path, truth_bed, padded_len,
               window_overlap_step, subread_permutations, inference):
    self.reference_fasta = reference_fasta
    self.example_width = example_width
    self.example_height = example_height
    self.truth_vcf = truth_vcf
    self.species = species
    self.split = split
    self.output_path = output_path
    self.truth_bed = truth_bed
    self.padded_len = padded_len or example_width
    self.window_overlap_step = window_overlap_step
    self.subread_permutations = subread_permutations
    self.inference = inference

  def expand(self, dc_input: deepconsensus_pb2.DeepConsensusInput) -> None:
    """Processes the input PCollection and writes out tf.Examples."""

    if not self.inference and self.species == 'human' and self.reference_fasta:
      dc_input = (
          dc_input
          | f'add_label_bases_position_{self.split}' >> beam.ParDo(
              AddLabelBasesPositionDoFn(reference_fasta=self.reference_fasta)))

    dc_input = (
        dc_input
        | f'chunk_windows_{self.example_width}_{self.split}' >> beam.ParDo(
            GetSmallerWindowDoFn(
                example_width=self.example_width,
                window_overlap_step=self.window_overlap_step,
                inference=self.inference)))

    if not self.inference and self.species == 'human' and self.truth_vcf:
      dc_input = (
          dc_input
          | f'filter_variants_{self.split}' >> beam.ParDo(
              FilterVariantWindowsDoFn(truth_vcf=self.truth_vcf)))

    if not self.inference and self.species == 'human' and self.truth_bed:
      dc_input = (
          dc_input
          | f'filter_nonconfident_regions_{self.split}' >> beam.ParDo(
              FilterNonconfidentRegionsDoFn(truth_bed=self.truth_bed)))

    if self.padded_len is not None:
      dc_input = (
          dc_input
          | f'pad_examples_{self.split}' >> beam.ParDo(
              PadExamplesDoFn(
                  padded_len=self.padded_len, inference=self.inference)))

    # Do not remove empty subreads.
    dc_input = (
        dc_input
        | f'remove_sequences_with_invalid_bases_{self.split}' >> beam.ParDo(
            RemoveSequencesWithInvalidBasesDoFn(inference=self.inference)))

    if self.subread_permutations > 0:
      dc_input = (
          dc_input
          | f'subread_permutations_{self.split}' >> beam.ParDo(
              SubreadPermutationsDoFn(self.subread_permutations)))

    # Order of elements does not matter at inference time.
    if not self.inference:
      dc_input = (dc_input | f'shuffle_{self.split}' >> beam.Reshuffle())

    _ = (
        dc_input
        | f'convert_to_tf_ex_{self.split}' >> beam.ParDo(
            ConvertToTfExamplesDoFn(
                example_height=self.example_height, inference=self.inference))
        | f'write_{self.split}' >> tfrecordio.WriteToTFRecord(
            os.path.join(self.output_path, f'{self.split}/{self.split}'),
            file_name_suffix='.tfrecords.gz',
            coder=beam.coders.ProtoCoder(tf.train.Example),
            compression_type=CompressionTypes.GZIP))


class FilterNonconfidentRegionsDoFn(beam.DoFn):
  """DoFn for filtering out windows from nonconfident regions."""

  def __init__(self, truth_bed: str):
    self.inside_confident_regions_counter = metrics.Metrics.counter(
        self.__class__, 'inside_confident_regions')
    self.outside_confident_regions_counter = metrics.Metrics.counter(
        self.__class__, 'outside_confident_regions')
    self.truth_bed = truth_bed
    self.confident_regions = ranges.RangeSet.from_bed(self.truth_bed)

  def process(
      self, dc_input: deepconsensus_pb2.DeepConsensusInput
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Yields the input proto if it is fully covered by confident regions."""

    start, end = tf_example_utils.get_label_start_end(
        dc_input.label.base_positions, dc_input.strand)
    # If we cannot determine the start and end for this window, try using start
    # and end for the full molecule.
    if start is None or end is None:
      start = dc_input.chrom_start
      end = dc_input.chrom_end
    if self.confident_regions.envelops(dc_input.chrom_name, start, end):
      self.inside_confident_regions_counter.inc()
      yield dc_input
    else:
      self.outside_confident_regions_counter.inc()


class AddLabelBasesPositionDoFn(beam.DoFn):
  """DoFn for adding the reference position of each base in the label."""

  def __init__(self, reference_fasta: str):
    self.discarded_molecules_counter = metrics.Metrics.counter(
        self.__class__, 'discarded_molecules')
    self.total_molecules_counter = metrics.Metrics.counter(
        self.__class__, 'total_molecules')
    self.reference_fasta = reference_fasta
    self.ref_reader = None

  def setup(self):
    if self.ref_reader is None:
      self.ref_reader = fasta.IndexedFastaReader(self.reference_fasta)

  def process(
      self, dc_input: deepconsensus_pb2.DeepConsensusInput
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Adds a list of reference positions for each label base to given input."""

    dc_input_range = ranges.make_range(dc_input.chrom_name,
                                       dc_input.chrom_start, dc_input.chrom_end)
    # <internal>
    # In case the setup function didn't run.
    if self.ref_reader is None:
      self.ref_reader = fasta.IndexedFastaReader(self.reference_fasta)
    forward_ref_sequence = self.ref_reader.query(dc_input_range)
    # The forward_ref_sequence might not be the correct orientation. Grab the
    # reference sequence of the same strand as dc_input.
    outputs = tf_example_utils.get_ref_and_start_and_offset(
        forward_ref_sequence, dc_input.strand, dc_input.chrom_start,
        dc_input.chrom_end)
    ref_sequence, curr_pos, to_add = outputs
    label_sequence = tf_example_utils.get_sequence_without_gaps_or_padding(
        dc_input.label.bases)

    if ref_sequence != label_sequence:
      self.discarded_molecules_counter.inc()
      return

    # Save the position for each label base, or -1 for padding and gap tokens.
    label_base_positions = []
    for base in dc_input.label.bases:
      if base not in [dc_constants.GAP_OR_PAD, dc_constants.GAP_OR_PAD]:
        label_base_positions.append(curr_pos)
        curr_pos += to_add
      else:
        label_base_positions.append(-1)

    dc_input_copy = copy.deepcopy(dc_input)
    dc_input_copy.label.base_positions.extend(label_base_positions)
    self.total_molecules_counter.inc()
    yield dc_input_copy


def get_windowed_subread(subread: deepconsensus_pb2.Subread,
                         molecule_name: str,
                         example_width: int,
                         start: int,
                         is_label: bool,
                         label_start: Optional[int] = None,
                         label_end: Optional[int] = None):
  """Returns a window of the input subread.

  A new subread of `example_width` from the `start` index is produced.

  Args:
    subread: deepconsensus_pb2.Subread. subread that will be windowed
    molecule_name: str. Name of molecule to which this subread belongs.
    example_width: int. Size of window to extract from input subread.
    start: int. Index at which window starts.
    is_label: bool. Whether the input is a label. Pulse width (pw) and
      interpulse distance (ip) will only be modified if the input is not a
      label, as labels do not contain these fields.
    label_start: int. Start index to slice for this window's label.
    label_end: int. End index to slice for this window's label.

  Returns:
    A new subread representing a window of `example_width` from the input
      subread.
  """

  end = start + example_width
  if is_label:
    start = label_start
    end = label_end
  windowed_bases = subread.bases[start:end]
  windowed_cigar = subread.expanded_cigar[start:end]
  windowed_pw = subread.pw[start:end]
  windowed_ip = subread.ip[start:end]
  windowed_base_positions = subread.base_positions[start:end]

  assert len(windowed_bases) == len(windowed_cigar)
  if len(windowed_bases) < example_width:
    pad_len = example_width - len(windowed_bases)
    str_padding = dc_constants.GAP_OR_PAD * pad_len
    lst_padding = [dc_constants.GAP_OR_PAD_INT] * pad_len
    windowed_bases += str_padding
    windowed_cigar += str_padding

    # pw and ip fields for label should be empty.
    if not is_label:
      windowed_pw += lst_padding
      windowed_ip += lst_padding

    # We only want to add padding if windowed_base_positions exists and is not
    # the correct length. If it doesn't exist, which will be the case for E.
    # Coli data, we do not add padding.
    if is_label and windowed_base_positions:
      base_positions_padding = [-1] * pad_len
      windowed_base_positions += base_positions_padding

  windowed_subread = deepconsensus_pb2.Subread(
      molecule_name=molecule_name,
      bases=windowed_bases,
      expanded_cigar=windowed_cigar,
      pw=windowed_pw,
      ip=windowed_ip,
      base_positions=windowed_base_positions,
      subread_strand=subread.subread_strand)

  return windowed_subread


@typehints.with_input_types(Any)
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class GetSmallerWindowDoFn(beam.DoFn):
  """Cuts DeepConsensusInput fields into smaller windows.

  Sequence, cigar, pulse width (pw) and interpulse distance (ip) for each
  window are broken into windows of `example_width` size.
  """

  def __init__(
      self,
      example_width: int,
      inference: bool,
      window_overlap_step: Optional[int] = None,
      proto_class: str = 'DeepConsensusInput',
  ):
    self.small_windows_counter = metrics.Metrics.counter(
        self.__class__, 'small_windows')
    self.unsup_insertions_total = metrics.Metrics.counter(
        self.__class__, 'unsup_insertions_total')

    self.example_width = example_width
    self.window_overlap_step = window_overlap_step
    if (proto_class is not None and
        proto_class not in ['DeepConsensusInput', 'Example']):
      raise ValueError('Unexpected proto_class: %s.' % proto_class)
    self.proto_class = proto_class
    self.inference = inference

  def process(
      self, input_proto: Any) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Breaks deepconsensus_input up into multiple windows of smaller width."""
    # If the input is a tf.Example, first parse it.
    if self.proto_class == 'Example':
      deepconsensus_input = deepconsensus_pb2.DeepConsensusInput()
      deepconsensus_input.ParseFromString(
          input_proto.features.feature['deepconsensus_input/encoded'].bytes_list
          .value[0])
    else:
      deepconsensus_input = input_proto

    # Explicitly cast to str for downstream type checks.
    # The default in Python 2 is unicode.
    molecule_name = str(deepconsensus_input.molecule_name)
    label_end = None
    range_step = self.window_overlap_step or self.example_width
    for start in range(0, len(deepconsensus_input.subreads[0].bases),
                       range_step):

      if not self.inference:
        # Get appropriate window from the label.
        if not deepconsensus_input.subread_indices:
          label_start = start
        elif len(deepconsensus_input.subread_indices) <= start:
          label_start = label_end
        else:
          # Shift back by at least one or more if the first position in the
          # window contains unsupported insertions.
          shift_back = 1 + deepconsensus_input.unsup_insertions_by_pos.get(
              start, 0)
          label_start = deepconsensus_input.subread_indices[start] - shift_back

        # Subtract one since we will grab one element using this index, not
        # slice.
        label_end_index = start + self.example_width - 1
        if not deepconsensus_input.subread_indices:
          label_end = label_start + self.example_width
        if len(deepconsensus_input.subread_indices) <= label_end_index:
          label_end = label_start + self.example_width
        else:
          label_end = deepconsensus_input.subread_indices[label_end_index]

        windowed_label = get_windowed_subread(
            subread=deepconsensus_input.label,
            molecule_name=molecule_name,
            example_width=self.example_width,
            start=start,
            is_label=True,
            label_start=label_start,
            label_end=label_end)

      # Get appropriate window from each subread.
      windowed_subreads = []
      for subread in deepconsensus_input.subreads:
        windowed_subread = get_windowed_subread(
            subread=subread,
            molecule_name=molecule_name,
            example_width=self.example_width,
            start=start,
            is_label=False)
        windowed_subreads.append(windowed_subread)

      # Discard windows that do not have any subread data.
      if not windowed_subreads:
        return
      # If CCS sequence is not present, windowed sequence will also be empty.
      end = start + self.example_width
      windowed_ccs_sequence = deepconsensus_input.ccs_sequence[start:end]

      # We also pad everything downstream in PadExamplesDoFn, so this code is
      # redundant in the full pipeline if we are adding padding to each window.
      # However, we need this if we aren't padding each window and also for
      # testing.
      if windowed_ccs_sequence and len(
          windowed_ccs_sequence) != self.example_width:
        pad_len = self.example_width - len(windowed_ccs_sequence)
        str_padding = dc_constants.GAP_OR_PAD * pad_len
        windowed_ccs_sequence += str_padding

      if self.inference:
        dc_input = deepconsensus_pb2.DeepConsensusInput(
            subreads=windowed_subreads,
            molecule_name=molecule_name,
            molecule_start=start,
            sn=deepconsensus_input.sn,
            ccs_sequence=windowed_ccs_sequence)
      else:
        unsup_insertion_count = sum(
            count for pos, count in
            deepconsensus_input.unsup_insertions_by_pos.items()
            if pos >= start and pos < end)
        self.unsup_insertions_total.inc(unsup_insertion_count)

        dc_input = deepconsensus_pb2.DeepConsensusInput(
            label=windowed_label,
            subreads=windowed_subreads,
            molecule_name=molecule_name,
            molecule_start=start,
            chrom_name=deepconsensus_input.chrom_name,
            chrom_start=deepconsensus_input.chrom_start,
            chrom_end=deepconsensus_input.chrom_end,
            sn=deepconsensus_input.sn,
            strand=deepconsensus_input.strand,
            ccs_sequence=windowed_ccs_sequence,
            unsup_insertion_count=unsup_insertion_count)
      self.small_windows_counter.inc()
      yield dc_input


@typehints.with_input_types(deepconsensus_pb2.DeepConsensusInput)
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class FilterVariantWindowsDoFn(beam.DoFn):
  """DoFn for filtering windows with known variants."""

  def __init__(self,
               truth_vcf: str,
               slack: Optional[int] = None):
    self.vcf_path = truth_vcf
    self.windows_with_variants_counter = metrics.Metrics.counter(
        self.__class__, 'windows_with_variants')
    self.windows_no_variants_counter = metrics.Metrics.counter(
        self.__class__, 'windows_no_variants')
    self.vcf_reader = None
    self.slack = slack

  def setup(self):
    if self.vcf_reader is None:
      self.vcf_reader = vcf.VcfReader(self.vcf_path)

  def process(self, deepconsensus_input):
    """Yields inputs if the corresponding region does not contain variants."""

    start, end = tf_example_utils.get_label_start_end(
        label_base_positions=deepconsensus_input.label.base_positions,
        strand=deepconsensus_input.strand)
    if start is None or end is None:
      self.windows_no_variants_counter.inc()
      return [deepconsensus_input]

    if self.slack is not None:
      start -= self.slack
      end += self.slack
    start = max(0, start)
    query_region = ranges.make_range(
        chrom=deepconsensus_input.chrom_name, start=start, end=end)
    # <internal>
    # In case the setup function didn't run.
    if self.vcf_reader is None:
      self.vcf_reader = vcf.VcfReader(self.vcf_path)
    with self.vcf_reader.query(query_region) as variants:
      variants_list = list(variants)
      if not variants_list:
        self.windows_no_variants_counter.inc()
        return [deepconsensus_input]
      else:
        self.windows_with_variants_counter.inc()
        return


@typehints.with_input_types(deepconsensus_pb2.DeepConsensusInput)
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class CreateExamplesDoFn(beam.DoFn):
  """DoFn for assigning examples to either the training or evaluation set."""

  def __init__(self, filter_set: str, species: str):
    self.filter_set = filter_set
    self.species = species

  def process(
      self, deepconsensus_input: deepconsensus_pb2.DeepConsensusInput
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Assigns input to train/eval/test set based on genomic end position."""

    in_train_region, in_eval_region, in_test_region = tf_example_utils.check_region(
        deepconsensus_input, self.species, {})

    assert sum([in_train_region, in_eval_region, in_test_region]) <= 1

    if in_train_region and self.filter_set == 'train':
      yield deepconsensus_input

    if in_eval_region and self.filter_set == 'eval':
      yield deepconsensus_input

    if in_test_region and self.filter_set == 'test':
      yield deepconsensus_input


@typehints.with_input_types(deepconsensus_pb2.DeepConsensusInput)
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class PadExamplesDoFn(beam.DoFn):
  """DoFn for padding DeepConsensusInput protos."""

  def __init__(self, padded_len: int, inference: bool):
    self.padded_len = padded_len
    self.windows_longer_than_padded_len_counter = metrics.Metrics.counter(
        self.__class__, 'windows_longer_than_padded_len')
    self.inference = inference

  def process(
      self, deepconsensus_input: deepconsensus_pb2.DeepConsensusInput
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Pads examples to allow for insertions."""
    deepconsensus_input_copy = copy.deepcopy(deepconsensus_input)

    # Pad the subreads and label to fixed width.
    all_reads = list(deepconsensus_input_copy.subreads)
    if not self.inference:
      all_reads += [deepconsensus_input_copy.label]
    for read in all_reads:
      if len(read.bases) > self.padded_len:
        self.windows_longer_than_padded_len_counter.inc()
        return
      tf_example_utils.pad_bases_pw_ip_cigar(read, self.padded_len)

    ccs_sequence = deepconsensus_input_copy.ccs_sequence
    pad_len = self.padded_len - len(ccs_sequence)
    padding = dc_constants.GAP_OR_PAD * pad_len
    padded_ccs_sequence = ccs_sequence + padding
    deepconsensus_input_copy.ccs_sequence = padded_ccs_sequence
    if deepconsensus_input_copy.subreads:
      yield deepconsensus_input_copy


@typehints.with_input_types(deepconsensus_pb2.DeepConsensusInput)
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class RemoveSequencesWithInvalidBasesDoFn(beam.DoFn):
  """DoFn for filtering out sequences containing invalid bases.

  If a subread contains bases outside the DeepConsensus vocab, the subread is
  ignored. If the label contains invalid bases, the entire example is thrown
  out.
  """

  def __init__(self, inference: bool):
    self.subreads_outside_vocab_counter = metrics.Metrics.counter(
        self.__class__, 'subreads_outside_vocab')
    self.labels_outside_vocab_counter = metrics.Metrics.counter(
        self.__class__, 'examples_with_label_outside_vocab')
    self.inference = inference

  def process(
      self, dc_input: deepconsensus_pb2.DeepConsensusInput
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    """Yields the input with sequences containing invalid bases removed."""
    dc_input_copy = copy.deepcopy(dc_input)
    subreads_to_keep = []
    # Keep only subreads with bases from the vocab.
    for subread in dc_input_copy.subreads:
      if not set(subread.bases) - set(dc_constants.VOCAB):
        subreads_to_keep.append(subread)
      else:
        self.subreads_outside_vocab_counter.inc()
    if self.inference and set(dc_input_copy.label.bases) - set(
        dc_constants.VOCAB):
      # If the label contains bases outside the vocab, ignore this example.
      self.labels_outside_vocab_counter.inc()
      return
    del dc_input_copy.subreads[:]
    dc_input_copy.subreads.extend(subreads_to_keep)
    yield dc_input_copy


@typehints.with_input_types(deepconsensus_pb2.DeepConsensusInput)
@typehints.with_output_types(tf.train.Example)
class ConvertToTfExamplesDoFn(beam.DoFn):
  """DoFn that writes out tensorflow.Examples with subread and label sequences.

  Each base in the sequence is encoded as a float. These tf.Examples are used
  as inputs for sequence-only models. See tf_example_utils.get_total_rows for
  details on the structure of the tf.Example.
  """

  def __init__(self, example_height: int, inference: bool):
    if example_height <= 0:
      raise ValueError('Example height must both be > 0.')
    self.example_height = example_height
    self.examples_with_discarded_subreads = metrics.Metrics.counter(
        self.__class__, 'example_with_discarded_subreads')
    self.examples_no_subreads_counter = metrics.Metrics.counter(
        self.__class__, 'examples_no_subreads')
    self.subreads_counter = metrics.Metrics.counter(self.__class__,
                                                    'subreads_counter')
    self.subreads_reverse_strand_counter = metrics.Metrics.counter(
        self.__class__, 'subreads_reverse_strand')
    self.total_examples_counter = metrics.Metrics.counter(
        self.__class__, 'total_examples')
    self.inference = inference

  def process(self, deepconsensus_input):
    """Yields tf.Example created from the given DeepConsensusInput proto."""
    counters = {
        'examples_with_discarded_subreads':
            self.examples_with_discarded_subreads,
        'examples_no_subreads_counter':
            self.examples_no_subreads_counter,
    }
    example = tf_example_utils.deepconsensus_input_to_example(
        deepconsensus_input=deepconsensus_input,
        example_height=self.example_height,
        counters=counters,
        inference=self.inference)
    if example is not None:
      n_reverse = sum([
          x.subread_strand == deepconsensus_pb2.Subread.REVERSE
          for x in deepconsensus_input.subreads
      ])
      self.subreads_reverse_strand_counter.inc(n_reverse)
      self.subreads_counter.inc(len(deepconsensus_input.subreads))
      self.total_examples_counter.inc()
      yield example


@typehints.with_input_types(deepconsensus_pb2.DeepConsensusInput)
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class SubreadPermutationsDoFn(beam.DoFn):
  """DoFn for permuting subreads."""

  def __init__(self, n_permutations: int):
    # Permutations + 1 for the original example
    self.n_examples = n_permutations + 1

  def process(
      self, dc_proto: deepconsensus_pb2.DeepConsensusInput
  ) -> Iterable[deepconsensus_pb2.DeepConsensusInput]:
    n_subreads = len(dc_proto.subreads)
    hash_set = {}
    for p_count, subread_set in enumerate(
        itertools.permutations(dc_proto.subreads)):
      if p_count >= self.n_examples:
        break
      dc_proto_out = copy.deepcopy(dc_proto)
      del dc_proto_out.subreads[:]
      subread_set = list(subread_set)
      hash_set = set()
      if n_subreads > self.n_examples:
        # When n_subreads > n_permutations, use hash to ensure no duplicates.
        while True:
          # Do not shuffle the first iteration as it is the unaugmented example.
          if p_count > 0:
            random.shuffle(subread_set)
          subread_set_str = str(subread_set)
          if subread_set_str not in hash_set:
            hash_set.add(subread_set_str)
            break
      dc_proto_out.subreads.extend(subread_set)
      yield dc_proto_out


@typehints.with_input_types(deepconsensus_pb2.DeepConsensusInput)
@typehints.with_output_types(deepconsensus_pb2.DeepConsensusInput)
class DownSample(beam.DoFn):
  """Sub samples elements from Pcollection."""

  def __init__(self, rate, input_dataset):
    self.rate = rate
    self.input_dataset = input_dataset
    self.before_subsample = metrics.Metrics.counter(
        self.__class__, f'before_subsample_count_{input_dataset}')
    self.after_subsample = metrics.Metrics.counter(
        self.__class__, f'after_subsample_count_{input_dataset}')

  def process(self, item):
    self.before_subsample.inc()
    if random.uniform(0, 1) < self.rate:
      self.after_subsample.inc()
      yield item
