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
"""DoFns for or producing a unified set of read_pb2.Read protos for subreads and labels."""

import copy
import logging
from typing import Iterable, List, Optional, Tuple

import apache_beam as beam
from apache_beam import metrics
from apache_beam import typehints

from deepconsensus.preprocess import preprocess_utils
from nucleus.protos import cigar_pb2
from nucleus.protos import reads_pb2
from nucleus.util import cigar as cigar_utils
from nucleus.util import sequence_utils
from nucleus.util import struct_utils

OPS_TO_CONSIDER = frozenset([
    cigar_pb2.CigarUnit.ALIGNMENT_MATCH, cigar_pb2.CigarUnit.SEQUENCE_MATCH,
    cigar_pb2.CigarUnit.INSERT, cigar_pb2.CigarUnit.DELETE,
    cigar_pb2.CigarUnit.SEQUENCE_MISMATCH, cigar_pb2.CigarUnit.CLIP_SOFT
])

OP_CHARS_TO_CONSIDER = frozenset(
    [cigar_utils.CIGAR_OPS_TO_CHAR[op] for op in OPS_TO_CONSIDER])


@typehints.with_input_types(reads_pb2.Read)
@typehints.with_output_types(reads_pb2.Read)
class RemoveReadsMissingSequenceDoFn(beam.DoFn):
  """DoFn that yields read_pb2.Read protos that contain a sequence.

  Reads that do not contain a sequence, such as in the case of secondary
  alignments produced through the minimap2 mapping software, are excluded.
  """

  def process(self, read: reads_pb2.Read) -> Iterable[reads_pb2.Read]:
    """Yields the given read proto if it contains a sequence."""
    if read.aligned_sequence:
      yield read


@typehints.with_input_types(reads_pb2.Read)
@typehints.with_output_types(reads_pb2.Read)
class RemoveIncorrectlyMappedReadsDoFn(beam.DoFn):
  """DoFn that yields read_pb2.Read protos that are mapped to correct molecule.

  Reads that have different molecule names in their fragment name and
  reference name are excluded. For PacBio data, the name is of the format
  '<movieName>/<zmw>/<indices_or_type>'. We remove the '/<indices_or_type>'
  suffix to produce the molecule name.
  """

  def process(self, read: reads_pb2.Read) -> Iterable[reads_pb2.Read]:
    """Yield the given read proto if fragment and reference match."""
    fragment_molecule = preprocess_utils.get_pacbio_molecule_name(
        read.fragment_name)
    reference_molecule = preprocess_utils.get_pacbio_molecule_name(
        read.alignment.position.reference_name)
    if fragment_molecule == reference_molecule:
      # Neither value should be None if we keep the read.
      assert fragment_molecule and reference_molecule
      yield read


@typehints.with_input_types(reads_pb2.Read)
@typehints.with_output_types(Tuple[str, reads_pb2.Read])
class GetReadNameDoFn(beam.DoFn):
  """DoFn that yields tuples of (read_name, read)."""

  def process(self,
              read: reads_pb2.Read) -> Iterable[Tuple[str, reads_pb2.Read]]:
    """Yields fragment name and the given read proto."""

    # When using Python 2, fragment_name is unicode. In other places in the
    # downstream pipeline, fragment_name is a str. When grouping by key, Beam
    # did not treat str and unicode keys as being the same. Call str here to
    # ensure that the grouping based on fragment_name happens correctly.
    yield str(read.fragment_name), read


@typehints.with_input_types(Tuple[str, Tuple[List[reads_pb2.Read],
                                             List[reads_pb2.Read]]])
@typehints.with_output_types(reads_pb2.Read)
class MergeSubreadsDoFn(beam.DoFn):
  """DoFn that adds fields from unaligned into aligned read.

  Sequence, pulse width (pw), interpulse distance (ip), and signal to noise
  ratios (sn) are added from unaligned into aligned read. Unaligned reads come
  from a PacBio-specific file. These reads are aligned to the output of PacBio's
  circular consensus sequence algorithm (pbccs) to produce the aligned reads.

  The input to the DoFn is of the form (read_name, (aligned_reads,
  unaligned_reads)), where read_name is a str, and aligned_reads and
  unaligned_reads are lists of read_pb2.Read protos that have the given
  read_name.

  Some implementation details:

  * Take the first aligned read if multiple correct alignments exist. For a
  given read, alignment is correct if 1) read has a sequence and 2) read is
  mapped to the correct molecule.

  * Throw an error if any unaligned reads are missing a sequence. All unaligned
  reads missing sequence should be filtered out prior to this function. This is
  done because it is not clear how to determine if reverse complementing the
  sequence, PW, IP is needed in these cases.

  * If unaligned read sequence is reverse complement of the aligned read
  sequence, reverse complement the PW and IP to match orientation of aligned
  read.

  * Ignore cases where unaligned read and aligned read have different sequence,
  even when one is reverse complemented. This may happen if we have hard clips.
  """

  def __init__(self):
    self.multiple_alignments_counter = metrics.Metrics.counter(
        self.__class__, 'reads_with_multiple_correct_alignments')
    self.mismatched_sequences_counter = metrics.Metrics.counter(
        self.__class__, 'reads_with_mismatching_sequences')

  def process(
      self, subread_data: Tuple[str, Tuple[List[reads_pb2.Read],
                                           List[reads_pb2.Read]]]
  ) -> Iterable[reads_pb2.Read]:
    """Merge aligned read proto and unaligned read proto."""

    (_, (aligned_reads, unaligned_reads)) = subread_data
    if not aligned_reads or not unaligned_reads:
      return
    aligned = aligned_reads[0]
    unaligned = unaligned_reads[0]

    # Do not error when there exist, for a given read, multiple alignments to
    # correct molecule. Some of the alignments may be supplementary alignments.
    # Keep a count and use the first alignment.
    if len(aligned_reads) > 1:
      logging.info('Unexpected: %d aligned reads for %s', len(aligned_reads),
                   aligned.fragment_name)
      self.multiple_alignments_counter.inc()

    # Error when an aligned read maps to multiple unaligned reads. There should
    # only be one unaligned reads in all cases.
    if len(unaligned_reads) > 1:
      raise ValueError('Unexpected: %d unaligned reads for %s' %
                       (len(unaligned_reads), unaligned.fragment_name))

    # This DoFn will be called after RemoveReadsMissingSequenceDoFn, so all
    # aligned reads should contain a sequence.
    if not aligned.aligned_sequence:
      raise ValueError('Unexpected: aligned read %s is missing a sequence.')

    sequence = aligned.aligned_sequence
    pw = struct_utils.get_int_field(unaligned.info, 'pw')
    ip = struct_utils.get_int_field(unaligned.info, 'ip')
    sn = struct_utils.get_number_field(unaligned.info, 'sn')

    # Reverse sequence and per-base fields if needed.
    if sequence != unaligned.aligned_sequence:
      reverse_complement_unaligned = sequence_utils.reverse_complement(
          unaligned.aligned_sequence)

      # <internal>
      # when aligned read had hard clipped bases, which are not returned with
      # read.aligned_sequence. For now, skip over such reads.
      if sequence != reverse_complement_unaligned:
        logging.info(
            'Unexpected: sequence present in aligned BAM is '
            'different than what is present in the unaligned BAM '
            'for %s', aligned.fragment_name)
        self.mismatched_sequences_counter.inc()
        return

      # Reverse the per-base fields since we reversed the sequence.
      # sn is not per-base so does not need to be reversed.
      sequence = reverse_complement_unaligned
      pw = pw[::-1]
      ip = ip[::-1]

    read_copy = copy.deepcopy(aligned)
    read_copy.aligned_sequence = sequence
    struct_utils.set_int_field(read_copy.info, 'pw', pw)
    struct_utils.set_int_field(read_copy.info, 'ip', ip)
    struct_utils.set_number_field(read_copy.info, 'sn', sn)
    yield read_copy


@typehints.with_input_types(Tuple[str, Tuple[List[reads_pb2.Read], List[str]]])
@typehints.with_output_types(reads_pb2.Read)
class MergeLabelsDoFn(beam.DoFn):
  """DoFn that yields a copy of the input read with missing sequence filled.

  Yields reads with sequence filled in. Sequences are either already present in
  the read or are filled in using the input sequence, which should come from a
  FASTA file.

  The input to the DoFn is of the form (read_name, (reads, sequences)), where
  read_name is a str, reads is a list of read_pb2.Read protos, and sequences is
  a list of str. Both lists are expected to have only one element.
  """

  def __init__(self):
    self.multiple_alignments_counter = metrics.Metrics.counter(
        self.__class__, 'labels_with_multiple_correct_alignments')
    self.no_full_seqs_counter = metrics.Metrics.counter(
        self.__class__, 'no_reads_with_full_seq_matching_truth')

  def process(
      self, grouped_reads_and_sequences: Tuple[str, Tuple[List[reads_pb2.Read],
                                                          str]]
  ) -> Optional[Iterable[reads_pb2.Read]]:
    """Merge aligned label read proto and label FASTA sequence."""

    (_, (reads, sequences)) = grouped_reads_and_sequences
    if not reads or not sequences:
      return

    sequence = sequences[0]

    # Keep count of when there exist, for a given label, multiple alignments to
    # correct molecule. Some of the alignments may be supplementary alignments.
    if len(reads) > 1:
      logging.info('%d aligned reads for %s', len(reads),
                   reads[0].fragment_name)
      self.multiple_alignments_counter.inc()

    if len(sequences) > 1:
      raise ValueError('Unexpected: multiple sequences for %s' %
                       reads[0].fragment_name)

    # We do not need to check if truth FASTA sequence is the reverse complement
    # of the truthToCcs BAM sequence. From PacBio README: "The truth sequence is
    # extracted from the BED interval specified in refCoords but is oriented to
    # the strand of the CCS read." Thus, truth FASTA and truthToCcs BAM
    # sequences should always be same orientation.

    # Find the read that contains a matching sequence to FASTA, or a read with
    # no sequence. We don't want reads that might have a different sequence due
    # to hard clips.
    for read in reads:
      if (read.aligned_sequence and
          read.aligned_sequence == sequence) or not read.aligned_sequence:
        read_copy = copy.deepcopy(read)
        read_copy.aligned_sequence = sequence
        return [read_copy]

    # If all reads have incorrect sequence present, skip this molecule.
    logging.info(
        'Unexpected: sequence present in truth read is '
        'different than what is present in the FASTA file for '
        '%s', reads[0].fragment_name)
    self.no_full_seqs_counter.inc()
    return
