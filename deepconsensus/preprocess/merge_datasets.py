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
r"""A pipeline for producing a unified set of read_pb2.Read protos for subreads and labels.

Example usage:

DIR=deepconsensus/testdata/ecoli
python3 -m deepconsensus.preprocess.merge_datasets \
  --input_bam=${DIR}/ecoli.subreadsToCcs.bam \
  --input_unaligned_bam=${DIR}/ecoli.subreads.bam \
  --inference=true \
  --output_path=/tmp/merge_datasets_inference

python3 -m deepconsensus.preprocess.merge_datasets \
  --input_bam=${DIR}/ecoli.subreadsToCcs.bam \
  --input_unaligned_bam=${DIR}/ecoli.subreads.bam \
  --input_label_bam=${DIR}/ecoli.truthToCcs.bam \
  --input_label_fasta=${DIR}/ecoli.truth.fasta \
  --inference=false \
  --output_path=/tmp/merge_datasets_train
"""

import os
from typing import Callable, Optional

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes

from deepconsensus.preprocess import beam_io
from deepconsensus.preprocess import merge_datasets_transforms

from nucleus.protos import reads_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('input_bam', None,
                    'Input BAM file - subreads aligned to ccs.')
flags.DEFINE_string('input_unaligned_bam', None,
                    'Input unaligned subreads BAM.')
flags.DEFINE_string(
    'input_label_bam', None,
    'Optional. Input BAM file - labels aligned to ccs. Not needed for inference.'
)
flags.DEFINE_string(
    'input_label_fasta', None,
    'Optional. Input FASTA file of label sequences. Not needed for inference.')
flags.DEFINE_string('output_path', None,
                    'Path at which output tfrecords.gz files will be created.')
flags.DEFINE_bool('inference', False,
                  'Whether we are in training or inference mode.')
flags.DEFINE_string(
    'runner', 'direct',
    'Beam runner to use. Only direct is available outside of Google.')


def create_pipeline(input_bam: str, input_unaligned_bam: str,
                    input_label_bam: Optional[str],
                    input_label_fasta: Optional[str], output_path: str,
                    inference: bool) -> Callable[..., None]:
  """Returns a pipeline for merging input datasets for subreads and label."""

  def pipeline(root):
    """Pipeline function for merging input datasets."""
    input_subreads = (
        root
        | 'read_reads' >> beam_io.ReadSam(input_bam)
        | 'reshuffle_contigs' >> beam.Reshuffle()  # to balance the shards.
        | 'remove_reads_missing_sequence' >> beam.ParDo(
            merge_datasets_transforms.RemoveReadsMissingSequenceDoFn())
        | 'filter_incorrectly_mapped_subreads' >> beam.ParDo(
            merge_datasets_transforms.RemoveIncorrectlyMappedReadsDoFn())
        | 'get_read_name' >> beam.ParDo(
            merge_datasets_transforms.GetReadNameDoFn()))

    input_unaligned_subreads = (
        root
        | 'read_unaligned_reads' >> beam_io.ReadSam(
            input_unaligned_bam, parse_aux_fields=True)
        | 'reshuffle_unaligned_reads' >> beam.Reshuffle()
        | 'get_unaligned_read_name' >> beam.ParDo(
            merge_datasets_transforms.GetReadNameDoFn()))

    _ = ((input_subreads, input_unaligned_subreads)
         | 'group_by_read_name' >> beam.CoGroupByKey()
         | 'merge_read_protos' >> beam.ParDo(
             merge_datasets_transforms.MergeSubreadsDoFn())
         | 'write_merged_subreads' >> tfrecordio.WriteToTFRecord(
             os.path.join(output_path, 'subreads/subreads'),
             file_name_suffix='.tfrecords.gz',
             coder=beam.coders.ProtoCoder(reads_pb2.Read),
             compression_type=CompressionTypes.GZIP))

    if not inference:
      label_reads = (
          root
          | 'read_label_from_bam' >> beam_io.ReadSam(input_label_bam)
          | 'reshuffle_label' >> beam.Reshuffle()  # to balance the shards
          | 'filter_incorrectly_mapped_label_reads' >> beam.ParDo(
              merge_datasets_transforms.RemoveIncorrectlyMappedReadsDoFn())
          | 'get_read_fragment_name' >> beam.ParDo(
              merge_datasets_transforms.GetReadNameDoFn()))

      label_sequences = (
          root
          | 'read_label_from_fasta' >> beam_io.ReadFastaFile(input_label_fasta)
          | 'reshuffle_label_bases' >> beam.Reshuffle()
      )  # to balance the shards

      _ = ((label_reads, label_sequences)
           | 'group_label_and_bases_by_molecule' >> beam.CoGroupByKey()
           | 'add_sequence_to_label_reads' >> beam.ParDo(
               merge_datasets_transforms.MergeLabelsDoFn())
           | 'write_merged_label' >> tfrecordio.WriteToTFRecord(
               os.path.join(output_path, 'labels/labels'),
               file_name_suffix='.tfrecords.gz',
               coder=beam.coders.ProtoCoder(reads_pb2.Read),
               compression_type=CompressionTypes.GZIP))

  return pipeline


def main(argv):
  """Main entry point."""
  if FLAGS.runner == 'direct':
    runner = beam.runners.DirectRunner()
  else:
    raise ValueError(f'Invalid runner type: {FLAGS.runner}')

  # We have to do flag validation in main rather than using
  # flags.mark_flags_as_required because beam workers don't set flags
  # appropriately.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not FLAGS.input_bam:
    raise app.UsageError('--input_bam must be specified.')
  if not FLAGS.input_unaligned_bam:
    raise app.UsageError('--input_unaligned_bam must be specified.')
  if not FLAGS.output_path:
    raise app.UsageError('--output_path must be specified.')
  if not FLAGS.inference and not FLAGS.input_label_bam:
    raise app.UsageError('--input_label_bam must specified in training mode.')
  if not FLAGS.inference and not FLAGS.input_label_fasta:
    raise app.UsageError('--input_label_fasta must specified in training mode.')

  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  runner.run(
      create_pipeline(FLAGS.input_bam, FLAGS.input_unaligned_bam,
                      FLAGS.input_label_bam, FLAGS.input_label_fasta,
                      FLAGS.output_path, FLAGS.inference), options)


if __name__ == '__main__':
  app.run(main)
