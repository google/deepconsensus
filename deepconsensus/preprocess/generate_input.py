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
r"""A pipeline for writing out DeepConsensusInput protos.

Subreads and labels, which are reads_pb2.Read protos, are processed and
joined with BED records to produce a DeepConsensusInput proto per molecule.
The DeepConsensusInput proto contains all information needed by models
downstream and can be transformed into tf.Example protos.

Example usage:

python3 -m deepconsensus.preprocess.generate_input \
  --merged_datasets_path=deepconsensus/testdata/ecoli/output \
  --input_bed=deepconsensus/testdata/ecoli/ecoli.refCoords.bed \
  --input_ccs_fasta=deepconsensus/testdata/ecoli/ecoli.ccs.fasta \
  --output_path=/tmp/generate_input
"""

import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes

from deepconsensus.preprocess import beam_io
from deepconsensus.preprocess import generate_input_transforms
from deepconsensus.protos import deepconsensus_pb2

from nucleus.protos import reads_pb2

FLAGS = flags.FLAGS

flags.DEFINE_string('merged_datasets_path', None,
                    'Path containing merged subreads and label TFRecords.')
flags.DEFINE_string('input_bed', None, 'Input BED file.')
flags.DEFINE_string('input_ccs_fasta', None, 'Input CCS fasta file.')
flags.DEFINE_string('output_path', None,
                    'Path at which output tfrecords.gz files will be created.')
flags.DEFINE_string(
    'runner', 'direct',
    'Beam runner to use. Only direct is available outside of Google.')
flags.DEFINE_bool('inference', False,
                  'Whether we are in training or inference mode.')


def create_pipeline(merged_datasets_path: str, input_bed: str,
                    input_ccs_fasta: str, output_path: str, inference: bool):
  """Returns a pipeline for writing out DeepConsensusInput protos."""

  def pipeline(root):
    """Pipeline function for writing out DeepConsensusInput protos."""

    aligned_subreads = (
        root
        | 'read_subreads' >> beam.io.ReadFromTFRecord(
            file_pattern=os.path.join(merged_datasets_path, 'subreads/*'),
            coder=beam.coders.ProtoCoder(reads_pb2.Read),
            compression_type=CompressionTypes.GZIP)
        | 'reshuffle_subreads' >> beam.Reshuffle()  # to balance the shards
        | 'get_subread_molecule_name' >> beam.ParDo(
            generate_input_transforms.GetReadMoleculeNameDoFn())
        | 'group_by_subread_molecule' >> beam.GroupByKey()
        | 'expand_subread_fields' >> beam.ParDo(
            generate_input_transforms.ExpandFieldsRemoveSoftClipsDoFn(
                is_label=False))
        | 'indent_subread' >> beam.ParDo(
            generate_input_transforms.IndentReadsDoFn())
        | 'align_subread_sequences' >> beam.ParDo(
            generate_input_transforms.AlignSubreadSequencesDoFn())
        | 'pad_subreads' >> beam.ParDo(
            generate_input_transforms.PadSubreadsDoFn())
        |
        'align_pw_ip' >> beam.ParDo(generate_input_transforms.AlignPwIpDoFn()))

    if inference:
      dc_inputs = (
          aligned_subreads
          | 'create_deepconsensus_input' >> beam.ParDo(
              generate_input_transforms.CreateInferenceDeepConsensusInputDoFn())
      )
    else:
      input_labels = (
          root
          | 'read_labels' >> beam.io.ReadFromTFRecord(
              os.path.join(merged_datasets_path, 'labels/*'),
              coder=beam.coders.ProtoCoder(reads_pb2.Read),
              compression_type=CompressionTypes.GZIP)
          | 'reshuffle_labels' >> beam.Reshuffle()  # to balance the shards
          | 'get_label_molecule_name' >> beam.ParDo(
              generate_input_transforms.GetReadMoleculeNameDoFn())
          | 'group_by_label_molecule' >> beam.GroupByKey()
          | 'expand_label_fields' >> beam.ParDo(
              generate_input_transforms.ExpandFieldsRemoveSoftClipsDoFn(
                  is_label=True))
          | 'indent_label' >> beam.ParDo(
              generate_input_transforms.IndentReadsDoFn()))

      aligned_subreads_and_label = (
          (aligned_subreads, input_labels)
          | 'group_subreads_and_labels' >> beam.CoGroupByKey()
          | 'align_label_sequences' >> beam.ParDo(
              generate_input_transforms.AlignLabelSequencesDoFn())
          | 'pad_subreads_and_labels' >> beam.ParDo(
              generate_input_transforms.PadSubreadsAndLabelDoFn()))

      bed_records = (
          root
          | 'read_bed_records' >> beam_io.ReadBed(input_bed)
          | 'reshuffle_bed' >> beam.Reshuffle()
          | 'get_bed_molecule_name' >> beam.ParDo(
              generate_input_transforms.GetBedRecordMoleculeNameDoFn()))

      dc_inputs = (
          (aligned_subreads_and_label, bed_records)
          | 'group_by_molecule' >> beam.CoGroupByKey()
          | 'create_deepconsensus_input' >> beam.ParDo(
              generate_input_transforms.CreateTrainDeepConsensusInputDoFn()))

    if input_ccs_fasta:
      ccs_sequences = (
          root
          | 'read_fasta' >> beam_io.ReadFastaFile(input_ccs_fasta)
          | 'reshuffle_ccs_fasta' >> beam.Reshuffle()
          | 'get_ccs_molecule_name' >> beam.ParDo(
              generate_input_transforms.GetMoleculeNameFromSequenceName()))

      dc_inputs = (
          dc_inputs
          |
          'get_dc_molecule_name' >> beam.Map(lambda dc: (dc.molecule_name, dc)))

      dc_inputs = ((dc_inputs, ccs_sequences)
                   | 'group_dc_and_ccs' >> beam.CoGroupByKey()
                   | beam.ParDo(generate_input_transforms.AddCcsSequenceDoFn()))

    _ = (
        dc_inputs
        | 'write_deepconsensus_input' >> tfrecordio.WriteToTFRecord(
            os.path.join(output_path, 'deepconsensus/deepconsensus'),
            file_name_suffix='.tfrecords.gz',
            coder=beam.coders.ProtoCoder(deepconsensus_pb2.DeepConsensusInput),
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
  if not FLAGS.merged_datasets_path:
    raise app.UsageError('--merged_datasets_path must be specified.')
  if not FLAGS.inference and not FLAGS.input_bed:
    raise app.UsageError('--input_bed must be specified for training mode.')
  if not FLAGS.output_path:
    raise app.UsageError('--output_path must be specified.')

  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  runner.run(
      create_pipeline(FLAGS.merged_datasets_path, FLAGS.input_bed,
                      FLAGS.input_ccs_fasta, FLAGS.output_path,
                      FLAGS.inference), options)


if __name__ == '__main__':
  app.run(main)
