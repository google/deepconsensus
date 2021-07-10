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
r"""A pipeline for writing out FASTA files from predictions.

Example usage:

INPUT=deepconsensus/testdata/ecoli/output/predictions/deepconsensus*.tfrecords.gz
python3 -m deepconsensus.postprocess.stitch_predictions \
  --input_file=${INPUT} \
  --output_path=/tmp/stitched_predictions \
  --example_width=100
"""

import json
import os
from typing import Optional

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io.filesystem import CompressionTypes
import tensorflow as tf

from deepconsensus.models import model_utils
from deepconsensus.postprocess import stitch_predictions_transforms
from deepconsensus.protos import deepconsensus_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', None, 'Input TFRecords files.')
flags.DEFINE_boolean(
    'fill_n', False, 'Output failed sequence windows using N'
    'placeholder sequences')
flags.DEFINE_string('output_path', None,
                    'Path at which output tfrecords.gz files will be created.')
flags.DEFINE_integer('example_width', None, 'Width of examples used.')
flags.DEFINE_string(
    'runner', 'direct',
    'Beam runner to use. Only direct is available outside of Google.')


def get_unpadded_example_width(input_file_pattern: str) -> int:
  """Returns the unpadded example width for data used to produce input files."""
  checkpoint_path = os.path.join(
      input_file_pattern.split('postprocess')[0], 'model')
  params = model_utils.read_params_from_json(checkpoint_path)
  dataset_dir = os.path.dirname(params.train_path)
  # Trying to be robust to changes in the directory structure.
  while dataset_dir:
    dataset_params_path = os.path.join(dataset_dir, 'dataset_summary.json')
    if tf.io.gfile.exists(dataset_params_path):
      break
    dataset_dir = os.path.dirname(dataset_dir)
  assert dataset_params_path
  dataset_params = json.load(tf.io.gfile.GFile(dataset_params_path, 'r'))
  return int(dataset_params['EXAMPLE_WIDTH'])


def create_pipeline(input_file: str,
                    output_path: str,
                    example_width: Optional[int] = None):
  """Returns a pipeline for writing out DeepConsensusInput protos."""

  if example_width is None:
    example_width = get_unpadded_example_width(input_file_pattern=input_file)

  def pipeline(root):
    """Pipeline function for writing out DeepConsensusInput protos."""
    _ = (
        root
        | 'read_deepconsensus_input' >> beam.io.ReadFromTFRecord(
            file_pattern=input_file,
            coder=beam.coders.ProtoCoder(deepconsensus_pb2.DeepConsensusInput),
            compression_type=CompressionTypes.GZIP)
        | 'reshuffle_subreads' >> beam.Reshuffle()  # to balance the shards
        | 'get_molecule_name' >> beam.Map(lambda dc: (dc.molecule_name, dc))
        | 'group_by_molecule' >> beam.GroupByKey()
        | 'get_full_sequence_and_molecule_name' >> beam.ParDo(
            stitch_predictions_transforms.GetFullSequenceDoFn(
                example_width=example_width, fill_n=FLAGS.fill_n))
        | 'convert_to_fasta_str' >> beam.ParDo(
            stitch_predictions_transforms.ConvertToFastqStrDoFn())
        # Note, writing a single shard can be slow for large amounts of data. If
        # the pipeline is taking too long to complete, you can comment out the
        # line below (`num_shards=1`) and instead manually join the shards into
        # one final FASTA.
        | 'write_fastq' >> beam.io.WriteToText(
            os.path.join(output_path, 'full_predictions'),
            file_name_suffix='.fastq',
            append_trailing_newlines=False,
            compression_type=CompressionTypes.UNCOMPRESSED,
        ))

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
  if not FLAGS.input_file:
    raise app.UsageError('--input_file must be specified.')
  if not FLAGS.output_path:
    raise app.UsageError('--output_path must be specified.')

  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  runner.run(
      create_pipeline(FLAGS.input_file, FLAGS.output_path, FLAGS.example_width),
      options)


if __name__ == '__main__':
  app.run(main)
