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
r"""A workflow to compute consensus sequence from per-position majority vote.

Example usage:


INPUT=deepconsensus/testdata/human/output/deepconsensus/deepconsensus-00000-of-00001.tfrecords.gz
python3 -m deepconsensus.models.run_majority_vote_model \
  --input_tfrecords_path=${INPUT} \
  --write_errors=true \
  --output_path=/tmp/ \
  --proto_class='DeepConsensusInput'

This command will not produce any output files unless you set the --write_errors
flag and provide an output path.

"""

import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io import filesystem
from apache_beam.io import tfrecordio
import tensorflow as tf


from deepconsensus.models import majority_vote_transforms
from deepconsensus.models import model_utils
from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.tf_examples import tf_example_transforms

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_tfrecords_path', None,
    'Full path containing the TFRecords directory along with '
    'the sharded naming pattern for TFRecords. For example, '
    '/path/to/directory/output@*.tfrecords.gz')
flags.DEFINE_integer(
    'example_width', None, 'Number of bases to split molecule '
    'into before running the model. Will not affect accuracy '
    'and is intended to allow us to write out smaller windows '
    'of the subread where we have incorrect predictions.')
flags.DEFINE_boolean(
    'write_errors', False,
    'Whether to write out windows in which majority vote was incorrect. '
    'If true, the errors will be written to deepconsensus/ directory under '
    '--output_path. Note that the prefix of the output changes based on the '
    '--proto_class flag.')
flags.DEFINE_string(
    'output_path', None,
    'Output path for examples in which majority vote was incorrect.')
flags.DEFINE_float('eval_fraction', 0.1, 'Fraction of examples in eval set.')
flags.DEFINE_integer('chromosome_size', 4642522,
                     'Number of bases in the chromosome being processed.')
flags.DEFINE_enum(
    'proto_class', 'DeepConsensusInput', ['DeepConsensusInput', 'Example'],
    'Class type for the input records in --input_tfrecords_path. '
    'The output records in --write_errors will also be the same '
    'type.')
flags.DEFINE_string(
    'runner', 'direct',
    'Beam runner to use. Only direct is available outside of Google.')


def create_pipeline(input_tfrecords_path: str, example_width: int,
                    write_errors: bool, output_path: str, proto_class: str):
  """Returns a pipeline for running the majority vote baseline."""

  def _extract_deepconsensus_input(example):
    retevl = deepconsensus_pb2.DeepConsensusInput()
    retevl.ParseFromString(
        example.features.feature['deepconsensus_input/encoded'].bytes_list
        .value[0])
    return retevl

  def pipeline(root):
    """Pipeline function for running the majority vote baseline."""
    if proto_class == 'DeepConsensusInput':
      deepconsensus_input = (
          root
          | 'read_deepconsensus_input' >> tfrecordio.ReadFromTFRecord(
              input_tfrecords_path,
              coder=beam.coders.ProtoCoder(
                  deepconsensus_pb2.DeepConsensusInput)))
    elif proto_class == 'Example':
      modified_input_tfrecords_path = input_tfrecords_path.replace(
          '@*.tfrecords.gz', '')
      # In case the pattern was * instead of @*.
      modified_input_tfrecords_path = modified_input_tfrecords_path.replace(
          '*.tfrecords.gz', '')
      example_height = model_utils.extract_example_height(
          modified_input_tfrecords_path)
      deepconsensus_input = (
          root
          | 'read_tf_example' >> tfrecordio.ReadFromTFRecord(
              input_tfrecords_path,
              coder=beam.coders.ProtoCoder(tf.train.Example))
          | 'get_deepconsensus_input' >> beam.Map(_extract_deepconsensus_input))
    else:
      raise ValueError('Unexpected record type: %s' % proto_class)

    mv_input = (deepconsensus_input
                | 'reshuffle' >> beam.Reshuffle())  # to balance the shards.

    if example_width is not None:
      mv_input = (
          mv_input
          | 'chunk_windows_%d' % example_width >> beam.ParDo(
              tf_example_transforms.GetSmallerWindowDoFn(
                  example_width, inference=False)))

    mv_output = (
        mv_input
        | 'get_consensus_from_majority_vote' >> beam.ParDo(
            majority_vote_transforms.GetConsensusFromMajorityVoteDoFn())
        | 'count_matches_sequence_only' >> beam.ParDo(
            majority_vote_transforms.CountMatchesFromSequenceDoFn()))

    if write_errors:
      hard_examples = (
          mv_output
          | 'get_hard_ex' >> beam.ParDo(
              majority_vote_transforms.GetHardExamplesDoFn()))
      if proto_class == 'Example':
        hard_examples = (
            hard_examples
            | 'convert_to_tf_ex_train' >> beam.ParDo(
                tf_example_transforms.ConvertToTfExamplesDoFn(
                    example_height=example_height, inference=False))
            | 'write_hard_ex' >> tfrecordio.WriteToTFRecord(
                os.path.join(output_path, 'deepconsensus/tf_examples'),
                file_name_suffix='.tfrecords.gz',
                coder=beam.coders.ProtoCoder(tf.train.Example),
                compression_type=filesystem.CompressionTypes.GZIP))
      else:
        _ = (
            hard_examples
            | 'write_hard_ex' >> tfrecordio.WriteToTFRecord(
                os.path.join(output_path, 'deepconsensus/deepconsensus'),
                file_name_suffix='.tfrecords.gz',
                coder=beam.coders.ProtoCoder(
                    deepconsensus_pb2.DeepConsensusInput),
                compression_type=filesystem.CompressionTypes.GZIP))

  return pipeline


def main(unused_args=None):
  """Main entry point."""
  if FLAGS.runner == 'direct':
    runner = beam.runners.DirectRunner()
  else:
    raise ValueError(f'Invalid runner type: {FLAGS.runner}')

  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)

  if FLAGS.input_tfrecords_path is None:
    raise app.UsageError('Must specify --input_tfrecords_path.')
  if FLAGS.write_errors and FLAGS.output_path is None:
    raise app.UsageError(
        'Must specify --output_path if --write_errors is True.')

  runner.run(
      create_pipeline(
          input_tfrecords_path=FLAGS.input_tfrecords_path,
          example_width=FLAGS.example_width,
          write_errors=FLAGS.write_errors,
          output_path=FLAGS.output_path,
          proto_class=FLAGS.proto_class), options)


if __name__ == '__main__':
  app.run(main)
