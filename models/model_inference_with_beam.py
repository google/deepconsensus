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

# Disable pyformat so that the command for running the binary does not get
# incorrectly split into multiple lines.
# pyformat: disable
r"""Inference binary for all neural network models.


<internal>

"""
# pyformat: enable

import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes

import ml_collections
from ml_collections.config_flags import config_flags
import pandas as pd
import tensorflow as tf

from deepconsensus.models import model_inference_transforms
from deepconsensus.models import model_utils
from deepconsensus.protos import deepconsensus_pb2


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('params', None, 'Training configuration.')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to checkpoint that will be loaded in.')
flags.DEFINE_string(
    'test_path', None,
    'Optional. Alternate dataset on which to run inference. If '
    'not provided, the dataset will be inferred from params.')
flags.DEFINE_string('out_dir', None,
                    'Output path for logs and optionally, model predictions.')
flags.DEFINE_string(
    'runner', 'direct',
    'Beam runner to use. Only direct is available outside of Google.')


def create_pipeline(
    out_dir: str,
    params: ml_collections.ConfigDict,
    checkpoint_path: str,
    test_path: str,
    testing: bool = False,
):
  """Returns a pipeline for running model inference."""

  def pipeline(root):
    """Pipeline function for running model inference."""
    model_utils.modify_params(params)
    if test_path:
      params.test_path = test_path

    records = (
        root
        | 'read_tf_examples' >> beam.io.ReadFromTFRecord(
            os.path.join(params.test_path, '*.tfrecords.gz'),
            compression_type=CompressionTypes.GZIP)
        | 'shuffle_tf_examples' >> beam.Reshuffle()
        | 'parse_tf_examples' >> beam.ParDo(
            model_inference_transforms.ParseTfExamplesDoFn(params=params))
        | 'run_forward_pass' >> beam.ParDo(
            model_inference_transforms.RunForwardPassDoFn(
                checkpoint_path, params)))

    # Calculate Metrics, stratified by set variables
    metric_set = [x.name for x in model_utils.get_deepconsensus_metrics()]
    metric_set += ['edit_distance']
    groups = [
        'all', 'num_passes', 'homopolymer_content', 'label_length',
        'unsup_insertion_count'
    ]
    if testing:
      # When testing, limit size of pipeline.
      metric_set = metric_set[:1]
      groups = groups[:2]
    for group in groups:


      for metric in metric_set:
        _ = records | f'{group}_{metric}' >> model_inference_transforms.StatsToCsv(
            out_dir, group, metric)


    _ = (
        records
        | 'get_proto' >> beam.Map(lambda x: x['dc_proto'])
        | 'write_preds' >> tfrecordio.WriteToTFRecord(
            os.path.join(out_dir, 'predictions/deepconsensus'),
            file_name_suffix='.tfrecords.gz',
            coder=beam.coders.ProtoCoder(deepconsensus_pb2.DeepConsensusInput),
            compression_type=CompressionTypes.GZIP))

  return pipeline


def combine_metrics(out_dir: str):
  """Combine all .stat.csv metric files for analysis."""
  metric_files = tf.io.gfile.glob(f'{out_dir}/metrics/**/*.stat.csv')
  metrics_combined = f'{out_dir}/metrics.stat.csv'
  df_set = [pd.read_csv(tf.io.gfile.GFile(f)) for f in metric_files]
  df = pd.concat(df_set).sort_values(['metric', 'group', 'group_val'])
  with tf.io.gfile.GFile(metrics_combined, 'w') as f:
    df.to_csv(f, index=False)


def main(unused_args=None):
  if FLAGS.runner == 'direct':
    runner = beam.runners.DirectRunner()
  else:
    raise ValueError(f'Invalid runner type: {FLAGS.runner}')

  if not FLAGS.params:
    params = model_utils.read_params_from_json(
        checkpoint_path=FLAGS.checkpoint_path)
  else:
    params = FLAGS.params
  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  runner.run(
      create_pipeline(FLAGS.out_dir, params, FLAGS.checkpoint_path,
                      FLAGS.test_path), options)
  combine_metrics(FLAGS.out_dir)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'out_dir',
      'checkpoint_path',
  ])
  app.run(main)
