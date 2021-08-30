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

# Disable pyformat so that the command for running the binary does not get
# incorrectly split into multiple lines.
# pyformat: disable
r"""Inference binary for all neural network models.


<internal>

"""
# pyformat: enable

import os
from typing import Optional

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
    'dataset_path', None,
    'Optional. Alternate dataset on which to run inference. If '
    'not provided, the dataset will be inferred from params.')
flags.DEFINE_string('out_dir', None,
                    'Output path for logs and optionally, model predictions.')
flags.DEFINE_bool('inference', False,
                  'Whether we are in training or inference mode.')
flags.DEFINE_integer('max_passes', 20, 'Maximum subreads in each input.')
flags.DEFINE_string(
    'runner', 'direct',
    'Beam runner to use. Only direct is available outside of Google.')


def create_pipeline(
    out_dir: str,
    params: ml_collections.ConfigDict,
    checkpoint_path: str,
    dataset_path: str,
    max_passes: Optional[int],
    inference: bool,
    testing: bool = False,
):
  """Returns a pipeline for running model inference."""

  def pipeline(root):
    """Pipeline function for running model inference."""
    # For inference externally, params.max_passes does not get set in the
    # model_utils.modify_params function, we need to set it here before we call
    # model_utils.modify_params.
    if max_passes:
      with params.unlocked():
        params.max_passes = max_passes
    model_utils.modify_params(params=params, dataset_path=dataset_path)
    records = (
        root
        | 'read_tf_examples' >> beam.io.ReadFromTFRecord(
            os.path.join(dataset_path, '*.tfrecords.gz'),
            compression_type=CompressionTypes.GZIP)
        | 'shuffle_tf_examples' >> beam.Reshuffle()
        | 'parse_tf_examples' >> beam.ParDo(
            model_inference_transforms.ParseTfExamplesDoFn(
                params=params, inference=inference))
        | 'run_forward_pass' >> beam.ParDo(
            model_inference_transforms.RunForwardPassDoFn(
                checkpoint_path, params, inference=inference)))
    _ = (
        records
        | 'get_proto' >> beam.Map(lambda x: x['dc_proto'])
        | 'write_preds' >> tfrecordio.WriteToTFRecord(
            os.path.join(out_dir, 'predictions/deepconsensus'),
            file_name_suffix='.tfrecords.gz',
            coder=beam.coders.ProtoCoder(deepconsensus_pb2.DeepConsensusInput),
            compression_type=CompressionTypes.GZIP))

    if not inference:
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
      create_pipeline(
          out_dir=FLAGS.out_dir,
          params=params,
          checkpoint_path=FLAGS.checkpoint_path,
          dataset_path=FLAGS.dataset_path,
          max_passes=FLAGS.max_passes,
          inference=FLAGS.inference), options)
  if not FLAGS.inference:
    combine_metrics(FLAGS.out_dir)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'out_dir',
      'checkpoint_path',
  ])
  app.run(main)
