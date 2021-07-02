# Disable pyformat so that the command for running the binary does not get
# incorrectly split into multiple lines.
# pyformat: disable
r"""Inference binary for all neural network models.

To use this binary for running inference with a specific model, the
model checkpoint should be specified as input. We build and run separately here
so that the binary can be built with `--config=cuda`. We also use
`--flume_dax_parallel_config_updates` to ensure each worker has a GPU, which
provides significant speedup. See go/flume-with-accelerators for the
documentation.

blaze build -c opt \
  //learning/genomics/deepconsensus/models:model_inference_with_beam.par \
  --config=cuda

DATE=$(TZ=US/Pacific date "+%Y%m%d")
OUT_DIR="/cns/is-d/home/brain-genomics/${USER}/deepconsensus/output_predictions/${DATE}"
CHECKPOINT_PATH=/cns/is-d/home/brain-genomics/gunjanbaid/deepconsensus/experiments/20201106/exp_18479117/wu_3/model

time ./blaze-bin/learning/genomics/deepconsensus/models/model_inference_with_beam.par \
  --out_dir ${OUT_DIR} \
  --checkpoint_path ${CHECKPOINT_PATH} \
  --flume_exec_mode=BORG \
  --flume_borg_user_name=${USER} \
  --flume_borg_accounting_charged_user_name=brain-genomics \
  --flume_batch_scheduler_strategy=RUN_SOON \
  --flume_use_batch_scheduler \
  --flume_worker_priority=100 \
  --flume_close_to_resources="/cns/is-d/home/brain-genomics" \
  --flume_backend=DAX \
  --flume_auto_retry=false \
  --flume_tmp_file_cells="is-d" \
  --flume_tmp_dir_group="brain-genomics" \
  --logtostderr \
  --flume_completion_email_address=${USER}@google.com \
  --flume_dax_parallel_config_updates="worker_pool_config { \
      [dist_proc.dax.workflow.borg_process_pool_config_ext] { \
        use_ml_allocator: true, accelerator: GPU_TESLA_V100 }}"
"""
# pyformat: enable

import os

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io import tfrecordio
from apache_beam.io.filesystem import CompressionTypes

import ml_collections
import pandas as pd
import tensorflow as tf

from deepconsensus.models import model_inference_transforms
from deepconsensus.models import model_utils
from deepconsensus.protos import deepconsensus_pb2
from google3.pipeline.flume.py import runner as flume_runner

FLAGS = flags.FLAGS
ml_collections.config_flags.DEFINE_config_file('params', None,
                                               'Training configuration.')
flags.DEFINE_string('checkpoint_path', None,
                    'Path to checkpoint that will be loaded in.')
flags.DEFINE_string(
    'test_path', None,
    'Optional. Alternate dataset on which to run inference. If '
    'not provided, the dataset will be inferred from params.')
flags.DEFINE_string('out_dir', None,
                    'Output path for logs and optionally, model predictions.')


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

      # Output error analysis for each configuration.
      if group != 'all':
        _ = (
            records
            | f'ea_{group}' >> model_inference_transforms.ErrorAnalysis(
                out_dir, group, params))

      for metric in metric_set:
        _ = records | f'{group}_{metric}' >> model_inference_transforms.StatsToCsv(
            out_dir, group, metric)

    # Per position accuracy
    _ = (records | model_inference_transforms.StratifiedPosAccToCSV(out_dir))

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
  if not FLAGS.params:
    params = model_utils.read_params_from_json(
        checkpoint_path=FLAGS.checkpoint_path)
  else:
    params = FLAGS.params
  runner = flume_runner.FlumeRunner()
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
