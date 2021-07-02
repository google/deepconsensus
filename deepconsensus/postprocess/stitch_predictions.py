r"""A pipeline for writing out FASTA files from predictions.

Example usage:

EXP_DIR=/cns/is-d/home/brain-genomics/gunjanbaid/deepconsensus/output_predictions/20201111_cuda_poa
INPUT_FILE=${EXP_DIR}/deepconsensus/deepconsensus@102.tfrecords.gz
OUTPUT_PATH=${EXP_DIR}/stitched_predictions

time blaze run -c opt \
//learning/genomics/deepconsensus/postprocess:stitch_predictions.par -- \
  --input_file ${INPUT_FILE} \
  --output_path ${OUTPUT_PATH} \
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
  --flume_completion_email_address=${USER}@google.com
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
from google3.pipeline.flume.py import runner as flume_runner

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file', None, 'Input TFRecords files.')
flags.DEFINE_boolean(
    'fill_n', False, 'Output failed sequence windows using N'
    'placeholder sequences')
flags.DEFINE_string('output_path', None,
                    'Path at which output tfrecords.gz files will be created.')
flags.DEFINE_integer('example_width', None, 'Width of examples used.')


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
  flume_runner.program_started()

  # We have to do flag validation in main rather than using
  # flags.mark_flags_as_required because beam workers don't set flags
  # appropriately.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not FLAGS.input_file:
    raise app.UsageError('--input_file must be specified.')
  if not FLAGS.output_path:
    raise app.UsageError('--output_path must be specified.')

  runner = flume_runner.FlumeRunner()
  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  runner.run(
      create_pipeline(FLAGS.input_file, FLAGS.output_path, FLAGS.example_width),
      options)


if __name__ == '__main__':
  app.run(main)
