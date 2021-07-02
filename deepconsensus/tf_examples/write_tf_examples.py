r"""A pipeline for writing out tf.Example protos.

Example usage:

DATE=$(TZ=US/Pacific date "+%Y%m%d")
CL=$(g4 client -o | grep SyncChange | cut -f 2)
PREPROCESS_PATHS=/cns/is-d/home/brain-genomics/gunjanbaid/deepconsensus/deepconsensus_input/human_m54238_180901_011437_alignedToPoa/20201106
PREPROCESS_DOWNSAMPLE="0.5"
OUTPUT_PATH=/cns/is-d/home/brain-genomics/gunjanbaid/deepconsensus/tfexamples/human_m54238_180901_011437_alignedToPoa/${DATE}_hybrid_approach
TRUTH_VCF=/cns/is-d/home/brain-genomics/gunjanbaid/deepconsensus/HG002_v42.vcf.gz
EXAMPLE_WIDTH=100
MAX_PASSES=20
SPECIES=human
REFERENCE_FASTA=/cns/is-d/home/brain-genomics/${USER}/deepconsensus/GCA_000001405.15_GRCh38_no_alt_plus_hs38d1_analysis_set.fna

time blaze run -c opt \
//learning/genomics/deepconsensus/tf_examples:write_tf_examples.par -- \
  --preprocess_paths ${PREPROCESS_PATHS} \
  --preprocess_downsample ${PREPROCESS_DOWNSAMPLE} \
  --output_path ${OUTPUT_PATH} \
  --example_width ${EXAMPLE_WIDTH} \
  --max_passes ${MAX_PASSES} \
  --truth_vcf ${TRUTH_VCF} \
  --species ${SPECIES} \
  --reference_fasta ${REFERENCE_FASTA} \
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

import os
from typing import Optional

from absl import app
from absl import flags
import apache_beam as beam
from apache_beam.io.filesystem import CompressionTypes
import tensorflow as tf

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.tf_examples import tf_example_transforms
from deepconsensus.tf_examples import tf_example_utils
from google3.pipeline.flume.py import runner as flume_runner

FLAGS = flags.FLAGS

flags.DEFINE_list(
    'preprocess_paths', [], 'DeepConsensus Preprocess Input Paths where '
    'TFRecords are stored.')
flags.DEFINE_list(
    'preprocess_downsample', [],
    'Comma-delimited downsample fraction for each preprocess'
    'path.')
flags.DEFINE_string('output_path', None,
                    'Path at which output tfrecords.gz files will be created.')
flags.DEFINE_integer('max_passes', 20, 'Maximum subreads in each input.')
flags.DEFINE_integer('example_width', 100, 'Number of bases in each input.')
flags.DEFINE_string(
    'species', 'ecoli',
    'Species for the data being used. Can be either ecoli or '
    'human.')
flags.DEFINE_string(
    'reference_fasta', None, 'Path to human reference genome. Only used with '
    'human data to help us filter out windows with variants')
flags.DEFINE_string('truth_vcf', None, 'Path to truth variants VCF.')
flags.DEFINE_string('truth_bed', None, 'Path to confident regions BED file.')
flags.DEFINE_integer('padded_len', None,
                     'Total length after padding to allow for insertions.')
flags.DEFINE_integer(
    'window_overlap_step', None,
    'Range step used to overlap windows for training. If this '
    'is None, there will be no overlap.')
flags.DEFINE_integer(
    'subread_permutations', 0,
    'Number of additional subread permutations to generate per example.'
    '0=No additional permutations generated.')


def create_pipeline(preprocess_paths: str, preprocess_downsample: str,
                    output_path: str, max_passes: int, example_width: int,
                    species: str, reference_fasta: str, truth_vcf: str,
                    truth_bed: str, padded_len: int,
                    window_overlap_step: Optional[int],
                    subread_permutations: Optional[int]):
  """Returns a pipeline for creating pileup examples."""

  contig_chrom = {}
  if reference_fasta:
    reference_name = os.path.basename(reference_fasta).split('.', 1)[0]
    reference_path = os.path.dirname(reference_fasta)

    # For diploid assemblies, read in contig to chrom.
    # https://bit.googleplex.com/#/danielecook/6429885700505600
    chrom_map_file = os.path.join(reference_path,
                                  f'{reference_name}.chrom_mapping.txt')
    if tf.io.gfile.exists(chrom_map_file):
      with tf.io.gfile.GFile(chrom_map_file) as f:
        contig_chrom = dict([x.split() for x in f.readlines()])

  def pipeline(root):
    """Pipeline function for creating TF examples."""

    dc_downsample = list(map(float, preprocess_downsample))
    example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
    assert len(preprocess_paths) == len(dc_downsample)
    tf_inputs = []
    for i, dc_path in enumerate(preprocess_paths):
      # Extracts p<date>_<hash> from dc_path.
      dc_name = os.path.basename(os.path.dirname(dc_path.strip('/')))
      result = (
          root
          | f'read_{dc_path}' >> beam.io.ReadFromTFRecord(
              os.path.join(dc_path,
                           'deepconsensus/deepconsensus*.tfrecords.gz'),
              coder=beam.coders.ProtoCoder(
                  deepconsensus_pb2.DeepConsensusInput),
              compression_type=CompressionTypes.GZIP)
          | f'shuffle_{dc_path}' >> beam.Reshuffle()
          | f'filter_{dc_path}' >> beam.ParDo(
              tf_example_transforms.DownSample(dc_downsample[i], dc_name)))
      tf_inputs.append(result)

    merged_tf_input = (tuple(tf_inputs) | 'flatten_tfrecords' >> beam.Flatten())

    num_partitions = 4
    train_set, eval_set, test_set, _ = (
        merged_tf_input
        | beam.Partition(
            tf_example_utils.train_eval_partition_fn,
            num_partitions,
            species=species,
            contig_chrom=contig_chrom))

    _ = (
        train_set
        | 'process_and_write_train' >>
        tf_example_transforms.ProcessAndWriteTfExamples(
            reference_fasta=reference_fasta,
            example_width=example_width,
            example_height=example_height,
            truth_vcf=truth_vcf,
            species=species,
            split='train',
            output_path=output_path,
            truth_bed=truth_bed,
            padded_len=padded_len,
            window_overlap_step=window_overlap_step,
            subread_permutations=subread_permutations))

    _ = (
        eval_set
        | 'process_and_write_eval' >>
        tf_example_transforms.ProcessAndWriteTfExamples(
            reference_fasta=reference_fasta,
            example_width=example_width,
            example_height=example_height,
            truth_vcf=truth_vcf,
            species=species,
            split='eval',
            output_path=output_path,
            truth_bed=truth_bed,
            padded_len=padded_len,
            window_overlap_step=None,
            subread_permutations=0))

    _ = (
        test_set
        | 'process_and_write_test' >>
        tf_example_transforms.ProcessAndWriteTfExamples(
            reference_fasta=reference_fasta,
            example_width=example_width,
            example_height=example_height,
            truth_vcf=None,
            truth_bed=None,
            species=species,
            split='test',
            output_path=output_path,
            padded_len=padded_len,
            window_overlap_step=None,
            subread_permutations=0))

  return pipeline


def main(argv):
  """Main entry point."""
  flume_runner.program_started()

  # We have to do flag validation in main rather than using
  # flags.mark_flags_as_required because beam workers don't set flags
  # appropriately.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  if not FLAGS.preprocess_paths:
    raise app.UsageError('--preprocess_paths must be specified.')
  if not FLAGS.preprocess_downsample:
    raise app.UsageError('--preprocess_downsample must be specified.')
  if not FLAGS.output_path:
    raise app.UsageError('--output_path must be specified.')
  if FLAGS.padded_len and FLAGS.padded_len < FLAGS.example_width:
    raise app.UsageError('--padded_len cannot be less than --example_width.')

  runner = flume_runner.FlumeRunner()
  options = beam.options.pipeline_options.PipelineOptions(
      pipeline_type_check=True, runtime_type_check=True)
  pipeline = create_pipeline(
      FLAGS.preprocess_paths, FLAGS.preprocess_downsample, FLAGS.output_path,
      FLAGS.max_passes, FLAGS.example_width, FLAGS.species,
      FLAGS.reference_fasta, FLAGS.truth_vcf, FLAGS.truth_bed, FLAGS.padded_len,
      FLAGS.window_overlap_step, FLAGS.subread_permutations)
  result = runner.run(pipeline, options)
  # Write counts summary to file.
  counts_path = os.path.join(FLAGS.output_path, 'counts.json')
  tf_example_utils.metrics_to_json(result, counts_path)


if __name__ == '__main__':
  app.run(main)
