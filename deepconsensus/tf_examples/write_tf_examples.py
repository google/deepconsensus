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
# pyformat: disable
r"""A pipeline for writing out tf.Example protos.

Example usage:

python3 -m deepconsensus.tf_examples.write_tf_examples \
  --preprocess_paths=deepconsensus/testdata/ecoli/output \
  --preprocess_downsample=0.5 \
  --output_path=/tmp/write_tf_examples \
  --species=ecoli

Outputs are in `/tmp/write_tf_examples`. Counters can be found in:

$ cat /tmp/write_tf_examples/counts.json
{
    "filter_deepconsensus/testdata/ecoli/output:after_subsample_count_ecoli": 1,
    "filter_deepconsensus/testdata/ecoli/output:before_subsample_count_ecoli": 1,
    "process_and_write_train/chunk_windows_100_train:small_windows": 253,
    "process_and_write_train/chunk_windows_100_train:unsup_insertions_total": 1,
    "process_and_write_train/convert_to_tf_ex_train:subreads_counter": 1764,
    "process_and_write_train/convert_to_tf_ex_train:subreads_reverse_strand": 756,
    "process_and_write_train/convert_to_tf_ex_train:total_examples": 252,
    "process_and_write_train/pad_examples_train:windows_longer_than_padded_len": 1
}

"""
# pyformat: enable

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
flags.DEFINE_string(
    'runner', 'direct',
    'Beam runner to use. Only direct is available outside of Google.')


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
  if FLAGS.runner == 'direct':
    runner = beam.runners.DirectRunner()
  else:
    raise ValueError(f'Invalid runner type: {FLAGS.runner}')

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
