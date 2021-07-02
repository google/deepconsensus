r"""Script to calculate metrics for pbccs and DeepConsensus on overlapping data.

Example usage:

PBCCS_BAM_CONCORDANCE_CSV=/bigstore/brain-genomics/deepconsensus/postprocess/m54238_180901_011437_pbccs_20210302181746/bam_concordance.csv
PBCCS_IDENTITY_METRICS_CSV=/bigstore/brain-genomics/deepconsensus/postprocess/m54238_180901_011437_pbccs_20210302181746/per_read_identity_metrics.csv
DC_BAM_CONCORDANCE_CSV=/bigstore/brain-genomics/deepconsensus/postprocess/m54238_180901_011437_20210302232749/bam_concordance.csv
DC_IDENTITY_METRICS_CSV=/bigstore/brain-genomics/deepconsensus/postprocess/m54238_180901_011437_20210302232749/per_read_identity_metrics.csv
OUTPUT_FILE=/tmp/compare_deepconsensus_to_pbccs.csv

blaze run -c opt \
  learning/genomics/deepconsensus/utils:compare_metrics_to_pbccs \
  -- \
  --pbccs_bam_concordance_csv=${PBCCS_BAM_CONCORDANCE_CSV} \
  --pbccs_identity_metrics_csv=${PBCCS_IDENTITY_METRICS_CSV} \
  --dc_bam_concordance_csv=${DC_BAM_CONCORDANCE_CSV} \
  --dc_identity_metrics_csv=${DC_IDENTITY_METRICS_CSV} \
  --output_file=${OUTPUT_FILE}
"""

import itertools
from typing import Sequence

from absl import app
from absl import flags
import pandas as pd
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('pbccs_bam_concordance_csv', None,
                    'PBCCS bamConcordance CSV output file.')
flags.DEFINE_string('pbccs_identity_metrics_csv', None,
                    'PBCCS identity metrics CSV output file.')
flags.DEFINE_string('dc_bam_concordance_csv', None,
                    'DeepConsensus bamConcordance metrics CSV output file.')
flags.DEFINE_string('dc_identity_metrics_csv', None,
                    'PBCCS identity metrics CSV output file.')
flags.DEFINE_string('output_file', None, 'Output CSV with comparable metrics.')


def get_comparable_metrics(pbccs_bam_concordance_csv: str,
                           pbccs_identity_metrics_csv: str,
                           dc_bam_concordance_csv: str,
                           dc_identity_metrics_csv: str,
                           output_file: str) -> None:
  """Writes out metrics for pbccs and DeepConsensus on overlapping molecules."""
  pbccs_bam_concordance_df = pd.read_csv(
      tf.io.gfile.GFile(pbccs_bam_concordance_csv))
  pbccs_identity_metrics_df = pd.read_csv(
      tf.io.gfile.GFile(pbccs_identity_metrics_csv))
  dc_bam_concordance_df = pd.read_csv(tf.io.gfile.GFile(dc_bam_concordance_csv))
  dc_identity_metrics_df = pd.read_csv(
      tf.io.gfile.GFile(dc_identity_metrics_csv))

  # Merge the identity and bamConcordance output files.
  pbccs_df = pbccs_bam_concordance_df.merge(
      pbccs_identity_metrics_df,
      left_on='#read',
      right_on='read_name',
      how='inner')
  dc_df = dc_bam_concordance_df.merge(
      dc_identity_metrics_df,
      left_on='#read',
      right_on='read_name',
      how='inner')

  # Get intersection of molecules from both DataFrames and primary alignments.
  pbccs_df = pbccs_df[pbccs_df['read_name'].isin(dc_df['read_name'])]
  dc_df = dc_df[dc_df['read_name'].isin(pbccs_df['read_name'])]
  pbccs_df = pbccs_df[pbccs_df['alignmentType'] == 'Primary']
  dc_df = dc_df[dc_df['alignmentType'] == 'Primary']
  assert len(pbccs_df) == len(dc_df)

  # Rename columns to include method name.
  pbccs_df = pbccs_df.rename(mapper=lambda name: f'pbccs_{name}', axis=1)
  dc_df = dc_df.rename(mapper=lambda name: f'dc_{name}', axis=1)
  one_df = pbccs_df.merge(
      dc_df, left_on='pbccs_read_name', right_on='dc_read_name')

  # Compute average and weighted average metrics.
  results = pd.DataFrame()
  metrics = [
      'concordance', 'identity', 'gap_compressed_identity', 'concordance'
  ]
  for method, metric in itertools.product(['pbccs', 'dc'], metrics):
    col_name = f'{method}_{metric}'
    results[f'ave_{col_name}'] = [one_df[col_name].mean()]
    total = sum(one_df[col_name] * one_df[f'{method}_readLengthBp'])
    weighed_ave = total / sum(one_df[f'{method}_readLengthBp'])
    results[f'ave_{col_name}_weighted'] = [weighed_ave]
  results.sort_index(axis=1).to_csv(
      tf.io.gfile.GFile(output_file, 'wt'), index=False)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  get_comparable_metrics(FLAGS.pbccs_bam_concordance_csv,
                         FLAGS.pbccs_identity_metrics_csv,
                         FLAGS.dc_bam_concordance_csv,
                         FLAGS.dc_identity_metrics_csv, FLAGS.output_file)


if __name__ == '__main__':
  app.run(main)
