"""Tests for deepconsensus .postprocess.stitch_predictions."""

import os

from absl.testing import absltest
import apache_beam as beam
import tensorflow as tf

from deepconsensus.postprocess import stitch_predictions
from deepconsensus.utils import dc_constants
from deepconsensus.utils.test_utils import deepconsensus_testdata
from google3.pipeline.flume.py import runner as flume_runner

from google3.third_party.nucleus.io import fastq


class StitchPredictionsTest(absltest.TestCase):

  def test_e2e(self):
    """Tests the full pipeline for joining all predictions for a molecule."""
    input_file = deepconsensus_testdata(
        'ecoli/output/predictions/deepconsensus*.tfrecords.gz')
    output_path = self.create_tempdir().full_path
    runner = flume_runner.FlumeRunner()
    # No padding here, just using full length of the subreads in the dc inputs.
    example_width = 100
    pipeline = stitch_predictions.create_pipeline(
        input_file=input_file,
        output_path=output_path,
        example_width=example_width)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    runner.run(pipeline, options)
    output_file_pattern = os.path.join(output_path, 'full_predictions*.fastq')
    total_contigs = 0
    output_files = tf.io.gfile.glob(output_file_pattern)
    for output_file in output_files:
      with fastq.FastqReader(output_file) as fastq_reader:
        for record in fastq_reader:
          total_contigs += 1
          self.assertTrue(record.id.endswith('/ccs'))
          self.assertTrue(set(record.sequence).issubset(dc_constants.VOCAB))
    self.assertGreater(total_contigs, 0)


if __name__ == '__main__':
  absltest.main()
