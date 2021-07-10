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
"""Tests for deepconsensus.models.model_inference_with_beam."""

import glob
import os

from absl.testing import absltest
import apache_beam as beam
import tensorflow as tf

from deepconsensus.models import model_configs
from deepconsensus.models import model_inference_with_beam
from deepconsensus.models import model_utils
from deepconsensus.utils import test_utils


class ModelInferenceWithBeamTest(absltest.TestCase):

  def test_end_to_end(self):
    """Test that the full pipeline runs without errors."""

    checkpoint_path = test_utils.deepconsensus_testdata('model/checkpoint-1')
    config_name = 'transformer_learn_values+test'
    params = model_configs.get_config(config_name)
    model_utils.modify_params(params)
    out_dir = self.create_tempdir().full_path
    # Run the pipeline.
    runner = beam.runners.DirectRunner()
    pipeline = model_inference_with_beam.create_pipeline(
        out_dir=out_dir,
        params=params,
        checkpoint_path=checkpoint_path,
        test_path=None,
        testing=True)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    runner.run(pipeline, options)

    output_files = glob.glob(
        os.path.join(out_dir, 'predictions/deepconsensus*'))
    self.assertNotEmpty(output_files)
    model_inference_with_beam.combine_metrics(out_dir)
    metrics_combined = f'{out_dir}/metrics.stat.csv'
    self.assertGreater(tf.io.gfile.stat(metrics_combined).length, 100)

  def test_alternate_test_path(self):
    """Test that the specified test_path is used and throws an error."""

    checkpoint_path = test_utils.deepconsensus_testdata('model/checkpoint-1')
    config_name = 'transformer_learn_values+test'
    params = model_configs.get_config(config_name)
    model_utils.modify_params(params)
    out_dir = self.create_tempdir().full_path
    # Run the pipeline.
    runner = beam.runners.DirectRunner()
    pipeline = model_inference_with_beam.create_pipeline(
        out_dir=out_dir,
        params=params,
        checkpoint_path=checkpoint_path,
        test_path=self.create_tempdir().full_path,
        testing=True)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    # test_path is an empty dir, so pipeline should throw an error.
    with self.assertRaises(OSError):
      runner.run(pipeline, options)


if __name__ == '__main__':
  absltest.main()
