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
"""Tests for deepconsensus.models.model_inference_with_beam."""

import glob
import os

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import tensorflow as tf

from deepconsensus.models import model_configs
from deepconsensus.models import model_inference_with_beam
from deepconsensus.models import model_utils
from deepconsensus.utils import test_utils


class ModelInferenceWithBeamTest(parameterized.TestCase):

  @parameterized.parameters([True, False])
  def test_end_to_end_inference(self, inference):
    """Test that the full pipeline runs without errors."""

    checkpoint_path = test_utils.deepconsensus_testdata('model/checkpoint-1')
    config_name = 'transformer_learn_values+test'
    params = model_configs.get_config(config_name)
    model_utils.modify_params(params)
    out_dir = self.create_tempdir().full_path
    # Run the pipeline.
    runner = beam.runners.DirectRunner()
    dataset_path = params.inference_path if inference else params.test_path
    pipeline = model_inference_with_beam.create_pipeline(
        out_dir=out_dir,
        params=params,
        checkpoint_path=checkpoint_path,
        dataset_path=dataset_path,
        testing=True,
        inference=inference,
        max_passes=None)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    runner.run(pipeline, options)

    output_files = glob.glob(
        os.path.join(out_dir, 'predictions/deepconsensus*'))
    self.assertNotEmpty(output_files)
    if not inference:
      model_inference_with_beam.combine_metrics(out_dir)
      metrics_combined = f'{out_dir}/metrics.stat.csv'
      self.assertGreater(tf.io.gfile.stat(metrics_combined).length, 100)

  @parameterized.parameters([True, False])
  def test_alternate_test_path(self, inference):
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
        dataset_path=self.create_tempdir().full_path,
        testing=True,
        inference=inference,
        max_passes=None)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    # test_path is an empty dir, so pipeline should throw an error.
    with self.assertRaises(OSError):
      runner.run(pipeline, options)


if __name__ == '__main__':
  absltest.main()
