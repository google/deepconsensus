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
"""Tests for deepconsensus.preprocess.write_tf_examples."""

import itertools
import os
import shutil

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import numpy as np
import tensorflow as tf

from deepconsensus.protos import deepconsensus_pb2
from deepconsensus.tf_examples import tf_example_utils
from deepconsensus.tf_examples import write_tf_examples
from deepconsensus.utils import dc_constants
from deepconsensus.utils.test_utils import deepconsensus_testdata
from nucleus.io import sharded_file_utils
from nucleus.io import tfrecord


class WriteTfExamplesTest(parameterized.TestCase):

  def setUp(self):
    self.temp_dir = self.create_tempdir().full_path
    super().setUp()

  def check_for_dataset(self, set_name: str, example_height: int,
                        expected_width: int, inference: bool):
    # Check eval examples.
    files = sharded_file_utils.glob_list_sharded_file_patterns(
        os.path.join(self.temp_dir, f'{set_name}/{set_name}*.tfrecords.gz'))
    file_pattern = os.path.join(
        self.temp_dir, f'{set_name}/{set_name}@%d.tfrecords.gz' % len(files))
    reader = tfrecord.read_tfrecords(file_pattern, proto=tf.train.Example)
    for example in reader:
      self.check_example(example, example_height, expected_width, inference)

  def check_example(self, example, example_height, example_width, inference):
    """Checks that example is of the expected format."""

    encoded_subreads = tf_example_utils.get_encoded_subreads_from_example(
        example)
    subreads_shape = tf_example_utils.get_subreads_shape_from_example(example)
    num_passes = tf_example_utils.get_num_passes_from_example(example)
    encoded_deepconsensus_input = tf_example_utils.get_encoded_deepconsensus_input_from_example(
        example)
    deepconsensus_input = deepconsensus_pb2.DeepConsensusInput.FromString(
        encoded_deepconsensus_input)

    # Sanity check the DeepConsensusInput proto and num_passes.
    self.assertGreater(num_passes, 0)
    self.assertLessEqual(num_passes, len(deepconsensus_input.subreads))
    self.assertNotEmpty(deepconsensus_input.subreads)

    # Check that saved shapes are correct and that arrays have the correct
    # number of elements.
    self.assertEqual(subreads_shape, [example_height, example_width, 1])
    self.assertEqual(
        np.fromstring(encoded_subreads, dc_constants.NP_DATA_TYPE).size,
        np.prod(subreads_shape))
    if not inference:
      encoded_label = tf_example_utils.get_encoded_label_from_example(example)
      label_shape = tf_example_utils.get_label_shape_from_example(example)
      self.assertEqual(label_shape, [example_width])
      self.assertEqual(
          np.fromstring(encoded_label, dc_constants.NP_DATA_TYPE).size,
          np.prod(label_shape))

  @parameterized.parameters(
      itertools.product(['ecoli', 'human'], [35], [True, False]))
  def test_end_to_end_train(self, species, padded_len, inference):
    """Tests the full pipeline, including IO."""
    output_path = 'inference_output' if inference else 'output'
    dc_test_data = deepconsensus_testdata(f'{species}/{output_path}')
    # Test writing out two dc input paths
    # Copy the dc_test_data to a tempdir so we have 'two' TFrecord inputs
    cp_test_data = self.create_tempdir().full_path + '/output'
    shutil.copytree(dc_test_data, cp_test_data)
    preprocess_paths = [dc_test_data, cp_test_data]
    example_width = 25
    max_passes = 2
    example_height = tf_example_utils.get_total_rows(max_passes=max_passes)
    if inference:
      preprocess_downsample = ['1.0', '1.0']
      reference_fasta = None
      truth_vcf = None
      truth_bed = None
      window_overlap_step = None
      subread_permutations = 0
    else:
      preprocess_downsample = ['0.5', '0.5']
      window_overlap_step = example_width // 4
      subread_permutations = 1
      if species == 'human':
        reference_fasta = deepconsensus_testdata(
            f'{species}/{species}.ref.fa.gz')
        truth_vcf = deepconsensus_testdata(
            f'{species}/{species}.variants.vcf.gz')
        truth_bed = deepconsensus_testdata(f'{species}/{species}.truth.bed')
      elif species == 'ecoli':
        reference_fasta = None
        truth_vcf = None
        truth_bed = None

    # Run the pipeline.
    runner = beam.runners.DirectRunner()
    pipeline = write_tf_examples.create_pipeline(
        preprocess_paths,
        preprocess_downsample,
        output_path=self.temp_dir,
        max_passes=max_passes,
        example_width=example_width,
        species=species,
        reference_fasta=reference_fasta,
        truth_vcf=truth_vcf,
        truth_bed=truth_bed,
        padded_len=padded_len,
        window_overlap_step=window_overlap_step,
        subread_permutations=subread_permutations,
        inference=inference)
    options = beam.options.pipeline_options.PipelineOptions(
        pipeline_type_check=True, runtime_type_check=True)
    result = runner.run(pipeline, options)
    tf_example_utils.metrics_to_json(result,
                                     self.temp_dir + '/tf_example.counts.json')

    # There is only one DeepConsensusInput proto (chrom_end=2347972) in the
    # testdata. This should be allocated to the training set and the eval set
    # should be empty.
    # <internal>
    # protos.

    # Check that each dataset was output.
    expected_width = padded_len or example_width
    if inference:
      self.check_for_dataset(
          'inference', example_height, expected_width, inference=inference)
    else:
      self.check_for_dataset(
          'train', example_height, expected_width, inference=inference)
      self.check_for_dataset(
          'eval', example_height, expected_width, inference=inference)
      self.check_for_dataset(
          'test', example_height, expected_width, inference=inference)


if __name__ == '__main__':
  absltest.main()
