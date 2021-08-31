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
"""Tests for run_deepconsensus."""

import io
import os
from unittest import mock

from absl.testing import absltest

from deepconsensus.opensource_only.scripts import run_deepconsensus

# This shows the output of one run as context for other tests to build on.
EXPECTED_OUTPUT = """# DRY-RUN: mkdir output_directory/1_merge_datasets
# DRY-RUN: mkdir output_directory/2_generate_input
# DRY-RUN: mkdir output_directory/3_write_tf_examples
# DRY-RUN: mkdir output_directory/4_model_inference_with_beam
# DRY-RUN: mkdir output_directory

***** DRY-RUN ONLY:*****
python3 -m deepconsensus.preprocess.merge_datasets   --input_bam=input_subreads_aligned   --input_unaligned_bam=input_subreads_unaligned   --output_path=output_directory/1_merge_datasets   --inference=true


***** DRY-RUN ONLY:*****
python3 -m deepconsensus.preprocess.generate_input   --merged_datasets_path=output_directory/1_merge_datasets   --output_path=output_directory/2_generate_input   --input_ccs_fasta=input_ccs_fasta   --inference=true


***** DRY-RUN ONLY:*****
python3 -m deepconsensus.tf_examples.write_tf_examples   --preprocess_paths=output_directory/2_generate_input   --output_path=output_directory/3_write_tf_examples   --preprocess_downsample=1   --species=human   --inference=true   --example_width=200   --max_passes=20   --padded_len=120   --subread_permutations=0


***** DRY-RUN ONLY:*****
python3 -m deepconsensus.models.model_inference_with_beam   --dataset_path=output_directory/3_write_tf_examples/inference   --out_dir=output_directory/4_model_inference_with_beam   --checkpoint_path=checkpoint   --inference=true   --max_passes=20


***** DRY-RUN ONLY:*****
python3 -m deepconsensus.postprocess.stitch_predictions   --input_file=output_directory/4_model_inference_with_beam/predictions/*.tfrecords.gz   --output_path=output_directory   --example_width=200   --inference=true   --min_quality=10

Outputs can be found at: output_directory
"""


class RunDeepconsensusTest(absltest.TestCase):

  def test_full_dry_run_matched_expected(self):
    mock_stdout = io.StringIO()
    with mock.patch('sys.stdout', mock_stdout):
      run_deepconsensus.run_deepconsensus(
          dry_run=True,
          input_subreads_aligned='input_subreads_aligned',
          input_subreads_unaligned='input_subreads_unaligned',
          input_ccs_fasta='input_ccs_fasta',
          output_directory='output_directory',
          checkpoint='checkpoint',
          min_quality=10,
          example_width=200)
      self.assertEqual(mock_stdout.getvalue(), EXPECTED_OUTPUT)

  def test_params_are_printed(self):
    mock_stdout = io.StringIO()
    with mock.patch('sys.stdout', mock_stdout):
      run_deepconsensus.run_deepconsensus(
          dry_run=True,
          input_subreads_aligned='input_subreads_aligned',
          input_subreads_unaligned='input_subreads_unaligned',
          input_ccs_fasta='input_ccs_fasta',
          output_directory='output_directory',
          checkpoint='checkpoint',
          min_quality=20,
          example_width=100)
      log_output = mock_stdout.getvalue()
      self.assertIn('example_width=100', log_output)
      self.assertIn('min_quality=20', log_output)

  def test_incomplete_params_raises(self):
    # Capture printed output for cleaner test logs.
    mock_stdout = io.StringIO()
    with mock.patch('sys.stdout', mock_stdout):
      with self.assertRaisesRegex(TypeError, 'input_ccs_fasta'):
        run_deepconsensus.run_deepconsensus(
            dry_run=True,
            input_subreads_aligned='input_subreads_aligned',
            input_subreads_unaligned='input_subreads_unaligned',
            # input_ccs_fasta='input_ccs_fasta',
            output_directory='output_directory',
            checkpoint='checkpoint',
            min_quality=10,
            example_width=200)

      with self.assertRaisesRegex(TypeError, 'input_subreads_aligned'):
        run_deepconsensus.run_deepconsensus(
            dry_run=True,
            # input_subreads_aligned='input_subreads_aligned',
            input_subreads_unaligned='input_subreads_unaligned',
            input_ccs_fasta='input_ccs_fasta',
            output_directory='output_directory',
            checkpoint='checkpoint',
            min_quality=10,
            example_width=200)

  def test_non_dry_run_fails_correctly(self):
    # The dry run fails when running as a bazel test because it tries to run
    # 'python3 -m deepconsensus.preprocess.merge_datasets' which doesn't exist
    # unless the pip package is installed.

    # This therefore can't test that it finishes, so instead it checks that
    # the correct directories are created and that the failure happens as
    # expected and is logged and shown to the user correctly.

    real_output_directory = self.create_tempdir()
    output_directory = os.path.join(real_output_directory.full_path,
                                    'output_for_run_deepconsensus_test')
    # Capture printed output for cleaner test logs.
    mock_stdout = io.StringIO()
    with mock.patch('sys.stdout', mock_stdout):
      with self.assertRaisesRegex(RuntimeError, 'Command failed'):
        run_deepconsensus.run_deepconsensus(
            dry_run=False,
            input_subreads_aligned='input_subreads_aligned',
            input_subreads_unaligned='input_subreads_unaligned',
            input_ccs_fasta='input_ccs_fasta',
            output_directory=output_directory,
            checkpoint='checkpoint',
            min_quality=20,
            example_width=100)

    # Check the right directories are all created.
    self.assertTrue(os.path.isdir(output_directory))
    directories_created = os.listdir(output_directory)
    self.assertContainsSubset([
        '1_merge_datasets', '2_generate_input', '3_write_tf_examples',
        '4_model_inference_with_beam'
    ], directories_created)

    # Check logs that were printed to the screen.
    self.assertIn(
        'Running the command',
        mock_stdout.getvalue(),
        msg='Printed log shows attempt at running the first command.')

    # Check logs that were saved to the log file.
    self.assertTrue(
        os.path.exists(output_directory + '/deepconsensus_log.txt'),
        msg='Creates log file.')
    with open(os.path.join(output_directory, 'deepconsensus_log.txt')) as f:
      self.assertIn(
          'Running the command',
          '\n'.join(f.readlines()),
          msg='Log file shows attempt at running the first command.')


if __name__ == '__main__':
  absltest.main()
