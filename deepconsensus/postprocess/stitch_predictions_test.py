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
"""Tests for deepconsensus.postprocess.stitch_predictions."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import apache_beam as beam
import tensorflow as tf

from deepconsensus.postprocess import stitch_predictions
from deepconsensus.utils import dc_constants
from deepconsensus.utils.test_utils import deepconsensus_testdata

from nucleus.io import fastq


class StitchPredictionsTest(parameterized.TestCase):

  @parameterized.parameters(
      (False, None, None),
      (True, 1, 1),
  )
  def test_e2e(self, inference, min_quality, min_length):
    """Tests the full pipeline for joining all predictions for a molecule."""
    input_file = deepconsensus_testdata(
        'ecoli/output/predictions/deepconsensus*.tfrecords.gz')
    output_path = self.create_tempdir().full_path
    runner = beam.runners.DirectRunner()
    # No padding here, just using full length of the subreads in the dc inputs.
    example_width = 100
    pipeline = stitch_predictions.create_pipeline(
        input_file=input_file,
        output_path=output_path,
        min_quality=min_quality,
        min_length=min_length,
        inference=inference,
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
