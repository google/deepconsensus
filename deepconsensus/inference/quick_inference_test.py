# Copyright (c) 2021, Google Inc.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 
# 3. Neither the name of Google Inc. nor the names of its contributors
#    may be used to endorse or promote products derived from this software without
#    specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Tests for quick_inference."""

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import pysam

from deepconsensus.inference import quick_inference
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils

FLAGS = flags.FLAGS


def setUpModule():
  logging.set_verbosity(logging.FATAL)


class QuickInferenceTest(parameterized.TestCase):

  @parameterized.parameters(
      dict(
          subreads='human_1m/subreads_to_ccs.bam',
          fasta='human_1m/ccs.fasta',
          expected_lengths=[17141, 16320]))
  @flagsaver.flagsaver
  def test_end_to_end(self, subreads, fasta, expected_lengths):
    FLAGS.subreads_to_ccs = test_utils.deepconsensus_testdata(subreads)
    FLAGS.ccs_fasta = test_utils.deepconsensus_testdata(fasta)
    output_path = test_utils.test_tmpfile('output_path.fastq')
    FLAGS.output = output_path
    FLAGS.checkpoint = test_utils.deepconsensus_testdata('model/checkpoint-1')
    FLAGS.min_quality = 0  # Qualities are lower due to tiny sample model.
    FLAGS.limit = 2
    outcomes = quick_inference.run()

    count = 0
    output_lengths = []
    with pysam.FastqFile(output_path) as fastq_reader:
      for record in fastq_reader:
        self.assertTrue(record.name.endswith('/ccs'))
        self.assertTrue(set(record.sequence).issubset(dc_constants.VOCAB))
        self.assertEqual(len(record.sequence), len(record.quality))
        # Length of the output read should be deterministic for the same model.
        output_lengths.append(len(record.sequence))
        count += 1
    self.assertEqual(count, 2)
    # TODO: Figure out why lengths are not deterministic.
    # Not deterministic, might be due to the test model used since other runs
    # with the release model have been deterministic so far.
    # self.assertEqual(expected_lengths, output_lengths)
    print('expected lengths:', expected_lengths, 'output lengths:',
          output_lengths)
    self.assertEqual(outcomes.success, 2)

  @parameterized.parameters(
      dict(cpus=0, batch_zmws=1), dict(cpus=0, batch_zmws=0),
      dict(cpus=1, batch_zmws=1), dict(cpus=1, batch_zmws=100))
  @flagsaver.flagsaver
  def test_end_to_end_multiprocessing(self, cpus, batch_zmws):
    FLAGS.subreads_to_ccs = test_utils.deepconsensus_testdata(
        'human_1m/subreads_to_ccs.bam')
    FLAGS.ccs_fasta = test_utils.deepconsensus_testdata('human_1m/ccs.fasta')
    FLAGS.checkpoint = test_utils.deepconsensus_testdata('model/checkpoint-1')
    output_path = test_utils.test_tmpfile('output_path.fastq')
    FLAGS.output = output_path
    FLAGS.batch_zmws = batch_zmws
    FLAGS.cpus = cpus
    FLAGS.min_quality = 0  # Qualities are lower due to tiny sample model.
    FLAGS.limit = 2
    outcomes = quick_inference.run()

    count = 0
    with pysam.FastqFile(output_path) as fastq_reader:
      for record in fastq_reader:
        self.assertTrue(record.name.endswith('/ccs'))
        self.assertTrue(set(record.sequence).issubset(dc_constants.VOCAB))
        self.assertEqual(len(record.sequence), len(record.quality))
        count += 1
    self.assertEqual(count, 2)
    self.assertEqual(outcomes.success, 2)


if __name__ == '__main__':
  absltest.main()
