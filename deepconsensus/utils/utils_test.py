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
"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from deepconsensus.utils import utils


class EncodedSequenceToString(parameterized.TestCase):

  @parameterized.parameters(([0], ' '), ([1], 'A'), ([2], 'T'), ([3], 'C'),
                            ([4], 'G'), ([2, 0, 1], 'T A'))
  def test_score_to_string(self, encoded_val, expected_char):
    encoded_val = np.asarray(encoded_val)
    self.assertEqual(
        utils.encoded_sequence_to_string(encoded_val), expected_char)


class QualityScoreToStringTest(parameterized.TestCase):

  @parameterized.parameters((0, '!'), (40, 'I'), (20, '5'))
  def test_score_to_string(self, score, expected_char):
    self.assertEqual(utils.quality_score_to_string(score), expected_char)

  @parameterized.parameters((np.array([]), ''),
                            (np.array([0, 10, 20, 30, 40]), '!+5?I'))
  def test_score_list_to_string(self, scores, expected_str):
    self.assertEqual(utils.quality_scores_to_string(scores), expected_str)


class QualityStringToArrayTest(parameterized.TestCase):

  @parameterized.parameters(('', []), ('!', [0]), ('I', [40]), ('5', [20]),
                            ('!+5?I', [0, 10, 20, 30, 40]))
  def test_string_to_int(self, string, expected_scores):
    self.assertEqual(utils.quality_string_to_array(string), expected_scores)


class TestAvgPhred(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='single_value',
          ccs_base_quality_scores=np.array([1]),
          expected_avg_quality=0.9999,
      ),
      dict(
          testcase_name='integer list',
          ccs_base_quality_scores=[1, 2, 3],
          expected_avg_quality=1.9235,
      ),
      dict(
          testcase_name='multiple_values',
          ccs_base_quality_scores=np.array([1, 2, 3]),
          expected_avg_quality=1.9235,
      ),
      dict(
          testcase_name='spacer values',
          ccs_base_quality_scores=np.array([1, -1, 3]),
          expected_avg_quality=1.8858,
      ),
      dict(
          testcase_name='no values',
          ccs_base_quality_scores=np.array([-1, -1, -1]),
          expected_avg_quality=0.0,
      ))
  def test_avg_ccs_quality(self, ccs_base_quality_scores, expected_avg_quality):
    np_phred = utils.avg_phred(ccs_base_quality_scores)
    self.assertAlmostEqual(np_phred, expected_avg_quality, 3)
    # Test tensorflow implementation
    tf_phred = utils.tf_avg_phred(tf.convert_to_tensor(ccs_base_quality_scores))
    self.assertAlmostEqual(np_phred, tf_phred, 3)


class TestLeftSeq(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_gap',
          input_seq=np.array([1, 2, 3, 4, 1, 2, 3, 4]),
          expected_seq=np.array([1, 2, 3, 4, 1, 2, 3, 4]),
      ),
      dict(
          testcase_name='single_gap',
          input_seq=np.array([1, 2, 3, 4, 0, 0, 1, 2, 3, 4]),
          expected_seq=np.array([1, 2, 3, 4, 1, 2, 3, 4, 0, 0]),
      ),
      dict(
          testcase_name='multiple_gaps',
          input_seq=np.array(
              [0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0]),
          expected_seq=np.array(
              [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0]),
      ))
  def test_avg_ccs_quality(self, input_seq, expected_seq):
    left_shifted_seq = utils.left_shift_seq(input_seq)
    self.assertTrue((left_shifted_seq == expected_seq).all())


class TestBatchLeftSeq(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='no_gap',
          input_seq=np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
          expected_seq=np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
      ),
      dict(
          testcase_name='staggered_gaps',
          input_seq=np.array([[0, 0, 1, 1, 0, 0, 2, 2],
                              [1, 1, 0, 0, 2, 2, 0, 0]]),
          expected_seq=np.array([[1, 1, 2, 2, 0, 0, 0, 0],
                                 [1, 1, 2, 2, 0, 0, 0, 0]]),
      ))
  def test_avg_ccs_quality(self, input_seq, expected_seq):
    left_shifted_seq = utils.left_shift(input_seq)
    self.assertTrue((left_shifted_seq == expected_seq).all())


if __name__ == '__main__':
  absltest.main()
