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

from deepconsensus.utils import utils


class QualityScoreToStringTest(parameterized.TestCase):

  @parameterized.parameters((0, '!'), (40, 'I'), (20, '5'))
  def test_score_to_string(self, score, expected_char):
    self.assertEqual(utils.quality_score_to_string(score), expected_char)

  @parameterized.parameters(([], ''), ([0, 10, 20, 30, 40], '!+5?I'))
  def test_score_list_to_string(self, scores, expected_str):
    self.assertEqual(utils.quality_scores_to_string(scores), expected_str)


class QualityStringToArrayTest(parameterized.TestCase):

  @parameterized.parameters(('', []), ('!', [0]), ('I', [40]), ('5', [20]),
                            ('!+5?I', [0, 10, 20, 30, 40]))
  def test_string_to_int(self, string, expected_scores):
    self.assertEqual(utils.quality_string_to_array(string), expected_scores)


if __name__ == '__main__':
  absltest.main()
