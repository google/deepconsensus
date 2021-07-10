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


if __name__ == '__main__':
  absltest.main()
