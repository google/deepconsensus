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
