"""Utilities for DeepConsensus."""

from typing import List


def quality_score_to_string(score: int) -> str:
  """Returns the string representation for the given quality score.

  We add 33 to the score because this is how the quality score encoding is
  defined. Source:
  https://support.illumina.com/help/BaseSpace_OLH_009008/Content/Source/Informatics/BS/QualityScoreEncoding_swBS.htm

  Args:
    score: The raw quality score value.

  Returns:
    Symbol for the input quality score.
  """
  ascii_code = score + 33
  return chr(ascii_code)


def quality_scores_to_string(scores: List[int]) -> str:
  """Returns the string representation for the given list of quality scores."""
  return ''.join([quality_score_to_string(score) for score in scores])
