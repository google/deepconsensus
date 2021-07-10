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
