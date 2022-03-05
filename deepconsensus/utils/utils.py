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
"""Utilities for DeepConsensus."""

import math
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


def quality_string_to_array(quality_string: str) -> List[int]:
  """Returns the int array representation for the given quality string."""
  return [ord(char) - 33 for char in quality_string]


def avg_phred(base_qualities: List[int]) -> float:
  """Returns the average phred quality of the provided base qualities."""
  if not base_qualities:
    return 0
  return -10 * math.log(
      sum([10**(i / -10) for i in base_qualities]) / len(base_qualities), 10)
