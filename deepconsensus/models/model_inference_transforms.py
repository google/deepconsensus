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
"""DoFns for running inference with Beam and writing out predictions."""

import itertools

from deepconsensus.utils import dc_constants


def edit_distance(s1: str, s2: str) -> int:
  """Calculates the Levenstein edit distance.

  Edit distance represents the number of insertions, deletions,
  and substitutions required to change s1 to s2. For example,

  CAT -> BAT  = 1
  CAT -> BATS = 2

  Args:
    s1: String 1
    s2: String 2

  Returns:
    The Levenstein edit distance.

  """
  if len(s1) > len(s2):
    s1, s2 = s2, s1

  # Remove all gaps/padding from strings.
  s1 = s1.replace(dc_constants.GAP_OR_PAD, '')
  s1 = s1.replace(dc_constants.GAP_OR_PAD, '')
  s2 = s2.replace(dc_constants.GAP_OR_PAD, '')
  s2 = s2.replace(dc_constants.GAP_OR_PAD, '')

  distances = range(len(s1) + 1)
  for i2, c2 in enumerate(s2):
    distances_ = [i2 + 1]
    for i1, c1 in enumerate(s1):
      if c1 == c2:
        distances_.append(distances[i1])
      else:
        distances_.append(1 + min((distances[i1], distances[i1 + 1],
                                   distances_[-1])))
    distances = distances_
  return distances[-1]


def homopolymer_content(seq: str) -> float:
  """Calculates proportion of seq composed of 3+ repeated bases."""
  seq = seq.replace(dc_constants.GAP_OR_PAD, '').strip(dc_constants.GAP_OR_PAD)
  if not seq:
    return 0.0
  run_length_encoding = [len(list(g)) for _, g in itertools.groupby(seq)]
  hcontent = sum([x for x in run_length_encoding if x >= 3]) / len(seq)
  return round(hcontent, 2)
