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
"""Tests for deepconsensus.models.model_inference_transforms."""

from absl.testing import absltest
from absl.testing import parameterized

from deepconsensus.models import model_inference_transforms


class EditDistanceTest(parameterized.TestCase):

  @parameterized.parameters([
      ['ATCG', 'ATCG', 0],
      ['ATCG', 'TT', 3],
      ['ATCG', 'ZZZZ', 4],
      [' A T C G  ', 'ATCG', 0],
  ])
  def test_edit_distance(self, str1, str2, expected_edit_distance):
    ed = model_inference_transforms.edit_distance(str1, str2)
    self.assertEqual(ed, expected_edit_distance)


class RepeatContentTest(parameterized.TestCase):

  @parameterized.parameters([['      ', 0.0], ['ABCD', 0.0], ['AAABBBCD', 0.75],
                             ['AAABBBCCCDDD', 1.0],
                             ['AAA BBB CCC DDD    ', 1.0]])
  def test_repeat_content(self, seq, expected_homopolymer_content):
    hcontent = model_inference_transforms.homopolymer_content(seq)
    self.assertEqual(hcontent, expected_homopolymer_content)


if __name__ == '__main__':
  absltest.main()
