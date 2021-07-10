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
"""Tests for deepconsensus.preprocess.preprocess_utils."""

from absl.testing import absltest
from absl.testing import parameterized

from deepconsensus.preprocess import preprocess_utils


class GetPacbioMoleculeNameTest(parameterized.TestCase):

  @parameterized.parameters(
      ('m54316_180808_005743/5636304/ccs', 'm54316_180808_005743/5636304'),
      ('m54316_180808_005743/5636304/truth', 'm54316_180808_005743/5636304'),
      ('m54316_180808_005743//5636304/ccs', None), ('', None))
  def test_correct_name(self, name, expected):
    """Tests that correct molecule name or None returned."""
    actual = preprocess_utils.get_pacbio_molecule_name(name)
    self.assertEqual(actual, expected)


if __name__ == '__main__':
  absltest.main()
