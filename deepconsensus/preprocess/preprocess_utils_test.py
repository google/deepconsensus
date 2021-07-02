"""Tests for deepconsensus .preprocess.preprocess_utils."""

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
