"""Tests for deepconsensus .models.model_inference."""

import glob
import os

from absl.testing import absltest

from deepconsensus.models import model_configs
from deepconsensus.models import model_inference
from deepconsensus.models import model_utils
from deepconsensus.utils import test_utils


class ModelInferenceTest(absltest.TestCase):

  def test_inference_e2e(self):
    """Tests that inference finishes running and an output file is created."""

    config_name = 'transformer_learn_values+test'
    out_dir = self.create_tempdir().full_path
    checkpoint_path = test_utils.deepconsensus_testdata(
        'ecoli/output/model/checkpoint-2')
    params = model_configs.get_config(config_name)
    tpu = None
    tpu_topology = None
    model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)
    model_inference.run_inference(
        out_dir=out_dir,
        params=params,
        checkpoint_path=checkpoint_path,
        master=None,
        tpu_topology=None,
        limit=-1)

    # Output directory should contain the CSV logger output.
    csv_output = glob.glob(os.path.join(out_dir, 'inference.csv'))
    self.assertLen(csv_output, 1)


if __name__ == '__main__':
  absltest.main()
