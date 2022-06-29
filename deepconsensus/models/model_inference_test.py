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
"""Tests for deepconsensus.models.model_inference."""

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
    checkpoint_path = test_utils.deepconsensus_testdata('model/checkpoint-1')
    params = model_configs.get_config(config_name)
    tpu = None
    tpu_topology = None
    model_utils.modify_params(params, tpu=tpu, tpu_topology=tpu_topology)
    model_inference.run_inference(
        out_dir=out_dir,
        params=params,
        checkpoint_path=checkpoint_path,
        tpu=None,
        tpu_topology=None,
        limit=-1)

    # Output directory should contain the CSV logger output.
    csv_output = glob.glob(os.path.join(out_dir, 'inference.csv'))
    self.assertLen(csv_output, 1)


if __name__ == '__main__':
  absltest.main()
