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
"""Architecture and training hyperparameters for networks."""
# pylint: disable=line-too-long
import os

import ml_collections
from deepconsensus.utils import dc_constants


############### Base params for different model architectures ###############


def _set_base_fc_hparams(params):
  """Updates given params with base values for the fully connected model."""
  # Architecture
  params.model_name = 'fc'
  params.fc_size = [256, 512, 256, 128]
  params.fc_dropout = 0.0

  params.use_bases = True
  params.use_pw = True
  params.use_ip = True
  params.use_strand = True
  params.use_ccs = True
  params.use_sn = True
  params.num_channels = 1

  params.per_base_hidden_size = 1
  params.pw_hidden_size = 1
  params.ip_hidden_size = 1
  params.strand_hidden_size = 1
  params.sn_hidden_size = 1

  # Training
  params.l2 = 0.0
  params.batch_size = 256
  params.num_epochs = 15
  params.learning_rate = 0.004
  params.buffer_size = 1000


def _set_base_conv_net_hparams(params, conv_model):
  """Updates given params with base values for ConvNet models.

  Args:
    params: configuration dictionary.
    conv_model: Defines the convolutional network architecture.
  """

  # Model
  params.model_name = 'conv_net'
  params.input_format = 'stack[base,pw,ip,sn]'
  params.conv_model = conv_model

  # Data
  params.num_channels = 5
  params.use_bases = True
  params.use_pw = True
  params.use_ip = True
  params.use_ccs = False
  params.use_sn = False
  params.use_strand = False
  params.per_base_hidden_size = 1
  params.pw_hidden_size = 1
  params.ip_hidden_size = 1
  params.sn_hidden_size = 1
  params.strand_hidden_size = 1

  # Training
  params.l2 = 0.0
  params.batch_size = 256
  params.num_epochs = 15
  params.learning_rate = 0.004
  params.buffer_size = 10000


def _set_base_transformer_hparams(params):
  """Updates given config with base values for the Transformer model."""
  # Architecture
  params.model_name = 'transformer'
  params.add_pos_encoding = True
  params.use_relative_pos_enc = True
  # Num heads should be divisible by hidden size. This value should be tuned for
  # the production setting. TODO: update this parameter after
  # tuning.
  params.num_heads = 2
  params.layer_norm = False
  params.dtype = dc_constants.TF_DATA_TYPE
  params.condense_transformer_input = False
  params.transformer_model_size = 'base'

  params.num_channels = 1
  params.use_bases = True
  params.use_pw = True
  params.use_ip = True
  params.use_ccs = True
  params.use_strand = True
  params.use_sn = True
  params.per_base_hidden_size = 1
  params.pw_hidden_size = 1
  params.ip_hidden_size = 1
  params.sn_hidden_size = 1
  params.strand_hidden_size = 1

  # Training
  params.batch_size = 256
  params.num_epochs = 50
  params.learning_rate = 1e-4
  params.buffer_size = 1000


def _set_transformer_learned_embeddings_hparams(params):
  """Updates given config with values for the learned embeddings transformer."""
  _set_base_transformer_hparams(params)
  params.model_name = 'transformer_learn_values'
  params.PW_MAX = dc_constants.PW_MAX
  params.IP_MAX = dc_constants.IP_MAX
  params.STRAND_MAX = dc_constants.STRAND_MAX
  params.SN_MAX = dc_constants.SN_MAX
  params.per_base_hidden_size = 8
  params.pw_hidden_size = 8
  params.ip_hidden_size = 8
  params.strand_hidden_size = 2
  params.sn_hidden_size = 8




############### Base params for different datasets ###############


def _set_test_data_hparams(params):
  """Updates the given config with values for a test dataset."""
  curr_dir = os.path.dirname(__file__)
  params.train_path = [
      os.path.join(curr_dir, '../testdata/human_1m/tf_examples/train/*')
  ]
  # Use same data for train/eval/hard eval because the eval test data is empty.
  params.eval_path = params.train_path
  params.test_path = params.train_path
  params.inference_path = os.path.join(
      curr_dir, '../testdata/human_1m/tf_examples/inference/*')
  params.n_examples_train = 253
  params.n_examples_eval = 253
  params.max_passes = 20

  # The test dataset uniquely sets these model-level parameters because the test
  # dataset is small and we want to keep model files small.
  params.batch_size = 1
  params.num_epochs = 1
  params.buffer_size = 10
  if params.model_name == 'fc':
    params.fc_size = [4, 4]


############### Core function for setting all config values ###############


def get_config(config_name: str) -> ml_collections.ConfigDict:
  """Returns the default configuration as instance of ConfigDict.

  Valid config names must consist of two parts: {model_name}+{dataset_name}. The
  "+" must be present as a separator between the two parts. For example,
  transformer_learn_bases+ccs is a valid name.

  Valid model names include:
    * fc
    * conv_net
    * transformer
    * transformer_learn_values

  Valid dataset names include:
    * ecoli
    * ccs
    * poa
    * test

  Args:
    config_name: String consisting of two parts, model name and dataset name,
      separated by a "+".

  Returns:
    A config dictionary containing the valid configs for the model and dataset
    specified.
  """
  params = ml_collections.ConfigDict()
  # Specify common configs here.
  params.num_classes = len(dc_constants.VOCAB)
  params.vocab_size = len(dc_constants.VOCAB)
  params.tensorboard_update_freq = 'batch'
  params.model_checkpoint_freq = 'epoch'
  params.seed = 1
  params.remove_label_gaps = False
  params.loss_function = 'alignment_loss'
  # AlignmentLoss-specific parameters here.
  params.del_cost = 10.0
  params.loss_reg = 0.1


  # Scaling factor to multiply the batch_size when using TPUs since they have
  # more memory than GPUs.
  params.tpu_scale_factor = 1
  model_config_name, dataset_config_name = config_name.split('+')
  params.model_config_name = model_config_name
  params.dataset_config_name = dataset_config_name
  params.tf_dataset = None
  params.limit = -1
  if model_config_name == 'fc':
    _set_base_fc_hparams(params)
  elif model_config_name.startswith('conv_net'):
    _, conv_sub_model = model_config_name.split('-')
    _set_base_conv_net_hparams(params, conv_sub_model)
  elif model_config_name == 'transformer':
    _set_base_transformer_hparams(params)
  elif model_config_name == 'transformer_learn_values':
    _set_transformer_learned_embeddings_hparams(params)
  else:
    raise ValueError('Unknown model_config_name: %s' % model_config_name)

  if dataset_config_name == 'poa':
    _set_human_aligned_to_poa_data_hparams(params)
  elif dataset_config_name == 'ccs':
    _set_human_aligned_to_ccs_data_hparams(params)
  elif dataset_config_name == 'ecoli':
    _set_ecoli_data_hparams(params)
  elif dataset_config_name == 'test':
    _set_test_data_hparams(params)
  else:
    raise ValueError('dataset_config_name is %s. Must be one of the following: '
                     'ccs, poa, ecoli, test' % dataset_config_name)
  return params
