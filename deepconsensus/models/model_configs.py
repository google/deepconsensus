"""Architecture and training hyperparameters for networks."""

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
  # Num heads should be divisible by hidden size.
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
  params.num_epochs = 25
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


def _set_dnabert_hparams(params):
  """Updates given config with values for DNA-BERT."""
  _set_transformer_learned_embeddings_hparams(params)
  params.use_dnabert = True
  params.dnabert_max_seq_length = 259
  params.dnabert_desired_hidden_size = 8
  params.bert_config_file = '/cns/yo-d/home/wammar/public/congenrep/configs/96hidden+maxlen259.json'
  params.pretrained_dnabert_checkpoint = '/cns/yo-d/home/wammar/public/congenrep/models/beamout/ecoli_small/ctl_step_40000.ckpt-80'

############### Base params for different datasets ###############


def _set_human_aligned_to_poa_data_hparams(params):
  """Updates the given config with values for human data aligned to POA."""
  params.train_path = '/readahead/1G/placer/prod/home/brain-genomics/gunjanbaid/deepconsensus/tfexamples/human_m54238_180901_011437_alignedToPoa/20201106/train'
  params.eval_path = '/readahead/1G/placer/prod/home/brain-genomics/gunjanbaid/deepconsensus/tfexamples/human_m54238_180901_011437_alignedToPoa/20201106/eval'
  params.hard_eval_path = '/readahead/1G/placer/prod/home/brain-genomics/gunjanbaid/deepconsensus/mv_hard_ex/human_m54238_180901_011437_alignedToPoa/20201106/deepconsensus'
  params.test_path = '/readahead/1G/placer/prod/home/brain-genomics/gunjanbaid/deepconsensus/inference_tfexamples/human_m54238_180901_011437_alignedToPoa/20210112/eval'
  params.train_data_size = 12141989
  params.eval_data_size = 321982
  params.max_passes = 20


def _set_human_aligned_to_ccs_data_hparams(params):
  """Updates the given config with values for human data aligned to CCS."""
  params.tf_dataset = 'tf20210608-6df586ca'


def _set_ecoli_data_hparams(params):
  """Updates the given config with values for E. Coli data."""
  params.train_path = '/readahead/1G/placer/prod/home/brain-genomics/gunjanbaid/deepconsensus/tfexamples/20200716_CL321602084_100x64/train'
  params.eval_path = '/readahead/1G/placer/prod/home/brain-genomics/gunjanbaid/deepconsensus/tfexamples/20200716_CL321602084_100x64/eval'
  params.hard_eval_path = '/readahead/1G/placer/prod/home/brain-genomics/gunjanbaid/deepconsensus/mv_hard_ex/20200819_CL327389574_100x64/eval'
  params.test_path = params.eval_path
  params.train_data_size = 17949878
  params.eval_data_size = 1774687
  params.max_passes = 20


def _set_test_data_hparams(params):
  """Updates the given config with values for a test dataset."""
  params.train_path = '/cns/is-d/home/brain-genomics/deepconsensus/testdata/20210611/ecoli_testdata/tf_examples/train'
  # Use same data for train/eval/hard eval because the eval test data is empty.
  params.eval_path = params.train_path
  params.hard_eval_path = params.train_path
  params.test_path = params.train_path
  params.train_data_size = 253
  params.eval_data_size = 253
  params.max_passes = 30

  # The test dataset uniquely sets these model-level parameters
  # because the test dataset is very small.
  params.batch_size = 1
  params.num_epochs = 2
  params.buffer_size = 10


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
    * transformer_learn_bases_dnabert

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
  # dnabert is disabled by default.
  params.use_dnabert = False
  params.dnabert_desired_hidden_size = 0
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
  elif model_config_name == 'transformer_learn_bases_dnabert':
    _set_dnabert_hparams(params)
    params.use_pw = False
    params.use_ip = False
    params.batch_size = 32
    params.transformer_input_size = 160
    params.condense_transformer_input = True
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
