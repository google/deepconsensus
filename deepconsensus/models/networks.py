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
"""TF2 + tf.keras implementations of networks for DeepConsensus."""

import logging
from typing import Any, Callable, Dict, Optional, Tuple

import ml_collections
import tensorflow as tf

from deepconsensus.models import data_providers
from deepconsensus.models import encoder_stack
from official.nlp.modeling import layers


# TODO: Looking into removing this eventually.
class ModifiedOnDeviceEmbedding(layers.OnDeviceEmbedding):
  """Subclass of OnDeviceEmbedding, init similar to EmbeddingSharedWeights."""

  def __init__(self, vocab_size, embedding_width, **kwargs):
    # Set initializer and scale_factor to match the original implementation in
    # tensorflow_models/official/legacy/transformer/embedding_layer.py
    super().__init__(
        vocab_size,
        embedding_width,
        initializer=tf.random_normal_initializer(
            mean=0., stddev=embedding_width**-0.5),
        scale_factor=embedding_width**0.5,
        **kwargs)

  def call(self, inputs):
    # make sure 0 ids match to zero emebeddings.
    embeddings = super().call(inputs)
    mask = tf.cast(tf.not_equal(inputs, 0), embeddings.dtype)
    embeddings *= tf.expand_dims(mask, -1)
    return embeddings


# pylint: disable=invalid-name
def FullyConnectedNet(params: ml_collections.ConfigDict) -> tf.keras.Model:
  """Fully connected neural network architecture."""

  inputs = tf.keras.Input(
      shape=(params.hidden_size, params.max_length, params.num_channels))
  l2_reg = tf.keras.regularizers.l2
  net = inputs
  net = tf.keras.layers.Flatten()(net)
  for i in range(len(params.fc_size)):
    net = tf.keras.layers.Dense(
        units=params.fc_size[i],
        activation=tf.nn.relu,
        kernel_regularizer=l2_reg(params.l2))(
            net)
    net = tf.keras.layers.Dropout(rate=params.fc_dropout)(net)

  net = tf.keras.layers.Dense(units=params.max_length * params.num_classes)(net)
  net = tf.keras.layers.Reshape((params.max_length, params.num_classes))(net)
  net = tf.keras.layers.Softmax(axis=-1)(net)
  outputs = net
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_conv_sub_model(
    conv_model
) -> Tuple[Callable[..., tf.Tensor], Callable[[tf.keras.Model],
                                              tf.keras.Model]]:
  """Returns a predefined convolutional architecture."""
  if conv_model == 'resnet50':
    return tf.keras.applications.ResNet50V2, tf.keras.applications.resnet_v2.preprocess_input
  elif conv_model == 'resnet101':
    return tf.keras.applications.ResNet101V2, tf.keras.applications.resnet_v2.preprocess_input
  elif conv_model == 'resnet152':
    return tf.keras.applications.ResNet152V2, tf.keras.applications.resnet_v2.preprocess_input
  else:
    raise NotImplementedError(f'conv model "{conv_model}" not found')


# pylint: disable=invalid-name
class ConvNet(tf.keras.Model):
  """Convolutional neural network architecture."""

  def __init__(self, params: ml_collections.ConfigDict, **kwargs):
    super(ConvNet, self).__init__(params, **kwargs)
    # Most conv models only accept 3 channels.
    self.resnet_input_shape = (params.hidden_size, params.max_length, 3)
    self.dimensions = params.max_length * params.num_classes

    model, self.conv_preprocess = get_conv_sub_model(params.conv_model)
    self.model = model(
        include_top=False,
        weights=None,
        input_shape=self.resnet_input_shape,
        pooling='avg')
    self.use_sn = params.use_sn
    self.max_length = params.max_length
    self.num_classes = params.num_classes

    # Define layers
    self.layer_dense = tf.keras.layers.Dense(units=self.dimensions)

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    # Most conv models only accept 3 channels;
    # The sn channel must be removed and optionally
    # added back at the end. CCS rows not being used currently for this model.
    input_rows, _, sn_rows = tf.split(inputs, [3, 1, 1], 3)

    cn_input = self.conv_preprocess(input_rows)
    net = self.model(cn_input, training=training)

    if self.use_sn:
      logging.info('Using SN Values')
      # sn_rows was padded previously to match the input dimensions
      # Crop it here back to 4 rows.
      sn_rows = tf.image.crop_to_bounding_box(sn_rows, 0, 0, 4, self.max_length)
      sn_rows = tf.keras.layers.Flatten()(sn_rows)
      net = tf.keras.layers.Flatten()(net)
      net = tf.concat([net, sn_rows], 1)
    else:
      net = tf.keras.layers.Flatten()(net)

    net = self.layer_dense(net)
    net = tf.keras.layers.Reshape((self.max_length, self.num_classes))(net)
    net = tf.keras.layers.Softmax(axis=-1)(net)
    output = net
    return output


class EncoderOnlyTransformer(tf.keras.Model):
  """Modified encoder-only transformer model for DeepConsensus.

  This implementation is similar to

  tensorflow_models/official/legacy/transformer/transformer.py
  tensorflow_models/official/nlp/modeling/models/seq2seq_transformer.py

  with some simplifications and extensions.

  The main changes are:

  * Removing logic relating to converting tokens to embeddings, since the
  DeepConsensus is already in the form of vectors for each position.

  * Removing the decoder, since we only want to run the encoder.

  * Adding additional layers on top of the encoder for the per-position
  classification task.
  """

  def __init__(self,
               params: ml_collections.ConfigDict,
               name: Optional[str] = None,
               **kwargs):
    super().__init__(**kwargs)
    self.params = params
    if self.params.add_pos_encoding:
      self.position_embedding = layers.RelativePositionEmbedding(
          hidden_size=self.params['hidden_size'])
    self.encoder_stack = encoder_stack.EncoderStack(params)
    self.fc1 = tf.keras.layers.Dense(
        units=(params['vocab_size']),
        activation=None,
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros')
    self.softmax = tf.keras.layers.Softmax()

  def get_config(self) -> Dict[str, Any]:
    return {
        'params': self.params,
    }

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    """Runs a forward pass of the model.

    Args:
      inputs: tensor of shape (batch_size, hidden_size, input_length
        num_channels).
      training: boolean, whether in training mode or not.

    Returns:
      Output from softmax layer, which is a distribution over the vocabulary at
      each position in the sequence.
    """
    with tf.name_scope('Transformer'):
      logits = self.get_logits(inputs, training=training)
      preds = self.softmax(logits)
      return preds

  def get_logits(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    """Get logits of the model.

    Args:
      inputs: tensor of shape (batch_size, hidden_size, input_length
        num_channels).
      training: boolean, whether in training mode or not.

    Returns:
      Output logits over the vocabulary at each position in the sequence. The
        output tensor is of shape (batch_size, length, vocab_size).
    """

    # Get rid of the channel dimension as we only have one channel.
    inputs = tf.squeeze(inputs, -1)

    # `inputs` is of shape (batch_size, hidden_size, input_length). For the
    # Transformer, we need to change the format to be the following:
    # (batch_size, input_length, hidden_size).
    inputs = tf.transpose(inputs, [0, 2, 1])

    # Attention_bias for our model should be all 0s with shape
    # (batch_size, 1, 1, input_length). See model_utils.get_padding_bias
    # to see how this is calculated in the base model.
    all_zeros = tf.reduce_sum(tf.zeros_like(inputs), -1)
    attention_bias = tf.expand_dims(tf.expand_dims(all_zeros, 1), 1)

    # Run inputs through the encoder. Encoder returns logits from dense layer.
    encoder_outputs = self.encode(inputs, attention_bias, training)
    return encoder_outputs

  def encode(self, inputs: tf.Tensor, attention_bias: tf.Tensor,
             training: bool) -> tf.Tensor:
    """Runs the input through Encoder stack and problem-specific layers."""

    with tf.name_scope('encode'):

      # The input for each position is already a vector, so we do not use
      # embeddings here, unlike the base model. Base model input is a token at
      # each position, which must first be embedded as a vector. In the future,
      # we may want to use embeddings for part of the input, such as the bases,
      # so that we can learn the scale of values.
      encoder_inputs = inputs

      # Positional embedding only works when we have an even value for the
      # hidden_size. If hidden_size is odd, add an empty row to make it even.
      if self.params.add_pos_encoding and encoder_inputs.shape[2] % 2 != 0:
        empty_row = tf.zeros(
            shape=(encoder_inputs.shape[0], encoder_inputs.shape[1], 1))
        encoder_inputs = tf.concat([encoder_inputs, empty_row], axis=-1)
        assert self.params.hidden_size == encoder_inputs.shape[2]

      # All values in `input_padding` should be 0 and shape should be
      # (batch_size, input_length). See model_utils.get_padding to see how this
      # is computed for the base model.
      inputs_padding = tf.reduce_sum(tf.zeros_like(encoder_inputs), -1)

      # Cast input `attention_bias` to correct type, as done in the base model.
      attention_bias = tf.cast(attention_bias, self.params['dtype'])

      # Add positional encoding to the input. The scale of the positional
      # encoding relative to the input values will matter since we are not
      # learning the input embedding.
      if self.params['add_pos_encoding']:
        with tf.name_scope('add_pos_encoding'):
          pos_encoding = self.position_embedding(inputs=encoder_inputs)
          pos_encoding = tf.cast(pos_encoding, self.params['dtype'])
          encoder_inputs += pos_encoding

      # Add dropout when training.
      if training:
        encoder_inputs = tf.nn.dropout(
            encoder_inputs, rate=self.params['layer_postprocess_dropout'])

      # Pass inputs through the encoder. As mentioned above, `inputs_padding` is
      # not actually used by EncoderStack.call. Encoder stack output has shape
      # (batch_size, input_length, hidden_size).
      encoder_outputs = self.encoder_stack(
          encoder_inputs, attention_bias, inputs_padding, training=training)

      # Pass through dense layer and output logits over vocab for each position.
      encoder_outputs = self.fc1(encoder_outputs)
      return encoder_outputs

  def decode(self, encoder_outputs: tf.Tensor, attention_bias: tf.Tensor,
             training: bool) -> tf.Tensor:
    """Returns the outputs from the encoder."""

    raise NotImplementedError

  def predict(self, encoder_inputs: tf.Tensor) -> tf.Tensor:
    """Returns the argmax of the decoder output, which comes from a softmax."""

    # The base model also has a predict method that behaves differently. This
    # predict function is consistent with how predict behaves for other
    # DeepConsensus models (conv, FC), but we may want to change this in the
    # future to match the transformer base class. For more details, see:
    # https://github.com/tensorflow/models/blob/bc71d8e9e155d34a38af8489ad4cbb2fde6fa152/official/nlp/transformer/transformer.py#L279
    return self.call(encoder_inputs, training=False)


class EncoderOnlyLearnedValuesTransformer(EncoderOnlyTransformer):
  """Modified transformer that learns embeddings for the bases."""

  def __init__(self,
               params: ml_collections.ConfigDict,
               name: Optional[str] = None):
    super(EncoderOnlyLearnedValuesTransformer, self).__init__(params, name=name)
    if params.use_bases:
      self.bases_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=params['vocab_size'],
          embedding_width=params['per_base_hidden_size'],
          name='bases_embedding')
    if params.use_pw:
      pw_vocab_size = params.PW_MAX + 1
      self.pw_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=pw_vocab_size,
          embedding_width=params['pw_hidden_size'],
          name='pw_embedding')
    if params.use_ip:
      ip_vocab_size = params.IP_MAX + 1
      self.ip_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=ip_vocab_size,
          embedding_width=params['ip_hidden_size'],
          name='ip_embedding')

    if params.use_sn:
      sn_vocab_size = params.SN_MAX + 1
      self.sn_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=sn_vocab_size,
          embedding_width=params['sn_hidden_size'],
          name='sn_embedding')

    if params.use_strand:
      strand_vocab_size = params.STRAND_MAX + 1
      self.strand_embedding_layer = ModifiedOnDeviceEmbedding(
          vocab_size=strand_vocab_size,
          embedding_width=params['strand_hidden_size'],
          name='strand_embedding')

    # Define a dense layer to linearly map the concatenated embeddings of
    # all subreads at a given position to a smaller dimension
    # (transformer_input_size) in order to keep the transformer layers small.
    if self.params.condense_transformer_input:
      logging.info('Condensing input.')
      self.transformer_input_condenser = tf.keras.layers.Dense(
          units=(params.transformer_input_size),
          activation=None,
          use_bias=False,
          kernel_initializer='glorot_uniform',
          bias_initializer='zeros')

  def encode(self, inputs: tf.Tensor, attention_bias: tf.Tensor,
             training: bool) -> tf.Tensor:
    """Runs the input through Encoder stack and problem-specific layers."""

    # Input to embedding layer is [batch_size, length] and output will be
    # [batch_size, length, embedding_size]. Embed each row of the input
    # separately and then concatenate.
    embedded_inputs = []
    base_indices, pw_indices, ip_indices, strand_indices, ccs_indices, sn_indices = data_providers.get_indices(
        self.params['max_passes'])
    if self.params.use_bases:
      for i in range(*base_indices):
        # Shape: [batch_size, length, per_base_hidden_size]
        embedded = self.bases_embedding_layer(
            tf.cast(inputs[:, :, i], tf.int32))
        embedded_inputs.append(embedded)

    if self.params.use_pw:
      for i in range(*pw_indices):
        # Shape: [batch_size, length, pw_hidden_size]
        embedded = self.pw_embedding_layer(tf.cast(inputs[:, :, i], tf.int32))
        embedded_inputs.append(embedded)

    if self.params.use_ip:
      for i in range(*ip_indices):
        # Shape: [batch_size, length, ip_hidden_size]
        embedded = self.ip_embedding_layer(tf.cast(inputs[:, :, i], tf.int32))
        embedded_inputs.append(embedded)

    if self.params.use_strand:
      for i in range(*strand_indices):
        embedded = self.strand_embedding_layer(
            tf.cast(inputs[:, :, i], tf.int32))
        embedded_inputs.append(embedded)

    if self.params.use_ccs:
      for i in range(*ccs_indices):
        embedded = self.bases_embedding_layer(
            tf.cast(inputs[:, :, i], tf.int32))
        embedded_inputs.append(embedded)

    # TODO: experiment with computing a weighted average using snr as
    # weights to aggregate subread-level embeddings (instead of concatenating).
    if self.params.use_sn:
      # The last four elements in the last dimension in the inputs tensor
      # correspond to the four signal-to-noise ratio scores for A, G, C, T.
      for i in range(*sn_indices):
        embedded = self.sn_embedding_layer(tf.cast(inputs[:, :, i], tf.int32))
        embedded_inputs.append(embedded)

    embedded_inputs = tf.concat(embedded_inputs, axis=-1)
    embedded_inputs = tf.cast(embedded_inputs, self.params['dtype'])

    if self.params.condense_transformer_input:
      # Condense the transformer input at each position to a smaller vector to
      # reduce the transformer hidden size, since the transformer model size is
      # quadratic in its hidden size.
      # Shape: [batch_size, length, transformer_input_size]
      transformer_input = self.transformer_input_condenser(embedded_inputs)
    else:
      transformer_input = embedded_inputs

    return super(EncoderOnlyLearnedValuesTransformer,
                 self).encode(transformer_input, attention_bias, training)
