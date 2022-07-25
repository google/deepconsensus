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
"""Defines the Transformer model in TF 2.0.

Model paper: https://arxiv.org/pdf/1706.03762.pdf
Transformer model code source: https://github.com/tensorflow/tensor2tensor
"""

from typing import Any, Dict, Union, Iterable

import ml_collections
import tensorflow as tf

from deepconsensus.models import attention_layer
from deepconsensus.models import ffn_layer


class PrePostProcessingWrapper(tf.keras.layers.Layer):
  """Wrapper class that applies layer pre-processing and post-processing."""

  def __init__(self, layer: tf.keras.layers.Layer,
               params: ml_collections.ConfigDict):
    super(PrePostProcessingWrapper, self).__init__()
    self.layer = layer
    self.params = params
    self.postprocess_dropout = params["layer_postprocess_dropout"]

  def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
    # Create normalization layer
    self.layer_norm = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(PrePostProcessingWrapper, self).build(input_shape)

  def get_config(self) -> Dict[str, Any]:
    return {
        "params": self.params,
    }

  def call(self, x: tf.Tensor, *args, **kwargs) -> Dict[str, tf.Tensor]:
    """Calls wrapped layer with same parameters."""
    # Preprocessing: apply layer normalization
    training = kwargs["training"]

    y = self.layer_norm(x)

    # Get layer output.
    layer_output = self.layer(y, *args, **kwargs)
    y = layer_output["main_output"]

    # Postprocessing: apply dropout and residual connection
    if training:
      y = tf.nn.dropout(y, rate=self.postprocess_dropout)
    layer_output["main_output"] = x + y
    return layer_output


class EncoderStack(tf.keras.layers.Layer):
  """Transformer encoder stack.

  The encoder stack is made up of N identical layers. Each layer is composed
  of the sublayers:
    1. Self-attention layer
    2. Feedforward network (which is 2 fully-connected layers)
  """

  def __init__(self, params: ml_collections.ConfigDict):
    super(EncoderStack, self).__init__()
    self.params = params
    self.layers = []

  def build(self, input_shape: Union[tf.TensorShape, Iterable[tf.TensorShape]]):
    """Builds the encoder stack."""
    params = self.params
    for _ in range(params["num_hidden_layers"]):
      # Create sublayers for each layer.
      self_attention_layer = attention_layer.SelfAttention(
          params["hidden_size"], params["num_heads"],
          params["attention_dropout"])
      feed_forward_network = ffn_layer.FeedForwardNetwork(
          params["hidden_size"], params["filter_size"], params["relu_dropout"])

      self.layers.append([
          PrePostProcessingWrapper(self_attention_layer, params),
          PrePostProcessingWrapper(feed_forward_network, params)
      ])

    # Create final layer normalization layer.
    self.output_normalization = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, dtype="float32")
    super(EncoderStack, self).build(input_shape)

  def get_config(self) -> Dict[str, Any]:
    return {
        "params": self.params,
    }

  def call(self, encoder_inputs: tf.Tensor, attention_bias: tf.Tensor,
           inputs_padding: tf.Tensor, training: bool) -> Dict[str, tf.Tensor]:
    """Return the output of the encoder layer stacks.

    Args:
      encoder_inputs: tensor with shape [batch_size, input_length, hidden_size]
      attention_bias: bias for the encoder self-attention layer. [batch_size, 1,
        1, input_length]
      inputs_padding: tensor with shape [batch_size, input_length], inputs with
        zero paddings.
      training: boolean, whether in training mode or not.

    Returns:
      Dictionary with the following (key:value) pairs:
        "self_attention_layer_{n}": Attention layer output for every layer in
        the encoder stack with shape [batch_size, input_length, hidden_size].
        "attention_scores_{n}" : Attention map for every layer in the
        encoder stack with shape [batch_size, num_heads, input_length,
        input_length].
        "ffn_layer_{n}": Feedforward network output for every layer in the
        encoder stack with shape [batch_size, input_length, hidden_size].
        "final_output": Final output of the entire encoder stack after
        normalization with shape [batch_size, input_length, hidden_size]. Used
        as input to the fully-connected layer which outputs logits.
    """
    outputs_dict = dict()
    for n, layer in enumerate(self.layers):
      # Run inputs through the sublayers.
      self_attention_layer = layer[0]
      feed_forward_network = layer[1]

      with tf.name_scope("layer_%d" % n):
        with tf.name_scope("self_attention"):
          layer_outputs = self_attention_layer(
              encoder_inputs, attention_bias, training=training)
          encoder_inputs = layer_outputs["main_output"]
          # Add attention layer outputs and attention map scores to outputs.
          outputs_dict[f"self_attention_layer_{n}"] = encoder_inputs
          outputs_dict[f"attention_scores_{n}"] = layer_outputs[
              "attention_scores"]
        with tf.name_scope("ffn"):
          layer_outputs = feed_forward_network(
              encoder_inputs, training=training)
          encoder_inputs = layer_outputs["main_output"]
          # Add output of the feedforward network to outputs.
          outputs_dict[f"ffn_layer_{n}"] = encoder_inputs

    # Add normalized final output of the entire encoder stack to outputs.
    outputs_dict["final_output"] = self.output_normalization(encoder_inputs)
    return outputs_dict
