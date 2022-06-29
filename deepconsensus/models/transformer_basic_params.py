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
"""Defines Transformer basic model parameters for each model size."""

import collections


BASE_PARAMS = collections.defaultdict(
    lambda: None,  # Set default value to None.

    # Input params
    default_batch_size=2048,  # Maximum number of tokens per batch of examples.
    default_batch_size_tpu=32768,
    max_length=256,  # Maximum number of tokens per example.

    # Model params
    initializer_gain=1.0,  # Used in trainable variable initialization.
    vocab_size=33708,  # Number of tokens defined in the vocabulary file.
    hidden_size=512,  # Model dimension in the hidden layers.
    num_hidden_layers=6,  # Number of layers in the encoder and decoder stacks.
    num_heads=8,  # Number of heads to use in multi-headed attention.
    filter_size=2048,  # Inner layer dimension in the feedforward network.

    # Dropout values (only used when training)
    layer_postprocess_dropout=0.1,
    attention_dropout=0.1,
    relu_dropout=0.1,

    # Training params
    label_smoothing=0.1,
    learning_rate=2.0,
    learning_rate_decay_rate=1.0,
    learning_rate_warmup_steps=16000,

    # Optimizer params
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,

    # Default prediction params
    extra_decode_length=50,
    beam_size=4,
    alpha=0.6,  # used to calculate length normalization in beam search

    # TPU specific parameters
    use_tpu=False,
    static_batch=False,
    allow_ffn_pad=True,
)

BIG_PARAMS = BASE_PARAMS.copy()
BIG_PARAMS.update(
    default_batch_size=4096,

    # default batch size is smaller than for BASE_PARAMS due to memory limits.
    default_batch_size_tpu=16384,

    hidden_size=1024,
    filter_size=4096,
    num_heads=16,
)

# Parameters for running the model in multi gpu. These should not change the
# params that modify the model shape (such as the hidden_size or num_heads).
BASE_MULTI_GPU_PARAMS = BASE_PARAMS.copy()
BASE_MULTI_GPU_PARAMS.update(
    learning_rate_warmup_steps=8000
)

BIG_MULTI_GPU_PARAMS = BIG_PARAMS.copy()
BIG_MULTI_GPU_PARAMS.update(
    layer_postprocess_dropout=0.3,
    learning_rate_warmup_steps=8000
)

# Parameters for testing the model
TINY_PARAMS = BASE_PARAMS.copy()
TINY_PARAMS.update(
    default_batch_size=1024,
    default_batch_size_tpu=1024,
    hidden_size=32,
    num_heads=4,
    filter_size=256,
)
