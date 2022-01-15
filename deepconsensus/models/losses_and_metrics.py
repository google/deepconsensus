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
"""Custom metrics and losses for DeepConsensus models."""

from typing import Callable, Optional, Tuple

import tensorflow as tf

from deepconsensus.utils import dc_constants


class PerExampleAccuracy(tf.keras.metrics.Accuracy):
  """Computes per-example accuracy."""

  def __init__(self, name: str = 'per_example_accuracy', **kwargs):
    super(PerExampleAccuracy, self).__init__(name=name, **kwargs)

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred_scores: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    """Accumulates running per-example accuracy."""
    del sample_weight  # We use the mask calculated here instead.

    # Left shift the label and prediction and compare.
    y_true = tf.cast(left_shift_sequence(y_true), dc_constants.TF_DATA_TYPE)
    # Convert pred scores and left shift.
    y_pred = tf.cast(
        tf.argmax(y_pred_scores, axis=-1), dc_constants.TF_DATA_TYPE)
    y_pred = left_shift_sequence(y_pred)

    # Count matching positions per row.
    y_pred_matches = tf.math.count_nonzero(tf.equal(y_true, y_pred), axis=-1)
    # Count total positions per row.
    y_true_counts = tf.math.count_nonzero(tf.ones_like(y_pred), axis=-1)
    # Calculate accuracy where matching positions == total by row.
    super().update_state(y_pred_matches, y_true_counts)


class PerClassAccuracy(tf.keras.metrics.Accuracy):
  """Compute per-position accuracy for the given class."""

  def __init__(self, class_value: int, name: Optional[str] = None, **kwargs):
    if not name:
      name = str(class_value)
    name = f'{name}_per_class_accuracy'
    self.class_value = class_value
    super(PerClassAccuracy, self).__init__(name=name, **kwargs)

  def update_state(self,
                   y_true: tf.Tensor,
                   y_pred_scores: tf.Tensor,
                   sample_weight: Optional[tf.Tensor] = None) -> None:
    """Accumulates running per-position accuracy for the given class."""
    del sample_weight  # We use the mask calculated here instead.
    y_pred = tf.cast(
        tf.argmax(y_pred_scores, axis=-1), dc_constants.TF_DATA_TYPE)
    mask = tf.cast(
        tf.equal(y_true, self.class_value), dc_constants.TF_DATA_TYPE)
    super().update_state(y_true, y_pred, sample_weight=mask)


@tf.function
def left_shift_sequence(y_true: tf.Tensor) -> tf.int32:
  """Removes internal gaps and shifts labels to the left.

  Args:
    y_true: Label tensor.

  Returns:
    left shifted y_true

  """
  gap_token = dc_constants.VOCAB.find(dc_constants.GAP_OR_PAD)
  shape = tf.shape(y_true)
  seq_length = shape[1]

  ixs = tf.broadcast_to(tf.range(seq_length), shape)
  # Sorting is performed in 2 stages. Sort internal gaps back by increasing
  # an index by the seq length, perform sort, then subtract to return
  # original index.
  sort_order = tf.sort(tf.where(y_true != gap_token, ixs, seq_length + ixs))
  sort_order = tf.where(sort_order < seq_length, sort_order,
                        sort_order - seq_length)
  y_true_left_aligned = tf.gather(y_true, sort_order, axis=1, batch_dims=-1)
  return y_true_left_aligned


# Type aliases to represent "pointwise" cost functions for alignment loss.
SubsCostFn = Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
InsCostFn = Callable[[tf.Tensor], tf.Tensor]


def xentropy_subs_cost_fn(y_true: tf.Tensor,
                          y_pred: tf.Tensor,
                          eps: float = 1e-7) -> tf.Tensor:
  """Pointwise cross-entropy substitution cost function for alignment loss.

  Args:
    y_true: A tf.Tensor<int>[batch, m] representing the ground-truth sequences.
    y_pred: A tf.Tensor<float>[batch, n, n_tokens] representing the scores for
      for predicted sequences. It is assumed that y_pred[b][l] lies in a k-dim
      probability simplex.
    eps: A small positive float. All scores in y_pred will be clipped to [eps, 1
      - eps] for numerical stability.

  Returns:
    A tf.Tensor<float>[batch, m, n] such that out[b][l1][l2] represents the
    (sparse) cross-entropy loss between y_true[b][l1] and y_pred[b][l2].
  """
  y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)
  y_true, y_pred = tf.expand_dims(y_true, 2), tf.expand_dims(y_pred, 1)
  return -tf.reduce_sum(tf.math.xlogy(y_true, y_pred), axis=-1)


def xentropy_ins_cost_fn(y_pred: tf.Tensor, eps=1e-7) -> tf.Tensor:
  """Pointwise cross-entropy insertion cost function for alignment loss.

  Args:
    y_pred: A tf.Tensor<float>[batch, n, n_tokens] representing the scores for
      for predicted sequences. It is assumed that y_pred[b][l] lies in a k-dim
      probability simplex.
    eps: A small positive float. All scores in y_pred will be clipped to [eps, 1
      - eps] for numerical stability.

  Returns:
    A tf.Tensor<float>[batch, n] such that out[b][l] represents the
    cross-entropy loss between dc_constants.GAP_OR_PAD and y_pred[b][l].
  """
  gap_token = dc_constants.VOCAB.find(dc_constants.GAP_OR_PAD)
  ins_scores = tf.clip_by_value(y_pred[..., gap_token], eps, 1 - eps)
  return -tf.math.log(ins_scores)


class AlignmentLoss(tf.keras.losses.Loss):
  r"""Implements a differentiable alignment loss for DeepConsensus.

  #TODO: support for from_logits argument.
  #TODO: support for annealing schedules (depending on DC API?).

  Attributes:
    subs_cost_fn: A (batched) function $\Delta^{B \times L_1} \times \Delta^{B
      \times L_2} \rightarrow \mathbb{R}_{+}^{B \times L_1 \times L_2}$
      computing the "outer product" per-position costs for a batch of B
      sequences `y_true` and their corresponding predictions `y_pred`. It is
      assumed that $L_2 \ge L_1$ and $\Delta$ represents the k-dimensional
      probability simplex.
    ins_cost_fn: A (batched_ function $\Delta^{B \times L} \rightarrow
      \mathbb{R}_{+}^{B \times L}$ computing the per-position insertion cost for
      a batch of B predictions `y_pred`. \Delta$ represents the k-dimensional
      probability simplex.
    del_cost: A float representing the (constant) cost of deletions.
    loss_reg: A float representing the regularization strength. Set to None to
      disable regularization (i.e. to compute hard alignments).
    width: An int representing the width of the alignement path. Set to None to
      remove this constraint.
    reduction: (Optional) type of `tf.keras.losses.Reduction` to apply to loss.
      Default value is `AUTO`. When used in custom training loops under the
      scope of `tf.distribute.Strategy`, must be set to `NONE` or `SUM`.
  """

  def __init__(
      self,
      subs_cost_fn: SubsCostFn = xentropy_subs_cost_fn,
      ins_cost_fn: InsCostFn = xentropy_ins_cost_fn,
      del_cost: float = 1.0,
      loss_reg: Optional[float] = 1.0,
      width: Optional[int] = None,
      reduction: tf.keras.losses.Reduction = tf.keras.losses.Reduction.AUTO):
    super(AlignmentLoss, self).__init__(reduction=reduction)
    self.subs_cost_fn = subs_cost_fn
    self.ins_cost_fn = ins_cost_fn
    self.del_cost = del_cost
    self.loss_reg = loss_reg
    self.width = width

  @staticmethod
  def preprocess_y_true(
      y_true: tf.Tensor,
      dtype: tf.DType = tf.float32,
  ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Applies AlignmentLoss-specific preprocessing to labels tensor.

    Args:
      y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
        sequences.
      dtype: The dtype for the one-hot encoded output tensor of sequence labels.

    Returns:
      A tuple (y_true_oh, seq_lens) such that
        +  y_true_oh is a tf.Tensor<dtype>[batch, m, n_tokens], where n_tokens
           is the number of tokens in dc_constants.VOCAB. It contains a one-hot
           representation of the input y_true, with dc_constants.GAP_OR_PAD
           tokens removed and extra dc_constants.GAP_OR_PAD tokens appended if
           necessary.
        +  seq_lens is a tf.Tensor<int>[batch] containing the length of each
           label sequence in y_true, excluding any pad and gap tokens.
    """
    # Ensures y_true is of integer type.
    y_true = tf.cast(y_true, tf.int32)
    # Removes internal gaps, shifting sequences left and adding padding when
    # necessary.
    y_true = left_shift_sequence(y_true)
    # Computes per-example label sequence length, excluding padding.
    pad_token = dc_constants.VOCAB.find(dc_constants.GAP_OR_PAD)
    seq_lens = tf.reduce_sum(tf.cast(y_true != pad_token, y_true.dtype), -1)
    # Converts y_true to one-hot.
    n_tokens = len(dc_constants.VOCAB)
    y_true_oh = tf.one_hot(y_true, depth=n_tokens, dtype=dtype)
    return y_true_oh, seq_lens

  @staticmethod
  def preprocess_y_pred(y_pred: tf.Tensor) -> tf.Tensor:
    # Ensures predicted scores add to one.
    y_pred = y_pred / tf.reduce_sum(y_pred, axis=-1, keepdims=True)
    return y_pred

  @staticmethod
  def wavefrontify(tensor: tf.Tensor) -> tf.Tensor:
    """Rearranges batch of input 2D tensors for vectorized wavefront algorithm.

    Args:
      tensor: A tf.Tensor<dtype>[batch, len1, len2].

    Returns:
      A single tf.Tensor<dtype>[len1 + len2 - 1, len1, batch] satisfying
        out[k][i][n] = t[n][i][k - i]
      if the RHS is well-defined, and 0 otherwise.
      In other words, for each len1 x len2 matrix t[n], out[..., n] is a
      (len1 + len2 - 1) x len1 matrix whose rows correspond to antidiagonals of
      t[n].
    """
    b, l1, l2 = tf.shape(tensor)[0], tf.shape(tensor)[1], tf.shape(tensor)[2]
    n_pad, padded_len = l1 - 1, l1 + l2 - 1

    ta = tf.TensorArray(tensor.dtype, size=l1, clear_after_read=True)
    for i in tf.range(l1):
      row_i = tf.squeeze(tf.slice(tensor, [0, i, 0], [b, 1, l2]), axis=1)
      row_i = tf.pad(row_i, [[0, 0], [n_pad, n_pad]])
      row_i = tf.slice(row_i, [0, n_pad - i], [b, padded_len])
      ta = ta.write(i, row_i)  # row_i[b, padded_len]
    ta = ta.stack()  # ta[l1, b, padded_len]

    return tf.transpose(ta, (2, 0, 1))  # out[padded_len, l1, b]

  @staticmethod
  def wavefrontify_vec(tensor: tf.Tensor, len1: int) -> tf.Tensor:
    """Rearranges batch of 1D input tensors for vectorized wavefront algorithm.

    Args:
      tensor: A tf.Tensor<dtype>[batch, len2].
      len1: An integer corresponding to the length of y_true plus one.

    Returns:
      A single tf.Tensor<dtype>[len1 + len2 - 1, len1, batch] satisfying
        out[k][i][n] = t[n][k - i]
      if the RHS is well-defined, and 0 otherwise.
    """
    b, len2 = tf.shape(tensor)[0], tf.shape(tensor)[1]
    n_pad, padded_len = len1 - 1, len1 + len2 - 1

    ta = tf.TensorArray(tensor.dtype, size=len1, clear_after_read=True)
    for i in tf.range(len1):
      row_i = tf.pad(tensor, [[0, 0], [n_pad, n_pad]])
      row_i = tf.slice(row_i, [0, n_pad - i], [b, padded_len])
      ta = ta.write(i, row_i)  # row_i[b, padded_len]
    ta = ta.stack()  # ta[len1, b, padded_len]

    return tf.transpose(ta, (2, 0, 1))  # out[padded_len, len1, b]

  def alignment(self, subs_costs, ins_costs, del_cost, seq_lens, inf, dtype):
    """Computes the alignment score values.

    Args:
      subs_costs: A tf.Tensor<float>[batch, len_1, len_2] input matrix of
        substitution costs.
      ins_costs: A tf.Tensor<float>[batch, len_1] input vector of insertion
        costs.
      del_cost: A float, the cost of deletion.
      seq_lens: A tf.Tensor<int>[batch] input matrix of true sequence lengths.
      inf: A float with very high value.
      dtype: the data type of y_pred

    Returns:
      A tf.Tensor<float>[batch] of values of the alignment scores.
    """
    # Gathers shape variables.
    b, m = tf.shape(subs_costs)[0], tf.shape(subs_costs)[1]
    n = tf.shape(subs_costs)[2]  # We assume tf.shape(y_pred)[0] equals b.
    # Computes and rearranges cost tensors for vectorized wavefront iterations.
    subs_costs = self.wavefrontify(subs_costs)
    ins_costs = self.wavefrontify_vec(ins_costs, m + 1)

    # Sets up reduction operators.
    if self.loss_reg is None:
      minop = lambda t: tf.reduce_min(t, 0)
    else:
      loss_reg = tf.convert_to_tensor(self.loss_reg, dtype)
      minop = lambda t: -loss_reg * tf.reduce_logsumexp(-t / loss_reg, 0)

    # Initializes recursion.
    v_opt = tf.fill([b], inf)
    v_p2 = tf.pad(tf.fill([m - 1, b], inf), [[1, 0], [0, 0]])
    v_p1 = tf.concat([
        tf.slice(ins_costs[0], [0, 0], [1, b]),
        tf.fill([1, b], del_cost),
        tf.fill([m - 1, b], inf)
    ], 0)
    # Precomputes auxiliary (constant) tensors used during the recursion.
    i_range = tf.range(m + 1, dtype=tf.int32)
    k_end = seq_lens + n  # Indexes antidiagonal containing last entry, w/o pad.
    # Indexes last entries in "wavefrontified" slices, accounting for padding.
    nd_indices = tf.stack([seq_lens, tf.range(b, dtype=seq_lens.dtype)], -1)

    # Runs forward recursion.
    for k in tf.range(2, m + n + 1):
      # Masks invalid entries in "wavefrontified" value tensor.
      j_range = k - i_range
      inv_mask = tf.logical_and(j_range >= 0, j_range <= n)[:, tf.newaxis]

      o_m = v_p2 + subs_costs[k - 2]  # [m, b]
      o_i = v_p1 + ins_costs[k - 1]  # [m + 1, b]
      v_p2 = v_p1[:-1]
      o_d = v_p2 + del_cost  # [m, b]

      v_p1 = tf.concat(
          [tf.slice(o_i, [0, 0], [1, b]),
           minop(tf.stack([o_m, o_i[1:], o_d]))], 0)
      v_p1 = tf.where(inv_mask, v_p1, inf)
      v_opt = tf.where(k_end == k, tf.gather_nd(v_p1, nd_indices), v_opt)

    return v_opt

  def weave_band(self, input_v: tf.Tensor, inf: float):
    """Transforms a band around the diagonal of the matrix in a tall matrix.

    Args:
      input_v: A tf.Tensor<float>[batch, len, len] batch of square input
        matrices.
      inf: a very large float.

    Returns:
      A tf.Tensor<float>[batch, 2 * len - 1, 2 * width + 1] such that
      input_v[i, j] is returned in out[i + j, i - j + width]. With input matrix
       A B C D
       E F G H
       I J K L
       M N O P
      the function returns, for width=1
       0 A 0
       E 0 B
       0 F 0
       J 0 G
       0 K 0
       O 0 L
       0 P 0
    """
    batch = tf.shape(input_v)[0]
    len_v = tf.shape(input_v)[1]
    width = tf.cast(self.width, dtype=tf.int32)
    n_diag = 2 * width + 1
    diags = tf.linalg.diag_part(
        input_v, k=(-width, width), padding_value=inf, align='LEFT_LEFT')
    weave = tf.reshape(
        tf.stack([diags, tf.fill(tf.shape(diags), inf)], -1),
        [batch, n_diag, -1])
    woven_band_tr = inf * tf.ones((n_diag, batch, 2 * len_v))
    for diff in tf.range(-width, width + 1):
      i = diff + width
      abs_diff = tf.abs(diff)
      padded_weave = tf.roll(weave[:, n_diag - 1 - i], abs_diff, -1)
      woven_band_tr = tf.tensor_scatter_nd_update(woven_band_tr, [[i]],
                                                  padded_weave[tf.newaxis, ...])
    return tf.transpose(woven_band_tr, (1, 2, 0))[:, :-1, :]

  def index_ending_band(self, len_1, seq_lens):
    """Computes the indices at which to fetch the solution of the program."""
    batch = tf.shape(seq_lens)[0]
    i = seq_lens
    j = len_1 - tf.nn.relu(len_1 - seq_lens - self.width)
    sum_index = i + j
    diff_index = j - i + self.width
    range_batch = tf.range(batch)
    return tf.concat([
        range_batch[..., tf.newaxis], sum_index[..., tf.newaxis],
        diff_index[..., tf.newaxis]
    ],
                     axis=-1)

  def banded_alignment(self, subs_costs, ins_costs, del_cost, seq_lens, inf,
                       dtype):
    """Computes the alignment score values, with a band-restriction on the path.

    Args:
      subs_costs: A tf.Tensor<float>[batch, len_1, len_2] input matrix of
        substitution costs.
      ins_costs: A tf.Tensor<float>[batch, len_1] input vector of insertion
        costs.
      del_cost: A float, the cost of deletion.
      seq_lens: A tf.Tensor<int>[batch] input matrix of true sequence lengths.
      inf: A float with very high value.
      dtype: The data type of y_pred.

    Returns:
      A tf.Tensor<float>[batch] of values of the alignment scores.
    """
    batch = tf.shape(subs_costs)[0]
    len_1 = tf.shape(subs_costs)[1]
    len_2 = tf.shape(subs_costs)[2]
    val_trans = tf.zeros((len_1 + 1, len_2 + 1, batch))
    updates = [
        del_cost * tf.tile(
            tf.range(len_1 + 1, dtype=tf.float32)[..., tf.newaxis],
            multiples=[1, batch])
    ]
    val_trans = tf.tensor_scatter_nd_update(val_trans, [[0]], updates)
    for i in tf.range(1, len_1 + 1):
      previous_row = val_trans[i - 1, 0]
      val_trans = tf.tensor_scatter_nd_update(
          val_trans, [[i, 0]], [previous_row + ins_costs[:, i - 1]])
      values = tf.transpose(val_trans, [2, 1, 0])
    input_band = self.weave_band(values, inf)
    subs_band = self.weave_band(subs_costs, inf)
    ins_costs_pad = tf.pad(ins_costs, [[0, 0], [1, 0]], constant_values=0.)
    # TODO: uphere
    insert_expand = tf.tile(
        ins_costs_pad[:, tf.newaxis, :], multiples=[1, len_1 + 1, 1])
    insert_band = self.weave_band(insert_expand, inf)
    length = tf.shape(input_band)[1]
    # Sets up reduction operators.
    if self.loss_reg is None:
      minop = lambda t: tf.reduce_min(t, axis=-1)
    else:
      loss_reg = tf.convert_to_tensor(self.loss_reg, dtype)
      minop = lambda t: -loss_reg * tf.reduce_logsumexp(-t / loss_reg, axis=-1)
    for k in tf.range(2, length):
      input_minus_one = tf.pad(
          input_band[..., :-1], [[0, 0], [0, 0], [1, 0]], constant_values=inf)
      input_plus_one = tf.pad(
          input_band[..., 1:], [[0, 0], [0, 0], [0, 1]], constant_values=inf)
      min_tens = tf.stack([
          input_band[:, k - 2, :] + subs_band[:, k - 2, :],
          input_plus_one[:, k - 1, :] + del_cost,
          input_minus_one[:, k - 1, :] + insert_band[:, k, :],
      ],
                          axis=-1)
      insert_mins = minop(min_tens)
      input_trans = tf.tensor_scatter_nd_update(
          tf.transpose(input_band, [1, 0, 2]), [[k]], [insert_mins])
      input_band = tf.transpose(input_trans, [1, 0, 2])
    fetch_indices = self.index_ending_band(len_1, seq_lens)
    return tf.gather_nd(input_band, fetch_indices)

  def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes the alignment loss for a batch of sequences.

    Args:
      y_true: A tf.Tensor<[float, int]>[batch, m] representing the ground-truth
        sequences.
      y_pred: A tf.Tensor<float>[batch, n, n_tokens], (n >= m) representing the
        scores for predicted sequences.

    Returns:
      A tf.Tensor<float>[batch] with the value of the loss for each example.
    """
    # Gathers type variables.
    dtype = y_pred.dtype
    # Defines an appropriate large positive float to represent "infinity".
    # inf = tf.dtypes.float16.max if dtype == tf.dtypes.float16 else 1e9
    inf = tf.convert_to_tensor(1e9, dtype)  # TODO: float16 support?

    # Removes internal gaps, computes length excl. pad and converts to one-hot.
    y_true, seq_lens = self.preprocess_y_true(y_true)
    # Combines pad and gap tokens and ensures predicted scores add to be one.
    y_pred = self.preprocess_y_pred(y_pred)
    subs_costs = self.subs_cost_fn(y_true, y_pred)
    ins_costs = self.ins_cost_fn(y_pred)
    del_cost = tf.convert_to_tensor(self.del_cost, dtype)
    if self.width is None:
      return self.alignment(subs_costs, ins_costs, del_cost, seq_lens, inf,
                            dtype)
    else:
      return self.banded_alignment(subs_costs, ins_costs, del_cost, seq_lens,
                                   inf, dtype)
