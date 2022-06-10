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
"""Utilities for DeepConsensus."""

from typing import List, Union
import numpy as np
import tensorflow as tf
from deepconsensus.utils import dc_constants


def encoded_sequence_to_string(encoded_sequence: np.ndarray) -> str:
  encoded_sequence = encoded_sequence.astype(int)
  return ''.join(np.vectorize(dc_constants.VOCAB.__getitem__)(encoded_sequence))


def quality_score_to_string(score: int) -> str:
  """Returns the string representation for the given quality score.

  We add 33 to the score because this is how the quality score encoding is
  defined. Source:
  https://support.illumina.com/help/BaseSpace_OLH_009008/Content/Source/Informatics/BS/QualityScoreEncoding_swBS.htm

  Args:
    score: The raw quality score value.

  Returns:
    Symbol for the input quality score.
  """
  ascii_code = score + 33
  return chr(ascii_code)


def quality_scores_to_string(scores: np.ndarray) -> str:
  """Returns the string representation for the given list of quality scores."""
  return ''.join([chr(score) for score in (scores + 33)])


def quality_string_to_array(quality_string: str) -> List[int]:
  """Returns the int array representation for the given quality string."""
  return [ord(char) - 33 for char in quality_string]


def tf_avg_phred(base_qualities: tf.Tensor) -> tf.float32:
  """Calculate the avg phred using tensorflow."""

  def un_phred(val):
    return tf.pow(10.0, (val / -10.0))

  base_qualities = tf.cast(base_qualities, dc_constants.TF_DATA_TYPE)
  base_qualities = base_qualities[base_qualities >= 0]
  if not tf.reduce_any(tf.greater(base_qualities, 0)):
    return 0.0
  else:
    probs = tf.map_fn(un_phred, base_qualities)
    probs_len = tf.cast(tf.shape(probs), dc_constants.TF_DATA_TYPE)
    avg_prob = tf.reduce_sum(probs) / probs_len
    avg_q = -10.0 * (tf.math.log(avg_prob) / tf.math.log(10.0))
    return float(avg_q)


def avg_phred(base_qualities: Union[np.ndarray, List[int]]) -> float:
  """Get the average phred quality given base qualities of a read.

  Args:
     base_qualities: A numpy array containing the base qualities of a read.

  Returns:
     The average error rate of the read quality list.
  """
  # Filter out base qualities that are set to -1
  # These are used to encode spacing.
  base_qualities = np.asarray(base_qualities)
  base_qualities = base_qualities[base_qualities >= 0]
  if not base_qualities.any():
    return 0.0
  probs = 10**(base_qualities / -10.)
  avg_prob = probs.sum() / len(probs)
  avg_q = -10 * np.log10(avg_prob)
  return avg_q


def left_shift_seq(seq: np.ndarray) -> np.ndarray:
  """Left shift a numeric-encoded sequence."""
  return np.concatenate([
      seq[seq != dc_constants.GAP_OR_PAD_INT],
      seq[seq == dc_constants.GAP_OR_PAD_INT]
  ])


def left_shift(batch_seq: np.ndarray, axis: int = 1) -> np.ndarray:
  """Performs left_shift on a batch of sequences."""
  return np.apply_along_axis(left_shift_seq, axis, batch_seq)
