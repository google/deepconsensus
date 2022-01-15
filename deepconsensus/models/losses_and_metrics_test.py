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
"""Tests for losses_and_metrics."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from deepconsensus.models import losses_and_metrics
from deepconsensus.utils import dc_constants
from deepconsensus.utils import test_utils


class PerExampleAccuracyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='all padding',
          y_true=np.array([
              [dc_constants.GAP_OR_PAD_INT, dc_constants.GAP_OR_PAD_INT],
          ]),

          # Using one hot inputs to create a 'distribution'. The metric will
          # compute the prediction by taking the argmax of the distribution.
          y_pred_scores=np.array([
              [test_utils.get_one_hot(dc_constants.GAP_OR_PAD_INT)] * 2,
          ]),
          # All windows are correct.
          exp_accuracy=1.0,
      ),
      dict(
          testcase_name='Left shift testing',
          y_true=np.stack([
              test_utils.seq_to_array('A T C G'),
              test_utils.seq_to_array('T T T T'),
              test_utils.seq_to_array('A A A A'),
          ]),

          # Using one hot inputs to create a 'distribution'. The metric will
          # compute the prediction by taking the argmax of the distribution.
          y_pred_scores=np.stack([
              test_utils.seq_to_one_hot('   ATCG'),
              test_utils.seq_to_one_hot('   GGGG'),
              test_utils.seq_to_one_hot('   AAAA'),
          ]),
          # Of the 3 examples, 1 and 3 are fully correct.
          exp_accuracy=0.6666667,
      ),
  )
  def test_accuracy(self, y_true, y_pred_scores, exp_accuracy):
    """Checks that accuracy is correct."""
    accuracy_obj = losses_and_metrics.PerExampleAccuracy()
    accuracy_obj.update_state(y_true, y_pred_scores)
    self.assertAlmostEqual(accuracy_obj.result().numpy(), exp_accuracy)

  def test_accuracy_multiple_updates(self):
    """Checks that accuracy is correct with multiple updates."""

    accuracy_obj = losses_and_metrics.PerExampleAccuracy()

    y_true = np.array([
        test_utils.seq_to_array('A T C G'),
        test_utils.seq_to_array('A T C G'),
        test_utils.seq_to_array('A T C G')
    ])
    y_pred_scores = np.array([
        test_utils.seq_to_one_hot('   ATCG'),
        test_utils.seq_to_one_hot('ATCG   '),
        test_utils.seq_to_one_hot('  ATCG ')
    ])

    # Update 1 is all correct
    accuracy_obj.update_state(y_true, y_pred_scores)
    self.assertEqual(accuracy_obj.result().numpy(), 1.0)

    y_true = np.array([
        test_utils.seq_to_array('C C C C'),
        test_utils.seq_to_array('A T C G'),
        test_utils.seq_to_array('C C C C')
    ])
    y_pred_scores = np.array([
        test_utils.seq_to_one_hot('   ATCG'),
        test_utils.seq_to_one_hot('ATCG   '),
        test_utils.seq_to_one_hot('  CCCC ')
    ])

    # Update 2 has 1 errors
    accuracy_obj.update_state(y_true, y_pred_scores)
    self.assertAlmostEqual(accuracy_obj.result().numpy(), 0.833333333)


class PerClassAccuracyTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='all correct',
          y_true=np.array([[0, 1, 0, 0]]),
          y_pred_scores=np.array([[
              test_utils.get_one_hot(0),
              test_utils.get_one_hot(1),
              test_utils.get_one_hot(0),
              test_utils.get_one_hot(0)
          ]]),
          class_value=1,
          exp_accuracy=1 / 1,
      ),
      dict(
          testcase_name='all positions correct for given class value',
          y_true=np.array([[0, 1, 0, 0]]),
          y_pred_scores=np.array([[
              test_utils.get_one_hot(0),
              test_utils.get_one_hot(1),
              test_utils.get_one_hot(1),
              test_utils.get_one_hot(1)
          ]]),
          class_value=1,
          exp_accuracy=1.0,
      ),
      dict(
          testcase_name='some positions incorrect for given class value',
          y_true=np.array([[0, 1, 1, 1]]),
          y_pred_scores=np.array([[
              test_utils.get_one_hot(0),
              test_utils.get_one_hot(1),
              test_utils.get_one_hot(0),
              test_utils.get_one_hot(0)
          ]]),
          class_value=1,
          exp_accuracy=1 / 3,
      ),
      dict(
          testcase_name='given class value not present',
          y_true=np.array([[0, 1, 1, 1]]),
          y_pred_scores=np.array([[
              test_utils.get_one_hot(0),
              test_utils.get_one_hot(1),
              test_utils.get_one_hot(0),
              test_utils.get_one_hot(0)
          ]]),
          class_value=4,
          # Metric is initialized as 0.
          exp_accuracy=0.0,
      ),
  )
  def test_accuracy(self, y_true, y_pred_scores, class_value, exp_accuracy):
    """Checks that per-class accuracy is correct."""
    accuracy_obj = losses_and_metrics.PerClassAccuracy(class_value=class_value)
    accuracy_obj.update_state(y_true, y_pred_scores)
    self.assertAlmostEqual(accuracy_obj.result().numpy(), exp_accuracy)


class LeftShiftTrueLabels(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Convert internal gaps',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['T T A G GC',
                                                    'A   G CTGG'])),
      dict(
          testcase_name='Do not convert internal gaps',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['T T A G GC',
                                                    'A   G CTGG'])),
  )
  def test_left_shift_sequence(self, sequences):
    """Checks that edit distance calculation matches expected value."""
    y_true, y_true_gapped = sequences
    y_true = test_utils.multiseq_to_array(y_true)
    y_true_gapped = test_utils.multiseq_to_array(y_true_gapped)

    y_true_ungapped = losses_and_metrics.left_shift_sequence(y_true_gapped)
    self.assertTrue(bool(tf.reduce_all(y_true == y_true_ungapped)))


class XentropySubsCostFn(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Equal lengths',
          b=2,
          m=4,
          n=4,
          seed=0,
          dtype=tf.float32,
      ),
      dict(
          testcase_name='Unequal lengths',
          b=2,
          m=4,
          n=6,
          seed=0,
          dtype=tf.float32,
      ),
  )
  def test_xentropy_subs_cost_fn(self, b, m, n, seed, dtype):
    """Checks that pointwise XEntropy values agree with tf.keras.losses."""
    # Generates random data.
    n_tokens = len(dc_constants.VOCAB)
    n_base_tokens = len(dc_constants.ALLOWED_BASES)

    y_true = tf.argmax(
        tf.random.stateless_normal([b, m, n_base_tokens], [seed, 0]), -1)
    y_true_oh = tf.one_hot(y_true, n_tokens, dtype=dtype)

    y_pred = tf.random.stateless_uniform([b, n, n_tokens], [seed, 1],
                                         dtype=dtype)
    y_pred = y_pred / tf.reduce_sum(y_pred, -1, True)

    xent = losses_and_metrics.xentropy_subs_cost_fn(y_true_oh, y_pred)
    # Compares with tf.losses.sparse_categorical_crossentropy as reference.
    for i in range(m):
      for j in range(n):
        y_true_i, y_pred_j = y_true[:, i], y_pred[:, j]
        xent_ij = tf.losses.sparse_categorical_crossentropy(y_true_i, y_pred_j)
        self.assertTrue(np.allclose(xent[:, i, j], xent_ij))


class XentropyInsCostFn(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Base case',
          b=4,
          n=8,
          seed=0,
          dtype=tf.float32,
      ),)
  def test_xentropy_subs_cost_fn(self, b, n, seed, dtype):
    """Checks that pointwise XEntropy values agree with tf.keras.losses."""
    # Generates random data.
    gap_token = dc_constants.VOCAB.find(dc_constants.GAP_OR_PAD)
    n_tokens = len(dc_constants.VOCAB)

    y_pred = tf.random.stateless_uniform([b, n, n_tokens], [seed, 0],
                                         dtype=dtype)
    y_pred = y_pred / tf.reduce_sum(y_pred, -1, True)

    xent = losses_and_metrics.xentropy_ins_cost_fn(y_pred)
    # Compares with tf.losses.sparse_categorical_crossentropy as reference.
    y_true = gap_token * tf.ones([b, n], dtype=tf.int32)
    xent_keras = tf.losses.sparse_categorical_crossentropy(y_true, y_pred)
    self.assertTrue(np.allclose(xent, xent_keras))


class AlignmentLossTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Hard, identical sequences, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGGC', 'AGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, identical sequences, with same pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTAGGC    ',
                                                    'AGCTGG    ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, identical sequences, with different pad',
          sequences=(['TTAGGCAT', 'AGCTGG  '], ['TTAGGCAT  ', 'AGCTGG    ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, correct insertions only, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['T TA G G C', 'AGC    TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, correct insertions only, with pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTA G GC  ',
                                                    'AGC    TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion at cost one, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGG ', 'GCTGG ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=1.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion at cost two, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TAGGC ', 'AGCGG ']),
          del_cost=2.0,
          loss_reg=None,
          expected_loss=2.0,
          width=None),
      dict(
          testcase_name='Hard, two deletions at cost one, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAG  ', 'GCGG  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=2.0,
          width=None),
      dict(
          testcase_name='Hard, one error, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['ATAGGC', 'TGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=16.118,  # log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, two errors, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['AAAGGC', 'TGCTGC']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=32.236,  # 2*log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, one erroneous insertion, no pad',
          sequences=(['TTAGGC', 'ATCGAC',
                      'AGCTGG'], ['TTAGGCA', 'ATCCGAC', 'CAGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=16.118,  # log(eps), with eps = 1e-7
          width=None),
      dict(
          testcase_name='Hard, one deletion, small deletion cost, with pad',
          sequences=(['ATCG ', 'ATCG '], ['TCG  ', 'TCG  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=1.0,
          width=None),
      dict(
          testcase_name='Hard, one deletion, large deletion cost, with pad',
          sequences=(['ATCG ', 'ATCG '], ['TCG  ', 'TCG  ']),
          del_cost=1e9,
          loss_reg=None,
          expected_loss=64.472,  # 4*log(eps), with eps = 1e-7
          width=None),
      # TODO: included test cases for soft alignment.
      dict(
          testcase_name='with band, identical sequences',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGGC', 'AGCTGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=2),
      dict(
          testcase_name='with band, one deletion at cost one, with pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['TTAGG ', 'GCTGG ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=1.0,
          width=2),
      dict(
          testcase_name='with band, identical sequences, with same pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTAGGC    ',
                                                    'AGCTGG    ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=1),
      dict(
          testcase_name='with band, correct insertions only, no pad',
          sequences=(['TTAGGC   ', 'AGCTG   G'], ['T TAG G C', 'AGC   TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=8),
      dict(
          testcase_name='with band, correct insertions only, with pad',
          sequences=(['TTAGGC    ', 'AGCTGG    '], ['TTA G GC  ',
                                                    'AGC    TGG']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=0.0,
          width=8),
      dict(
          testcase_name='with band, two errors, no pad',
          sequences=(['TTAGGC', 'AGCTGG'], ['AAAGGC', 'TGCTGC']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=32.236,  # 2*log(eps), with eps = 1e-7
          width=4),
      dict(
          testcase_name='with band of 2, two dels, one align, two pads',
          sequences=(['TTA', 'GGC'], ['A  ', 'C  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=2.0,
          width=2),
      dict(
          testcase_name='with band of 1,one del, one align, two pads, one del',
          sequences=(['TTA', 'GGC'], ['A  ', 'C  ']),
          del_cost=1.0,
          loss_reg=None,
          expected_loss=18.118,  # 2.0 + log(eps), with eps = 1e-7
          width=1),
  )
  def test_alignment_loss(self, sequences, del_cost, loss_reg, width,
                          expected_loss):
    """Checks that edit distance calculation matches expected value."""
    y_true, y_pred_scores = test_utils.convert_seqs(sequences)
    loss_obj = losses_and_metrics.AlignmentLoss(
        del_cost=del_cost, loss_reg=loss_reg, width=width)
    loss = loss_obj(y_true, y_pred_scores)
    self.assertAlmostEqual(float(loss), expected_loss, places=2)


if __name__ == '__main__':
  absltest.main()
