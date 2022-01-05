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
"""Tests for preprocess."""

import json
import os

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import tensorflow as tf

from deepconsensus.preprocess import preprocess
from deepconsensus.preprocess import utils
from deepconsensus.utils.test_utils import deepconsensus_testdata
from absl import app


testdata = deepconsensus_testdata

FLAGS = flags.FLAGS


def load_summary(tmp_dir, path):
  summary_path = os.path.join(tmp_dir, path)
  return json.load(open(summary_path, 'r'))


def load_dataset(output, dataset):
  # Load inference, train, eval, or test tfrecord.gz files.
  tf_record = output.replace('@split', dataset)
  dataset = tf.data.TFRecordDataset(tf_record, compression_type='GZIP')
  examples = list(dataset.as_numpy_iterator())
  return examples


def get_unique_zmws(examples):
  zmws = []
  for example in examples:
    features = utils.tf_example_to_features_dict(example)
    zmws.append(int(features['name'].split('/')[1]))
  return len(set(zmws))


class PreprocessE2E(parameterized.TestCase):

  @parameterized.parameters([0, 2])
  def test_e2e_inference(self, n_cpus):
    """Tests preprocessing inference in both single and multiprocess mode."""
    n_zmws = 3
    FLAGS.subreads_to_ccs = testdata('human_1m/subreads_to_ccs.bam')
    FLAGS.ccs_fasta = testdata('human_1m/ccs.fasta')
    FLAGS.cpus = n_cpus
    FLAGS.limit = n_zmws
    tmp_dir = self.create_tempdir()
    output = os.path.join(tmp_dir, 'tf-@split.tfrecord.gz')
    FLAGS.output = output
    preprocess.main([])
    examples = load_dataset(output, 'inference')
    features = utils.tf_example_to_features_dict(examples[0], inference=True)

    # Check that window_pos incr. monotonically for each ZMW.
    last_pos = -1
    last_zmw = -1
    for example in examples:
      features = utils.tf_example_to_features_dict(example, inference=True)
      zmw = int(features['name'].split('/')[1])
      if zmw != last_zmw:
        last_zmw = zmw
        last_pos = -1
      window_pos = int(features['window_pos'])
      self.assertGreater(window_pos, last_pos)
      last_zmw = zmw
      last_pos = window_pos

    summary = load_summary(tmp_dir, 'tf-summary.inference.json')

    self.assertEqual(summary['n_zmw_pass'], n_zmws)
    self.assertLen(examples, summary['n_examples'])

  @parameterized.parameters([0, 2])
  def test_e2e_train(self, n_cpus):
    """Tests preprocessing training in both single and multiprocess mode."""
    n_zmws = 10
    FLAGS.subreads_to_ccs = testdata('human_1m/subreads_to_ccs.bam')
    FLAGS.ccs_fasta = testdata('human_1m/ccs.fasta')
    FLAGS.truth_to_ccs = testdata('human_1m/truth_to_ccs.bam')
    FLAGS.truth_bed = testdata('human_1m/truth.bed')
    FLAGS.truth_split = testdata('human_1m/truth_split.tsv')
    FLAGS.cpus = n_cpus
    FLAGS.limit = n_zmws
    tmp_dir = self.create_tempdir()
    output = os.path.join(tmp_dir, 'tf-@split.tfrecord.gz')
    FLAGS.output = output
    preprocess.main([])
    train_examples = load_dataset(output, 'train')
    eval_examples = load_dataset(output, 'eval')
    test_examples = load_dataset(output, 'test')
    all_examples = train_examples + eval_examples + test_examples

    # Check that window_pos incr. monotonically for each ZMW.
    last_pos = -1
    last_zmw = -1
    for example in all_examples:
      features = utils.tf_example_to_features_dict(example, inference=False)
      zmw = int(features['name'].split('/')[1])
      if zmw != last_zmw:
        last_zmw = zmw
        last_pos = -1
      window_pos = int(features['window_pos'])
      self.assertGreater(window_pos, last_pos)
      last_zmw = zmw
      last_pos = window_pos

    summary = load_summary(tmp_dir, 'tf-summary.training.json')

    # Total count
    self.assertLen(all_examples, summary['n_examples'])

    # Test ZMW counts match
    n_zmw_train = get_unique_zmws(train_examples)
    n_zmw_eval = get_unique_zmws(eval_examples)
    n_zmw_test = get_unique_zmws(test_examples)
    self.assertLessEqual(summary['n_zmw_pass'], n_zmws)
    self.assertEqual(n_zmw_train + n_zmw_eval + n_zmw_test,
                     summary['n_zmw_pass'])
    self.assertEqual(n_zmw_train, summary['n_zmw_train'])
    self.assertEqual(n_zmw_eval, summary['n_zmw_eval'])
    self.assertEqual(n_zmw_test, summary['n_zmw_test'])

    # Test n example counts match
    self.assertLen(train_examples, summary['n_examples_train'])
    self.assertLen(eval_examples, summary['n_examples_eval'])
    self.assertLen(test_examples, summary['n_examples_test'])

    features = utils.tf_example_to_features_dict(train_examples[0])
    self.assertIn('label', features)
    self.assertIn('label/shape', features)
    self.assertTrue(
        (features['subreads'].shape == features['subreads/shape']).all())


if __name__ == '__main__':
  absltest.main()
