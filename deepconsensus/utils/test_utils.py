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
"""Utilities to help with testing code."""

import os
from typing import List, Text, Tuple, Union

from absl import flags
from absl.testing import absltest
import numpy as np

from deepconsensus.utils import dc_constants
from tensorflow.python.platform import gfile

DEEPCONSENSUS_DATADIR = ''

# In the OSS version these will be ''.
DEFAULT_WORKSPACE = ''

FLAGS = flags.FLAGS


def test_tmpfile(name, contents=None):
  """Returns a path to a tempfile named name in the test_tmpdir.

  Args:
    name: str; the name of the file, should not contain any slashes.
    contents: bytes, or None. If not None, tmpfile's contents will be set to
      contents before returning the path.

  Returns:
    str path to a tmpfile with filename name in our test tmpfile directory.
  """
  path = os.path.join(absltest.get_default_test_tmpdir(), name)
  if contents is not None:
    with gfile.Open(path, 'wb') as fout:
      fout.write(contents)
  return path


def genomics_testdata(path, datadir):
  """Gets the path to a testdata file in genomics at relative path.

  Args:
    path: A path to a testdata file *relative* to the genomics root directory.
      For example, if you have a test file in "datadir/test/testdata/foo.txt",
      path should be "test/testdata/foo.txt" to get a path to it.
    datadir: The path of the genomics root directory *relative* to the testing
      source directory.

  Returns:
    The absolute path to a testdata file.
  """
  if hasattr(FLAGS, 'test_srcdir'):
    # Google code uses FLAG.test_srcdir
    # TensorFlow uses a routine googletest.test_src_dir_path.
    test_workspace = os.environ.get('TEST_WORKSPACE', DEFAULT_WORKSPACE)
    test_srcdir = os.path.join(FLAGS.test_srcdir, test_workspace)
  else:
    # In bazel TEST_SRCDIR points at the runfiles directory, and
    # TEST_WORKSPACE names the workspace.  We need to append to the
    # path the name of the workspace in order to get to the root of our
    # source tree.
    test_workspace = os.environ['TEST_WORKSPACE']
    test_srcdir = os.path.join(os.environ['TEST_SRCDIR'], test_workspace)
  return os.path.join(test_srcdir, datadir, path)


def deepconsensus_testdata(filename):
  """Gets the path to filename in testdata.

  These paths are only known at runtime, after flag parsing
  has occurred.

  Args:
    filename: The name of a testdata file in the core genomics testdata
      directory. For example, if you have a test file in
      "DEEPCONSENSUS_DATADIR/deepconsensus/testdata/foo.txt", filename should be
      "foo.txt" to get a path to it.

  Returns:
    The absolute path to a testdata file.
  """
  return genomics_testdata(
      os.path.join('deepconsensus/testdata', filename), DEEPCONSENSUS_DATADIR)


def get_one_hot(value: Union[int, np.ndarray]) -> np.ndarray:
  """Returns a one-hot vector for a given value."""
  return np.eye(len(dc_constants.VOCAB), dtype=dc_constants.NP_DATA_TYPE)[value]


def seq_to_array(seq: str) -> List[int]:
  return [dc_constants.VOCAB.index(i) for i in seq]


def multiseq_to_array(sequences: Union[Text, List[Text]]) -> np.ndarray:
  """Converts ATCG sequences to DC numeric format."""
  return np.array(list(map(seq_to_array, sequences)))


def seq_to_one_hot(sequences: Union[Text, List[Text]]) -> np.ndarray:
  """Converts ATCG to one-hot format."""
  result = []
  for seq in sequences:
    result.append(get_one_hot(multiseq_to_array(seq)))
  result = np.squeeze(result)
  return result.astype(dc_constants.NP_DATA_TYPE)


def convert_seqs(sequences: List[str]) -> Tuple[np.ndarray, np.ndarray]:
  """Creates label and associated y_pred tensor.

  Args:
    sequences: string array inputs for label and prediction

  Returns:
    y_true as array
    y_pred_scores as probability array

  """
  y_true, y_pred_scores = sequences
  y_true = multiseq_to_array(y_true).astype(dc_constants.NP_DATA_TYPE)
  y_pred_scores = seq_to_one_hot(y_pred_scores)
  return y_true, y_pred_scores
