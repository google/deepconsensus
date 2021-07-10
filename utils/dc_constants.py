# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Common constants shared across the DeepConsensus codebase."""

import numpy as np
import tensorflow as tf

from nucleus.protos import cigar_pb2
from nucleus.util import cigar as cigar_utils

# Do not include soft clips.
OPS_TO_CONSIDER = frozenset([
    cigar_pb2.CigarUnit.ALIGNMENT_MATCH, cigar_pb2.CigarUnit.SEQUENCE_MATCH,
    cigar_pb2.CigarUnit.INSERT, cigar_pb2.CigarUnit.DELETE,
    cigar_pb2.CigarUnit.SEQUENCE_MISMATCH
])

READ_ADVANCING_OPS = frozenset([
    cigar_pb2.CigarUnit.ALIGNMENT_MATCH, cigar_pb2.CigarUnit.SEQUENCE_MATCH,
    cigar_pb2.CigarUnit.INSERT, cigar_pb2.CigarUnit.SEQUENCE_MISMATCH
])

OP_CHARS_TO_CONSIDER = frozenset(
    [cigar_utils.CIGAR_OPS_TO_CHAR[op] for op in OPS_TO_CONSIDER])

GAP_OR_PAD = ' '
ALLOWED_BASES = 'ATCG'
VOCAB = GAP_OR_PAD + ALLOWED_BASES

GAP_OR_PAD = ' '

GAP_OR_PAD_INT = VOCAB.index(GAP_OR_PAD)

# Value used to fill in empty rows in the tf.Examples.
GAP_OR_PAD_INT = VOCAB.index(GAP_OR_PAD)

TF_DATA_TYPE = tf.float32
NP_DATA_TYPE = np.float32

PW_MAX = 9
IP_MAX = 9
SN_MAX = 15
STRAND_MAX = 2

# E. Coli eval region is first 10% of the genome,
# Test region is last 10% of the genome
# Total genome length is 4642522.
ECOLI_REGIONS = {
    'TRAIN': (464253, 4178270),
    'EVAL': (0, 464252),
    'TEST': (4178271, 4642522)
}
# chrs 1-18, X, and Y. All with and without 'chr'.
HUMAN_TRAIN_REGIONS = [str(i) for i in range(1, 19)] + [
    'chr%d' % i for i in range(1, 19)
] + ['X', 'Y', 'chrX', 'chrY']
# chrs 21 and 22, both with and without 'chr'.
HUMAN_EVAL_REGIONS = ['21', '22', 'chr21', 'chr22']
HUMAN_TEST_REGIONS = ['19', '20', 'chr19', 'chr20']

MAX_QUAL = 60
EMPTY_QUAL = 0
