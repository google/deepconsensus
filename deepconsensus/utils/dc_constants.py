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
"""Common constants shared across the DeepConsensus codebase."""
import enum

import numpy as np
import pysam
import tensorflow as tf

# DeepConsensus Version
__version__ = '1.2.0'

# Vocab
GAP = ' '
ALLOWED_BASES = 'ATCG'
SEQ_VOCAB = GAP + ALLOWED_BASES
SEQ_VOCAB_SIZE = len(SEQ_VOCAB)

# Value used to fill in empty rows in the tf.Examples.
GAP_INT = SEQ_VOCAB.index(GAP)

PYSAM_READ_ADVANCING_OPS = list(
    map(int, [pysam.CMATCH, pysam.CINS, pysam.CEQUAL, pysam.CDIFF])
)


class Issue(int, enum.Enum):
  TRUTH_ALIGNMENT_NOT_FOUND = 1
  SUPP_TRUTH_ALIGNMENT = 2


class Strand(int, enum.Enum):
  UNKNOWN = 0
  FORWARD = 1  # read.is_reverse == False
  REVERSE = 2  # read.is_reverse == True


CIGAR_OPS = {
    'M': pysam.CMATCH,
    'I': pysam.CINS,
    'D': pysam.CDEL,
    'N': pysam.CREF_SKIP,
    'S': pysam.CSOFT_CLIP,
    'H': pysam.CHARD_CLIP,
    'P': pysam.CPAD,
    '=': pysam.CEQUAL,
    'X': pysam.CDIFF,
    'B': pysam.CBACK,
}

# Defining this as ints makes comparison operations faster.
PYSAM_CMATCH = int(pysam.CMATCH)
PYSAM_CINS = int(pysam.CINS)
PYSAM_CDEL = int(pysam.CDEL)
PYSAM_CSOFT_CLIP = int(pysam.CSOFT_CLIP)
PYSAM_CHARD_CLIP = int(pysam.CHARD_CLIP)

# Dtypes
TF_DATA_TYPE = tf.float32
NP_DATA_TYPE = np.float32

# E. Coli eval region is first 10% of the genome,
# Test region is last 10% of the genome
# Total genome length is 4642522.
ECOLI_REGIONS = {
    'TRAIN': (464253, 4178270),
    'EVAL': (0, 464252),
    'TEST': (4178271, 4642522),
}
TRAIN_REGIONS = {
    'HUMAN': (
        [str(i) for i in range(1, 19)]
        + ['chr%d' % i for i in range(1, 19)]
        + ['X', 'Y', 'chrX', 'chrY']
    ),
    'MAIZE': [str(i) for i in range(1, 9)] + ['chr%d' % i for i in range(1, 9)],
}

EVAL_REGIONS = {
    'HUMAN': ['21', '22', 'chr21', 'chr22'],
    'MAIZE': ['9', 'chr9'],
}
TEST_REGIONS = {
    'HUMAN': ['19', '20', 'chr19', 'chr20'],
    'MAIZE': ['10', 'chr10'],
}

# List of features in DC examples.
DC_FEATURES = [
    'rows',
    'label',
    'num_passes',
    'window_pos',
    'name',
    'ccs_base_quality_scores',
    'ec',
    'np_num_passes',
    'rq',
    'rg',
]

EMPTY_QUAL = 0

# The name of the eval metric to optimize (choosing checkpoints and vizier).
MAIN_EVAL_METRIC_NAME = 'eval/per_example_accuracy'
