#!/bin/bash
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
# bash learning/genomics/deepconsensus/testdata/generate_testdata.sh

source gbash.sh || exit

# Input sample
MOVIE_NAME=m54238_180901_011437
SHARD="00000-of-00300"
INPUT_PATH=gs://<bucket>pacbio/datasets/human_1m/subreads/test_data/${MOVIE_NAME}
CHROM_SPLIT=gs://brain-genomics/gunjanbaid/deepconsensus/data/hg002_hifi_hicanu_combined.chrom_mapping.txt
N_SEQS=10

# Outputs
TEST_DIR=$(gbash::get_google3_dir)/learning/genomics/deepconsensus/testdata/human_1m
mkdir -p ${TEST_DIR}
mkdir -p ${TEST_DIR}/tf_examples
SUBREADS_TO_CCS=${TEST_DIR}/subreads_to_ccs.bam
CCS_FASTA=${TEST_DIR}/ccs.fasta
DATASET_SUMMARY=${TEST_DIR}/summary.training.json

TRUTH_TO_CCS=${TEST_DIR}/truth_to_ccs.bam
TRUTH_BED=${TEST_DIR}/truth.bed
TRUTH_SPLIT=${TEST_DIR}/truth_split.tsv

#=================#
# subreads_to_ccs #
#=================#

gsutil cat ${INPUT_PATH}/subreads_to_ccs/${MOVIE_NAME}-${SHARD}.subreads_to_ccs.bam | \
samtools view -h | \
awk -v n_seqs=${N_SEQS} 'BEGIN { split("", seqs) }
     $0 ~ "^@" && $0 !~ "^@SQ" { print }
     $0 ~ "^@SQ" {
                     if (sq_lines < n_seqs) {
                       seqs[length(seqs)+1] = gensub("SN:", "", "g", $2);
                       sq_lines = sq_lines + 1;
                       print
                     }
                 }
     $0 !~ "^@" {
                   for (k in seqs) {
                     if (seqs[k] == $3) {
                       print
                     }
                 }
     }' | \
     samtools view -bh > ${SUBREADS_TO_CCS}

samtools view ${TRUTH_TO_CCS} | cut -f 1 | cut -f 1,2 -d '/' > /tmp/truth_seqs.txt
samtools view ${SUBREADS_TO_CCS} | cut -f 1 | cut -f 1,2 -d '/' | uniq > /tmp/ccs_seqs.txt

#===========#
# ccs_fasta #
#===========#

gsutil cat ${INPUT_PATH}/ccs_fasta/${MOVIE_NAME}-${SHARD}.ccs.fasta | \
grep -A 1 -f /tmp/ccs_seqs.txt > "${CCS_FASTA}"
samtools faidx "${CCS_FASTA}"

#==============#
# truth_to_ccs #
#==============#

# For the truth dataset, fetch ZMWs up to the last ZMW contained in the subread bam
last_zmw=$(cut -d '/' -f 2 /tmp/ccs_seqs.txt | tail -n 1)
gsutil cat ${INPUT_PATH}/truth_to_ccs/${MOVIE_NAME}-${SHARD}.truth_to_ccs.bam | \
samtools view -h | \
awk -v last_zmw=${last_zmw} '$0 ~ "^@" && $0 !~ "^@SQ" { print }
                             $0 ~ "^@SQ" { split($2, a, "/"); if (a[2] <= last_zmw) { print } }
                             $0 !~ "^@SQ" { split($1, a, "/"); if (a[2] <= last_zmw) { print } }' | \
     samtools view -bh > ${TRUTH_TO_CCS}
samtools index ${TRUTH_TO_CCS}

#===========#
# truth_bed #
#===========#

gsutil cat ${INPUT_PATH}/truth_bed/${MOVIE_NAME}-${SHARD}.truth.bed | \
grep -f /tmp/ccs_seqs.txt > ${TRUTH_BED}

#=============#
# truth_split #
#=============#

cut -f 1 ${TRUTH_BED} > /tmp/truth_contigs.txt
gsutil cat ${CHROM_SPLIT} | \
grep -f /tmp/truth_contigs.txt > ${TRUTH_SPLIT}

#======================#
# generate tf examples #
#======================#

# Use cpus=0 to ensure deterministic output.

# Inference examples
blaze run -c opt //learning/genomics/deepconsensus/preprocess:preprocess -- \
  --subreads_to_ccs=${SUBREADS_TO_CCS} \
  --ccs_fasta=${CCS_FASTA} \
  --cpus=0 \
  --output="${TEST_DIR}/tf_examples/@split/@split.tfrecord.gz"


# Training examples
blaze run -c opt //learning/genomics/deepconsensus/preprocess:preprocess -- \
  --subreads_to_ccs=${SUBREADS_TO_CCS} \
  --ccs_fasta=${CCS_FASTA} \
  --truth_to_ccs=${TRUTH_TO_CCS} \
  --truth_bed=${TRUTH_BED} \
  --truth_split=${TRUTH_SPLIT} \
  --cpus=0 \
  --output="${TEST_DIR}/tf_examples/@split/@split.tfrecord.gz"
