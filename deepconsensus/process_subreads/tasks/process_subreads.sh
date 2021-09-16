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
set -euox

# Generate CCS Sequence
ccs --all \
    -j "$(nproc)" \
    --chunk="${n}"/"${n_total}" \
    "${subreads_bam}" \
    "${ccs_shard_bam}"
pbindex "${ccs_shard_bam}"

# Generate CCS Fasta
samtools fasta --threads "$(nproc)" "${ccs_shard_bam}" > "${ccs_shard_fasta}"

# TEMPORARY: Subreads sharding should not be required later.
pbindex "${ccs_shard_bam}"
pbindexdump "${ccs_shard_bam}.pbi" | jq '.reads[].holeNumber' > zmws.txt
zmwfilter --include zmws.txt "${subreads_bam}" "${subreads_shard_bam}"

# Extract HIFI
extracthifi "${ccs_shard_bam}" "${hifi_shard_bam}"

# Generate HIFI Fasta
samtools fasta --threads "$(nproc)" "${hifi_shard_bam}" > "${hifi_shard_fasta}"

# Generate subreads to bam alignment
actc -j "$(nproc)" \
     "${subreads_bam}" "${ccs_shard_bam}" "${subreads_to_ccs_shard_bam}"
