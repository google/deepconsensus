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
set -euox pipefail


# Store hg002 index
pbmm2 align --preset HIFI \
            --sort \
            -c 0 \
            -y 70 \
            "${truth_reference}" \
            "${ccs_shard_bam}" \
            ccs_truth_ref.bam

samtools view -h -F 0x900 ccs_truth_ref.bam | \
  awk 'BEGIN{OFS="\t";} { if($1 !~ /@/) { $1=$1":"length($10); } print; }' | \
  samtools view -b | bedtools bamtobed -i - | \
  awk '{ OFS="\t"; split($4,A,":"); $4=A[1]"\t"A[2]; print; }' | \
  awk '{ READLEN=$5; D=(READLEN-($3-$2)); if(D<0){D=-D;} ED=(D+$NF)/READLEN; { print $1 "\t" $2 "\t" $3 "\t" $4 "\t" ED "\t" $7; } }' \
  > "${truth_shard_bed}"

for STRAND in "+" "-"; do
  if [[ "${STRAND}" == "-" ]]; then
    RC="-r"
  else
    RC=""
  fi
  echo "RC=${RC}"
  paste <(awk -v STRAND="${STRAND}" '($6==STRAND) { print ">" $4; }' "${truth_shard_bed}" | sed -e 's%/ccs%/truth%') \
    <(awk -v STRAND="${STRAND}" '($6==STRAND) { print  $1 ":" 1+$2 "-" $3; }' "${truth_shard_bed}" | \
    xargs samtools faidx "${truth_reference}" | seqtk seq ${RC} -l 0 /dev/stdin | grep -v '>') | \
    awk '{ print $1; print $2; }' >> unfiltered.truth_shard_fasta
done

awk '(NR%2==1) { n=$0; } (NR%2==0) { split(n,A,"/"); print A[2] "\t" n "\t" $0; }' unfiltered.truth_shard_fasta | \
  sort -k1,1g | \
  awk '{ print $2; print $3; }' > "${truth_shard_fasta}"

samtools faidx "${truth_shard_fasta}"


# Align truth to CCS.
minimap2 -t "$(nproc)" \
         -a \
         --secondary=no \
         --split-prefix=/tmp/tmp.fasta \
        "${ccs_shard_fasta}" "${truth_shard_fasta}" > unfiltered.bam


# Keep only alignments where the truth and CCS reference molecules match.
samtools view -h unfiltered.bam \
  | awk '{ if($1 ~ /^@/) { print; } else { split($1,A,"/"); split($3,B,"/"); if(A[2]==B[2]) { split(A[3],C,"_"); print $0 "\tqs:i:" C[1]; } } }' \
  | samtools sort \
  | samtools view -h -o "${truth_to_ccs_shard_bam}"

# samtools index "${truth_to_ccs_shard_bam}"
