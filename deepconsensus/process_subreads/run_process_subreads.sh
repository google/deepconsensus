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
# Example Usage:
#
# bash run_process_subreads.sh --shards=10 \
#   --subreads=gs://<bucket>pacbio/datasets/test_set/subreads/test_set.subreads.bam \
#   --truth_reference=gs://brain-genomics/deepconsensus/hg002_diploid_assembly/hg002_hifi_hicanu_combined.fasta
set -euo pipefail

source gbash.sh || exit 1
source module lib/colors.sh

DEFINE_string subreads --required "" "Subreads"
DEFINE_int shards --required "100" "Shards"
DEFINE_string truth_reference "" "Required for label generation. Reference genome to use for labels."
DEFINE_bool dry_run false "Print json used to submit jobs."

gbash::init_google "$@"

if ! gcertstatus --check_remaining=12h; then
  LOG FATAL "You need at least 12 hours of prod access; Run gcert."
fi

VERSION=v1
ALIGNMENT_IMAGE="gcr.io/google.com/brain-genomics/alignment:210910"

function gs_to_mnt {
  echo "/mnt/data/mount/${1/:\//}"
}

function to_label {
  echo "${1}" | tr "[:upper:]-." "[:lower:]_-"
}

function print_tsv {
  # Prints an array as a tsv line
  arr=("$@")
  echo -e "$( IFS=$'\t'; echo -e "${arr[*]}" )"
}

function check_gs {
  if ! gsutil -q stat "${1}"; then LOG FATAL "${bred}ERROR: File ${1} Not found${endcolor}"; fi;
}

function job_log {
  echo -e "\n$(colors::get_color bluebg bold black)JOB → ${endcolor} ${green}${1}${endcolor}\n"
}

SOURCE="$(dirname "${BASH_SOURCE[0]}")"
DSUB=dsub
DSUB_COMMON_ARGS=(
   --provider "google-v2"
   --project "<google project id>"
   --zone us-*
   --logging "gs://<bucket>logs"
   --ssh
   --machine-type n1-standard-32
   --boot-disk-size 30
)

if [[ ${FLAGS_dry_run} == 1 ]]; then
  DSUB_COMMON_ARGS+=( --dry-run )
fi;

#======================#
# run process subreads #
#======================#

check_gs "${FLAGS_subreads}"

n_total=$(printf "%05g" "${FLAGS_shards}")
dataset_path=$(dirname "$(dirname "${FLAGS_subreads}")")
dataset=$(basename "${dataset_path}")
subreads_fname=$(basename "${FLAGS_subreads}")
movie_name=$(echo "${subreads_fname}" | cut -f 1 -d '.')
output_dir=${dataset_path}/${VERSION}/${movie_name}

header=(
  --env=n
  --env=n_total
  --env=subreads_bam
  --label=subreads
  --output=ccs_shard_bam
  --output=ccs_shard_fasta
  --output=hifi_shard_bam
  --output=hifi_shard_fasta
  --output=subreads_shard_bam
  --output=subreads_to_ccs_shard_bam
)
# Construct tasks file
task_tsv=$(mktemp)
{
  print_tsv "${header[@]}" | sed 's/=/ /g';
  for n in $(seq -f "%05g" 1 "${FLAGS_shards}"); do
    # Output files
    ccs_shard_bam=${output_dir}/ccs_bam/${movie_name}.${n}_of_${n_total}.ccs.bam
    ccs_shard_fasta=${output_dir}/ccs_fasta/${movie_name}.${n}_of_${n_total}.ccs.fasta
    hifi_shard_bam=${output_dir}/hifi_bam/${movie_name}.${n}_of_${n_total}.hifi.bam
    hifi_shard_fasta=${output_dir}/hifi_fasta/${movie_name}.${n}_of_${n_total}.hifi.fasta
    subreads_shard_bam=${output_dir}/subreads/${movie_name}.${n}_of_${n_total}.subreads.bam
    subreads_to_ccs_shard_bam=${output_dir}/subreads_to_ccs/${movie_name}.${n}_of_${n_total}.subreads_to_ccs.bam
    vals=(
      "${n}"
      "${n_total}"
      $(gs_to_mnt "${FLAGS_subreads}")
      $(to_label "${subreads_fname}")
      "${ccs_shard_bam}"
      "${ccs_shard_fasta}"
      "${hifi_shard_bam}"
      "${hifi_shard_fasta}"
      "${subreads_shard_bam}"
      "${subreads_to_ccs_shard_bam}"
    )
    print_tsv "${vals[@]}"
  done;
} > "${task_tsv}"

job_log "process subreads"
PROCESS_SUBREADS_TASK=$(
  ${DSUB} \
  "${DSUB_COMMON_ARGS[@]}" \
  --mount BUCKET="gs://deepconsensus" \
  --label process_subreads \
  --image ${ALIGNMENT_IMAGE} \
  --script "${SOURCE}/tasks/process_subreads.sh" \
  --tasks "${task_tsv}" \
  --disk-size 100 \
  --skip
)

#===============#
# combine fasta #
#===============#

function combine_task {
  # Combines files. If fasta output an index is generated.
  input_pattern=${1}
  output=${2}
  after=${3}

  # Optionally generate fasta index
  add_output_fasta_index=""
  if [[ "${output: -5}" == "fasta" ]]; then
    add_output_fasta_index="--output output_fasta_index=${output}.fai"
  fi

  # shellcheck disable=SC2086
  ${DSUB} \
  "${DSUB_COMMON_ARGS[@]}" \
  --label combine \
  --input input_pattern="${input_pattern}" \
  --output output="${output}" \
  ${add_output_fasta_index} \
  --disk-type pd-ssd \
  --image ${ALIGNMENT_IMAGE} \
  --script "${SOURCE}/tasks/combine.sh" \
  --disk-size 500 \
  --after="${after}" \
  --skip
}

job_log "combine ccs fasta"
ccs_fasta_pattern="${output_dir}/ccs_fasta/*"
ccs_fasta_combined="${output_dir}/${movie_name}.ccs.fasta"
combine_task "${ccs_fasta_pattern}" "${ccs_fasta_combined}" "${PROCESS_SUBREADS_TASK}"

job_log "combine hifi fasta"
hifi_fasta_pattern="${output_dir}/hifi_fasta/*"
hifi_fasta_combined="${output_dir}/${movie_name}.hifi.fasta"
combine_task "${hifi_fasta_pattern}" "${hifi_fasta_combined}" "${PROCESS_SUBREADS_TASK}"

#=================#
# generate_labels #
#=================#

if [[ "${FLAGS_truth_reference}" == "" ]]; then
  LOG INFO "No further work to be done."
  exit 0
fi;

header=(
  --env=n
  --env=n_total
  --input=ccs_shard_bam
  --input=ccs_shard_fasta
  --input=truth_reference
  --output=truth_shard_bed
  --output=truth_shard_fasta
  --output=truth_to_ccs_shard_bam
)
# Construct tasks tsv for label generation.
label_task_tsv=$(mktemp)
{
  print_tsv "${header[@]}" | sed 's/=/ /g';
  for n in $(seq -f "%05g" 1 "${FLAGS_shards}"); do
    # Output files
    ccs_shard_bam=${output_dir}/ccs_bam/${movie_name}.${n}_of_${n_total}.ccs.bam
    ccs_shard_fasta=${output_dir}/ccs_fasta/${movie_name}.${n}_of_${n_total}.ccs.fasta
    truth_reference=${FLAGS_truth_reference}
    truth_shard_bed=${output_dir}/truth_bed/${movie_name}.${n}_of_${n_total}.truth.bed
    truth_shard_fasta=${output_dir}/truth_fasta/${movie_name}.${n}_of_${n_total}.truth.fasta
    truth_to_ccs_shard_bam=${output_dir}/truth_to_ccs/${movie_name}.${n}_of_${n_total}.truth_to_ccs.bam
    vals=(
      "${n}"
      "${n_total}"
      "${ccs_shard_bam}"
      "${ccs_shard_fasta}"
      "${truth_reference}"
      "${truth_shard_bed}"
      "${truth_shard_fasta}"
      "${truth_to_ccs_shard_bam}"
    )
    print_tsv "${vals[@]}"
  done;
} > "${label_task_tsv}"

job_log "generate labels"
GENERATE_LABELS_TASK=$(
${DSUB} \
  "${DSUB_COMMON_ARGS[@]}" \
  --label generate_labels \
  --image ${ALIGNMENT_IMAGE} \
  --script "${SOURCE}/tasks/generate_labels.sh" \
  --disk-size 100 \
  --tasks "${label_task_tsv}" \
  --after="${PROCESS_SUBREADS_TASK}" \
  --skip
)

job_log "combine truth fasta"
truth_fasta_pattern="${output_dir}/truth_fasta/*"
truth_fasta_combined="${output_dir}/${movie_name}.truth.fasta"
combine_task "${truth_fasta_pattern}" "${truth_fasta_combined}" "${GENERATE_LABELS_TASK}"

job_log "combine truth bed"
truth_bed_pattern="${output_dir}/truth_bed/*"
truth_bed_combined="${output_dir}/${movie_name}.truth.bed"
combine_task "${truth_bed_pattern}" "${truth_bed_combined}" "${GENERATE_LABELS_TASK}"
