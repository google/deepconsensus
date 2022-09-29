# Generate Training Examples

This tutorial explains how to generate DeepConsensus training examples from
PacBio subreads and their CCS sequence. In this tutorial we will use a small
test dataset to demonstrate how training examples are generated. We will shard
the input and then generate examples for just one shard. These steps can be used
as a reference to implement a highly parallelized pipeline.

## Generate a CCS Sequence and subreads to ccs bam alignment

Follow the steps in [Quick Start](quick_start.md) up to and including
[Run actc](quick_start.md#run-actc). The following files should be generated:

```
${shard_id}.ccs.bam  # CCS sequence (one shard)
${shard_id}.subreads_to_ccs.bam # Subreads to CCS alignment (one shard)
```

Note: the shard_id variable is defined in [Quick Start](quick_start.md).

## Download truth files

```bash
# Create a work directory and place to store input and output files.
QS_DIR="${HOME}/deepconsensus_training"
mkdir -p "${QS_DIR}"
```

```bash
# Truth Reference
gsutil cp gs://deepconsensus/pacbio/datasets/chm13/chm13v2.0_noY.fa "${QS_DIR}"/
# Truth exclude BED
gsutil cp gs://deepconsensus/pacbio/datasets/chm13/chm13v2.0_noY_hifi.issues.bed "${QS_DIR}"/
# Truth split
gsutil cp gs://deepconsensus/pacbio/datasets/chm13/chm13v2.0_noY.chrom_mapping.txt "${QS_DIR}"/
```

## Generate labels

For convenience the following commands can be run inside DeepConsensus docker
container that contains all necessary tools and utilities. See
[Quick Start](quick_start.md#running-the-docker-image) for more details.

```bash
# Input:
export truth_reference="${QS_DIR}/chm13v2.0_noY.fa"
export ccs_shard_bam="${QS_DIR}/${shard_id}.ccs.bam"
export truth_split=${QS_DIR}/chm13v2.0_noY.chrom_mapping.txt
export subreads_to_ccs_shard_bam="${QS_DIR}/${shard_id}.subreads_to_ccs.bam"
# Output
TF_EXAMPLES_DIR="${QS_DIR}/tf_examples"
mkdir "${TF_EXAMPLES_DIR}"
mkdir "${TF_EXAMPLES_DIR}/train"
mkdir "${TF_EXAMPLES_DIR}/eval"
mkdir "${TF_EXAMPLES_DIR}/test"
export ccs_shard_to_truth_alignment_unfiltered="${QS_DIR}/${shard_id}.ccs_to_truth_ref.unfiltered.bam"
export  ccs_shard_to_truth_alignment_filtered="${QS_DIR}/${shard_id}.ccs_to_truth_ref.filtered.bam"
export truth_shard_bed="${QS_DIR}/${shard_id}.truth.bed"
export truth_shard_fasta="${QS_DIR}/${shard_id}.truth.fasta
export truth_to_ccs_shard_bam="${QS_DIR}/${shard_id}.truth_to_ccs.bam
tf_example_fname_output="tf-@split-${shard_id}.tfrecord.gz"

train_tf_record=${TF_EXAMPLES_DIR}/train/${tf_example_fname_output/@split/train}
eval_tf_record=${TF_EXAMPLES_DIR}/eval/${tf_example_fname_output/@split/eval}
test_tf_record=${TF_EXAMPLES_DIR}/test/${tf_example_fname_output/@split/test}

```

Note, that shard_id is defined in [Quick Start](quick_start.md)

### Align ccs to diploid truth reference

```bash
pbmm2 align --preset HIFI \
            --sort \
            -c 0 \
            -y 70 \
            "${truth_reference}" \
            "${ccs_shard_bam}" \
            "${ccs_shard_to_truth_alignment_unfiltered}"

samtools view -h -F 0x900 "${ccs_shard_to_truth_alignment_filtered}" | \
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
```

### Align truth to ccs

Still within the docker container, create and run a shell script from the
following code.

```bash
#!/bin/bash
set -euox pipefail

function align_truth_to_ccs_zmw {
  # Align a single truth sequence to its respective ccs sequence based on ZMW.
  ccs_seq=$(mktemp --tmpdir=/dev/shm --suffix .fa)
  truth_seq=$(mktemp --tmpdir=/dev/shm --suffix .fa)
  out_bam=$(mktemp --tmpdir=/dev/shm --suffix .bam)

  cat - > "${ccs_seq}"
  truth_seq_name=$(head -n 1 "${ccs_seq}" | sed 's/>//g' | sed 's#/ccs#/truth#g')
  samtools faidx "${truth_shard_fasta}" "${truth_seq_name}" > "${truth_seq}"

  pbmm2 align --preset HIFI \
              --log-level FATAL \
              --num-threads 1 \
              -c 0 \
              -y 70 \
              "${ccs_seq}" \
              "${truth_seq}" \
              "${out_bam}"

  samtools view "${out_bam}"
  # Cleanup
  rm "${ccs_seq}" "${out_bam}" "${truth_seq}"
}

export truth_shard_fasta
export -f align_truth_to_ccs_zmw

# Align truth to ccs
{
  # Construct header
  samtools view -H "${ccs_shard_to_truth_alignment_filtered}" | grep -v '@SQ';
  samtools view "${ccs_shard_to_truth_alignment_filtered}" | awk '{ print "@SQ\tSN:" $1 "\tLN:" length($10) }' | sort | uniq;
  # Convert bam to fasta, and use parallel to process.
  # parallel --pipe -N2 splits the fasta into individual sequences that are
  # processed using align_truth_to_ccs_zmw.
  samtools fasta "${ccs_shard_to_truth_alignment_filtered}" | \
  parallel -j "$(nproc)" \
           --linebuffer \
           --keep-order \
           --pipe \
           -N2 \
           align_truth_to_ccs_zmw;
} | samtools view -bh - > unsorted.bam

samtools sort -O BAM --threads "$(nproc)" unsorted.bam > "${truth_to_ccs_shard_bam}"
samtools index "${truth_to_ccs_shard_bam}"
samtools quickcheck "${truth_to_ccs_shard_bam}"
```

### Filter for reads that do not align in flagged regions

This step filters reads aligned to poor quality regions that are likely to hurt
training.

```bash
samtools view -@ "$(nproc)" -F 0x904 -b "${ccs_shard_to_truth_alignment_unfiltered}" > "${ccs_shard_to_truth_alignment_filtered}"
samtools index -@"$(nproc)" "${ccs_shard_to_truth_alignment_filtered}"
```

## Generate TensorFlow examples

Finally, we can generate training examples using the `deepconsensus preprocess`
command:

```bash
deepconsensus preprocess \
      --subreads_to_ccs="${subreads_to_ccs_shard_bam}" \
      --ccs_bam="${ccs_shard_bam}" \
      --truth_bed="${truth_shard_bed}" \
      --truth_to_ccs="${truth_to_ccs_shard_bam}" \
      --truth_split="${truth_split}" \
      --output="${tf_example_fname_output}" \
      --cpus="$(nproc)"

mv "${tf_example_fname_output/@split/train}" "${train_tf_record}"
mv "${tf_example_fname_output/@split/eval}" "${eval_tf_record}"
mv "${tf_example_fname_output/@split/test}" "${test_tf_record}"
```

We generated TensorFlow examples!

```bash
${train_tf_record} - Train examples
${eval_tf_record} - Evaluation examples
${test_tf_record} - Test examples
```
