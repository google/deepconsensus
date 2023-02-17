# Generate Training Examples

This tutorial explains how to generate DeepConsensus training examples from
PacBio subreads and their CCS sequence. In this tutorial we will use a small
test dataset to demonstrate how training examples are generated. We will shard
the input and then generate examples for just one shard. These steps can be used
as a reference to implement a highly parallelized pipeline.

## Setting up a VM

DeepConsensus can be run on Unix systems. The command below can be used to spin
up a compatible virtual machine (VM) on Google Cloud Platform (GCP). This
command will spin up a
[n1-standard-16 machine on GCP](https://cloud.google.com/compute/docs/general-purpose-machines#n1_machines).

```bash
VM="${USER}-deepconsensus-model-train"
gcloud compute instances create "${VM}" \
  --scopes "compute-rw,storage-full,cloud-platform" \
  --maintenance-policy "TERMINATE" \
  --image-family "ubuntu-2004-lts" \
  --image-project "ubuntu-os-cloud" \
  --machine-type "n1-standard-16" \
  --boot-disk-size "200" \
  --zone "us-west1-b" \
  --min-cpu-platform "Intel Skylake"
```

This instance will have the following configuration:

```bash
OS: Ubuntu 20.04.3 LTS (x86_64)
Python version: Python 3.8.10
CPUs: 16vCPUs (x86_64, GenuineIntel, SkyLake)
Memory: 60G
```

You can log into the new VM using `gcloud`. Please note, that it may take couple
minutes for the VM to initialize before you can ssh there.

```bash
gcloud compute ssh "${VM}" --zone=us-west1-b
```

See the [runtime metrics page](runtime_metrics.md) for an overview of runtimes
using different GCP compute VM configurations.

## Download test data

```bash
# Create a work directory and place to store our model for the quick start.
BASE_DIR="${HOME}/deepconsensus_model_train"
mkdir -p "${BASE_DIR}"
TF_EXAMPLES_DIR="${BASE_DIR}/tf_examples"
mkdir -p "${TF_EXAMPLES_DIR}"
mkdir "${TF_EXAMPLES_DIR}/train"
mkdir "${TF_EXAMPLES_DIR}/eval"
mkdir "${TF_EXAMPLES_DIR}/test"

# Download the input PacBio Subread data.
gsutil cp gs://brain-genomics-public/research/deepconsensus/quickstart/v1.2/n1000.subreads.bam "${BASE_DIR}"/

# Truth Reference
gsutil cp gs://deepconsensus/pacbio/datasets/chm13/chm13v2.0_noY.fa "${BASE_DIR}"/
# Truth exclude BED
gsutil cp gs://deepconsensus/pacbio/datasets/chm13/chm13v2.0_noY_hifi.issues.bed "${BASE_DIR}"/
# Truth split
gsutil cp gs://deepconsensus/pacbio/datasets/chm13/chm13v2.0_noY.chrom_mapping.txt "${BASE_DIR}"/

cd "${BASE_DIR}"
```

This directory should now contain the following files:

```
n1000.subreads.bam
chm13v2.0_noY.fa
chm13v2.0_noY_hifi.issues.bed
chm13v2.0_noY.chrom_mapping.txt
```

## Create script to align tuth to CCS

Create the script that we'll run later.

```bash
tee align_truth_to_ccs.sh << END
#!/bin/bash

function align_truth_to_ccs_zmw {
  # Align a single truth sequence to its respective ccs sequence based on ZMW.
  ccs_seq=\$(mktemp --tmpdir=/dev/shm --suffix .fa)
  truth_seq=\$(mktemp --tmpdir=/dev/shm --suffix .fa)
  out_bam=\$(mktemp --tmpdir=/dev/shm --suffix .bam)

  cat - > "\${ccs_seq}"
  truth_seq_name=\$(head -n 1 "\${ccs_seq}" | sed 's/>//g' | sed 's#/ccs#/truth#g')
  samtools faidx "\${truth_shard_fasta}" "\${truth_seq_name}" > "\${truth_seq}"

  pbmm2 align --preset HIFI \\
              --log-level FATAL \\
              --num-threads 1 \\
              -c 0 \\
              -y 70 \\
              "\${ccs_seq}" \\
              "\${truth_seq}" \\
              "\${out_bam}"

  samtools view "\${out_bam}"
  # Cleanup
  rm "\${ccs_seq}" "\${out_bam}" "\${truth_seq}"
}

export truth_shard_fasta
export -f align_truth_to_ccs_zmw

# Align truth to ccs
{
  # Construct header
  samtools view -H "\${ccs_shard_to_truth_alignment_filtered}" | grep -v '@SQ';
  samtools view "\${ccs_shard_to_truth_alignment_filtered}" | awk '{ print "@SQ\tSN:" \$1 "\tLN:" length(\$10) }' | sort | uniq;
  # Convert bam to fasta using align_truth_to_ccs_zmw.
  samtools fasta "\${ccs_shard_to_truth_alignment_filtered}" | \\
  parallel -j "\$(nproc)" \\
           --linebuffer \\
           --keep-order \\
           --pipe \\
           -N2 \\
           align_truth_to_ccs_zmw;
} | samtools view -bh - > unsorted.bam

samtools sort -O BAM --threads "\$(nproc)" unsorted.bam > "\${truth_to_ccs_shard_bam}"
samtools index "\${truth_to_ccs_shard_bam}"
samtools quickcheck "\${truth_to_ccs_shard_bam}"
END
```

Set execution right for the script:

```bash
chmod +x align_truth_to_ccs.sh
```

## Process Subread Data

Now we can process subread data to generate the appropriate inputs for
DeepConsensus. We will use the following tools to do this:

*   [`pbindex`](https://github.com/PacificBiosciences/pbbam) - generates a
    pacbio index (`.pbi`) on subread bams that allows us to process data in a
    sharded/chunked manner. (Note: `pbindex` is installed as part of the `pbbam`
    package).
*   [`ccs`](https://ccs.how/) - generates a draft consensus sequence.
*   [`actc`](https://github.com/PacificBiosciences/actc) - aligns subreads to
    the draft consensus sequence.

For convenience, we have packaged these tools in a Docker image. Follow
https://docs.docker.com/engine/install/ubuntu/ to install Docker.

```bash
# Define DOCKER_IMAGE *once* depending on whether you will be using CPU or GPU:
DOCKER_IMAGE=google/deepconsensus:1.2.0  # For CPU
sudo docker pull ${DOCKER_IMAGE}
```

Alternatively, you can install `pbindex`, `ccs` and `actc` using
[conda](https://docs.conda.io/en/latest/):

```bash
# pbindex is installed as part of the pbbam package.
# pbccs is the package name for ccs.
conda install -c bioconda pbbam pbccs actc pbmm2 samtools bedtools seqtk
```

## Running the Docker Image

If you are using Docker, you can launch the docker image using the following
command, which will also mount the quickstart directory into our container. Be
sure to use the appropriate command for your use case. These commands will
launch a container with an interactive terminal where you can execute commands.
**For the rest of the tutorial it assumed that all commands are run inside
docker.**

```bash
# Launching Docker when using a CPU:
sudo docker run \
  -it \
  -w /data \
  -v "${BASE_DIR}":/data \
  ${DOCKER_IMAGE} /bin/bash
```

Here are some details on what these docker commands are doing:

*   `-i / --interactive` - Run a docker container interactively.
*   `-t / --tty` - Allocate a pseudo-TTY. This makes working interactively
    operate like a traditional terminal session.
*   `-w / --workdir` - Sets the working directory inside the container.
*   `-v / --volume` - Binds a volume. You can specify a path and a corresponding
    path inside your container. Here we specify the quickstart directory
    (`${BASE_DIR}`) to be mounted as a directory called `/data`, which also is
    what we set as our working directory.

## Index the subreads BAM with `pbindex`

Our example `subreads.bam` is small - so indexing will be fast. But indexing a
full subreads BAM can take a long time. If you already have access to a `.pbi`
index, you should skip this step.

```bash
pbindex n1000.subreads.bam
```

This will generate `subreads.bam.pbi`.

## Run `ccs`

We will run `ccs` to generate a draft consensus. We will illustrate how sharding
can be accomplished using the `--chunk` flag. However, we will only process the
first of two chunks from our example dataset, which corresponds to processing
the first half of our subreads dataset.

```bash
n=1  # Set this to the shard you would like to process.
n_total=2  # For a full dataset, set to a larger number such as 500.

function to_shard_id {
  # ${1}: n=1-based counter
  # ${2}: n_total=1-based count
  echo "$( printf %05g "${1}")-of-$(printf "%05g" "${2}")"
}

shard_id="$(to_shard_id "${n}" "${n_total}")"

ccs --min-rq=0.88 \
      -j "$(nproc)" \
      --chunk="${n}"/"${n_total}" \
      n1000.subreads.bam \
      "${shard_id}.ccs.bam"
```

This command should generate a `00001-of-00002.ccs.bam` file. Here is an
explanation of the flags we ran `ccs` with:

*   `--min-rq=0.88` - this flag will filter out very low quality reads that are
    normally filtered using a Q>=20 read filter. Poor quality reads are unlikely
    to benefit enough from DeepConsensus polishing to be rescued from the Q>=20
    filter. A `--min-rq=0.88` corresponds to a read with ~Q9.
*   `-j` - sets the number of processors to use. `$(nproc)` will equal the
    number of available processors on our VM.
*   `--chunk` - defines a subset of the subread bam to process. We set a
    corresponding output filename with the `${shard_id}.ccs.bam` variable.

Another VM, in parallel, could process the second chunk by specifying
`--chunk=2/2`. Sharded output files can then be processed independently.

`ccs` will filter ZMWs with poor quality. Running ccs will also output a file
called `00001-of-00002.ccs.ccs_report.txt` that shows which ZMWs are filtered
and why:

```
ZMWs input               : 500

ZMWs pass filters        : 178 (35.60%)
ZMWs fail filters        : 322 (64.40%)
ZMWs shortcut filters    : 0 (0.000%)

ZMWs with tandem repeats : 3 (0.932%)

Exclusive failed counts
Below SNR threshold      : 4 (1.242%)
Median length filter     : 0 (0.000%)
Lacking full passes      : 312 (96.89%)
Heteroduplex insertions  : 3 (0.932%)
Coverage drops           : 0 (0.000%)
Insufficient draft cov   : 0 (0.000%)
Draft too different      : 0 (0.000%)
Draft generation error   : 3 (0.932%)
Draft above --max-length : 0 (0.000%)
Draft below --min-length : 0 (0.000%)
Reads failed polishing   : 0 (0.000%)
Empty coverage windows   : 0 (0.000%)
CCS did not converge     : 0 (0.000%)
CCS below minimum RQ     : 0 (0.000%)
Unknown error            : 0 (0.000%)

Additional passing metrics
ZMWs missing adapters    : 1 (0.562%)
```

## Run `actc`

Next, we will process the first chunk of our dataset by aligning subreads to the
draft consensus sequence using `actc`.

```bash
actc -j "$(nproc)" \
    n1000.subreads.bam \
    "${shard_id}.ccs.bam" \
    "${shard_id}.subreads_to_ccs.bam"
```

This command will output `00001-of-00002.subreads_to_ccs.bam`.

Both the `${shard_id}.ccs.bam` and `${shard_id}.subreads_to_ccs.bam` files will
be used as input for DeepConsensus.

## Generate labels

```bash
# Input:
export truth_reference=chm13v2.0_noY.fa
export ccs_shard_bam="${shard_id}.ccs.bam"
export truth_split=chm13v2.0_noY.chrom_mapping.txt
export subreads_to_ccs_shard_bam="${shard_id}.subreads_to_ccs.bam"
# If true, incorporate CCS Base Quality scores into tf.examples (DC v1.2).
export use_ccs_bq=True

# Output
TF_EXAMPLES_DIR="tf_examples"
export ccs_shard_to_truth_alignment_unfiltered="${shard_id}.ccs_to_truth_ref.unfiltered.bam"
export  ccs_shard_to_truth_alignment_filtered="${shard_id}.ccs_to_truth_ref.filtered.bam"
export truth_shard_bed="${shard_id}.truth.bed"
export truth_shard_fasta="${shard_id}.truth.fasta"
export truth_to_ccs_shard_bam="${shard_id}.truth_to_ccs.bam"
tf_example_fname_output="tf-@split-${shard_id}.tfrecord.gz"

train_tf_record=${TF_EXAMPLES_DIR}/train/${tf_example_fname_output/@split/train}
eval_tf_record=${TF_EXAMPLES_DIR}/eval/${tf_example_fname_output/@split/eval}
test_tf_record=${TF_EXAMPLES_DIR}/test/${tf_example_fname_output/@split/test}

```

## Align ccs to diploid truth reference

```bash
pbmm2 align --preset HIFI \
            --sort \
            -c 0 \
            -y 70 \
            "${truth_reference}" \
            "${ccs_shard_bam}" \
            "${ccs_shard_to_truth_alignment_unfiltered}"

# This step filters reads aligned to poor quality regions that are likely to
# hurt training.
samtools view -@ "$(nproc)" -F 0x904 -b "${ccs_shard_to_truth_alignment_unfiltered}" > "${ccs_shard_to_truth_alignment_filtered}"
samtools index -@"$(nproc)" "${ccs_shard_to_truth_alignment_filtered}"

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

## Align truth to ccs

Still within the docker container, create and run a shell script from the
following code.

```bash
./align_truth_to_ccs.sh
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
      --use_ccs_bq="${use_ccs_bq}" \
      --output="${tf_example_fname_output}" \
      --cpus="$(nproc)"

mv "${tf_example_fname_output/@split/train}" "${train_tf_record}"
mv "${tf_example_fname_output/@split/eval}" "${eval_tf_record}"
mv "${tf_example_fname_output/@split/test}" "${test_tf_record}"
```

## Combine sharded summary statistic files

Summary statistics file is generated for each shard. Sharded summary statistics
need to be combined into one file. This file is required for the training
script.

```bash
cat tf-summary-*.training.json | \
jq -n 'reduce
  (inputs | to_entries[]) as {$key, $value}
  ({};
    (
    if ($value|type=="string") then
      .[$key] = $value
    else
      .[$key] += $value end
    )
  )
'  > summary.training.json
```

We generated TensorFlow examples!

```bash
${train_tf_record} - Train examples
${eval_tf_record} - Evaluation examples
${test_tf_record} - Test examples
```
