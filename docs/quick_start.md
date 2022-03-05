# Quick start for DeepConsensus

This Quick Start provides an example of running DeepConsensus on a small example
dataset. This will cover the steps of running from a subreads BAM file and
generate a FASTQ of consensus reads.

This covers the following stages:

1.  How to easily parallelize the work across multiple machines.
2.  Running *[ccs]* with the `--all` option to output all reads (it is possible
    to use DeepConsensus from existing *ccs* reads, but yield will be higher
    when including all reads).
3.  Aligning subreads to the *ccs* consensus with *[actc]*
4.  Running DeepConsensus using either pip or Docker

## System configuration

We tested the DeepConsensus quickstart with the following configuration:

```bash
OS: Ubuntu 20.04.3 LTS (x86_64)
Python version: Python 3.8.10
CPUs: 16vCPUs (x86_64, GenuineIntel, SkyLake)
Memory: 60G
GPU: 1 nvidia-tesla-p100
```

DeepConsensus can be run on any compatible Unix systems. In this case, we used a
[n1-standard-16 machine on GCP](https://cloud.google.com/compute/docs/general-purpose-machines#n1_machines),
with an NVIDIA P100 GPU. To reproduce on GCP, use
[this command for P100 GPU](runtime_metrics.md#p100-gpu-16vcpus-skylake-n1-standard-16-with-nvidia-tesla-p100-on-gcp).

See the [runtime metrics page](runtime_metrics.md) for a few examples of runtime
on different compute setups on Google Cloud.

## Parallelization

Let's do a little back-of-the-envelope calculation to determine what
parallelization setup to use and what runtime to expect.

One SMRTcell produces something like 3-4 million ZMWs depending on the fragment
length of the sequencing library. That is based on an 8M SMRTcell. We will
assume 4 million ZMWs for this estimate.

For this small calculation, we'll assume 1.1 seconds/ZMW runtime to match the
16vCPU (no GPU) machines that most people should have access to. See the
[runtime metrics page](runtime_metrics.md) to get an estimate matching your
compute setup.

(1.1 seconds/ZMW) * (4 million ZMWs) = 4.4 million seconds = 1,222 hours.

If we split this into 500 shards, that is about 2.4 hours per shard. There is
some variability between shards, but this should give you an idea of what to
expect. This is only for the DeepConsensus step itself, not including the
preprocessing with *ccs* and *actc*.

We recommend running a single small shard first so you have an idea of the
runtime to expect on your compute setup and with your sequencing run, since
factors from compute hardware to library fragment length can make a big
difference.

Note that we recommend running each shard on a separate machine/VM as opposed to
using a tool like `parallel` to run DeepConsensus on multiple shards on the same
machine, since currently each DeepConsensus run will use all the available
resources. We are working on ways to restrict this to enable making better use
of machines with many cores. The [runtime metrics page](runtime_metrics.md) is
showing runtimes where DeepConsensus is not competing with any other jobs on the
same machine, so expect worse runtimes when these conditions are not ideal.

Here is a quick example of setting up variables and file names for parallelizing
across shards.

```bash
# Format shard number nicely for file names:
function to_shard_id {
  # ${1}: n=1-based counter
  # ${2}: n_total=1-based count
  echo "$( printf %05g "${1}")-of-$(printf "%05g" "${2}")"
}

# Here's what this looks like with 10 shards:
n_total=10
for n in $(seq 1 $n_total); do
  echo "run ccs with --chunk=${n}/${n_total}"
  shard_id="$(to_shard_id "${n}" "${n_total}")"
  echo "for file names: ${shard_id}"
done
```

In this quick start example, we are using such a small dataset that sharding is
unnecessary, but we will show it as if we are running 1 shard out of 1 total
shards so it is easy to adapt.

## Download example data

This will download about 142 MB of data and the model is another 245 MB.

```bash
# Set directory where all data and model will be placed.
QUICKSTART_DIRECTORY="${HOME}/deepconsensus_quick_start"
# This will soon have 2 subfolders: data, model.

DATA="${QUICKSTART_DIRECTORY}/data"
MODEL_DIR="${QUICKSTART_DIRECTORY}/model"
mkdir -p "${DATA}"
mkdir -p "${MODEL_DIR}"

# Download the input data, which is PacBio subreads.
gsutil cp gs://brain-genomics-public/research/deepconsensus/quickstart/v0.2/subreads.bam* "${DATA}"/

# Download the DeepConsensus model.
gsutil cp gs://brain-genomics-public/research/deepconsensus/models/v0.2/* "${MODEL_DIR}"/
```

## If running with GPU, set up your GPU machine correctly.

In our example run, because we're using GPU, we used:

```bash
curl https://raw.githubusercontent.com/google/deepvariant/r1.3/scripts/install_nvidia_docker.sh -o install_nvidia_docker.sh
bash install_nvidia_docker.sh
```

to make sure our GPU is set up correctly.

## Process the data with *ccs* and *actc*

You can install *[ccs]* and *[actc]* on your own. For convenience, we put them
in a Docker image:

```bash
# DOCKER_IMAGE=google/deepconsensus:0.2.0  # For CPU
DOCKER_IMAGE=google/deepconsensus:0.2.0-gpu  # For GPU
sudo docker pull ${DOCKER_IMAGE}
```

DeepConsensus operates on subreads aligned to a draft consensus. We use *ccs*
to generate this.

The *ccs* software helpfully shards its output for us when used with `--chunk`,
all from the same subreads bam. You simply run the whole rest of this quick
start on a different machine for each value of `n` up to your desired `n_total`.

```bash
# In this quick start example, we'll use the variables for sharding but just
# show it with 1 shard so we run with all 1000 ZMWs in the sample dataset.
n_total=1  # For a real run, try 500 here.
n=1  # Set this for each machine as shown in the example loop above.

function to_shard_id {
  # ${1}: n=1-based counter
  # ${2}: n_total=1-based count
  echo "$( printf %05g "${1}")-of-$(printf "%05g" "${2}")"
}

shard_id="$(to_shard_id "${n}" "${n_total}")"
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  ccs --all \
    -j "$(nproc)" \
    --chunk="${n}"/"${n_total}" \
    "/data/subreads.bam" \
    "/data/${shard_id}.ccs.bam"
```

Note that the `--all` flag is a required setting for DeepConsensus to work
optimally. This allows DeepConsensus to rescue reads previously below the
quality threshold.

Then, we create `subreads_to_ccs.bam` by running *actc*:

```bash
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  actc -j "$(nproc)" \
    "/data/subreads.bam" \
    "/data/${shard_id}.ccs.bam" \
    "/data/${shard_id}.subreads_to_ccs.bam"
```

DeepConsensus will take the consensus sequences output by *ccs* in FASTA format.

*actc* already converted the BAM into FASTA. Rename and index it.

```bash
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  mv /data/${shard_id}.subreads_to_ccs.fasta /data/${shard_id}.ccs.fasta

sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  samtools faidx /data/${shard_id}.ccs.fasta
```

## Run DeepConsensus

### Install and run DeepConsensus via pip install

You can install DeepConsensus using `pip`:

```bash
# GPU ONLY:
pip install deepconsensus[gpu]==0.2.0
```

NOTE: If you're using a CPU machine, do this instead:

```bash
# CPU ONLY:
pip install deepconsensus[cpu]==0.2.0
```

To make sure the `deepconsensus` command-line interface works, set the PATH:

```bash
export PATH="/home/${USER}/.local/bin:${PATH}"
```

The step above is important. Otherwise you might see an error like:
`deepconsensus: command not found`.

```bash
CHECKPOINT=${MODEL_DIR}/checkpoint-50

time deepconsensus run \
  --subreads_to_ccs=${DATA}/${shard_id}.subreads_to_ccs.bam  \
  --ccs_fasta=${DATA}/${shard_id}.ccs.fasta \
  --checkpoint=${CHECKPOINT} \
  --output=${DATA}/${shard_id}.output.fastq \
  --batch_zmws=100
```

At the end of your run, you should see:

```
Processed 1000 ZMWs in 341.3297851085663 seconds
Outcome counts: OutcomeCounter(empty_sequence=0, only_gaps_and_padding=50, failed_quality_filter=424, failed_length_filter=0, success=526)
```

The final output FASTQ can be found at the following path:

```bash
ls "${DATA}"/${shard_id}.output.fastq
```

### (Optional) Run DeepConsensus using Docker

If `pip install` didn't work well for you, we encourage you to file
[a GitHub issue] to let us know.

You can also try running DeepConsensus with Docker:

```bash
time sudo docker run --gpus all \
  -v "${DATA}":"/data" -v "${MODEL_DIR}":"/model" ${DOCKER_IMAGE} \
  deepconsensus run \
  --subreads_to_ccs=/data/${shard_id}.subreads_to_ccs.bam  \
  --ccs_fasta=/data/${shard_id}.ccs.fasta \
  --checkpoint=/model/checkpoint-50 \
  --output=/data/${shard_id}.output.fastq \
  --batch_zmws=100
```

At the end of your run, you should see:

```
Processed 1000 ZMWs in 428.84565114974976 seconds
Outcome counts: OutcomeCounter(empty_sequence=0, only_gaps_and_padding=50, failed_quality_filter=424, failed_length_filter=0, success=526)
```

Currently we notice that the Docker GPU version is slower. We're still trying
to improve this. If you have any suggestions, please let us know through
[a GitHub issue].


## Tweaking for speed

You might be able to tweak parameters like `--batch_zmws` depending on your
hardware limit. You can also see [runtime_metrics.md](runtime_metrics.md) for
runtime on different CPU or GPU machines.

[ccs]: https://ccs.how
[actc]: https://github.com/PacificBiosciences/align-clr-to-ccs
[a GitHub issue]: https://github.com/google/deepconsensus/issues
