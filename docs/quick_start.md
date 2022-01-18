# Quick start for DeepConsensus

This Quick Start provides an example of running DeepConsensus on a small example
dataset. This will cover the steps of running from a subreads BAM file and
generate a FASTQ of consensus reads.

This covers the following stages:
1. Running *[ccs]* with the `--all` option to output all reads (it is possible
   to use DeepConsensus from existing *ccs* reads, but yield will be higher when
   including all reads)
2. Aligning subreads to the *ccs* consensus with *[actc]*
3. Running DeepConsensus using either pip or Docker

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
[n1-standard-16 machine on GCP](https://cloud.google.com/compute/docs/general-purpose-machines#n1_machines), with an NVIDIA P100 GPU.

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

You can install *[ccs]* and *[actc]* on your own. For convenience, we put them in
a Docker image:

```bash
DOCKER_IMAGE=google/deepconsensus:0.2.0-gpu
sudo docker pull ${DOCKER_IMAGE}
```

DeepConsensus operates on subreads aligned to a draft consensus. We use *ccs*
to generate this.

```bash
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  ccs --all \
    -j "$(nproc)" \
    /data/subreads.bam \
    /data/ccs.bam
```

Note that the `--all` flag is a required setting for DeepConsensus to work
optimally. This allows DeepConsensus to rescue reads previously below the
quality threshold.
If you want to split up the task for parallelization, we recommend using the
`--chunk` option in *ccs*.

Then, we create `subreads_to_ccs.bam` by running *actc*:

```bash
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  actc -j "$(nproc)" \
    /data/subreads.bam \
    /data/ccs.bam \
    /data/subreads_to_ccs.bam
```

DeepConsensus will take the consensus sequences output by *ccs* in FASTA format.

*actc* already converted the BAM into FASTA. Rename and index it.

```bash
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  mv /data/subreads_to_ccs.fasta /data/ccs.fasta

sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  samtools faidx /data/ccs.fasta
```

## Run DeepConsensus

### Install and run DeepConsensus via pip install

You can install DeepConsensus using `pip`:

```bash
pip install deepconsensus[gpu]==0.2.0
```

NOTE: If you're using a CPU machine, install with `deepconsensus[cpu]` instead.

To make sure the `deepconsensus` CLI works, set the PATH:

```bash
export PATH="/home/${USER}/.local/bin:${PATH}"
```

The step above is important. Otherwise you might see an error like:
`deepconsensus: command not found`.

```bash
CHECKPOINT=${MODEL_DIR}/checkpoint-50

time deepconsensus run \
  --subreads_to_ccs=${DATA}/subreads_to_ccs.bam  \
  --ccs_fasta=${DATA}/ccs.fasta \
  --checkpoint=${CHECKPOINT} \
  --output=${DATA}/output.fastq \
  --batch_zmws=100
```

At the end of your run, you should see:

```
Processed 1000 ZMWs in 341.3297851085663 seconds
Outcome counts: OutcomeCounter(empty_sequence=0, only_gaps_and_padding=50, failed_quality_filter=424, failed_length_filter=0, success=526)
```

The final output FASTQ can be found at the following path:

```bash
ls "${DATA}"/output.fastq
```

### (Optional) Run DeepConsensus using Docker

If `pip install` didn't work well for you, we encourage you to file
[a GitHub issue] to let us know.

You can also try running DeepConsensus with Docker:

```bash
time sudo docker run --gpus all \
  -v "${DATA}":"/data" -v "${MODEL_DIR}":"/model" ${DOCKER_IMAGE} \
  deepconsensus run \
  --subreads_to_ccs=/data/subreads_to_ccs.bam  \
  --ccs_fasta=/data/ccs.fasta \
  --checkpoint=/model/checkpoint-50 \
  --output=/data/output.fastq \
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
