# Quick start for DeepConsensus

This Quick Start provides an example of running DeepConsensus on a small example
dataset. This will cover the steps of running from a subreads BAM file and
generate a FASTQ of consensus reads.

This covers the following stages:
1. Running [pbccs] with the `--all` option to output all reads (it is possible
   to use DeepConsensus from existing pbccs reasd, but yield will be higher when
   including all reads)
2. Aligning subreads to the pbccs consensus with [actc]
3. Running DeepConsensus using one of two options (with pip or using Docker)

## System configuration

We tested the DeepConsensus quickstart with the following configuration:

```bash
OS: Ubuntu 20.04.3 LTS (x86_64)
Python version: Python 3.8.10
CPUs: 64vCPUs (x86_64, GenuineIntel, Cascade Lake)
Memory: 256G
```

DeepConsensus can be run on any compatible Unix systems. In this case, we used a
[n2-standard-64 machine on GCP](https://cloud.google.com/compute/docs/general-purpose-machines#n2_machines).

## Download data for testing

This will download about 142 MB of data and the model is another 245 MB.

```bash
# Set directory where all data and model will be placed.
QUICKSTART_DIRECTORY="${HOME}/deepconsensus_quick_start"
# This will soon have 2 subfolders: data, model.

DATA="${QUICKSTART_DIRECTORY}/data"
MODEL_DIR="${QUICKSTART_DIRECTORY}/model"
mkdir -p "${DATA}"
mkdir -p "${MODEL_DIR}"

# Download the input data which is PacBio subreads.
# <internal>
gsutil cp gs://brain-genomics/pichuan/b208710498/v0.2/subreads.bam* "${DATA}"/

# Download DeepConsensus model.
# <internal>
gsutil cp gs://brain-genomics-public/research/deepconsensus/models/v0.1/* "${MODEL_DIR}"/
```

## Process the data with [pbccs] and [actc]

You can install `ccs` and `actc` on your own. For convenience, we put them in
a Docker image:

```
<internal>
DOCKER_IMAGE=gcr.io/google.com/brain-genomics/deepconsensus:211222
sudo docker pull ${DOCKER_IMAGE}
```

<internal>

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
`--chunk` option in `ccs`.

Then, we create `subreads_to_ccs.bam` was created by running [actc]:

```bash
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  actc -j "$(nproc)" \
    /data/subreads.bam \
    /data/ccs.bam \
    /data/subreads_to_ccs.bam
```

DeepConsensus will take FASTA format of ccs, so we use samtools to generate.

```bash
sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  samtools fasta --threads "$(nproc)" /data/ccs.bam > ${DATA}/ccs.fasta

sudo docker run -v "${DATA}":"/data" ${DOCKER_IMAGE} \
  samtools faidx /data/ccs.fasta
```

## Run DeepConsensus

### (Option 1) Run DeepConsensus using Docker

You can directly run DeepConsensus with Docker:

```bash
sudo docker run -v "${DATA}":"/data" -v "${MODEL_DIR}":"/model" ${DOCKER_IMAGE} \
  deepconsensus run \
  --subreads_to_ccs=/data/subreads_to_ccs.bam  \
  --ccs_fasta=/data/ccs.fasta \
  --checkpoint=/model/checkpoint-50 \
  --output=/data/output.fastq \
  --batch_zmws=100
```

This took ~12.5 minutes on the 64-vCPU instance on GCP we tested on.
At the end of your run, you should see:

```
Processed 1000 ZMWs in 715.4207527637482 seconds
Outcome counts: OutcomeCounter(empty_sequence=0, only_gaps_and_padding=50, failed_quality_filter=435, failed_length_filter=0, success=515)
```

### (Option 2) Install and run DeepConsensus via pip install

Empirically, sometimes we see running in Docker might cause a slight overhead.
Another option is to install DeepConsensus via `pip install`:

<internal>
```bash
pip install \
  --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple \
  deepconsensus[cpu]==0.2.0b208710498
```

If you're using a GPU machine, install with `deepconsensus[gpu]` instead.

<internal>
I do.

To make sure the `deepconsensus` CLI works, set the PATH:

```bash
export PATH="/home/${USER}/.local/bin:${PATH}"
```

```bash
CHECKPOINT=${MODEL_DIR}/checkpoint-50

time deepconsensus run \
  --subreads_to_ccs=${DATA}/subreads_to_ccs.bam  \
  --ccs_fasta=${DATA}/ccs.fasta \
  --checkpoint=${CHECKPOINT} \
  --output=${DATA}/output.fastq \
  --batch_zmws=100
```

<internal>
`W1223 05:23:07.523110 140307704698688 utils.py:474] window at 10900 has no ccs alignment.`
, which is not great for users. Rerun this whole thing and remove
this <internal>

This took ~12 minutes on the 64-vCPU instance on GCP we tested on.

At the end of your run, you should see:
```
Processed 1000 ZMWs in 721.7911970615387 seconds
Outcome counts: OutcomeCounter(empty_sequence=0, only_gaps_and_padding=50, failed_quality_filter=435, failed_length_filter=0, success=515)
```
the outputs can be found at the following paths:

```bash
# Final output fastq file which has DeepConsensus reads.
ls "${DATA}"/output.fastq
```

[pbccs]: https://github.com/PacificBiosciences/ccs
[actc]: https://github.com/PacificBiosciences/align-clr-to-ccs
