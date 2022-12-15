# Quick Start for DeepConsensus

This quick start tutorial walks through how to process PacBio Hi-Fi sequence
data that originates as a subread BAM file, and how to run DeepConsensus to
generate polished reads.

This tutorial is organized as follows:

1.  [Setting up a VM](#setting-up-a-vm)
2.  [Parallelization](#parallelization)
3.  [Download Example Data](#download-example-data)
4.  [Process Subread Data](#process-subread-data)
5.  [Run DeepConsensus](#run-deepconsensus)
6.  [Tips for Optimizing](#tips-for-optimizing)

## Setting up a VM

DeepConsensus can be run on Unix systems. The command below can be used to spin
up a compatible virtual machine (VM) on Google Cloud Platform (GCP). This
command will spin up a
[n1-standard-16 machine on GCP](https://cloud.google.com/compute/docs/general-purpose-machines#n1_machines).

```bash
VM=deepconsensus-quick-start
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

You can log into the new VM using `gcloud`:

```bash
gcloud compute ssh "${VM}" --zone=us-west1-b
```

See the [runtime metrics page](runtime_metrics.md) for an overview of runtimes
using different GCP compute VM configurations.

### GPU Setup

If you are planning on running DeepConsensus with an NVIDIA GPU, you can use the
command below to install Docker and the GPU libraries required:

```bash
# For GPU only:
curl https://raw.githubusercontent.com/google/deepvariant/r1.4/scripts/install_nvidia_docker.sh -o install_nvidia_docker.sh
bash install_nvidia_docker.sh
```

### CPU Setup

Follow https://docs.docker.com/engine/install/ubuntu/ to install Docker.

## Parallelization

One 8M SMRT Cell can take ~1000 hours to run (without parallelization) depending
on the fragment lengths of the sequencing library - see the
[yield metrics page](yield_metrics.md). If we split this into 500 shards, that
is about 2 hours per shard. There is some variability between shards, but this
should give you an idea of what to expect. This estimate is only for the
DeepConsensus processing step, and does not include the preprocessing required
with *ccs* and *actc*.

We recommend running a single small shard first so you have an idea of the
runtime to expect on your compute setup and with your sequencing run, since
factors from compute hardware to library fragment length can make a big
difference.

Keep in mind that pre-processing tools (`pbccs`, `actc`) and DeepConsensus are
set up to make use of all available compute resources. However, subread datasets
are very large so distributing this work via sharding across multiple VMs will
allow for processing over reasonable timeframes.

## Download example data

Next we will download example data which contains 1000 ZMWs and a DeepConsensus
model. The example data is about 210 MB and the model is 38.18 MB.

We will download data using `gsutil` which is pre-installed on GCP VMs, but you
can install it in other environments using `pip install gsutil`.

```bash
# Create a work directory and place to store our model for the quick start.
QS_DIR="${HOME}/deepconsensus_quick_start"
mkdir -p "${QS_DIR}" "${QS_DIR}/model"

# Download the input PacBio Subread data.
gsutil cp gs://brain-genomics-public/research/deepconsensus/quickstart/v1.1/n1000.subreads.bam "${QS_DIR}"/

# Download the DeepConsensus model.
gsutil cp -r gs://brain-genomics-public/research/deepconsensus/models/v1.1/model_checkpoint/* "${QS_DIR}"/model/
```

This directory should now contain the following files:

```
n1000.subreads.bam
model/checkpoint.data-00000-of-00001
model/checkpoint.index
model/params.json
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

For convenience, we have packaged these tools in a Docker image. Be sure to use
the appropriate version (CPU / GPU) depending on your use case.

```bash
# Define DOCKER_IMAGE *once* depending on whether you will be using CPU or GPU:
DOCKER_IMAGE=google/deepconsensus:1.1.0  # For CPU
DOCKER_IMAGE=google/deepconsensus:1.1.0-gpu  # For GPU
sudo docker pull ${DOCKER_IMAGE}
```

Alternatively, you can install `pbindex`, `ccs` and `actc` using
[conda](https://docs.conda.io/en/latest/):

```bash
# pbindex is installed as part of the pbbam package.
# pbccs is the package name for ccs.
conda install -c bioconda pbbam pbccs actc
```

## Running the Docker Image

If you are using Docker, you can launch the docker image using the following
command, which will also mount the quickstart directory into our container. Be
sure to use the appropriate command for your use case. These commands will
launch a container with an interactive terminal where you can execute commands.

```bash
# Launching Docker when using a CPU:
sudo docker run \
  -it \
  -w /data \
  -v "${QS_DIR}":/data \
  ${DOCKER_IMAGE} /bin/bash

# Launching Docker when using a GPU:
sudo docker run \
  --gpus all \
  -it \
  -w /data \
  -v "${QS_DIR}":/data \
  ${DOCKER_IMAGE} /bin/bash
```

Here are some details on what these docker commands are doing:

*   `-i / --interactive` - Run a docker container interactively.
*   `-t / --tty` - Allocate a pseudo-TTY. This makes working interactively
    operate like a traditional terminal session.
*   `-w / --workdir` - Sets the working directory inside the container.
*   `-v / --volume` - Binds a volume. You can specify a path and a corresponding
    path inside your container. Here we specify the quickstart directory
    (`${QS_DIR}`) to be mounted as a directory called `/data`, which also is
    what we set as our working directory.

### Index the subreads BAM with `pbindex`

Our example `subreads.bam` is small - so indexing will be fast. But indexing a
full subreads BAM can take a long time. If you already have access to a `.pbi`
index, you should skip this step.

```bash
pbindex n1000.subreads.bam
```

This will generate `subreads.bam.pbi`.

### Run `ccs`

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

### Run `actc`

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

## Run DeepConsensus

If using the Docker container, DeepConsensus was installed alongside ccs and
actc above. Alternatively, you can install DeepConsensus using `pip` (see the
[README](../README.md)).

```bash
deepconsensus run \
  --subreads_to_ccs=${shard_id}.subreads_to_ccs.bam  \
  --ccs_bam=${shard_id}.ccs.bam \
  --checkpoint=model/checkpoint \
  --output=${shard_id}.output.fastq
```

At the end of your run, you should see:

```
Processed 178 ZMWs in 230.602 seconds
Outcome counts: OutcomeCounter(empty_sequence=0, only_gaps=0, failed_quality_filter=1, failed_length_filter=0, success=177)
```

## Optimizing Runtime

You may be able to tweak the `--batch_size` and `--batch_zmws` parameters to
optimize for runtime specific to your hardware. You can also see
[runtime_metrics.md](runtime_metrics.md) for runtime on different CPU or GPU
machines.
