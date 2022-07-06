# DeepConsensus

DeepConsensus uses gap-aware sequence transformers to correct errors in Pacific
Biosciences (PacBio) Circular Consensus Sequencing (CCS) data.

This results in greater yield of high-quality reads. See
[yield metrics](docs/yield_metrics.md) for results on three full SMRT Cells with
different chemistries and read length distributions.

## Usage

See the [quick start](docs/quick_start.md) for how to run DeepConsensus, along
with guidance on how to shard and parallelize most effectively.

### `ccs` settings matter

To get the most out of DeepConsensus, we **highly** recommend that you run `ccs`
with the parameters given in the [quick start](docs/quick_start.md). This is
because `ccs` by default filters out reads below a predicted quality of 20,
which then cannot be rescued by DeepConsensus. The runtime of `ccs` is low
enough that it is definitely worth doing this extra step whenever you are using
DeepConsensus.

### Compute setup

The recommended compute setup for DeepConsensus is to shard each SMRT Cell into
at least 500 shards, each of which can run on a 16-CPU machine (or smaller). We
find that having more than 16 CPUs available for each shard does not
significantly improve runtime. See the
[runtime metrics page](docs/runtime_metrics.md) for more information.

## Where does DeepConsensus fit into my pipeline?

After a PacBio sequencing run, DeepConsensus is meant to be run on the subreads
to create new corrected reads in FASTQ format that can take the place of the
CCS/HiFi reads for downstream analyses.

### For variant-calling downstream

For context, we are the team that created and maintains both DeepConsensus and
DeepVariant. For variant calling with DeepVariant, we tested different models
and found that the best performance is with DeepVariant v1.4 using the normal
pacbio model rather than the model trained on DeepConsensus v0.1 output. We plan
to include DeepConsensus v0.3 outputs when training the next DeepVariant model,
so if there is a DeepVariant version later than v1.4 when you read this, we
recommend using that latest version.

### For assembly downstream

We have confirmed that v0.3 outperforms v0.2 in terms of downstream assembly
contiguity and accuracy. See the
[assembly metrics page](docs/assembly_metrics.md) for details.

## How to cite

If you are using DeepConsensus in your work, please cite:

[DeepConsensus: Gap-Aware Sequence Transformers for Sequence Correction](https://www.biorxiv.org/content/10.1101/2021.08.31.458403v1)

## How DeepConsensus works

![DeepConsensus overview diagram](https://raw.githubusercontent.com/google/deepconsensus/main/docs/images/pipeline_figure.png)

Watch [How DeepConsensus Works](https://youtu.be/TlWtIao2i9E) for a quick
overview.

See this
[notebook](notebooks/Inspecting_DeepConsensus_examples_and_running_model.ipynb)
to inspect some example model inputs and outputs.

## Installation

### From pip package

If you're on a GPU machine:

```bash
pip install deepconsensus[gpu]==0.3.0
# To make sure the `deepconsensus` CLI works, set the PATH:
export PATH="/home/${USER}/.local/bin:${PATH}"
```

If you're on a CPU machine:

```bash
pip install deepconsensus[cpu]==0.3.0
# To make sure the `deepconsensus` CLI works, set the PATH:
export PATH="/home/${USER}/.local/bin:${PATH}"
```

### From Docker image

For GPU:

```bash
sudo docker pull google/deepconsensus:0.3.0-gpu
```

For CPU:

```bash
sudo docker pull google/deepconsensus:0.3.0
```

### From source

```bash
git clone https://github.com/google/deepconsensus.git
cd deepconsensus
source install.sh
```

If you have GPU, run `source install-gpu.sh` instead. Currently the only
difference is that the GPU version installs `tensorflow-gpu` instead of
`intel-tensorflow`.

(Optional) After `source install.sh`, if you want to run all unit tests, you can
do:

```bash
./run_all_tests.sh
```

## Disclaimer

This is not an official Google product.

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind, including
but not limited to diagnosis or prognosis.
