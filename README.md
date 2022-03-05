# DeepConsensus

DeepConsensus uses gap-aware sequence transformers to correct errors in Pacific
Biosciences (PacBio) Circular Consensus Sequencing (CCS) data.

![DeepConsensus overview diagram](https://raw.githubusercontent.com/google/deepconsensus/main/docs/images/pipeline_figure.png)

## Installation

### From pip package

If you're on a GPU machine:

```bash
pip install deepconsensus[gpu]==0.2.0
# To make sure the `deepconsensus` CLI works, set the PATH:
export PATH="/home/${USER}/.local/bin:${PATH}"
```

If you're on a CPU machine:

```bash
pip install deepconsensus[cpu]==0.2.0
# To make sure the `deepconsensus` CLI works, set the PATH:
export PATH="/home/${USER}/.local/bin:${PATH}"
```

### From Docker image

For GPU:

```bash
sudo docker pull google/deepconsensus:0.2.0-gpu
```

For CPU:

```bash
sudo docker pull google/deepconsensus:0.2.0
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

## Usage

See the [quick start](https://github.com/google/deepconsensus/blob/main/docs/quick_start.md).

## Where does DeepConsensus fit into my pipeline?

After a PacBio sequencing run, DeepConsensus is meant to be run on the subreads
to create new corrected reads in FASTQ format that can take the place of the CCS
reads for downstream analyses.

See the [quick start](https://github.com/google/deepconsensus/blob/main/docs/quick_start.md)
for an example of inputs and outputs.

## How to cite

If you are using DeepConsensus in your work, please cite:

[DeepConsensus: Gap-Aware Sequence Transformers for Sequence Correction](https://www.biorxiv.org/content/10.1101/2021.08.31.458403v1)

## Disclaimer

This is not an official Google product.

NOTE: the content of this research code repository (i) is not intended to be a
medical device; and (ii) is not intended for clinical use of any kind, including
but not limited to diagnosis or prognosis.
