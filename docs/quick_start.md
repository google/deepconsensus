# Quick start for DeepConsensus

## System configuration

We tested the DeepConsensus quickstart with the following configuration:

```bash
OS: Ubuntu 18.04.5 LTS (x86_64)
Python version: Python 3.6.9
CPUs: 64vCPUs (x86_64, GenuineIntel)
Memory: 126G
```

## Download data for testing

This will download about 252 MB of data and the model is another 245 MB.

```bash
# Set directory where inputs will be placed.
QUICKSTART_DIRECTORY="${HOME}/deepconsensus_quick_start/"
# This will soon have 3 subfolders: inputs, model, and outputs.

INPUTS="${QUICKSTART_DIRECTORY}/inputs"
MODEL_DIR="${QUICKSTART_DIRECTORY}/model"
mkdir -p "${INPUTS}"
mkdir -p "${MODEL_DIR}"

# Download the input data which is PacBio subreads and CCS reads.
gsutil cp gs://brain-genomics-public/research/deepconsensus/quickstart/v0.1/subreads.bam "${INPUTS}"/
gsutil cp gs://brain-genomics-public/research/deepconsensus/quickstart/v0.1/ccs.fasta "${INPUTS}"/

# Optionally skip the alignment step below by downloading this instead:
# gsutil cp gs://brain-genomics-public/research/deepconsensus/quickstart/v0.1/subreads_to_ccs.bam "${INPUTS}"/

# Download DeepConsensus model.
gsutil cp gs://brain-genomics-public/research/deepconsensus/models/v0.1/* "${MODEL_DIR}"/
```

## Prepare input files

```bash
cd "${INPUTS}"/

# We used pbmm2 v1.4.0 for mapping subreads to CCS reads.
pbmm2 align --sample "sample" --preset SUBREAD --sort \
  "ccs.fasta" "subreads.bam" "aligned.subreads.bam"

# Only subread alignments to the correct molecule were retained.
# We used samtools and awk to filter incorrect alignments using the below
# command:
samtools view -h "aligned.subreads.bam" | \
  awk '{ if($1 ~ /^@/) { print; } else { split($1,A,"/"); \
  split($3,B,"/"); if(A[2]==B[2]) { split(A[3],C,"_"); \
  print $0 "\tqs:i:" C[1]; } } }' | samtools view -b > "subreads_to_ccs.bam"
```

## Install DeepConsensus

```bash
pip install deepconsensus==0.1.0
```

## Run DeepConsensus

```bash
# Set directory where outputs will be placed and set the model for DeepConsensus
OUTPUTS="${QUICKSTART_DIRECTORY}/outputs"
CHECKPOINT_PATH=${MODEL_DIR}/checkpoint-50

# Run DeepConsensus
python3 -m deepconsensus.scripts.run_deepconsensus \
  --input_subreads_aligned=${INPUTS}/subreads_to_ccs.bam \
  --input_subreads_unaligned=${INPUTS}/subreads.bam \
  --input_ccs_fasta=${INPUTS}/ccs.fasta \
  --output_directory=${OUTPUTS} \
  --checkpoint=${CHECKPOINT_PATH}
```

Expected runtime of DeepConsensus on 64-vCPU instance on GCP is ~10 minutes.

Once DeepConsensus finishes, the outputs can be found at the following paths:

```bash
# Final output fastq file which has DeepConsensus reads.
ls "${OUTPUTS}"/full_predictions-00000-of-00001.fastq
# The logs can be found in:
ls "${OUTPUTS}"/deepconsensus_log.txt

# The log for each stage of DeepConsensus can be found in separate logs:
ls "${OUTPUTS}/1_merge_datasets"
ls "${OUTPUTS}/2_generate_input"
ls "${OUTPUTS}/3_write_tf_examples"
ls "${OUTPUTS}/4_model_inference_with_beam"
```
