# Quick start for DeepConsensus

## Download data

This will download about 162 MB of data and the model is another 244 MB.

<internal>

```bash
INPUTS="${HOME}/deepconsensus_quick_start/inputs"
mkdir -p "${INPUTS}"

# Sample data:
gsutil cp gs://brain-genomics/marianattestad/deepconsensus/quick_start_inputs/*tig00030431* "${INPUTS}"

# Model:
gsutil cp gs://brain-genomics/marianattestad/deepconsensus/quick_start_inputs/models/checkpoint-50* "${INPUTS}"
gsutil cp gs://brain-genomics/marianattestad/deepconsensus/quick_start_inputs/models/params.json "${INPUTS}"
```

## Install

```bash
# <internal>
cd deepconsensus
source install.sh && ./build_pip_package.sh
```

## Run

```bash
OUTPUTS="${HOME}/deepconsensus_quick_start/outputs"
CHECKPOINT_PATH=${INPUTS}/checkpoint-50

python3 -m scripts.run_deepconsensus \
  --input_subreads_aligned=${INPUTS}/aligned_chr20.ccs_from_tig00030431.bam \
  --input_subreads_unaligned=${INPUTS}/unaligned_chr20.ccs_from_tig00030431.bam \
  --input_ccs_fasta=${INPUTS}/m64014_181209_091052.ccs_from_tig00030431.fasta \
  --output_directory=${OUTPUTS} \
  --checkpoint=${CHECKPOINT_PATH}
```

This took about 10 minutes on a 64-CPU instance on GCP.

Outputs can then be found at the following paths:

```bash
# Final output FastQ:
ls "${OUTPUTS}"/full_predictions-00000-of-00001.fastq
# Logs:
ls "${OUTPUTS}"/deepconsensus_log.txt

# Intermediate outputs from the first 4 stages:
ls "${OUTPUTS}/1_merge_datasets"
ls "${OUTPUTS}/2_generate_input"
ls "${OUTPUTS}/3_write_tf_examples"
ls "${OUTPUTS}/4_model_inference_with_beam"
```
