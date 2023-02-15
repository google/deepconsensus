# Genome assembly analysis with DeepConsensus reads.

DeepConsensus improves the consensus accuracy and yield of the PacBio CCS reads
which downstream can improve the quality of genome analysis. To assess the
improvement we can achieve by using DeepConsensus reads for genome assembly, we
used `hifiasm (v0.16.1)` assembler to assemble the HG002 sample and used `YAK
v0.1-r56` to assess the quality of the assembly.

Notably, the `DeepConsensus v0.3, v1.0, v1.1 and v1.2` models are trained on the
`T2T v2.0` assembly of CHM13 (with `v1.2` being additionally trained on maize),
so HG002 is a held out sample used for this analysis.

## Results

We performed genome assembly on reads from two PacBio SMRT Cells with insert
sizes of 16kb and 24kb.

**Table 1: HG002 assembly metrics with 2 SMRT Cells of PacBio CCS reads with
16kb insert size.**

Method        | Version | Assembly N50 | hap-1 QV | hap-2 QV | Mean QV
------------- | ------- | ------------ | -------- | -------- | -------
HiFi          | -       | 32253089     | 51.878   | 51.731   | 51.8045
DeepConsensus | v0.2    | 34721278     | 52.74    | 52.526   | 52.633
DeepConsensus | v0.3    | 35794972     | 52.819   | 52.79    | 52.8045
DeepConsensus | v1.0    | 33141115     | 53.092   | 53.033   | 53.0625
DeepConsensus | v1.1    | 34129352     | 52.930   | 52.910   | 52.92
DeepConsensus | v1.2    | 36019127     | 53.105   | 53.056   | 53.0805

**Table 2: HG002 assembly metrics with 2 SMRT Cells of PacBio CCS reads with
24kb insert size.**

Method        | Version | Assembly N50 | hap-1 QV | hap-2 QV | Mean QV
------------- | ------- | ------------ | -------- | -------- | -------
HiFi          | -       | 30789663     | 49.878   | 49.847   | 49.8625
DeepConsensus | v0.2    | 33130436     | 51.285   | 51.154   | 51.2195
DeepConsensus | v0.3    | 31478064     | 51.478   | 51.446   | 51.462
DeepConsensus | v1.0    | 31480574     | 51.932   | 51.775   | 51.8535
DeepConsensus | v1.1    | 34046473     | 52.001   | 51.923   | 51.962
DeepConsensus | v1.2    | 34052833     | 52.223   | 51.975   | 52.099


## Methods

Here we describe the methods used to perform the assembly and to estimate the
quality of the assemblies.

### Assembly

We performed the genome assemblies with `hifiasm v0.16.1` assembler using the
following command:

```bash
hifiasm -o OUTPUT_DIR/OUTPUT_PREFIX -t NUMBER_PROCESSES FASTQ_FILE
```

### Reference free QV estimation

We used [YAK](https://github.com/lh3/yak) `v0.1-r56` to evaluate the assembly
quality. YAK uses the k-mer distribution observed in short-reads (Illumina) to
evaluate the quality of the sequence present in the assembly. We used the
following command for YAK:

```bash
yak qv -t NUMBER_PROCESSES \
-p -K 3.2g -l 100k KMER_DB ASSEMBLY_FASTA > OUTPUT_qv.txt
```

## Data availability

The data used for this analysis is available publicly.

### PacBio CCS reads (FASTQ)

Sample | Insert size | Method        | Version | Location
------ | ----------- | ------------- | ------- | --------
HG002  | 16kb        | HiFi          | -       | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/fastqs/HG002_16kb_2SMRT_cells.hifi.q20.fastq
HG002  | 16kb        | DeepConsensus | v0.2    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/fastqs/HG002_16kb_2SMRT_cells.dc.v0.2.q20.fastq.gz
HG002  | 16kb        | DeepConsensus | v0.3    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/fastqs/HG002_16kb_2SMRT_cells.dc.v0.3.q20.fastq.gz
HG002  | 16kb        | DeepConsensus | v0.3    | gs://brain-genomics-public/research/deepconsensus/data/v1.0/assembly_analysis/fastqs/HG002_16kb_2SMRT_cells.dc.v1.0.q20.fastq.gz
HG002  | 16kb        | DeepConsensus | v1.0    | gs://brain-genomics-public/research/deepconsensus/data/v1.0/assembly_analysis/fastqs/HG002_16kb_2SMRT_cells.dc.v1.0.q20.fastq.gz
HG002  | 16kb        | DeepConsensus | v1.1    | gs://brain-genomics-public/research/deepconsensus/data/v1.1/assembly_analysis/fastqs/HG002_16kb_2SMRT_cells.dc.v1.1.q20.fastq.gz
HG002  | 16kb        | DeepConsensus | v1.2    | gs://brain-genomics-public/research/deepconsensus/data/v1.2/assembly_analysis/fastqs/HG002_16kb_2SMRT_cells.dc.v1.2.q20.fastq.gz
HG002  | 24kb        | HiFi          | -       | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/fastqs/HG002_24kb_2SMRT_cells.hifi.q20.fastq
HG002  | 24kb        | DeepConsensus | v0.2    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/fastqs/HG002_24kb_2SMRT_cells.dc.v0.2.q20.fastq.gz
HG002  | 24kb        | DeepConsensus | v0.3    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/fastqs/HG002_24kb_2SMRT_cells.dc.v0.3.q20.fastq.gz
HG002  | 24kb        | DeepConsensus | v1.0    | gs://brain-genomics-public/research/deepconsensus/data/v1.0/assembly_analysis/fastqs/HG002_24kb_2SMRT_cells.dc.v1.0.q20.fastq.gz
HG002  | 24kb        | DeepConsensus | v1.1    | gs://brain-genomics-public/research/deepconsensus/data/v1.1/assembly_analysis/fastqs/HG002_24kb_2SMRT_cells.dc.v1.1.q20.fastq.gz
HG002  | 24kb        | DeepConsensus | v1.2    | gs://brain-genomics-public/research/deepconsensus/data/v1.2/assembly_analysis/fastqs/HG002_24kb_2SMRT_cells.dc.v1.2.q20.fastq.gz

### Assembly outputs

Sample | Insert size | Method        | Version | Location
------ | ----------- | ------------- | ------- | --------
HG002  | 16kb        | HiFi          | -       | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/hifiasm_outputs/HG002_16kb_2SMRT_cells_hifi_q20/
HG002  | 16kb        | DeepConsensus | v0.2    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/hifiasm_outputs/HG002_16kb_2SMRT_cells_dc_v0.2_q20/
HG002  | 16kb        | DeepConsensus | v0.3    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/hifiasm_outputs/HG002_16kb_2SMRT_cells_dc_v0.3_q20/
HG002  | 16kb        | DeepConsensus | v1.0    | gs://brain-genomics-public/research/deepconsensus/data/v1.0/assembly_analysis/hifiasm_outputs/HG002_16kb_2SMRT_cells_dc_v1.0_q20/
HG002  | 16kb        | DeepConsensus | v1.1    | gs://brain-genomics-public/research/deepconsensus/data/v1.1/assembly_analysis/hifiasm_outputs/HG002_16kb_2SMRT_cells_dc_v1.1_q20/
HG002  | 16kb        | DeepConsensus | v1.2    | gs://brain-genomics-public/research/deepconsensus/data/v1.2/assembly_analysis/hifiasm_outputs/HG002_16kb_2SMRT_cells_dc_v1.2_q20/
HG002  | 24kb        | HiFi          | -       | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/hifiasm_outputs/HG002_24kb_2SMRT_cells_hifi_q20/
HG002  | 24kb        | DeepConsensus | v0.2    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/hifiasm_outputs/HG002_24kb_2SMRT_cells_dc_v0.2_q20/
HG002  | 24kb        | DeepConsensus | v0.3    | gs://brain-genomics-public/research/deepconsensus/data/v0.3/assembly_analysis/hifiasm_outputs/HG002_24kb_2SMRT_cells_dc_v0.3_q20/
HG002  | 24kb        | DeepConsensus | v1.0    | gs://brain-genomics-public/research/deepconsensus/data/v1.0/assembly_analysis/hifiasm_outputs/HG002_24kb_2SMRT_cells_dc_v1.0_q20/
HG002  | 24kb        | DeepConsensus | v1.1    | gs://brain-genomics-public/research/deepconsensus/data/v1.1/assembly_analysis/hifiasm_outputs/HG002_24kb_2SMRT_cells_dc_v1.1_q20/
HG002  | 24kb        | DeepConsensus | v1.2    | gs://brain-genomics-public/research/deepconsensus/data/v1.2/assembly_analysis/hifiasm_outputs/HG002_24kb_2SMRT_cells_dc_v1.2_q20/
