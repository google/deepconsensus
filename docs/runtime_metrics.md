# Runtime on different hardware configurations

We processed 10,000 ZMWs with `ccs` and `actc`. After filtering, we were left
with 3,577 ZMWs, which we used to profile runtimes using `deepconsensus run`
across several hardware configurations. These estimates only reflect the runtime
required to perform inference, after subread data has been preprocessed.

In general, larger values of `batch_zmws` and `batch_size` result in faster
performance, but if set too large you can exhaust available memory.

You can use this table and your own hardware configuration to get a sense of
optimal settings.

machine        | gpu   | batch_size | batch_zmws | max_mem (Gb) | per_zmw (seconds) | duration (minutes)
:------------- | :---- | ---------: | ---------: | -----------: | ----------------: | -----------------:
n1-standard-16 | True  | 1024       | 1000       | 103.2        | 0.836679          | 49.88
n2-standard-64 | False | 4096       | 500        | 97.1         | 1.03109           | 61.47
n1-standard-16 | False | 2048       | 1000       | 157.3        | 1.06933           | 63.75
n2-standard-16 | False | 2048       | 100        | 34.7         | 1.20554           | 71.87

## Runtime Profiles

![DeepConsensus runtime profiling](images/runtimes.png)

Runtime, max memory usage, `batch_size`, and `batch_zmws` are shown across
different machine types. We observe that `batch_zmws` has a large impact on
memory usage. Our current implementation requires setting `batch_zmws` and
`batch_size` carefully to achieve optimal performance. We are working to improve
performance further and allow for more predictable runtimes based on
DeepConsensus settings.

We only show GPU runtimes for `n1-standard-16`. Note that we observe job
failures when using GPU with larger batch sizes (>=4096), with larger values of
`batch_zmws` (500, 1000).

## Runtime Test Configurations

### `n2-standard-64`: 64vCPUs (Cascade Lake)

This command shows what machine we tested on:

```bash
gcloud compute instances create "${USER}-n2-64" \
  --scopes "compute-rw,storage-full,cloud-platform" \
  --image-family "ubuntu-2004-lts" \
  --image-project "ubuntu-os-cloud" \
  --machine-type "n2-standard-64" \
  --boot-disk-size "200" \
  --zone "us-west1-b"
```

### `n2-standard-16`: 16vCPUs (Cascade Lake)

This command shows what machine we tested on:

```bash
gcloud compute instances create "${USER}-n2-16" \
  --scopes "compute-rw,storage-full,cloud-platform" \
  --image-family "ubuntu-2004-lts" \
  --image-project "ubuntu-os-cloud" \
  --machine-type "n2-standard-16" \
  --boot-disk-size "200" \
  --zone "us-west1-b"
```

### `n1-standard-16`, 16vCPUs

This command shows what machine we tested on:

```bash
gcloud compute instances create "${USER}-gpu" \
  --scopes "compute-rw,storage-full,cloud-platform" \
  --maintenance-policy "TERMINATE" \
  --image-family "ubuntu-2004-lts" \
  --image-project "ubuntu-os-cloud" \
  --machine-type "n1-standard-16" \
  --boot-disk-size "200" \
  --zone "us-west1-b"
```

### `n1-standard-16 + P100 GPU`, 16vCPUs (SkyLake)

This command shows what machine we tested on:

```bash
gcloud compute instances create "${USER}-gpu" \
  --scopes "compute-rw,storage-full,cloud-platform" \
  --maintenance-policy "TERMINATE" \
  --accelerator=type=nvidia-tesla-p100,count=1 \
  --image-family "ubuntu-2004-lts" \
  --image-project "ubuntu-os-cloud" \
  --machine-type "n1-standard-16" \
  --boot-disk-size "200" \
  --zone "us-west1-b" \
  --min-cpu-platform "Intel Skylake"
```

## Test Dataset

The 10k ZMW test dataset is available on Google Storage:

`gs://brain-genomics-public/research/deepconsensus/quickstart/v0.3/10000.subreads.bam`
