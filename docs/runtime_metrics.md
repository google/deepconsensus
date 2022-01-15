# Runtime on different hardware configurations

We use the data in [Quick Start](quick_start.md) to test various different
hardware configurations.

## 64vCPUs (Cascade Lake) (n2-standard-64 on GCP)

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

* With pip: 735.94 seconds / 1000 ZMWs
* With Docker: 760.54 seconds / 1000 ZMWs

## 16vCPUs (Cascade Lake) (n2-standard-16 on GCP)

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

* With pip: 1131.88 seconds / 1000 ZMWs
* With Docker: 1129.49 seconds / 1000 ZMWs

## P100 GPU, 16vCPUs (SkyLake) (n1-standard-16 with nvidia-tesla-p100 on GCP)

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

* With pip: 346.73 seconds / 1000 ZMWs
* With Docker: 433.64 seconds / 1000 ZMWs
