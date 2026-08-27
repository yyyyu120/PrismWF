# Datasets

## ARES Benchmark

The closed-world and open-world 2--5-tab experiments use the released ARES
benchmark and the official WFlib instance-level splitting protocol:

- source and instructions: https://github.com/Xinhao-Deng/Multitab-WF-Datasets
- archived data: https://zenodo.org/records/13732130

The benchmark contains concurrently collected multi-tab Tor sessions; it is
not produced by simply adding independently collected single-tab traces.

The Zenodo archives contain one combined NPZ per setting. The complete
train/validation/test partitions and exact split indices used by PrismWF are
available from the
[PrismWF ARES Data Splits](https://huggingface.co/datasets/yuuu94/PrismWF-ARES-Splits)
repository. Download all eight multi-tab archives, verify their published MD5
checksums, and apply these indices with:

```bash
bash scripts/download_ares_multitab.sh
```

Large archives are downloaded with eight resumable HTTP range connections by
default. Set `DOWNLOAD_CONNECTIONS` to adjust this for the local network, for
example `DOWNLOAD_CONNECTIONS=16 bash scripts/download_ares_multitab.sh`.

The script reproduces the published indices locally using the released WFlib
protocol with random seed 2024: a 90/10 train-test split followed by a 90/10
train-validation split of the first partition. This produces 81% training, 9%
validation, and 10% test instances. It saves `split_indices.npz` and
`split_metadata.json` in every dataset directory. The published Hugging Face
copy can be used to verify these generated indices. The multi-label split is
random at the trace-instance level and is not stratified by website
combination, collection time, or client location.

## Expected NPZ Format

Each raw split is an NPZ file with:

- `X`: signed timestamp traces in seconds. The sign denotes packet direction
  and the absolute value denotes time;
- `y`: a float or integer multi-hot array of shape `(samples, classes)`.

The paper preprocessing first right-truncates or zero-pads each raw trace to
10,000 events. It then converts the trace into the six-channel slot-based
Robust Trace Representation using 20 ms slots over a 160 s window. The
resulting files contain `X` with shape `(samples, 6, 8000)` and the unchanged
`y` array. Their historical `hg8_*.npz` prefix is retained only for
compatibility with the released scripts and checkpoints.

```text
datasets/
  closed_2tab/
    train.npz
    valid.npz
    test.npz
    split_indices.npz
    split_metadata.json
    hg8_train.npz
    hg8_valid.npz
    hg8_test.npz
  ...
  open_5tab/
    ...
```

## Real-World Deployment Data

The controlled Tor deployment produces substantially larger trace files and
security-sensitive collection metadata. This repository provides the capture,
audit, and conversion code, but does not distribute raw deployment traces,
browser profiles, Tor keys, bridge descriptors, IP addresses, credentials, or
cloud state.

## ARES Redistribution

The separate Hugging Face dataset repository contains the derived NPZ
partitions, integer split indices, and metadata. These files originate from
the official ARES release; retain its original terms and citation when using
or redistributing them.
