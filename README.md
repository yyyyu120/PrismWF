# PrismWF: A Multi-Granularity Patch-Based Transformer for Robust Website Fingerprinting Attack

> **Accepted by IEEE Transactions on Dependable and Secure Computing (TDSC), 2026.**

[Paper](https://doi.org/10.1109/TDSC.2026.3746307) | [Checkpoints](https://huggingface.co/yuuu94/PrismWF) | [Dataset splits](https://huggingface.co/datasets/yuuu94/PrismWF-ARES-Splits) | [Installation](docs/INSTALLATION.md)

This repository is the official PyTorch implementation of **PrismWF**, a
multi-granularity model for identifying concurrently visited websites from
mixed Tor traffic.

## Highlights

- six-channel slot-based Robust Trace Representation;
- multi-branch CNN extraction at different temporal granularities;
- fine-to-coarse attention and router-token interaction;
- fixed-tab, mixed-tab, and unknown-tab-count evaluation;
- deterministic multi-seed training, statistical analysis, and efficiency
  profiling;
- real Tor collection tools and an online DeTorrent integration for
  WFDefProxy.

## Quick Start

Create the pinned environment:

```bash
conda env create -f environment.yml
conda activate prismwf
python -m pip install -e . --no-deps
```

Download the prepared ARES train/validation/test partitions:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="yuuu94/PrismWF-ARES-Splits",
    repo_type="dataset",
    local_dir="datasets",
)
```

Run one complete feature-extraction, training, and evaluation pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_one_fixed_k.sh closed_2tab 2 2024
```

The default configuration uses 10,000 raw events, 20 ms slots, a 160 s
observation window, and 80 training epochs. See
[Installation](docs/INSTALLATION.md) and [Datasets](docs/DATASETS.md) for the
full setup.

## Pretrained Checkpoints

Seed-2024 checkpoints for closed-world and open-world 2--5-tab settings are
hosted on [Hugging Face](https://huggingface.co/yuuu94/PrismWF). Each checkpoint
includes its run configuration, training history, and SHA-256 checksum.

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="yuuu94/PrismWF",
    filename="checkpoints/closed_2tab/seed2024.pt",
)
```

## Reproduce the Experiments

| Experiment | Command |
|---|---|
| Closed/open 2--5-tab, three seeds | `DEVICE=cuda bash scripts/run_fixed_k.sh` |
| Mixed-tab and unknown-`K` | `SCENARIO=closed DEVICE=cuda bash scripts/run_mixed_unknown_k.sh` |
| Architecture ablation | `DEVICE=cuda bash scripts/run_architecture_ablation.sh` |
| Feature ablation | `DEVICE=cuda bash scripts/run_feature_ablation.sh` |
| Attention-block depth | `DEVICE=cuda bash scripts/run_depth_ablation.sh` |
| Loading-window and slot sensitivity | `DEVICE=cuda bash scripts/run_sensitivity.sh` |
| Simulated defense datasets | `DEVICE=cuda bash scripts/run_defense_datasets.sh` |
| Network perturbation robustness | `DEVICE=cuda bash scripts/run_network_robustness.sh` |

Training records the full run configuration, epoch history, validation-based
checkpoint selection, and sampled mixed-training indices. Evaluation writes
machine-readable metrics and prediction arrays under `results/`.

## Data and Documentation

The prepared 18.23 GiB ARES release contains exact train/validation/test
partitions and split indices for every closed-world and open-world 2--5-tab
setting. To reconstruct these partitions from the original Zenodo archives
instead, run:

```bash
bash scripts/download_ares_multitab.sh
```

Detailed documentation:

- [Installation and environment](docs/INSTALLATION.md)
- [Datasets and preprocessing](docs/DATASETS.md)
- [Baseline protocols](docs/BASELINES.md)
- [Reproducibility and metrics](docs/REPRODUCIBILITY.md)
- [Real-world Tor collection](docs/REAL_WORLD_COLLECTION.md)
- [Online DeTorrent integration](deployment/wfdef-detorrent)

## Citation

If PrismWF is useful in your research, please cite the TDSC article:

```bibtex
@ARTICLE{11745895,
  author  = {Pan, Yuhao and Xu, Wenchao and Huo, Fushuo and Wang, Haozhao and Wang, Xiucheng and Cheng, Nan},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  title   = {{PrismWF}: A Multi-Granularity Patch-Based Transformer for Robust Website Fingerprinting Attack},
  year    = {2026},
  pages   = {1--16},
  doi     = {10.1109/TDSC.2026.3746307}
}
```

## Responsible Use and License

PrismWF is a research prototype. Use it only on traffic and systems for which
you have authorization. The source code is released under the MIT License;
see [NOTICE](NOTICE) for third-party attribution.
