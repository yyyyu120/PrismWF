# PrismWF: A Multi-Granularity Patch-Based Transformer for Robust Website Fingerprinting Attack

This repository contains the implementation and reproducibility tools for
PrismWF.

PrismWF is a research prototype for evaluating multi-tab website
fingerprinting attacks. Use it only on traffic and systems for which you have
authorization.

## Framework and Requirements

PrismWF is implemented in **PyTorch**. The public artifact and checkpoint
reproduction environment use Ubuntu Linux, Python 3.10.20, PyTorch 2.4.1 with
CUDA 12.1, and NVIDIA A800 GPUs. The A800 is the validation platform, not a
strict hardware requirement. Feature extraction and lightweight evaluation
utilities can run on CPU, while model training is intended for an NVIDIA CUDA
GPU. See [docs/INSTALLATION.md](docs/INSTALLATION.md) for complete setup and
verification instructions.

## Included

- the PrismWF model used in the paper;
- the six-channel slot-based Robust Trace Representation;
- deterministic fixed-tab and mixed-tab training;
- fixed-`K` and unknown-`K` evaluation;
- three-seed aggregation, paired bootstrap confidence intervals, and
  computational-efficiency profiling;
- exact run metadata and mixed-training sampling indices;
- sanitized controlled-Tor collection, audit, and trace-conversion tools;
- the custom WFDefProxy transport used for the online DeTorrent deployment.

Datasets, trained checkpoints, and third-party baseline implementations are
not bundled in the source archive. The partitioned ARES train/validation/test
files, exact split indices, and metadata are released separately on
[Hugging Face](https://huggingface.co/datasets/yuuu94/PrismWF-ARES-Splits).
See [docs/DATASETS.md](docs/DATASETS.md),
[docs/BASELINES.md](docs/BASELINES.md), and
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
The controlled deployment workflow is described in
[docs/REAL_WORLD_COLLECTION.md](docs/REAL_WORLD_COLLECTION.md).
The DeTorrent integration, parameters, and example Tor configurations are in
[deployment/wfdef-detorrent](deployment/wfdef-detorrent).

## Quick Start

Create the pinned Conda environment and install PrismWF:

```bash
conda env create -f environment.yml
conda activate prismwf
python -m pip install -e . --no-deps
```

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for pip installation,
environment verification, and test commands.

## Pretrained Checkpoints

Verified seed-2024 PrismWF checkpoints are available from the
[PrismWF Hugging Face repository](https://huggingface.co/yuuu94/PrismWF).
The release includes the closed-world and open-world 2--5-tab settings. Each
checkpoint is distributed with its exact run configuration, training history,
and SHA-256 checksum.

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="yuuu94/PrismWF",
    filename="checkpoints/closed_2tab/seed2024.pt",
)
```

The local `checkpoints/` path used by the commands below is created
automatically during training and is excluded from version control.

## Data Preparation

The complete partitioned ARES data used by PrismWF are available from the
[PrismWF ARES Data Splits](https://huggingface.co/datasets/yuuu94/PrismWF-ARES-Splits)
repository. The 18.23 GiB release contains `train.npz`, `valid.npz`, and
`test.npz` for every closed-world and open-world 2--5-tab setting, together
with the exact split indices and metadata. Download all settings with:

```python
from huggingface_hub import snapshot_download

patterns = [
    f"{scenario}_{tabs}tab/*"
    for scenario in ("closed", "open")
    for tabs in range(2, 6)
]
snapshot_download(
    repo_id="yuuu94/PrismWF-ARES-Splits",
    repo_type="dataset",
    local_dir="datasets",
    allow_patterns=patterns,
)
```

For independent reconstruction, download the original ARES archives from
Zenodo, verify their published checksums, and reproduce the same seed-2024
instance-level splits with:

```bash
bash scripts/download_ares_multitab.sh
```

This creates each split under `datasets/<dataset>/`:

```text
datasets/closed_2tab/
  train.npz
  valid.npz
  test.npz
```

Generate the six-channel representation:

```bash
for split in train valid test; do
  python extract_features.py \
    --input datasets/closed_2tab/${split}.npz \
    --output datasets/closed_2tab/hg8_${split}.npz
done
```

The defaults reproduce the paper protocol: 10,000 raw events, a 20 ms slot,
and a 160 s observation window (8,000 slots). The historical `hg8` filename
prefix is retained only for compatibility with the released scripts and
checkpoints.

## Fixed-Tab Training and Evaluation

```bash
python train.py \
  --datasets closed_2tab \
  --checkpoint checkpoints/closed_2tab/seed2024.pt \
  --checkpoint-k 2 \
  --seed 2024

python evaluate.py \
  --dataset closed_2tab \
  --checkpoint checkpoints/closed_2tab/seed2024.pt \
  --tabs 2 \
  --seed 2024 \
  --output results/closed_2tab/seed2024
```

To reproduce all closed-world and open-world fixed-tab settings with seeds
`2024`, `2025`, and `2026`, including automatic aggregation of the mean and
sample standard deviation, run:

```bash
DEVICE=cuda bash scripts/run_fixed_k.sh
```

Per-seed metrics are written to `results/<dataset>/seed<seed>.json`, and each
three-seed summary is written to `results/<dataset>/summary.json`.

## Mixed-Tab and Unknown-K Evaluation

Mixed-tab training pools 30% of each 2--5-tab training split. The checkpoint
is selected by pooled validation MAP@5.

```bash
python train.py \
  --datasets closed_2tab closed_3tab closed_4tab closed_5tab \
  --sample-ratio 0.3 \
  --checkpoint-k 5 \
  --checkpoint checkpoints/mixed/seed2024.pt \
  --seed 2024
```

Run `evaluate.py` on each validation and test split to save score arrays, then
select one global threshold using only the pooled validation predictions:

```bash
python tools/evaluate_unknown_k.py \
  --validation results/mixed/valid_2tab.npz results/mixed/valid_3tab.npz \
               results/mixed/valid_4tab.npz results/mixed/valid_5tab.npz \
  --test results/mixed/test_2tab.npz results/mixed/test_3tab.npz \
         results/mixed/test_4tab.npz results/mixed/test_5tab.npz \
  --output results/mixed/unknown_k.json
```

The evaluator reports micro-precision, micro-recall, micro-F1, exact-set
accuracy, and label-count MAE without using the true tab count at test time.

## Repository Layout

| Path | Purpose |
|---|---|
| `train.py`, `evaluate.py`, `extract_features.py` | Main training, evaluation, and feature-extraction entry points |
| `prismwf/` | PrismWF model, Robust Trace Representation, metrics, data loading, and training engine |
| `configs/` | Paper configuration and baseline protocol |
| `datasets/` | Local dataset staging area and download guidance |
| `scripts/` | End-to-end reproduction and experiment launchers |
| `deployment/collection/` | Tor traffic collection, auditing, and trace conversion |
| `deployment/wfdef-detorrent/` | Custom online DeTorrent integration for WFDefProxy |
| `tools/` | Dataset splitting, unknown-`K` evaluation, statistics, aggregation, and efficiency profiling |
| `docs/` | Installation, dataset, baseline, deployment, and reproducibility documentation |

## Acknowledgments

The dataset organization and parts of the experimental interface build on the
open-source ARES / Website Fingerprinting Library ecosystem. Their copyright
notice and MIT license are retained in `LICENSE`; see `NOTICE` for attribution.

## Citation

If you find PrismWF useful in your research, please cite:

```bibtex
@article{pan2026prismwf,
  title   = {PrismWF: A Multi-Granularity Patch-Based Transformer for Robust Website Fingerprinting Attack},
  author  = {Pan, Yuhao and Xu, Wenchao and Huo, Fushuo and Wang, Haozhao and Wang, Xiucheng and Cheng, Nan},
  journal = {arXiv preprint arXiv:2603.21117},
  year    = {2026}
}
```
