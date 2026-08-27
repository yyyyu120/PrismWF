# Installation

## Supported Stack

PrismWF is implemented with PyTorch. The public artifact and checkpoint
reproduction environment use:

```text
Operating system  Ubuntu Linux
Python            3.10.20
PyTorch           2.4.1+cu121
CUDA runtime      12.1
cuDNN              9.1.0
NumPy             1.26.4
scikit-learn      1.7.2
timm               1.0.19
tqdm              4.68.1
GPU                NVIDIA A800-SXM4-80GB
```

An A800 is not required to execute the code, but training and the reported
latency measurements require an NVIDIA CUDA GPU. CPU execution is suitable for
feature extraction, metric calculation, and small smoke tests.

## Recommended Conda Installation

Install Miniconda or Anaconda, clone the repository, and create the pinned
environment:

```bash
git clone https://github.com/yyyyu120/PrismWF.git
cd PrismWF
conda env create -f environment.yml
conda activate prismwf
python -m pip install -e . --no-deps
```

The `--no-deps` flag prevents pip from replacing the CUDA-enabled PyTorch
package installed by Conda.

## Pip Installation

For a Python 3.10 virtual environment, install the CUDA 12.1 PyTorch wheel
before installing PrismWF:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -e .
```

Use the official PyTorch selector when a different CUDA runtime is required.
Runs performed with a different PyTorch, CUDA, cuDNN, driver, or GPU stack may
not be bitwise identical to the paper environment.

## Verify the Environment

```bash
nvidia-smi
python - <<'PY'
import torch
import prismwf

print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("PrismWF import: OK")
PY
```

For a CUDA training machine, `torch.cuda.is_available()` must print `True`.
If it prints `False`, verify the NVIDIA driver and make sure the CUDA-enabled
PyTorch build, rather than a CPU-only build, is installed.

## Run the Tests

```bash
python -m pip install pytest ruff
pytest -q
```

The tests check the six-channel representation, fixed- and unknown-tab-count
metrics, and a PrismWF forward pass. They are smoke tests and do not replace
reproducing the complete paper experiments.

## First Training Run

After preparing the dataset as described in the main README, launch one
closed-world 2-tab run:

```bash
python train.py \
  --datasets closed_2tab \
  --checkpoint checkpoints/closed_2tab/seed2024.pt \
  --checkpoint-k 2 \
  --seed 2024
```

Training writes the checkpoint together with the run configuration, epoch
history, and any sampled training indices. Preserve these sidecar files when
publishing or transferring a checkpoint.
