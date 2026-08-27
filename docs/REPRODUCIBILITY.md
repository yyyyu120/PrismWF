# Reproducibility

## Deterministic Settings

`prismwf.reproducibility.set_seed` fixes Python, NumPy, CPU and CUDA random
seeds and configures deterministic cuDNN execution. Every DataLoader receives
an explicit generator and deterministic worker seeds.

Determinism is expected for repeated runs on the same PyTorch, CUDA, cuDNN,
GPU, and driver stack. Bitwise identity is not guaranteed across different
hardware or library versions. The paper reports three independent seeds to
measure training variation rather than relying on one run. The exact software
and hardware stack is documented in [INSTALLATION.md](INSTALLATION.md).

## Run Records

For a checkpoint `seed2024.pt`, training creates:

- `seed2024.run.json`: all command-line settings;
- `seed2024.history.json`: epoch-level loss, validation MAP@K, and learning
  rate;
- `seed2024.sampling.json`: exact mixed-training sample indices when
  `--sample-ratio` is below one.

If the sampling file already exists, `train.py` reloads it and validates the
dataset list and seed before training. Preserve these files with checkpoints.

## Metrics

- AUC is macro-averaged ROC-AUC over website labels.
- P@K is the mean fraction of ground-truth labels in the top-K predictions.
- MAP@K follows the ARES protocol: the mean of P@1 through P@K.
- Unknown-K uses one global threshold selected by pooled-validation micro-F1.
- Sample standard deviation uses `n-1` in the denominator.
- Paired bootstrap analysis reports percentile-based 95% confidence intervals;
  it does not label bootstrap tail proportions as null-hypothesis p-values.

## Controlled Ablations

`--ablation no-router`, `no-router-no-cross`, and `single-granularity` retain
the full model's feature representation, optimization settings, attention
widths, and remaining convolution branches. Thus each comparison changes only
the named architectural component. `scripts/run_architecture_ablation.sh`
records the chosen variant in every run JSON.

## Benchmark-Compatible Tail Handling

The released WFlib/ARES-style implementation of the six-channel Robust Trace
Representation assigns all events remaining after the penultimate boundary to
the final slot. The default encoder preserves that behavior so existing
benchmark features and checkpoints remain exactly compatible. It is therefore
not a strict time truncation when a perturbed trace extends beyond the nominal
window. Pass `--strict-window` to feature extraction or robustness generation
to discard later events; results produced with that flag must be identified
separately and require retraining where training features also change.
