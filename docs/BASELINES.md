# Baseline Protocol

The paper compares PrismWF with AWF, DF, Tik-Tok, Var-CNN, RF, BAPM, TMWF,
CountMamba, and ARES. We do not redistribute their implementations here.
Install each method from its upstream release and observe its license.

The principal upstream repositories are listed in
`configs/baseline_protocol.csv`. BAPM did not provide an official source-code
release; its reported benchmark values follow the comparison protocol stated
in the manuscript.

## Common Evaluation Rules

- Use the same released train, validation, and test partitions for every
  method.
- Preserve each baseline's released feature extractor and backbone.
- Replace only the final output layer and loss where required for multi-label
  prediction.
- Train the rerun configurations for 80 epochs.
- Select fixed-tab checkpoints by validation MAP@K.
- Select mixed-tab checkpoints by pooled validation MAP@5.
- Never tune thresholds or hyperparameters on a test split.
- Report seeds 2024, 2025, and 2026 as mean and sample standard deviation.

The public launch configurations record the effective training and evaluation
protocol for each method. The evaluation scripts generate metric JSON files
and prediction arrays locally, allowing users to reproduce the reported
comparisons without requiring those generated outputs to be redistributed.

`configs/baseline_protocol.csv` records the effective optimizer settings used
by the reruns. In particular, the released CountMamba parser declares a
`weight_decay` option, but its optimizer constructor does not pass that option
to `torch.optim.AdamW`; the effective PyTorch default is therefore 0.01, which
is the value recorded in the protocol file.
