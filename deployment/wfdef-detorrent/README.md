# DeTorrent WFDefProxy Integration

DeTorrent's public artifact provides an offline implementation of its padding
policy, but it does not include a WFDefProxy-compatible online transport.  The
PrismWF deployment therefore adapts the released DeTorrent generator and
padding logic into a custom transport for the `v2` branch of WFDefProxy.

This directory releases the integration used for the controlled Tor
deployment.  It does not redistribute DeTorrent model weights, calibration
data, Tor keys, bridge descriptors, browser profiles, or cloud credentials.

## Upstream Requirements

1. Clone the WFDefProxy `v2` branch from
   <https://github.com/websitefingerprinting/wfdef>.
2. Obtain the official DeTorrent artifact from
   <https://github.com/jkhollandjr/PETS_DeTorrent>.
   Check out commit `c08ad14c1e14539f22cf657dbb170524452e1283`.
3. Use Python 3.8 for the official DeTorrent generator environment and Go
   1.20 or later for WFDefProxy.

## Build the Custom Transport

From the root of a clean WFDefProxy `v2` checkout:

```bash
cp -R /path/to/PrismWF/deployment/wfdef-detorrent/transports/detorrent \
  transports/detorrent
git apply /path/to/PrismWF/deployment/wfdef-detorrent/register_detorrent.patch
go build -o build/obfs4proxy ./obfs4proxy
go test ./transports/detorrent
```

The registration patch adds `detorrent` to WFDefProxy's transport registry.
The transport retains WFDefProxy's authenticated framing and requests a
causal padding schedule from the generator service after observing each
completed time bin.

## Generator Reproduction and Service

The DeTorrent authors did not distribute the pretrained weight used in this
deployment. We reproduced the fold-4 generator checkpoint with the unmodified
`wf_preprocessing.py` and `wf_defense.py` files from the fixed commit above and
the official `be_dataset`. The resulting file follows the artifact's
conventional path `official_checkpoints/fold_4/wf_generator.pth`; the directory
name does not mean that the weight itself was released by the DeTorrent
authors.

The generator service loads this reproduced checkpoint and its calibration
histogram without modifying the released model architecture:

```bash
python detorrent_generator_service.py \
  --artifact-root /path/to/PETS_DeTorrent \
  --checkpoint /path/to/PETS_DeTorrent/official_checkpoints/fold_4/wf_generator.pth \
  --calibration-hist /path/to/PETS_DeTorrent/wf_preprocessed_data/lstm_train_hist_256_4.npy \
  --calibration-samples 1000 \
  --host 127.0.0.1 \
  --port 19991 \
  --seed 2024 \
  --audit-log /data/detorrent-generator-audit.jsonl
```

Both tab settings use the same generator architecture and reproduced
checkpoint; the generator is not retrained for either setting. The paper uses
padding budgets of `7000` for 2-tab sessions and `23000` for 5-tab sessions.
These values were calibrated on the collected real multi-tab traffic and are
not DeTorrent's original default parameters. A budget only scales the online
dummy scheduling volume. It is passed to both client and bridge transport
options. Example `torrc` fragments are provided in `torrc/`.

## End-to-End Check

`scripts/detorrent_online_smoke.sh` starts a local echo endpoint, generator,
client transport, and bridge transport.  It verifies payload integrity and
requires both generated dummy packets and WFDefProxy trace records. Supply the
local DeTorrent artifact, its Python interpreter, and the compiled WFDefProxy
binary explicitly:

```bash
DETORRENT_ARTIFACT_ROOT=/path/to/PETS_DeTorrent \
PYTHON_BIN=/path/to/detorrent-python \
PT_BINARY=/path/to/wfdef/build/obfs4proxy \
bash scripts/detorrent_online_smoke.sh
```

Set `RUN_DIR` to retain the smoke-test logs outside the default temporary
directory. `GENERATOR_CHECKPOINT`, `CALIBRATION_HIST`, and the three local
ports can also be overridden when needed.

## Scope

This is the authors' WFDefProxy adaptation of the released DeTorrent policy,
not an online implementation distributed by the DeTorrent authors.  The
deployment applies padding while a Tor connection is active; it does not
transform stored traces after collection.
