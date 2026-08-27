# Controlled Tor Deployment

PrismWF's controlled deployment collects concurrent Tor Browser sessions over
real Tor circuits and records them through WFDefProxy. The public repository
excludes Tor keys, bridge descriptors, browser profiles, cloud addresses, and
credentials.

## Components

- Tor Browser and a compatible geckodriver;
- Selenium (`pip install -e '.[collection]'`);
- the official WFDefProxy release: https://github.com/websitefingerprinting/wfdef;
- one isolated SOCKS port, control port, control cookie, PT log, Tor data
  directory, and browser profile copy per parallel lane.

WFDefProxy must be built and configured from its own release. Select its null
transport for undefended collection. For DeTorrent collection, apply the
custom transport released in
[`deployment/wfdef-detorrent`](../deployment/wfdef-detorrent). The
collector reads WFDefProxy `[TRACE_LOG]` records; it does not synthesize
defended traces offline.

## Manifest Collection

Copy `deployment/collection/lanes.example.json` outside the repository and
replace its placeholder paths with local lane paths. Never commit that private
lane file.

```bash
python deployment/collection/run_manifest.py \
  deployment/collection/manifest.example.jsonl \
  --lanes /private/path/lanes.json \
  --output-root /data/prismwf-collection \
  --browser-binary /path/to/tor-browser/Browser/firefox \
  --default-profile /path/to/profile.default \
  --geckodriver /path/to/geckodriver \
  --capture-seconds 160
```

Each manifest row defines the requested URLs, labels, split, optional
inter-tab delays, and defense metadata. `run_manifest.py` assigns traces to
isolated lanes; `collect_multitab.py` creates all tabs before dispatching the
navigations and records the actual dispatch timestamps.

The paper collections retain a 160-second observation window and allow a
5-second tail to drain in-flight traffic before the session is closed. Model
features include only events within the 160-second observation window. Both
DeTorrent settings use the same generator checkpoint reproduced with the
unmodified official artifact and dataset. The 2-tab and 5-tab transports use
dummy-cell budgets of 7,000 and 23,000, respectively. These calibrated budgets
only scale online dummy scheduling and do not retrain the generator.

## Collection Arms

All four reported datasets use live Tor Browser sessions, the same strict
536-byte WFDefProxy recording pipeline, and 3--10-second delays between
consecutive page loads. The 2-tab Null and DeTorrent arms each contain about
20,000 sessions and originate from the same source manifest, which aligns the
planned website pair, tab order, and inter-tab delay. They were collected at
different times and are not counterfactual recordings of the same visits.

The 5-tab arms use the same online observation pipeline but different
composition protocols. The Null dataset covers 3,327 ARES-style website
combinations. The DeTorrent dataset contains about 20,000 sessions drawn from
about 1,000 combinations. Results therefore compare attacks within each
dataset under identical partitions; cross-condition 5-tab differences should
not be interpreted as causal estimates of the defense effect.

## Audit and Conversion

Build an auditable clean index without modifying raw traces:

```bash
python deployment/collection/prepare_clean_dataset.py \
  /data/prismwf-collection/2tab \
  --sites /data/sites.csv \
  --tab-count 2 \
  --output-dir /data/prismwf-clean-index
```

Convert fixed-size WFDefProxy byte records back into Tor-cell events and
generate the PrismWF six-channel Robust Trace Representation and ARES MTAF
features:

```bash
python deployment/collection/trace_csv_to_features.py \
  --index /data/prismwf-clean-index/clean_index.jsonl \
  --output-dir datasets/real_null_2tab \
  --interval-ms 20 \
  --max-slots 8000 \
  --frame-bytes 536 \
  --cell-bytes 536
```

The conversion preserves a per-direction byte remainder and checks cell
conservation. Events after the 160-second observation window, including any
records captured during the 5-second drain tail, are excluded from the
real-deployment feature files.

## Scope

This is a controlled page-loading deployment, not a comprehensive evaluation
of authenticated SPA interaction, indefinite scrolling, or prolonged media
playback. Preserve that limitation when describing the released data.
