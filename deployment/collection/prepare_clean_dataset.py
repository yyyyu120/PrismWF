#!/usr/bin/env python3
"""Build an auditable training index without modifying raw WFDef traces."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STRONG_INVALID_TITLE_RE = re.compile(
    r"client challenge|attention required|access denied|captcha|"
    r"just a moment|problem loading page|robot or human|security check|"
    r"secure connection failed|site unavailable|^403(?:\s|$)|^404(?:\s|$)|"
    r"^502(?:\s|$)|^503(?:\s|$)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_root", type=Path)
    parser.add_argument("--sites", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tab-count", type=int, choices=[2, 3, 4, 5], required=True)
    parser.add_argument(
        "--queue-snapshot",
        type=Path,
        help="Optional JSONL snapshot; only queue-complete trace IDs are considered.",
    )
    parser.add_argument(
        "--exclude-domain",
        action="append",
        default=[],
        help="Drop every trace containing this requested domain (repeatable).",
    )
    parser.add_argument("--low-content-threshold", type=int, default=50)
    return parser.parse_args()


def read_sites(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    sites = {int(row["label"]): row for row in rows}
    if len(sites) != len(rows):
        raise RuntimeError("site file contains duplicate labels")
    return sites


def read_queue_snapshot(path: Path | None) -> tuple[set[str] | None, Counter[str]]:
    if path is None:
        return None, Counter()
    statuses: dict[str, str] = {}
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            trace_id = item["trace_id"]
            if trace_id in statuses:
                raise RuntimeError(
                    f"duplicate queue trace_id {trace_id!r} at line {line_number}"
                )
            statuses[trace_id] = item["status"]
    return (
        {trace_id for trace_id, status in statuses.items() if status == "complete"},
        Counter(statuses.values()),
    )


def tab_invalid_reason(tab: dict[str, Any]) -> str | None:
    title = str(tab.get("title") or "").strip()
    url = str(tab.get("url") or "").strip()
    if not url or url == "about:blank":
        return "blank_final_url"
    if not title:
        return "blank_title"
    if STRONG_INVALID_TITLE_RE.search(title):
        return "challenge_or_browser_error_title"
    return None


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> int:
    args = parse_args()
    sites = read_sites(args.sites)
    authoritative_ids, queue_statuses = read_queue_snapshot(args.queue_snapshot)
    excluded_domains = set(args.exclude_domain)

    metadata_by_id: dict[str, tuple[Path, dict[str, Any]]] = {}
    duplicate_ids: list[str] = []
    raw_statuses: Counter[str] = Counter()
    for metadata_path in sorted(args.trace_root.glob("*/metadata.json")):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw_statuses["unreadable"] += 1
            continue
        status = str(metadata.get("status") or "missing")
        trace_id = str(metadata.get("trace_id") or metadata_path.parent.name)
        if metadata_path.parent.name != trace_id:
            raw_statuses[f"archived_{status}"] += 1
            continue
        raw_statuses[status] += 1
        if status != "complete":
            continue
        if authoritative_ids is not None and trace_id not in authoritative_ids:
            continue
        if trace_id in metadata_by_id:
            duplicate_ids.append(trace_id)
            continue
        metadata_by_id[trace_id] = (metadata_path, metadata)

    if duplicate_ids:
        raise RuntimeError(f"duplicate complete trace IDs: {duplicate_ids[:10]}")

    observed_labels: set[int] = set()
    for _, metadata in metadata_by_id.values():
        observed_labels.update(int(label) for label in metadata.get("manifest", {}).get("labels", []))
    excluded_labels = {
        label
        for label in observed_labels
        if sites.get(label, {}).get("domain") in excluded_domains
    }
    retained_labels = sorted(observed_labels - excluded_labels)
    label_map = {old: new for new, old in enumerate(retained_labels)}

    clean: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    exclusion_reasons: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    flag_counts: Counter[str] = Counter()

    for trace_id, (metadata_path, metadata) in sorted(metadata_by_id.items()):
        manifest = metadata.get("manifest") or {}
        domains = [str(domain) for domain in manifest.get("domains", [])]
        original_labels = [int(label) for label in manifest.get("labels", [])]
        traffic = metadata.get("traffic") or {}
        tabs = metadata.get("tabs") or []
        reasons: list[str] = []
        invalid_tabs: list[dict[str, Any]] = []

        for domain in domains:
            if domain in excluded_domains:
                reasons.append(f"excluded_domain:{domain}")
        for tab in tabs:
            reason = tab_invalid_reason(tab)
            if reason:
                reasons.append(reason)
                invalid_tabs.append({
                    "index": tab.get("index"),
                    "requested_url": tab.get("requested_url"),
                    "final_url": tab.get("url"),
                    "title": tab.get("title"),
                    "reason": reason,
                })
        if (
            len(tabs) != args.tab_count
            or len(original_labels) != args.tab_count
            or len(domains) != args.tab_count
        ):
            reasons.append("invalid_tab_count_metadata")
        if int(traffic.get("real_bytes") or 0) <= 0:
            reasons.append("no_real_bytes")
        if int(traffic.get("dummy_bytes") or 0) <= 0:
            reasons.append("no_dummy_bytes")
        if int(traffic.get("packet_records") or 0) <= 0:
            reasons.append("no_packet_records")
        trace_path = metadata_path.parent / "trace.csv"
        if not trace_path.is_file() or trace_path.stat().st_size == 0:
            reasons.append("missing_trace_csv")

        low_content_tabs = [
            int(tab.get("index", index))
            for index, tab in enumerate(tabs)
            if int(tab.get("text_length") or 0) < args.low_content_threshold
        ]
        incomplete_tabs = [
            int(tab.get("index", index))
            for index, tab in enumerate(tabs)
            if tab.get("ready_state") not in {"complete", None}
        ]
        reasons = sorted(set(reasons))
        if reasons:
            for reason in reasons:
                exclusion_reasons[reason] += 1
            excluded.append({
                "trace_id": trace_id,
                "metadata_path": str(metadata_path),
                "trace_path": str(trace_path),
                "split": manifest.get("split"),
                "labels": original_labels,
                "domains": domains,
                "reasons": reasons,
                "invalid_tabs": invalid_tabs,
            })
            continue

        if low_content_tabs:
            flag_counts["low_content_traces_retained"] += 1
            flag_counts["low_content_tabs_retained"] += len(low_content_tabs)
        if incomplete_tabs:
            flag_counts["incomplete_ready_state_traces_retained"] += 1
            flag_counts["incomplete_ready_state_tabs_retained"] += len(incomplete_tabs)

        remapped_labels = [label_map[label] for label in original_labels]
        split = str(manifest.get("split") or "unknown")
        split_counts[split] += 1
        label_counts.update(remapped_labels)
        clean.append({
            "trace_id": trace_id,
            "metadata_path": str(metadata_path),
            "trace_path": str(trace_path),
            "split": split,
            "labels": remapped_labels,
            "original_labels": original_labels,
            "domains": domains,
            "pair_id": manifest.get("pair_id"),
            "repetition": manifest.get("repetition"),
            "low_content_tabs": low_content_tabs,
            "incomplete_ready_state_tabs": incomplete_tabs,
            "traffic": {
                key: traffic.get(key)
                for key in (
                    "packet_records", "real_bytes", "dummy_bytes", "total_bytes",
                    "dummy_fraction", "first_timestamp_ns", "last_timestamp_ns",
                )
            },
        })

    if authoritative_ids is not None:
        missing_complete_ids = sorted(authoritative_ids - set(metadata_by_id))
    else:
        missing_complete_ids = []

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "clean_index.jsonl", clean)
    write_jsonl(args.output_dir / "excluded_traces.jsonl", excluded)

    with (args.output_dir / "clean_sites.csv").open(
        "w", newline="", encoding="utf-8"
    ) as output:
        fieldnames = ("label", "original_label", "tranco_rank", "domain", "url")
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for old_label in retained_labels:
            site = sites[old_label]
            writer.writerow({
                "label": label_map[old_label],
                "original_label": old_label,
                "tranco_rank": site["tranco_rank"],
                "domain": site["domain"],
                "url": site["url"],
            })

    summary = {
        "trace_root": str(args.trace_root),
        "tab_count": args.tab_count,
        "queue_snapshot": str(args.queue_snapshot) if args.queue_snapshot else None,
        "queue_statuses": dict(sorted(queue_statuses.items())),
        "raw_metadata_statuses": dict(sorted(raw_statuses.items())),
        "authoritative_complete": (
            len(authoritative_ids) if authoritative_ids is not None else None
        ),
        "complete_metadata_found": len(metadata_by_id),
        "missing_authoritative_complete": len(missing_complete_ids),
        "missing_authoritative_complete_ids": missing_complete_ids,
        "clean_traces": len(clean),
        "excluded_traces": len(excluded),
        "exclusion_reasons": dict(sorted(exclusion_reasons.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "retained_flags": dict(sorted(flag_counts.items())),
        "observed_original_labels": sorted(observed_labels),
        "excluded_original_labels": sorted(excluded_labels),
        "retained_classes": len(retained_labels),
        "label_map": {str(old): new for old, new in label_map.items()},
        "per_class_trace_occurrences": {
            str(label): label_counts[label] for label in sorted(label_counts)
        },
        "per_class_min": min(label_counts.values(), default=0),
        "per_class_max": max(label_counts.values(), default=0),
    }
    (args.output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if not missing_complete_ids else 2


if __name__ == "__main__":
    raise SystemExit(main())
