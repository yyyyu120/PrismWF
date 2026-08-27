#!/usr/bin/env python3
"""Collect one concurrent multi-tab Tor Browser trace."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


TRACE_RE = re.compile(r"\[TRACE_LOG\]\s+(\d+)\s+(-?\d+)\s+(-?\d+)")
ERROR_TITLE_RE = re.compile(
    r"access denied|attention required|captcha|error|forbidden|"
    r"checking your browser|client challenge|internal server|just a moment|"
    r"not found|problem loading|robot or human|not a robot|"
    r"security check|security verification|secure connection failed|"
    r"secure site not available|site unavailable|403|404|502|503|\u043d\u0435 \u0440\u043e\u0431\u043e\u0442",
    re.IGNORECASE,
)
ERROR_URL_RE = re.compile(
    r"/(?:bgn_verification|captcha|challenge(?:\.html)?|vpn-block)(?:[/?#]|$)|"
    r"/cdn-cgi/challenge-platform/|[?&](?:captcha|js_challenge|verifyCode)=",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect one 2-5 tab trace through a WFDefProxy transport."
    )
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--url", action="append", dest="urls", required=True)
    parser.add_argument("--defense", default="null (undefended)")
    parser.add_argument(
        "--defense-parameters-json",
        default="{}",
    )
    parser.add_argument("--capture-seconds", type=float, default=160.0)
    parser.add_argument("--tail-seconds", type=float, default=5.0)
    parser.add_argument("--warmup-seconds", type=float, default=3.0)
    parser.add_argument("--schedule-lead-seconds", type=float, default=2.0)
    parser.add_argument(
        "--inter-tab-delay-seconds",
        action="append",
        type=float,
        default=[],
        help="Delay after each navigation before dispatching the next tab.",
    )
    parser.add_argument("--newnym-wait-seconds", type=float, default=10.0)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--browser-binary", type=Path, required=True)
    parser.add_argument("--default-profile", type=Path, required=True)
    parser.add_argument("--geckodriver", type=Path, required=True)
    parser.add_argument("--pt-log", type=Path, required=True)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=19151)
    parser.add_argument("--control-cookie", type=Path, required=True)
    parser.add_argument("--socks-host", default="127.0.0.1")
    parser.add_argument("--socks-port", type=int, default=19150)
    parser.add_argument("--skip-tor-check", action="store_true")
    parser.add_argument("--skip-newnym", action="store_true")
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument("--min-packet-records", type=int, default=1000)
    parser.add_argument("--min-real-bytes", type=int, default=50000)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 2 <= len(args.urls) <= 5:
        raise ValueError("each mixed trace must contain 2 to 5 URLs")
    for url in args.urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"invalid HTTP(S) URL: {url}")
    for path in (args.browser_binary, args.default_profile, args.geckodriver,
                 args.pt_log, args.control_cookie):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.capture_seconds <= 0 or args.schedule_lead_seconds <= 0:
        raise ValueError("capture and scheduling durations must be positive")
    if args.inter_tab_delay_seconds and (
        len(args.inter_tab_delay_seconds) != len(args.urls) - 1
    ):
        raise ValueError(
            "inter-tab delays must be omitted or supplied once per tab gap"
        )
    if any(delay < 0 for delay in args.inter_tab_delay_seconds):
        raise ValueError("inter-tab delays must be non-negative")
    try:
        args.defense_parameters = json.loads(args.defense_parameters_json)
    except json.JSONDecodeError as exc:
        raise ValueError("defense parameters must be valid JSON") from exc
    if not isinstance(args.defense_parameters, dict):
        raise ValueError("defense parameters must decode to a JSON object")


def read_control_reply(stream) -> list[str]:
    lines: list[str] = []
    while True:
        raw = stream.readline()
        if not raw:
            raise RuntimeError("Tor control port closed the connection")
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        lines.append(line)
        if len(line) >= 4 and line[:3].isdigit() and line[3] == " ":
            return lines


def request_new_circuit(args: argparse.Namespace) -> list[str]:
    cookie = args.control_cookie.read_bytes().hex()
    replies: list[str] = []
    with socket.create_connection(
        (args.control_host, args.control_port), timeout=10
    ) as sock:
        stream = sock.makefile("rwb", buffering=0)
        for command in (f"AUTHENTICATE {cookie}", "SIGNAL NEWNYM", "QUIT"):
            stream.write((command + "\r\n").encode("ascii"))
            lines = read_control_reply(stream)
            replies.extend(lines)
            if not lines[-1].startswith("250"):
                raise RuntimeError(f"Tor control command failed: {lines}")
    return replies


def configure_browser(args: argparse.Namespace, profile: Path) -> Options:
    options = Options()
    options.binary_location = str(args.browser_binary)
    options.page_load_strategy = "none"
    options.add_argument("-headless")
    options.add_argument("-profile")
    options.add_argument(str(profile))

    preferences = {
        "network.proxy.type": 1,
        "network.proxy.socks": args.socks_host,
        "network.proxy.socks_port": args.socks_port,
        "network.proxy.socks_version": 5,
        "network.proxy.socks_remote_dns": True,
        "network.proxy.no_proxies_on": "",
        "network.proxy.failover_direct": False,
        "extensions.torlauncher.start_tor": False,
        "extensions.torlauncher.prompt_at_startup": False,
        "browser.startup.page": 0,
        "browser.startup.homepage": "about:blank",
        "browser.privatebrowsing.autostart": True,
        "browser.tabs.warnOnClose": False,
        "browser.shell.checkDefaultBrowser": False,
        "browser.aboutwelcome.enabled": False,
        "app.update.enabled": False,
        "extensions.update.enabled": False,
        "datareporting.healthreport.uploadEnabled": False,
        "datareporting.policy.dataSubmissionEnabled": False,
        "toolkit.telemetry.enabled": False,
        "dom.disable_open_during_load": False,
    }
    for key, value in preferences.items():
        options.set_preference(key, value)
    return options


def wait_for_tor_check(driver, timeout: float = 45.0) -> dict:
    url = "https://check.torproject.org/api/ip"
    driver.get(url)
    deadline = time.time() + timeout
    last_text = ""
    while time.time() < deadline:
        try:
            last_text = driver.find_element("tag name", "body").text.strip()
            try:
                payload = json.loads(last_text)
                if payload.get("IsTor") is True:
                    return payload
            except json.JSONDecodeError:
                # Tor Browser's built-in JSON viewer renders keys and values
                # as a tree instead of preserving the raw response body.
                if re.search(r"\bIsTor\s+true\b", last_text, re.IGNORECASE):
                    ip_match = re.search(
                        r"\bIP\s+[\"']?([0-9a-f:.]+)", last_text, re.IGNORECASE
                    )
                    return {
                        "IsTor": True,
                        "IP": ip_match.group(1) if ip_match else None,
                        "rendered_by": "Firefox JSON viewer",
                    }
        except (WebDriverException, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Tor Browser proxy check failed: {last_text[:200]}")


def schedule_tabs(driver, urls: list[str], target_ms: int, trace_id: str,
                  delays_seconds: list[float]) -> tuple[list[str], list[int]]:
    """Create blank tabs, then dispatch navigations on a fixed schedule."""
    handles = [driver.current_window_handle]
    for _ in urls[1:]:
        driver.switch_to.new_window("tab")
        handles.append(driver.current_window_handle)
    for handle in handles:
        driver.switch_to.window(handle)
        driver.get("about:blank")

    if not delays_seconds:
        delays_seconds = [0.0] * (len(urls) - 1)
    cumulative_delay_ms = 0
    dispatch_times: list[int] = []
    for index, (handle, url) in enumerate(zip(handles, urls)):
        if index:
            cumulative_delay_ms += round(delays_seconds[index - 1] * 1000)
        planned_dispatch_ms = target_ms + cumulative_delay_ms
        time.sleep(max(0.0, planned_dispatch_ms / 1000 - time.time()))
        driver.switch_to.window(handle)
        dispatched_ms = time.time_ns() // 1_000_000
        marker = {
            "trace_id": f"{trace_id}:{index}",
            "scheduled_ms": planned_dispatch_ms,
            "navigation_dispatch_ms": dispatched_ms,
        }
        driver.execute_script("window.name = arguments[0];", json.dumps(marker))
        driver.execute_script("window.location.replace(arguments[0]);", url)
        dispatch_times.append(dispatched_ms)
    return handles, dispatch_times


def inspect_tabs(driver, handles: list[str], requested_urls: list[str], out_dir: Path,
                 screenshots: bool) -> list[dict]:
    results: list[dict] = []
    for index, (handle, requested_url) in enumerate(zip(handles, requested_urls)):
        result = {"index": index, "requested_url": requested_url}
        try:
            driver.switch_to.window(handle)
            values = driver.execute_script(
                "return {url: location.href, title: document.title, "
                "ready_state: document.readyState, marker: window.name, "
                "text_length: document.body ? document.body.innerText.length : 0, "
                "image_count: document.images ? document.images.length : 0, "
                "body_text_preview: document.body ? "
                "document.body.innerText.slice(0, 1000) : ''};"
            )
            result.update(values)
            try:
                result["marker"] = json.loads(result.get("marker") or "{}")
            except json.JSONDecodeError:
                pass
            if screenshots:
                screenshot = out_dir / f"tab-{index}.png"
                driver.save_screenshot(str(screenshot))
                result["screenshot"] = screenshot.name
        except WebDriverException as exc:
            result["error"] = str(exc)
        results.append(result)
    return results


def validate_loaded_tabs(tabs: list[dict], expected_count: int) -> None:
    if len(tabs) != expected_count:
        raise RuntimeError(
            f"expected {expected_count} browser tabs, observed {len(tabs)}"
        )
    failures = []
    for tab in tabs:
        title = (tab.get("title") or "").strip()
        url = tab.get("url") or ""
        content_loaded = (
            int(tab.get("text_length") or 0) >= 50
            or int(tab.get("image_count") or 0) >= 1
        )
        if (
            tab.get("error")
            or not url.startswith(("http://", "https://"))
            or not title
            or ERROR_TITLE_RE.search(title)
            or ERROR_URL_RE.search(url)
            or not content_loaded
        ):
            failures.append(tab)
    if failures:
        raise RuntimeError(f"one or more tabs did not navigate: {failures}")


def extract_trace(log_path: Path, start_offset: int, end_offset: int,
                  start_ns: int, end_ns: int, out_dir: Path) -> dict:
    with log_path.open("rb") as source:
        source.seek(start_offset)
        raw = source.read(max(0, end_offset - start_offset)).decode(
            "utf-8", errors="replace"
        )

    rows: list[tuple[int, int, int, str]] = []
    for line in raw.splitlines():
        match = TRACE_RE.search(line)
        if not match:
            continue
        timestamp_ns, real_bytes, dummy_bytes = map(int, match.groups())
        if start_ns <= timestamp_ns <= end_ns:
            rows.append((timestamp_ns, real_bytes, dummy_bytes, line))

    # WFDefProxy logs events from concurrent goroutines, so adjacent records can
    # arrive a few microseconds out of order even though their timestamps are
    # authoritative.
    rows.sort(key=lambda row: row[0])

    with (out_dir / "trace.csv").open("w", encoding="ascii") as output:
        output.write("timestamp_ns,real_bytes,dummy_bytes\n")
        for timestamp_ns, real_bytes, dummy_bytes, _ in rows:
            output.write(f"{timestamp_ns},{real_bytes},{dummy_bytes}\n")
    with (out_dir / "trace.raw.log").open("w", encoding="utf-8") as output:
        for *_, line in rows:
            output.write(line + "\n")

    real_total = sum(abs(row[1]) for row in rows)
    dummy_total = sum(abs(row[2]) for row in rows)
    total = real_total + dummy_total
    return {
        "packet_records": len(rows),
        "real_bytes": real_total,
        "dummy_bytes": dummy_total,
        "total_bytes": total,
        "dummy_fraction": (dummy_total / total) if total else None,
        "outgoing_records": sum(1 for row in rows if row[1] >= 0),
        "incoming_records": sum(1 for row in rows if row[1] < 0),
        "first_timestamp_ns": rows[0][0] if rows else None,
        "last_timestamp_ns": rows[-1][0] if rows else None,
    }


def write_metadata(path: Path, metadata: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    validate_args(args)
    out_dir = args.output_root / f"{len(args.urls)}tab" / args.trace_id
    if out_dir.exists():
        raise FileExistsError(f"trace output already exists: {out_dir}")
    out_dir.mkdir(parents=True)

    metadata = {
        "status": "starting",
        "trace_id": args.trace_id,
        "tab_count": len(args.urls),
        "urls": args.urls,
        "defense": args.defense,
        "defense_parameters": args.defense_parameters,
        "transport_log_format": "WFDefProxy TRACE_LOG",
        "capture_seconds": args.capture_seconds,
        "tail_seconds": args.tail_seconds,
        "planned_inter_tab_delays_seconds": args.inter_tab_delay_seconds,
        "created_at_ns": time.time_ns(),
    }
    write_metadata(out_dir / "metadata.json", metadata)

    profile_root = Path(tempfile.mkdtemp(prefix=f"wfdef-{args.trace_id}-"))
    profile = profile_root / "profile"
    driver = None
    try:
        shutil.copytree(args.default_profile, profile)
        if args.skip_newnym:
            metadata["tor_control_replies"] = {
                "newnym_skipped": True,
                "scope": "prewarmed worker batch",
            }
        else:
            metadata["tor_control_replies"] = request_new_circuit(args)
            time.sleep(args.newnym_wait_seconds)

        os.environ["TOR_SKIP_LAUNCH"] = "1"
        os.environ["TOR_SOCKS_PORT"] = str(args.socks_port)
        os.environ["TOR_CONTROL_PORT"] = str(args.control_port)
        os.environ["TOR_CONTROL_COOKIE_AUTH_FILE"] = str(args.control_cookie)
        service = Service(
            executable_path=str(args.geckodriver),
            log_output=str(out_dir / "geckodriver.log"),
        )
        driver = webdriver.Firefox(
            service=service, options=configure_browser(args, profile)
        )
        metadata["browser_capabilities"] = {
            key: driver.capabilities.get(key)
            for key in ("browserName", "browserVersion", "platformName")
        }
        if not args.skip_tor_check:
            metadata["tor_check"] = wait_for_tor_check(driver)
        else:
            metadata["tor_check"] = {
                "scope": "worker startup preflight",
                "per_trace_check_skipped": True,
            }

        driver.get("about:blank")
        time.sleep(args.warmup_seconds)
        start_offset = args.pt_log.stat().st_size
        target_ms = int((time.time() + args.schedule_lead_seconds) * 1000)
        handles, dispatch_times = schedule_tabs(
            driver,
            args.urls,
            target_ms,
            args.trace_id,
            args.inter_tab_delay_seconds,
        )
        metadata["scheduled_navigation_ms"] = target_ms
        metadata["navigation_dispatch_times_ms"] = dispatch_times
        metadata["navigation_dispatch_intervals_ms"] = [
            right - left
            for left, right in zip(dispatch_times, dispatch_times[1:])
        ]
        metadata["navigation_dispatch_skew_ms"] = (
            max(dispatch_times) - min(dispatch_times)
        )
        metadata["pt_log_start_offset"] = start_offset
        write_metadata(out_dir / "metadata.json", metadata)

        capture_end = target_ms / 1000 + args.capture_seconds
        time.sleep(max(0.0, capture_end - time.time()))
        metadata["tabs"] = inspect_tabs(
            driver, handles, args.urls, out_dir, args.screenshots
        )
        validate_loaded_tabs(metadata["tabs"], len(args.urls))
        metadata["capture_end_ns"] = time.time_ns()
        time.sleep(args.tail_seconds)
        end_ns = time.time_ns()
        end_offset = args.pt_log.stat().st_size

        metadata["trace_window_start_ns"] = target_ms * 1_000_000 - 500_000_000
        metadata["trace_window_end_ns"] = end_ns
        metadata["pt_log_end_offset"] = end_offset
        metadata["traffic"] = extract_trace(
            args.pt_log,
            start_offset,
            end_offset,
            metadata["trace_window_start_ns"],
            end_ns,
            out_dir,
        )
        if metadata["traffic"]["packet_records"] < args.min_packet_records:
            raise RuntimeError(
                "trace has too few packet records: "
                f"{metadata['traffic']['packet_records']} < "
                f"{args.min_packet_records}"
            )
        if metadata["traffic"]["real_bytes"] < args.min_real_bytes:
            raise RuntimeError(
                "trace has too few real bytes: "
                f"{metadata['traffic']['real_bytes']} < {args.min_real_bytes}"
            )
        metadata["status"] = "complete"
        write_metadata(out_dir / "metadata.json", metadata)
        print(json.dumps({"output": str(out_dir), **metadata["traffic"]}, indent=2))
        return 0
    except Exception as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        write_metadata(out_dir / "metadata.json", metadata)
        raise
    finally:
        if driver is not None:
            try:
                driver.quit()
            except WebDriverException:
                pass
        shutil.rmtree(profile_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"collection failed: {error}", file=sys.stderr)
        raise
