#!/usr/bin/env bash
set -euo pipefail

for name in DETORRENT_ARTIFACT_ROOT PYTHON_BIN PT_BINARY; do
  if [[ -z "${!name:-}" ]]; then
    echo "Required environment variable is unset: $name" >&2
    exit 2
  fi
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
GENERATOR_SERVICE=${GENERATOR_SERVICE:-$SCRIPT_DIR/../detorrent_generator_service.py}
GENERATOR_CHECKPOINT=${GENERATOR_CHECKPOINT:-$DETORRENT_ARTIFACT_ROOT/official_checkpoints/fold_4/wf_generator.pth}
CALIBRATION_HIST=${CALIBRATION_HIST:-$DETORRENT_ARTIFACT_ROOT/wf_preprocessed_data/lstm_train_hist_256_4.npy}
RUN_DIR=${RUN_DIR:-${TMPDIR:-/tmp}/prismwf-detorrent-smoke}
GENERATOR_PORT="${GENERATOR_PORT:-19991}"
PT_PORT="${PT_PORT:-31080}"
ECHO_PORT="${ECHO_PORT:-31081}"

for executable in "$PYTHON_BIN" "$PT_BINARY"; do
  if [[ ! -x "$executable" ]]; then
    echo "Required executable is unavailable: $executable" >&2
    exit 2
  fi
done

for file in "$GENERATOR_SERVICE" "$GENERATOR_CHECKPOINT" "$CALIBRATION_HIST"; do
  if [[ ! -f "$file" ]]; then
    echo "Required file is unavailable: $file" >&2
    exit 2
  fi
done

if [[ -z "$RUN_DIR" || "$RUN_DIR" == "/" ]]; then
  echo "Refusing unsafe RUN_DIR: $RUN_DIR" >&2
  exit 2
fi

pids=()
cleanup() {
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT

rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR/server" "$RUN_DIR/client"

"$PYTHON_BIN" -u - "$ECHO_PORT" >"$RUN_DIR/echo.log" 2>&1 <<'PY' &
import socket
import sys
import threading

port = int(sys.argv[1])
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", port))
listener.listen()

def echo(connection):
    try:
        while True:
            payload = connection.recv(65536)
            if not payload:
                return
            connection.sendall(payload)
    finally:
        connection.close()

while True:
    connection, _ = listener.accept()
    threading.Thread(target=echo, args=(connection,), daemon=True).start()
PY
pids+=("$!")

"$PYTHON_BIN" -u "$GENERATOR_SERVICE" \
  --artifact-root "$DETORRENT_ARTIFACT_ROOT" \
  --checkpoint "$GENERATOR_CHECKPOINT" \
  --calibration-hist "$CALIBRATION_HIST" \
  --calibration-samples 1000 --host 127.0.0.1 --port "$GENERATOR_PORT" \
  --seed 2024 --audit-log "$RUN_DIR/generator_audit.jsonl" \
  >"$RUN_DIR/generator.log" 2>&1 &
pids+=("$!")

for _ in $(seq 1 60); do
  nc -z 127.0.0.1 "$GENERATOR_PORT" 2>/dev/null && break
  sleep 1
done

tail -f /dev/null | env \
  TOR_PT_MANAGED_TRANSPORT_VER=1 \
  TOR_PT_STATE_LOCATION="$RUN_DIR/server" \
  TOR_PT_SERVER_TRANSPORTS=detorrent \
  TOR_PT_SERVER_BINDADDR="detorrent-127.0.0.1:$PT_PORT" \
  TOR_PT_SERVER_TRANSPORT_OPTIONS="detorrent:budget=3000;detorrent:generator-addr=127.0.0.1:$GENERATOR_PORT" \
  TOR_PT_ORPORT="127.0.0.1:$ECHO_PORT" \
  "$PT_BINARY" -enableLogging=true -unsafeLogging=true -logLevel=DEBUG \
  >"$RUN_DIR/server.stdout" 2>&1 &
pids+=("$!")

for _ in $(seq 1 60); do
  grep -q '^SMETHOD detorrent ' "$RUN_DIR/server.stdout" 2>/dev/null && break
  sleep 0.25
done
grep -q '^SMETHOD detorrent ' "$RUN_DIR/server.stdout"

tail -f /dev/null | env \
  TOR_PT_MANAGED_TRANSPORT_VER=1 \
  TOR_PT_STATE_LOCATION="$RUN_DIR/client" \
  TOR_PT_CLIENT_TRANSPORTS=detorrent \
  "$PT_BINARY" -enableLogging=true -unsafeLogging=true -logLevel=DEBUG \
  >"$RUN_DIR/client.stdout" 2>&1 &
pids+=("$!")

for _ in $(seq 1 60); do
  grep -q '^CMETHOD detorrent ' "$RUN_DIR/client.stdout" 2>/dev/null && break
  sleep 0.25
done
grep -q '^CMETHOD detorrent ' "$RUN_DIR/client.stdout"

cert=$(sed -n 's/^SMETHOD detorrent .* ARGS:cert=\([^ ]*\)$/\1/p' "$RUN_DIR/server.stdout")
client_port=$(sed -n 's/^CMETHOD detorrent socks5 127\.0\.0\.1:\([0-9]*\)$/\1/p' "$RUN_DIR/client.stdout")
test -n "$cert"
test -n "$client_port"

"$PYTHON_BIN" - "$client_port" "$PT_PORT" "$GENERATOR_PORT" "$cert" <<'PY'
import hashlib
import socket
import struct
import sys
import time

client_port, pt_port, generator_port = map(int, sys.argv[1:4])
cert = sys.argv[4]
arguments = (
    f"cert={cert};budget=3000;generator-addr=127.0.0.1:{generator_port}"
).encode()
username = arguments[:255]
password = arguments[255:] or b"\x00"

connection = socket.create_connection(("127.0.0.1", client_port), 10)
connection.settimeout(30)
connection.sendall(b"\x05\x01\x02")
assert connection.recv(2) == b"\x05\x02"
connection.sendall(
    bytes([1, len(username)]) + username + bytes([len(password)]) + password
)
assert connection.recv(2) == b"\x01\x00"
target = b"127.0.0.1"
connection.sendall(
    b"\x05\x01\x00\x03" + bytes([len(target)]) + target + struct.pack("!H", pt_port)
)
reply = connection.recv(10)
assert len(reply) == 10 and reply[1] == 0, reply

payload = (b"detorrent-online-e2e-" * 900)[:18000]
connection.sendall(payload)
received = b""
while len(received) < len(payload):
    received += connection.recv(len(payload) - len(received))
assert received == payload
print(f"payload_bytes={len(payload)}")
print(f"sha256={hashlib.sha256(received).hexdigest()}")
time.sleep(3)
connection.close()
PY

sleep 2
trace_log="$RUN_DIR/client/obfs4proxy.log"
records=$(grep -c 'TRACE_LOG' "$trace_log" || true)
dummies=$(awk '/TRACE_LOG/ && $0 ~ / 0 -?[1-9][0-9]*$/ {count++} END {print count+0}' "$trace_log")
real=$(awk '/TRACE_LOG/ && $0 !~ / 0 -?[1-9][0-9]*$/ {count++} END {print count+0}' "$trace_log")
generated=$("$PYTHON_BIN" - "$RUN_DIR/generator_audit.jsonl" <<'PY'
import json
import sys

print(sum(json.loads(line).get("dummy_packets", 0) for line in open(sys.argv[1])))
PY
)
test "$records" -gt 0
test "$dummies" -gt 0
test "$generated" -gt 0

cat >"$RUN_DIR/smoke_summary.json" <<EOF
{
  "payload_bytes": 18000,
  "payload_integrity": true,
  "trace_records": $records,
  "dummy_records": $dummies,
  "other_records": $real,
  "generator_scheduled_dummy_packets": $generated,
  "generator_checkpoint": "fold_4/wf_generator.pth",
  "budget": 3000
}
EOF
cat "$RUN_DIR/smoke_summary.json"
