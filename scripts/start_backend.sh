#!/usr/bin/env bash
# Start the FastAPI backend as a detached process group.
# Child processes (uvicorn reloader, multiprocessing workers) can be cleanly
# terminated via scripts/stop_backend.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$ROOT_DIR/run"
LOG_FILE="$ROOT_DIR/backend.log"
PID_FILE="$RUN_DIR/backend.pid"

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Backend already running (PID $existing_pid). Use stop_backend.sh first." >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

cd "$ROOT_DIR"

setsid "$ROOT_DIR/.venv/bin/python" -m src.backend.api --host 0.0.0.0 \
    >>"$LOG_FILE" 2>&1 </dev/null &

pid=$!
echo "$pid" >"$PID_FILE"

sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    echo "Backend failed to start. See $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
fi

echo "Backend started (PID $pid). Logs: $LOG_FILE"
