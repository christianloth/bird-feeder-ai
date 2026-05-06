#!/usr/bin/env bash
# Start the bird detection pipeline as a detached process group.
# Child processes can be cleanly terminated via scripts/stop_pipeline.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_DIR="$ROOT_DIR/run"
LOG_FILE="$ROOT_DIR/pipeline.log"
PID_FILE="$RUN_DIR/pipeline.pid"

mkdir -p "$RUN_DIR"

if [[ -f "$PID_FILE" ]]; then
    existing_pid="$(cat "$PID_FILE")"
    if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
        echo "Pipeline already running (PID $existing_pid). Use stop_pipeline.sh first." >&2
        exit 1
    fi
    rm -f "$PID_FILE"
fi

cd "$ROOT_DIR"

# setsid puts the python process in a new session (PGID == PID), so all
# child processes (e.g., multiprocessing workers) can be killed as a group.
setsid "$ROOT_DIR/.venv/bin/python" -m src.pipeline.pipeline \
    --mode hailo --day --log-level INFO \
    >>"$LOG_FILE" 2>&1 </dev/null &

pid=$!
echo "$pid" >"$PID_FILE"

sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
    echo "Pipeline failed to start. See $LOG_FILE" >&2
    rm -f "$PID_FILE"
    exit 1
fi

echo "Pipeline started (PID $pid). Logs: $LOG_FILE"
