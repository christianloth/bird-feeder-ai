#!/usr/bin/env bash
# Stop the FastAPI backend and all of its child processes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PID_FILE="$ROOT_DIR/run/backend.pid"
TIMEOUT="${STOP_TIMEOUT:-15}"

stop_group() {
    local pid="$1"
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    if [[ -z "$pgid" ]]; then
        pgid="$pid"
    fi

    echo "Sending SIGTERM to process group $pgid..."
    kill -TERM -- "-$pgid" 2>/dev/null || true

    for ((i = 0; i < TIMEOUT; i++)); do
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "Backend stopped."
            return 0
        fi
        sleep 1
    done

    echo "Process group $pgid did not exit after ${TIMEOUT}s; sending SIGKILL."
    kill -KILL -- "-$pgid" 2>/dev/null || true
    sleep 1
}

if [[ ! -f "$PID_FILE" ]]; then
    echo "No pidfile at $PID_FILE. Searching for a running backend..."
    mapfile -t pids < <(pgrep -f 'src\.backend\.api' || true)
    if [[ ${#pids[@]} -eq 0 ]]; then
        echo "No backend process found."
        exit 0
    fi
    for pid in "${pids[@]}"; do
        stop_group "$pid"
    done
    exit 0
fi

pid="$(cat "$PID_FILE")"
if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    echo "Backend not running (stale pidfile)."
    rm -f "$PID_FILE"
    exit 0
fi

stop_group "$pid"
rm -f "$PID_FILE"
