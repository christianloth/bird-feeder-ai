#!/usr/bin/env bash
# Backs up the bird feeder database and detection images to Dropbox via rclone.
# Requires: rclone configured with a "dropbox" remote (run `rclone config` once).
#
# Database backup strategy (rolling, multi-version):
#   db/birds.db.gz                      - latest, refreshed every night
#   db/weekly/birds-YYYY-MM-DD.db.gz    - one snapshot every Sunday, newest 8 kept (~2 months)
#   db/quarantine/birds-corrupt-*.db.gz - snapshots that FAILED integrity_check
# A snapshot is uploaded only after `PRAGMA integrity_check` passes, so a
# corrupt database can never overwrite the known-good copies. Restore with
# scripts/restore_dropbox.sh. Detection images and rotated logs are copied
# additively (nothing is ever deleted on the remote).
#
# On any failure the script logs an ERROR, sends a Telegram alert, and exits
# non-zero (so `systemctl --failed` shows the unit too).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_SRC="${PROJECT_DIR}/db/birds.db"
DETECTIONS_SRC="${PROJECT_DIR}/detections"
REMOTE="dropbox:bird-feeder"
WEEKLY_KEEP=8
# Unpredictable name + 0600 perms so other local users can't read the DB copy
# (mktemp avoids the predictable /tmp/...$$ symlink/pre-creation race).
TMP_DB="$(mktemp "${TMPDIR:-/tmp}/birds_backup.XXXXXX.db")"
chmod 600 "${TMP_DB}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

NOTIFIED=0
notify_failure() {
    local msg="$1"
    log "ERROR: ${msg}"
    [ "${NOTIFIED}" = "1" ] && return 0
    NOTIFIED=1
    if ! python3 "${PROJECT_DIR}/scripts/notify_telegram.py" \
            "🔴 Bird Feeder backup FAILED ($(hostname)): ${msg}"; then
        log "WARN: could not send Telegram failure alert"
    fi
}
on_err() {
    local rc=$?
    notify_failure "aborted (exit ${rc}) at line ${BASH_LINENO[0]}: ${BASH_COMMAND}"
}
trap on_err ERR
trap 'rm -f "${TMP_DB}"' EXIT

log "Starting Dropbox backup..."

# --- Database: integrity-gated, rolling multi-version -------------------------
log "Snapshotting database..."
# Safe SQLite copy (works even while the pipeline is running with WAL mode).
sqlite3 "${DB_SRC}" ".backup '${TMP_DB}'"

log "Verifying snapshot integrity..."
set +e
integrity="$(sqlite3 "${TMP_DB}" 'PRAGMA integrity_check;' 2>&1 | head -1)"
set -e
if [ "${integrity}" != "ok" ]; then
    ts="$(date '+%Y-%m-%dT%H-%M-%S')"
    log "Snapshot is NOT ok — quarantining instead of overwriting good backups."
    gzip -c "${TMP_DB}" \
        | rclone rcat "${REMOTE}/db/quarantine/birds-corrupt-${ts}.db.gz" --log-level INFO \
        || log "WARN: could not upload quarantine copy"
    notify_failure "DB integrity_check failed: ${integrity}. Snapshot quarantined; good backups untouched."
    exit 1
fi
log "Integrity OK."

log "Uploading latest database (db/birds.db.gz)..."
gzip -c "${TMP_DB}" | rclone rcat "${REMOTE}/db/birds.db.gz" --log-level INFO

# Weekly snapshot on Sundays (date +%u == 7), then prune to newest $WEEKLY_KEEP.
if [ "$(date '+%u')" = "7" ]; then
    snap="birds-$(date '+%Y-%m-%d').db.gz"
    log "Sunday — writing weekly snapshot db/weekly/${snap}"
    gzip -c "${TMP_DB}" | rclone rcat "${REMOTE}/db/weekly/${snap}" --log-level INFO

    log "Pruning weekly snapshots (keeping newest ${WEEKLY_KEEP})..."
    # Names embed ISO dates, so a plain sort is chronological; head -n -KEEP
    # prints everything except the newest KEEP — i.e. the ones to delete.
    mapfile -t stale < <(
        rclone lsf "${REMOTE}/db/weekly/" --include "birds-*.db.gz" 2>/dev/null \
            | sort | head -n "-${WEEKLY_KEEP}"
    )
    for f in "${stale[@]:-}"; do
        [ -z "${f}" ] && continue
        log "Pruning old weekly snapshot ${f}"
        rclone deletefile "${REMOTE}/db/weekly/${f}" || log "WARN: could not delete ${f}"
    done
fi

rm -f "${TMP_DB}"

# --- Detection images (additive: skip existing, delete nothing on remote) -----
log "Syncing detection images..."
rclone copy "${DETECTIONS_SRC}" "${REMOTE}/detections" \
    --checksum \
    --transfers 4 \
    --exclude ".DS_Store" \
    --log-level INFO

# --- Rotated (compressed) log archives — skip the live .log files -------------
log "Syncing log archives..."
rclone copy "${PROJECT_DIR}/logs" "${REMOTE}/logs" \
    --include "*.log-*.gz" \
    --checksum \
    --log-level INFO

# --- Config / scripts (so a bare-metal restore has everything) ----------------
log "Backing up config files..."
rclone copyto "${HOME}/.config/rclone/rclone.conf" "${REMOTE}/config/rclone.conf" \
    --checksum --log-level INFO
for s in backup_dropbox.sh restore_dropbox.sh notify_telegram.py; do
    rclone copyto "${PROJECT_DIR}/scripts/${s}" "${REMOTE}/config/${s}" \
        --checksum --log-level INFO
done

log "Backup complete."
