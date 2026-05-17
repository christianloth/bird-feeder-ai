#!/usr/bin/env bash
# Backs up the bird feeder database and detection images to Dropbox via rclone.
# Requires: rclone configured with a "dropbox" remote (run `rclone config` once).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DB_SRC="${PROJECT_DIR}/db/birds.db"
DETECTIONS_SRC="${PROJECT_DIR}/detections"
REMOTE="dropbox:bird-feeder"
TMP_DB="/tmp/birds_backup_$$.db"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Starting Dropbox backup..."

# Safe SQLite copy (works even while the pipeline is running with WAL mode)
log "Backing up database..."
sqlite3 "${DB_SRC}" ".backup '${TMP_DB}'"
rclone copyto "${TMP_DB}" "${REMOTE}/db/birds.db" \
    --checksum \
    --log-level INFO
rm -f "${TMP_DB}"

# Sync detection images (skip already-uploaded files, delete nothing on remote)
log "Syncing detection images..."
rclone copy "${DETECTIONS_SRC}" "${REMOTE}/detections" \
    --checksum \
    --transfers 4 \
    --log-level INFO

# Back up the rclone config and this script itself
log "Backing up config files..."
rclone copyto "${HOME}/.config/rclone/rclone.conf" "${REMOTE}/config/rclone.conf" \
    --checksum \
    --log-level INFO
rclone copyto "${PROJECT_DIR}/scripts/backup_dropbox.sh" "${REMOTE}/config/backup_dropbox.sh" \
    --checksum \
    --log-level INFO

log "Backup complete."
