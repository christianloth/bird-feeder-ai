#!/usr/bin/env bash
# Restore the bird feeder database from a Dropbox backup made by
# backup_dropbox.sh. Always downloads to a staging area, decompresses, and runs
# PRAGMA integrity_check BEFORE touching the live database.
#
# Usage:
#   restore_dropbox.sh                  # list available snapshots
#   restore_dropbox.sh latest           # fetch + verify the nightly latest
#   restore_dropbox.sh 2026-06-01       # fetch + verify a weekly snapshot by date
#   restore_dropbox.sh <sel> --apply    # also swap it into db/birds.db (stops services)
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="dropbox:bird-feeder"
DB_DST="${PROJECT_DIR}/db/birds.db"
STAGE_DIR="${PROJECT_DIR}/db/restore"
SERVICES=(bird-feeder-pipeline.service bird-feeder-backend.service)

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

list_snapshots() {
    echo "Latest (nightly):"
    rclone lsf "${REMOTE}/db/" --include "birds.db.gz" 2>/dev/null \
        | sed 's/^birds.db.gz$/  latest/' || true
    echo
    echo "Weekly snapshots (oldest → newest):"
    if ! rclone lsf "${REMOTE}/db/weekly/" --include "birds-*.db.gz" 2>/dev/null \
            | sort | sed -E 's/^birds-(.*)\.db\.gz$/  \1/'; then
        echo "  (none)"
    fi
    echo
    echo "Restore with:  $0 <YYYY-MM-DD|latest> [--apply]"
}

sel="${1:-}"
apply=0
[ "${2:-}" = "--apply" ] && apply=1

if [ -z "${sel}" ] || [ "${sel}" = "list" ]; then
    list_snapshots
    exit 0
fi

if [ "${sel}" = "latest" ]; then
    remote_file="${REMOTE}/db/birds.db.gz"
    label="latest"
else
    remote_file="${REMOTE}/db/weekly/birds-${sel}.db.gz"
    label="${sel}"
fi

mkdir -p "${STAGE_DIR}"
staged_gz="${STAGE_DIR}/birds-${label}.db.gz"
staged_db="${STAGE_DIR}/birds-${label}.db"

log "Fetching ${remote_file} ..."
if ! rclone copyto "${remote_file}" "${staged_gz}" --log-level INFO; then
    log "ERROR: could not fetch ${remote_file}. Run '$0 list' to see what exists."
    exit 1
fi

log "Decompressing ..."
gzip -dc "${staged_gz}" > "${staged_db}"

log "Verifying integrity ..."
integrity="$(sqlite3 "${staged_db}" 'PRAGMA integrity_check;' 2>&1 | head -1)"
if [ "${integrity}" != "ok" ]; then
    log "ERROR: restored snapshot FAILED integrity_check (${integrity}). Not safe to use."
    exit 1
fi
log "Integrity OK. Verified database staged at: ${staged_db}"

if [ "${apply}" -ne 1 ]; then
    cat <<EOF

Not applied (staging only). To swap it in manually:
  sudo systemctl stop ${SERVICES[*]}
  cp "${DB_DST}" "${DB_DST}.bak-\$(date +%Y%m%d-%H%M%S)"   # keep current
  rm -f "${DB_DST}-wal" "${DB_DST}-shm"                    # drop stale WAL
  cp "${staged_db}" "${DB_DST}"
  sudo systemctl start ${SERVICES[*]}

Or re-run with --apply to do all of that automatically:
  $0 ${label} --apply
EOF
    exit 0
fi

# --- --apply: swap the verified snapshot into the live database ---------------
log "About to replace ${DB_DST} with the '${label}' snapshot."
read -r -p "Stop services, back up current DB, and swap? [y/N] " ans
case "${ans}" in
    y|Y) ;;
    *) log "Aborted; verified copy left at ${staged_db}"; exit 1 ;;
esac

log "Stopping services: ${SERVICES[*]}"
sudo systemctl stop "${SERVICES[@]}"

if [ -f "${DB_DST}" ]; then
    bak="${DB_DST}.bak-$(date +%Y%m%d-%H%M%S)"
    log "Backing up current database -> ${bak}{,-wal,-shm}"
    for ext in "" "-wal" "-shm"; do
        [ -f "${DB_DST}${ext}" ] && cp "${DB_DST}${ext}" "${bak}${ext}"
    done
fi

# A restored snapshot is self-contained; remove any stale WAL/SHM so SQLite
# does not replay old frames over the freshly installed database.
log "Installing restored database ..."
rm -f "${DB_DST}-wal" "${DB_DST}-shm"
cp "${staged_db}" "${DB_DST}"

log "Starting services: ${SERVICES[*]}"
sudo systemctl start "${SERVICES[@]}"

log "Restore complete — live DB is now the '${label}' snapshot."
log "Previous database preserved as ${DB_DST}.bak-*"
