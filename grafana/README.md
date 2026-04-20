# Grafana dashboard

A Grafana stack for visualising the bird feeder database, fronted by Caddy with basic auth so it can be safely exposed via ngrok.

## Architecture

```
[browser] ──▶ Caddy :80 (basic auth)
                ├── /grafana/*  ──▶ grafana:3000
                └── /*          ──▶ host FastAPI :8000
```

- **Same origin** — no CORS headaches; image URLs in the dashboard use relative paths like `/api/detections/<id>/crop`.
- **One password** at the Caddy gate covers everything inside (Grafana runs in anonymous-viewer mode so you aren't prompted twice).
- SQLite DB is bind-mounted **read-only** into the Grafana container.

## One-time setup

1. Generate a bcrypt hash for the Caddy basic-auth password:

   ```bash
   docker run --rm caddy:2 caddy hash-password --plaintext 'prettybirds'
   ```

2. Copy `.env.example` to `.env` and paste the hash. **Escape every `$` as `$$`** so docker-compose doesn't try to interpolate it:

   ```bash
   cp grafana/.env.example grafana/.env
   # edit grafana/.env, set CADDY_BASIC_AUTH_HASH to the $$2a$$... string
   ```

3. Start the stack:

   ```bash
   cd grafana
   docker compose --env-file .env up -d
   ```

4. Visit `http://<pi-ip>/grafana/` — log in with the Caddy credentials (admin / prettybirds). The dashboard is provisioned automatically under the "Bird Feeder" folder.

## Exposing externally via ngrok

Claim a free static domain at ngrok.com, then:

```bash
ngrok http --domain=your-name.ngrok-free.app 80
```

Update `GF_ROOT_URL` in `.env` to `https://your-name.ngrok-free.app/grafana/` and restart: `docker compose up -d`.

Run ngrok as a service (recommended):

```bash
sudo tee /etc/systemd/system/ngrok.service <<'EOF'
[Unit]
Description=ngrok tunnel
After=network.target

[Service]
ExecStart=/usr/local/bin/ngrok http --domain=your-name.ngrok-free.app 80
Restart=always
User=pi

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl enable --now ngrok
```

## Editing dashboards

`allowUiUpdates: true` is set, so you can edit panels in the Grafana UI freely. To persist changes to git:

1. Dashboard settings → JSON Model → Copy to clipboard
2. Paste over `grafana/dashboards/bird-feeder.json`
3. Commit

Provisioning rescans every 30s, so saved file changes reload automatically.

## Panels included

- **At a glance** — detections in range / today / all-time, unique species, most common today, avg confidence
- **Activity over time** — bucketed bar chart (bucket selectable), stacked by top-N species, hour-of-day × day-of-week heatmap
- **Species breakdown** — top species table, donut, most-frequent-per-day winner, rare visitors
- **Quality & curation** — confidence histogram, false-positive rate, % reviewed, classifier-model breakdown
- **Weather correlations** — detections vs temperature overlay, detections bucketed by temperature / cloud cover
- **Detection gallery** — latest detection spotlight (Dalvany image panel), recent detections table with thumbnail crop + annotated frame cells

## Troubleshooting

- **No data**: confirm `db/birds.db` exists (`ls ../db`) and the bind mount in `docker-compose.yml` matches.
- **Images are broken**: confirm the FastAPI backend is running on :8000 and Caddy can reach `host.docker.internal:8000` (on Linux this works via the `extra_hosts: host-gateway` mapping).
- **`Login or register` loop**: clear the GF_ROOT_URL — if it doesn't match the URL you're hitting, asset paths 404.
- **Grafana plugin install fails on first boot**: the Pi may be rate-limited downloading from grafana.com; `docker compose logs grafana` will show it. Retry with `docker compose restart grafana`.
