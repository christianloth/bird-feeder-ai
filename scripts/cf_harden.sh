#!/usr/bin/env bash
# Cloudflare zone hardening for the public site (loth.me).
#
# Flips on the protections that the official CLIs (wrangler/flarectl) can't set:
#   1. Always Use HTTPS   — redirect any plaintext HTTP to HTTPS at the edge
#   2. HSTS               — tell browsers to never use HTTP for this host again
#   3. Rate limit /api/*  — block abusive clients before they reach the Pi
#
# Idempotent: safe to re-run. Steps 1-2 need a token with "Zone Settings: Edit";
# step 3 also needs "Zone WAF: Edit". Step 3 fails gracefully if not permitted.
#
# Create the token at: dash.cloudflare.com -> My Profile -> API Tokens -> Create Token
#   Permissions: Zone > Zone Settings > Edit   (and Zone > WAF > Edit for rate limiting)
#   Zone Resources: Include > Specific zone > loth.me
#
# Run WITHOUT putting the token in your shell history / this chat:
#   echo -n 'PASTE_TOKEN_HERE' > ~/.cf_token && chmod 600 ~/.cf_token   # (in your own editor ideally)
#   CF_API_TOKEN="$(cat ~/.cf_token)" bash scripts/cf_harden.sh
#   rm -f ~/.cf_token
set -euo pipefail

ZONE_NAME="${CF_ZONE:-loth.me}"
API="https://api.cloudflare.com/client/v4"
: "${CF_API_TOKEN:?Set CF_API_TOKEN — see the header of this script for how to create one}"

AUTH=(-H "Authorization: Bearer ${CF_API_TOKEN}" -H "Content-Type: application/json")

# Print OK / FAILED(+errors) from a Cloudflare API JSON response, tolerant of junk.
result() {
  printf '%s' "$1" | python3 -c '
import sys, json
try:
    d = json.load(sys.stdin)
    if d.get("success"):
        print("OK")
    else:
        print("FAILED:", json.dumps(d.get("errors")))
except Exception:
    print("FAILED: non-JSON response (check token / network)")'
}

echo "Resolving zone ${ZONE_NAME} ..."
ZRESP="$(curl -sS "${AUTH[@]}" "${API}/zones?name=${ZONE_NAME}")"
ZID="$(printf '%s' "$ZRESP" | python3 -c 'import sys,json
r=json.load(sys.stdin)
print(r["result"][0]["id"] if r.get("result") else "")' 2>/dev/null || true)"
if [ -z "${ZID}" ]; then
  echo "ERROR: could not resolve zone id. API said:"; printf '%s\n' "$ZRESP"; exit 1
fi
echo "  zone id: ${ZID}"

echo "[1/3] Always Use HTTPS ..."
result "$(curl -sS -X PATCH "${AUTH[@]}" \
  "${API}/zones/${ZID}/settings/always_use_https" --data '{"value":"on"}')"

echo "[2/3] HSTS (max-age 1y; no preload / no includeSubDomains — reversible) ..."
result "$(curl -sS -X PATCH "${AUTH[@]}" \
  "${API}/zones/${ZID}/settings/security_header" \
  --data '{"value":{"strict_transport_security":{"enabled":true,"max_age":31536000,"include_subdomains":false,"nosniff":true,"preload":false}}}')"

echo "[3/3] Rate limit: block an IP doing >100 req / 10s to /api/* ..."
echo "      (NOTE: this REPLACES the zone's http_ratelimit ruleset — fine if you have no other rate rules)"
RL_RULES='{"rules":[{
  "action":"block",
  "expression":"(starts_with(http.request.uri.path, \"/api/\"))",
  "description":"Throttle /api/* per client IP",
  "ratelimit":{
    "characteristics":["ip.src","cf.colo.id"],
    "period":10,
    "requests_per_period":100,
    "mitigation_timeout":60
  }
}]}'
result "$(curl -sS -X PUT "${AUTH[@]}" \
  "${API}/zones/${ZID}/rulesets/phases/http_ratelimit/entrypoint" --data "${RL_RULES}")"

echo
echo "Done. Verify HSTS is live:"
echo "  curl -sI https://feedercam.loth.me/ | grep -i strict-transport-security"
