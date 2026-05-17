#!/usr/bin/env python3
"""
One-shot Dropbox OAuth2 helper for headless rclone setup.

Usage:
    python3 dropbox_auth.py <client_id> <client_secret>

Listens on 0.0.0.0:8765, handles the OAuth redirect from Dropbox,
exchanges the code for access+refresh tokens, and writes them into
~/.config/rclone/rclone.conf so rclone can use them immediately.
"""

import configparser
import json
import secrets
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

PORT = 8765
SCOPES = " ".join([
    "account_info.read",
    "files.content.read",
    "files.content.write",
    "files.metadata.read",
    "files.metadata.write",
])


def write_rclone_config(token_data: dict, client_id: str, client_secret: str) -> Path:
    config_path = Path.home() / ".config" / "rclone" / "rclone.conf"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    expiry = token_data.get("expires_in")
    expiry_str = "0001-01-01T00:00:00Z"
    if expiry:
        from datetime import timedelta
        exp_dt = datetime.now(timezone.utc) + timedelta(seconds=int(expiry))
        expiry_str = exp_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    token_json = json.dumps({
        "access_token": token_data["access_token"],
        "token_type": "bearer",
        "refresh_token": token_data.get("refresh_token", ""),
        "expiry": expiry_str,
    })

    cfg = configparser.ConfigParser()
    if config_path.exists():
        cfg.read(config_path)

    cfg["dropbox"] = {
        "type": "dropbox",
        "client_id": client_id,
        "client_secret": client_secret,
        "token": token_json,
    }

    with open(config_path, "w") as f:
        cfg.write(f)

    return config_path


class OAuthHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # silence request logs

    def _send(self, body: str, status: int = 200):
        encoded = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(encoded))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/":
            state = secrets.token_urlsafe(16)
            self.server.state = state
            redirect_uri = f"http://{self.server.server_address[0]}:{PORT}/callback"
            params = urllib.parse.urlencode({
                "client_id": self.server.client_id,
                "redirect_uri": redirect_uri,
                "response_type": "code",
                "state": state,
                "token_access_type": "offline",
                "scope": SCOPES,
            })
            auth_url = f"https://www.dropbox.com/oauth2/authorize?{params}"
            self._send(f"""<!DOCTYPE html>
<html><body style="font-family:sans-serif;max-width:500px;margin:4em auto;text-align:center">
<h2>Bird Feeder — Dropbox Authorization</h2>
<p>Tap the button below to log in and authorize access to your Dropbox.</p>
<a href="{auth_url}" style="display:inline-block;padding:12px 24px;background:#0061ff;
color:#fff;border-radius:6px;text-decoration:none;font-size:1.1em">Connect Dropbox</a>
</body></html>""")

        elif parsed.path == "/callback":
            qs = urllib.parse.parse_qs(parsed.query)
            code = qs.get("code", [None])[0]
            state = qs.get("state", [None])[0]

            if not code or state != getattr(self.server, "state", None):
                self._send("<h2>Error: invalid state or missing code.</h2>", 400)
                return

            redirect_uri = f"http://{self.server.server_address[0]}:{PORT}/callback"
            data = urllib.parse.urlencode({
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            }).encode()
            req = urllib.request.Request(
                "https://api.dropboxapi.com/oauth2/token",
                data=data,
                method="POST",
            )
            import base64
            creds = base64.b64encode(
                f"{self.server.client_id}:{self.server.client_secret}".encode()
            ).decode()
            req.add_header("Authorization", f"Basic {creds}")
            req.add_header("Content-Type", "application/x-www-form-urlencoded")

            try:
                with urllib.request.urlopen(req) as resp:
                    token_data = json.loads(resp.read())
            except Exception as e:
                self._send(f"<h2>Token exchange failed: {e}</h2>", 500)
                return

            config_path = write_rclone_config(
                token_data,
                self.server.client_id,
                self.server.client_secret,
            )
            self._send(f"""<!DOCTYPE html>
<html><body style="font-family:sans-serif;max-width:500px;margin:4em auto;text-align:center">
<h2 style="color:green">Authorization successful!</h2>
<p>rclone is now configured. You can close this page.</p>
<p style="font-size:.85em;color:#666">Config saved to {config_path}</p>
</body></html>""")
            print(f"\n[OK] Token saved to {config_path}")
            self.server.done = True

        else:
            self._send("Not found", 404)


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <client_id> <client_secret>")
        sys.exit(1)

    client_id, client_secret = sys.argv[1], sys.argv[2]

    import socket
    local_ip = socket.gethostbyname(socket.gethostname())

    server = HTTPServer(("0.0.0.0", PORT), OAuthHandler)
    server.client_id = client_id
    server.client_secret = client_secret
    server.state = None
    server.done = False

    print(f"\nOpen this URL on your phone:\n\n  http://{local_ip}:{PORT}/\n")
    print("Waiting for authorization...")

    while not server.done:
        server.handle_request()

    print("Done — you can now run the backup script.")


if __name__ == "__main__":
    main()
