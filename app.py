"""ChaseIQ server: Python standard-library HTTP server + JSON API + static frontend.

Run:            python app.py
Options:        python app.py --port 8080
                python app.py --build     (re-ingest data/raw/*.zip and retrain the model first)
                python app.py --no-browser

No third-party web framework is required. Analytics use pandas/NumPy and the model uses scikit-learn.
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
import sys
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

import config
from src import api
from src.api import ApiError
from src.ingest import find_raw_zips
from src.store import DataStore

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
STORE = DataStore()

GET_ROUTES = {
    "/api/status": api.status,
    "/api/filters": api.filters,
    "/api/overview": api.overview,
    "/api/matches": api.matches,
    "/api/match": api.match,
    "/api/strategy": api.strategy,
    "/api/winprob-grid": api.winprob_grid,
    "/api/players/search": api.player_search,
    "/api/player": api.player,
    "/api/compare": api.compare,
    "/api/matchup": api.matchup,
    "/api/teams": api.teams,
    "/api/explorer": api.explorer,
    "/api/model": api.model_report,
}


def _json_safe(obj):
    """Recursively turn NaN/inf into None and numpy scalars into Python values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    return obj


class Handler(BaseHTTPRequestHandler):
    server_version = "ChaseIQ/1.0"

    # ---- helpers -----------------------------------------------------------
    def _send_json(self, payload, status=HTTPStatus.OK):
        body = json.dumps(_json_safe(payload), default=str, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, body: bytes, content_type: str, status=HTTPStatus.OK, filename=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str):
        self._send_json({"error": message}, status=status)

    def _serve_static(self, path: str):
        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (config.STATIC_DIR / rel).resolve()
        if config.STATIC_DIR.resolve() not in target.parents and target != config.STATIC_DIR.resolve():
            return self._error(HTTPStatus.FORBIDDEN, "Forbidden")
        if not target.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "Not found")
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send_bytes(target.read_bytes(), ctype)

    def log_message(self, fmt, *args):  # quieter console
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), fmt % args))

    # ---- routing -------------------------------------------------------------
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
        try:
            if url.path == "/api/export/matches.csv":
                csv_text = api.csv_matches(STORE)
                return self._send_bytes(csv_text.encode("utf-8"), "text/csv; charset=utf-8",
                                        filename="chaseiq_matches.csv")
            handler = GET_ROUTES.get(url.path)
            if handler is None:
                if url.path.startswith("/api/"):
                    return self._error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {url.path}")
                return self._serve_static(url.path)
            return self._send_json(handler(STORE, q))
        except ApiError as exc:
            return self._error(exc.status, exc.message)
        except Exception as exc:  # keep the server alive; report the error to the browser
            traceback.print_exc()
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Server error: {exc}")

    def do_POST(self):
        url = urlparse(self.path)
        try:
            if url.path == "/api/upload":
                return self._handle_upload(parse_qs(url.query))
            if url.path == "/api/build":
                result = STORE.build()
                return self._send_json({"ok": True, **result})
            return self._error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {url.path}")
        except FileNotFoundError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:
            traceback.print_exc()
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Build failed: {exc}")

    def _handle_upload(self, query):
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._error(HTTPStatus.BAD_REQUEST, "Empty upload.")
        if length > MAX_UPLOAD_BYTES:
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "File is larger than 300 MB.")
        raw_name = (query.get("name") or ["upload.zip"])[0]
        name = Path(raw_name).name
        if not re.fullmatch(r"[A-Za-z0-9._-]+\.zip", name):
            return self._error(HTTPStatus.BAD_REQUEST, "File name must be a .zip with letters, digits, '.', '_' or '-'.")
        body = self.rfile.read(length)
        if body[:4] != b"PK\x03\x04":
            return self._error(HTTPStatus.BAD_REQUEST, "The file is not a ZIP archive.")
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        (config.RAW_DIR / name).write_bytes(body)
        result = {"saved_as": name, "bytes": length}
        if (query.get("build") or ["1"])[0] == "1":
            result.update(STORE.build())
        return self._send_json({"ok": True, **result})


def start(port: int, open_browser: bool) -> None:
    server = ThreadingHTTPServer((config.HOST, port), Handler)
    url = f"http://{config.HOST}:{port}/"
    print(f"ChaseIQ running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ChaseIQ cricket analytics server")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--build", action="store_true", help="re-ingest data/raw ZIPs and retrain")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if args.build:
        print("Rebuilding from data/raw ...")
        STORE.build(log=print)
    elif not STORE.load():
        if find_raw_zips():
            print("Processed data not found; building from data/raw ZIPs (one-off, a few minutes) ...")
            STORE.build(log=print)
        else:
            print("No data yet. Add a Cricsheet T20 JSON ZIP to data/raw/ or upload it on the Data page.")
    start(args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
