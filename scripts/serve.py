#!/usr/bin/env python3
"""Static server for web/ + POST /score endpoint that runs DetectionScorer.

Usage:
    export OPENROUTER_API_KEY=...
    python scripts/serve.py [--port 8000]
"""
import argparse, asyncio, json, sys, traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_explanation import score_explanation

WEB = Path(__file__).resolve().parent.parent / "web"


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/score":
            return self._json(404, {"error": "not found"})
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n))
            feature = int(req["feature"])
            description = str(req["description"]).strip()
            if not description:
                raise ValueError("empty description")
        except Exception as e:
            return self._json(400, {"error": f"bad request: {e}"})
        try:
            r = asyncio.run(score_explanation(feature, description))
            self._json(200, r)
        except Exception as e:
            traceback.print_exc()
            self._json(500, {"error": f"{type(e).__name__}: {e}"})


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8000)
    a = p.parse_args()
    print(f"serving {WEB} on http://localhost:{a.port}")
    ThreadingHTTPServer(("", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
