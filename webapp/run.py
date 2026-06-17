"""Launch the CobbleGen web server.

Usage:
    python -m webapp.run            # serve on http://127.0.0.1:8000
    python -m webapp.run --port 9000 --host 0.0.0.0
"""
from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CobbleGen web UI.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default 8000).")
    parser.add_argument("--reload", action="store_true", help="Enable autoreload for development.")
    args = parser.parse_args()

    print(f"\n  CobbleGen -> http://{args.host}:{args.port}\n")
    uvicorn.run("webapp.server:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
