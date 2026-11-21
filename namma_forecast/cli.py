"""Command line interface.
  python -m namma_forecast.cli run                      # full pipeline -> outputs/
  python -m namma_forecast.cli dashboard                # open the finished dashboard in your browser
  python -m namma_forecast.cli run --open               # run, then open it
  python -m namma_forecast.cli run --as-of 2026-11-20   # pretend it is that day (default: today)
  python -m namma_forecast.cli run --fast               # skip SARIMA (about 1 minute faster)
  python -m namma_forecast.cli refresh --hourly new.csv [--hourly-measure exits] --daily tracker.csv   # register newer data (FR-16)
  python -m namma_forecast.cli refresh ... --run        # register and rerun
  python -m namma_forecast.cli actuals --observed obs.csv   # fill actual_value of stored forecasts (FR-18)
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import pandas as pd
from .config import load_config, resolve


def _serve(cfg, port=8765):
    """Serve outputs/ over http://localhost so the page always opens as a real web page, then open the browser."""
    import functools, http.server, socketserver, threading, webbrowser
    out = resolve(cfg, cfg["paths"]["outputs"]) if "outputs" in cfg["paths"] else Path("outputs")
    Handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out))
    Handler.extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".html": "text/html; charset=utf-8", ".json": "application/json; charset=utf-8"}
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as httpd:
        url = f"http://localhost:{port}/dashboard.html"
        print(f"Dashboard running at {url}   (press Ctrl+C to stop)")
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("stopped")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="namma_forecast")
    ap.add_argument("--config", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--fast", action="store_true"); r.add_argument("--open", action="store_true", help="open the dashboard when finished")
    r.add_argument("--as-of", default=None, help="run date YYYY-MM-DD (default: today); the forecast covers the 7 days after it")
    sub.add_parser("dashboard", help="serve outputs/ on localhost and open the dashboard in your browser")
    f = sub.add_parser("refresh")
    f.add_argument("--hourly", action="append", type=Path); f.add_argument("--hourly-measure", choices=["entries", "exits"])
    f.add_argument("--daily", type=Path); f.add_argument("--run", action="store_true")
    a = sub.add_parser("actuals"); a.add_argument("--observed", type=Path, required=True)
    for _s in (sys.stdout, sys.stderr):      # Windows consoles default to cp1252; log lines contain non-ASCII
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.cmd == "run":
        from .pipeline import run
        run(args.config, skip_sarima=args.fast, as_of=args.as_of)
        if args.open:
            _serve(cfg)
    elif args.cmd == "dashboard":
        _serve(cfg)
    elif args.cmd == "refresh":
        from . import ingest
        added = ingest.refresh(cfg, args.hourly, args.daily, args.hourly_measure)
        print("registered:", added)
        if args.run:
            from .pipeline import run
            run(args.config)
    elif args.cmd == "actuals":
        from .store import update_actuals
        obs = pd.read_csv(args.observed)
        print("rows updated:", update_actuals(resolve(cfg, cfg["paths"]["db"]), obs))


if __name__ == "__main__":
    main()
