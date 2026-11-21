"""FR-01 / FR-16 / NFR-07: source loading (CSV, Parquet, Excel), raw-data registry and refresh.

Raw files are never modified. `refresh` copies new files into data/raw/ and appends a registry entry.
"""
from __future__ import annotations
import hashlib, re, shutil
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd

from .config import resolve, load_sources, save_sources

HOURLY_COLS = ["Date", "Hour", "Station", "Ridership"]


def _sniff_sep(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig") as f:
        head = f.readline()
    return ";" if head.count(";") > head.count(",") else ","


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_wide_rti_xlsx(path: Path, measure_hint: str | None) -> list[tuple[str, pd.DataFrame]]:
    """Parse the original RTI Excel layout: BUSINESS DATE | STATION ('11-Baiyappanahalli') | 24 hour columns | TOTAL.
    Sheet name containing 'exit' => exits, otherwise entries. Hour headers are 'H00'.. or '00:00 Hrs To 01:00 Hrs'."""
    out = []
    xl = pd.ExcelFile(path)
    for sh in xl.sheet_names:
        raw = xl.parse(sh)
        raw.columns = [str(c).strip() for c in raw.columns]
        if "BUSINESS DATE" not in raw.columns or "STATION" not in raw.columns:
            continue  # e.g. station-pair workbooks use another layout (handled by od loader)
        raw = raw.dropna(subset=["BUSINESS DATE", "STATION"])
        hour_cols = {}
        for c in raw.columns:
            m = re.match(r"^H(\d{2})$", c) or re.match(r"^(\d{2}):00\s*Hrs", c)
            if m:
                hour_cols[c] = int(m.group(1))
        if len(hour_cols) != 24:
            raise ValueError(f"{path.name}/{sh}: expected 24 hour columns, found {len(hour_cols)}")
        long = raw.melt(id_vars=["BUSINESS DATE", "STATION"], value_vars=list(hour_cols), var_name="hc", value_name="Ridership")
        long["Hour"] = long["hc"].map(hour_cols)
        long["Date"] = pd.to_datetime(long["BUSINESS DATE"]).dt.strftime("%Y-%m-%d")
        long["Station"] = long["STATION"].astype(str).str.replace(r"^\d+-", "", regex=True).str.strip()
        measure = "exits" if "exit" in sh.lower() else (measure_hint or "entries")
        out.append((measure, long[HOURLY_COLS]))
    return out


def load_hourly_source(cfg: dict, entry: dict) -> dict[str, pd.DataFrame]:
    """Return {measure: long DataFrame[Date, Hour, Station, Ridership, source_id]} for one registry entry."""
    path = resolve(cfg, f"{cfg['paths']['raw_dir']}/{entry['path']}")
    fmt = entry.get("format", "long_csv")
    res: dict[str, pd.DataFrame] = {}
    if fmt == "long_csv":
        df = pd.read_csv(path, sep=_sniff_sep(path), encoding="utf-8-sig")
        res[entry["measure"]] = df
    elif fmt == "parquet":
        res[entry["measure"]] = pd.read_parquet(path)
    elif fmt in ("rti_xlsx", "xlsx"):
        for measure, df in _read_wide_rti_xlsx(path, entry.get("measure")):
            res[measure] = df
    else:
        raise ValueError(f"Unsupported hourly format {fmt!r}")
    for m, df in res.items():
        missing = [c for c in HOURLY_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path.name}: missing columns {missing}")
        df = df[HOURLY_COLS].copy()
        df["source_id"] = entry["source_id"]
        res[m] = df
    return res


def load_all_hourly(cfg: dict, sources: dict) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Load every registered hourly source. Returns raw frames per measure and a provenance log (AC-01)."""
    frames: dict[str, list[pd.DataFrame]] = {"entries": [], "exits": []}
    prov = []
    for entry in sources["hourly"]:
        path = resolve(cfg, f"{cfg['paths']['raw_dir']}/{entry['path']}")
        got = load_hourly_source(cfg, entry)
        for m, df in got.items():
            frames[m].append(df)
        d = pd.concat(got.values())
        prov.append({
            "source_id": entry["source_id"], "file": entry["path"], "format": entry.get("format"),
            "sha256": sha256(path), "rows": int(sum(len(x) for x in got.values())),
            "measures": ",".join(got), "date_min": str(d["Date"].min()), "date_max": str(d["Date"].max()),
            "dates": int(d["Date"].nunique()), "stations": int(d["Station"].nunique()),
            "url": entry.get("url", ""), "downloaded": entry.get("downloaded", ""), "license": entry.get("license", ""),
            "origin": entry.get("origin", ""), "definition": entry.get("definition", ""), "limitations": entry.get("limitations", ""),
        })
    out = {m: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=HOURLY_COLS + ["source_id"]) for m, v in frames.items()}
    return out, prov


def load_tracker(cfg: dict, sources: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Daily network tracker (raw file, ticket-mode columns). Several daily sources are concatenated; later wins."""
    parts, prov = [], []
    for entry in sources["daily"]:
        path = resolve(cfg, f"{cfg['paths']['raw_dir']}/{entry['path']}")
        df = pd.read_csv(path, sep=_sniff_sep(path))
        df["date"] = pd.to_datetime(df["Record Date"], format="%d-%m-%Y", errors="coerce")
        if df["date"].isna().any():
            df["date"] = pd.to_datetime(df["Record Date"], errors="coerce")
        df["source_id"] = entry["source_id"]
        parts.append(df)
        prov.append({
            "source_id": entry["source_id"], "file": entry["path"], "format": entry.get("format"), "sha256": sha256(path),
            "rows": int(len(df)), "measures": "daily_total", "date_min": str(df["date"].min().date()),
            "date_max": str(df["date"].max().date()), "dates": int(df["date"].nunique()), "stations": 0,
            "url": entry.get("url", ""), "downloaded": entry.get("downloaded", ""), "license": entry.get("license", ""),
            "origin": entry.get("origin", ""), "definition": entry.get("definition", ""), "limitations": entry.get("limitations", ""),
        })
    df = pd.concat(parts, ignore_index=True).sort_values(["date", "source_id"]).drop_duplicates("date", keep="last")
    return df.reset_index(drop=True), prov


def load_reference(cfg: dict, sources: dict) -> dict[str, pd.DataFrame]:
    ref = sources["reference"]
    rd = lambda k, **kw: pd.read_csv(resolve(cfg, f"{cfg['paths']['raw_dir']}/{ref[k]}"), **kw)
    return {
        "stations": rd("stations_dataset"),
        "codes": rd("station_codes", sep=";"),
        "aliases": rd("station_names", sep=";"),
        "events": rd("significant_dates", parse_dates=["Date"]),
    }


def refresh(cfg: dict, hourly: list[Path] | None = None, daily: Path | None = None, measure: str | None = None,
            fmt: str | None = None) -> list[str]:
    """FR-16: register newer files (copied unchanged into data/raw/refresh/) so the next pipeline run uses them."""
    sources = load_sources(cfg)
    added = []
    rawdir = resolve(cfg, cfg["paths"]["raw_dir"]) / "refresh"
    rawdir.mkdir(parents=True, exist_ok=True)
    stamp = date.today().isoformat()
    for p in hourly or []:
        p = Path(p)
        dst = rawdir / f"{stamp}_{p.name}"
        shutil.copy2(p, dst)
        f = fmt or ("rti_xlsx" if p.suffix.lower() in (".xlsx", ".xls") else "parquet" if p.suffix == ".parquet" else "long_csv")
        sid = f"REFRESH_{stamp}_{p.stem}".upper().replace(" ", "_")
        sources["hourly"].append({"source_id": sid, "measure": measure or ("exits" if "exit" in p.name.lower() else "entries"),
                                  "path": f"refresh/{dst.name}", "format": f, "downloaded": stamp,
                                  "url": "user-supplied", "license": "user-supplied", "origin": "user refresh",
                                  "definition": "user-supplied hourly station file", "limitations": "not yet reviewed"})
        added.append(sid)
    if daily:
        p = Path(daily)
        dst = rawdir / f"{stamp}_{p.name}"
        shutil.copy2(p, dst)
        sid = f"REFRESH_{stamp}_{p.stem}".upper().replace(" ", "_")
        sources["daily"].append({"source_id": sid, "path": f"refresh/{dst.name}", "format": "tracker_raw", "downloaded": stamp,
                                 "url": "user-supplied", "license": "user-supplied", "origin": "user refresh",
                                 "definition": "user-supplied daily network file", "limitations": "not yet reviewed"})
        added.append(sid)
    save_sources(cfg, sources)
    return added
