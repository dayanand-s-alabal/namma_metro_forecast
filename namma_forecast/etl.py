"""FR-02, FR-03, FR-04, NFR-07: validation, cleaning (with log), station-hour integration and aggregation.

Design rules (SRS 6.3):
  * missing station-hours are never filled with zero; pre-opening hours are masked (NaN + NOT_OPEN)
  * invalid values are flagged/excluded and logged, unusual values are flagged SUSPECT but not altered
  * entries and exits stay separate targets
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .dimensions import alias_lookup, canonical_name, build_dim_station

OBS, MISSING_DATE, NOT_OPEN = 1, 0, 2   # status codes used in coverage matrices


@dataclass
class Cube:
    """Integrated station x date x hour grid (the analytical grain of SRS 6.5)."""
    dates: pd.DatetimeIndex
    stations: list[str]
    entries: np.ndarray            # (S, D, 24) float, NaN where not observed / not open
    exits: np.ndarray
    status: np.ndarray             # (S, D) int8: 1 observed, 0 date missing from sources, 2 station not yet open
    suspect_e: np.ndarray          # (S, D, 24) bool
    suspect_x: np.ndarray
    open_idx: np.ndarray           # (S,) first date index on which station is open (-1 if never)
    date_has_data: np.ndarray      # (D,) bool
    log: list[dict] = field(default_factory=list)

    @property
    def S(self): return len(self.stations)
    @property
    def D(self): return len(self.dates)
    def arr(self, measure: str) -> np.ndarray:
        return self.entries if measure == "entries" else self.exits


def _log(log, step, severity, scope, detail, n=0, action=""):
    log.append({"step": step, "severity": severity, "scope": scope, "detail": detail, "n_records": int(n), "action": action})


def clean_long(df: pd.DataFrame, measure: str, lookup: dict, log: list) -> pd.DataFrame:
    """Parse, canonicalise station names, resolve duplicates, flag invalid counts."""
    n0 = len(df)
    df = df.copy()
    df["date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["hour"] = pd.to_numeric(df["Hour"], errors="coerce")
    df["value"] = pd.to_numeric(df["Ridership"], errors="coerce")
    bad_date = df["date"].isna() | df["hour"].isna() | ~df["hour"].between(0, 23) | (df["hour"] % 1 != 0)
    if bad_date.any():
        _log(log, "parse", "ERROR", measure, "unparseable date/hour", bad_date.sum(), "rows excluded")
        df = df[~bad_date]
    df["hour"] = df["hour"].astype(int)
    # station names -> canonical
    names = df["Station"].astype(str).unique()
    mapping = {n: canonical_name(n, lookup) for n in names}
    changed = {n: c for n, (c, ch) in mapping.items() if ch}
    if changed:
        _log(log, "station_mapping", "INFO", measure, "source station names mapped to canonical: " + "; ".join(f"{k} -> {v}" for k, v in changed.items()),
             df["Station"].isin(changed).sum(), "renamed via station-names/codes dictionary")
    df["station"] = df["Station"].map(lambda n: mapping[n][0])
    # invalid counts
    neg = df["value"] < 0
    nonint = (df["value"] % 1 != 0) & df["value"].notna()
    nan = df["value"].isna()
    inv = neg | nonint | nan
    if inv.any():
        _log(log, "invalid_values", "ERROR", measure, f"negative={int(neg.sum())}, non-integer={int(nonint.sum())}, non-numeric={int(nan.sum())}", inv.sum(),
             "excluded (set to missing) - not altered")
        df = df[~inv]
    else:
        _log(log, "invalid_values", "INFO", measure, "no negative, non-integer or non-numeric counts", 0, "none needed")
    # duplicates
    key = ["date", "hour", "station"]
    dup = df.duplicated(key, keep=False)
    if dup.any():
        g = df[dup].groupby(key)["value"].nunique()
        conflicting = int((g > 1).sum())
        _log(log, "duplicates", "WARN", measure, f"{int(dup.sum())} rows share a (date,hour,station) key; {conflicting} keys with conflicting values",
             dup.sum(), "identical duplicates dropped; conflicting keys keep the LAST source (later refresh wins)")
        df = df.drop_duplicates(key, keep="last")
    else:
        _log(log, "duplicates", "INFO", measure, "no duplicate (date,hour,station) keys", 0, "none needed")
    _log(log, "rows", "INFO", measure, f"{n0} raw rows -> {len(df)} clean rows", len(df), "")
    return df[["date", "hour", "station", "value"]]


def _to_grid(df: pd.DataFrame, dates: pd.DatetimeIndex, st_idx: dict) -> np.ndarray:
    A = np.full((len(st_idx), len(dates), 24), np.nan)
    di = dates.get_indexer(df["date"])
    si = df["station"].map(st_idx).to_numpy()
    A[si, di, df["hour"].to_numpy()] = df["value"].to_numpy()
    return A


def _suspect(A: np.ndarray, dates: pd.DatetimeIndex, z_thr: float, min_abs: float) -> np.ndarray:
    """Robust z-score per (station, hour, day-type) across dates; flags only, never alters."""
    out = np.zeros(A.shape, bool)
    dow = dates.dayofweek.to_numpy()
    for grp in (dow < 5, dow == 5, dow == 6):
        idx = np.where(grp)[0]
        if len(idx) < 4:
            continue
        sub = A[:, idx, :]
        with np.errstate(all="ignore"):
            med = np.nanmedian(sub, axis=1, keepdims=True)
            mad = np.nanmedian(np.abs(sub - med), axis=1, keepdims=True)
            z = 0.6745 * (sub - med) / np.maximum(mad, 1.0)
            out[:, idx, :] = (np.abs(z) > z_thr) & (np.abs(sub - med) > min_abs)
    return out


def build_cube(raw: dict[str, pd.DataFrame], refs: dict, cfg: dict) -> tuple[Cube, pd.DataFrame, list[dict]]:
    log: list[dict] = []
    lookup = alias_lookup(refs["aliases"], refs["codes"])
    clean = {m: clean_long(df, m, lookup, log) for m, df in raw.items() if len(df)}
    all_df = pd.concat(clean.values())
    dmin, dmax = all_df["date"].min(), all_df["date"].max()
    dates = pd.date_range(dmin, dmax, freq="D")
    station_order = list(dict.fromkeys(clean.get("entries", all_df)["station"]))
    for s in all_df["station"].unique():
        if s not in station_order:
            station_order.append(s)
            _log(log, "station_mapping", "WARN", "dimension", f"station '{s}' present only in exits - added to dimension", 1, "added")
    st_idx = {s: i for i, s in enumerate(station_order)}
    S, D = len(station_order), len(dates)

    # ---- date coverage (missing hours / dates)
    present = pd.DatetimeIndex(sorted(all_df["date"].unique()))
    missing = dates.difference(present)
    if len(missing):
        runs = pd.Series(missing).groupby((pd.Series(missing).diff().dt.days != 1).cumsum()).agg(["min", "max", "count"])
        for _, r in runs.iterrows():
            _log(log, "coverage", "WARN", "date", f"no hourly data for {r['min'].date()} .. {r['max'].date()} ({int(r['count'])} days)", int(r["count"]) * S * 24,
                 "left as MISSING_DATE (not zero-filled, not imputed for evaluation)")
    for m, df in clean.items():
        per = df.groupby(["date", "station"])["hour"].nunique()
        part = per[per < 24]
        if len(part) == 0:
            _log(log, "coverage", "INFO", m, "every station-day present in the source has all 24 hour buckets", 0, "none needed")
        else:
            _log(log, "coverage", "WARN", m, f"{len(part)} station-days have <24 hour buckets", int(part.size), "missing hours kept as missing (NaN)")

    E = _to_grid(clean["entries"], dates, st_idx) if "entries" in clean else np.full((S, D, 24), np.nan)
    X = _to_grid(clean["exits"], dates, st_idx) if "exits" in clean else np.full((S, D, 24), np.nan)

    # ---- station opening dates (pre-opening / trial-run vs genuine service)
    # Rule: a station is "open" from the first day whose entries+exits reach opening_threshold_frac of its median
    # positive daily traffic. Smaller earlier values are trial-run / test traffic and are masked as NOT_OPEN.
    frac = cfg["validation"].get("opening_threshold_frac", 0.10)
    tot = np.nan_to_num(E).sum(axis=2) + np.nan_to_num(X).sum(axis=2)       # (S, D)
    rows_any = ~np.isnan(E).all(axis=2) | ~np.isnan(X).all(axis=2)
    open_idx = np.full(S, -1)
    for s_ in range(S):
        v = tot[s_]
        pos_v = v[v > 0]
        if len(pos_v):
            open_idx[s_] = int(np.argmax(v >= frac * np.median(pos_v)))
    first_pos = np.where((tot > 0).any(axis=1), (tot > 0).argmax(axis=1), -1)
    status = np.zeros((S, D), np.int8)
    date_has = np.isin(dates, present)
    for s_ in range(S):
        for d in range(D):
            status[s_, d] = MISSING_DATE if not date_has[d] else (NOT_OPEN if (open_idx[s_] < 0 or d < open_idx[s_]) else OBS)
    pre = status == NOT_OPEN
    trial = float(np.nansum(np.where(pre[:, :, None], E, 0)) + np.nansum(np.where(pre[:, :, None], X, 0)))
    n_trial_days = int(((first_pos >= 0) & (first_pos < open_idx)).sum())
    E[pre] = np.nan
    X[pre] = np.nan
    late = [(station_order[s_], str(dates[open_idx[s_]].date())) for s_ in range(S) if open_idx[s_] > 0]
    if late:
        _log(log, "pre_opening", "WARN", "dimension",
             f"{len(late)} stations (Yellow Line extension) enter service after {dates[0].date()}: " + ", ".join(f"{n} {d}" for n, d in late[:5]) + ", ...",
             int(pre.sum()) * 24, "masked as NOT_OPEN (NaN); entries rows were absent and exits were zero-filled by the source - not treated as zero demand")
    _log(log, "pre_opening", "INFO", "trial_runs", f"{n_trial_days} stations show tiny pre-service/trial traffic ({trial:.0f} passengers in total) before the commercial start",
         n_trial_days, f"treated as trial-run: masked as NOT_OPEN (threshold {frac:.0%} of the station's median daily traffic)")

    z, ma = cfg["validation"]["suspect_mad_z"], cfg["validation"]["suspect_min_abs"]
    sus_e, sus_x = _suspect(E, dates, z, ma), _suspect(X, dates, z, ma)
    _log(log, "suspect_values", "INFO", "entries", f"{int(sus_e.sum())} station-hours deviate strongly from their own hour/day-type pattern (|robust z|>{z}, >{ma} pax)",
         int(sus_e.sum()), "flagged SUSPECT, kept unchanged (may be genuine events)")
    _log(log, "suspect_values", "INFO", "exits", f"{int(sus_x.sum())} station-hours flagged SUSPECT", int(sus_x.sum()), "flagged SUSPECT, kept unchanged")

    non_op = cfg["non_operating_hours"]
    nz = {m: float(np.nansum(A[:, :, non_op] > 0)) for m, A in (("entries", E), ("exits", X))}
    _log(log, "operating_hours", "INFO", "hours", f"hours {non_op} are non-operating: positive counts entries={nz['entries']:.0f}, exits={nz['exits']:.0f} station-hours",
         int(sum(nz.values())), "forecast fixed at 0, flagged is_operating_hour=0")

    cube = Cube(dates, station_order, E, X, status, sus_e, sus_x, open_idx, date_has, log)
    first_obs = {station_order[s]: (str(dates[open_idx[s]].date()) if open_idx[s] >= 0 else None) for s in range(S)}
    dim = build_dim_station(station_order, first_obs, refs["stations"], refs["codes"], refs["aliases"], str(dates[0].date()))
    unmatched = dim.loc[dim.ref_match == "UNMATCHED", "station_name"].tolist()
    _log(log, "station_mapping", "INFO" if not unmatched else "WARN", "dimension",
         f"{S} canonical stations; {S - len(unmatched)} matched to reference station dataset (line/interchange)" + (f"; unmatched: {unmatched}" if unmatched else ""), S, "")
    return cube, dim, log


# --------------------------------------------------------------------------------------- outputs
def station_hourly_fact(cube: Cube, dim: pd.DataFrame, cal: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """fact_station_hourly_ridership: one row per station x date x hour (only dates present in the sources)."""
    S, D = cube.S, cube.D
    di = np.where(cube.date_has_data)[0]
    sid = dim["station_id"].to_numpy()
    s_ix, d_ix, h_ix = np.meshgrid(np.arange(S), di, np.arange(24), indexing="ij")
    s_ix, d_ix, h_ix = s_ix.ravel(), d_ix.ravel(), h_ix.ravel()
    st = cube.status[s_ix, d_ix]
    sus_e, sus_x = cube.suspect_e[s_ix, d_ix, h_ix], cube.suspect_x[s_ix, d_ix, h_ix]
    flag_e = np.where(st == NOT_OPEN, "NOT_OPEN", np.where(sus_e, "SUSPECT", "OBSERVED"))
    flag_x = np.where(st == NOT_OPEN, "NOT_OPEN", np.where(sus_x, "SUSPECT", "OBSERVED"))
    flag = np.where((flag_e == "NOT_OPEN"), "NOT_OPEN", np.where((flag_e == "SUSPECT") | (flag_x == "SUSPECT"), "SUSPECT", "OBSERVED"))
    dts = cube.dates[d_ix]
    calx = cal.set_index("date")
    df = pd.DataFrame({
        "date": dts.strftime("%Y-%m-%d"), "hour": h_ix.astype(np.int8),
        "datetime": (dts + pd.to_timedelta(h_ix, unit="h")).strftime("%Y-%m-%d %H:00"),
        "station_id": sid[s_ix],
        "entries": cube.entries[s_ix, d_ix, h_ix], "exits": cube.exits[s_ix, d_ix, h_ix],
        "rainfall_mm": np.nan, "temperature_c": np.nan, "humidity": np.nan,   # no verified aligned weather available (FR-17)
        "is_weekend": calx["is_weekend"].reindex(dts).to_numpy().astype(int),
        "is_holiday": calx["is_holiday"].reindex(dts).to_numpy().astype(int),
        "is_festival": calx["is_festival"].reindex(dts).to_numpy().astype(int),
        "is_operating_hour": (~np.isin(h_ix, cfg["non_operating_hours"])).astype(int),
        "entries_flag": flag_e, "exits_flag": flag_x, "data_quality_flag": flag,
    })
    df["entries"] = df["entries"].astype("Int64")
    df["exits"] = df["exits"].astype("Int64")
    return df


def station_daily(cube: Cube, dim: pd.DataFrame) -> pd.DataFrame:
    """Aggregate hourly -> station-day (sum of the 24 hour buckets; NaN when the station is not open / date missing)."""
    rows = []
    for m in ("entries", "exits"):
        A = cube.arr(m)
        tot = A.sum(axis=2)  # NaN propagates -> NOT_OPEN days stay NaN
        s_ix, d_ix = np.where(cube.status == OBS)
        rows.append(pd.DataFrame({"date": cube.dates[d_ix], "station_id": dim["station_id"].to_numpy()[s_ix], "measure": m, "value": tot[s_ix, d_ix]}))
    df = pd.concat(rows).pivot_table(index=["date", "station_id"], columns="measure", values="value").reset_index()
    return df.rename(columns={"entries": "entries", "exits": "exits"})


def network_daily_from_stations(sd: pd.DataFrame) -> pd.DataFrame:
    g = sd.groupby("date").agg(entries=("entries", "sum"), exits=("exits", "sum"), n_stations=("station_id", "nunique")).reset_index()
    return g


def prepare_tracker(tr: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    """Tracker daily series -> fact_network_daily_ridership. QR total = sum of app columns when the raw total disagrees."""
    t = tr.copy()
    apps = ["QR NammaMetro", "QR WhatsApp", "QR Paytm", "QR ONDC"]
    t["qr_apps_sum"] = t[apps].sum(axis=1)
    bad = t["Total QR"] != t["qr_apps_sum"]
    _log(log, "tracker_qr", "WARN", "daily_tracker", f"'Total QR' disagrees with the sum of QR app columns on {int(bad.sum())} of {len(t)} days "
         f"(100% of days Sep-2025..Jul-2026); max |diff|={int((t['Total QR']-t['qr_apps_sum']).abs().max())}", int(bad.sum()),
         "QR = sum of app columns (also the rule used in the tracker's processed file); original column kept as qr_total_raw")
    out = pd.DataFrame({
        "date": t["date"], "official_daily_total": t["Total Smart Cards"] + t["Total Tokens"] + t["Total NCMC"] + t["Group Ticket"] + t["qr_apps_sum"],
        "smart_cards": t["Total Smart Cards"], "tokens": t["Total Tokens"], "ncmc": t["Total NCMC"], "group_ticket": t["Group Ticket"],
        "qr": t["qr_apps_sum"], "qr_total_raw": t["Total QR"], "qr_inconsistent": bad.astype(int), "source_id": t["source_id"],
        "source_definition": "system-wide daily ridership, all ticket modes (tracker); QR = sum of apps",
    })
    full = pd.date_range(out["date"].min(), out["date"].max(), freq="D")
    miss = full.difference(out["date"])
    _log(log, "tracker_gaps", "WARN", "daily_tracker", f"{len(miss)} of {len(full)} calendar days missing ({out['date'].min().date()}..{out['date'].max().date()})", len(miss),
         "kept missing; imputed ONLY inside model fitting (never scored)")
    out["quality_status"] = np.where(out["qr_inconsistent"] == 1, "QR_RECOMPUTED", "OBSERVED")
    return out.sort_values("date").reset_index(drop=True)
