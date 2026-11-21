"""Real-time operation: the forecast window follows the run date, missing station-hour history is filled (imputed), and the production forecast is produced.

Why this module exists
  The station-hour source ends long before the run date, but the independent daily tracker (all ticket modes) is much fresher.
  Method ("tracker-driven top-down"):
      station-hour value  =  network tracker total  x  ratio(station-derived / tracker)  x  station share  x  hour-of-day profile
  The ratio, shares and profiles are learned from the most recent *observed* station-hour days only. For past days with a tracker value the
  tracker total is the actual; for later days it is the forecast of the best tracker model. Every filled value is flagged IMPUTED.
"""
from __future__ import annotations
from dataclasses import replace
from datetime import date
import numpy as np
import pandas as pd

from . import network_daily as nd
from .evaluate import make_ctx, metrics
from .models import trk_td_days, MODELS


def resolve_window(cfg: dict, last_hourly: pd.Timestamp, tracker_last: pd.Timestamp, as_of=None) -> dict:
    rt = cfg.get("realtime", {})
    as_of = pd.Timestamp(as_of or rt.get("as_of") or date.today()).normalize()
    start = as_of + pd.Timedelta(days=int(rt.get("start_offset_days", 1)))
    end = start + pd.Timedelta(days=cfg["horizon_days"] - 1)
    if start <= last_hourly:
        raise ValueError(f"as-of date {as_of.date()} is not after the last hourly observation ({last_hourly.date()}); use a later date")
    return {"as_of": as_of, "start": start, "end": end, "tracker_last": tracker_last, "last_hourly": last_hourly,
            "h_trk": int((end - tracker_last).days), "stale_hourly_days": int((start - last_hourly).days), "stale_tracker_days": int((start - tracker_last).days)}


def tracker_path(sdf: pd.DataFrame, sel_t: str, cfg: dict, win: dict) -> tuple[pd.Series, pd.Series]:
    """Network tracker totals for every day up to the end of the forecast window: actual, seasonal fill for missing days, then forecast."""
    o = sdf.index.get_loc(win["tracker_last"])
    y = sdf["y"].to_numpy()[:o + 1]
    yi = nd._impute(y)
    f = nd._forecast(sel_t, sdf, o, win["h_trk"], cfg)
    idx = sdf.index[:o + 1 + len(f)]
    T = pd.Series(np.r_[yi, f], index=idx)
    src = pd.Series(np.r_[np.where(np.isnan(y), "filled", "observed"), np.repeat("forecast", len(f))], index=idx)
    return T, src


def _missing_runs(has: np.ndarray):
    runs, a = [], None
    for i, v in enumerate(has):
        if not v and a is None: a = i
        if v and a is not None: runs.append((a, i - 1)); a = None
    return runs


def fill_and_forecast(cube, cal: pd.DataFrame, cfg: dict, T: pd.Series, src: pd.Series, win: dict, y_actual: pd.Series) -> dict:
    """Fill (a) dates missing inside the observed range and (b) every day from the last observation to the run date, and forecast the window."""
    Hp = int((win["end"] - cube.dates[-1]).days)
    ctx, arrs = make_ctx(cube, cal, cfg, horizon=Hp, trk={"y": y_actual})
    S, Dx, o = cube.S, cube.D + Hp, cube.D - 1
    filled = {m: np.full((S, Dx, 24), np.nan) for m in ("entries", "exits")}
    kind = np.array([""] * Dx, dtype=object)
    for a, b in _missing_runs(cube.date_has_data):                       # internal gaps: parameters from the observed days that follow
        days = np.arange(a, b + 1); Tg = T.reindex(ctx.dates[days]).to_numpy()
        for m in filled:
            filled[m][:, days] = trk_td_days(ctx, arrs[m], o, days, Tg)
        kind[days] = "gap"
    days = np.arange(o + 1, o + 1 + Hp); Tp = T.reindex(ctx.dates[days]).to_numpy()
    for m in filled:
        filled[m][:, days] = trk_td_days(ctx, arrs[m], o, days, Tp)
    d = ctx.dates[days]
    kind[days] = np.where(d < win["start"], np.where(d <= win["tracker_last"], "extension", "bridge"), "target")
    tgt = np.where(kind == "target")[0]
    pred = {m: filled[m][:, tgt, :].copy() for m in filled}
    return {"ctx": ctx, "arrs": arrs, "filled": filled, "kind": kind, "target_idx": tgt, "pred": pred, "dates": ctx.dates[tgt], "T": T, "src": src, "Hp": Hp, "origin_idx": o}


def validate_imputation(cube, cal: pd.DataFrame, cfg: dict, y_actual: pd.Series) -> pd.DataFrame:
    """Hide observed days, fill them with the method, compare with what was actually observed (and with a no-tracker baseline)."""
    ctx, arrs = make_ctx(cube, cal, cfg, horizon=14, trk={"y": y_actual})
    D = cube.D
    T_all = pd.Series(nd._impute(y_actual.reindex(ctx.dates).to_numpy()), index=ctx.dates)
    scen = [("forward fill, 7 days", D - 8, np.arange(D - 7, D), None), ("forward fill, 14 days", D - 15, np.arange(D - 14, D), None),
            ("interior block, 7 days", D - 1, np.arange(D - 21, D - 14), np.arange(D - 21, D - 14))]
    thr = cfg["evaluation"]
    rows = []
    for name, o, days, hide in scen:
        c2 = ctx
        A2 = {m: arrs[m].copy() for m in arrs}
        if hide is not None:
            obs2 = ctx.obs.copy(); obs2[:, hide] = False
            c2 = replace(ctx, obs=obs2)
            for m in A2: A2[m][:, hide] = np.nan
        Tg = T_all.reindex(ctx.dates[days]).to_numpy()
        for m in ("entries", "exits"):
            act = arrs[m][:, days, :]
            ok = ctx.obs[:, days] & ~np.isnan(act).any(axis=2)
            variants = {"tracker-driven (used)": trk_td_days(c2, A2[m], o, days, Tg)}
            if hide is None:       # baseline without the tracker: repeat the last 3 weeks (same hour of week)
                variants["no-tracker baseline (3-week average)"] = MODELS["ma_3wk"](c2, A2[m], o)[:, :len(days), :]
            for meth, P in variants.items():
                v = ok[:, :, None] & ~np.isnan(act); v[:, :, cfg["non_operating_hours"]] = False
                h = metrics(P[v], act[v], thr["mape_min_actual"])
                pd_, ad_ = P.sum(axis=2), np.nansum(act, axis=2)
                dd = metrics(pd_[ok], ad_[ok], thr["daily_mape_min_actual"])
                nn = metrics(np.where(ok, pd_, 0).sum(axis=0), np.where(ok, ad_, 0).sum(axis=0), 1.0)
                rows.append({"scenario": name, "method": meth, "measure": m, "hourly_WAPE": h["WAPE"], "hourly_MAE": h["MAE"], "daily_WAPE": dd["WAPE"],
                             "network_WAPE": nn["WAPE"], "network_bias_pct": float((np.where(ok, pd_, 0).sum() / np.where(ok, ad_, 0).sum() - 1) * 100)})
    return pd.DataFrame(rows)
