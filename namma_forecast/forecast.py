"""FR-10 / FR-11 / FR-18 / NFR-10: final 7-day forecasts, reconciliation, prediction intervals and traceability records.

Aggregation rules (documented, reproducible):
  station-day  = sum of the 24 hourly forecasts of that station and measure
  network-day  = sum of the station-day forecasts (entries and exits stay separate targets)
If the best hourly model (M_h) differs from the best daily model (M_d), hourly values of M_h are rescaled per station-day so that
their sum equals M_d's station-day forecast ("reconciliation"). The reconciled pipeline is itself evaluated in the backtest.
"""
from __future__ import annotations
import hashlib, json
import numpy as np
import pandas as pd

from .models import MODELS, MODEL_INFO, ENSEMBLE_MEMBERS
from .evaluate import make_ctx, fold_origins

HOUR_GROUPS = {"night/early (0-5)": range(0, 6), "AM peak (6-10)": range(6, 11), "midday (11-15)": range(11, 16),
               "PM peak (16-20)": range(16, 21), "evening (21-23)": range(21, 24)}


def hour_group(h: int) -> str:
    for k, r in HOUR_GROUPS.items():
        if h in r:
            return k
    return "other"


def reconcile(hourly_h: np.ndarray, daily_d: np.ndarray) -> np.ndarray:
    """Scale (S,H,24) hourly forecasts so each station-day sums to daily_d (S,H). Zero-sum days fall back to a flat profile of daily_d."""
    s = hourly_h.sum(axis=2)
    with np.errstate(all="ignore"):
        f = np.where(s > 0, daily_d / s, 1.0)
    return hourly_h * f[:, :, None]


def predict_model(name: str, ctx, A, o):
    """Forecast of any candidate (including the derived ensemble) at origin o."""
    if name == "ensemble":
        return np.mean([MODELS[k](ctx, A, o) for k in ENSEMBLE_MEMBERS], axis=0)
    return MODELS[name](ctx, A, o)


def add_pipeline_backtest(bt: dict, sel_h: str, sel_d: str, ctx):
    """Add the 'pipeline' candidate (M_h hourly shape reconciled to M_d daily totals) to the backtest predictions."""
    for (name, m, fi), _ in list(bt["pred"].items()):
        if name != sel_h:
            continue
        ph = bt["pred"][(sel_h, m, fi)]
        pd_ = bt["pred"][(sel_d, m, fi)].sum(axis=2)
        out = ph if sel_h == sel_d else reconcile(ph, pd_)
        bt["pred"][("pipeline", m, fi)] = out
    bt["runtime"]["pipeline"] = bt["runtime"][sel_h] + (0 if sel_h == sel_d else bt["runtime"][sel_d])


def interval_params(bt: dict, cube, ctx, cfg: dict) -> dict:
    """Empirical (split-conformal style) multiplicative bands from out-of-sample ratios actual/predicted of the pipeline."""
    cov = cfg["evaluation"]["interval_coverage"]; a = (1 - cov) / 2
    H = ctx.horizon
    out = {"coverage": cov, "hourly": {}, "daily": {}, "network": {}}
    for m in ("entries", "exits"):
        rat = {g: [] for g in [f"{g}|{w}" for g in HOUR_GROUPS for w in ("weekday", "weekend")]}
        dr, nr = [], []
        for fi, o in enumerate(bt["origins"]):
            pred, act = bt["pred"][("pipeline", m, fi)], bt["act"][(m, fi)]
            dates = ctx.dates[o + 1:o + 1 + H]
            for k in range(H):
                wk = "weekend" if dates[k].dayofweek >= 5 else "weekday"
                for h in range(24):
                    if h in ctx.non_op:
                        continue
                    p, y = pred[:, k, h], act[:, k, h]
                    ok = (p >= 10) & ~np.isnan(y) & ctx.obs[:, o + 1 + k]
                    rat[f"{hour_group(h)}|{wk}"].extend((y[ok] / p[ok]).tolist())
            dv = ctx.obs[:, o + 1:o + 1 + H] & ~np.isnan(act).any(axis=2)
            pdy, ady = pred.sum(axis=2), np.nansum(act, axis=2)
            ok = dv & (pdy >= 200)
            dr.extend((ady[ok] / pdy[ok]).tolist())
            nr.extend((np.where(dv, ady, 0).sum(axis=0) / np.where(dv, pdy, 0).sum(axis=0)).tolist())
        out["hourly"][m] = {g: [float(np.quantile(v, a)), float(np.quantile(v, 1 - a)), len(v)] for g, v in rat.items() if len(v) > 20}
        out["daily"][m] = [float(np.quantile(dr, a)), float(np.quantile(dr, 1 - a)), len(dr)]
        out["network"][m] = [float(np.quantile(nr, a)), float(np.quantile(nr, 1 - a)), len(nr)]
    return out


def apply_hourly_bands(pred: np.ndarray, m: str, dates: pd.DatetimeIndex, ip: dict) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = np.zeros_like(pred), np.zeros_like(pred)
    z = 1.2816 * 1.5   # Poisson-style fallback for small expected counts (<10 pax/hour)
    for k, d in enumerate(dates):
        wk = "weekend" if d.dayofweek >= 5 else "weekday"
        for h in range(24):
            q = ip["hourly"][m].get(f"{hour_group(h)}|{wk}", [0.7, 1.3, 0])
            p = pred[:, k, h]
            small = p < 10
            lo[:, k, h] = np.where(small, np.maximum(0, p - z * np.sqrt(p)), p * q[0])
            hi[:, k, h] = np.where(small, p + z * np.sqrt(p) + 1, p * q[1])
    return lo, hi


def model_version(name: str, cfg: dict) -> str:
    from . import __version__
    h = hashlib.sha1(json.dumps({k: v for k, v in cfg.items() if not k.startswith("_")}, sort_keys=True, default=str).encode()).hexdigest()[:8]
    return f"{name}@{__version__}+cfg{h}"


def forecast_station_hourly(cube, cal, cfg, sel_h: str, sel_d: str, horizon: int | None = None, take: int | None = None) -> dict:
    """Production forecast from the last observed day with a station model. With horizon > 7 (run date later than the last observation)
    the model forecasts through the gap and only the last `take` days (the forecast window) are returned: (S, take, 24) per measure."""
    ctx, arrs = make_ctx(cube, cal, cfg, horizon=horizon)
    o = cube.D - 1; H = ctx.horizon; sl = slice(H - (take or H), H)
    res = {"origin": cube.dates[o], "dates": ctx.dates[o + 1:o + 1 + H][sl], "pred": {}, "ctx": ctx}
    for m in ("entries", "exits"):
        ph = predict_model(sel_h, ctx, arrs[m], o)
        if sel_d != sel_h:
            ph = reconcile(ph, predict_model(sel_d, ctx, arrs[m], o).sum(axis=2))
        ph[:, :, ctx.non_op] = 0.0
        res["pred"][m] = ph[:, sl, :]
    # baseline comparator (SRS FR-14): seasonal naive
    res["baseline"] = {m: predict_model("snaive_week", ctx, arrs[m], o)[:, sl, :] for m in ("entries", "exits")}
    return res


def production_from_fill(imp: dict, cube, win: dict) -> dict:
    """Package the tracker-driven forecast; the baseline is 'same hour last week' taken from the (filled) history."""
    tgt = imp["target_idx"]; base = {}
    for m in ("entries", "exits"):
        A = imp["arrs"][m].copy()
        A[:, ~np.isnan(imp["filled"][m]).all(axis=(0, 2))] = imp["filled"][m][:, ~np.isnan(imp["filled"][m]).all(axis=(0, 2))]
        base[m] = np.stack([A[:, tgt[0] + k - 7, :] for k in range(len(tgt))], axis=1)
    return {"origin": win["tracker_last"], "dates": imp["dates"], "pred": imp["pred"], "ctx": imp["ctx"], "baseline": base}


def inflate_intervals(ip: dict, k: float) -> dict:
    """Widen every empirical ratio band by factor k around 1 (used when station shares/profiles are old; heuristic, documented)."""
    wid = lambda q: [1 - (1 - q[0]) * k, 1 + (q[1] - 1) * k] + list(q[2:])
    out = {"coverage": ip["coverage"], "hourly": {}, "daily": {}, "network": {}, "inflation": k}
    for m in ("entries", "exits"):
        out["hourly"][m] = {g: wid(v) for g, v in ip["hourly"][m].items()}
        out["daily"][m] = wid(ip["daily"][m]); out["network"][m] = wid(ip["network"][m])
    return out


def imputed_tables(imp: dict, cube, dim, cal) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputed station-day and station-hour values (flagged) for gaps, the stretch up to the tracker end and the run-date bridge. The forecast window itself is excluded."""
    ctx = imp["ctx"]; sid = dim["station_id"].to_numpy(); S = cube.S
    keep = np.where(np.isin(imp["kind"], ["gap", "extension", "bridge"]))[0]
    dts = ctx.dates[keep]
    d_rows, h_rows = [], []
    kinds = imp["kind"][keep]
    for m in ("entries", "exits"):
        F = imp["filled"][m][:, keep, :]
        D = F.sum(axis=2)
        s_ix, k_ix = np.meshgrid(np.arange(S), np.arange(len(keep)), indexing="ij")
        d_rows.append(pd.DataFrame({"date": dts[k_ix.ravel()].strftime("%Y-%m-%d"), "station_id": sid[s_ix.ravel()], "measure": m, "value": D.ravel().round(1),
                                    "imputation_kind": kinds[k_ix.ravel()]}))
        s_ix, k_ix, h_ix = np.meshgrid(np.arange(S), np.arange(len(keep)), np.arange(24), indexing="ij")
        h_rows.append(pd.DataFrame({"date": dts[k_ix.ravel()].strftime("%Y-%m-%d"), "hour": h_ix.ravel().astype(np.int8), "station_id": sid[s_ix.ravel()], "measure": m,
                                    "value": np.round(F.ravel(), 1)}))
    dd = pd.concat(d_rows, ignore_index=True)
    dd = dd.pivot_table(index=["date", "station_id", "imputation_kind"], columns="measure", values="value").reset_index()
    dd["data_status"] = "IMPUTED"
    dd["method"] = "tracker-driven top-down: tracker total x station share x scaling ratio"
    hh = pd.concat(h_rows, ignore_index=True).pivot_table(index=["date", "hour", "station_id"], columns="measure", values="value").reset_index()
    hh["data_status"] = "IMPUTED"
    return dd, hh


def results_frame(cube, dim, fc: dict, ip: dict, cfg: dict, sel_h: str, sel_d: str, tracker_fc: pd.DataFrame | None, tracker_model: str | None,
                  tracker_band: tuple, generated_at: str, net_bands: list | None = None) -> pd.DataFrame:
    """forecast_results (SRS 6.5): one row per forecast value; includes hourly, station-daily, network-daily (station-derived and tracker)."""
    sid = dim["station_id"].to_numpy(); dates = fc["dates"]; ctx = fc["ctx"]
    ver = model_version("pipeline[" + sel_h + "->" + sel_d + "]", cfg)
    rows = []
    for m in ("entries", "exits"):
        P = fc["pred"][m]
        lo, hi = apply_hourly_bands(P, m, dates, ip)
        S, H, _ = P.shape
        s_ix, k_ix, h_ix = np.meshgrid(np.arange(S), np.arange(H), np.arange(24), indexing="ij")
        s_ix, k_ix, h_ix = s_ix.ravel(), k_ix.ravel(), h_ix.ravel()
        dts = dates[k_ix] + pd.to_timedelta(h_ix, unit="h")
        rows.append(pd.DataFrame({
            "forecast_generated_at": generated_at, "forecast_origin": str(fc["origin"].date()), "target_datetime": dts.strftime("%Y-%m-%d %H:00"),
            "target_date": dates[k_ix].strftime("%Y-%m-%d"), "station_id": sid[s_ix], "forecast_granularity": "hourly", "scope": "station",
            "target_measure": m, "predicted_value": P[s_ix, k_ix, h_ix], "lower_80": lo[s_ix, k_ix, h_ix], "upper_80": hi[s_ix, k_ix, h_ix],
            "actual_value": np.nan, "model_name": sel_h if sel_h == sel_d else f"{sel_h}->reconciled:{sel_d}", "model_version": ver,
            "horizon_step": k_ix * 24 + h_ix + 1, "is_operating_hour": (~np.isin(h_ix, cfg["non_operating_hours"])).astype(int),
            "definition_note": "boardings (entries) / alightings (exits) per station-hour; start-of-hour bucket"}))
        D = P.sum(axis=2)
        q = ip["daily"][m]
        s_ix, k_ix = np.meshgrid(np.arange(S), np.arange(H), indexing="ij"); s_ix, k_ix = s_ix.ravel(), k_ix.ravel()
        rows.append(pd.DataFrame({
            "forecast_generated_at": generated_at, "forecast_origin": str(fc["origin"].date()), "target_datetime": dates[k_ix].strftime("%Y-%m-%d"),
            "target_date": dates[k_ix].strftime("%Y-%m-%d"), "station_id": sid[s_ix], "forecast_granularity": "daily", "scope": "station",
            "target_measure": m, "predicted_value": D[s_ix, k_ix], "lower_80": D[s_ix, k_ix] * q[0], "upper_80": D[s_ix, k_ix] * q[1], "actual_value": np.nan,
            "model_name": "sum of hourly forecasts", "model_version": ver, "horizon_step": k_ix + 1, "is_operating_hour": np.nan,
            "definition_note": "station-day = sum of the 24 hourly forecasts"}))
        N = D.sum(axis=0); qn = ip["network"][m]
        rows.append(pd.DataFrame({
            "forecast_generated_at": generated_at, "forecast_origin": str(fc["origin"].date()), "target_datetime": dates.strftime("%Y-%m-%d"),
            "target_date": dates.strftime("%Y-%m-%d"), "station_id": None, "forecast_granularity": "daily", "scope": "network_station_derived",
            "target_measure": m, "predicted_value": N, "lower_80": N * (np.array([b[0] for b in net_bands]) if net_bands else qn[0]), "upper_80": N * (np.array([b[1] for b in net_bands]) if net_bands else qn[1]), "actual_value": np.nan,
            "model_name": "sum of station-day forecasts", "model_version": ver, "horizon_step": np.arange(1, len(dates) + 1), "is_operating_hour": np.nan,
            "definition_note": "network-day = sum over stations of station-day forecasts (RTI station boardings definition, NOT the tracker total)"}))
    if tracker_fc is not None:
        rows.append(pd.DataFrame({
            "forecast_generated_at": generated_at, "forecast_origin": str((tracker_fc["date"].iloc[0] - pd.Timedelta(days=1)).date()),
            "target_datetime": tracker_fc["date"].dt.strftime("%Y-%m-%d"), "target_date": tracker_fc["date"].dt.strftime("%Y-%m-%d"), "station_id": None,
            "forecast_granularity": "daily", "scope": "network_tracker", "target_measure": "total_ridership", "predicted_value": tracker_fc["pred"],
            "lower_80": tracker_fc["lo"] if "lo" in tracker_fc else tracker_fc["pred"] * tracker_band[0], "upper_80": tracker_fc["hi"] if "hi" in tracker_fc else tracker_fc["pred"] * tracker_band[1], "actual_value": np.nan,
            "model_name": tracker_model, "model_version": model_version(tracker_model, cfg), "horizon_step": tracker_fc["horizon_step"], "is_operating_hour": np.nan,
            "definition_note": "independent daily tracker series (all ticket modes), modelled separately"}))
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "run_type", "production")
    return df


def backtest_results_frame(bt, cube, dim, ctx, cfg, ip, generated_at: str, tracker_bt_records: pd.DataFrame | None = None, tracker_model: str | None = None) -> pd.DataFrame:
    """Backtest rows (actual_value filled) for the selected pipeline: hourly + station-day + network-day, last fold only for hourly to bound size."""
    sid = dim["station_id"].to_numpy(); H = ctx.horizon
    rows = []
    for fi, o in enumerate(bt["origins"]):
        dates = ctx.dates[o + 1:o + 1 + H]
        for m in ("entries", "exits"):
            P, A = bt["pred"][("pipeline", m, fi)], bt["act"][(m, fi)]
            S = P.shape[0]
            obs = ctx.obs[:, o + 1:o + 1 + H]
            D, AD = P.sum(axis=2), np.nansum(A, axis=2)
            s_ix, k_ix = np.where(obs & ~np.isnan(A).any(axis=2))
            rows.append(pd.DataFrame({"run_type": "backtest", "fold": fi, "forecast_generated_at": generated_at, "forecast_origin": str(ctx.dates[o].date()),
                                      "target_datetime": dates[k_ix].strftime("%Y-%m-%d"), "target_date": dates[k_ix].strftime("%Y-%m-%d"), "station_id": sid[s_ix],
                                      "forecast_granularity": "daily", "scope": "station", "target_measure": m, "predicted_value": D[s_ix, k_ix],
                                      "actual_value": AD[s_ix, k_ix], "model_name": "pipeline", "horizon_step": k_ix + 1}))
            if fi == len(bt["origins"]) - 1:   # hourly rows only for the final fold
                s_ix, k_ix, h_ix = np.where(obs[:, :, None] & ~np.isnan(A))
                rows.append(pd.DataFrame({"run_type": "backtest", "fold": fi, "forecast_generated_at": generated_at, "forecast_origin": str(ctx.dates[o].date()),
                                          "target_datetime": (dates[k_ix] + pd.to_timedelta(h_ix, unit="h")).strftime("%Y-%m-%d %H:00"),
                                          "target_date": dates[k_ix].strftime("%Y-%m-%d"), "station_id": sid[s_ix], "forecast_granularity": "hourly", "scope": "station",
                                          "target_measure": m, "predicted_value": P[s_ix, k_ix, h_ix], "actual_value": A[s_ix, k_ix, h_ix],
                                          "model_name": "pipeline", "horizon_step": k_ix * 24 + h_ix + 1}))
    df = pd.concat(rows, ignore_index=True)
    df["model_version"] = model_version("pipeline", cfg)
    if tracker_bt_records is not None and tracker_model:
        g = tracker_bt_records.query("model == @tracker_model").dropna(subset=["actual"])
        t = pd.DataFrame({"run_type": "backtest", "fold": g["fold"], "forecast_generated_at": generated_at, "forecast_origin": g["origin"].dt.strftime("%Y-%m-%d"),
                          "target_datetime": g["date"].dt.strftime("%Y-%m-%d"), "target_date": g["date"].dt.strftime("%Y-%m-%d"), "station_id": None,
                          "forecast_granularity": "daily", "scope": "network_tracker", "target_measure": "total_ridership", "predicted_value": g["pred"],
                          "actual_value": g["actual"], "model_name": tracker_model, "horizon_step": g["horizon"], "model_version": model_version(tracker_model, cfg)})
        df = pd.concat([df, t], ignore_index=True)
    return df
