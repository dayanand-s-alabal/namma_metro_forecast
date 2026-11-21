"""FR-12: forecasting of the INDEPENDENT daily network series (tracker) and reconciliation with station-derived totals.

The tracker (all ticket modes, system-wide) is a different definition from the station-hour data (boardings by RTI).
It is therefore modelled on its own and only *compared* with station-derived totals via a documented scaling ratio.
"""
from __future__ import annotations
import time, warnings
import numpy as np
import pandas as pd

from .evaluate import metrics

TRACKER_MODELS = {
    "naive": "Naive (last observed day)", "snaive7": "Seasonal naive (same weekday last week)", "ma4w": "Moving average (same weekday, 4 wks)",
    "ses": "Simple exp. smoothing", "holt": "Holt linear trend (damped)", "hw": "Holt-Winters / ETS (weekly)",
    "sarima": "SARIMA(1,0,1)(0,1,1)7", "xgb": "XGBoost (lags>=7d, level-normalised)",
    "hw_h": "Holt-Winters, holiday-adjusted", "sarima_h": "SARIMA, holiday-adjusted",
}
FAMILY = {"naive": "Baseline", "snaive7": "Baseline", "ma4w": "Moving average", "ses": "Statistical", "holt": "Statistical", "hw": "Statistical",
          "sarima": "Statistical", "xgb": "Machine learning", "hw_h": "Statistical + calendar", "sarima_h": "Statistical + calendar"}


def prepare_series(tracker: pd.DataFrame, cal: pd.DataFrame, end: str | None = None, future: int = 7) -> pd.DataFrame:
    """Continuous daily frame (observed total or NaN) with calendar features for the observed range + 7 future days."""
    s = tracker.set_index("date")["official_daily_total"].astype(float)
    idx = pd.date_range(s.index.min(), (pd.Timestamp(end) if end else s.index.max()) + pd.Timedelta(days=future), freq="D")
    df = pd.DataFrame({"y": s.reindex(idx)}, index=idx)
    c = cal.set_index("date").reindex(idx)
    df["dow"] = idx.dayofweek
    df["hol"] = c["is_holiday"].fillna(0).to_numpy(float)
    df["event"] = c["is_special_event"].fillna(0).to_numpy(float)
    return df


def _impute(y: np.ndarray) -> np.ndarray:
    out = y.copy()
    n = len(y)
    for d in np.where(np.isnan(y))[0]:
        v = [y[d + k * 7] for k in (-3, -2, -1, 1, 2, 3) if 0 <= d + k * 7 < n and not np.isnan(y[d + k * 7])]
        if v:
            out[d] = np.mean(v)
    if np.isnan(out).any():
        out = pd.Series(out).interpolate(limit_direction="both").to_numpy()
    return out


def _forecast(name: str, df: pd.DataFrame, o: int, H: int, cfg: dict) -> np.ndarray:
    y = df["y"].to_numpy()[: o + 1]
    win = cfg["network_tracker"].get("fit_window_days", 240)
    lo = max(0, o + 1 - win)
    if name == "naive":
        last = y[~np.isnan(y)][-1]
        return np.repeat(last, H)
    if name in ("snaive7", "ma4w"):
        out = []
        for k in range(1, H + 1):
            t = o + k
            vals = [df["y"].to_numpy()[t - 7 * j] for j in range(1, 5) if 0 <= t - 7 * j <= o and not np.isnan(df["y"].to_numpy()[t - 7 * j])]
            if not vals:
                vals = [y[~np.isnan(y)][-1]]
            out.append(vals[0] if name == "snaive7" else np.mean(vals))
        return np.array(out)
    if name in ("hw_h", "sarima_h"):
        # public holidays distort the weekly seasonal pattern the next weeks inherit: remove them from the fit (seasonal fill),
        # forecast the normal pattern, then re-apply a holiday multiplier (median holiday / seasonal-expectation ratio, history only)
        hol = df["hol"].to_numpy(); yy = df["y"].to_numpy()[: o + 1].copy(); hm = hol[: o + 1] > 0
        adj = yy.copy(); adj[hm] = np.nan
        yi = _impute(adj)
        ok = hm & ~np.isnan(yy)
        mult = float(np.clip(np.median(yy[ok] / yi[ok]), 0.5, 1.05)) if ok.sum() >= 3 else 1.0
        d2 = df.copy(); v = d2["y"].to_numpy().copy(); v[: o + 1] = yi; v[o + 1:] = np.nan; d2["y"] = v
        f = _forecast(name[:-2], d2, o, H, cfg)
        return f * np.where(hol[o + 1: o + 1 + H] > 0, mult, 1.0)
    if name in ("ses", "holt", "hw", "sarima"):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        yi = _impute(y)[lo:]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if name == "ses":
                    f = SimpleExpSmoothing(yi, initialization_method="estimated").fit().forecast(H)
                elif name == "holt":
                    f = ExponentialSmoothing(yi, trend="add", damped_trend=True, initialization_method="estimated").fit().forecast(H)
                elif name == "hw":
                    f = ExponentialSmoothing(yi, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7, initialization_method="estimated").fit().forecast(H)
                else:
                    f = SARIMAX(yi, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=60).forecast(H)
            f = np.asarray(f, float)
            if np.isfinite(f).all() and f.max() < 3 * yi.max() and f.min() > 0:
                return f
        except Exception:
            pass
        return _forecast("snaive7", df, o, H, cfg)
    if name == "xgb":
        import xgboost as xgb
        p = cfg["xgboost"]
        Y = df["y"].to_numpy().copy(); Y[o + 1:] = np.nan                         # nothing after the origin is visible
        n = len(Y)
        def lag(k): out = np.full(n, np.nan); out[k:] = Y[:n - k]; return out
        l7, l14, l21, l28 = lag(7), lag(14), lag(21), lag(28)
        lvl = np.full(n, np.nan)
        for d in range(n):
            seg = Y[max(d - 13, 0): d - 6] if d - 6 > 0 else np.array([])
            if len(seg) and not np.isnan(seg).all():
                lvl[d] = np.nanmean(seg)
        with np.errstate(all="ignore"):
            lm = np.nanmean(np.stack([l7, l14, l21, l28]), axis=0)
            X = np.column_stack([l7 / lvl, l14 / lvl, l21 / lvl, lm / lvl, df["dow"], df["hol"], df["event"], (df["dow"] >= 5).astype(float)])
            tgt = Y / lvl
        tr = np.arange(max(28, o + 1 - 365), o + 1)
        ok = ~np.isnan(tgt[tr]) & ~np.isnan(lvl[tr]) & ~np.isnan(X[tr, 3])
        mdl = xgb.XGBRegressor(n_estimators=250, learning_rate=p["learning_rate"], max_depth=4, min_child_weight=3, subsample=p["subsample"],
                               colsample_bytree=p["colsample_bytree"], tree_method="hist", n_jobs=1, random_state=cfg["seed"], verbosity=0)
        mdl.fit(X[tr][ok], tgt[tr][ok])
        fd = np.arange(o + 1, o + 1 + H)
        r = mdl.predict(X[fd])
        base = lvl[fd]
        base = np.where(np.isnan(base), np.nanmean(Y[max(0, o - 6): o + 1]), base)
        return np.clip(r * base, 0, None)
    raise ValueError(name)


def backtest(df: pd.DataFrame, cfg: dict, log=print) -> dict:
    nt = cfg["network_tracker"]; H = cfg["horizon_days"]
    last_obs = df["y"].last_valid_index()
    first_origin = _first_origin(df, nt)
    origins = [o for o in range(first_origin, df.index.get_loc(last_obs), nt["step_days"]) if o + 1 < len(df)]
    rows, preds, runtime = [], {}, {}
    yv = df["y"].to_numpy()
    for name in TRACKER_MODELS:
        t0 = time.time()
        for fi, o in enumerate(origins):
            f = _forecast(name, df, o, H, cfg)
            for k in range(H):
                a = yv[o + 1 + k] if o + 1 + k < len(yv) else np.nan
                preds.setdefault(name, []).append((fi, o, k + 1, df.index[o + 1 + k], f[k], a))
        runtime[name] = time.time() - t0
        log(f"  tracker backtest {name:8s} {runtime[name]:5.1f}s")
    recs = []
    for name, lst in preds.items():
        for fi, o, k, d, p, a in lst:
            recs.append({"model": name, "fold": fi, "origin": df.index[o], "horizon": k, "date": d, "pred": p, "actual": a})
    return {"records": pd.DataFrame(recs), "origins": origins, "runtime": runtime}


def evaluate(bt: dict, cfg: dict) -> pd.DataFrame:
    r = bt["records"].dropna(subset=["actual"])
    nt = cfg["network_tracker"]
    rows = []
    n_f = r["fold"].max() + 1
    for name, g in r.groupby("model"):
        full = metrics(g["pred"], g["actual"], 1.0)
        rec = g[g["fold"] >= n_f - nt.get("recent_folds", 8)]
        recm = metrics(rec["pred"], rec["actual"], 1.0)
        fw = [metrics(x["pred"], x["actual"], 1.0)["WAPE"] for _, x in g.groupby("fold") if len(x) >= 3]
        rows.append({"model": name, "label": TRACKER_MODELS[name], "family": FAMILY[name], **full, "recent_MAPE": recm["MAPE"], "recent_MAE": recm["MAE"],
                     "recent_WAPE": recm["WAPE"], "wape_fold_mean": float(np.mean(fw)), "wape_fold_std": float(np.std(fw)),
                     "runtime_s": round(bt["runtime"][name], 2), "folds": int(g["fold"].nunique())})
    t = pd.DataFrame(rows).sort_values("wape_fold_mean").reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    return t


def interval_quantiles(bt: dict, model: str, coverage: float) -> tuple[float, float]:
    g = bt["records"].query("model == @model").dropna(subset=["actual"])
    ratio = (g["actual"] / g["pred"]).to_numpy()
    a = (1 - coverage) / 2
    return float(np.quantile(ratio, a)), float(np.quantile(ratio, 1 - a))


def _first_origin(df: pd.DataFrame, nt: dict) -> int:
    """Configured first origin, or (for an earlier run date / short series) the origin 26 weeks before the last observation."""
    last = df.index.get_loc(df["y"].last_valid_index())
    ts = pd.Timestamp(nt["first_origin"])
    if ts in df.index and df.index.get_loc(ts) < last - 28:
        return df.index.get_loc(ts)
    return max(120, last - 26 * 7)


def backtest_model(df: pd.DataFrame, cfg: dict, model: str, H: int) -> pd.DataFrame:
    """Backtest only the selected tracker model over a longer horizon H (needed because the run date can be days after the last tracker date)."""
    nt = cfg["network_tracker"]
    last_obs = df["y"].last_valid_index()
    first_origin = _first_origin(df, nt)
    origins = [o for o in range(first_origin, df.index.get_loc(last_obs), nt["step_days"]) if o + 1 < len(df)]
    yv = df["y"].to_numpy(); recs = []
    for fi, o in enumerate(origins):
        f = _forecast(model, df, o, H, cfg)
        for k in range(H):
            a = yv[o + 1 + k] if o + 1 + k < len(yv) else np.nan
            recs.append({"model": model, "fold": fi, "origin": df.index[o], "horizon": k + 1, "date": df.index[o + 1 + k], "pred": f[k], "actual": a})
    return pd.DataFrame(recs)


def step_bands(recs: pd.DataFrame, coverage: float, max_step: int) -> list[tuple[float, float]]:
    """Empirical actual/forecast ratio quantiles by horizon bucket (1-7, 8-14); beyond 14 days the 8-14 band is widened by sqrt(step/14)."""
    g = recs.dropna(subset=["actual"]); a = (1 - coverage) / 2
    def q(lo, hi):
        r = (g[(g.horizon >= lo) & (g.horizon <= hi)]["actual"] / g[(g.horizon >= lo) & (g.horizon <= hi)]["pred"]).to_numpy()
        return float(np.quantile(r, a)), float(np.quantile(r, 1 - a))
    b1, b2 = q(1, 7), q(8, 14)
    out = []
    for s in range(1, max_step + 1):
        if s <= 7: out.append(b1)
        elif s <= 14: out.append(b2)
        else:
            w = (s / 14) ** 0.5
            out.append((1 - (1 - b2[0]) * w, 1 + (b2[1] - 1) * w))
    return out


def forecast_next(df: pd.DataFrame, model: str, cfg: dict, H: int | None = None) -> pd.DataFrame:
    o = df.index.get_loc(df["y"].last_valid_index())
    H = H or cfg["horizon_days"]
    f = _forecast(model, df, o, H, cfg)
    return pd.DataFrame({"date": df.index[o + 1:o + 1 + H], "pred": f, "horizon_step": np.arange(1, H + 1)})
