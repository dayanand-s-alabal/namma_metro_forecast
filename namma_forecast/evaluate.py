"""FR-08 / FR-09 / SR-03: chronological rolling-origin evaluation, metrics, model comparison and selection."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd

from .etl import Cube, OBS
from .models import MODELS, MODEL_INFO, ENSEMBLE_MEMBERS, Ctx


def make_ctx(cube: Cube, cal: pd.DataFrame, cfg: dict, horizon: int | None = None, trk: dict | None = None) -> tuple[Ctx, dict]:
    """Extend the cube by the forecast horizon and attach calendar features (known in advance)."""
    H = horizon or cfg["horizon_days"]
    ext = pd.date_range(cube.dates[0], periods=cube.D + H, freq="D")
    c = cal.set_index("date").reindex(ext)
    obs = np.zeros((cube.S, cube.D + H), bool)
    obs[:, :cube.D] = cube.status == OBS
    ctx = Ctx(dates=ext, open_idx=cube.open_idx, obs=obs, hol=c["is_holiday"].fillna(0).to_numpy(float), event=c["is_special_event"].fillna(0).to_numpy(float),
              horizon=H, non_op=cfg["non_operating_hours"], profile_window=cfg["evaluation"]["profile_window_days"], xgb=cfg["xgboost"], seed=cfg["seed"])
    if trk is not None:      # tracker actuals (only ever read up to the origin) and per-origin tracker forecasts
        ctx.trk_hist = trk["y"].reindex(ext).to_numpy(float)
        ctx.trk_fc = trk.get("fc", {})
    arrs = {m: np.concatenate([cube.arr(m), np.full((cube.S, H, 24), np.nan)], axis=1) for m in ("entries", "exits")}
    return ctx, arrs


# ------------------------------------------------------------------------------------------ metrics
def metrics(pred: np.ndarray, act: np.ndarray, mape_min: float) -> dict:
    pred, act = np.asarray(pred, float).ravel(), np.asarray(act, float).ravel()
    m = ~np.isnan(act) & ~np.isnan(pred)
    p, a = pred[m], act[m]
    if len(a) == 0:
        return {k: np.nan for k in ("MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "bias", "n", "n_mape_excluded")}
    e = p - a
    keep = a >= mape_min
    den = np.abs(a) + np.abs(p)
    sm = np.where(den > 0, 2 * np.abs(e) / np.where(den > 0, den, 1), 0.0)   # both zero => 0 error
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE": float(np.mean(np.abs(e[keep]) / a[keep]) * 100) if keep.any() else np.nan,
            "WAPE": float(np.abs(e).sum() / max(a.sum(), 1e-9) * 100), "sMAPE": float(sm.mean() * 100),
            "bias": float(e.mean()), "n": int(len(a)), "n_mape_excluded": int((~keep).sum())}


def fold_origins(cube: Cube, n_folds: int, test_days: int) -> list[int]:
    """Origins such that the LAST test window ends on the last observed date; windows are consecutive and chronological."""
    last = cube.D - 1
    return [last - test_days * (n_folds - i) for i in range(n_folds)]


def run_backtests(cube: Cube, cal: pd.DataFrame, cfg: dict, models: list[str] | None = None, log=print, trk: dict | None = None) -> dict:
    """Run every candidate on every fold. Returns {'pred': {(model, measure, fold): (S,H,24)}, 'act': {...}, 'runtime': {...}}."""
    ctx, arrs = make_ctx(cube, cal, cfg, trk=trk)
    H = ctx.horizon
    origins = fold_origins(cube, cfg["evaluation"]["n_folds"], H)
    models = models or list(MODELS)
    out = {"pred": {}, "act": {}, "origins": origins, "runtime": {}, "ctx": ctx}
    for fi, o in enumerate(origins):
        for m in ("entries", "exits"):
            out["act"][(m, fi)] = arrs[m][:, o + 1:o + 1 + H, :].copy()
    for name in models:
        t0 = time.time()
        for fi, o in enumerate(origins):
            for m in ("entries", "exits"):
                A = arrs[m]
                out["pred"][(name, m, fi)] = MODELS[name](ctx, A, o)
        out["runtime"][name] = time.time() - t0
        log(f"  backtest {name:14s} {out['runtime'][name]:6.1f}s")
    if all(k in out["runtime"] for k in ENSEMBLE_MEMBERS):
        for fi, o in enumerate(origins):
            for m in ("entries", "exits"):
                out["pred"][("ensemble", m, fi)] = np.mean([out["pred"][(k, m, fi)] for k in ENSEMBLE_MEMBERS], axis=0)
        out["runtime"]["ensemble"] = sum(out["runtime"][k] for k in ENSEMBLE_MEMBERS)
    return out


def _valid_hourly(ctx, act, fi_origin, S, H):
    """station-hours that are observed (open, date present) and operating."""
    obs = ctx.obs[:, fi_origin + 1: fi_origin + 1 + H][:, :, None] & ~np.isnan(act)
    obs[:, :, ctx.non_op] = False
    return obs


def evaluate(bt: dict, cube: Cube, cfg: dict) -> dict:
    """Hourly (station-hour) and daily (station-day, network-day) metrics per model, measure and fold."""
    ctx = bt["ctx"]; H = ctx.horizon; ev = cfg["evaluation"]
    rows_h, rows_d, rows_n = [], [], []
    models = sorted({k[0] for k in bt["pred"]}, key=lambda k: list(MODEL_INFO).index(k))
    for name in models:
        for m in ("entries", "exits"):
            for fi, o in enumerate(bt["origins"]):
                pred, act = bt["pred"][(name, m, fi)], bt["act"][(m, fi)]
                v = _valid_hourly(ctx, act, o, cube.S, H)
                rows_h.append({"model": name, "measure": m, "fold": fi, "origin": str(ctx.dates[o].date()), **metrics(pred[v], act[v], ev["mape_min_actual"])})
                # daily: station-day sums over days where the station is open & observed
                dv = ctx.obs[:, o + 1:o + 1 + H] & ~np.isnan(act).any(axis=2)
                pd_, ad_ = pred.sum(axis=2), np.nansum(act, axis=2)
                rows_d.append({"model": name, "measure": m, "fold": fi, "origin": str(ctx.dates[o].date()), **metrics(pd_[dv], ad_[dv], ev["daily_mape_min_actual"])})
                # network-day: sum across stations open that day
                pn = np.where(dv, pd_, 0).sum(axis=0); an = np.where(dv, ad_, 0).sum(axis=0)
                rows_n.append({"model": name, "measure": m, "fold": fi, "origin": str(ctx.dates[o].date()), **metrics(pn, an, 1.0)})
    return {"hourly": pd.DataFrame(rows_h), "daily": pd.DataFrame(rows_d), "network": pd.DataFrame(rows_n)}


def pool(bt: dict, cube: Cube, cfg: dict, level: str) -> pd.DataFrame:
    """Pooled-over-folds metrics per model & measure (+ per-fold WAPE spread = stability, runtime)."""
    ctx = bt["ctx"]; H = ctx.horizon; ev = cfg["evaluation"]
    models = sorted({k[0] for k in bt["pred"]}, key=lambda k: list(MODEL_INFO).index(k))
    rows = []
    for name in models:
        for m in ("entries", "exits"):
            P, A, folds = [], [], []
            for fi, o in enumerate(bt["origins"]):
                pred, act = bt["pred"][(name, m, fi)], bt["act"][(m, fi)]
                if level == "hourly":
                    v = _valid_hourly(ctx, act, o, cube.S, H); p, a, thr = pred[v], act[v], ev["mape_min_actual"]
                else:
                    dv = ctx.obs[:, o + 1:o + 1 + H] & ~np.isnan(act).any(axis=2)
                    pdy, ady = pred.sum(axis=2), np.nansum(act, axis=2)
                    if level == "daily":
                        p, a, thr = pdy[dv], ady[dv], ev["daily_mape_min_actual"]
                    else:
                        p, a, thr = np.where(dv, pdy, 0).sum(axis=0), np.where(dv, ady, 0).sum(axis=0), 1.0
                P.append(p); A.append(a)
                folds.append(metrics(p, a, thr)["WAPE"])
            r = metrics(np.concatenate(P), np.concatenate(A), thr)
            r.update({"model": name, "label": MODEL_INFO[name][0], "family": MODEL_INFO[name][1], "measure": m,
                      "wape_fold_mean": float(np.mean(folds)), "wape_fold_std": float(np.std(folds)), "fold_wape": [round(x, 2) for x in folds],
                      "runtime_s": round(bt["runtime"][name], 2)})
            rows.append(r)
    return pd.DataFrame(rows)


def select(tbl: pd.DataFrame) -> pd.DataFrame:
    """Rank models: mean fold WAPE over both measures (primary); fold-to-fold std and runtime are shown as tie-breakers."""
    g = tbl.groupby(["model", "label", "family"]).agg(score=("wape_fold_mean", "mean"), stability=("wape_fold_std", "mean"), runtime_s=("runtime_s", "first")).reset_index()
    g = g.sort_values(["score", "stability"]).reset_index(drop=True)
    g["rank"] = np.arange(1, len(g) + 1)
    base = g.loc[g.model == "snaive_week", "score"]
    g["vs_seasonal_naive_pct"] = (g["score"] / base.iloc[0] - 1) * 100 if len(base) else np.nan
    return g
