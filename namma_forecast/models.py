"""FR-06 / FR-07: leakage-safe features and forecasting models for station-level hourly series.

Every model has the signature  f(ctx, A, o) -> (S, H, 24)  where
    A   : (S, Dx, 24) observation cube (NaN = not observed). Only A[:, :o+1] may be read.
    o   : index of the forecast origin (last observed day). Forecast days are o+1 .. o+H.
All models return non-negative counts with non-operating hours fixed to 0.
Top-down models forecast station-day totals and distribute them with day-of-week hour-share profiles.
"""
from __future__ import annotations
import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


@dataclass
class Ctx:
    dates: pd.DatetimeIndex        # length Dx (observed range extended by the forecast horizon)
    open_idx: np.ndarray           # (S,) first open day index
    obs: np.ndarray                # (S, Dx) bool, True where the station-day is an observed, open day
    hol: np.ndarray                # (Dx,) public holiday flag (known in advance)
    event: np.ndarray              # (Dx,) special event / disruption flag
    horizon: int
    non_op: list
    profile_window: int
    xgb: dict
    seed: int
    trk_hist: np.ndarray | None = None   # (Dx,) tracker (all-ticket-mode) daily totals aligned with dates, NaN where unknown
    trk_fc: dict | None = None           # {origin index: (H,) tracker forecast made with data up to that origin}

    @property
    def dow(self): return self.dates.dayofweek.to_numpy()
    @property
    def S(self): return len(self.open_idx)


MODEL_INFO = {
    "naive_day":   ("Naive (repeat last day)",                  "Baseline"),
    "snaive_week": ("Seasonal naive (same hour last week)",     "Baseline"),
    "ma_3wk":      ("Moving average (same hour-of-week, 3 wks)", "Moving average"),
    "ma7_td":      ("Moving average 7-day (top-down)",          "Moving average"),
    "ses_td":      ("Simple exp. smoothing (top-down)",         "Statistical"),
    "holt_td":     ("Holt linear trend (top-down)",             "Statistical"),
    "hw_td":       ("Holt-Winters / ETS weekly (top-down)",     "Statistical"),
    "sarima_td":   ("SARIMA(1,0,1)(0,1,1)7 (top-down)",         "Statistical"),
    "xgb_daily_td": ("XGBoost daily (top-down)",                "Machine learning"),
    "xgb_hourly":  ("XGBoost hourly (global, lag-168h)",        "Machine learning"),
    "trk_td":      ("Tracker-driven top-down (tracker forecast x station share x hour profile)", "Hybrid"),
    "ensemble":    ("Ensemble mean (MA-3wk + HW + XGB-hourly)",   "Ensemble"),
    "pipeline":    ("Selected pipeline (best hourly shape x best daily totals)", "Pipeline"),
}
ENSEMBLE_MEMBERS = ["ma_3wk", "hw_td", "xgb_hourly"]


# ------------------------------------------------------------------------------------------ helpers
def daily_totals(A: np.ndarray) -> np.ndarray:
    return A.sum(axis=2)            # NaN-propagating on purpose: partial / missing days stay NaN


def shift(X: np.ndarray, k: int, axis: int = 1) -> np.ndarray:
    """X[:, d-k] aligned at d (NaN fill)."""
    out = np.full_like(X, np.nan, dtype=float)
    if k < X.shape[axis]:
        src = [slice(None)] * X.ndim; dst = [slice(None)] * X.ndim
        src[axis] = slice(0, X.shape[axis] - k); dst[axis] = slice(k, None)
        out[tuple(dst)] = X[tuple(src)]
    return out


def impute_daily(T: np.ndarray, o: int, ctx: Ctx) -> np.ndarray:
    """Fill missing station-days (after opening, up to origin) with the mean of same-weekday observations within +-3 weeks
    (only days <= origin). Used for statistical model *fitting* only; never for scoring."""
    Ti = T[:, : o + 1].copy()
    S = Ti.shape[0]
    for s in range(S):
        for d in range(max(ctx.open_idx[s], 0), o + 1):
            if np.isnan(Ti[s, d]) and ctx.open_idx[s] >= 0:
                vals = [T[s, d + k * 7] for k in (-3, -2, -1, 1, 2, 3) if 0 <= d + k * 7 <= o and not np.isnan(T[s, d + k * 7])]
                if vals:
                    Ti[s, d] = np.mean(vals)
        # leading gap values (after opening) already handled; remaining NaN -> linear interpolation
        row = Ti[s, max(ctx.open_idx[s], 0):]
        if np.isnan(row).any() and (~np.isnan(row)).any():
            Ti[s, max(ctx.open_idx[s], 0):] = pd.Series(row).interpolate(limit_direction="both").to_numpy()
    return Ti


def hour_shares(A: np.ndarray, T: np.ndarray, o: int, ctx: Ctx) -> np.ndarray:
    """(S, 7, 24) share of the daily total falling in each hour, per weekday, from the last `profile_window` observed days."""
    lo = max(0, o - ctx.profile_window + 1)
    S = A.shape[0]
    dow = ctx.dow
    sh = np.full((S, 7, 24), np.nan)
    days = np.arange(lo, o + 1)
    ok = ctx.obs[:, lo:o + 1] & ~np.isnan(T[:, lo:o + 1])
    A_w = np.nan_to_num(A[:, lo:o + 1, :]); T_w = np.nan_to_num(T[:, lo:o + 1])
    pooled_num = (A_w * ok[:, :, None]).sum(axis=1); pooled_den = (T_w * ok).sum(axis=1)
    with np.errstate(all="ignore"):
        pooled = pooled_num / pooled_den[:, None]
        net_num = pooled_num.sum(0); net_share = net_num / net_num.sum()
        for k in range(7):
            m = ok & (dow[lo:o + 1] == k)[None, :]
            num = (A_w * m[:, :, None]).sum(axis=1); den = (T_w * m).sum(axis=1)
            sh[:, k, :] = num / den[:, None]
    # fallbacks: dow without data -> pooled station profile -> network profile
    bad = np.isnan(sh).any(axis=2) | (np.nansum(sh, axis=2) <= 0)
    for s, k in zip(*np.where(bad)):
        sh[s, k, :] = pooled[s] if not np.isnan(pooled[s]).any() and pooled[s].sum() > 0 else net_share
    sh[:, :, ctx.non_op] = 0.0
    sh = sh / sh.sum(axis=2, keepdims=True)
    return sh


def _finish(F: np.ndarray, ctx: Ctx) -> np.ndarray:
    F = np.nan_to_num(F, nan=0.0)
    F = np.clip(F, 0, None)
    F[:, :, ctx.non_op] = 0.0
    return F


def _topdown(daily: np.ndarray, A, T, o, ctx) -> np.ndarray:
    H = ctx.horizon
    sh = hour_shares(A, T, o, ctx)
    dows = ctx.dow[o + 1:o + 1 + H]
    F = np.stack([daily[:, k][:, None] * sh[:, dows[k], :] for k in range(H)], axis=1)
    return _finish(F, ctx)


# ------------------------------------------------------------------------------------------ simple models
def naive_day(ctx, A, o):
    last = A[:, o, :]
    return _finish(np.repeat(last[:, None, :], ctx.horizon, axis=1), ctx)


def _weekly_lags(A, o, H, n=3):
    out = []
    for k in range(1, H + 1):
        t = o + k
        out.append(np.stack([A[:, t - 7 * j, :] if 0 <= t - 7 * j <= o else np.full(A[:, 0, :].shape, np.nan) for j in range(1, n + 1)], axis=0))
    return out  # list over k of (n, S, 24)


def snaive_week(ctx, A, o):
    H = ctx.horizon
    lag = _weekly_lags(A, o, H)
    F = np.empty((A.shape[0], H, 24))
    fallback = A[:, o, :]
    for k in range(H):
        v = np.full(A[:, 0, :].shape, np.nan)
        for j in range(3):                      # first available of 1, 2, 3 weeks back
            v = np.where(np.isnan(v), lag[k][j], v)
        F[:, k, :] = np.where(np.isnan(v), fallback, v)
    return _finish(F, ctx)


def ma_3wk(ctx, A, o):
    H = ctx.horizon
    lag = _weekly_lags(A, o, H)
    F = np.empty((A.shape[0], H, 24))
    fallback = A[:, o, :]
    for k in range(H):
        m = np.nanmean(lag[k], axis=0)
        F[:, k, :] = np.where(np.isnan(m), fallback, m)
    return _finish(F, ctx)


# ------------------------------------------------------------------------------------------ top-down statistical
def _station_stat_forecast(kind: str, y: np.ndarray, H: int) -> np.ndarray | None:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if kind == "ses":
                r = SimpleExpSmoothing(y, initialization_method="estimated").fit()
            elif kind == "holt":
                r = ExponentialSmoothing(y, trend="add", damped_trend=True, initialization_method="estimated").fit()
            elif kind == "hw":
                if len(y) < 14:
                    return None
                r = ExponentialSmoothing(y, trend="add", damped_trend=True, seasonal="add", seasonal_periods=7,
                                         initialization_method="estimated").fit()
            elif kind == "sarima":
                if len(y) < 21:
                    return None
                r = SARIMAX(y, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7), enforce_stationarity=False,
                            enforce_invertibility=False).fit(disp=False, maxiter=40)
            else:
                raise ValueError(kind)
            f = np.asarray(r.forecast(H), float)
        # numerical-divergence guard: reject non-finite or absurd forecasts (> 3x series maximum) -> seasonal-naive fallback
        return f if np.isfinite(f).all() and f.max() <= 3 * max(float(np.max(y)), 1.0) else None
    except Exception:
        return None


def _stat_td(kind):
    def f(ctx, A, o):
        H = ctx.horizon
        T = daily_totals(A[:, :o + 1])
        T_full = np.full((A.shape[0], A.shape[1]), np.nan); T_full[:, :o + 1] = T
        Ti = impute_daily(T_full, o, ctx)
        daily = np.full((A.shape[0], H), np.nan)
        sn = snaive_week(ctx, A, o).sum(axis=2)           # fall-back when a series is too short / fit fails
        for s in range(A.shape[0]):
            if ctx.open_idx[s] < 0 or ctx.open_idx[s] > o:
                continue
            y = Ti[s, ctx.open_idx[s]:o + 1]
            f = _station_stat_forecast(kind, y, H) if len(y) >= 7 else None
            daily[s] = f if f is not None else sn[s]
        return _topdown(daily, A[:, :o + 1], T_full[:, :o + 1], o, ctx)
    return f


def ma7_td(ctx, A, o):
    T = daily_totals(A[:, :o + 1]); H = ctx.horizon
    Ti = impute_daily(T, o, ctx)
    daily = np.repeat(np.nanmean(Ti[:, max(0, o - 6):o + 1], axis=1)[:, None], H, axis=1)
    return _topdown(daily, A[:, :o + 1], T, o, ctx)


# ------------------------------------------------------------------------------------------ XGBoost
def _xgb(ctx):
    import xgboost as xgb
    p = ctx.xgb
    return xgb.XGBRegressor(objective="count:poisson", n_estimators=p["n_estimators"], learning_rate=p["learning_rate"], max_depth=p["max_depth"],
                            min_child_weight=p["min_child_weight"], subsample=p["subsample"], colsample_bytree=p["colsample_bytree"],
                            tree_method="hist", n_jobs=p["n_jobs"], random_state=ctx.seed, verbosity=0)


def _roll_nanmean(T: np.ndarray, lo_off: int, hi_off: int) -> np.ndarray:
    """out[:, d] = nanmean(T[:, d-hi_off : d-lo_off+1]) using only days strictly before d-lo_off+1 (offsets >= 7)."""
    S, D = T.shape
    out = np.full((S, D), np.nan)
    for d in range(D):
        a, b = d - hi_off, d - lo_off
        if b < 0:
            continue
        seg = T[:, max(a, 0):b + 1]
        if seg.shape[1]:
            out[:, d] = np.nanmean(seg, axis=1)
    return out


def _calendar_feats(ctx, D):
    return ctx.dow[:D], ctx.hol[:D], ctx.event[:D]


def xgb_hourly(ctx, A, o):
    """Global hourly model with lags >= 168h only (so any step of a 168-step horizon is a legal direct forecast)."""
    Dx, S, H = A.shape[1], A.shape[0], ctx.horizon
    Aw = A.copy(); Aw[:, o + 1:, :] = np.nan                    # hard guarantee: nothing after the origin is visible
    T = daily_totals(Aw)
    l1, l2, l3 = shift(Aw, 7), shift(Aw, 14), shift(Aw, 21)
    with np.errstate(all="ignore"):
        lm = np.nanmean(np.stack([l1, l2, l3]), axis=0)
    T7 = shift(T, 7)
    lvl = _roll_nanmean(T, 7, 13); lvl_prev = _roll_nanmean(T, 14, 20)
    dow, hol, ev = _calendar_feats(ctx, Dx)
    dso = np.arange(Dx)[None, :] - ctx.open_idx[:, None]
    def feats(d_idx):
        d = np.asarray(d_idx)
        n = len(d)
        cols = [
            l1[:, d, :], l2[:, d, :], l3[:, d, :], lm[:, d, :],
            np.broadcast_to(T7[:, d, None], (S, n, 24)), np.broadcast_to(lvl[:, d, None], (S, n, 24)),
            np.broadcast_to(lvl_prev[:, d, None], (S, n, 24)),
            np.broadcast_to(np.clip(dso[:, d, None], 0, 60), (S, n, 24)),
            np.broadcast_to(np.arange(24)[None, None, :], (S, n, 24)), np.broadcast_to(dow[d][None, :, None], (S, n, 24)),
            np.broadcast_to(hol[d][None, :, None], (S, n, 24)), np.broadcast_to(ev[d][None, :, None], (S, n, 24)),
        ]
        X = np.stack([c.astype(float) for c in cols], axis=-1)       # (S, n, 24, F)
        with np.errstate(all="ignore"):
            ratio = (X[..., 5] / X[..., 6])[..., None]
            share7 = (X[..., 0] / (X[..., 4] + 1.0))[..., None]
        return np.concatenate([X, ratio, share7], axis=-1)
    train_d = np.arange(7, o + 1)
    X = feats(train_d); y = Aw[:, train_d, :]
    oper = ~np.isin(np.arange(24), ctx.non_op)[None, None, :]
    ok = ctx.obs[:, train_d][:, :, None] & ~np.isnan(y) & oper & ~(np.isnan(X[..., 0]) & np.isnan(X[..., 1]))
    mdl = _xgb(ctx).fit(X[ok], y[ok])
    fd = np.arange(o + 1, o + 1 + H)
    Xf = feats(fd)
    F = mdl.predict(Xf.reshape(-1, Xf.shape[-1])).reshape(S, H, 24)
    return _finish(F, ctx)


def xgb_daily_td(ctx, A, o):
    Dx, S, H = A.shape[1], A.shape[0], ctx.horizon
    Aw = A.copy(); Aw[:, o + 1:, :] = np.nan
    T = daily_totals(Aw)
    t1, t2, t3 = shift(T, 7), shift(T, 14), shift(T, 21)
    with np.errstate(all="ignore"):
        tm = np.nanmean(np.stack([t1, t2, t3]), axis=0)
    lvl = _roll_nanmean(T, 7, 13); lvl_prev = _roll_nanmean(T, 14, 20)
    dow, hol, ev = _calendar_feats(ctx, Dx)
    dso = np.clip(np.arange(Dx)[None, :] - ctx.open_idx[:, None], 0, 60)
    def feats(d):
        n = len(d)
        with np.errstate(all="ignore"):
            r = lvl[:, d] / lvl_prev[:, d]
        cols = [t1[:, d], t2[:, d], t3[:, d], tm[:, d], lvl[:, d], lvl_prev[:, d], r, dso[:, d],
                np.broadcast_to(dow[d][None, :], (S, n)), np.broadcast_to(hol[d][None, :], (S, n)), np.broadcast_to(ev[d][None, :], (S, n))]
        return np.stack([c.astype(float) for c in cols], axis=-1)
    td = np.arange(7, o + 1)
    X = feats(td); y = T[:, td]
    ok = ctx.obs[:, td] & ~np.isnan(y) & ~np.isnan(X[..., 3])
    mdl = _xgb(ctx).fit(X[ok], y[ok])
    fd = np.arange(o + 1, o + 1 + H)
    daily = mdl.predict(feats(fd).reshape(-1, X.shape[-1])).reshape(S, H)
    return _topdown(daily, Aw[:, :o + 1], T[:, :o + 1], o, ctx)


# ------------------------------------------------------------------------------------------ tracker-driven top-down
def _daytype(dow, hol):
    """0 weekday, 1 Saturday, 2 Sunday or public holiday."""
    return np.where(np.asarray(hol) > 0, 2, np.where(np.asarray(dow) == 6, 2, np.where(np.asarray(dow) == 5, 1, 0))).astype(int)


def trk_params(ctx: Ctx, A: np.ndarray, o: int):
    """Learn from the last `profile_window` observed days up to origin o (A[:, :o+1] only):
       r[dt]        station-derived network total / tracker total (all ticket modes), by day type
       share[S, dt] each station's share of the network total
       prof[S,dt,24] hour-of-day share of the station-day total"""
    lo = max(0, o + 1 - ctx.profile_window); S = A.shape[0]
    days = np.arange(lo, o + 1)
    Aw = A[:, lo:o + 1, :]
    ok = ctx.obs[:, lo:o + 1] & ~np.isnan(Aw).any(axis=2)
    day = np.where(ok, np.nan_to_num(Aw).sum(axis=2), np.nan)
    opened = (ctx.open_idx[:, None] >= 0) & (days[None, :] >= ctx.open_idx[:, None])
    full = (ok | ~opened).all(axis=0) & opened.any(axis=0)            # every open station observed that day
    net = np.where(full, np.nansum(day, axis=0), np.nan)
    th = ctx.trk_hist[lo:o + 1]
    dt = _daytype(ctx.dow[lo:o + 1], ctx.hol[lo:o + 1])
    good = full & ~np.isnan(th) & (th > 0)
    if not good.any():
        raise ValueError("trk_td needs overlapping station and tracker data")
    r = np.array([np.median(net[good & (dt == k)] / th[good & (dt == k)]) if (good & (dt == k)).sum() >= 2 else np.nan for k in range(3)])
    r = np.where(np.isnan(r), np.median(net[good] / th[good]), r)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sh_day = day / net[None, :]
        all_sh = np.nanmean(np.where(full[None, :], sh_day, np.nan), axis=1)
        share = np.stack([np.nanmean(np.where((full & (dt == k))[None, :], sh_day, np.nan), axis=1) for k in range(3)], axis=1)
        share = np.where(np.isnan(share), all_sh[:, None], share)
        share = np.nan_to_num(share)
        Aw0 = np.nan_to_num(Aw)
        pool_num = (Aw0 * (ok)[:, :, None]).sum(axis=1)
        pool = pool_num / np.nansum(np.where(ok, day, np.nan), axis=1)[:, None]
        netp = pool_num.sum(0) / max(pool_num.sum(), 1e-9)
        prof = np.zeros((S, 3, 24))
        for k in range(3):
            m = ok & (dt == k)[None, :]
            num = (Aw0 * m[:, :, None]).sum(axis=1); den = np.where(m, day, 0).sum(axis=1)
            p = num / den[:, None]
            bad = ~np.isfinite(p).all(axis=1) | (den <= 0)
            p[bad] = pool[bad]
            bad = ~np.isfinite(p).all(axis=1) | (p.sum(axis=1) <= 0)
            p[bad] = netp
            prof[:, k, :] = p
    prof[:, :, ctx.non_op] = 0.0
    prof = prof / np.maximum(prof.sum(axis=2, keepdims=True), 1e-12)
    return r, share, prof


def trk_td_days(ctx: Ctx, A: np.ndarray, o: int, days: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Station-hour values for arbitrary day indices `days` (past or future) given the network tracker totals T for those days."""
    r, share, prof = trk_params(ctx, A, o)
    S = A.shape[0]
    F = np.zeros((S, len(days), 24))
    for j, d in enumerate(days):
        dt = int(_daytype(ctx.dow[d], ctx.hol[d]))
        open_t = (ctx.open_idx >= 0) & (ctx.open_idx <= d)
        s = np.where(open_t, share[:, dt], 0.0)
        s = s / max(s.sum(), 1e-12)
        F[:, j, :] = (r[dt] * T[j] * s)[:, None] * prof[:, dt, :]
    return _finish(F, ctx)


def trk_td(ctx, A, o):
    T = np.asarray(ctx.trk_fc[o], float)[:ctx.horizon]
    return trk_td_days(ctx, A, o, np.arange(o + 1, o + 1 + len(T)), T)


MODELS = {
    "naive_day": naive_day, "snaive_week": snaive_week, "ma_3wk": ma_3wk, "ma7_td": ma7_td,
    "ses_td": _stat_td("ses"), "holt_td": _stat_td("holt"), "hw_td": _stat_td("hw"), "sarima_td": _stat_td("sarima"),
    "xgb_daily_td": xgb_daily_td, "xgb_hourly": xgb_hourly, "trk_td": trk_td,
}
