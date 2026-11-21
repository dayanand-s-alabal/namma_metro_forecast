"""Tests for the real-time requirement: forecast window from the run date, tracker-driven fill, flags and no leakage."""
import numpy as np, pandas as pd, pytest
from dataclasses import replace
from test_core import synth, cube_from, CFG
from namma_forecast import dimensions, realtime as rt, forecast as fc
from namma_forecast.evaluate import make_ctx
from namma_forecast.models import trk_td_days


def setup(n_days=42, horizon=12):
    cube, dim, log = cube_from(synth(n_days))
    cal = dimensions.build_dim_calendar(pd.DataFrame({"Date": pd.to_datetime([]), "Event": []}), "2025-08-04", "2026-12-31", set(), set())
    net = np.nansum(cube.entries, axis=(0, 2))
    y = pd.Series(net / 0.8, index=cube.dates)             # tracker = station total / 0.8 -> ratio 0.8
    ctx, arrs = make_ctx(cube, cal, CFG, horizon=horizon, trk={"y": y})
    return cube, cal, y, ctx, arrs


def test_window_follows_run_date_and_rejects_stale_asof():
    last_h, last_t = pd.Timestamp("2025-09-30"), pd.Timestamp("2026-11-15")
    w = rt.resolve_window(CFG, last_h, last_t, "2026-11-20")
    assert w["start"] == pd.Timestamp("2026-11-21") and w["end"] == pd.Timestamp("2026-11-27") and w["h_trk"] == 12
    w2 = rt.resolve_window(CFG, last_h, last_t, "2026-11-21")
    assert w2["start"] == pd.Timestamp("2026-11-22")        # moves with the run date
    with pytest.raises(ValueError):
        rt.resolve_window(CFG, last_h, last_t, "2025-09-29")


def test_trk_td_recovers_tracker_level_and_ratio():
    cube, cal, y, ctx, arrs = setup()
    o = cube.D - 1
    T = np.full(10, float(y.iloc[-7:].mean()))
    F = trk_td_days(ctx, arrs["entries"], o, np.arange(o + 1, o + 11), T)
    assert F.shape == (cube.S, 10, 24)
    assert np.allclose(F.sum(axis=(0, 2)), 0.8 * T * np.array([1.0] * 10), rtol=0.05)   # station sum = ratio x tracker
    assert (F >= 0).all() and (F[:, :, CFG["non_operating_hours"]] == 0).all()


def test_filled_values_are_positive_not_zero_filled():
    cube, cal, y, ctx, arrs = setup()
    o = cube.D - 1
    F = trk_td_days(ctx, arrs["entries"], o, np.arange(o + 1, o + 8), y.iloc[-7:].to_numpy())
    op = [h for h in range(24) if h not in CFG["non_operating_hours"]]
    assert np.isfinite(F).all() and (F[:, :, op].sum(axis=2) > 0).all()              # every operating hour of every station gets a positive filled value


def test_trk_td_does_not_read_data_after_origin():
    cube, cal, y, ctx, arrs = setup()
    o = cube.D - 15
    T = y.iloc[o + 1:o + 8].to_numpy()
    full = trk_td_days(ctx, arrs["entries"], o, np.arange(o + 1, o + 8), T)
    A2 = arrs["entries"].copy(); A2[:, o + 1:, :] = np.nan
    ctx2 = replace(ctx, trk_hist=np.where(np.arange(len(ctx.trk_hist)) <= o, ctx.trk_hist, np.nan))   # tracker after the origin unknown too
    assert np.allclose(full, trk_td_days(ctx2, A2, o, np.arange(o + 1, o + 8), T))


def test_extended_horizon_forecast_returns_only_the_window():
    cube, dim, log = cube_from(synth(42))
    cal = dimensions.build_dim_calendar(pd.DataFrame({"Date": pd.to_datetime([]), "Event": []}), "2025-08-04", "2026-12-31", set(), set())
    f = fc.forecast_station_hourly(cube, cal, CFG, "ma_3wk", "ma_3wk", horizon=10, take=7)
    assert f["pred"]["entries"].shape == (cube.S, 7, 24)
    assert f["dates"][0] == cube.dates[-1] + pd.Timedelta(days=4) and len(f["dates"]) == 7
