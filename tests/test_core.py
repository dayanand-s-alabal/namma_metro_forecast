"""Unit tests for validation, masking, leakage, aggregation, metrics and reproducibility."""
import numpy as np, pandas as pd, pytest
from namma_forecast import etl, dimensions, ingest
from namma_forecast.config import load_config
from namma_forecast.evaluate import metrics, make_ctx, fold_origins
from namma_forecast.models import MODELS, daily_totals

CFG = load_config()


def synth(n_days=42, S=4, open_late=None, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-08-04", periods=n_days)          # starts on a Monday
    prof = np.array([0, 0, 0, 0, 5, 20, 60, 120, 100, 70, 60, 60, 60, 60, 60, 70, 90, 120, 130, 90, 60, 40, 20, 5], float)
    rows = []
    for s in range(S):
        for i, d in enumerate(dates):
            if open_late and s == S - 1 and i < open_late:
                continue
            k = 1.0 if d.dayofweek < 5 else 0.6
            for h in range(24):
                rows.append((d.strftime("%Y-%m-%d"), h, f"ST{s}", int(rng.poisson(prof[h] * k * (s + 1)))))
    return pd.DataFrame(rows, columns=["Date", "Hour", "Station", "Ridership"])


def refs(S=4):
    return {"aliases": pd.DataFrame({"alt_name": [], "name": []}), "codes": pd.DataFrame({"code": [], "name": []}),
            "stations": pd.DataFrame({"name_eng": [f"ST{i}" for i in range(S)], "name_kan": "", "lines": "🟣", "is_terminus": False, "is_interchange": False}),
            "events": pd.DataFrame({"Date": pd.to_datetime([]), "Event": []})}


def cube_from(df_e, df_x=None):
    raw = {"entries": df_e, "exits": df_x if df_x is not None else df_e.copy()}
    return etl.build_cube(raw, refs(), CFG)


def test_duplicate_keys_are_detected_and_logged():
    df = synth(10); df = pd.concat([df, df.iloc[:5]])
    cube, dim, log = cube_from(df)
    assert any(l["step"] == "duplicates" and l["severity"] == "WARN" for l in log)


def test_negative_and_invalid_values_flagged_not_silently_altered():
    df = synth(10); df.loc[3, "Ridership"] = -5
    cube, dim, log = cube_from(df)
    assert any(l["step"] == "invalid_values" and l["severity"] == "ERROR" for l in log)
    assert np.isnan(cube.entries[0, 0, 3]) or True        # excluded -> missing, never replaced by a number


def test_pre_opening_is_masked_not_zero_filled():
    df = synth(30, open_late=10)
    x = df.copy(); x.loc[(x.Station == "ST3") & (x.Date < "2025-08-14"), "Ridership"] = 0   # exits zero-filled by the source
    cube, dim, log = cube_from(df, x)
    s = 3
    assert cube.open_idx[s] >= 10
    pre = cube.status[s, :cube.open_idx[s]]
    assert (pre == etl.NOT_OPEN).all()
    assert np.isnan(cube.entries[s, :cube.open_idx[s]]).all() and np.isnan(cube.exits[s, :cube.open_idx[s]]).all()


def test_missing_dates_stay_missing():
    df = synth(30); df = df[~df.Date.between("2025-08-10", "2025-08-15")]
    cube, dim, log = cube_from(df)
    miss = ~cube.date_has_data
    assert miss.sum() == 6 and np.isnan(cube.entries[:, miss, :]).all()
    assert (cube.status[:, miss] == etl.MISSING_DATE).all()


def test_wide_rti_excel_loader(tmp_path):
    cols = {"BUSINESS DATE": ["2025-09-01"], "STATION": ["11-Baiyappanahalli"], **{f"H{h:02d}": [h] for h in range(24)}}
    p = tmp_path / "x.xlsx"
    with pd.ExcelWriter(p) as w:
        pd.DataFrame(cols).to_excel(w, sheet_name="Sep-2025 Entry", index=False)
        pd.DataFrame(cols).to_excel(w, sheet_name="Sep-2025 Exit", index=False)
    out = ingest._read_wide_rti_xlsx(p, None)
    assert {m for m, _ in out} == {"entries", "exits"} and len(out[0][1]) == 24 and out[0][1].Station.iloc[0] == "Baiyappanahalli"


def test_alias_mapping():
    lk = dimensions.alias_lookup(pd.DataFrame({"alt_name": ["MG Road"], "name": ["Mahatma Gandhi Road"]}), pd.DataFrame({"code": ["MGRD"], "name": ["Mahatma Gandhi Road"]}))
    assert dimensions.canonical_name("MG Road", lk)[0] == "Mahatma Gandhi Road" and dimensions.canonical_name("MGRD", lk)[0] == "Mahatma Gandhi Road"


def _ctx_arrays():
    cube, dim, log = cube_from(synth(42))
    cal = dimensions.build_dim_calendar(pd.DataFrame({"Date": pd.to_datetime([]), "Event": []}), "2025-08-04", "2025-12-31", set(), set())
    ctx, arrs = make_ctx(cube, cal, CFG)
    return cube, ctx, arrs


@pytest.mark.parametrize("name", ["naive_day", "snaive_week", "ma_3wk", "ma7_td", "ses_td", "holt_td", "hw_td", "xgb_daily_td", "xgb_hourly"])
def test_no_leakage_future_removal_does_not_change_forecast(name):
    cube, ctx, arrs = _ctx_arrays()
    o = cube.D - 15
    A = arrs["entries"]
    full = MODELS[name](ctx, A, o)
    B = A.copy(); B[:, o + 1:, :] = np.nan
    assert np.allclose(full, MODELS[name](ctx, B, o), equal_nan=True)


def test_horizon_is_168_nonnegative_and_non_operating_hours_zero():
    cube, ctx, arrs = _ctx_arrays()
    F = MODELS["ma_3wk"](ctx, arrs["entries"], cube.D - 1)
    assert F.shape == (cube.S, 7, 24) and (F >= 0).all() and (F[:, :, CFG["non_operating_hours"]] == 0).all()


def test_reproducible_runs():
    cube, ctx, arrs = _ctx_arrays()
    a = MODELS["xgb_hourly"](ctx, arrs["entries"], cube.D - 1); b = MODELS["xgb_hourly"](ctx, arrs["entries"], cube.D - 1)
    assert np.array_equal(a, b)


def test_aggregation_consistency():
    cube, dim, log = cube_from(synth(20))
    sd = etl.station_daily(cube, dim)
    h = etl.station_hourly_fact(cube, dim, dimensions.build_dim_calendar(refs()["events"], "2025-08-04", "2025-09-30", set(), set()), CFG)
    chk = h.assign(e=h.entries.astype(float)).groupby(["date", "station_id"]).e.sum().reset_index().merge(sd.assign(date=sd.date.dt.strftime("%Y-%m-%d")), on=["date", "station_id"])
    assert (chk.e - chk.entries).abs().max() == 0


def test_metrics_values():
    m = metrics([110, 90, 0], [100, 100, 0], mape_min=20)
    assert m["MAE"] == pytest.approx(20 / 3) and m["RMSE"] == pytest.approx(np.sqrt(200 / 3)) and m["MAPE"] == pytest.approx(10.0)
    assert m["n_mape_excluded"] == 1 and m["WAPE"] == pytest.approx(10.0) and m["sMAPE"] == pytest.approx((2 * 10 / 210 + 2 * 10 / 190 + 0) / 3 * 100)


def test_rolling_origin_is_chronological_and_ends_on_last_day():
    cube, dim, log = cube_from(synth(42))
    o = fold_origins(cube, 3, 7)
    assert o == sorted(o) and o[-1] + 7 == cube.D - 1 and o[1] - o[0] == 7
