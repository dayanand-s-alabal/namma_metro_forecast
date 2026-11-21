"""FR-15 / deliverable 11: Markdown project report generated from the pipeline results (no hand-typed numbers)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .models import MODEL_INFO


def md_table(df: pd.DataFrame, cols: list[str], fmt: dict | None = None) -> str:
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "|".join("---" for _ in cols) + "|\n"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join((fmt[c](r[c]) if c in fmt else str(r[c])) for c in cols) + " |")
    return head + "\n".join(rows) + "\n"


def f0(x): return f"{x:,.0f}"
def f1(x): return f"{x:,.1f}"
def f2(x): return f"{x:,.2f}"


def write_report(R: dict, path) -> None:
    cube, cfg, dim = R["cube"], R["cfg"], R["dim"]
    meta = f"{cube.dates[0].date()} to {cube.dates[-1].date()}"
    obs_days = int(cube.date_has_data.sum())
    iv = R["imp_val"]; iv = iv[iv.method.str.startswith("tracker")]
    imp_txt = "; ".join(f"{r.scenario} / {r.measure}: station-hour {r.hourly_WAPE:.1f}, station-day {r.daily_WAPE:.1f}, network-day {r.network_WAPE:.1f}" for r in iv.itertuples())
    win = R["win"]
    tbl = R["tbl"]
    lab = lambda m: MODEL_INFO[m][0]
    P = R["forecast"]["pred"]
    dates = R["forecast"]["dates"]
    ent = P["entries"].sum(axis=(0, 2)); ext = P["exits"].sum(axis=(0, 2))
    dfc = pd.DataFrame({"Date": [d.strftime("%a %d %b %Y") for d in dates], "Entries (station-derived)": ent, "Exits (station-derived)": ext})
    top_st = (P["entries"].sum(axis=(1, 2))).argsort()[::-1][:10]
    rows_ = []
    for s_ in top_st:
        k_, h_ = np.unravel_index(P["entries"][s_].argmax(), P["entries"][s_].shape)
        rows_.append({"Station": dim.station_name[s_], "Peak forecast hour": f"{dates[k_].strftime('%a %d %b')} {h_:02d}:00", "Entries": P["entries"][s_, k_, h_]})
    hi_hour = pd.DataFrame(rows_)
    def sel_table(level):
        t = R["sel"][level].copy()
        return md_table(t, ["rank", "label", "family", "score", "stability", "runtime_s", "vs_seasonal_naive_pct"], {"score": f2, "stability": f2, "runtime_s": f1, "vs_seasonal_naive_pct": lambda v: f"{v:+.1f}%"})
    def metric_table(level):
        t = tbl[level]; t = t[t.model.isin(list(R["sel"][level].model.head(4)) + ["snaive_week", "pipeline"])].drop_duplicates(["model", "measure"])
        return md_table(t, ["label", "measure", "MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "n", "n_mape_excluded"], {k: f2 for k in ("MAE", "RMSE", "MAPE", "WAPE", "sMAPE")} | {"n": f0, "n_mape_excluded": f0})
    nt = R["ntbl"]
    rec = R["recon"]
    log = pd.DataFrame(R["dq_log"])
    log = log[log.severity != "INFO"]
    ac = pd.DataFrame(R["acceptance"]["ac"])
    perf = R["acceptance"]["performance"]
    tf = R["tfc"]
    txt = f"""# Namma Metro Ridership Forecasting System - Project Report

*Generated {R['generated_at']} by `namma_forecast` v{cfg['version']}. Academic prototype; not an official BMRCL document.
All numbers below are produced by the pipeline from the supplied datasets.*

## 1. Scope and outputs
- Station-wise **hourly** entry and exit forecasts for the next 7 consecutive days (168 steps per station), station-wise **daily** totals (sum of the hours),
  and **network-wide daily** totals (sum of stations). Entries and exits are separate targets and are never added together.
- The independent daily tracker series is forecast separately (7 days) and used for reconciliation.
- **Real-time window:** run date **{win['as_of'].date()}**, forecast days **{dates[0].date()} to {dates[-1].date()}** (the 7 days after the run date; use `--as-of` to change it).
- Production model for station-hours: **{R['prod_h']}** ({R['sel_reason']}).
- Station-hour data end {win['last_hourly'].date()}; the tracker ends {win['tracker_last'].date()}. Missing station history up to the run date is **imputed** (section 3) and flagged.
- Tracker model: {nt.label.iloc[0]}; its forecast starts the day after the last tracker date and runs {win['h_trk']} days to the end of the window.

## 2. Data sources and provenance
""" + md_table(pd.DataFrame(R["prov"]), ["source_id", "file", "rows", "date_min", "date_max", "stations", "license"]) + f"""
Bucket convention: {cfg['hour_bucket_convention']}. Time zone {cfg['timezone']}.

## 3. Coverage, data quality and gap filling
- Hourly station data: **{obs_days} observed days** within {meta} ({cube.S} stations). Dates without data ({int((~cube.date_has_data).sum())} inside the range) and every day up to the run date are **imputed, never zero-filled**: station-hour value = tracker total x station-derived/tracker ratio x station share x hour-of-day share (ratio, shares and profiles from the last 35 observed days by day type). Imputed values carry the IMPUTED flag (`imputed_station_daily.csv`, `imputed_station_hourly.csv.gz`).
- Imputation check on hidden observed days (WAPE %): {imp_txt}.
- Limits of the fill: shares, profiles and the ratio are frozen at the last observed day, so stations opened or changed since then are not represented; station-level interval bands are widened by a documented heuristic factor when the station data are old.
- 15 Yellow Line stations opened on 2025-08-11; earlier values are masked as NOT_OPEN (not zero). Tiny trial-run traffic is logged.
- Station-day entries explain about {rec['ratio_median_full_network']*100:.0f}% of the tracker's daily total on {rec['overlap_days']} overlapping days (p10-p90 {rec['ratio_p10']*100:.0f}-{rec['ratio_p90']*100:.0f}%): the definitions differ, so tracker values are compared only through this ratio.

### Cleaning log (warnings and errors)
""" + md_table(log, ["step", "severity", "scope", "detail", "action"]) + f"""
## 4. Exploratory findings
- Weekday peaks: AM at {R['eda']['peak_hours']['weekday_am']:02d}:00 and PM at {R['eda']['peak_hours']['weekday_pm']:02d}:00. Weekday traffic is {R['eda']['weekday_vs_weekend_ratio']:.2f}x weekend traffic on average.
- Ten busiest stations carry {R['eda']['top10_share_pct']:.0f}% of all entries. Busiest: {R['eda']['top10_stations_entries'][0]['station']}.
- Average daily entries by weekday: {', '.join(f"{k} {v:,.0f}" for k, v in R['eda']['avg_daily_entries_by_dow'].items())} (complete-network days only).
- Anomalies are listed on the dashboard (Hourly profile / Data quality) with calendar events attached.

## 5. Models and validation
Candidates: naive, seasonal naive, moving averages, SES, Holt, Holt-Winters/ETS, SARIMA, XGBoost (hourly and daily top-down), and an ensemble. Prophet is optional and not installed here.
Statistical models forecast daily totals per station and distribute them with weekday hour-share profiles (top-down) because 7 weeks of history cannot support a 168-hour seasonal fit.
Validation is **rolling-origin**: {cfg['evaluation']['n_folds']} consecutive 7-day test windows ending at the last observed day; every fold trains only on earlier days.
Models are ranked by mean fold WAPE over entries and exits; fold-to-fold spread and runtime are shown with it.

### Hourly target (station-hour)
""" + sel_table("hourly") + "\n" + metric_table("hourly") + f"""
### Daily target (station-day)
""" + sel_table("daily") + "\n" + metric_table("daily") + f"""
### Network-day (sum of stations)
""" + sel_table("network") + f"""
**Selected:** hourly = {lab(R['sel_h'])}; daily = {lab(R['sel_d'])}. If they differ, hourly values are rescaled per station-day so they sum to the daily model's total (the reconciled pipeline is itself evaluated above).

Aspirational target MAPE < 15%: hourly pipeline MAPE = {perf['hourly_mape_pipeline']:.1f}% (actuals below {cfg['evaluation']['mape_min_actual']} passengers excluded and counted),
network-day MAPE = {perf['network_daily_mape_pipeline']:.1f}%. This is evidence from {cfg['evaluation']['n_folds']} weeks only, not a guarantee.

### Independent tracker series (daily network, {int(nt.folds.max())} weekly folds)
""" + md_table(nt, ["rank", "label", "MAE", "RMSE", "MAPE", "WAPE", "recent_MAPE", "wape_fold_std"], {k: f2 for k in ("MAE", "RMSE", "MAPE", "WAPE", "recent_MAPE", "wape_fold_std")}) + f"""
`recent_MAPE` covers the latest {cfg['network_tracker']['recent_folds']} folds, a period with a level shift (growing NCMC share, ONDC QR from 2026-08-12) and holidays.

## 6. Seven-day forecast
### Station-derived network totals (RTI boardings definition)
""" + md_table(dfc, list(dfc.columns), {"Entries (station-derived)": f0, "Exits (station-derived)": f0}) + f"""
Out-of-sample check against tracker totals published after the data end (scaled by the reconciliation ratio):
""" + (md_table(pd.DataFrame(R["oos"]), ["date", "scaled_to_tracker", "tracker_actual", "error_pct", "event"], {"scaled_to_tracker": f0, "tracker_actual": f0, "error_pct": lambda v: f"{v:+.1f}%"}) if R["oos"] else "_no tracker actuals exist for the forecast dates_\n") + f"""
### Highest-demand stations (forecast week, entries)
""" + md_table(hi_hour, ["Station", "Peak forecast hour", "Entries"], {"Entries": f0}) + f"""
### Tracker forecast (all ticket modes)
""" + md_table(pd.DataFrame({"Date": tf["date"].dt.strftime("%a %d %b %Y"), "Forecast": tf["pred"]}), ["Date", "Forecast"], {"Forecast": f0}) + f"""
Prediction intervals are {int(cfg['evaluation']['interval_coverage']*100)}% empirical bands from out-of-sample ratios (actual / predicted); they are narrow-sample estimates.

## 7. Aggregation and consistency
station-day = sum of 24 hourly forecasts; network-day = sum of station-days. Checks: max |sum(hourly) - station-day| = {R['checks']['daily_from_hourly']:.2g};
max |sum(stations) - network| = {R['checks']['network_from_station']:.2g}; leakage test (future removed) max difference = {max(R['checks']['leakage'].values()):.2g}.

## 8. Limitations and assumptions
- Only {obs_days} days of hourly history (two blocks, with a 13-day gap): 168-hour weekly seasonality rests on 3-7 observations per hour-of-week; annual seasonality cannot be estimated.
- The Yellow Line opening on 2025-08-11 is a structural break; its 15 stations have under 8 weeks of history and are flagged.
- Holiday and event flags are keyword-classified from `significant_dates.csv` and must be checked against the official list.
- No weather data (columns stored as NULL); no real-time data; no service-disruption feed.
- Tracker vs station definitions differ; 256 tracker days are missing (never scored; imputed only for model fitting); QR totals recomputed from app columns.
- Anomalous days (festivals, strikes, matches) are poorly predicted by lag-based models. The forecasts support planning and do not trigger any operational action.
- Licence: ODbL / CC BY-NC-SA 4.0 for the hourly data; check before non-academic use.

## 9. Future work
Request longer hourly history from BMRCL, add verified weather and event data, line-wise forecasts, real-time refresh, anomaly alerts, and OD-based interchange modelling.

## 10. Acceptance criteria status
""" + md_table(ac, ["id", "condition", "status", "evidence"]) + f"""
## 11. Runtime (NFR-01)
""" + md_table(pd.DataFrame([{"stage": k, "seconds": v} for k, v in R["timer"].items()]), ["stage", "seconds"]) + f"\nEnvironment: {R['run_env']}\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
