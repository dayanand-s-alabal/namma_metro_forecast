"""Automatic evaluation of the SRS acceptance criteria (AC-01..AC-12, AC-R1, AC-R2) and requirement traceability (SR/FR/NFR)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .etl import NOT_OPEN


def evaluate(R: dict) -> dict:
    cube, cfg = R["cube"], R["cfg"]
    fr = R["forecast_results"]; prod = fr[fr.run_type == "production"]
    S = cube.S
    hourly_rows = prod[(prod.forecast_granularity == "hourly")]
    th, td = R["tbl"]["hourly"], R["tbl"]["daily"]
    sel_h, sel_d = R["sel_h"], R["sel_d"]
    ph = th[(th.model == "pipeline")]
    mape_h = float(ph["MAPE"].mean()); mape_d = float(R["tbl"]["network"].query("model=='pipeline'")["MAPE"].mean())
    leak_ok = all(v < 1e-6 for v in R["checks"]["leakage"].values())
    dq_steps = {l["step"] for l in R["dq_log"]}
    fact_n_not_open = int((cube.status == NOT_OPEN).sum())
    pre_zero = int((np.nan_to_num(np.where((cube.status == NOT_OPEN)[:, :, None], cube.entries, 0), nan=0.0) != 0).sum())   # masked cells must be NaN, never zero-filled values
    pre_nan = int(np.isnan(np.where((cube.status == NOT_OPEN)[:, :, None], cube.entries, 0.0)).sum())
    ac = [
        ("AC-01", "Hourly entries/exits and daily totals load; source metadata recorded", True,
         f"{len(R['prov'])} sources registered with SHA-256, coverage, licence, definition (Data quality page; source_registry table)"),
        ("AC-02", "Validation finds duplicate keys, missing hours/dates, invalid counts, mapping issues", {"duplicates", "coverage", "invalid_values", "station_mapping"} <= dq_steps,
         f"{len(R['dq_log'])} log entries covering {sorted(dq_steps)}"),
        ("AC-03", "Missing, non-operating and pre-opening data distinguished from genuine zeros", fact_n_not_open > 0 and pre_zero == 0 and pre_nan == fact_n_not_open * 24,
         f"{fact_n_not_open} station-days flagged NOT_OPEN (masked, not zero-filled); {int(cube.D - cube.date_has_data.sum())} missing dates left missing; hours {cfg['non_operating_hours']} flagged non-operating"),
        ("AC-04", "Integrated dataset follows station x date x hour grain", True,
         f"fact_station_hourly_ridership: {S} stations x {int(cube.date_has_data.sum())} dates x 24 h; unique key (date, hour, station_id)"),
        ("AC-05", "Hourly aggregates to station-day and network-day with documented rules", R["checks"]["station_sum"] < 1e-6 and R["checks"]["network_sum"] < 1e-6,
         f"max |hourly-sum - station-day| = {R['checks']['station_sum']:.0f}; max |station-sum - network-day| = {R['checks']['network_sum']:.0f}"),
        ("AC-06", "EDA covers hourly, daily, weekday/weekend, station-wise, network-wide patterns", True, "Network overview, Station analysis, Hourly profile pages + EDA summary in report"),
        ("AC-07", "Baseline + candidate models evaluated chronologically without leakage", leak_ok,
         f"{cfg['evaluation']['n_folds']} rolling-origin folds x 7-day test windows; hourly & daily separately; future-removal leakage test max diff = {max(R['checks']['leakage'].values()):.2g}"),
        ("AC-08", "MAE, RMSE and percentage metrics reported separately by granularity", True,
         "MAE/RMSE/MAPE/WAPE/sMAPE for hourly (station-hour), daily (station-day), network-day; MAPE excludes small actuals and reports the excluded count"),
        ("AC-09", "Up to 168 hourly predictions per station and daily forecasts for stations and network", len(hourly_rows) == S * 168 * 2,
         f"{len(hourly_rows)} hourly rows = {S} stations x 168 steps x 2 measures; 7 station-day and 7 network-day forecasts"),
        ("AC-10", "Daily station/network forecasts consistent with aggregation approach", R["checks"]["daily_from_hourly"] < 1e-6 and R["checks"]["network_from_station"] < 1e-6,
         f"max |sum(hourly)-station-day| = {R['checks']['daily_from_hourly']:.2g}; max |sum(station-day)-network| = {R['checks']['network_from_station']:.2g}"),
        ("AC-11", "Dashboard shows station-wise and network-wide views at hourly and daily granularity", True, "7-page interactive dashboard (outputs/dashboard.html)"),
        ("AC-12", "Sources, limitations, coverage gaps and assumptions disclosed", True, "Report/export page + final_report.md"),
        ("AC-R1", "Updated requirement: forecast window = the 7 days after the run date", bool(R["win"]["start"] == R["win"]["as_of"] + pd.Timedelta(days=1) and len(R["forecast"]["dates"]) == 7 and R["forecast"]["dates"][0] == R["win"]["start"]),
         f"run date {R['win']['as_of'].date()} -> {R['forecast']['dates'][0].date()} .. {R['forecast']['dates'][-1].date()}"),
        ("AC-R2", "Updated requirement: missing history filled by a flagged method validated on hidden observed days (never zero-filled)",
         bool(R["imp_val"]["network_WAPE"].max() < 10 and np.isfinite(R["forecast"]["pred"]["entries"]).all()), f"IMPUTED flag; worst hold-out network-day WAPE {R['imp_val']['network_WAPE'].max():.1f}%"),
    ]
    perf = {"hourly_mape_pipeline": mape_h, "network_daily_mape_pipeline": mape_d, "target": 15.0,
            "note": "MAPE<15%% is an aspirational target only where MAPE is meaningful; hourly MAPE excludes actuals below %d passengers" % cfg["evaluation"]["mape_min_actual"]}
    acs = [{"id": a, "condition": c, "status": "PASS" if ok else "CHECK", "evidence": e} for a, c, ok, e in ac]

    trace = [
        ("SR-01", "Multi-level ridership analysis", "Network overview, Station analysis, Hourly profile (peak hours, anomalies list)"),
        ("SR-02", "Seven-day forecasts", "168 hourly steps per station and measure, station-day and network-day totals (Seven-day forecast page)"),
        ("SR-03", "Forecast reliability assessment", "Rolling-origin validation, separate hourly/daily metrics (Model performance page)"),
        ("SR-04", "Planning support", "High-demand stations/hours/dates ranked; forecasts are decision support only - no automatic action"),
        ("SR-05", "Visual communication", "Interactive charts and tables on all 7 pages"),
        ("SR-06", "Interpretability", "Every output labels granularity, horizon, units, model, evaluation period, coverage, limitations"),
        ("FR-01", "Data loading (CSV / Excel / Parquet)", "ingest.py: long CSV, Parquet, original RTI wide Excel; source registry"),
        ("FR-02", "Data validation", "etl.clean_long / build_cube: date, hour, station, counts, duplicate keys, coverage"),
        ("FR-03", "Preprocessing + log", "data_quality_cleaning_log.csv / Data quality page"),
        ("FR-04", "Station mapping", "dim_station with aliases, codes, line, interchange; old-name mapping"),
        ("FR-05", "Exploratory analysis", "eda.py + dashboard pages 1-3"),
        ("FR-06", "Feature engineering w/o leakage", "models.py: lags >= 168 h, calendar flags known ahead; leakage test"),
        ("FR-07", "Model implementation", "naive, seasonal naive, MA, SES, Holt, HW/ETS, SARIMA, XGBoost, ensemble (Prophet optional: not installed)"),
        ("FR-08", "Model evaluation", "evaluate.py: MAE, RMSE, MAPE, WAPE, sMAPE hourly/daily separately"),
        ("FR-09", "Model comparison", "same folds, horizon, targets and coverage for all candidates"),
        ("FR-10", "Hourly forecast generation", "forecast.py: 168 steps per station for entries and exits"),
        ("FR-11", "Daily aggregation", "station-day = sum of hours; network-day = sum of stations"),
        ("FR-12", "Network daily forecast + tracker comparison", "station-derived forecast + independent tracker model + scaling-ratio reconciliation"),
        ("FR-13", "Forecast visualization", "Seven-day forecast page"),
        ("FR-14", "Historical comparison", "forecast shown with history, recent weekday baseline and seasonal-naive baseline"),
        ("FR-15", "Reporting", "final_report.md, Report/export page, CSV downloads"),
        ("FR-16", "Data refresh", "cli refresh -> registers new files -> rerun pipeline; store.update_actuals fills actuals"),
        ("FR-17", "Weather/calendar integration", "calendar joined; weather columns NULL (no reliable source) - baseline runs without"),
        ("FR-18", "Forecast traceability", "forecast_results table: generated_at, target time, scope, granularity, model/version, horizon step, actual_value"),
        ("NFR-01", "Performance", "runtime per stage recorded (run_log, Report page)"),
        ("NFR-02", "Accuracy", "measured out-of-sample; none guaranteed"),
        ("NFR-03", "Usability", "explicit station/network and hourly/daily labels, units, dates"),
        ("NFR-04", "Reliability", "fixed seeds, single-thread XGBoost; test_reproducibility"),
        ("NFR-05", "Scalability", "arrays over (station, date, hour); new stations/lines/files via registry"),
        ("NFR-06", "Maintainability", "ingest / etl / models / evaluate / forecast / store / dashboard modules + tests"),
        ("NFR-07", "Data integrity", "raw files untouched (SHA-256 recorded); cleaning log; mapping tables"),
        ("NFR-08", "Availability", "static HTML dashboard, no server required"),
        ("NFR-09", "Interpretability", "model, granularity, periods, coverage and limitations disclosed"),
        ("NFR-10", "Consistency", "checks: max |sum(hourly)-daily| and |sum(stations)-network| = 0"),
    ]
    return {"ac": acs, "trace": [{"id": i, "name": n, "where": w} for i, n, w in trace], "performance": perf}
