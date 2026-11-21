"""End-to-end orchestration: ingest -> validate/clean -> integrate -> EDA -> backtest -> select -> forecast -> store -> report -> dashboard.
Stages are separate modules (NFR-06); each stage's runtime is recorded (NFR-01)."""
from __future__ import annotations
import json, platform, time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from . import __version__, ingest, etl, dimensions, evaluate as ev, forecast as fc, network_daily as nd, eda, store, realtime as rt
from .config import load_config, load_sources, resolve
from .models import MODEL_INFO

CAL_START = "2024-10-26"


class Timer:
    def __init__(self): self.t = {}
    @contextmanager
    def stage(self, name, log=print):
        t0 = time.time(); log(f"[{name}] ...")
        yield
        self.t[name] = round(time.time() - t0, 2); log(f"[{name}] done in {self.t[name]}s")


def _round_tbl(df: pd.DataFrame, nd_: int = 2) -> list[dict]:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(nd_)
    return json.loads(out.to_json(orient="records"))


def run(config_path=None, log=print, skip_sarima: bool = False, as_of=None) -> dict:
    cfg = load_config(config_path)
    T = Timer()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out_dir = resolve(cfg, cfg["paths"]["outputs"]); out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(cfg["seed"])

    # ------------------------------------------------------------------ 1. ingest
    with T.stage("ingest", log):
        sources = load_sources(cfg)
        raw, prov = ingest.load_all_hourly(cfg, sources)
        tracker_raw, prov_d = ingest.load_tracker(cfg, sources)
        refs = ingest.load_reference(cfg, sources)
        prov += prov_d
        for k in ("stations_dataset", "station_codes", "station_names", "significant_dates"):
            p = resolve(cfg, f"{cfg['paths']['raw_dir']}/{sources['reference'][k]}")
            prov.append({"source_id": "REF_" + k.upper(), "file": sources["reference"][k], "format": "csv", "sha256": ingest.sha256(p), "rows": int(len(
                {"stations_dataset": refs["stations"], "station_codes": refs["codes"], "station_names": refs["aliases"], "significant_dates": refs["events"]}[k])),
                "measures": "reference", "date_min": "", "date_max": "", "dates": 0, "stations": 0, "url": "", "downloaded": "2026-11-17", "license": "see README_collection.md",
                "origin": "collected reference table", "definition": k, "limitations": "calendar classification is keyword-based; verify against official holiday list" if k == "significant_dates" else ""})

    # ------------------------------------------------------------------ 2. validate / integrate
    with T.stage("validate_integrate", log):
        cube, dim, dq_log = etl.build_cube(raw, refs, cfg)
        fact_net = etl.prepare_tracker(tracker_raw, dq_log)
        # real-time: the run date decides what is "known". Tracker rows dated after the run date are not yet available.
        as_of_ts = pd.Timestamp(as_of or cfg["realtime"].get("as_of") or datetime.now().date()).normalize()
        n_future = int((fact_net["date"] > as_of_ts).sum())
        if n_future:
            fact_net = fact_net[fact_net["date"] <= as_of_ts].reset_index(drop=True)
            dq_log.append({"step": "as_of", "severity": "INFO", "scope": "daily_tracker", "detail": f"{n_future} tracker rows dated after the run date {as_of_ts.date()} ignored", "n_records": n_future, "action": "treated as not yet available"})
        win = rt.resolve_window(cfg, cube.dates[-1], fact_net["date"].max(), as_of_ts)
        CAL_END = str((win["end"] + pd.Timedelta(days=60)).date())
        hourly_dates = set(cube.dates[cube.date_has_data].strftime("%Y-%m-%d"))
        cal = dimensions.build_dim_calendar(refs["events"], CAL_START, CAL_END, hourly_dates, set(fact_net["date"]))
        fact_h = etl.station_hourly_fact(cube, dim, cal, cfg)
        sd = etl.station_daily(cube, dim)
        sd_net = etl.network_daily_from_stations(sd)
        unmapped = [e for e in refs["events"]["Event"] if dimensions.classify_event(e) == "observance"]
        dq_log.append({"step": "calendar", "severity": "INFO", "scope": "dim_calendar", "detail": f"{len(refs['events'])} dated events classified into holiday / festival / event / disruption / observance; {len(unmapped)} observances (eclipses, solstices...) carry no demand flag",
                       "n_records": len(refs["events"]), "action": "keyword classification; verify against the official Karnataka holiday list"})
        dq_log.append({"step": "weather", "severity": "INFO", "scope": "weather", "detail": "no verified date/hour-aligned weather source available in this environment (IMD/Open-Meteo unreachable)",
                       "n_records": 0, "action": "rainfall_mm, temperature_c, humidity stored as NULL; baseline and all models run without them (FR-17)"})
        # aggregation integrity (AC-05 / NFR-10)
        chk_station = float(np.nanmax(np.abs(
            fact_h.assign(e=fact_h["entries"].astype(float)).groupby(["date", "station_id"])["e"].sum(min_count=1).reset_index()
            .merge(sd.assign(date=sd["date"].dt.strftime("%Y-%m-%d")), on=["date", "station_id"], suffixes=("_h", "_d"))
            .eval("e - entries").to_numpy())))
        chk_net = float(abs(sd.groupby("date")["entries"].sum().to_numpy() - sd_net["entries"].to_numpy()).max())

    # ------------------------------------------------------------------ 3. EDA
    with T.stage("eda", log):
        eda_s = eda.eda_summary(cube, dim, sd, cal)
        anomalies = eda.anomalies(cube, dim, cal, fact_net)
        recon = eda.reconcile_with_tracker(sd_net, fact_net)
        lookup = dimensions.alias_lookup(refs["aliases"], refs["codes"])
        od = eda.od_summary(cfg, dim, lookup, resolve(cfg, f"{cfg['paths']['raw_dir']}/{sources['od']['path']}") if sources.get("od") else None,
                            resolve(cfg, cfg["paths"]["od_cache"]))

    # ------------------------------------------------------------------ 4a. tracker (independent network series), run first because the station
    #                                                                       candidate "trk_td" and the production path need its forecasts
    with T.stage("network_tracker", log):
        sdf = nd.prepare_series(fact_net, cal, future=max(win["h_trk"], 7))
        nbt = nd.backtest(sdf, cfg, log=log)
        ntbl = nd.evaluate(nbt, cfg)
        sel_t = ntbl.model.iloc[0]
        tband = nd.interval_quantiles(nbt, sel_t, cfg["evaluation"]["interval_coverage"])
        rec_long = nd.backtest_model(sdf, cfg, sel_t, max(14, win["h_trk"]))
        step_q = nd.step_bands(rec_long, cfg["evaluation"]["interval_coverage"], win["h_trk"])
        tracker_T, tracker_src = rt.tracker_path(sdf, sel_t, cfg, win)
        fut = tracker_T.index > win["tracker_last"]
        tfc_all = pd.DataFrame({"date": tracker_T.index[fut], "pred": tracker_T[fut].to_numpy(), "horizon_step": np.arange(1, win["h_trk"] + 1)})
        tfc_all["lo"] = tfc_all["pred"] * np.array([b[0] for b in step_q]); tfc_all["hi"] = tfc_all["pred"] * np.array([b[1] for b in step_q])
        tfc = tfc_all[tfc_all["date"] >= win["start"]].reset_index(drop=True)          # the 7 days of the forecast window
        y_actual = fact_net.set_index("date")["official_daily_total"].astype(float)
        H7 = cfg["horizon_days"]
        trk_fc = {o: nd._forecast(sel_t, sdf, sdf.index.get_loc(cube.dates[o]), H7, cfg) for o in ev.fold_origins(cube, cfg["evaluation"]["n_folds"], H7)}
        trk_pack = {"y": y_actual, "fc": trk_fc}
        log(f"  run date {win['as_of'].date()} -> forecast window {win['start'].date()}..{win['end'].date()}; tracker ends {win['tracker_last'].date()} ({win['stale_tracker_days']} d before the window), "
            f"station-hour data ends {win['last_hourly'].date()} ({win['stale_hourly_days']} d before); tracker model {sel_t}, horizon {win['h_trk']} d")

    # ------------------------------------------------------------------ 4. station backtests
    models = [m for m in MODEL_INFO if m not in ("ensemble", "pipeline")]
    if skip_sarima:
        models.remove("sarima_td")
    with T.stage("backtest_station", log):
        bt = ev.run_backtests(cube, cal, cfg, models=models, log=log, trk=trk_pack)
        ctx = bt["ctx"]
        # leakage guard: forecasts must not change when everything after the origin is removed
        leak = {}
        arrs = {m: np.concatenate([cube.arr(m), np.full((cube.S, ctx.horizon, 24), np.nan)], axis=1) for m in ("entries", "exits")}
        o0 = bt["origins"][0]
        for name in ("snaive_week", "ma_3wk", "xgb_hourly", "xgb_daily_td", "hw_td", "trk_td"):
            if name not in models:
                continue
            A = arrs["entries"].copy(); A[:, o0 + 1:, :] = np.nan
            from .models import MODELS
            leak[name] = float(np.nanmax(np.abs(MODELS[name](ctx, A, o0) - bt["pred"][(name, "entries", 0)])))
    with T.stage("evaluate_select", log):
        tbl_h0, tbl_d0, tbl_n0 = (ev.pool(bt, cube, cfg, lv) for lv in ("hourly", "daily", "network"))
        sel_h_tbl, sel_d_tbl, sel_n_tbl = ev.select(tbl_h0), ev.select(tbl_d0), ev.select(tbl_n0)
        sel_h, sel_d = sel_h_tbl.model.iloc[0], sel_d_tbl.model.iloc[0]
        # parsimony rule: when the hourly winner is within 3% (relative) of the daily winner on the daily target, use it for both,
        # so the published daily totals are exactly the sum of that model's hourly forecasts (no reconciliation needed)
        sc = sel_d_tbl.set_index("model")["score"]
        if sel_h != sel_d and sc[sel_h] <= sc[sel_d] * 1.03:
            log(f"  daily winner {sel_d} is within 3% of {sel_h} -> using {sel_h} for both targets")
            sel_d = sel_h
        # production model: the candidate that won on the observed folds, unless the station-hour data is too old to say anything about the run
        # date; then only the tracker-driven top-down can carry post-gap information (documented override)
        use_trk = win["stale_hourly_days"] > cfg["realtime"]["stale_hourly_days"]
        eff_h, eff_d = ("trk_td", "trk_td") if use_trk else (sel_h, sel_d)
        sel_reason = (f"station-hour data ends {win['last_hourly'].date()}, {win['stale_hourly_days']} days before the first forecast day (more than {cfg['realtime']['stale_hourly_days']}): "
                      f"production uses the tracker-driven top-down; the best candidate on the observed folds was {sel_h}") if use_trk else "best candidate on the observed folds"
        fc.add_pipeline_backtest(bt, eff_h, eff_d, ctx)
        tbl_h, tbl_d, tbl_n = (ev.pool(bt, cube, cfg, lv) for lv in ("hourly", "daily", "network"))
        per_fold = ev.evaluate(bt, cube, cfg)
        ip = fc.interval_params(bt, cube, ctx, cfg)
        log(f"  best on observed folds: hourly={sel_h} daily={sel_d};  production: {eff_h}/{eff_d}")
        if use_trk:
            ip = fc.inflate_intervals(ip, cfg["realtime"]["band_inflation_stale"])

    # ------------------------------------------------------------------ 6. final forecasts
    with T.stage("forecast", log):
        imp = rt.fill_and_forecast(cube, cal, cfg, tracker_T, tracker_src, win, y_actual)      # filled history (gaps, run-date bridge) + tracker-driven forecast
        imp_val = rt.validate_imputation(cube, cal, cfg, y_actual)
        if use_trk or "trk_td" in (eff_h, eff_d):          # the tracker-driven model needs the tracker path, which fill_and_forecast already built
            f = fc.production_from_fill(imp, cube, win)
        else:
            f = fc.forecast_station_hourly(cube, cal, cfg, eff_h, eff_d, horizon=imp["Hp"], take=cfg["horizon_days"])
        net_bands = step_q[-cfg["horizon_days"]:]
        res_prod = fc.results_frame(cube, dim, f, ip, cfg, eff_h, eff_d, tfc, sel_t, tband, generated_at, net_bands=net_bands)
        res_bt = fc.backtest_results_frame(bt, cube, dim, ctx, cfg, ip, generated_at, nbt["records"], sel_t)
        forecast_results = pd.concat([res_prod, res_bt], ignore_index=True)
        # NFR-10 consistency checks
        P = f["pred"]["entries"]
        fr = res_prod
        d_from_h = fr[(fr.forecast_granularity == "hourly") & (fr.target_measure == "entries")].groupby(["station_id", "target_date"])["predicted_value"].sum()
        d_rows = fr[(fr.forecast_granularity == "daily") & (fr.scope == "station") & (fr.target_measure == "entries")].set_index(["station_id", "target_date"])["predicted_value"]
        c1 = float((d_from_h - d_rows.reindex(d_from_h.index)).abs().max())
        n_rows = fr[(fr.scope == "network_station_derived") & (fr.target_measure == "entries")].set_index("target_date")["predicted_value"]
        c2 = float((d_rows.groupby("target_date").sum() - n_rows).abs().max())
        # out-of-sample sanity check against tracker values that appeared after the training cut-off
        tr_i = fact_net.set_index("date")["official_daily_total"]
        ratio = recon["ratio_median_full_network"]
        oos = []
        for d, v in zip(f["dates"], P.sum(axis=(0, 2))):
            if d in tr_i.index:
                oos.append({"date": str(d.date()), "forecast_station_entries": float(v), "scaled_to_tracker": float(v / ratio), "tracker_actual": float(tr_i[d]),
                            "error_pct": float((v / ratio / tr_i[d] - 1) * 100), "event": (cal.set_index("date").loc[d, "all_events"] if isinstance(cal.set_index("date").loc[d, "all_events"], str) else "")})

    # ------------------------------------------------------------------ 7. store
    with T.stage("store", log):
        imp_daily_tbl, imp_hourly_tbl = fc.imputed_tables(imp, cube, dim, cal)
        tracker_path_tbl = pd.DataFrame({"date": tracker_T.index.strftime("%Y-%m-%d"), "tracker_total": tracker_T.to_numpy().round(0), "source": tracker_src.to_numpy()})
        dq_df = pd.DataFrame(dq_log)
        prov_df = pd.DataFrame(prov)
        metrics_all = pd.concat([tbl_h.assign(level="hourly_station"), tbl_d.assign(level="daily_station"), tbl_n.assign(level="daily_network_station_derived")])
        metrics_all["fold_wape"] = metrics_all["fold_wape"].astype(str)
        trk_metrics = ntbl.assign(level="daily_network_tracker")
        run_log = pd.DataFrame([{"generated_at": generated_at, "version": __version__, "stage": k, "seconds": v} for k, v in T.t.items()])
        fact_net_db = fact_net.copy()
        tables = {
            "dim_station": dim, "dim_calendar": cal.assign(date=cal["date"].dt.strftime("%Y-%m-%d")),
            "fact_station_hourly_ridership": fact_h,
            "fact_station_daily": sd.assign(date=sd["date"].dt.strftime("%Y-%m-%d")),
            "fact_network_daily_ridership": fact_net_db.assign(date=fact_net_db["date"].dt.strftime("%Y-%m-%d")),
            "fact_network_daily_station_derived": sd_net.assign(date=sd_net["date"].dt.strftime("%Y-%m-%d")),
            "fact_station_daily_imputed": imp_daily_tbl, "imputation_validation": imp_val, "tracker_path": tracker_path_tbl,
            "forecast_results": forecast_results, "model_metrics": pd.concat([metrics_all, trk_metrics], ignore_index=True).astype({"fold_wape": str}, errors="ignore"),
            "data_quality_log": dq_df, "source_registry": prov_df, "run_log": run_log,
        }
        store.write_db(resolve(cfg, cfg["paths"]["db"]), tables)
        for name in ("forecast_results",):
            res_prod.to_csv(out_dir / "forecast_results_production.csv", index=False)
        res_prod[(res_prod.forecast_granularity == "hourly")].to_csv(out_dir / "forecast_station_hourly_7day.csv", index=False)
        res_prod[(res_prod.forecast_granularity == "daily") & (res_prod.scope == "station")].to_csv(out_dir / "forecast_station_daily_7day.csv", index=False)
        res_prod[res_prod.scope.str.startswith("network")].to_csv(out_dir / "forecast_network_daily_7day.csv", index=False)
        dq_df.to_csv(out_dir / "data_quality_cleaning_log.csv", index=False)
        imp_daily_tbl.to_csv(out_dir / "imputed_station_daily.csv", index=False)
        imp_hourly_tbl.to_csv(out_dir / "imputed_station_hourly.csv.gz", index=False)
        imp_val.to_csv(out_dir / "imputation_validation.csv", index=False)
        tracker_path_tbl.to_csv(out_dir / "tracker_path_actual_filled_forecast.csv", index=False)
        prov_df.to_csv(out_dir / "source_provenance.csv", index=False)
        dim.to_csv(out_dir / "dim_station.csv", index=False)
        metrics_all.to_csv(out_dir / "model_metrics_station.csv", index=False)
        trk_metrics.to_csv(out_dir / "model_metrics_network_tracker.csv", index=False)

    results = dict(cfg=cfg, generated_at=generated_at, timer=T.t, cube=cube, dim=dim, cal=cal, fact_net=fact_net, sd=sd, sd_net=sd_net, prov=prov, dq_log=dq_log,
                   eda=eda_s, anomalies=anomalies, recon=recon, od=od, bt=bt, ctx=ctx, tbl=dict(hourly=tbl_h, daily=tbl_d, network=tbl_n), sel=dict(hourly=sel_h_tbl, daily=sel_d_tbl, network=sel_n_tbl),
                   sel_h=sel_h, sel_d=sel_d, prod_h=eff_h, prod_d=eff_d, sel_reason=sel_reason, use_trk=use_trk, win=win, imp=imp, imp_val=imp_val, tfc_all=tfc_all, net_bands=net_bands,
                   tracker_T=tracker_T, tracker_src=tracker_src, per_fold=per_fold, ip=ip, nbt=nbt, ntbl=ntbl, sel_t=sel_t, tband=tband, tfc=tfc, tracker_series=sdf, forecast=f,
                   forecast_results=forecast_results, oos=oos, checks=dict(station_sum=chk_station, network_sum=chk_net, daily_from_hourly=c1, network_from_station=c2, leakage=leak),
                   run_env=dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__), sarima_skipped=skip_sarima)
    from . import acceptance, dashboard, report
    results["acceptance"] = acceptance.evaluate(results)
    with T.stage("report_dashboard", log):
        report.write_report(results, out_dir / "final_report.md")
        dashboard.build(results, out_dir / "dashboard.html")
    # persist total runtime
    results["timer"] = T.t
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"generated_at": generated_at, "stage_seconds": T.t, "total_seconds": round(sum(T.t.values()), 1), "selected_hourly": sel_h, "selected_daily": sel_d, "production_model": eff_h, "run_date": str(win["as_of"].date()), "forecast_window": [str(win["start"].date()), str(win["end"].date())],
                   "selected_tracker": sel_t, "checks": results["checks"], "env": results["run_env"]}, fh, indent=2)
    log(f"Done. Total {sum(T.t.values()):.1f}s. Outputs in {out_dir}")
    return results
