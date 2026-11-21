"""Builds the interactive dashboard payload (JSON) and the self-contained HTML file (SRS section 8)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

from .etl import OBS, NOT_OPEN
from .models import MODEL_INFO
from . import forecast as fc

TEMPLATE = Path(__file__).parent / "dashboard_template.html"


def _r(a, nd=1):
    return np.round(np.asarray(a, float), nd).tolist()


def _tbl(df: pd.DataFrame, keep=None):
    d = df.copy()
    if keep:
        d = d[[c for c in keep if c in d.columns]]
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(2)
    return json.loads(d.to_json(orient="records"))


def payload(R: dict) -> dict:
    cube, dim, cal, cfg = R["cube"], R["dim"], R["cal"], R["cfg"]
    S, D = cube.S, cube.D
    f = R["forecast"]; ctx = R["ctx"]; bt = R["bt"]
    dates = f["dates"]; H = len(dates)
    imp, win = R["imp"], R["win"]
    ext = pd.date_range(cube.dates[0], win["start"] - pd.Timedelta(days=1)); nExt = len(ext)      # observed + imputed days before the forecast window
    kind = imp["kind"][:nExt]
    cali = cal.set_index("date")
    def calrow(d):
        r = cali.loc[d]
        g = lambda k: r[k] if isinstance(r[k], str) else ""
        return {"d": str(d.date()), "dow": int(d.dayofweek), "hol": g("holiday_name"), "fest": g("festival_name"), "dis": g("disruption_name"), "obs": g("observance_name")}
    flat = lambda A: np.where(np.isnan(A), -1, A).reshape(S, -1)
    obs_days = (cube.status == OBS).sum(axis=1)
    stations = [{"id": r.station_id, "name": r.station_name, "line": r.line_name or "", "ich": bool(r.is_interchange), "open": r.opening_date or "",
                 "first": r.first_observed_date or "", "days": int(obs_days[i]), "code": r.station_code or "", "alias": r.source_aliases or "", "match": r.ref_match,
                 "flag": "RECENTLY_OPENED" if cube.open_idx[i] > 0 else ("LIMITED_HISTORY" if obs_days[i] < cfg["validation"]["min_obs_days_for_forecast"] else "OK")}
                for i, r in enumerate(dim.itertuples())]
    fh = R["forecast_results"]
    fh = fh[(fh.run_type == "production") & (fh.forecast_granularity == "hourly")]
    fcd = {}
    for m in ("entries", "exits"):
        sub = fh[fh.target_measure == m].sort_values(["station_id", "horizon_step"])
        n = len(sub) // S
        fcd[m] = {"pred": _r(sub.predicted_value.to_numpy().reshape(S, n)), "lo": _r(sub.lower_80.to_numpy().reshape(S, n)), "hi": _r(sub.upper_80.to_numpy().reshape(S, n)),
                  "base": _r(f["baseline"][m].reshape(S, -1)), "dq": R["ip"]["daily"][m], "nq": R["ip"]["network"][m], "nqd": [[round(b[0], 4), round(b[1], 4)] for b in R["net_bands"]]}
    o_last = bt["origins"][-1]
    btd = {"origin": str(ctx.dates[o_last].date()), "dates": [str(d.date()) for d in ctx.dates[o_last + 1:o_last + 1 + H]]}
    for m in ("entries", "exits"):
        btd[m] = {"pred": _r(bt["pred"][("pipeline", m, len(bt["origins"]) - 1)].reshape(S, -1)), "act": _r(flat(bt["act"][(m, len(bt["origins"]) - 1)]))}
    # tracker
    tr = R["fact_net"]; sdf = R["tracker_series"]
    tfc = R["tfc"]; q = R["tband"]
    nb = R["nbt"]["records"].query("model == @R['sel_t']").dropna(subset=["actual"])
    tracker = {"dates": [str(d.date()) for d in tr["date"]], "total": tr["official_daily_total"].astype(int).tolist(),
               "parts": {k: tr[k].astype(int).tolist() for k in ("smart_cards", "tokens", "ncmc", "group_ticket", "qr")}, "qr_bad": tr["qr_inconsistent"].tolist(),
               "fc": {"dates": [str(d.date()) for d in R["tfc_all"]["date"]], "pred": _r(R["tfc_all"]["pred"], 0), "lo": _r(R["tfc_all"]["lo"], 0), "hi": _r(R["tfc_all"]["hi"], 0),
                  "t0": int((R["tfc_all"]["date"] < win["start"]).sum())},
               "filled": {"dates": [str(d.date()) for d in R["tracker_src"].index[R["tracker_src"] == "filled"]], "vals": _r(R["tracker_T"][R["tracker_src"] == "filled"].to_numpy(), 0)},
               "bt": [{"d": str(r.date.date()), "p": round(r.pred), "a": round(r.actual), "f": int(r.fold), "h": int(r.horizon)} for r in nb.itertuples()],
               "events": {str(d.date()): (cali.loc[d, "all_events"] if isinstance(cali.loc[d, "all_events"], str) else "") for d in tr["date"] if isinstance(cali.loc[d, "all_events"], str)}}
    cols = ["model", "label", "family", "measure", "MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "bias", "n", "n_mape_excluded", "wape_fold_mean", "wape_fold_std", "fold_wape", "runtime_s"]
    metrics = {k: _tbl(v, cols) for k, v in R["tbl"].items()}
    sel = {k: _tbl(v) for k, v in R["sel"].items()}
    ntbl = _tbl(R["ntbl"])
    flags = R["forecast_results"]  # placeholder to keep structure simple
    cov = []
    for s_ in range(S):          # 1 observed, 0 date missing, 2 not open, 3 imputed (model-filled)
        ch = ["3" if kind[i] == "gap" else str(int(cube.status[s_, i])) for i in range(cube.D)]
        cov.append("".join(ch) + ("3" if cube.open_idx[s_] >= 0 else "2") * (nExt - cube.D))
    impd = {}
    for m in ("entries", "exits"):
        daily = imp["filled"][m][:, :nExt, :].sum(axis=2)
        mask = np.isin(kind, ["gap", "extension", "bridge"])
        impd[m] = [[int(round(daily[s_, i])) if mask[i] else None for i in range(nExt)] for s_ in range(S)]
    ic = {k: int((kind == k).sum()) for k in ("gap", "extension", "bridge")}
    dq = {"log": R["dq_log"], "prov": R["prov"], "coverage": cov, "missing_dates": [str(d.date()) for d in cube.dates[~cube.date_has_data]], "imputed_counts": ic,
          "flag_counts": {"OBSERVED": int(((cube.status == OBS).sum()) * 24), "NOT_OPEN": int((cube.status == NOT_OPEN).sum() * 24),
                          "SUSPECT_entries": int(cube.suspect_e.sum()), "SUSPECT_exits": int(cube.suspect_x.sum())},
          "suspect_by_station": {dim.station_name[s]: int(cube.suspect_e[s].sum() + cube.suspect_x[s].sum()) for s in range(S) if (cube.suspect_e[s].sum() + cube.suspect_x[s].sum()) > 3}}
    return {
        "meta": {"generated_at": R["generated_at"], "version": cfg["version"], "hourly_first": str(cube.dates[0].date()), "hourly_last": str(cube.dates[-1].date()),
                 "tracker_first": tracker["dates"][0], "tracker_last": tracker["dates"][-1], "n_stations": S, "origin": str(f["origin"].date()), "fc_dates": [str(d.date()) for d in dates],
                 "rt": {"run_date": str(win["as_of"].date()), "fc_start": str(win["start"].date()), "fc_end": str(win["end"].date()), "tracker_last": str(win["tracker_last"].date()), "hourly_last": str(win["last_hourly"].date()),
                       "stale_hourly_days": win["stale_hourly_days"], "stale_tracker_days": win["stale_tracker_days"], "h_trk": win["h_trk"], "use_trk": bool(R["use_trk"]), "prod": R["prod_h"], "best_obs": R["sel_h"],
                       "reason": R["sel_reason"], "inflation": cfg["realtime"]["band_inflation_stale"] if R["use_trk"] else 1.0, "imp_counts": ic, "imp_val": _tbl(R["imp_val"]),
                       "tracker_filled": int((R["tracker_src"] == "filled").sum())},
                 "sel": {"hourly": R["prod_h"], "daily": R["prod_d"], "best_observed_hourly": R["sel_h"], "best_observed_daily": R["sel_d"], "tracker": R["sel_t"], "labels": {k: v[0] for k, v in MODEL_INFO.items()},
                         "tracker_label": R["ntbl"].set_index("model").label[R["sel_t"]]},
                 "timer": R["timer"], "env": R["run_env"], "cfg": {k: cfg[k] for k in ("hour_bucket_convention", "timezone", "non_operating_hours", "horizon_days")},
                 "evaluation": {"n_folds": cfg["evaluation"]["n_folds"], "mape_min": cfg["evaluation"]["mape_min_actual"], "cov": cfg["evaluation"]["interval_coverage"],
                                "fold_origins": [str(ctx.dates[o].date()) for o in bt["origins"]],
                                "test_windows": [[str(ctx.dates[o + 1].date()), str(ctx.dates[o + H].date())] for o in bt["origins"]],
                                "tracker_folds": int(R["ntbl"].folds.max()), "tracker_first_origin": cfg["network_tracker"]["first_origin"]},
                 "checks": R["checks"], "sarima_skipped": R["sarima_skipped"]},
        "stations": stations, "dates": [calrow(d) for d in ext], "fdates": [calrow(d) for d in dates],
        "hist": {"entries": _r(flat(cube.entries), 0), "exits": _r(flat(cube.exits), 0), "status": cov, "impd": impd},
        "fc": fcd, "bt": btd, "tracker": tracker, "metrics": metrics, "sel": sel, "tracker_metrics": ntbl,
        "per_fold": {k: _tbl(v, ["model", "measure", "fold", "origin", "MAE", "RMSE", "MAPE", "WAPE", "sMAPE", "n"]) for k, v in R["per_fold"].items()},
        "eda": R["eda"], "anomalies": R["anomalies"], "recon": R["recon"], "oos": R["oos"], "od": R["od"], "dq": dq,
        "acceptance": R["acceptance"], "recon_ratio": R["recon"]["ratio_median_full_network"],
        "ip": R["ip"],
    }


def _clean(o):
    """Recursively replace NaN/inf by None and numpy scalars by Python ones so the payload is strict JSON."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        o = o.item()
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def build(R: dict, out_html: Path) -> None:
    data = _clean(payload(R))
    out_html = Path(out_html)
    blob = json.dumps(data, separators=(",", ":"), allow_nan=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    (out_html.parent / "dashboard_data.json").write_text(blob, encoding="utf-8")
    tpl = TEMPLATE.read_text(encoding="utf-8")
    shell = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
             '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
             '<style>body{margin:0}:root{color-scheme:light}</style>\n')
    out_html.write_text(shell + tpl.replace("/*__DATA__*/", blob.replace("</", "<\\/")), encoding="utf-8")     # self-contained, works offline
    (out_html.parent / "dashboard_page.html").write_text(tpl.replace("/*__DATA__*/", ""), encoding="utf-8")                                  # data fetched from dashboard_data.json (for hosting)
