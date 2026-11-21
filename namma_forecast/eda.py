"""FR-05 / SR-01 / FR-12: exploratory statistics, anomaly detection and station-vs-tracker reconciliation."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

from .etl import Cube, OBS
from .dimensions import alias_lookup, canonical_name

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def eda_summary(cube: Cube, dim: pd.DataFrame, sd: pd.DataFrame, cal: pd.DataFrame) -> dict:
    E = cube.entries
    obs_days = np.where(cube.date_has_data)[0]
    dow = cube.dates.dayofweek.to_numpy()
    full = np.where(((cube.status == OBS).sum(axis=0) == cube.S) & cube.date_has_data)[0]    # days with the complete 83-station network
    out: dict = {}
    prof = {}
    for lab, sel in (("weekday", full[dow[full] < 5]), ("saturday", full[dow[full] == 5]), ("sunday", full[dow[full] == 6])):
        prof[lab] = np.nanmean(np.nansum(E[:, sel, :], axis=0), axis=0).round(0).tolist()
    out["network_hour_profile_entries"] = prof
    wd = np.array(prof["weekday"])
    out["peak_hours"] = {"weekday_am": int(np.argmax(wd[5:12]) + 5), "weekday_pm": int(np.argmax(wd[15:21]) + 15), "weekday_peak_value": float(wd.max())}
    net = pd.Series(np.nansum(np.where(cube.status[:, :, None] == OBS, E, 0), axis=(0, 2)), index=cube.dates)[cube.date_has_data]
    nfull = net[net.index.isin(cube.dates[full])]
    byd = nfull.groupby(nfull.index.dayofweek).mean()
    out["avg_daily_entries_by_dow"] = {DOW[k]: float(v) for k, v in byd.items()}
    out["weekday_vs_weekend_ratio"] = float(byd.loc[0:4].mean() / byd.loc[5:6].mean())
    out["network_days_full_network"] = int(len(full))
    tot = sd.groupby("station_id")["entries"].sum()
    top = tot.sort_values(ascending=False)
    names = dim.set_index("station_id")["station_name"]
    out["top10_stations_entries"] = [{"station": names[i], "total": float(v)} for i, v in top.head(10).items()]
    out["top10_share_pct"] = float(top.head(10).sum() / top.sum() * 100)
    out["lowest_stations_entries"] = [{"station": names[i], "total": float(v)} for i, v in top.tail(5).items()]
    return out


def anomalies(cube: Cube, dim: pd.DataFrame, cal: pd.DataFrame, tracker: pd.DataFrame, limit: int = 25) -> dict:
    """Station-day anomalies (entries vs same-weekday median) and network-day anomalies (station-derived and tracker)."""
    cal_i = cal.set_index("date")
    ev = lambda d: (cal_i.loc[d, "all_events"] if d in cal_i.index and isinstance(cal_i.loc[d, "all_events"], str) else "")
    T = np.nansum(np.where(cube.status[:, :, None] == OBS, cube.entries, 0), axis=2)
    dow = cube.dates.dayofweek.to_numpy()
    rows = []
    for s in range(cube.S):
        ok = (cube.status[s] == OBS)
        for k in range(7):
            idx = np.where(ok & (dow == k))[0]
            idx = idx[idx >= cube.open_idx[s] + 0]
            if len(idx) < 4:
                continue
            v = T[s, idx]; med = np.median(v); mad = np.median(np.abs(v - med)); mad = max(mad, 0.03 * med, 1)
            z = 0.6745 * (v - med) / mad
            for i, zz, vv in zip(idx, z, v):
                if abs(zz) > 4 and med > 500:
                    rows.append({"date": str(cube.dates[i].date()), "scope": dim.station_name[s], "level": "station-day", "actual": float(vv), "expected": float(med),
                                 "deviation_pct": float((vv / med - 1) * 100), "robust_z": float(zz), "event": ev(cube.dates[i])})
    st = pd.DataFrame(rows)
    st = st.reindex(st["robust_z"].abs().sort_values(ascending=False).index).head(limit) if len(st) else st
    full = np.where(((cube.status == OBS).sum(axis=0) == cube.S) & cube.date_has_data)[0]
    net = pd.Series(T.sum(axis=0), index=cube.dates).iloc[full]
    rows = []
    for d, v in net.items():
        same = net[(net.index.dayofweek == d.dayofweek) & (net.index != d)]
        if len(same) >= 3:
            med = same.median()
            if abs(v / med - 1) > 0.08:
                rows.append({"date": str(d.date()), "scope": "network (station-derived)", "level": "network-day", "actual": float(v), "expected": float(med),
                             "deviation_pct": float((v / med - 1) * 100), "event": ev(d)})
    nd = pd.DataFrame(rows)
    t = tracker.set_index("date")["official_daily_total"]
    rows = []
    for d, v in t.items():
        win = t[(t.index.dayofweek == d.dayofweek) & (abs((t.index - d).days) <= 35) & (t.index != d)]
        if len(win) >= 3:
            med = win.median()
            if abs(v / med - 1) > 0.25:
                rows.append({"date": str(d.date()), "scope": "network (tracker)", "level": "network-day", "actual": float(v), "expected": float(med),
                             "deviation_pct": float((v / med - 1) * 100), "event": ev(d)})
    tk = pd.DataFrame(rows)
    tk = tk.reindex(tk["deviation_pct"].abs().sort_values(ascending=False).index).head(limit) if len(tk) else tk
    return {"station_day": st.to_dict("records") if len(st) else [], "network_station_derived": nd.to_dict("records") if len(nd) else [],
            "network_tracker": tk.to_dict("records") if len(tk) else []}


def reconcile_with_tracker(sd_net: pd.DataFrame, tracker: pd.DataFrame) -> dict:
    """FR-12: station-derived entries vs independent tracker totals on overlapping dates (definitions differ -> scaling ratio)."""
    t = tracker.set_index("date")["official_daily_total"]
    j = sd_net.set_index("date")[["entries", "exits", "n_stations"]].join(t, how="inner")
    j["ratio"] = j["entries"] / j["official_daily_total"]
    j["diff"] = j["entries"] - j["official_daily_total"]
    j["full_network"] = j["n_stations"] == j["n_stations"].max()
    use = j[j["full_network"]]
    return {"table": [{"date": str(i.date()), "station_entries": float(r.entries), "station_exits": float(r.exits), "tracker_total": float(r.official_daily_total),
                       "ratio": float(r.ratio), "diff": float(r["diff"]), "n_stations": int(r.n_stations)} for i, r in j.iterrows()],
            "overlap_days": int(len(j)), "ratio_median_full_network": float(use["ratio"].median()) if len(use) else None,
            "ratio_p10": float(use["ratio"].quantile(0.1)) if len(use) else None, "ratio_p90": float(use["ratio"].quantile(0.9)) if len(use) else None,
            "ratio_mean": float(use["ratio"].mean()) if len(use) else None}


def od_summary(cfg: dict, dim: pd.DataFrame, lookup: dict, path, cache_path) -> dict | None:
    """Optional exploratory origin-destination summary (not used by the core forecast). Cached so the raw 80 MB file is optional."""
    import os
    if path and os.path.exists(path):
        df = pd.read_csv(path, sep=";")
        names = {n: i for i, n in enumerate(dim["station_name"])}
        o = df["Origin Station"].map(lambda n: names.get(canonical_name(n, lookup)[0], -1)); d = df["Destination Station"].map(lambda n: names.get(canonical_name(n, lookup)[0], -1))
        S = len(names)
        M = np.zeros((S, S), dtype=np.int64)
        ok = (o >= 0) & (d >= 0)
        np.add.at(M, (o[ok].to_numpy(), d[ok].to_numpy()), df.loc[ok, "Ridership"].to_numpy())
        byhour = df.groupby("Hour").agg(trips=("Ridership", "sum"), revenue=("Revenue", "sum"))
        byfare = df.groupby("Fare").apply(lambda g: pd.Series({"trips": g["Ridership"].sum(), "revenue": g["Revenue"].sum()}), include_groups=False)
        bystops = df.groupby("Travelled Stations")["Ridership"].sum()
        res = {"dates": [df["Date"].min(), df["Date"].max()], "trips": int(df["Ridership"].sum()), "revenue": int(df["Revenue"].sum()),
               "avg_fare": float(df["Revenue"].sum() / df["Ridership"].sum()), "matrix": M.tolist(),
               "by_hour": {int(h): {"trips": int(r.trips), "revenue": int(r.revenue)} for h, r in byhour.iterrows()},
               "by_fare": {int(f): {"trips": int(r.trips), "revenue": int(r.revenue)} for f, r in byfare.iterrows()},
               "by_stops": {int(k): int(v) for k, v in bystops.items()}, "unmapped_rows": int((~ok).sum()), "source": str(path)}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(res, f)
        return res
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)
    return None
