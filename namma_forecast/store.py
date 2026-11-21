"""NFR-07 / FR-18: SQLite store for dimensions, facts, forecast_results, metrics, data-quality log and run log."""
from __future__ import annotations
import sqlite3
from pathlib import Path
import pandas as pd

INDEXES = {
    "fact_station_hourly_ridership": ["(date, hour, station_id)", "(station_id)"],
    "fact_station_daily": ["(date, station_id)"],
    "forecast_results": ["(run_type, scope, forecast_granularity, target_measure, station_id, target_datetime)"],
    "dim_station": ["(station_id)"], "dim_calendar": ["(date)"],
}


def write_db(path: Path, tables: dict[str, pd.DataFrame]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    for name, df in tables.items():
        df = df.copy()
        for c in df.columns:
            if str(df[c].dtype).startswith("datetime"):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
        df.to_sql(name, con, index=False, if_exists="replace")
    for t, idx in INDEXES.items():
        if t in tables:
            for i, cols in enumerate(idx):
                con.execute(f"CREATE INDEX IF NOT EXISTS ix_{t}_{i} ON {t} {cols}")
    con.commit(); con.close()


def update_actuals(db_path: Path, observed: pd.DataFrame) -> int:
    """FR-18: fill actual_value of stored forecasts once the target time has been observed.
    `observed` columns: target_datetime, station_id (None for network), forecast_granularity, scope, target_measure, actual_value."""
    con = sqlite3.connect(db_path)
    n = 0
    for r in observed.itertuples(index=False):
        cur = con.execute("UPDATE forecast_results SET actual_value=? WHERE target_datetime=? AND forecast_granularity=? AND scope=? AND target_measure=? "
                          "AND COALESCE(station_id,'')=COALESCE(?,'') AND run_type='production'",
                          (float(r.actual_value), r.target_datetime, r.forecast_granularity, r.scope, r.target_measure, r.station_id))
        n += cur.rowcount
    con.commit(); con.close()
    return n
