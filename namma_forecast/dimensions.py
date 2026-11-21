"""FR-04: canonical station dimension (dim_station) and date dimension (dim_calendar)."""
from __future__ import annotations
import re
import numpy as np
import pandas as pd

LINE_EMOJI = {"🟣": "Purple", "🟢": "Green", "🟡": "Yellow", "🌸": "Pink", "🔵": "Blue", "🔴": "Red", "🟠": "Orange", "⚪": "Grey/White"}
# Reference-dataset spellings that differ from the hourly (RTI-derived) spellings
MANUAL_REF_MATCH = {
    "Nallurahalli": "Nallurhalli",
    "Pantharapalya - Nayandahalli": "Pantharapalya–Nayandahalli",
    "Mysore Road": "Mysuru Road",
    "Seetharampalya": "Seetharamapalya",
    "Sir M. Visvesvaraya Stn., Central College": "Sir.M.Visvesvaraya Stn., Central College",
    "Yeshwantpur": "Yeshwanthpur",
    "Krishnarajapura": "Krishnarajapura (K.R.Pura)",
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def alias_lookup(aliases: pd.DataFrame, codes: pd.DataFrame) -> dict[str, str]:
    """normalised alias -> canonical station name (old names and station codes -> current names)."""
    m = {}
    for _, r in aliases.iterrows():
        m[norm(r["alt_name"])] = r["name"]
        m[norm(r["name"])] = r["name"]
    for _, r in codes.iterrows():
        m[norm(r["code"])] = r["name"]
        m[norm(r["name"])] = r["name"]
    return m


def canonical_name(name: str, lookup: dict[str, str]) -> tuple[str, bool]:
    """Return (canonical name, was_changed). Unknown names are kept as-is (new station)."""
    clean = re.sub(r"\s+", " ", str(name)).strip()
    c = lookup.get(norm(clean), clean)
    return c, c != name


def build_dim_station(station_order: list[str], first_obs: dict[str, str | None], ref_stations: pd.DataFrame,
                      codes: pd.DataFrame, aliases: pd.DataFrame, first_data_date: str) -> pd.DataFrame:
    ref = ref_stations.copy()
    ref["_k"] = ref["name_eng"].map(norm)
    ref_idx = {k: i for i, k in enumerate(ref["_k"])}
    code_by_name = codes.groupby("name")["code"].first().to_dict()
    alias_by_name = aliases.groupby("name")["alt_name"].apply(lambda s: "; ".join(sorted(set(s)))).to_dict()
    rows = []
    for i, name in enumerate(station_order, 1):
        key = norm(MANUAL_REF_MATCH.get(name, name))
        r = ref.iloc[ref_idx[key]] if key in ref_idx else None
        lines = ""
        if r is not None:
            lines = "/".join(LINE_EMOJI.get(ch, "?") for ch in re.findall(r"🟣|🟢|🟡|🌸|🔵|🔴|🟠|⚪", str(r["lines"])))
        fo = first_obs.get(name)
        opened_in_window = fo is not None and fo > first_data_date
        rows.append({
            "station_id": f"STN{i:03d}", "station_name": name,
            "station_name_kn": None if r is None else r["name_kan"],
            "station_code": code_by_name.get(name),
            "line_name": lines or None,
            "is_interchange": None if r is None else bool(r["is_interchange"]),
            "is_terminus": None if r is None else bool(r["is_terminus"]),
            "opening_date": fo if opened_in_window else None,
            "opening_date_basis": "first entries record in RTI data (opened inside the data window)" if opened_in_window
                                  else f"open on/before {first_data_date} (true opening date not in data)",
            "first_observed_date": fo,
            "is_active": True,
            "source_aliases": alias_by_name.get(name),
            "ref_match": "matched" if r is not None else "UNMATCHED",
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------- calendar
HOLIDAY_KW = ["diwali", "deepavali", "rajayotsava", "rajyothsava", "christmas day", "republic day", "independence day",
              "makara sankranti", "mahashivratri", "holi", "ugadi", "eid", "good friday", "mahavir", "ambedkar", "basava jayanti",
              "may day", "muharram", "varalakshmi", "varamahalakshmi", "janmashtami", "ganesh chaturthi", "ayudha", "gandhi jayanti",
              "vijaya dashami", "vijayadashami", "dussehra", "mahanavami", "valmiki", "gurunanak", "guru nanak", "kanakadasa",
              "rama navami", "bakrid", "raksha bandhan", "naraka chaturdashi", "balipadyami", "mahalaya"]
FESTIVAL_KW = ["karaga", "huttari", "kadalekai", "bhai duj", "govardhan", "akshaya", "new year", "valentine", "christmas eve", "easter"]
EVENT_KW = ["ipl", "world cup", "t20", "odi", "t20i", "ranji", "champions trophy", "asia cup", "marathon", "concert", "aero india",
            "festival d", "music festival", "summit", "ed sheeran", "guns n", "bandland", "echoes"]
DISRUPTION_KW = ["bandh", "tragedy", "fare hike", "yellow line begins"]
NON_EVENT_KW = ["eclipse", "solstice", "equinox", "not visible"]


def classify_event(ev: str) -> str:
    e = ev.lower()
    if any(k in e for k in NON_EVENT_KW):
        return "observance"
    if any(k in e for k in DISRUPTION_KW):
        return "disruption"
    if any(k in e for k in EVENT_KW):
        return "event"
    if any(k in e for k in HOLIDAY_KW):
        return "holiday"
    if any(k in e for k in FESTIVAL_KW):
        return "festival"
    return "observance"


def build_dim_calendar(events: pd.DataFrame, start: str, end: str, hourly_dates: set, tracker_dates: set) -> pd.DataFrame:
    idx = pd.date_range(start, end, freq="D")
    cal = pd.DataFrame({"date": idx})
    cal["day_name"] = cal["date"].dt.day_name()
    cal["day_of_week"] = cal["date"].dt.dayofweek
    cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["month"] = cal["date"].dt.month
    cal["quarter"] = cal["date"].dt.quarter
    cal["year"] = cal["date"].dt.year
    cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(int)
    ev = events.copy()
    ev["etype"] = ev["Event"].map(classify_event)
    agg = ev.groupby("Date").apply(lambda g: pd.Series({
        "holiday_name": "; ".join(g.loc[g.etype == "holiday", "Event"]) or None,
        "festival_name": "; ".join(g.loc[g.etype.isin(["festival", "event"]), "Event"]) or None,
        "disruption_name": "; ".join(g.loc[g.etype == "disruption", "Event"]) or None,
        "observance_name": "; ".join(g.loc[g.etype == "observance", "Event"]) or None,
        "all_events": "; ".join(g.Event),
    }), include_groups=False).reset_index().rename(columns={"Date": "date"})
    for c in ["date", "holiday_name", "festival_name", "disruption_name", "observance_name", "all_events"]:
        if c not in agg.columns:
            agg[c] = pd.Series(dtype="datetime64[ns]" if c == "date" else object)
    agg["date"] = pd.to_datetime(agg["date"])
    cal = cal.merge(agg, on="date", how="left")
    cal["is_holiday"] = cal["holiday_name"].notna().astype(int)
    cal["is_festival"] = cal["festival_name"].notna().astype(int)
    cal["is_disruption"] = cal["disruption_name"].notna().astype(int)
    cal["is_special_event"] = ((cal["is_festival"] + cal["is_disruption"]) > 0).astype(int)
    cal["has_hourly_data"] = cal["date"].dt.strftime("%Y-%m-%d").isin(hourly_dates).astype(int)
    cal["has_tracker_data"] = cal["date"].isin(tracker_dates).astype(int)
    cal["calendar_source"] = "significant_dates.csv (keyword-classified; verify against official Karnataka holiday list)"
    return cal
