"""Real-time data acquisition and real-dataset building.

Sources (public, no credentials required):
  - IBTrACS v04r01 historical North Indian Ocean records (NOAA NCEI)
  - IBTrACS v04r01 ACTIVE list (currently active / season-in-progress storms)

Pipeline:
  fetch_ibtracs()   -> download & cache raw CSVs        (data/external/)
  normalize_ibtracs() -> clean to a uniform 6-hourly schema (SID/timestamp/lat/lon/wind/pressure)
  build_real_dataset() -> build storm-centric samples + strict seasonal splits + SHA256 manifest
                          into a dedicated data dir (e.g. data/real/)
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from .build_dataset import DatasetBuilder

IBTRACS_BASE = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-"
    "climate-stewardship-ibtracs/v04r01/access/csv/"
)
IBTRACS_FILES = {
    "ni": "ibtracs.NI.list.v04r01.csv",
    "active": "ibtracs.ACTIVE.list.v04r01.csv",
}

WIND_FALLBACK_COLS = ["WMO_WIND", "USA_WIND", "NEWDELHI_WIND"]
PRES_FALLBACK_COLS = ["WMO_PRES", "USA_PRES", "NEWDELHI_PRES"]

DATA_SOURCE_TAG = "ibtracs_v04r01_real"


def fetch_ibtracs(cache_dir: str = "data/external",
                  which: Tuple[str, ...] = ("ni", "active"),
                  force: bool = False) -> Dict[str, Path]:
    """Download IBTrACS CSVs to cache_dir unless present.

    Returns {key: local Path}.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}
    for key in which:
        fname = IBTRACS_FILES[key]
        dest = cache_dir / fname
        if dest.exists() and not force:
            print(f"[fetch] cached {fname}")
        else:
            url = IBTRACS_BASE + fname
            print(f"[fetch] downloading {url}")
            r = requests.get(url, timeout=300)
            r.raise_for_status()
            dest.write_bytes(r.content)
            print(f"[fetch] saved {len(r.content)} bytes -> {dest}")
        paths[key] = dest
    return paths


def clean_ibtracs_df(df: pd.DataFrame, six_hourly: bool = True) -> pd.DataFrame:
    """Clean an already-loaded IBTrACS dataframe to a uniform storm-centric schema.

    Input (raw IBTrACS columns): SID, ISO_TIME, BASIN, NATURE, LAT, LON,
    WMO_WIND, WMO_PRES, USA_WIND, USA_PRES, NEWDELHI_WIND, NEWDELHI_PRES ...
    Also accepts an already-normalized table (storm_id/timestamp/lat/lon/wind/pressure).

    Output columns: storm_id, timestamp, season, lat, lon, wind, pressure [, basin]
    IBTrACS raw files embed descriptor/unit rows inside the column block
    ("degrees_north", "degrees_east", "kts", "mb", ...) — these are dropped here.
    """
    is_raw = ("SID" in df.columns or "LAT" in df.columns)

    if not is_raw:
        # Already-normalized table (produced by fetch_real_data.py).
        for col in ["lat", "lon", "wind"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["lat", "lon", "wind"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["season"] = df["timestamp"].dt.year
        df = df.sort_values(["storm_id", "timestamp"])
        if six_hourly:
            df = resample_six_hourly(df)
        return df.reset_index(drop=True)

    # --- raw IBTrACS schema below ---
    # Remove descriptor/unit rows interleaved in the raw CSV.
    if "SID" in df.columns:
        df = df[df["SID"].notna() & (df["SID"].astype(str).str.strip() != "")]
    for col in ["LAT", "LON"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["LAT", "LON"])
    if "ISO_TIME" in df.columns:
        df["timestamp"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

    # Prefer tropical-cyclone nature records; keep older rows with blank NATURE.
    if "NATURE" in df.columns and "TC" in set(df["NATURE"].astype(str)):
        df = df[df["NATURE"].astype(str).isin(["TC", "nan", ""])]

    # Wind: WMO -> USA -> NEWDELHI fallback chain
    wind = df[[c for c in WIND_FALLBACK_COLS if c in df.columns]].bfill(axis=1).iloc[:, 0]
    df["wind"] = pd.to_numeric(wind, errors="coerce")
    pres = df[[c for c in PRES_FALLBACK_COLS if c in df.columns]].bfill(axis=1).iloc[:, 0]
    df["pressure"] = pd.to_numeric(pres, errors="coerce")
    df = df.dropna(subset=["wind"])

    df = df.rename(columns={"SID": "storm_id", "LAT": "lat", "LON": "lon"})

    # Pressure may have gaps across agencies -> forward fill per storm.
    if "pressure" in df.columns and not df["pressure"].isna().all():
        df["pressure"] = df.groupby("storm_id")["pressure"].transform(
            lambda s: s.ffill().bfill().fillna(990.0))

    columns = ["storm_id", "timestamp", "lat", "lon", "wind", "pressure"]
    if "BASIN" in df.columns:
        df["basin"] = df["BASIN"].astype(str).str.strip()
        columns.append("basin")
    df = df[columns].copy()
    df["season"] = df["timestamp"].dt.year
    df = df.sort_values(["storm_id", "timestamp"])

    if six_hourly:
        df = resample_six_hourly(df)
    return df.reset_index(drop=True)


def normalize_ibtracs(path: str, six_hourly: bool = True) -> pd.DataFrame:
    """Load + clean a raw IBTrACS CSV from disk."""
    df = pd.read_csv(path, low_memory=False)
    return clean_ibtracs_df(df, six_hourly=six_hourly)


def resample_six_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Subsample to a strict 6-hourly grid (00/06/12/18 UTC).

    IBTrACS NI records are 3-hourly; the rest of the TC-AI pipeline assumes a
    6-hour cadence (history windows, forecast horizons, translation speed).
    """
    keep = df["timestamp"].dt.hour.isin([0, 6, 12, 18])
    df = df[keep]
    # Keep storms with at least 6h history + 24h horizon (11+ points).
    counts = df.groupby("storm_id")["timestamp"].count()
    valid = counts[counts >= 11].index
    return df[df["storm_id"].isin(valid)].reset_index(drop=True)


def consolidate(ni_df: pd.DataFrame, active_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Merge historical NI tracks with any currently-active NI storms."""
    frames = [ni_df]
    if active_df is not None and len(active_df):
        active_ni = active_df[active_df["BASIN"].astype(str).str.strip().isin(["NI", "ni"])]
        if len(active_ni):
            frames.append(clean_ibtracs_df(active_ni))
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["storm_id", "timestamp"])
    return out.sort_values(["storm_id", "timestamp"]).reset_index(drop=True)


def build_real_dataset(csv_path: str, data_dir: str,
                       history_length: int = 6,
                       horizons: List[int] = (6, 12, 24),
                       recent_since: int = 2024,
                       min_season: int = 2012) -> Dict[str, object]:
    """Build the REAL storm-centric dataset into `data_dir`.

    Seasonal split (plan.md, zero temporal leakage):
      train <= 2018, val 2019-2020, test >= 2021.
    An additional `recent` index (seasons >= recent_since) captures the
    newest real storms — the "live / real-time" holdout window.
    """
    data_dir = Path(data_dir)
    df = normalize_ibtracs(csv_path, six_hourly=True)
    df = df[df["season"] >= min_season].reset_index(drop=True)

    builder = DatasetBuilder(
        data_dir=str(data_dir),
        synthetic_satellite=False,  # no fake imagery on real tracks
        synthetic_weather=False,    # no synthetic ERA5 on real tracks
        structural_labels=False,    # plan: structural labels are a stretch, not a core claim
        history_length=history_length,
        horizons=list(horizons),
    )

    total_samples = 0
    storm_count = 0
    for storm_id, storm_df in df.groupby("storm_id"):
        storm_df = storm_df.sort_values("timestamp")
        n = builder.build_storm(storm_df, split="all",
                                era5_base=None, satellite_base=None,
                                data_source=DATA_SOURCE_TAG)
        total_samples += n
        storm_count += 1

    builder.generate_negative_samples(n=0)
    builder.write_index_files(split_by_season=True)

    recent_n = _write_recent_index(str(data_dir), recent_since=recent_since)

    source_meta = {
        "data_source": DATA_SOURCE_TAG,
        "provider": "NOAA NCEI IBTrACS v04r01",
        "basin": "North Indian Ocean (NI)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "min_season": int(min_season),
        "storm_count": storm_count,
        "track_rows": int(len(df)),
        "sample_count": total_samples,
        "recent_since_season": int(recent_since),
        "recent_sample_count": recent_n,
        "season_range": [int(df["season"].min()), int(df["season"].max())],
        "six_hourly": True,
    }
    (data_dir / "source.json").write_text(json.dumps(source_meta, indent=2))
    print(f"[real] source metadata -> {data_dir / 'source.json'}")
    return source_meta


def _write_recent_index(data_dir: str, recent_since: int) -> int:
    """Write recent_index.csv: real test-season samples from the newest storms."""
    data_dir = Path(data_dir)
    samples_dir = data_dir / "samples"
    if not samples_dir.exists():
        return 0
    rows = []
    for sample_p in sorted(samples_dir.iterdir()):
        if not sample_p.is_dir():
            continue
        meta_p = sample_p / "meta.json"
        if not meta_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        if meta.get("negative"):
            continue
        season = meta.get("season")
        if season is None or int(season) < recent_since:
            continue
        rows.append({
            "sample_id": sample_p.name,
            "split": "recent",
            "storm_id": meta.get("storm_id", ""),
            "season": int(season),
        })
    if rows:
        recent_df = pd.DataFrame(rows)
        recent_df.to_csv(data_dir / "recent_index.csv", index=False)
    print(f"[real] recent/live samples (season >= {recent_since}): {len(rows)}")
    return len(rows)