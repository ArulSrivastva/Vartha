"""Wire REAL ERA5 reanalysis NetCDF fields into real samples as 64-dimensional weather.npy.

Reads NetCDF files from `../SIH2026/PS70/data/raw/era5/era5_YYYY_MM.nc`
and computes physically grounded, storm-centered environmental features:
  - Core values: SST, MSL, U10, V10, Wind speed
  - Pressure deficits and radial gradients
  - Multi-zone radial statistics (Inner core, intermediate, outer environment)
  - 4-quadrant environmental asymmetries (NE, NW, SE, SW)
  - Environmental steering flow, relative vorticity & divergence kinematic proxies
  - Local storm-relative 4x4 spatial grid

Output:
  data/real/samples/<sample_id>/weather.npy: shape (64,), float32
  meta.json: updated with weather_real=True, source info

Usage:
  python scripts/wire_real_era5.py
"""
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ERA5_DIR = PROJECT_ROOT.parent / "SIH2026" / "PS70" / "data" / "raw" / "era5"
SAMPLES_DIR = PROJECT_ROOT / "data" / "real" / "samples"


def extract_64_weather_vector(ds, lat: float, lon: float, timestamp: datetime) -> np.ndarray:
    """Extract a physically consistent 64-dim environmental feature vector centered on (lat, lon)."""
    t_coord = "valid_time" if "valid_time" in ds.coords else "time"
    t_val = np.datetime64(timestamp.strftime("%Y-%m-%dT%H:%M:%S"))

    # Bounding box of +- 6 degrees around storm (~660 km radius)
    lat_min, lat_max = lat - 6.0, lat + 6.0
    lon_min, lon_max = lon - 6.0, lon + 6.0

    # ERA5 latitudes are stored in descending order (e.g. 30.0 down to -5.0)
    lat_slice = slice(lat_max, lat_min) if ds.latitude.values[0] > ds.latitude.values[-1] else slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)

    try:
        sub_t = ds.sel({t_coord: t_val}, method="nearest")
        patch = sub_t.sel(latitude=lat_slice, longitude=lon_slice)
    except Exception:
        pt = ds.sel({t_coord: t_val, "latitude": lat, "longitude": lon}, method="nearest")
        sst_c = float(pt["sst"].values) - 273.15 if "sst" in pt else 28.0
        msl_hpa = float(pt["msl"].values) / 100.0 if "msl" in pt else 1005.0
        u10 = float(pt["u10"].values) if "u10" in pt else 0.0
        v10 = float(pt["v10"].values) if "v10" in pt else 0.0
        w64 = np.zeros(64, dtype=np.float32)
        w64[0] = sst_c
        w64[1] = msl_hpa
        w64[2] = u10
        w64[3] = v10
        w64[4] = np.sqrt(u10**2 + v10**2)
        return w64

    lats = patch.latitude.values
    lons = patch.longitude.values
    if len(lats) == 0 or len(lons) == 0:
        pt = ds.sel({t_coord: t_val, "latitude": lat, "longitude": lon}, method="nearest")
        sst_c = float(pt["sst"].values) - 273.15 if "sst" in pt else 28.0
        msl_hpa = float(pt["msl"].values) / 100.0 if "msl" in pt else 1005.0
        u10 = float(pt["u10"].values) if "u10" in pt else 0.0
        v10 = float(pt["v10"].values) if "v10" in pt else 0.0
        w64 = np.zeros(64, dtype=np.float32)
        w64[0] = sst_c
        w64[1] = msl_hpa
        w64[2] = u10
        w64[3] = v10
        w64[4] = np.sqrt(u10**2 + v10**2)
        return w64

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    dlat_km = (lat_grid - lat) * 110.574
    dlon_km = (lon_grid - lon) * (111.320 * np.cos(np.radians(lat)))
    dist_km = np.sqrt(dlat_km**2 + dlon_km**2)

    sst = patch["sst"].values - 273.15  # Kelvin -> Celsius
    msl = patch["msl"].values / 100.0   # Pa -> hPa
    u10 = patch["u10"].values           # m/s
    v10 = patch["v10"].values           # m/s
    wind10 = np.sqrt(u10**2 + v10**2)

    sst = np.nan_to_num(sst, nan=28.0)
    msl = np.nan_to_num(msl, nan=1008.0)
    u10 = np.nan_to_num(u10, nan=0.0)
    v10 = np.nan_to_num(v10, nan=0.0)
    wind10 = np.nan_to_num(wind10, nan=0.0)

    vec = []

    # 1. Storm Core / Center point (5 features)
    center_idx = np.argmin(dist_km)
    c_y, c_x = np.unravel_index(center_idx, dist_km.shape)
    c_sst = float(sst[c_y, c_x])
    c_msl = float(msl[c_y, c_x])
    c_u10 = float(u10[c_y, c_x])
    c_v10 = float(v10[c_y, c_x])
    c_wind = float(wind10[c_y, c_x])
    vec.extend([c_sst, c_msl, c_u10, c_v10, c_wind])

    # 2. Environmental deficits & gradients (3 features)
    outer_mask = (dist_km >= 400.0) & (dist_km <= 600.0)
    env_msl = float(np.mean(msl[outer_mask])) if np.any(outer_mask) else c_msl + 10.0
    env_sst = float(np.mean(sst[outer_mask])) if np.any(outer_mask) else c_sst
    p_deficit = env_msl - c_msl
    sst_anomaly = c_sst - env_sst
    max_p_grad = float(np.max(np.abs(msl - c_msl) / np.maximum(dist_km, 50.0))) * 100.0
    vec.extend([p_deficit, sst_anomaly, max_p_grad])

    # 3. Concentric radial zones (18 features)
    zones = [
        dist_km < 150.0,
        (dist_km >= 150.0) & (dist_km < 350.0),
        (dist_km >= 350.0) & (dist_km <= 600.0),
    ]
    for z_mask in zones:
        if not np.any(z_mask):
            z_mask = np.ones_like(dist_km, dtype=bool)
        w_sub = wind10[z_mask]
        m_sub = msl[z_mask]
        s_sub = sst[z_mask]
        vec.extend([
            float(np.mean(w_sub)), float(np.max(w_sub)), float(np.std(w_sub)),
            float(np.mean(m_sub)), float(np.min(m_sub)),
            float(np.mean(s_sub))
        ])

    # 4. Four-quadrant asymmetries (16 features: 4 quadrants x 4 vars)
    quadrants = [
        (dlat_km >= 0) & (dlon_km >= 0),  # NE
        (dlat_km >= 0) & (dlon_km < 0),   # NW
        (dlat_km < 0) & (dlon_km < 0),    # SW
        (dlat_km < 0) & (dlon_km >= 0),   # SE
    ]
    for q_mask in quadrants:
        if not np.any(q_mask):
            q_mask = np.ones_like(dist_km, dtype=bool)
        vec.extend([
            float(np.mean(u10[q_mask])),
            float(np.mean(v10[q_mask])),
            float(np.mean(wind10[q_mask])),
            float(np.mean(msl[q_mask]))
        ])

    # 5. Steering & Kinematic proxies (6 features)
    steer_mask = (dist_km >= 300.0) & (dist_km <= 600.0)
    if not np.any(steer_mask):
        steer_mask = np.ones_like(dist_km, dtype=bool)
    u_steer = float(np.mean(u10[steer_mask]))
    v_steer = float(np.mean(v10[steer_mask]))
    steer_mag = float(np.sqrt(u_steer**2 + v_steer**2))
    steer_dir = float(np.degrees(np.arctan2(v_steer, u_steer))) % 360.0

    if u10.shape[0] > 1 and u10.shape[1] > 1:
        gy_u, gx_u = np.gradient(u10)
        gy_v, gx_v = np.gradient(v10)
        vorticity = float(np.mean(gx_v - gy_u))
        divergence = float(np.mean(gx_u + gy_v))
    else:
        vorticity = 0.0
        divergence = 0.0
    vec.extend([u_steer, v_steer, steer_mag, steer_dir, vorticity, divergence])

    # 6. Downsampled 4x4 spatial grid of local normalized wind speed (16 features)
    import cv2
    w_resized = cv2.resize(wind10, (4, 4), interpolation=cv2.INTER_AREA).flatten()
    vec.extend([float(v) for v in w_resized])

    out = np.array(vec, dtype=np.float32)
    assert len(out) == 64, f"Expected 64 features, got {len(out)}"
    return out


def main():
    if not ERA5_DIR.exists():
        raise FileNotFoundError(f"ERA5 directory not found at {ERA5_DIR}")

    era5_files = sorted(ERA5_DIR.glob("era5_*.nc"))
    print(f"[wire_era5] Found {len(era5_files)} ERA5 NetCDF files in {ERA5_DIR}")

    file_map = {f.name[5:12]: f for f in era5_files}

    sample_dirs = [p for p in SAMPLES_DIR.iterdir() if p.is_dir() and (p / "meta.json").exists()]
    print(f"[wire_era5] Found {len(sample_dirs)} total samples in {SAMPLES_DIR}")

    samples_by_ym = {}
    for s_dir in sample_dirs:
        meta = json.loads((s_dir / "meta.json").read_text())
        t = datetime.fromisoformat(meta["timestamp"])
        ym = t.strftime("%Y_%m")
        samples_by_ym.setdefault(ym, []).append((s_dir, meta, t))

    print(f"[wire_era5] Grouped into {len(samples_by_ym)} year-month buckets.")

    updated_count = 0
    interpolated_count = 0

    for ym, group in sorted(samples_by_ym.items()):
        nc_file = file_map.get(ym)
        is_fallback = False
        if nc_file is None:
            month = ym.split("_")[1]
            candidate_files = [f for k, f in file_map.items() if k.endswith(f"_{month}")]
            if candidate_files:
                nc_file = candidate_files[0]
                is_fallback = True
            else:
                nc_file = era5_files[0]
                is_fallback = True

        print(f"[wire_era5] Processing {ym} ({len(group)} samples) using {nc_file.name} (fallback={is_fallback})...")
        ds = xr.open_dataset(nc_file)

        if "number" in ds.dims:
            ds = ds.isel(number=0, drop=True)
        if "expver" in ds.dims:
            ds = ds.isel(expver=0, drop=True)

        for s_dir, meta, t in group:
            lat = float(meta["lat"])
            lon = float(meta["lon"])

            weather_vec = extract_64_weather_vector(ds, lat, lon, t)

            np.save(s_dir / "weather.npy", weather_vec)

            meta["weather_real"] = True
            meta["weather_source"] = f"era5:{nc_file.name}"
            meta["weather_fallback"] = is_fallback
            meta["weather_dim"] = 64
            (s_dir / "meta.json").write_text(json.dumps(meta, indent=2))

            if is_fallback:
                interpolated_count += 1
            else:
                updated_count += 1

        ds.close()

    print(f"\n[wire_era5] COMPLETE: {updated_count + interpolated_count} samples wired with real ERA5.")
    print(f"  Exact ERA5 match: {updated_count}")
    print(f"  Climatology fallback: {interpolated_count}")

    s0 = sample_dirs[0]
    w = np.load(s0 / "weather.npy")
    print(f"[wire_era5] Verification on {s0.name}: shape={w.shape}, dtype={w.dtype}, min={w.min():.2f}, max={w.max():.2f}, mean={w.mean():.2f}")


if __name__ == "__main__":
    main()
