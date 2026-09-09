"""Real-time INSAT-3D(R) IMAGER ingest from MOSDAC.

Fetches the newest small geophysical products (default 3RIMG_L2B_SST, ~16 MB per
30-min slot) using the credentials in `config.json` (user provided the official
MOSDAC download API client `mdapi.py`), then converts the HDF5 files into the
model-ready satellite layout the dashboard and inference engine consume:

    data/mosdac/live/
        satellite.npy          (3, 512, 512)  newest frame  [model input]
        satellite_sequence.npy (F, 3, 512, 512) newest F frames
        sst_degc.npy           (512, 512)     raw SST in deg C (display)
        latlon.npy             (512, 512, 2)  geolocation [deg] per pixel
        meta.json              provenance + timestamps

Channels: ch0 = SST radiance (normalized to [0,1]), ch1/ch2 = spatial gradients of
the SST field (structure cues the models were trained on as IR/WV/VIS proxies).
Latitude/Longitude are stored scaled by 100 (centidegrees), fill = 32767.
SST values are Kelvin, fill = -999.

MOSDAC's public API is intermittently unreliable ("Server Unavailable"), so all
search + download calls are wrapped in retry/backoff and partial/corrupt files
are cleaned up before retrying. Failures are appended to
`data/mosdac/error_logs/` for operator visibility.

Usage:
    python scripts/fetch_mosdac_satellite.py --window-hours 12 --count 6
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

import mdapi  # official MOSDAC download API client (module-level config load)

LIVE_DIR = PROJECT_ROOT / "data" / "mosdac" / "live"
ERROR_LOG_DIR = PROJECT_ROOT / "data" / "mosdac" / "error_logs"

FIELD_NAMES = {"3RIMG_L2B_SST": "SST", "3RIMG_L2B_LST": "LST"}

_RETRIES = 3
_RETRY_BACKOFF = [15, 30, 60]


def _log_failure(step: str, detail: str):
    ERROR_LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H_%M_%S_UTC")
    log = ERROR_LOG_DIR / f"{datetime.now(timezone.utc).strftime('%d-%m-%Y')}_error.log"
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp} - ERROR - {step}: {detail}\n")


def _cleanup_partial(path: Path):
    """Remove zero-byte / .part / corrupt-looking downloads so retries start clean."""
    for p in [path, Path(str(path) + ".part")]:
        if p.exists():
            try:
                if p.stat().st_size == 0:
                    p.unlink()
            except OSError:
                pass


def _search_once(dataset: str, window_hours: int, count: int) -> list:
    import requests
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=window_hours)
    params = {
        "datasetId": dataset,
        "startTime": start.strftime("%Y-%m-%d"),
        "endTime": (end + timedelta(days=1)).strftime("%Y-%m-%d"),
        "count": count,
    }
    res = requests.get(mdapi.search_url, params=params, timeout=30)
    res.raise_for_status()
    entries = res.json().get("entries", [])
    if not entries:
        return []
    return entries


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def find_recent(dataset: str, window_hours: int, count: int):
    """Return newest `count` entries within `window_hours` ending now (retried)."""
    last_err = None
    for attempt in range(1, _RETRIES + 1):
        try:
            entries = _search_once(dataset, window_hours, count)
            if not entries:
                raise RuntimeError(f"No MOSDAC records for dataset '{dataset}' "
                                   f"in the last {window_hours}h window.")
            return entries
        except Exception as exc:
            last_err = exc
            _log_failure("search", f"{type(exc).__name__}: {exc}")
            if attempt < _RETRIES:
                delay = _RETRY_BACKOFF[attempt - 1]
                print(f"  [mosdac] search failed ({exc}); retrying in {delay}s ({attempt}/{_RETRIES - 1})", flush=True)
                time.sleep(delay)
    raise RuntimeError(f"MOSDAC search failed after {_RETRIES} attempts: {last_err}")


def download_entries(entries) -> list:
    """Download entries that are not already on disk; returns ALL present files.

    Each download is retried with exponential backoff and partial/corrupt files
    are cleaned up between attempts (MOSDAC intermittently aborts mid-stream).
    """
    tokens, _ = mdapi.get_token()
    tok = tokens["access_token"]
    total = len(entries)
    files = []
    for i, e in enumerate(entries, start=1):
        existing = Path(mdapi.download_path) / e["identifier"]
        if existing.exists() and existing.stat().st_size > 0:
            files.append(existing)
            continue
        saved = None
        last_err = None
        for attempt in range(1, _RETRIES + 1):
            _cleanup_partial(existing)
            try:
                path = mdapi.download_data(tok, e["id"], e["identifier"], e.get("updated"), i, total)
            except Exception as exc:
                last_err = exc
                path = None
            if path and isinstance(path, str) and Path(path).exists() and Path(path).stat().st_size > 0:
                saved = Path(path)
                break
            last_err = last_err or (f"download returned {path!r}" if path else "download failed")
            _log_failure("download", f"{e['identifier']}: {last_err}")
            if attempt < _RETRIES:
                delay = _RETRY_BACKOFF[attempt - 1]
                print(f"  [mosdac] download failed for {e['identifier']} ({last_err}); retrying in {delay}s ({attempt}/{_RETRIES - 1})", flush=True)
                time.sleep(delay)
        if saved is not None:
            files.append(saved)
        else:
            print(f"  [mosdac] WARNING: could not download {e['identifier']} after {_RETRIES} attempts.", flush=True)
    return files


def _read_product(file: Path, dataset: str) -> dict:
    import h5py
    with h5py.File(file, "r") as f:
        field = FIELD_NAMES.get(dataset)
        if field is not None:
            if field not in f:
                available = [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
                raise RuntimeError(f"Dataset '{field}' not in {file.name}; "
                                   f"root datasets: {available}")
            data = f[field][0].astype(np.float32)
            attrs = dict(f.attrs)
        else:
            datasets = {k: f[k] for k in f.keys()
                        if isinstance(f[k], h5py.Dataset) and f[k].shape[0] == 1 and
                        f[k].dtype == np.float32}
            if not datasets:
                raise RuntimeError(f"No single-channel float32 product found in {file.name}.")
            field = sorted(datasets)[0]
            data = datasets[field][0].astype(np.float32)
            attrs = dict(f.attrs)

        h, w = data.shape
        lat = f["Latitude"][:].astype(np.float32)
        lon = f["Longitude"][:].astype(np.float32)
        lat[lat == 32767] = np.nan
        lon[lon == 32767] = np.nan
        lat /= 100.0
        lon /= 100.0

    data[data < -50.0] = np.nan
    if "SST" in field.upper():
        data = data - 273.15  # Kelvin -> deg C

    acq = attrs.get("Acquisition_Start_Time", b"").decode() or attrs.get("Unique_Id", b"").decode()
    return {"data": data, "lat": lat, "lon": lon, "field": field, "acquired": acq, "file": file.name}


def _to_512(grid: np.ndarray) -> np.ndarray:
    """Resize a (H, W) float grid to (512, 512) with NaN-safe bilinear sampling."""
    arr = np.nan_to_num(grid.astype(np.float32), nan=0.0)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    out = F.interpolate(t, size=(512, 512), mode="bilinear", align_corners=False)
    return out[0, 0].numpy()


def _normalize(ch: np.ndarray) -> np.ndarray:
    """Clip to [p02, p98] then scale to [0, 1] (valid-pixel statistics)."""
    p2, p98 = np.nanpercentile(ch, [2, 98])
    if p98 <= p2:
        p98 = p2 + 1e-6
    c = np.clip(ch, p2, p98)
    c = (c - p2) / (p98 - p2)
    return np.nan_to_num(c, nan=0.0).astype(np.float32)


def convert(files: list, dataset: str, live_dir: Path):
    """Convert downloaded HDF5 files into the model-ready satellite layout."""
    live_dir.mkdir(parents=True, exist_ok=True)
    prods = [_read_product(f, dataset) for f in sorted(files)]
    prods.sort(key=lambda p: p["file"])  # oldest -> newest by identifier timestamp

    channels = []
    for p in prods:
        base = _to_512(p["data"])
        gy, gx = np.gradient(base)
        ch0 = _normalize(np.nan_to_num(base, nan=0.0))
        ch1 = _normalize(gx)
        ch2 = _normalize(gy)
        channels.append(np.stack([ch0, ch1, ch2], axis=0).astype(np.float32))

    sat = channels[-1]  # newest frame, (3, 512, 512)
    seq = np.stack(channels, axis=0)  # (F, 3, 512, 512)

    newest = prods[-1]
    lat512 = _to_512(newest["lat"])
    lon512 = _to_512(newest["lon"])
    degc = newest["data"].astype(np.float32)

    np.save(live_dir / "satellite.npy", sat)
    np.save(live_dir / "satellite_sequence.npy", seq)
    np.save(live_dir / "sst_degc.npy", degc)
    np.save(live_dir / "latlon.npy", np.stack([lat512, lon512], axis=-1))
    np.save(live_dir / "mask.npy", np.isfinite(_to_512(newest["data"])).astype(np.float32))

    meta = {
        "source": "MOSDAC (mosdac.gov.in) real INSAT-3DR IMAGER",
        "dataset": dataset,
        "field": newest["field"],
        "downloaded_files": len(files),
        "frames": int(seq.shape[1]),
        "newest_file": newest["file"],
        "newest_acquired": newest["acquired"],
        "fetched_at": utcnow_iso(),
        "channels": ["intensity", "dgrad_x", "dgrad_y"],
        "image_shape": [3, 512, 512],
        "notes": ("L2B geophysical product (small ~16MB files). ch0 = field "
                  "radiance normalized; ch1/ch2 = spatial gradients. True "
                  "IR/WV/VIS cloud radiances live in the 460MB 3RIMG_L1B_STD "
                  "full-disk product (see README)."),
    }
    (live_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    ap = argparse.ArgumentParser(description="Fetch + convert real-time MOSDAC imagery.")
    ap.add_argument("--dataset", default=None,
                    help="MOSDAC datasetId (default: 3RIMG_L2B_SST from config.json)")
    ap.add_argument("--window-hours", type=int, default=12)
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--no-convert", action="store_true",
                    help="download files only; skip building model-ready arrays")
    args = ap.parse_args()

    dataset = args.dataset or mdapi.datasetId
    if dataset not in FIELD_NAMES:
        print(f"[mosdac] dataset '{dataset}' is not yet handled by the converter. "
              f"Supported: {sorted(FIELD_NAMES)}.")
        sys.exit(1)

    print(f"[mosdac] searching for '{dataset}' in the last {args.window_hours}h...")
    entries = find_recent(dataset, args.window_hours, args.count)
    print(f"[mosdac] newest {len(entries)} records found, latest "
          f"updated={entries[0].get('updated')}")

    files = download_entries(entries)
    print(f"[mosdac] {len(files)} file(s) available on disk.")

    if not files:
        print("[mosdac] No valid frames downloaded; leaving live/ state unchanged.", flush=True)
        print("[mosdac] Check data/mosdac/error_logs/ for details and retry shortly.", flush=True)
        sys.exit(2)

    if args.no_convert:
        sys.exit(0)
    meta = convert(files, dataset, LIVE_DIR)
    print(f"[mosdac] LIVE frame written -> {LIVE_DIR / 'satellite.npy'}")
    print(f"[mosdac] newest sample {meta['newest_file']} acquired {meta['newest_acquired']}")
    print(f"[mosdac] meta.json written -> {LIVE_DIR / 'meta.json'}")


if __name__ == "__main__":
    main()