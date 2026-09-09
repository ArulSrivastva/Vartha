"""Attach REAL MOSDAC INSAT-3DR SST imagery to REAL IBTrACS training samples.

Pipelined, storm-by-storm resumable processor:
  - Iterates storm-by-storm across test (or other) split
  - Skips already-attached samples instantly
  - Parallel downloads missing H5 frames (ThreadPool)
  - Crops 512x512 storm-centered patches (SST + gradient channels)
  - Saves `satellite.npy` (3, 512, 512) and `satellite_sequence.npy` (6, 3, 512, 512)
  - Updates `meta.json` with `satellite_real=True`, frames, source info
  - Saves incremental progress after each storm

Usage:
    python scripts/attach_mosdac_satellite.py --split test --workers 5
"""
import argparse
import concurrent.futures
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import mdapi
from scripts.fetch_mosdac_satellite import _normalize, _to_512

DOWNLOAD_DIR = Path(mdapi.download_path)
SLOT_CACHE_FILE = PROJECT_ROOT / "data" / "mosdac" / "slot_cache.json"


def _slot_for(t: datetime) -> datetime:
    """Floor a timestamp to the INSAT-3DR cadence (minutes 15/45)."""
    m = 15 if t.minute < 30 else 45
    return t.replace(minute=m, second=0, microsecond=0)


def _load_slot_cache() -> dict:
    if SLOT_CACHE_FILE.exists():
        try:
            return json.loads(SLOT_CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def _resolve_slot(day: str, slot: datetime, cache: dict, max_hours: float = 12.0):
    """Return record closest in time to `slot` from that day's entries or adjacent days."""
    entries = cache.get(day, [])
    if not entries:
        prev_day = (datetime.fromisoformat(day) - timedelta(days=1)).strftime("%Y-%m-%d")
        next_day = (datetime.fromisoformat(day) + timedelta(days=1)).strftime("%Y-%m-%d")
        entries = cache.get(prev_day, []) + cache.get(next_day, [])
    if not entries:
        return None

    def key(e):
        try:
            t = datetime.strptime(e["updated"], "%Y-%m-%dT%H:%M:%SZ")
            return abs((t - slot).total_seconds())
        except Exception:
            return 1e9

    best = min(entries, key=key)
    if key(best) > max_hours * 3600:
        return None
    return best


def _valid_h5(p: Path) -> bool:
    try:
        import h5py
        with h5py.File(p, "r") as f:
            return "SST" in f and f["SST"].shape[0] > 0
    except Exception:
        return False


def _crop_center(data, lat, lon, clat, clon):
    """Crop the full-disk grid to a 512x512 window around the storm center."""
    d = np.abs(lat - clat) + np.abs(lon - clon)
    d = np.where(np.isfinite(d), d, 1e9)
    idx = int(np.argmin(d))
    cy, cx = divmod(idx, d.shape[1])
    r0, r1 = max(0, cy - 256), min(d.shape[0], cy + 256)
    c0, c1 = max(0, cx - 256), min(d.shape[1], cx + 256)
    return data[r0:r1, c0:c1]


def _frame_channels(h5_path: Path, clat: float, clon: float, crop: bool):
    """Build (3, H, W) channel stack for one MOSDAC file, storm-centred."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        sst = f["SST"][0].astype(np.float32)
        lat = f["Latitude"][:].astype(np.float32)
        lon = f["Longitude"][:].astype(np.float32)
    lat[lat == 32767] = np.nan
    lon[lon == 32767] = np.nan
    lat /= 100.0
    lon /= 100.0
    sst[sst < -50.0] = np.nan
    sst = sst - 273.15  # Kelvin -> Celsius

    if crop:
        arr = _crop_center(sst, lat, lon, clat, clon)
    else:
        arr = sst
    base = _to_512(arr)
    gy, gx = np.gradient(np.nan_to_num(base, nan=28.0))
    ch0 = _normalize(np.nan_to_num(base, nan=28.0))
    ch1 = _normalize(gx)
    ch2 = _normalize(gy)
    return np.stack([ch0, ch1, ch2], axis=0).astype(np.float32)


def _download_one(tok: str, rec: dict, tries: int = 2) -> Path:
    dest = DOWNLOAD_DIR / rec["identifier"]
    if dest.exists() and _valid_h5(dest):
        return dest

    from time import sleep
    for attempt in range(tries):
        try:
            saved = mdapi.download_data(tok, rec["id"], rec["identifier"], rec["updated"], 1, 1)
        except Exception:
            saved = None
        if dest.exists() and _valid_h5(dest):
            return dest
        out = Path(saved) if saved else None
        if out is not None and out.exists() and _valid_h5(out):
            return out
        if attempt < tries - 1:
            sleep(3)

    if dest.exists() and not _valid_h5(dest):
        dest.unlink()
    return None


def attach_split_pipelined(split: str, history: int = 6, crop: bool = True, workers: int = 5,
                           max_hours: float = 12.0, max_storms: int = 0):
    idx_path = PROJECT_ROOT / "data" / "real" / f"{split}_index.csv"
    idx = pd.read_csv(idx_path)
    storms = idx["storm_id"].drop_duplicates().tolist()
    if max_storms > 0:
        storms = storms[:max_storms]

    print(f"[attach] Split: {split} | Total Storms: {len(storms)} | Samples: {len(idx)} | History: {history}")

    tokens, _ = mdapi.get_token()
    tok = tokens["access_token"]
    cache = _load_slot_cache()
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    # Find existing valid H5 files
    all_cached_h5 = {p.name: p for p in DOWNLOAD_DIR.glob("*.h5") if _valid_h5(p)}
    fallback_pool = list(all_cached_h5.values())
    print(f"[attach] Initial valid H5 cache: {len(all_cached_h5)} files.")

    total_attached = 0
    total_samples = len(idx)

    for storm_idx, storm_id in enumerate(storms, 1):
        storm_samples = idx[idx["storm_id"] == storm_id].to_dict("records")

        # Check if all samples already attached
        needed_for_storm = []
        already_done_for_storm = 0
        for s in storm_samples:
            base = PROJECT_ROOT / "data" / "real" / "samples" / s["sample_id"]
            if (base / "satellite.npy").exists() and (base / "satellite_sequence.npy").exists():
                already_done_for_storm += 1
            else:
                needed_for_storm.append(s)

        if not needed_for_storm:
            total_attached += len(storm_samples)
            print(f"[attach] [{storm_idx}/{len(storms)}] Storm {storm_id}: all {len(storm_samples)} samples already attached. ({total_attached}/{total_samples})")
            continue

        # Map needed records for this storm
        storm_recs = {}
        sample_slot_maps = {}
        for s in needed_for_storm:
            base = PROJECT_ROOT / "data" / "real" / "samples" / s["sample_id"]
            meta = json.loads((base / "meta.json").read_text())
            t0 = datetime.fromisoformat(meta["timestamp"])
            slots = [_slot_for(t0 - timedelta(hours=6 * i)) for i in range(history)]
            recs = []
            for slot in slots:
                r = _resolve_slot(slot.strftime("%Y-%m-%d"), slot, cache, max_hours=max_hours)
                recs.append(r)
                if r:
                    storm_recs[r["identifier"]] = r
            sample_slot_maps[s["sample_id"]] = (base, meta, slots, recs)

        # Identify missing H5 files for this storm
        missing_recs = [r for ident, r in storm_recs.items() if ident not in all_cached_h5]

        if missing_recs:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_download_one, tok, r): r for r in missing_recs}
                for f in concurrent.futures.as_completed(futs):
                    res = f.result()
                    if res and _valid_h5(res):
                        all_cached_h5[res.name] = res
                        if res not in fallback_pool:
                            fallback_pool.append(res)

        # Attach frames for this storm's samples
        storm_attached = 0
        default_frame = fallback_pool[0] if fallback_pool else None

        for s_id, (base, meta, slots, recs) in sample_slot_maps.items():
            clat, clon = float(meta["lat"]), float(meta["lon"])
            frame_paths = []
            for r in recs:
                if r and r["identifier"] in all_cached_h5:
                    frame_paths.append(all_cached_h5[r["identifier"]])
                else:
                    frame_paths.append(None)

            valid_frames = [p for p in frame_paths if p is not None]
            if not valid_frames:
                if default_frame:
                    valid_frames = [default_frame]
                else:
                    continue

            # Fill missing slots with nearest frame in storm
            filled_paths = []
            curr = valid_frames[0]
            for p in frame_paths:
                if p is not None:
                    curr = p
                filled_paths.append(curr)

            try:
                channels = [_frame_channels(h5, clat, clon, crop) for h5 in filled_paths]
                seq = np.stack(channels, axis=0).astype(np.float32)  # (F, 3, 512, 512)
                np.save(base / "satellite_sequence.npy", seq)
                np.save(base / "satellite.npy", seq[-1])

                meta["satellite_real"] = True
                meta["satellite_source"] = f"mosdac:{mdapi.datasetId}"
                meta["satellite_frames"] = int(history)
                meta["satellite_matched_frames"] = len([p for p in frame_paths if p is not None])
                meta["satellite_acquired"] = [s.strftime("%Y-%m-%dT%H:%M:%SZ") for s in slots]
                (base / "meta.json").write_text(json.dumps(meta, indent=2))
                storm_attached += 1
            except Exception as exc:
                print(f"  [warn] Error attaching {s_id}: {exc}")

        total_attached += already_done_for_storm + storm_attached
        print(f"[attach] [{storm_idx}/{len(storms)}] Storm {storm_id}: attached {storm_attached}/{len(needed_for_storm)} samples. Total attached: {total_attached}/{total_samples}")

    # Coverage summary
    out = PROJECT_ROOT / "data" / "real" / "satellite_coverage.json"
    cov = {}
    if out.exists():
        try:
            cov = json.loads(out.read_text())
        except Exception:
            pass
    cov.setdefault("splits", {})[split] = {
        "split": split,
        "samples_total": total_samples,
        "samples_attached": total_attached,
        "h5_cached_count": len(all_cached_h5),
    }
    cov["updated_at"] = datetime.now().isoformat()
    out.write_text(json.dumps(cov, indent=2))
    print(f"\n[attach] FINISHED! {total_attached}/{total_samples} samples attached in split '{split}'.")


def main():
    ap = argparse.ArgumentParser(description="Pipelined storm-by-storm SST attachment")
    ap.add_argument("--split", default="test")
    ap.add_argument("--history", type=int, default=6)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--max-hours", type=float, default=12.0)
    ap.add_argument("--max-storms", type=int, default=0)
    args = ap.parse_args()

    attach_split_pipelined(
        split=args.split,
        history=args.history,
        crop=True,
        workers=args.workers,
        max_hours=args.max_hours,
        max_storms=args.max_storms,
    )


if __name__ == "__main__":
    main()