"""DPE verification (plan section 21/23 - Module 12).

When the actual track position arrives at horizon +h, compare against the
model's forecast and compute DPE as well as hit-rate vs targets.
"""
import sqlite3
from contextlib import closing
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List

from ..utils.geo import direct_positional_error, bearing_angle, direction_error


class DPEVerifier:
    """Stores forecasts, matches verifying observations, computes DPE."""

    def __init__(self, db_path: str = "./data/dpe_verification.db",
                 targets: Optional[Dict[int, float]] = None):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.targets = targets or {6: 25.0, 12: 50.0, 24: 100.0}
        self._init_db()

    def _init_db(self):
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS forecasts (
                    forecast_id TEXT PRIMARY KEY,
                    cyclone_id TEXT,
                    init_time TEXT,
                    horizon INTEGER,
                    pred_lat REAL,
                    pred_lon REAL,
                    obs_lat REAL,
                    obs_lon REAL,
                    dpe_km REAL,
                    verified INTEGER DEFAULT 0,
                    created_at TEXT
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_forecast_cyc_hor "
                         "ON forecasts(cyclone_id, init_time, horizon)")
            conn.commit()

    def save_forecast(self, cyclone_id: str, init_time: datetime,
                      horizon_h: int, pred_lat: float, pred_lon: float) -> str:
        """Store a forecast point. Returns forecast_id."""
        import uuid
        forecast_id = str(uuid.uuid4())
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO forecasts "
                "(forecast_id, cyclone_id, init_time, horizon, pred_lat, pred_lon, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (forecast_id, cyclone_id, init_time.isoformat(), horizon_h,
                 pred_lat, pred_lon, datetime.utcnow().isoformat()),
            )
            conn.commit()
        return forecast_id

    def verify(self, cyclone_id: str, init_time: datetime, horizon_h: int,
               obs_lat: float, obs_lon: float) -> Optional[float]:
        """Verify a forecast against observed position. Returns DPE in km."""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            row = conn.execute(
                "SELECT forecast_id, pred_lat, pred_lon FROM forecasts "
                "WHERE cyclone_id=? AND init_time=? AND horizon=?",
                (cyclone_id, init_time.isoformat(), horizon_h),
            ).fetchone()
            if row is None:
                return None
            forecast_id, pred_lat, pred_lon = row
            dpe = direct_positional_error(pred_lat, pred_lon, obs_lat, obs_lon)
            conn.execute(
                "UPDATE forecasts SET obs_lat=?, obs_lon=?, dpe_km=?, verified=1 "
                "WHERE forecast_id=?",
                (obs_lat, obs_lon, dpe, forecast_id),
            )
            conn.commit()
        return dpe

    def verify_storm(self, cyclone_id: str, best_track: List[tuple]) -> Dict[int, float]:
        """Verify all outstanding forecasts for a storm against its best track.

        best_track: list of (datetime, lat, lon).
        """
        results = {}
        for (ts, lat, lon) in best_track:
            # Find forecasts initialized at ts - horizon
            for h in self.targets:
                init = ts - timedelta(hours=h)
                dpe = self.verify(cyclone_id, init, h, lat, lon)
                if dpe is not None:
                    results[h] = dpe
        return results

    def report(self, horizon_h: int) -> Dict[str, float]:
        """Summary statistics for a given horizon over verified forecasts."""
        with closing(sqlite3.connect(str(self.db_path))) as conn:
            rows = conn.execute(
                "SELECT dpe_km FROM forecasts WHERE horizon=? AND verified=1",
                (horizon_h,),
            ).fetchall()
        dpes = [r[0] for r in rows]
        if not dpes:
            return {}
        dpe_arr = np.array(dpes)
        target = self.targets.get(horizon_h)
        result = {
            "n": len(dpes),
            "mean_dpe_km": float(np.mean(dpe_arr)),
            "median_dpe_km": float(np.median(dpe_arr)),
            "p95_dpe_km": float(np.percentile(dpe_arr, 95)),
            "rmse_dpe_km": float(np.sqrt(np.mean(dpe_arr ** 2))),
        }
        if target:
            result["hit_rate"] = float(np.mean(dpe_arr <= target))
            result["target_km"] = target
        return result

    def all_reports(self) -> Dict[int, Dict[str, float]]:
        return {h: self.report(h) for h in self.targets}
