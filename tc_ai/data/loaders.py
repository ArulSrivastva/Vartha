"""Data acquisition and loading for multi-source satellite, track, and reanalysis data."""
import os
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


class TrackDataLoader:
    """Loader for tropical cyclone best-track data (IMD/RSMC + IBTrACS)."""

    def __init__(self, ibtracs_path: Optional[str] = None):
        self.ibtracs_path = Path(ibtracs_path) if ibtracs_path else None
        self.cache: Dict[str, pd.DataFrame] = {}

    def load_ibtracs(self, path: Optional[str] = None) -> pd.DataFrame:
        """Load global IBTrACS track data (subset for North Indian Ocean)."""
        p = Path(path) if path else self.ibtracs_path
        if p is None or not p.exists():
            raise FileNotFoundError(f"IBTrACS data not found at {p}. "
                                    "Download from https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/")
        if str(p) in self.cache:
            return self.cache[str(p)]

        # heuristics for format detection
        if p.suffix == ".csv":
            df = pd.read_csv(p, low_memory=False)
        elif p.suffix in (".nc", ".grib", ".grib2"):
            import xarray as xr
            ds = xr.open_dataset(p)
            df = ds.to_dataframe().reset_index()
        else:
            raise ValueError(f"Unsupported format: {p.suffix}")

        self.cache[str(p)] = df
        return df

    def filter_north_indian_ocean(self, df: pd.DataFrame, lon_min=50, lon_max=100,
                                  lat_min=0, lat_max=30) -> pd.DataFrame:
        """Filter tracks to North Indian Ocean region."""
        for lon_col in ["LON", "Longitude", "lon"]:
            if lon_col in df.columns:
                return df[(df[lon_col] >= lon_min) & (df[lon_col] <= lon_max) &
                          (df["LAT"] >= lat_min) & (df["LAT"] <= lat_max)]
        raise ValueError("Cannot find longitude column")

    def get_cyclone_track(self, df: pd.DataFrame, storm_id: str) -> pd.DataFrame:
        """Extract single cyclone track."""
        for id_col in ["SID", "storm_id", "ID"]:
            if id_col in df.columns:
                return df[df[id_col] == storm_id].sort_values("ISO_TIME")
        raise ValueError(f"Cannot find storm id column")

    def load_insat_track(self, path: str) -> pd.DataFrame:
        """Load IMD/RSMC New Delhi best-track data for INSAT-visible storms."""
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["storm_id", "timestamp"])
        return df


class Era5Loader:
    """Loader for ERA5 reanalysis data (historical atmospheric state)."""

    def __init__(self, base_dir: str = "./data/era5"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def download_era5(self, variables: List[str], years: List[int], months: List[int],
                      area: Tuple[float, float, float, float]):
        """Download ERA5 data from CDS API.
        area: [North, West, South, East]
        """
        import cdsapi
        c = cdsapi.Client()
        for year in years:
            for month in months:
                filename = self.base_dir / f"era5_{year}_{month:02d}.grib"
                if filename.exists():
                    continue
                c.retrieve(
                    "reanalysis-era5-pressure-levels",
                    {
                        "product_type": "reanalysis",
                        "variable": variables,
                        "year": str(year),
                        "month": str(month),
                        "day": [f"{d:02d}" for d in range(1, 32)],
                        "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
                        "pressure_level": ["200", "500", "700", "850", "1000"],
                        "area": area,
                        "data_format": "grib",
                        "format": "grib",
                    },
                    str(filename),
                )
                print(f"Downloaded {filename}")

    def load_era5(self, variable: str, year: int, month: int) -> np.ndarray:
        """Load ERA5 variable for a month as numpy array (time, lat, lon)."""
        filename = self.base_dir / f"era5_{year}_{month:02d}.grib"
        if not filename.exists():
            raise FileNotFoundError(f"ERA5 file not found: {filename}")
        import xarray as xr
        ds = xr.open_dataset(filename, engine="cfgrib")
        data = ds[variable].values
        return data

    def load_era5_single_level(self, variable: str, year: int, month: int) -> np.ndarray:
        """Load single-level ERA5 variable (e.g., sst, t2m, msl)."""
        filename = self.base_dir / f"era5_single_{year}_{month:02d}.grib"
        if not filename.exists():
            raise FileNotFoundError(f"ERA5 single-level file not found: {filename}")
        import xarray as xr
        ds = xr.open_dataset(filename, engine="cfgrib")
        return ds[variable].values


class InsatLoader:
    """Loader for INSAT-3D/3DR satellite imagery (IR, WV, VIS)."""

    def __init__(self, base_dir: str, api_key: Optional[str] = None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key

    def load_image(self, timestamp: datetime, channel: str, product: str) -> np.ndarray:
        """Load a satellite image for a given timestamp and channel.

        channels: 'IR' (infrared), 'WV' (water vapor), 'VIS' (visible)
        """
        pattern = f"insat_{timestamp:%Y%m%d_%H%M}_{channel}_{product}.npy"
        path = self.base_dir / pattern
        if path.exists():
            return np.load(path)
        # Fallback: try common formats
        for ext in [".tif", ".tiff", ".png", ".nc"]:
            p = self.base_dir / pattern.replace(".npy", ext)
            if p.exists():
                if ext == ".nc":
                    import xarray as xr
                    return xr.open_dataset(p).to_array().values
                from PIL import Image
                return np.array(Image.open(p))
        raise FileNotFoundError(f"Satellite image not found for {pattern}")

    def download_from_mosdac(self, timestamp: datetime, channel: str) -> bool:
        """Download and cache imagery from MOSDAC (requires API credentials)."""
        if self.api_key is None:
            raise ValueError("MOSDAC API key required. Request access at mosdac.gov.in")
        base_url = "https://mosdac.gov.in/insat3d/data"
        # Simplified; real implementation requires MOSDAC API specifics
        url = f"{base_url}/{channel}/{timestamp:%Y/%m/%d}/INSAT3D_{timestamp:%Y%m%d_%H%M}_{channel}.tif"
        r = requests.get(url, timeout=60)
        if r.status_code == 200:
            out_path = self.base_dir / f"insat_{timestamp:%Y%m%d_%H%M}_{channel}.tif"
            out_path.write_bytes(r.content)
            return True
        return False


class NwpLoader:
    """Loader for NWP/AI forecast guidance (ECMWF AIFS/IFS)."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_forecast(self, init_time: datetime, lead_time_hours: int,
                      variables: List[str]) -> Dict[str, np.ndarray]:
        """Load NWP forecast fields for given initialization and lead time."""
        fields = {}
        filename = self.base_dir / f"nwp_{init_time:%Y%m%d_%H}_{lead_time:03d}h.grib"
        if not filename.exists():
            raise FileNotFoundError(f"NWP forecast not found: {filename}")
        import xarray as xr
        ds = xr.open_dataset(filename, engine="cfgrib")
        for var in variables:
            if var in ds:
                fields[var] = ds[var].values
        return fields


class OceanDataLoader:
    """Loader for oceanic variables (SST, ocean heat content)."""

    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def load_sst(self, timestamp: datetime) -> np.ndarray:
        """Load satellite-derived sea surface temperature."""
        filename = self.base_dir / f"sst_{timestamp:%Y%m%d}.nc"
        if not filename.exists():
            raise FileNotFoundError(f"SST data not found: {filename}")
        import xarray as xr
        ds = xr.open_dataset(filename)
        return ds["sst"].values

    def load_ohc(self, timestamp: datetime) -> np.ndarray:
        """Load ocean heat content (or TCHP)."""
        filename = self.base_dir / f"ohc_{timestamp:%Y%m%d}.nc"
        if not filename.exists():
            raise FileNotFoundError(f"OHC data not found: {filename}")
        import xarray as xr
        ds = xr.open_dataset(filename)
        return ds["ohc"].values
