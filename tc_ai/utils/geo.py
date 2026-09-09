import numpy as np
import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km (Haversine formula)."""
    R = 6371.0
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def direct_positional_error(pred_lat, pred_lon, true_lat, true_lon) -> float:
    """DPE (km) between predicted and true position."""
    return haversine_distance(pred_lat, pred_lon, true_lat, true_lon)


def vectorized_dpe(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Vectorized DPE.
    pred, true: arrays of shape (N, 2) with columns [lat, lon].
    Returns array of shape (N,) with distances in km.
    """
    R = 6371.0
    lat1 = np.radians(pred[:, 0])
    lon1 = np.radians(pred[:, 1])
    lat2 = np.radians(true[:, 0])
    lon2 = np.radians(true[:, 1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def bearing_angle(lat1, lon1, lat2, lon2) -> float:
    """Initial bearing (degrees) from point 1 to point 2."""
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlon_r = lon2_r - lon1_r
    x = math.sin(dlon_r) * math.cos(lat2_r)
    y = math.cos(lat1_r) * math.sin(lat2_r) - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(dlon_r)
    bearing = math.degrees(math.atan2(x, y))
    return (bearing + 360) % 360


def cross_track_error(pred_lat, pred_lon, start_lat, start_lon, end_lat, end_lon) -> float:
    """Cross-track distance (km) of pred_from great circle from start to end."""
    R = 6371.0
    d_13 = haversine_distance(start_lat, start_lon, pred_lat, pred_lon)
    d_12 = haversine_distance(start_lat, start_lon, end_lat, end_lon)

    # Angular distances
    lat1_r = math.radians(start_lat)
    lon1_r = math.radians(start_lon)
    lat2_r = math.radians(end_lat)
    lon2_r = math.radians(end_lon)
    lat3_r = math.radians(pred_lat)
    lon3_r = math.radians(pred_lon)

    a = math.sin(lat1_r) * math.sin(lat2_r) + math.cos(lat1_r) * math.cos(lat2_r) * math.cos(lon2_r - lon1_r)
    a = max(-1, min(1, a))
    delta_12 = math.acos(a)

    b = math.sin(lat1_r) * math.sin(lat3_r) + math.cos(lat1_r) * math.cos(lat3_r) * math.cos(lon3_r - lon1_r)
    b = max(-1, min(1, b))
    delta_13 = math.acos(b)

    if delta_12 == 0:
        return 0.0

    # Bearings
    theta_12 = bearing_angle(start_lat, start_lon, end_lat, end_lon)
    theta_13 = bearing_angle(start_lat, start_lon, pred_lat, pred_lon)
    theta_13 = math.radians(theta_13)
    theta_12 = math.radians(theta_12)

    xtd_angle = math.asin(max(-1, min(1, math.sin(delta_13) * math.sin(theta_13 - theta_12))))
    return abs(xtd_angle) * R


def along_track_error(pred_lat, pred_lon, start_lat, start_lon, end_lat, end_lon) -> float:
    """Along-track distance (km) of pred from start along the start->end path."""
    R = 6371.0
    d_13 = haversine_distance(start_lat, start_lon, pred_lat, pred_lon)
    d_12 = haversine_distance(start_lat, start_lon, end_lat, end_lon)

    lat1_r = math.radians(start_lat)
    lon1_r = math.radians(start_lon)
    lat2_r = math.radians(end_lat)
    lon2_r = math.radians(end_lon)
    lat3_r = math.radians(pred_lat)
    lon3_r = math.radians(pred_lon)

    a = math.sin(lat1_r) * math.sin(lat2_r) + math.cos(lat1_r) * math.cos(lat2_r) * math.cos(lon2_r - lon1_r)
    a = max(-1, min(1, a))
    delta_12 = math.acos(a)

    b = math.sin(lat1_r) * math.sin(lat3_r) + math.cos(lat1_r) * math.cos(lat3_r) * math.cos(lon3_r - lon1_r)
    b = max(-1, min(1, b))
    delta_13 = math.acos(b)

    theta_12 = bearing_angle(start_lat, start_lon, end_lat, end_lon)
    theta_13 = bearing_angle(start_lat, start_lon, pred_lat, pred_lon)

    xtd_angle = math.asin(max(-1, min(1, math.sin(delta_13) * math.sin(math.radians(theta_13) - math.radians(theta_12)))))
    atd_angle = math.acos(max(-1, min(1, math.cos(delta_13) / math.cos(xtd_angle))))
    return atd_angle * R


def direction_error(pred_bearing_deg: float, true_bearing_deg: float) -> float:
    """Absolute angular difference (degrees)."""
    diff = abs(pred_bearing_deg - true_bearing_deg) % 360
    return min(diff, 360 - diff)


def convert_km_to_degrees_deg(x_km, y_km, lat):
    """Approximate conversion of km offset to degrees."""
    deg_per_km_lat = 1 / 110.574
    deg_per_km_lon = 1 / (111.320 * math.cos(math.radians(lat)))
    return x_km * deg_per_km_lon, y_km * deg_per_km_lat
