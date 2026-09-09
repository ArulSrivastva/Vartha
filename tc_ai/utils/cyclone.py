import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple


def get_imd_stage_class(wind_speed_kt: float) -> str:
    """Classify cyclone stage based on IMD/RSMC New Delhi classification for the North Indian Ocean.

    Wind speed in knots. IMD categories for the NIO:
      - Low Pressure Area: < 17 kt
      - Depression: 17-27 kt
      - Deep Depression: 28-33 kt
      - Cyclonic Storm: 34-47 kt
      - Severe Cyclonic Storm: 48-63 kt
      - Very Severe Cyclonic Storm: 64-89 kt
      - Extremely Severe Cyclonic Storm: 90-119 kt
    """
    if wind_speed_kt < 17:
        return "low_pressure"
    elif wind_speed_kt <= 27:
        return "depression"
    elif wind_speed_kt <= 33:
        return "deep_depression"
    elif wind_speed_kt <= 47:
        return "cyclonic_storm"
    elif wind_speed_kt <= 63:
        return "severe_cyclonic_storm"
    elif wind_speed_kt <= 89:
        return "very_severe_cyclonic_storm"
    else:
        return "extremely_severe_cyclonic_storm"


def get_imd_stage_index(wind_speed_kt: float) -> int:
    """Get numerical index for IMD stage (0-7)."""
    mapping = {
        "no_disturbance": 0,
        "low_pressure": 1,
        "depression": 2,
        "deep_depression": 3,
        "cyclonic_storm": 4,
        "severe_cyclonic_storm": 5,
        "very_severe_cyclonic_storm": 6,
        "extremely_severe_cyclonic_storm": 7,
    }
    return mapping[get_imd_stage_class(wind_speed_kt)]


def get_saffir_simpson_category(wind_kt: float) -> str:
    """Saffir-Simpson Hurricane Wind Scale (SSHWS) category."""
    if wind_kt < 34:
        return "tropical_depression"
    elif wind_kt <= 63:
        return "tropical_storm"
    elif wind_kt <= 82:
        return "category_1"
    elif wind_kt <= 95:
        return "category_2"
    elif wind_kt <= 112:
        return "category_3"
    elif wind_kt <= 136:
        return "category_4"
    else:
        return "category_5"


def classify_structure(stage: int, wind_change_24h: float, shear: float,
                       has_eye: bool = False, asymmetric: bool = False) -> List[str]:
    """Classify structural pattern (multi-label) based on available features.

    Parameters:
      stage: IMD stage index (0-7)
      wind_change_24h: change in wind speed over 24h (positive = intensifying)
      shear: vertical wind shear (m/s, higher = more sheared)
      has_eye: whether an eye is present
      asymmetric: whether structure is highly asymmetric

    Returns list of structural labels.
    """
    labels = []
    if wind_change_24h > 15:
        labels.append("rapidly_intensifying")
    elif wind_change_24h > 5:
        labels.append("intensifying")
    elif wind_change_24h < -5:
        labels.append("weakening")
    elif abs(wind_change_24h) <= 5:
        labels.append("mature")

    if shear > 15:
        labels.append("sheared")
    if has_eye:
        if stage <= 3:
            labels.append("eye_forming")
        else:
            labels.append("eye_present")
        labels.append("eyewall_organized")
    if asymmetric:
        labels.append("highly_asymmetric")

    if not labels:
        labels.append("developing")
    return labels


def get_pattern_multilabel(history: List[float], wind_now: float, shear: float) -> np.ndarray:
    """Encode temporal pattern as a multi-label binary vector.

    history: list of wind speeds at T-12h, T-9h, T-6h, T-3h, T0
    """
    n = len(history)
    if n < 2:
        return np.zeros(9)

    changes = [history[i] - history[i-1] for i in range(1, n)]
    total_change_24h = history[-1] - history[0]
    recent_change = history[-1] - history[-2]

    labels = [0] * 9
    # 0: developing, 1: intensifying, 2: rapid_intensification, 3: mature
    # 4: weakening, 5: sheared, 6: recurring, 7: land_interaction, 8: dissipating

    if total_change_24h > 30 and any(c > 10 for c in changes):
        labels[2] = 1  # rapid intensification
    elif total_change_24h > 5 or recent_change > 0:
        labels[1] = 1  # intensifying
    elif total_change_24h < -15:
        labels[8] = 1  # dissipating
    elif total_change_24h < -3:
        labels[4] = 1  # weakening
    elif abs(total_change_24h) <= 5:
        labels[3] = 1  # mature

    if shear > 15:
        labels[5] = 1  # sheared

    if not any(labels[0:5]) and not any(labels[6:9]):
        labels[0] = 1  # developing
    return np.array(labels, dtype=np.float32)


def compute_wind_shear(u200, v200, u850, v850) -> float:
    """Compute magnitude of 200-850 hPa vertical wind shear (m/s)."""
    du = u200 - u850
    dv = v200 - v850
    return float(np.sqrt(du**2 + dv**2))


def estimate_ri_probability(wind, wind_history, sst, sst_change, shear, tchp, mslp) -> float:
    """Heuristic estimation of Rapid Intensification probability (0-1).

    Based on operational criteria (Kaplan & DeMaria RI index tradition).
    """
    score = 0.0
    weights = 0.0

    # Inner core intensity (current wind)
    if wind > 45:  # kt
        inner = min((wind - 45) / 30, 1.0) * 0.25
    else:
        inner = 0.0
    score += inner
    weights += 0.25

    # SST warm enough
    sst_term = max(min((sst - 26.5) / 3.5, 1.0), 0.0) * 0.25
    score += sst_term
    weights += 0.25

    # SST change
    sstc = max(min((sst_change + 0.5) / 2.0, 1.0), 0.0) * 0.1
    score += sstc
    weights += 0.1

    # Low shear
    shear_term = max(min((20 - shear) / 15, 1.0), 0.0) * 0.25
    score += shear_term
    weights += 0.25

    # Low TCHP -> here use as proxy
    tchp_term = max(min((tchp - 20) / 60, 1.0), 0.0) * 0.1
    score += tchp_term
    weights += 0.1

    # Recent pressure fall
    mslp_term = max(min((1005 - mslp) / 30, 1.0), 0.0) * 0.05
    score += mslp_term
    weights += 0.05

    return min(score / (weights + 1e-6), 1.0)
