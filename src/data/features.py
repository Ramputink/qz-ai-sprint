"""Toolbox de features de señal — utilidades reutilizables de diagnóstico.

Estas funciones vienen del trabajo paralelo en `origin/master` (commit 48b217b) y se
conservan aquí, separadas del pipeline de `preprocess.py`, porque son de otro nivel:
`preprocess.py` orquesta datasets completos, esto son primitivas sobre una señal.

`bearing_fault_freqs` es la pieza clave para rodamientos: si se conoce la geometría,
la energía en BPFO/BPFI/BSF/FTF discrimina mucho mejor que la curtosis genérica.
"""
from __future__ import annotations

import math
from pathlib import Path


def _np():
    import numpy as np
    return np


def time_features(window) -> dict[str, float]:
    np = _np()
    x = np.asarray(window, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    mean = float(np.mean(x))
    std = float(np.std(x) + 1e-12)
    return {
        "rms": rms,
        "peak": peak,
        "crest_factor": peak / (rms + 1e-12),
        "kurtosis": float(np.mean(((x - mean) / std) ** 4)),      # sube con impactos (fallo incipiente)
        "skewness": float(np.mean(((x - mean) / std) ** 3)),
        "p2p": float(np.max(x) - np.min(x)),
    }


def freq_features(window, fs: float, bands: int = 8) -> dict[str, float]:
    np = _np()
    x = np.asarray(window, dtype=np.float64)
    x = x - np.mean(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    out: dict[str, float] = {"spec_centroid": float(np.sum(freqs * spec) / (np.sum(spec) + 1e-12))}
    # energía por bandas logarítmicas
    edges = np.linspace(0, len(spec), bands + 1).astype(int)
    for i in range(bands):
        seg = spec[edges[i]:edges[i + 1]]
        out[f"band_{i}_energy"] = float(np.sum(seg ** 2))
    return out


def bearing_fault_freqs(rpm: float, n_balls: int, ball_dia: float, pitch_dia: float,
                        contact_angle_deg: float = 0.0) -> dict[str, float]:
    """Frecuencias características del rodamiento (Hz). Si se conoce la geometría,
    concentrar la energía en estas bandas es el mejor predictor de fallo."""
    fr = rpm / 60.0
    ratio = (ball_dia / pitch_dia) * math.cos(math.radians(contact_angle_deg))
    return {
        "BPFO": n_balls / 2 * fr * (1 - ratio),   # outer race
        "BPFI": n_balls / 2 * fr * (1 + ratio),   # inner race
        "BSF": pitch_dia / (2 * ball_dia) * fr * (1 - ratio ** 2),  # ball spin
        "FTF": fr / 2 * (1 - ratio),              # cage
    }


def window_signal(signal, win: int, hop: int):
    """Genera ventanas solapadas (para RUL y features)."""
    np = _np()
    x = np.asarray(signal)
    n = (len(x) - win) // hop + 1
    for i in range(max(0, n)):
        yield x[i * hop:i * hop + win]


def features_dataframe(signals: dict[str, list], fs: float, win: int, hop: int):
    """De {canal: señal} a un DataFrame de features por ventana. pandas perezoso."""
    import pandas as pd
    rows = []
    for ch, sig in signals.items():
        for j, w in enumerate(window_signal(sig, win, hop)):
            feat = {"channel": ch, "window": j}
            feat.update({f"t_{k}": v for k, v in time_features(w).items()})
            feat.update({f"f_{k}": v for k, v in freq_features(w, fs).items()})
            rows.append(feat)
    return pd.DataFrame(rows)


def save_parquet(df, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
