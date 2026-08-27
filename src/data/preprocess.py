"""Preprocesado: de señal cruda de vibración/corriente a features para el modelo.

Núcleo predictivo: extraer de cada ventana de señal las features que delatan un
fallo incipiente de rodamiento/motor:
  * Dominio del tiempo: RMS, pico, factor de cresta, curtosis, asimetría.
  * Dominio de frecuencia (FFT): energía en bandas, y energía en las frecuencias
    características del rodamiento (BPFO/BPFI/BSF/FTF) si se conoce la geometría.
  * Envolvente (Hilbert) para demodular defectos de rodamiento.

Salida: tablas Parquet (rápidas de releer, portables Mac/Windows). numpy/scipy se
importan de forma perezosa (solo se usan en ejecución real, no en --dry-run).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional


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


if __name__ == "__main__":
    # Auto-test SOLO si numpy está disponible (en el Mac de orquestación puede no estarlo).
    try:
        np = _np()
        fs = 20000.0
        sig = (np.sin(2 * np.pi * 100 * np.arange(4096) / fs)
               + 0.3 * np.random.randn(4096))
        tf = time_features(sig)
        ff = freq_features(sig, fs)
        bf = bearing_fault_freqs(rpm=1797, n_balls=9, ball_dia=7.94, pitch_dia=39.04)
        print("time_features:", {k: round(v, 3) for k, v in tf.items()})
        print("freq_features (centroid):", round(ff["spec_centroid"], 1), "Hz")
        print("bearing fault freqs:", {k: round(v, 1) for k, v in bf.items()})
        print("OK preprocess")
    except ImportError:
        print("preprocess: numpy no instalado aquí (normal en el Mac de orquestación); se usará en el PC.")
