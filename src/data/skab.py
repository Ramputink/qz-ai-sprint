"""Loader de SKAB (Skoltech Anomaly Benchmark) — testbed de bomba/motor.

Cada CSV (separador ';') tiene 8 sensores (acelerómetros, corriente, presión,
temperatura, termopar, voltaje, caudal) + etiquetas `anomaly` y `changepoint`.
Carpetas: anomaly-free (normal), valve1/valve2/other (experimentos con fallo).

Construimos features por ventana deslizante (estadísticos rolling que delatan la
degradación) y devolvemos X, y listos para entrenar. numpy/pandas perezosos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

SENSORS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure",
           "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]


def _find_root(data_dir: Path) -> Path:
    for cand in (data_dir / "SKAB-master" / "data", data_dir / "skab" / "SKAB-master" / "data",
                 data_dir / "data", data_dir):
        if cand.exists() and any(cand.glob("**/*.csv")):
            return cand
    raise FileNotFoundError(f"No encuentro los CSV de SKAB bajo {data_dir}. Descárgalo primero.")


def _window_features(df, win: int, hop: int):
    """De un DataFrame (una serie de sensores) a filas de features por ventana."""
    import numpy as np
    import pandas as pd
    X, y = [], []
    vals = df[SENSORS].to_numpy(dtype=np.float64)
    lab = df["anomaly"].to_numpy(dtype=np.float64)
    n = (len(vals) - win) // hop + 1
    for i in range(max(0, n)):
        w = vals[i * hop:i * hop + win]
        feat = []
        for c in range(w.shape[1]):
            col = w[:, c]
            feat += [col.mean(), col.std(), col.min(), col.max(),
                     np.abs(col).max(), (col[-1] - col[0])]  # tendencia intra-ventana
        X.append(feat)
        y.append(1.0 if lab[i * hop:i * hop + win].max() > 0 else 0.0)
    return X, y


def load(data_dir: str | Path, win: int = 60, hop: int = 20) -> dict[str, Any]:
    """Carga SKAB completo. Devuelve X (features), y (0 sano / 1 fallo), y máscara
    de 'normal' (para entrenar el autoencoder solo con datos sanos)."""
    import numpy as np
    import pandas as pd
    root = _find_root(Path(data_dir))
    files = sorted(root.glob("**/*.csv"))
    allX, ally, groups = [], [], []
    feat_names = [f"{s}_{stat}" for s in SENSORS
                  for stat in ("mean", "std", "min", "max", "absmax", "trend")]
    for f in files:
        try:
            df = pd.read_csv(f, sep=";")
        except Exception:
            continue
        if "anomaly" not in df.columns or not set(SENSORS).issubset(df.columns):
            continue
        X, y = _window_features(df, win, hop)
        allX.extend(X); ally.extend(y); groups.extend([f.parent.name] * len(y))
    X = np.asarray(allX, dtype=np.float64)
    y = np.asarray(ally, dtype=np.float64)
    if len(X) == 0:
        raise RuntimeError("SKAB: 0 ventanas cargadas (revisa la descarga).")
    # normalización (media/std por feature, calculada sobre los sanos)
    normal_mask = y == 0
    mu = X[normal_mask].mean(axis=0)
    sd = X[normal_mask].std(axis=0) + 1e-9
    Xn = (X - mu) / sd
    return {"X": Xn, "y": y, "normal_mask": normal_mask, "feat_names": feat_names,
            "n_features": Xn.shape[1], "n_samples": len(y),
            "n_normal": int(normal_mask.sum()), "n_fault": int((y == 1).sum()),
            "norm": {"mu": mu.tolist(), "sd": sd.tolist()}}


if __name__ == "__main__":
    import sys
    d = load(sys.argv[1] if len(sys.argv) > 1 else "data")
    print(f"SKAB cargado: {d['n_samples']} ventanas, {d['n_features']} features")
    print(f"  sanos: {d['n_normal']}  ·  con fallo: {d['n_fault']}")
