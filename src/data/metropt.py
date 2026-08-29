"""Loader de MetroPT-3 — run-to-failure REAL de un compresor (APU de metro).

15.169.480 muestras @ 1 Hz (feb–ago 2020), 7 sensores analógicos predictivos.
Fallos documentados (fuga de aire, alta severidad), fuente: descripción oficial:
  #1 2020-04-18 00:00 → 23:59
  #2 2020-05-29 23:30 → 2020-05-30 06:00
  #3 2020-06-05 10:00 → 2020-06-07 14:30
  #4 2020-07-15 14:30 → 19:00

Construimos:
  * RUL = minutos hasta el INICIO del próximo fallo (limitado a un horizonte).
  * label = 1 si la ventana cae en los `lead_days` previos a un fallo ('pre-fallo').
Lectura por chunks + resampleo a 1 min (media+std por minuto) para caber en RAM.
Ventana deslizante de features sobre el minuto → predice el fallo con antelación.
numpy/pandas perezosos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

ANALOG = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Motor_current", "Oil_temperature"]

FAILURES = [  # (inicio, fin) en UTC naive
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


def _find_csv(data_dir: Path) -> Path:
    for cand in data_dir.rglob("MetroPT3*.csv"):
        return cand
    for cand in data_dir.rglob("*.csv"):
        if "metro" in cand.name.lower():
            return cand
    raise FileNotFoundError(f"No encuentro el CSV de MetroPT-3 bajo {data_dir}")


def load(data_dir: str | Path, resample: str = "1min", win: int = 60, hop: int = 10,
         lead_days: float = 10.0, rul_horizon_min: float = 20160.0,
         progress=None) -> dict[str, Any]:
    """Carga MetroPT-3 → X (features), y (0 sano / 1 pre-fallo), rul (min), timestamps.
    rul_horizon_min por defecto = 14 días (más allá se satura)."""
    import numpy as np
    import pandas as pd

    csv = _find_csv(Path(data_dir))
    fails = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in FAILURES]
    onsets = [a for a, _ in fails]

    # --- lectura por chunks + resampleo a 1 min (media y std por minuto) ---
    cols = ["timestamp"] + ANALOG
    agg_chunks = []
    reader = pd.read_csv(csv, usecols=lambda c: c in cols or c == "timestamp",
                         parse_dates=["timestamp"], chunksize=1_000_000)
    total = 0
    for i, ch in enumerate(reader):
        ch = ch[["timestamp"] + [c for c in ANALOG if c in ch.columns]].dropna()
        ch = ch.set_index("timestamp")
        r = ch.resample(resample).agg(["mean", "std"])
        r.columns = [f"{a}_{s}" for a, s in r.columns]
        agg_chunks.append(r)
        total += len(ch)
        if progress:
            progress("resample", i, total)
    df = pd.concat(agg_chunks).groupby(level=0).mean().sort_index()
    df = df.interpolate().dropna()
    # --- features de DEGRADACIÓN: desviación vs baseline lento (24 h) ---
    # capta el "alejarse de lo normal" que precede al fallo (mejor lead-time, menos ruido).
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    base24 = df[mean_cols].rolling(1440, min_periods=120).mean()
    ewm = df[mean_cols].ewm(span=720, min_periods=120).mean()  # tendencia suave
    for c in mean_cols:
        df[c + "_dev24h"] = df[c] - base24[c]      # desviación del baseline diario
        df[c + "_ewmdev"] = df[c] - ewm[c]         # desviación de la tendencia
    df = df.dropna()
    feat_cols = list(df.columns)

    # --- etiquetas por minuto ---
    idx = df.index
    # RUL: minutos hasta el próximo onset de fallo (o horizonte si no hay más)
    rul = np.full(len(idx), rul_horizon_min, dtype=np.float64)
    for k, ts in enumerate(idx):
        future = [(o - ts).total_seconds() / 60.0 for o in onsets if o >= ts]
        if future:
            rul[k] = min(min(future), rul_horizon_min)
    # label pre-fallo: dentro de lead_days antes de un onset (y no dentro del propio fallo)
    lead_min = lead_days * 24 * 60
    label = ((rul <= lead_min) & (rul > 0)).astype(np.float64)

    # --- ventanas deslizantes de features + tendencia ---
    V = df[feat_cols].to_numpy(dtype=np.float64)
    X, y, ruls, tstamps = [], [], [], []
    n = (len(V) - win) // hop + 1
    for w in range(max(0, n)):
        s = w * hop
        block = V[s:s + win]
        feat = []
        for c in range(block.shape[1]):
            col = block[:, c]
            feat += [col.mean(), col.std(), col.min(), col.max(), col[-1] - col[0]]
        X.append(feat)
        y.append(1.0 if label[s:s + win].max() > 0 else 0.0)
        ruls.append(float(rul[s + win - 1]))       # RUL al final de la ventana
        tstamps.append(idx[s + win - 1])
        if progress and w % 2000 == 0:
            progress("window", w, n)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ruls = np.asarray(ruls, dtype=np.float64)
    # normalización con estadísticas de los sanos
    normal = y == 0
    mu = X[normal].mean(axis=0); sd = X[normal].std(axis=0) + 1e-9
    Xn = (X - mu) / sd
    feat_names = [f"{c}_{stat}" for c in feat_cols
                  for stat in ("mean", "std", "min", "max", "trend")]
    return {"X": Xn, "y": y, "rul_min": ruls, "timestamps": [str(t) for t in tstamps],
            "normal_mask": normal, "feat_names": feat_names, "n_features": Xn.shape[1],
            "n_samples": len(y), "n_normal": int(normal.sum()), "n_fault": int((y == 1).sum()),
            "raw_rows": total, "resample": resample, "win": win, "hop": hop,
            "lead_days": lead_days, "onsets": [str(o) for o in onsets],
            "norm": {"mu": mu.tolist(), "sd": sd.tolist()}}


if __name__ == "__main__":
    import sys, time
    t0 = time.time()
    d = load(sys.argv[1] if len(sys.argv) > 1 else "data",
             progress=lambda ph, i, n: print(f"  {ph} {i} ({n})", flush=True) if i % 3 == 0 else None)
    print(f"\nMetroPT-3: {d['raw_rows']:,} filas crudas → {d['n_samples']} ventanas, "
          f"{d['n_features']} features")
    print(f"  sanos: {d['n_normal']}  ·  pre-fallo: {d['n_fault']}  ·  {time.time()-t0:.0f}s")
