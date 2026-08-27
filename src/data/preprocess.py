"""Preprocesado: de los datos crudos a dos productos canonicos.

Cada dataset trae un layout distinto (ASCII de vibracion a 20 kHz, HDF5 de vuelo,
CSV de sensores, .mat de Matlab). Este modulo los normaliza a dos formatos que el
entrenador consume sin saber de donde vienen:

  1) PRODUCTO RUL  (kind=run_to_failure)  ->  data/processed/<clave>_rul.npz
       X       (N, T, F) float32   ventanas deslizantes de sensores/features
       y_rul   (N,)      float32   vida util restante, en "unidades" del dataset
       unit    (N,)      int32     id de la maquina/ensayo (para partir train/val
                                   POR MAQUINA y no filtrar informacion)
     meta.hours_per_unit convierte esas unidades a horas -> de ahi salen los DIAS
     de anticipacion que exige el objetivo.

  2) PRODUCTO CLASIFICACION (kind=vibration_fault | anomaly) -> <clave>_cls.npz
       Xf      (M, F) float32   features de vibracion por registro
       y       (M,)   int32     0 = sano, 1 = fallo
     Alimenta el baseline de boosting, el autoencoder de salud y el umbral por coste.

Las features de vibracion son las clasicas de diagnostico de rodamiento (RMS,
curtosis, factor de cresta...) mas la energia por bandas de FFT: es lo que hace que
un modelo pequeno funcione en el edge sin procesar la senal cruda.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

ProgressCB = Callable[[str, float, str], None]

WINDOW_DEFAULT = 32          # longitud de ventana temporal (pasos)
STRIDE_DEFAULT = 1
RUL_CAP_DEFAULT = 130.0      # techo de RUL: al principio de la vida "queda mucho" y
                             # el valor exacto no es aprendible ni util (practica estandar)

# Horas reales que representa un "paso" de cada dataset. Es la constante que traduce
# la RUL aprendida a DIAS DE ANTICIPACION, que es lo que mide el objetivo del proyecto.
HOURS_PER_UNIT = {
    "cmapss": 24.0,            # 1 ciclo de vuelo, tratado como 1 dia de operacion (supuesto)
    "ncmapss": 24.0,           # idem
    "nasa_ims_bearing": 1 / 6,   # 1 instantanea cada 10 min (medido)
    "metropt3": 1.0,           # remuestreado a 1 hora (medido)
}

# Ventana/salto por dataset: la ventana debe cubrir contexto suficiente para ver la
# TENDENCIA de degradacion, y el salto mantiene el numero de ventanas manejable.
_TUNING = {
    "cmapss":           {"window": 32, "stride": 1},    # ~1 mes de vuelos
    "ncmapss":          {"window": 32, "stride": 1},
    "nasa_ims_bearing": {"window": 64, "stride": 2},    # ~10,7 h de contexto
    "metropt3":         {"window": 48, "stride": 2},    # 2 dias de contexto
}

# Techo de RUL = 4x el horizonte de aviso. Predecir mas alla no aporta: lo que decide
# el exito es acertar la ventana de los ultimos `lead_time_days` dias.
_RUL_CAP_FACTOR = 4.0


def lead_horizon_units(key: str, lead_time_days: float) -> float:
    """Cuantos pasos de ESTE dataset son los dias de anticipacion exigidos."""
    hpu = HOURS_PER_UNIT.get(key, 24.0)
    return lead_time_days * 24.0 / hpu

# --- features de vibracion --------------------------------------------------
_FEATURE_NAMES = ("rms", "std", "kurtosis", "skew", "peak", "p2p", "crest",
                  "shape", "impulse", "clearance", "entropy")
_N_BANDS = 8


def vibration_features(sig: np.ndarray, n_bands: int = _N_BANDS) -> np.ndarray:
    """Features de diagnostico de una senal cruda 1-D + energia por bandas de FFT."""
    x = np.asarray(sig, dtype=np.float64).ravel()
    if x.size == 0:
        return np.zeros(len(_FEATURE_NAMES) + n_bands, dtype=np.float32)
    x = x - x.mean()
    absx = np.abs(x)
    rms = float(np.sqrt(np.mean(x ** 2))) or 1e-12
    mean_abs = float(absx.mean()) or 1e-12
    peak = float(absx.max())
    std = float(x.std()) or 1e-12
    kurt = float(np.mean((x / std) ** 4))
    skew = float(np.mean((x / std) ** 3))
    sqrt_mean = float(np.mean(np.sqrt(absx))) ** 2 or 1e-12

    spec = np.abs(np.fft.rfft(x)) ** 2
    total = float(spec.sum()) or 1e-12
    bands = np.array([b.sum() / total for b in np.array_split(spec, n_bands)])
    p = spec / total
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())

    base = np.array([rms, std, kurt, skew, peak, float(x.max() - x.min()),
                     peak / rms, rms / mean_abs, peak / mean_abs,
                     peak / sqrt_mean, entropy], dtype=np.float64)
    return np.concatenate([base, bands]).astype(np.float32)


def feature_names(n_channels: int = 1, n_bands: int = _N_BANDS) -> list[str]:
    names = list(_FEATURE_NAMES) + [f"band{i}" for i in range(n_bands)]
    if n_channels == 1:
        return names
    return [f"ch{c}_{n}" for c in range(n_channels) for n in names]


# --- utilidades de ventaneo -------------------------------------------------
def _windows_from_trajectory(mat: np.ndarray, window: int, stride: int):
    """Ventanas causales de una trayectoria (T, F). Devuelve (idx_fin, ventanas)."""
    T = mat.shape[0]
    if T < window:                      # trayectoria corta: se rellena por delante
        pad = np.repeat(mat[:1], window - T, axis=0)
        mat = np.concatenate([pad, mat], axis=0)
        T = window
    ends = np.arange(window - 1, T, stride)
    win = np.stack([mat[e - window + 1: e + 1] for e in ends]).astype(np.float32)
    return ends, win


def _build_rul_product(trajs: dict[int, np.ndarray], window: int, stride: int,
                       rul_cap: float) -> dict[str, np.ndarray]:
    """De {unit: (T,F)} a ventanas X + RUL. La RUL se mide desde el final de cada
    trayectoria (que por construccion es el instante del fallo)."""
    Xs, ys, us, ts = [], [], [], []
    for unit, mat in trajs.items():
        if mat.shape[0] < 2:
            continue
        ends, win = _windows_from_trajectory(mat, window, stride)
        T = mat.shape[0]
        rul = np.clip((T - 1) - np.minimum(ends, T - 1), 0, rul_cap).astype(np.float32)
        Xs.append(win); ys.append(rul)
        us.append(np.full(len(ends), unit, dtype=np.int32))
        ts.append(ends.astype(np.int32))
    if not Xs:
        raise RuntimeError("ninguna trayectoria utilizable")
    return {"X": np.concatenate(Xs), "y_rul": np.concatenate(ys),
            "unit": np.concatenate(us), "t_idx": np.concatenate(ts)}


def _save(out_dir: Path, key: str, suffix: str, arrays: dict[str, np.ndarray],
          meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{key}_{suffix}.npz"
    # numpy anade ".npz" si el nombre no acaba en .npz, asi que el temporal debe acabar
    # en .npz para que os.replace encuentre el fichero que realmente escribio.
    tmp = out_dir / f".{key}_{suffix}.partial.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(npz)
    meta_out = {**meta, "arrays": {k: list(v.shape) for k, v in arrays.items()}}
    (out_dir / f"{key}_{suffix}.meta.json").write_text(
        json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return npz


# =====================  ADAPTADORES POR DATASET  =============================

def prep_cmapss(raw: Path, out: Path, cb: ProgressCB, window: int, stride: int,
                rul_cap: float) -> dict[str, Any]:
    """C-MAPSS FD001-FD004: 26 columnas (unidad, ciclo, 3 op-settings, 21 sensores).

    Cada unidad del fichero de entrenamiento corre hasta el fallo, asi que la RUL es
    (ultimo ciclo - ciclo actual). Un ciclo = un vuelo completo.
    """
    files = sorted(raw.rglob("train_FD00*.txt"))
    if not files:
        raise FileNotFoundError("no se encontraron train_FD00*.txt")

    trajs: dict[int, np.ndarray] = {}
    offset = 0
    for fi, f in enumerate(files):
        cb("cmapss", 100.0 * fi / len(files), f"leyendo {f.name}")
        arr = np.loadtxt(f)
        units = arr[:, 0].astype(int)
        feats = arr[:, 2:]                      # op-settings + sensores
        for u in np.unique(units):
            trajs[offset + int(u)] = feats[units == u].astype(np.float32)
        offset += int(units.max()) + 1

    prod = _build_rul_product(trajs, window, stride, rul_cap)
    prod = _normalize_rul_product(prod)
    meta = {"key": "cmapss", "product": "rul", "n_units": len(trajs),
            "window": window, "stride": stride, "rul_cap": rul_cap,
            "unit_name": "ciclo de vuelo", "hours_per_unit": 24.0,
            "hours_per_unit_note": "SUPUESTO documentado: 1 ciclo C-MAPSS = 1 dia de "
                                   "operacion. C-MAPSS no publica duracion fisica del "
                                   "ciclo; la anticipacion en dias hereda este supuesto.",
            "features": [f"op{i}" for i in range(3)] + [f"s{i}" for i in range(1, 22)],
            "source_files": [f.name for f in files]}
    _save(out, "cmapss", "rul", prod, meta)
    return meta


def prep_ncmapss(raw: Path, out: Path, cb: ProgressCB, window: int, stride: int,
                 rul_cap: float, max_files: int = 3) -> dict[str, Any]:
    """N-CMAPSS (DS01-DS08, HDF5): perfiles de vuelo reales.

    Cada fichero trae millones de muestras a 1 Hz. Se agrega POR CICLO DE VUELO
    (media de cada sensor) para tener trayectorias del mismo orden que C-MAPSS:
    a nivel de flota, la degradacion se ve entre vuelos, no dentro de un vuelo.
    """
    import h5py

    files = sorted(raw.rglob("N-CMAPSS_DS*.h5"))
    files = [f for f in files if "Sample" not in f.name][:max_files]
    if not files:
        raise FileNotFoundError("no se encontraron N-CMAPSS_DS*.h5")

    trajs: dict[int, np.ndarray] = {}
    offset = 0
    var_names: list[str] = []
    for fi, f in enumerate(files):
        cb("ncmapss", 100.0 * fi / len(files), f"leyendo {f.name}")
        with h5py.File(f, "r") as h:
            X_s = np.array(h["X_s_dev"], dtype=np.float32)       # sensores
            W = np.array(h["W_dev"], dtype=np.float32)           # condiciones de vuelo
            A = np.array(h["A_dev"], dtype=np.float32)           # aux: unit, cycle, Fc, hs
            if not var_names:
                def _names(k):
                    return [s.decode() if isinstance(s, bytes) else str(s)
                            for s in np.array(h[k]).ravel()]
                var_names = _names("W_var") + _names("X_s_var")
        feats = np.concatenate([W, X_s], axis=1)
        unit_col, cycle_col = A[:, 0].astype(int), A[:, 1].astype(int)

        for u in np.unique(unit_col):
            m = unit_col == u
            cyc, fu = cycle_col[m], feats[m]
            order = np.argsort(cyc, kind="stable")
            cyc, fu = cyc[order], fu[order]
            # media por ciclo de vuelo (reduce ~1e6 muestras a ~100 ciclos)
            bounds = np.searchsorted(cyc, np.unique(cyc), side="left")
            traj = np.add.reduceat(fu, bounds, axis=0) / np.diff(
                np.append(bounds, len(cyc)))[:, None]
            trajs[offset + int(u)] = traj.astype(np.float32)
        offset += int(unit_col.max()) + 1

    prod = _build_rul_product(trajs, window, stride, rul_cap)
    prod = _normalize_rul_product(prod)
    meta = {"key": "ncmapss", "product": "rul", "n_units": len(trajs),
            "window": window, "stride": stride, "rul_cap": rul_cap,
            "unit_name": "ciclo de vuelo", "hours_per_unit": 24.0,
            "hours_per_unit_note": "mismo supuesto que C-MAPSS (1 ciclo = 1 dia).",
            "features": var_names, "source_files": [f.name for f in files]}
    _save(out, "ncmapss", "rul", prod, meta)
    return meta


# Rodamiento que efectivamente rompio en cada ensayo IMS (documentado en el readme
# del dataset). El ensayo termina cuando ese rodamiento falla, asi que los cuatro
# rodamientos del banco comparten el instante de fallo.
_IMS_FAILED = {"1st_test": (2, 3), "2nd_test": (0,), "3rd_test": (2,), "4th_test": (2,)}

# Numero de instantaneas que el readme del dataset documenta para cada ensayo.
#
# CRITICO: la carpeta del 3er ensayo (que en disco se llama `4th_test`) trae 6.324
# ficheros, pero el ensayo documentado termina en el 4.448 -- 2004.04.04 19:01:57,
# cuando falla el rodamiento 3. Los 1.876 restantes llegan hasta el 18 de abril, ya
# despues del fallo. Como la RUL se mide desde el FINAL de la trayectoria, no truncar
# desplaza la etiqueta 13 dias en un tercio de los rodamientos: el modelo aprende que
# al fallo le quedan casi dos semanas mas de las que le quedan.
_IMS_DOCUMENTED_SNAPSHOTS = {"1st_test": 2156, "2nd_test": 984, "3rd_test": 4448, "4th_test": 4448}


def _ims_snapshot(path: Path) -> np.ndarray | None:
    """Lee una instantanea IMS (ASCII tabulado, 20480 muestras x 4 u 8 canales)."""
    import pandas as pd
    try:
        return pd.read_csv(path, sep="\t", header=None, dtype=np.float32).to_numpy()
    except Exception:
        return None


def prep_ims(raw: Path, out: Path, cb: ProgressCB, window: int, stride: int,
             rul_cap: float) -> dict[str, Any]:
    """IMS (NASA/Univ. Cincinnati): rodamientos hasta rotura, ASCII a 20 kHz.

    Un fichero = 1 s de vibracion cada 10 min. La secuencia de instantaneas ES la
    trayectoria de degradacion y el ultimo fichero es el fallo. Es el caso mas
    parecido al motor/rotor de planta del proyecto.

    UNA TRAYECTORIA POR RODAMIENTO (4 por ensayo). El 1er ensayo tiene 8 canales
    (dos acelerometros por rodamiento) y los otros 4 (uno): se toma el primer
    acelerometro de cada rodamiento para que las tres tandas tengan el mismo numero
    de features y puedan entrenarse juntas.
    """
    from concurrent.futures import ThreadPoolExecutor

    # El volcado de NASA anida de forma irregular (el 3er ensayo cuelga de
    # `3rd_test/4th_test/txt/`), asi que se buscan por CONTENIDO: toda carpeta con
    # suficientes ficheros cuyo nombre sea una marca de tiempo.
    stamp = re.compile(r"\d{4}\.\d\d\.\d\d\.\d\d\.\d\d\.\d\d$")
    test_dirs = [p for p in raw.rglob("*") if p.is_dir()
                 and sum(1 for f in p.iterdir() if f.is_file() and stamp.match(f.name)) >= 50]
    if not test_dirs:
        raise FileNotFoundError("no se encontraron carpetas de ensayo IMS")

    def test_name(p: Path) -> str:
        """Nombre del ensayo, subiendo por la ruta (la hoja puede llamarse `txt`)."""
        for part in reversed(p.parts):
            if re.fullmatch(r"\dst_test|\dnd_test|\drd_test|\dth_test", part):
                return part
        return p.name

    trajs: dict[int, np.ndarray] = {}
    unit_meta: list[dict[str, Any]] = []
    unit_id = 0
    for td in sorted(set(test_dirs), key=lambda p: str(p)):
        snaps = sorted((p for p in td.iterdir() if p.is_file() and stamp.match(p.name)),
                       key=lambda p: p.name)
        if len(snaps) < 50:
            continue                               # carpeta contenedora, no de datos
        # truncar al fallo documentado (ver _IMS_DOCUMENTED_SNAPSHOTS)
        documented = _IMS_DOCUMENTED_SNAPSHOTS.get(test_name(td))
        if documented and len(snaps) > documented:
            cb("nasa_ims_bearing", 0.0,
               f"{test_name(td)}: {len(snaps)} ficheros -> {documented} (fallo documentado)")
            snaps = snaps[:documented]

        rows: list[np.ndarray | None] = [None] * len(snaps)
        done = 0

        def work(i_sp):
            i, sp = i_sp
            sig = _ims_snapshot(sp)
            if sig is None or sig.ndim != 2:
                return i, None
            n_bear = 4
            step = sig.shape[1] // n_bear or 1     # 2 canales/rodamiento en el 1er ensayo
            per = [vibration_features(sig[:, b * step]) for b in range(n_bear)]
            return i, per

        with ThreadPoolExecutor(max_workers=8) as ex:
            for i, per in ex.map(work, list(enumerate(snaps))):
                rows[i] = per
                done += 1
                if done % 200 == 0:
                    cb("nasa_ims_bearing", 100.0 * done / len(snaps),
                       f"{test_name(td)} {done}/{len(snaps)}")

        good = [r for r in rows if r is not None]
        if len(good) < 50:
            continue
        failed = _IMS_FAILED.get(test_name(td), ())
        for b in range(4):
            trajs[unit_id] = np.stack([g[b] for g in good]).astype(np.float32)
            unit_meta.append({"unit": unit_id, "test": test_name(td), "bearing": b + 1,
                              "failed": b in failed, "snapshots": len(good)})
            unit_id += 1

    if not trajs:
        raise RuntimeError("IMS: ninguna trayectoria con suficientes instantaneas")

    prod = _build_rul_product(trajs, window, stride, rul_cap)
    prod = _normalize_rul_product(prod)
    failed_units = np.array([u["unit"] for u in unit_meta if u["failed"]], dtype=np.int32)
    prod["failed_units"] = failed_units
    meta = {"key": "nasa_ims_bearing", "product": "rul", "n_units": len(trajs),
            "window": window, "stride": stride, "rul_cap": rul_cap,
            "unit_name": "instantanea de 10 min", "hours_per_unit": 1 / 6,
            "hours_per_unit_note": "IMS graba 1 s de vibracion cada 10 min (medido, no supuesto).",
            "features": feature_names(1), "units": unit_meta,
            "failed_units": failed_units.tolist(),
            "nota_unidades": "los 4 rodamientos de un banco comparten el instante de fallo "
                             "(el ensayo se detiene ahi); 'failed' marca cual rompio."}
    _save(out, "nasa_ims_bearing", "rul", prod, meta)
    return meta


# Periodos de fallo publicados de MetroPT-3 (compresor APU). Fuente: la ficha del
# dataset en UCI y el articulo que lo acompana. Se usan para derivar la RUL.
_METROPT_FAILURES = [
    ("2020-04-18 00:00:00", "2020-04-18 23:59:00"),
    ("2020-05-29 23:30:00", "2020-05-30 06:00:00"),
    ("2020-06-05 10:00:00", "2020-06-07 14:30:00"),
    ("2020-07-15 14:30:00", "2020-07-15 19:00:00"),
]


def prep_metropt3(raw: Path, out: Path, cb: ProgressCB, window: int, stride: int,
                  rul_cap: float) -> dict[str, Any]:
    """MetroPT-3: compresor de aire de un metro, 15 meses a 1 Hz, fallos fechados.

    Se remuestrea a 1 minuto (media) y cada tramo entre fallos se trata como una
    trayectoria run-to-failure independiente.
    """
    import pandas as pd

    csvs = [p for p in raw.rglob("*.csv") if p.stat().st_size > 10_000_000]
    if not csvs:
        raise FileNotFoundError("no se encontro el CSV de MetroPT-3")
    cb("metropt3", 5.0, f"leyendo {csvs[0].name}")
    df = pd.read_csv(csvs[0])
    tcol = next((c for c in df.columns if "time" in c.lower() or "date" in c.lower()), None)
    if tcol is None:
        raise RuntimeError("MetroPT-3: sin columna temporal reconocible")
    df[tcol] = pd.to_datetime(df[tcol])
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")])
    df = df.set_index(tcol).select_dtypes("number")

    # 1 hora: el objetivo es avisar con >=10 dias, no detectar transitorios de segundos.
    cb("metropt3", 40.0, "remuestreando a 1 h")
    df = df.resample("1h").mean().interpolate(limit=6).dropna()

    fails = [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in _METROPT_FAILURES]
    fails = [(a, b) for a, b in fails if df.index.min() <= a <= df.index.max()]
    if not fails:
        raise RuntimeError("MetroPT-3: los fallos publicados caen fuera del rango del CSV")

    cb("metropt3", 70.0, f"{len(fails)} fallos -> trayectorias")
    trajs: dict[int, np.ndarray] = {}
    start = df.index.min()
    for i, (fa, fb) in enumerate(fails):
        seg = df.loc[start:fa]                     # desde el fallo anterior hasta este
        if len(seg) > window * 4:
            trajs[i] = seg.to_numpy(dtype=np.float32)
        start = fb                                 # el periodo de fallo no se entrena

    if not trajs:
        raise RuntimeError("MetroPT-3: sin tramos suficientemente largos")
    prod = _build_rul_product(trajs, window, stride, rul_cap)
    prod = _normalize_rul_product(prod)
    meta = {"key": "metropt3", "product": "rul", "n_units": len(trajs),
            "window": window, "stride": stride, "rul_cap": rul_cap,
            "unit_name": "hora", "hours_per_unit": 1.0,
            "hours_per_unit_note": "remuestreado a 1 h desde 1 Hz (medido).",
            "features": list(df.columns), "failures": _METROPT_FAILURES}
    _save(out, "metropt3", "rul", prod, meta)
    return meta


def prep_cwru(raw: Path, out: Path, cb: ProgressCB) -> dict[str, Any]:
    """CWRU: .mat de Matlab, vibracion etiquetada. Producto de CLASIFICACION.

    Los ficheros 97-100 son el baseline sano; el resto son fallos (pista interna,
    bola, pista externa) a distintas severidades. Cada senal se trocea en segmentos
    de 4096 muestras y de cada segmento salen las features de vibracion.
    """
    from scipy.io import loadmat

    mats = sorted(raw.rglob("*.mat"))
    if not mats:
        raise FileNotFoundError("no se encontraron .mat de CWRU")
    healthy = {"97", "98", "99", "100"}
    seg_len = 4096
    Xs, ys = [], []
    for mi, m in enumerate(mats):
        cb("cwru_bearing", 100.0 * mi / len(mats), m.name)
        try:
            d = loadmat(m)
        except Exception:
            continue
        de = [v for k, v in d.items() if k.endswith("_DE_time")]
        if not de:
            continue
        sig = np.asarray(de[0]).ravel()
        label = 0 if m.stem in healthy else 1
        n_seg = min(60, len(sig) // seg_len)
        for s in range(n_seg):
            Xs.append(vibration_features(sig[s * seg_len:(s + 1) * seg_len]))
            ys.append(label)
    if not Xs:
        raise RuntimeError("CWRU: ningun segmento extraido")
    arrays = {"Xf": np.stack(Xs).astype(np.float32), "y": np.array(ys, dtype=np.int32)}
    meta = {"key": "cwru_bearing", "product": "cls", "segment_len": seg_len,
            "n_healthy": int((arrays["y"] == 0).sum()), "n_fault": int((arrays["y"] == 1).sum()),
            "features": feature_names(1), "source_files": len(mats)}
    _save(out, "cwru_bearing", "cls", arrays, meta)
    return meta


def prep_mfpt(raw: Path, out: Path, cb: ProgressCB) -> dict[str, Any]:
    """MFPT: baseline sano + fallo de pista interna/externa. Producto de CLASIFICACION."""
    from scipy.io import loadmat

    mats = sorted(raw.rglob("*.mat"))
    if not mats:
        raise FileNotFoundError("no se encontraron .mat de MFPT")
    seg_len = 4096
    Xs, ys = [], []
    for mi, m in enumerate(mats):
        cb("mfpt_bearing", 100.0 * mi / len(mats), m.name)
        try:
            d = loadmat(m, struct_as_record=False, squeeze_me=True)
        except Exception:
            continue
        bundle = d.get("bearing")
        sig = None
        if bundle is not None:
            sig = getattr(bundle, "gs", None)
        if sig is None:
            cand = [v for k, v in d.items()
                    if not k.startswith("__") and isinstance(v, np.ndarray) and v.size > seg_len]
            if not cand:
                continue
            sig = cand[0]
        sig = np.asarray(sig, dtype=np.float64).ravel()
        label = 0 if "baseline" in str(m).lower() else 1
        n_seg = min(60, len(sig) // seg_len)
        for s in range(n_seg):
            Xs.append(vibration_features(sig[s * seg_len:(s + 1) * seg_len]))
            ys.append(label)
    if not Xs:
        raise RuntimeError("MFPT: ningun segmento extraido")
    arrays = {"Xf": np.stack(Xs).astype(np.float32), "y": np.array(ys, dtype=np.int32)}
    meta = {"key": "mfpt_bearing", "product": "cls", "segment_len": seg_len,
            "n_healthy": int((arrays["y"] == 0).sum()), "n_fault": int((arrays["y"] == 1).sum()),
            "features": feature_names(1), "source_files": len(mats)}
    _save(out, "mfpt_bearing", "cls", arrays, meta)
    return meta


def prep_skab(raw: Path, out: Path, cb: ProgressCB) -> dict[str, Any]:
    """SKAB: banco con bomba de agua, anomalias etiquetadas por instante.

    Producto de CLASIFICACION directamente sobre los sensores (ya son features).
    """
    import pandas as pd

    csvs = [p for p in raw.rglob("*.csv") if "anomaly-free" not in p.name]
    csvs = [p for p in csvs if p.parent.name in ("other", "valve1", "valve2")]
    if not csvs:
        csvs = [p for p in raw.rglob("*.csv")]
    Xs, ys = [], []
    cols: list[str] = []
    for ci, c in enumerate(csvs):
        cb("skab", 100.0 * ci / max(1, len(csvs)), c.name)
        try:
            df = pd.read_csv(c, sep=";", index_col=0)
        except Exception:
            continue
        if "anomaly" not in df.columns:
            continue
        y = df["anomaly"].to_numpy(dtype=np.int32)
        f = df.drop(columns=[c2 for c2 in ("anomaly", "changepoint") if c2 in df.columns])
        f = f.select_dtypes("number")
        if not cols:
            cols = list(f.columns)
        Xs.append(f.to_numpy(dtype=np.float32)); ys.append(y)
    if not Xs:
        raise RuntimeError("SKAB: ningun CSV con columna 'anomaly'")
    arrays = {"Xf": np.concatenate(Xs), "y": np.concatenate(ys)}
    meta = {"key": "skab", "product": "cls", "features": cols,
            "n_healthy": int((arrays["y"] == 0).sum()),
            "n_fault": int((arrays["y"] == 1).sum()), "source_files": len(csvs)}
    _save(out, "skab", "cls", arrays, meta)
    return meta


# --- normalizacion ----------------------------------------------------------
def _normalize_rul_product(prod: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """z-score por feature sobre todas las ventanas. Columnas constantes -> 0.

    Se guarda mu/sigma en el propio npz: el edge necesita exactamente la misma
    normalizacion en inferencia.
    """
    X = prod["X"]
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0)
    sd = flat.std(axis=0)
    sd[sd < 1e-8] = 1.0
    prod["X"] = ((X - mu) / sd).astype(np.float32)
    prod["norm_mu"] = mu.astype(np.float32)
    prod["norm_sd"] = sd.astype(np.float32)
    np.nan_to_num(prod["X"], copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return prod


# =====================  ORQUESTACION  ========================================

_ADAPTERS: dict[str, Any] = {
    "cmapss": prep_cmapss,
    "ncmapss": prep_ncmapss,
    "nasa_ims_bearing": prep_ims,
    "metropt3": prep_metropt3,
    "cwru_bearing": prep_cwru,
    "mfpt_bearing": prep_mfpt,
    "skab": prep_skab,
}

_RUL_ADAPTERS = {"cmapss", "ncmapss", "nasa_ims_bearing", "metropt3"}


def run_all(data_dir: str | Path, cfg: dict[str, Any] | None = None,
            cb: ProgressCB | None = None, force: bool = False) -> dict[str, Any]:
    """Preprocesa todos los datasets descargados que tengan adaptador.

    Idempotente: si ya existe el .npz de un dataset, lo salta (salvo `force`).
    Tolerante: si un adaptador falla, lo registra y sigue con el resto.
    """
    cb = cb or (lambda k, p, m: None)
    cfg = cfg or {}
    lead_days = float((cfg.get("target") or {}).get("lead_time_days", 10))
    over = cfg.get("preprocess") or {}

    root = Path(data_dir)
    raw_root = root / "raw"
    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)

    from .download import available
    keys = available(root)
    summary: dict[str, Any] = {"ok": [], "skipped": [], "failed": [], "products": {}}

    for key in keys:
        fn = _ADAPTERS.get(key)
        if fn is None:
            summary["skipped"].append({"key": key, "reason": "sin adaptador"})
            continue
        suffix = "rul" if key in _RUL_ADAPTERS else "cls"
        npz = out / f"{key}_{suffix}.npz"
        if npz.exists() and not force:
            summary["skipped"].append({"key": key, "reason": "ya preprocesado"})
            summary["products"][key] = str(npz)
            continue
        try:
            cb(key, 0.0, "preprocesando")
            if key in _RUL_ADAPTERS:
                tune = _TUNING.get(key, {})
                window = int(over.get("window", tune.get("window", WINDOW_DEFAULT)))
                stride = int(over.get("stride", tune.get("stride", STRIDE_DEFAULT)))
                rul_cap = float(over.get("rul_cap",
                                         _RUL_CAP_FACTOR * lead_horizon_units(key, lead_days)))
                meta = fn(raw_root / key, out, cb, window, stride, rul_cap)
                # el adaptador ya escribio el meta.json; le anadimos el horizonte
                mp = out / f"{key}_rul.meta.json"
                meta = {**json.loads(mp.read_text(encoding="utf-8")),
                        "lead_horizon_units": lead_horizon_units(key, lead_days),
                        "lead_time_days_target": lead_days}
                mp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            else:
                meta = fn(raw_root / key, out, cb)
            summary["ok"].append(key)
            summary["products"][key] = str(out / f"{key}_{meta['product']}.npz")
            cb(key, 100.0, f"listo ({meta.get('n_units', meta.get('n_fault', '?'))})")
        except Exception as e:
            summary["failed"].append({"key": key, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            cb(key, 100.0, f"FALLO: {str(e)[:120]}")

    (out / "_PREPROCESS_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def load_product(data_dir: str | Path, key: str, product: str = "rul"):
    """Carga (arrays, meta) de un producto preprocesado."""
    out = Path(data_dir) / "processed"
    npz = out / f"{key}_{product}.npz"
    meta_p = out / f"{key}_{product}.meta.json"
    if not npz.exists():
        raise FileNotFoundError(npz)
    data = dict(np.load(npz))
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return data, meta


def list_products(data_dir: str | Path) -> dict[str, list[str]]:
    """Productos disponibles en disco, agrupados por tipo."""
    out = Path(data_dir) / "processed"
    if not out.exists():
        return {"rul": [], "cls": []}
    return {"rul": sorted(p.stem[:-4] for p in out.glob("*_rul.npz")),
            "cls": sorted(p.stem[:-4] for p in out.glob("*_cls.npz"))}
