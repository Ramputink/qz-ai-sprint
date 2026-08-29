"""Consumo electrico — adaptadores a dos productos canonicos.

Este es el bloque que responde a la pregunta central del proyecto: **si se puede
optimizar el consumo electrico**. Se separa de `preprocess.py` (mantenimiento
predictivo) porque la pregunta es otra y la forma del dato tambien.

  1) PRODUCTO CONSUMO  ->  data/processed/<clave>_consumo.npz
       y            (S, T) float32   carga por serie (NaN donde falta)
       time_feats   (T, Ct)          calendario: hora, dia de semana, mes, finde
       weather      (W, T, Cw)       meteorologia por emplazamiento
       site_of_serie(S,)  int32      a que emplazamiento pertenece cada serie
       timestamps   (T,)  int64      epoch en segundos
     Se guardan las SERIES, no las ventanas: 1578 edificios x 17544 horas caben en
     110 MB, mientras que sus ventanas ocuparian gigabytes. El entrenador las
     trocea sobre la marcha en GPU, que ademas permite cambiar el horizonte sin
     reprocesar nada.

  2) PRODUCTO NILM  ->  data/processed/<clave>_nilm.npz
       mains        (T,)   float32   potencia agregada de la acometida
       appliances   (T, A) float32   potencia por aparato/maquina
     Para desagregar: saber DONDE se va la energia sin instrumentar cada maquina.

Por que importa el contrafactual: no se puede demostrar un ahorro comparando el
consumo de este mes con el del anterior, porque cambian el clima y la produccion.
Hace falta un modelo entrenado ANTES de la intervencion que prediga lo que habria
pasado DESPUES (protocolo IPMVP Opcion C). Por eso el producto guarda los
timestamps: sin ellos no se puede partir en periodo pre y post.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

ProgressCB = Callable[[str, float, str], None]

_TIME_FEATURES = ("hora_sin", "hora_cos", "dow_sin", "dow_cos",
                  "mes_sin", "mes_cos", "finde")


def calendar_features(index) -> np.ndarray:
    """Calendario ciclico. Seno/coseno en vez del numero de hora crudo porque la
    hora 23 y la 0 son contiguas, y un modelo alimentado con 23 y 0 no lo sabe."""
    import pandas as pd

    idx = pd.DatetimeIndex(index)
    h, dow, mes = idx.hour.values, idx.dayofweek.values, idx.month.values
    return np.stack([
        np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * (mes - 1) / 12), np.cos(2 * np.pi * (mes - 1) / 12),
        (dow >= 5).astype(float),
    ], axis=1).astype(np.float32)


def _epoch(index) -> np.ndarray:
    """Epoch en segundos. `DatetimeIndex.view("int64")` ya devuelve un ndarray en
    pandas 2.x, asi que encadenar `.to_numpy()` revienta."""
    return (np.asarray(index.view("int64")) // 10 ** 9).astype(np.int64)


def _save(out_dir: Path, key: str, suffix: str, arrays: dict[str, np.ndarray],
          meta: dict[str, Any]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{key}_{suffix}.npz"
    tmp = out_dir / f".{key}_{suffix}.partial.npz"
    np.savez_compressed(tmp, **arrays)
    tmp.replace(npz)
    (out_dir / f"{key}_{suffix}.meta.json").write_text(
        json.dumps({**meta, "arrays": {k: list(v.shape) for k, v in arrays.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    return npz


def _cover(y: np.ndarray, minimo: float = 0.9) -> np.ndarray:
    """Series con al menos `minimo` de datos presentes. Una serie medio vacia mete
    mas ruido en la normalizacion que senal aporta."""
    return np.mean(np.isfinite(y), axis=1) >= minimo


# =====================  BDG2  ================================================
def prep_bdg2(raw: Path, out: Path, cb: ProgressCB,
              min_cobertura: float = 0.9) -> dict[str, Any]:
    """Building Data Genome 2: 1636 edificios reales, 2 anos horarios + meteorologia.

    Es el dataset de referencia para esto porque trae las tres cosas a la vez:
    consumo medido, el clima que lo explica en gran parte, y metadatos del edificio
    (uso, superficie). Sin el clima no se puede separar "consumo mas alto" de
    "hizo mas frio", que es justo lo que distingue un ahorro real de uno aparente.
    """
    import pandas as pd

    elec = next(raw.rglob("electricity_cleaned.csv"), None)
    wx = next(raw.rglob("weather.csv"), None)
    md = next(raw.rglob("metadata.csv"), None)
    if elec is None:
        raise FileNotFoundError("no se encontro electricity_cleaned.csv")

    cb("building_data_genome_2", 5.0, "leyendo consumo horario")
    df = pd.read_csv(elec, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    df = df.select_dtypes("number")
    y = df.to_numpy(dtype=np.float32).T                      # (S, T)
    series = list(df.columns)

    keep = _cover(y, min_cobertura)
    y, series = y[keep], [s for s, k in zip(series, keep) if k]
    cb("building_data_genome_2", 35.0, f"{len(series)} edificios con cobertura suficiente")

    # el nombre del edificio empieza por su emplazamiento: Panther_office_Hannah
    site_names = sorted({s.split("_")[0] for s in series})
    site_idx = {s: i for i, s in enumerate(site_names)}
    site_of_serie = np.array([site_idx[s.split("_")[0]] for s in series], dtype=np.int32)

    wx_cols = ["airTemperature", "dewTemperature", "windSpeed",
               "cloudCoverage", "precipDepth1HR"]
    weather = np.zeros((len(site_names), y.shape[1], len(wx_cols)), dtype=np.float32)
    if wx is not None:
        cb("building_data_genome_2", 55.0, "alineando meteorologia por emplazamiento")
        w = pd.read_csv(wx, parse_dates=["timestamp"])
        for name, i in site_idx.items():
            sub = w[w["site_id"] == name].set_index("timestamp").sort_index()
            # la meteo trae timestamps repetidos en algunos emplazamientos y reindex
            # no admite etiquetas duplicadas: nos quedamos con la primera lectura
            sub = sub[~sub.index.duplicated(keep="first")].reindex(df.index)
            for j, c in enumerate(wx_cols):
                col = sub[c] if c in sub.columns else pd.Series(index=df.index, dtype=float)
                weather[i, :, j] = col.interpolate(limit_direction="both").fillna(0.0).to_numpy()

    usos: dict[str, str] = {}
    if md is not None:
        m = pd.read_csv(md)
        col_id = "building_id" if "building_id" in m.columns else m.columns[0]
        if "primaryspaceusage" in m.columns:
            usos = dict(zip(m[col_id].astype(str), m["primaryspaceusage"].astype(str)))

    arrays = {"y": y, "time_feats": calendar_features(df.index), "weather": weather,
              "site_of_serie": site_of_serie,
              "timestamps": _epoch(df.index)}
    meta = {"key": "building_data_genome_2", "product": "consumo",
            "frecuencia": "1h", "unidad": "kWh", "n_series": len(series),
            "n_pasos": int(y.shape[1]), "series": series, "sitios": site_names,
            "time_features": list(_TIME_FEATURES), "weather_features": wx_cols,
            "usos": {s: usos.get(s, "") for s in series[:50]},
            "cobertura_minima": min_cobertura,
            "periodo": [str(df.index[0]), str(df.index[-1])]}
    _save(out, "building_data_genome_2", "consumo", arrays, meta)
    return meta


# =====================  UCI ElectricityLoadDiagrams  =========================
def prep_electricity_load(raw: Path, out: Path, cb: ProgressCB,
                          min_cobertura: float = 0.9) -> dict[str, Any]:
    """370 clientes, 15 min, 2011-2014. Referencia clasica: permite comparar contra
    resultados publicados en vez de contra uno mismo. Se agrega a hora para que sea
    homogeneo con el resto y porque el objetivo es la factura, no el transitorio."""
    import pandas as pd

    txt = next(raw.rglob("LD2011_2014.txt"), None)
    if txt is None:
        raise FileNotFoundError("no se encontro LD2011_2014.txt")
    cb("electricity_load_diagrams", 10.0, "leyendo 370 clientes a 15 min")
    df = pd.read_csv(txt, sep=";", decimal=",", index_col=0, parse_dates=True,
                     low_memory=False)
    df = df.select_dtypes("number").resample("1h").sum()
    y = df.to_numpy(dtype=np.float32).T
    keep = _cover(y, min_cobertura)
    y, series = y[keep], [c for c, k in zip(df.columns, keep) if k]

    arrays = {"y": y, "time_feats": calendar_features(df.index),
              "weather": np.zeros((1, y.shape[1], 0), dtype=np.float32),
              "site_of_serie": np.zeros(len(series), dtype=np.int32),
              "timestamps": _epoch(df.index)}
    meta = {"key": "electricity_load_diagrams", "product": "consumo",
            "frecuencia": "1h", "unidad": "kW", "n_series": len(series),
            "n_pasos": int(y.shape[1]), "series": list(series), "sitios": ["unico"],
            "time_features": list(_TIME_FEATURES), "weather_features": [],
            "nota": "sin meteorologia: mide cuanto se puede predecir solo con el "
                    "calendario y el propio historico",
            "periodo": [str(df.index[0]), str(df.index[-1])]}
    _save(out, "electricity_load_diagrams", "consumo", arrays, meta)
    return meta


# =====================  Steel Industry  ======================================
def prep_steel(raw: Path, out: Path, cb: ProgressCB) -> dict[str, Any]:
    """Planta siderurgica: el unico dataset realmente INDUSTRIAL del lote.

    Trae lo que de verdad se toca para optimizar una factura industrial: potencia
    reactiva adelantada y retrasada, factor de potencia y tipo de carga. La reactiva
    es dinero directo -- se penaliza en factura y se corrige con condensadores, sin
    tocar la produccion -- asi que aqui hay una via de ahorro que no existe en los
    datasets de edificios.
    """
    import pandas as pd

    csv = next(raw.rglob("Steel_industry_data.csv"), None)
    if csv is None:
        raise FileNotFoundError("no se encontro Steel_industry_data.csv")
    cb("steel_industry_energy", 20.0, "leyendo planta siderurgica")
    df = pd.read_csv(csv, encoding="utf-8-sig")
    tcol = df.columns[0]
    df[tcol] = pd.to_datetime(df[tcol], dayfirst=True)
    df = df.set_index(tcol).sort_index()

    carga = "Usage_kWh"
    extras = [c for c in df.columns
              if c != carga and pd.api.types.is_numeric_dtype(df[c])]
    horas = df.resample("1h").mean(numeric_only=True)
    y = horas[[carga]].to_numpy(dtype=np.float32).T
    cov = horas[extras].interpolate(limit_direction="both").fillna(0.0)

    arrays = {"y": y, "time_feats": calendar_features(horas.index),
              "weather": cov.to_numpy(dtype=np.float32)[None, :, :],
              "site_of_serie": np.zeros(1, dtype=np.int32),
              "timestamps": _epoch(horas.index)}
    meta = {"key": "steel_industry_energy", "product": "consumo",
            "frecuencia": "1h", "unidad": "kWh", "n_series": 1,
            "n_pasos": int(y.shape[1]), "series": [carga], "sitios": ["planta"],
            "time_features": list(_TIME_FEATURES), "weather_features": extras,
            "nota": "las 'weather_features' aqui son variables de proceso "
                    "(reactiva, factor de potencia, CO2), no meteorologia",
            "periodo": [str(horas.index[0]), str(horas.index[-1])]}
    _save(out, "steel_industry_energy", "consumo", arrays, meta)
    return meta


# =====================  Low Carbon London  ===================================
def prep_lcl(raw: Path, out: Path, cb: ProgressCB, freq_h: int = 1,
             min_cobertura: float = 0.8, chunk: int = 8_000_000) -> dict[str, Any]:
    """Low Carbon London: 5.567 hogares REALES, media hora, 2011-2014.

    Es el salto de escala: ~167 millones de lecturas de contador inteligente frente
    a los 1.440 edificios de BDG2. Y son medidas de campo, no simulacion.

    El CSV viene en formato LARGO (una fila por hogar y marca de tiempo) y pesa
    8,5 GB, asi que se lee por bloques y se vuelca directamente sobre una matriz
    preasignada (hogares x horas): pivotar con pandas en memoria pediria decenas de
    GB para no ganar nada.
    """
    import pandas as pd

    csv = next(raw.rglob("*FullData.csv"), None)
    if csv is None:
        raise FileNotFoundError("no se encontro el CSV de Low Carbon London")

    col_id, col_t, col_v = "LCLid", "DateTime", "KWH/hh (per half hour) "
    cb("low_carbon_london", 2.0, "primera pasada: hogares y rango temporal")

    ids: dict[str, int] = {}
    t_min = t_max = None
    lector = pd.read_csv(csv, usecols=[col_id, col_t], chunksize=chunk,
                         parse_dates=[col_t])
    n_filas = 0
    for i, ch in enumerate(lector):
        for v in ch[col_id].unique():
            ids.setdefault(v, len(ids))
        lo, hi = ch[col_t].min(), ch[col_t].max()
        t_min = lo if t_min is None else min(t_min, lo)
        t_max = hi if t_max is None else max(t_max, hi)
        n_filas += len(ch)
        cb("low_carbon_london", 2 + 38 * min(1.0, n_filas / 1.7e8),
           f"{n_filas/1e6:.0f}M filas · {len(ids)} hogares")

    idx = pd.date_range(t_min.floor("h"), t_max.ceil("h"), freq=f"{freq_h}h")
    pos = {t: i for i, t in enumerate(idx)}
    S, T = len(ids), len(idx)
    acum = np.zeros((S, T), dtype=np.float32)
    visto = np.zeros((S, T), dtype=np.int16)

    cb("low_carbon_london", 42.0, f"segunda pasada: volcando {S}x{T}")
    lector = pd.read_csv(csv, usecols=[col_id, col_t, col_v], chunksize=chunk,
                         parse_dates=[col_t])
    hechas = 0
    for ch in lector:
        ch[col_v] = pd.to_numeric(ch[col_v], errors="coerce")
        ch = ch.dropna(subset=[col_v])
        fila = ch[col_id].map(ids).to_numpy()
        hora = ch[col_t].dt.floor(f"{freq_h}h").map(pos).to_numpy()
        ok = np.isfinite(hora.astype(float))
        fila, hora = fila[ok].astype(np.int64), hora[ok].astype(np.int64)
        vals = ch[col_v].to_numpy(dtype=np.float32)[ok]
        np.add.at(acum, (fila, hora), vals)      # media hora -> suma horaria
        np.add.at(visto, (fila, hora), 1)
        hechas += len(ch)
        cb("low_carbon_london", 42 + 50 * min(1.0, hechas / 1.7e8),
           f"{hechas/1e6:.0f}M filas volcadas")

    y = np.where(visto > 0, acum, np.nan)
    keep = _cover(y, min_cobertura)
    nombres = [k for k, v in sorted(ids.items(), key=lambda kv: kv[1])]
    y, nombres = y[keep], [n for n, k in zip(nombres, keep) if k]
    cb("low_carbon_london", 95.0, f"{len(nombres)} hogares con cobertura suficiente")

    arrays = {"y": np.nan_to_num(y).astype(np.float32),
              "time_feats": calendar_features(idx),
              "weather": np.zeros((1, T, 0), dtype=np.float32),
              "site_of_serie": np.zeros(len(nombres), dtype=np.int32),
              "timestamps": _epoch(idx)}
    meta = {"key": "low_carbon_london", "product": "consumo",
            "frecuencia": f"{freq_h}h", "unidad": "kWh", "n_series": len(nombres),
            "n_pasos": int(T), "series": nombres, "sitios": ["londres"],
            "time_features": list(_TIME_FEATURES), "weather_features": [],
            "filas_leidas": int(n_filas), "hogares_totales": len(ids),
            "nota": "medidas de campo de contador inteligente; media hora agregada a hora",
            "periodo": [str(idx[0]), str(idx[-1])]}
    _save(out, "low_carbon_london", "consumo", arrays, meta)
    return meta


# =====================  AMPds2 (NILM)  =======================================
# Medidor 1 = acometida (agregado). El resto son submedidas por circuito.
_AMPDS_MAINS = 1


def prep_ampds2(raw: Path, out: Path, cb: ProgressCB,
                freq: str = "1min") -> dict[str, Any]:
    """AMPds2 en formato NILMTK: acometida + 20 submedidas, 2 anos a 1 minuto.

    Producto para DESAGREGACION: aprender a descomponer la potencia total en sus
    consumidores. Es lo que permite decir "el 40 % se te va en climatizacion" sin
    poner un contador en cada maquina, que es la barrera de entrada real en planta.
    """
    import pandas as pd

    h5 = next(raw.rglob("AMPds2.h5"), None)
    if h5 is None:
        raise FileNotFoundError("no se encontro AMPds2.h5")

    # Se lee con h5py y no con pandas: el fichero lo escribio NILMTK con una version
    # antigua de PyTables y `pd.read_hdf` revienta al interpretar sus metadatos.
    # Ademas los bloques van comprimidos con blosc, que h5py solo abre si se ha
    # importado `hdf5plugin` (registra los filtros).
    import pickle

    import hdf5plugin  # noqa: F401  (registra los filtros de compresion)
    import h5py

    def columnas(g) -> list[str]:
        ejes = pickle.loads(g.attrs["non_index_axes"].tobytes()
                            if hasattr(g.attrs["non_index_axes"], "tobytes")
                            else g.attrs["non_index_axes"])
        return [" ".join(x for x in c if x).strip() for c in ejes[0][1]]

    def leer(f, meter: int, que: str = "power active"):
        g = f[f"building1/elec/meter{meter}"]
        cols = columnas(g)
        if que not in cols:
            return None, None
        j = cols.index(que)
        t = g["table"]
        datos = t[:]
        return datos["index"], datos["values_block_0"][:, j].astype(np.float32)

    with h5py.File(h5, "r") as f:
        cb("ampds2", 5.0, "leyendo acometida (potencia activa)")
        idx, mains_v = leer(f, _AMPDS_MAINS)
        if idx is None:
            raise RuntimeError("AMPds2: la acometida no expone potencia activa")

        sub, nombres = [], []
        for m in range(2, 22):
            cb("ampds2", 5.0 + 85.0 * (m - 1) / 21, f"submedida {m}/21")
            i2, v = leer(f, m)
            if v is None or len(v) != len(mains_v):
                continue
            sub.append(v)
            nombres.append(f"meter{m}")

    if not sub:
        raise RuntimeError("AMPds2: no se pudo leer ninguna submedida")

    ts = pd.DatetimeIndex(idx)
    A = np.stack(sub, axis=1)
    arrays = {"mains": np.nan_to_num(mains_v),
              "appliances": np.nan_to_num(A),
              "valido": np.isfinite(mains_v).astype(np.int8),
              "timestamps": _epoch(ts)}
    meta = {"key": "ampds2", "product": "nilm", "frecuencia": "1min", "unidad": "W",
            "n_pasos": int(len(mains_v)), "n_aparatos": len(nombres),
            "aparatos": nombres, "magnitud": "power active",
            "nota": "el fichero trae ademas potencia reactiva y aparente por circuito; "
                    "la reactiva es la palanca de ahorro que no toca la produccion",
            "periodo": [str(ts[0]), str(ts[-1])]}
    _save(out, "ampds2", "nilm", arrays, meta)
    return meta


# =====================  Orquestacion  ========================================
ADAPTERS = {
    "building_data_genome_2": prep_bdg2,
    "low_carbon_london": prep_lcl,
    "electricity_load_diagrams": prep_electricity_load,
    "steel_industry_energy": prep_steel,
    "ampds2": prep_ampds2,
}
_NILM = {"ampds2"}


def run_all(data_dir: str | Path, cb: ProgressCB | None = None,
            force: bool = False) -> dict[str, Any]:
    """Preprocesa los datasets de consumo descargados. Tolerante e idempotente."""
    cb = cb or (lambda k, p, m: None)
    root = Path(data_dir)
    out = root / "processed"
    out.mkdir(parents=True, exist_ok=True)

    from .download import available
    resumen: dict[str, Any] = {"ok": [], "saltados": [], "fallos": [], "productos": {}}
    for key in available(root):
        fn = ADAPTERS.get(key)
        if fn is None:
            continue
        suf = "nilm" if key in _NILM else "consumo"
        npz = out / f"{key}_{suf}.npz"
        if npz.exists() and not force:
            resumen["saltados"].append(key)
            resumen["productos"][key] = str(npz)
            continue
        try:
            cb(key, 0.0, "preprocesando consumo")
            meta = fn(root / "raw" / key, out, cb)
            resumen["ok"].append(key)
            resumen["productos"][key] = str(npz)
            cb(key, 100.0, f"listo ({meta.get('n_series', meta.get('n_aparatos'))})")
        except Exception as e:
            resumen["fallos"].append({"key": key, "error": f"{type(e).__name__}: {str(e)[:200]}"})
            cb(key, 100.0, f"FALLO: {str(e)[:120]}")
    (out / "_CONSUMO_SUMMARY.json").write_text(
        json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


def load_consumo(data_dir: str | Path, key: str, product: str = "consumo"):
    out = Path(data_dir) / "processed"
    npz = out / f"{key}_{product}.npz"
    if not npz.exists():
        raise FileNotFoundError(npz)
    meta_p = out / f"{key}_{product}.meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    return dict(np.load(npz)), meta
