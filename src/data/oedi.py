"""OEDI / NREL End-Use Load Profiles — el salto de escala real.

Que es: el catalogo publico de NREL con perfiles de carga de todo el parque
edificatorio de EE. UU. (ComStock para terciario, ResStock para residencial), a 15
minutos y un ano completo, por edificio individual. Escala de terabytes.

Que aporta que no tengamos ya:
  * RESOLUCION de 15 min en vez de horaria.
  * DESGLOSE POR USO FINAL en cada edificio (climatizacion, iluminacion, ventiladores,
    enchufes, agua caliente...). BDG2 y Low Carbon London solo dan el total, asi que
    con ellos se puede predecir el consumo pero no decir DONDE actuar.
  * VOLUMEN suficiente para responder si el modelo esta limitado por datos. El
    barrido de capacidad mostro que no lo esta por parametros: de 0,9 a 25,7 M
    apenas movio el resultado. La pregunta que queda es si con 10 veces mas
    edificios mejora, y para eso hacen falta 10 veces mas edificios.

Se baja por HTTPS del bucket publico (sin credenciales) y en paralelo, porque son
miles de ficheros pequenos y el cuello de botella es la latencia, no el ancho de banda.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

ProgressCB = Callable[[str, float, str], None]

BUCKET = "https://oedi-data-lake.s3.amazonaws.com"
RELEASE = ("nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/"
           "2024/comstock_amy2018_release_2")

# La columna del total electrico y el prefijo de los usos finales.
_TOTAL = "out.electricity.total.energy_consumption"
_PREFIJO_USO = "out.electricity."


def listar_edificios(estado: str, limite: int = 500,
                     cb: ProgressCB | None = None) -> list[str]:
    """Claves S3 de los parquet por edificio de un estado. Pagina si hace falta."""
    import requests

    prefijo = f"{RELEASE}/timeseries_individual_buildings/by_state/upgrade=0/state={estado}/"
    claves: list[str] = []
    token = None
    while len(claves) < limite:
        params = {"list-type": "2", "prefix": prefijo, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = requests.get(BUCKET, params=params, timeout=60)
        r.raise_for_status()
        claves += re.findall(r"<Key>([^<]+\.parquet)</Key>", r.text)
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", r.text)
        if cb:
            cb("oedi", 0.0, f"{estado}: {len(claves)} edificios listados")
        if not m:
            break
        token = m.group(1)
    return claves[:limite]


def descargar(estados: list[str], por_estado: int, destino: Path,
              cb: ProgressCB | None = None, hilos: int = 24) -> dict[str, Any]:
    """Baja `por_estado` edificios de cada estado. Idempotente: salta los ya bajados."""
    import requests

    cb = cb or (lambda k, p, m: None)
    destino.mkdir(parents=True, exist_ok=True)
    tareas: list[tuple[str, Path]] = []
    for est in estados:
        claves = listar_edificios(est, por_estado, cb)
        for k in claves:
            out = destino / est / k.split("/")[-1]
            if not out.exists():
                tareas.append((k, out))
    total = len(tareas)
    cb("oedi", 0.0, f"{total} ficheros por descargar de {len(estados)} estados")

    sesion = requests.Session()
    hechos = {"ok": 0, "fallo": 0}

    def bajar(t):
        clave, out = t
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = sesion.get(f"{BUCKET}/{clave}", timeout=120)
            r.raise_for_status()
            tmp = out.with_suffix(".part")
            tmp.write_bytes(r.content)
            tmp.replace(out)
            hechos["ok"] += 1
        except Exception:
            hechos["fallo"] += 1
        n = hechos["ok"] + hechos["fallo"]
        if n % 50 == 0:
            cb("oedi", 100.0 * n / max(1, total), f"{n}/{total} ficheros")

    with ThreadPoolExecutor(max_workers=hilos) as ex:
        list(ex.map(bajar, tareas))

    bajados = sorted(destino.rglob("*.parquet"))
    resumen = {"estados": estados, "por_estado": por_estado, **hechos,
               "ficheros_en_disco": len(bajados),
               "gb": round(sum(f.stat().st_size for f in bajados) / 1e9, 2)}
    (destino / "_DOWNLOAD_OK.json").write_text(
        json.dumps({"key": "oedi_comstock", "method": "s3-http", "kind": "consumption",
                    **resumen}, ensure_ascii=False, indent=2), encoding="utf-8")
    cb("oedi", 100.0, f"{resumen['ficheros_en_disco']} edificios · {resumen['gb']} GB")
    return resumen


def preprocesar(raw: Path, out: Path, cb: ProgressCB | None = None,
                freq: str = "1h", max_edificios: int = 4000) -> dict[str, Any]:
    """De los parquet por edificio al producto canonico de consumo.

    Se agrega a hora para poder compararlo de igual a igual con BDG2 y Low Carbon
    London: si un dataset va a 15 min y otro a hora, cualquier diferencia de error
    puede ser de la resolucion y no del modelo.

    Ademas del total se guarda el reparto medio por uso final de cada edificio, que
    es la informacion que ni BDG2 ni LCL tienen y la que dice donde actuar.
    """
    import pandas as pd

    cb = cb or (lambda k, p, m: None)
    ficheros = sorted(raw.rglob("*.parquet"))[:max_edificios]
    if not ficheros:
        raise FileNotFoundError(f"no hay parquet en {raw}")

    import pyarrow.parquet as pq

    # Se leen SOLO las columnas necesarias. Cada parquet trae ~50 series en float64:
    # cargarlo entero y encima copiarlo con sort_index() pide ~27 MB por edificio y
    # agota la RAM mucho antes de llegar a los 4.200. Con la seleccion de columnas
    # baja a menos de 1 MB por edificio.
    esquema = pq.ParquetFile(ficheros[0]).schema_arrow.names
    tcol = next((c for c in esquema if "timestamp" in c.lower()), None)
    col_total = next((c for c in esquema if c.startswith(_TOTAL)), None)
    cols_uso = [c for c in esquema
                if c.startswith(_PREFIJO_USO) and not c.startswith(_TOTAL)]
    if tcol is None or col_total is None:
        raise RuntimeError(f"OEDI: el parquet no trae '{_TOTAL}' ni marca de tiempo")

    series, nombres, usos = [], [], []
    idx_ref = None
    for i, f in enumerate(ficheros):
        if i % 200 == 0:
            cb("oedi", 100.0 * i / len(ficheros), f"{i}/{len(ficheros)} edificios")
        try:
            df = pd.read_parquet(f, columns=[tcol, col_total] + cols_uso)
        except Exception:
            continue
        df = df.set_index(pd.to_datetime(df.pop(tcol)))
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        serie = df[col_total].astype("float32").resample(freq).sum()
        if idx_ref is None:
            idx_ref = serie.index
        elif len(serie) != len(idx_ref):
            serie = serie.reindex(idx_ref)
        if float(serie.isna().mean()) > 0.1:
            continue
        series.append(serie.to_numpy(dtype=np.float32))
        nombres.append(f"{f.parent.name}_{f.stem}")
        # reparto por uso final (fraccion del total): es la informacion que dice
        # DONDE actuar, y la que ni BDG2 ni Low Carbon London tienen.
        finales = {c.split(".")[2]: float(df[c].to_numpy(dtype=np.float32).sum())
                   for c in cols_uso}
        tot = sum(finales.values()) or 1.0
        usos.append({k: round(v / tot, 4) for k, v in sorted(
            finales.items(), key=lambda kv: -kv[1])[:8]})
        del df

    if not series:
        raise RuntimeError("OEDI: ningun edificio utilizable")

    from .consumption import _epoch, _save, _TIME_FEATURES, calendar_features

    y = np.nan_to_num(np.stack(series))
    arrays = {"y": y, "time_feats": calendar_features(idx_ref),
              "weather": np.zeros((1, y.shape[1], 0), dtype=np.float32),
              "site_of_serie": np.zeros(len(nombres), dtype=np.int32),
              "timestamps": _epoch(idx_ref)}
    meta = {"key": "oedi_comstock", "product": "consumo", "frecuencia": freq,
            "unidad": "kWh", "n_series": len(nombres), "n_pasos": int(y.shape[1]),
            "series": nombres, "sitios": ["eeuu"],
            "time_features": list(_TIME_FEATURES), "weather_features": [],
            "usos_finales_muestra": usos[:20],
            "nota": "ComStock (terciario simulado con meteorologia real 2018). Trae "
                    "desglose por uso final, que BDG2 y LCL no tienen.",
            "periodo": [str(idx_ref[0]), str(idx_ref[-1])]}
    _save(out, "oedi_comstock", "consumo", arrays, meta)
    return meta
