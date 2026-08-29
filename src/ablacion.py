"""Ablacion de preprocesados y de tipos de entrenamiento.

Por que existe: en esta maquina un entrenamiento completo de 6.000 pasos son ~70
segundos. A ese coste, discutir que preprocesado sera mejor cuesta mas que
probarlos todos. Este modulo convierte esas decisiones en medidas.

Cada variante cambia UNA cosa respecto a la referencia y se mide con el mismo
dato, la misma semilla y los mismos pasos. Todo se reporta contra la base ingenua
(misma hora, semana pasada) y con el CV(RMSE) POR EMPLAZAMIENTO, que es como
ASHRAE Guideline 14 decide si una linea base de ahorro es acreditable.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

# (nombre, kwargs de la tarea, kwargs del entrenamiento)
VARIANTES: list[tuple[str, dict, dict]] = [
    ("referencia",            {}, {}),
    ("log1p",                 {"log1p": True}, {}),
    ("normalizacion robusta", {"norma": "robusta"}, {}),
    ("sin meteorologia",      {"usar_meteo": False}, {}),
    ("contexto 336 h",        {"contexto": 336}, {}),
    ("contexto 720 h",        {"contexto": 720}, {}),
    ("horizonte 168 h",       {"horizonte": 168}, {}),
    ("log1p + robusta",       {"log1p": True, "norma": "robusta"}, {}),
]


def correr(data_dir: Path, dataset: str, logger, pasos: int = 6000,
           seed: int = 20260827, variantes=None, salida: Path | None = None) -> list[dict[str, Any]]:
    """Entrena una variante por fila y devuelve la tabla comparativa."""
    import torch

    from .consumo import TareaPrevision, entrenar_previsor

    filas: list[dict[str, Any]] = []
    for nombre, kt, ke in (variantes or VARIANTES):
        # NO se cachean las tareas entre variantes: cada una reserva su copia del
        # dataset en la GPU, y acumularlas agota la memoria a la tercera o cuarta
        # variante. Reconstruirla cuesta segundos; quedarse sin VRAM cuesta el barrido.
        tarea = TareaPrevision(data_dir, dataset, **kt)
        torch.manual_seed(seed)
        np.random.seed(seed)
        t0 = time.time()
        try:
            m = entrenar_previsor(tarea, pasos=pasos, perdida="l1", logger=logger,
                                  **ke)["metricas"]
        except torch.OutOfMemoryError as e:
            logger.warn("ablacion_sin_memoria", variante=nombre, error=str(e)[:120])
            filas.append({"variante": nombre, "error": "sin memoria en GPU"})
            del tarea
            torch.cuda.empty_cache()
            continue
        ps = m.get("por_serie", {})
        filas.append({"variante": nombre, "mae": m["mae"], "mae_train": m["mae_train"],
                      "brecha": m["brecha_train_test"], "skill": m["skill_vs_ingenua"],
                      "cv_rmse_mediana_pct": ps.get("cv_rmse_mediana_pct"),
                      "pct_series_acreditables": ps.get("pct_series_acreditables"),
                      "contexto_h": m["contexto_h"], "horizonte_h": m["horizonte_h"],
                      "segundos": round(time.time() - t0, 1)})
        logger.info("ablacion", **filas[-1])
        print(_fila(filas[-1]), flush=True)
        del tarea
        torch.cuda.empty_cache()
    if salida:
        salida.write_text(json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")
    return filas


def _fila(f: dict[str, Any]) -> str:
    if "error" in f:
        return f"  {f['variante']:24}  -- {f['error']} --"
    return (f"  {f['variante']:24}{f['mae']:>9.3f}{f['mae_train']:>10.3f}{f['brecha']:>8.3f}"
            f"{f['skill']:>+9.4f}{(f['cv_rmse_mediana_pct'] or 0):>8.2f}%"
            f"{(f['pct_series_acreditables'] or 0):>8.1f}%{f['segundos']:>7.0f}")


def cabecera() -> str:
    return (f"  {'variante':24}{'MAEtest':>9}{'MAEtrain':>10}{'brecha':>8}"
            f"{'skill':>9}{'CVmed':>9}{'%acred':>8}{'seg':>7}")
