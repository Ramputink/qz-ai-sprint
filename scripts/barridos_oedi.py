"""Barridos sobre OEDI, para lanzar cuando la GPU quede libre (scripts/).

Corre los MISMOS tres barridos que se hicieron sobre BDG2, para poder comparar de
igual a igual:
  1. capacidad     -> saber si el modelo esta limitado por parametros
  2. preprocesado  -> que transformaciones aportan
  3. escala        -> LA PREGUNTA NUEVA: si el modelo esta limitado por DATOS.
     En BDG2 el barrido de capacidad mostro que no lo limita el tamano del modelo
     (de 0,9 a 25,7 M de parametros, el % de series acreditables no se movio del
     72-73 %). Queda por saber si lo limita el numero de edificios, y para eso hay
     que entrenar el mismo modelo con 100, 400, 1600 y todos los edificios.
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.ablacion import cabecera, correr  # noqa: E402
from src.consumo import TareaPrevision, entrenar_previsor  # noqa: E402
from src.logging_utils import RunLogger  # noqa: E402

DATASET = "oedi_comstock"
DATA = BASE / "data"
ART = BASE / "artifacts"
log = RunLogger(BASE / "logs")


def barrido_capacidad(pasos=6000):
    print("\n=== OEDI · CAPACIDAD ===", flush=True)
    print(f"{'ancho':>6}{'bloq':>6}{'params':>11}{'MAEtest':>9}{'MAEtrain':>10}"
          f"{'brecha':>8}{'skill':>9}{'CVmed':>8}{'%acred':>8}{'seg':>6}", flush=True)
    filas = []
    for ancho, bloq in [(256, 3), (512, 3), (512, 12), (1024, 6)]:
        tarea = TareaPrevision(DATA, DATASET)
        torch.manual_seed(20260827); np.random.seed(20260827)
        try:
            m = entrenar_previsor(tarea, pasos=pasos, perdida="l1", ancho=ancho,
                                  bloques=bloq, logger=log)["metricas"]
        except torch.OutOfMemoryError:
            print(f"{ancho:>6}{bloq:>6}   -- sin memoria --", flush=True)
            del tarea; torch.cuda.empty_cache(); continue
        ps = m.get("por_serie", {})
        filas.append(m)
        print(f"{ancho:>6}{bloq:>6}{m['parametros']:>11,}{m['mae']:>9.3f}"
              f"{m['mae_train']:>10.3f}{m['brecha_train_test']:>8.3f}"
              f"{m['skill_vs_ingenua']:>+9.4f}{ps.get('cv_rmse_mediana_pct',0):>7.2f}%"
              f"{ps.get('pct_series_acreditables',0):>7.1f}%{m['segundos']:>6.0f}", flush=True)
        del tarea; torch.cuda.empty_cache()
    (ART / "oedi_capacidad.json").write_text(
        json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")


def barrido_escala(pasos=6000):
    """Curva de escala: mismo modelo, cada vez mas edificios."""
    print("\n=== OEDI · ESCALA (¿limita el numero de edificios?) ===", flush=True)
    print(f"{'series':>8}{'MAEtest':>9}{'MAEtrain':>10}{'brecha':>8}{'skill':>9}"
          f"{'CVmed':>8}{'%acred':>8}{'seg':>6}", flush=True)
    filas = []
    for n in (100, 400, 1600, 4000):
        try:
            tarea = TareaPrevision(DATA, DATASET, max_series=n)
        except Exception as e:
            print(f"{n:>8}  -- {type(e).__name__} --", flush=True)
            continue
        torch.manual_seed(20260827); np.random.seed(20260827)
        try:
            m = entrenar_previsor(tarea, pasos=pasos, perdida="l1", logger=log)["metricas"]
        except torch.OutOfMemoryError:
            print(f"{n:>8}  -- sin memoria --", flush=True)
            del tarea; torch.cuda.empty_cache(); continue
        ps = m.get("por_serie", {})
        filas.append(m)
        print(f"{tarea.n_series:>8}{m['mae']:>9.3f}{m['mae_train']:>10.3f}"
              f"{m['brecha_train_test']:>8.3f}{m['skill_vs_ingenua']:>+9.4f}"
              f"{ps.get('cv_rmse_mediana_pct',0):>7.2f}%"
              f"{ps.get('pct_series_acreditables',0):>7.1f}%{m['segundos']:>6.0f}", flush=True)
        del tarea; torch.cuda.empty_cache()
    (ART / "oedi_escala.json").write_text(
        json.dumps(filas, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    log.info("oedi_barridos_start", dataset=DATASET)
    barrido_escala()
    barrido_capacidad()
    print("\n=== OEDI · PREPROCESADO ===", flush=True)
    print(cabecera(), flush=True)
    correr(DATA, DATASET, log, pasos=6000, salida=ART / "oedi_ablacion_preproceso.json")
    print(f"\nTOTAL {(time.time()-t0)/60:.1f} min", flush=True)
    log.info("oedi_barridos_done", minutos=round((time.time() - t0) / 60, 1))
