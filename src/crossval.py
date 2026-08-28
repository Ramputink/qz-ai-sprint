"""Validacion cruzada dejando una maquina fuera (leave-one-unit-out).

POR QUE EXISTE ESTE MODULO
--------------------------
Con pocas maquinas, reservar el 20 % da uno o dos equipos de validacion y el
resultado depende de cuales toquen. En IMS (12 rodamientos, 2 de validacion) la
accuracy medida oscilaba entre 0,34 y 0,77 cambiando solo la semilla, y algun
sorteo ni siquiera era evaluable. Con esas cifras no se puede decidir nada.

Aqui se entrena un modelo por maquina, reservando esa y solo esa. Se obtienen
tantas estimaciones como maquinas, y lo que se reporta es la DISTRIBUCION: mediana,
rango y cuantos pliegues cumplen el objetivo. Un pliegue bueno ya no puede pasar por
el resultado del dataset.

Pliegues degenerados: si la maquina reservada es mas corta que el horizonte de aviso
(en IMS el 2o ensayo dura 164 h y el objetivo son 240), todas sus ventanas caen del
mismo lado y no hay clase negativa. Ese pliegue NO se puede evaluar y se reporta
como tal, en vez de colar un numero sin sentido.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _fold_metrics(cfg: dict[str, Any], base_dir: Path, dataset: str, units: list[int],
                  logger, max_calls: int) -> dict[str, Any]:
    """Entrena un pliegue reservando `units` y devuelve sus mejores metricas."""
    from .trainer import StageTrainer

    fold_cfg = json.loads(json.dumps(cfg))
    fold_cfg.setdefault("train", {})
    fold_cfg["train"]["primary_dataset"] = dataset
    fold_cfg["train"]["val_units"] = [int(u) for u in units]
    fold_cfg["train"]["torch_compile"] = False
    unit = units[0]

    trainer = StageTrainer(fold_cfg, base_dir, 1, "1a", logger)
    trainer.setup()
    best = None
    for i in range(max_calls):
        m = trainer.step(i)
        if "coste" in m and (best is None or m["coste"] < best["coste"]):
            best = m
        if trainer.stop:
            break
    if best is None:
        return {"unit": int(unit), "units": [int(u) for u in units], "evaluable": False,
                "motivo": "el horizonte de aviso no parte las ventanas reservadas "
                          "(duran menos que el aviso exigido): sin clase negativa"}
    return {"unit": int(unit), "units": [int(u) for u in units], "evaluable": True,
            **{k: best[k] for k in ("accuracy", "tp", "tn", "fp", "fn", "recall_fallo",
                                    "precision_fallo", "coste", "mae_rul",
                                    "pr_auc", "pr_auc_base", "roc_auc",
                                    "lead_time_days", "lead_time_days_min",
                                    "maquinas_sin_aviso") if k in best}}


def leave_one_unit_out(cfg: dict[str, Any], base_dir: Path, dataset: str, logger,
                       max_calls: int = 40, only_failed: bool = False,
                       group_by: str | None = None) -> dict[str, Any]:
    """Un modelo por maquina (o por GRUPO de maquinas). Devuelve la distribucion.

    `group_by` reserva de golpe todas las maquinas que comparten un atributo del meta
    -- en IMS, `test`: los 4 rodamientos del mismo banco.

    POR QUE HACE FALTA AGRUPAR: los 4 rodamientos de un ensayo IMS van en el mismo eje,
    con la misma carga, y el ensayo se detiene para todos en el mismo instante. Si se
    reserva uno solo, los otros tres estan en entrenamiento con ESE MISMO instante de
    fallo y una vibracion correlacionada a traves de la carcasa. El modelo puede
    acertar el momento sin haber aprendido nada de la degradacion del rodamiento
    reservado. Dejar fuera el ensayo entero elimina esa fuga.
    """
    from .data.preprocess import load_product

    data_dir = Path(base_dir) / cfg["paths"]["data_dir"]
    arrays, meta = load_product(data_dir, dataset, "rul")
    units = sorted(int(u) for u in np.unique(arrays["unit"]))
    if only_failed:
        failed = set(int(u) for u in arrays.get("failed_units", np.array([])).tolist())
        units = [u for u in units if u in failed]
        if not units:
            raise RuntimeError(f"{dataset}: no hay maquinas marcadas como rotas")

    info_by_unit = {m.get("unit"): m for m in meta.get("units", [])}

    def describe(us: list[int]) -> str:
        infos = [info_by_unit.get(u, {}) for u in us]
        if group_by and infos and infos[0].get(group_by):
            roto = sum(1 for i in infos if i.get("failed"))
            return f"{infos[0][group_by]} ({len(us)} rod., {roto} roto/s)"
        i = infos[0] if infos else {}
        return (f"{i.get('test', '?')} rod.{i.get('bearing', '?')}"
                + (" (rompio)" if i.get("failed") else "")) if i else str(us)

    if group_by:
        groups: dict[str, list[int]] = {}
        for u in units:
            groups.setdefault(str(info_by_unit.get(u, {}).get(group_by, u)), []).append(u)
        folds_units = list(groups.values())
    else:
        folds_units = [[u] for u in units]

    logger.info("loo_start", dataset=dataset, pliegues=len(folds_units),
                agrupado_por=group_by or "maquina",
                objetivo_dias=cfg["target"]["lead_time_days"])
    folds = []
    for i, us in enumerate(folds_units):
        f = _fold_metrics(cfg, Path(base_dir), dataset, us, logger, max_calls)
        f["descripcion"] = describe(us)
        folds.append(f)
        logger.info("loo_fold", pliegue=f"{i + 1}/{len(folds_units)}", unidades=us,
                    evaluable=f["evaluable"], accuracy=f.get("accuracy"),
                    aviso_dias=f.get("lead_time_days"))

    ok = [f for f in folds if f["evaluable"]]
    min_acc = float(cfg["target"]["min_accuracy"])
    lead_target = float(cfg["target"]["lead_time_days"])
    summary: dict[str, Any] = {
        "dataset": dataset, "pliegues": len(folds), "evaluables": len(ok),
        "no_evaluables": [f["descripcion"] for f in folds if not f["evaluable"]],
    }
    if ok:
        acc = np.array([f["accuracy"] for f in ok], dtype=float)
        lead = np.array([f.get("lead_time_days", 0.0) for f in ok], dtype=float)
        fn = np.array([f.get("fn", 0) for f in ok], dtype=float)
        summary.update({
            "accuracy": {"mediana": round(float(np.median(acc)), 4),
                         "media": round(float(acc.mean()), 4),
                         "min": round(float(acc.min()), 4), "max": round(float(acc.max()), 4),
                         "pliegues_que_cumplen": int((acc >= min_acc).sum())},
            "aviso_dias": {"mediana": round(float(np.median(lead)), 2),
                           "media": round(float(lead.mean()), 2),
                           "min": round(float(lead.min()), 2), "max": round(float(lead.max()), 2),
                           "pliegues_que_cumplen": int((lead >= lead_target).sum())},
            "fn_total": int(fn.sum()),
            **_auc_summary(ok),
            "veredicto": ("cumple en la mediana" if float(np.median(acc)) >= min_acc
                          and float(np.median(lead)) >= lead_target else "NO cumple"),
        })
    summary["pliegues_detalle"] = folds
    logger.info("loo_done", dataset=dataset, evaluables=len(ok),
                veredicto=summary.get("veredicto"))
    return summary


def _auc_summary(ok: list[dict[str, Any]]) -> dict[str, Any]:
    """Resumen del AUC sobre los pliegues evaluables.

    El PR-AUC solo dice algo comparado con su linea base (la prevalencia de la clase
    fallo en ese pliegue), asi que se reporta tambien el margen sobre ella: es lo que
    de verdad mide cuanto aporta el modelo frente a alarmar al azar.
    """
    pr = [f["pr_auc"] for f in ok if f.get("pr_auc") is not None]
    roc = [f["roc_auc"] for f in ok if f.get("roc_auc") is not None]
    if not pr:
        return {}
    base = [f.get("pr_auc_base", 0.0) for f in ok if f.get("pr_auc") is not None]
    lift = [p - b for p, b in zip(pr, base)]
    return {"pr_auc": {"mediana": round(float(np.median(pr)), 4),
                       "min": round(float(np.min(pr)), 4), "max": round(float(np.max(pr)), 4),
                       "base_mediana": round(float(np.median(base)), 4),
                       "margen_sobre_base_mediana": round(float(np.median(lift)), 4)},
            "roc_auc": {"mediana": round(float(np.median(roc)), 4),
                        "min": round(float(np.min(roc)), 4),
                        "max": round(float(np.max(roc)), 4)} if roc else {}}


def save_report(summary: dict[str, Any], base_dir: Path, cfg: dict[str, Any],
                tag: str = "") -> Path:
    out = Path(base_dir) / cfg["paths"]["artifacts_dir"]
    out.mkdir(parents=True, exist_ok=True)
    name = f"crossval_{summary['dataset']}" + (f"_{tag}" if tag else "") + ".json"
    path = out / name
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def print_report(summary: dict[str, Any]) -> None:
    print(f"\n=== VALIDACION CRUZADA leave-one-out · {summary['dataset']} ===")
    print(f"  {summary['evaluables']}/{summary['pliegues']} pliegues evaluables")
    if summary["no_evaluables"]:
        print(f"  no evaluables (maquina mas corta que el aviso exigido): {summary['no_evaluables']}")
    hdr = (f"  {'maquina':24}{'accuracy':>10}{'PR-AUC':>9}{'(base)':>9}"
           f"{'ROC-AUC':>9}{'FN':>6}{'FP':>6}{'aviso d':>9}")
    print(hdr)
    for f in summary["pliegues_detalle"]:
        if not f["evaluable"]:
            print(f"  {f['descripcion']:24}{'— no evaluable —':>32}")
            continue
        print(f"  {f['descripcion']:24}{f['accuracy']:>10}{f.get('pr_auc', 0):>9}"
              f"{f.get('pr_auc_base', 0):>9}{f.get('roc_auc', 0):>9}{f.get('fn', 0):>6}"
              f"{f.get('fp', 0):>6}{f.get('lead_time_days', 0):>9}")
    if "accuracy" in summary:
        a, l = summary["accuracy"], summary["aviso_dias"]
        print(f"\n  accuracy : mediana {a['mediana']}  rango [{a['min']}, {a['max']}]"
              f"  · cumplen {a['pliegues_que_cumplen']}/{summary['evaluables']}")
        print(f"  aviso    : mediana {l['mediana']} d  rango [{l['min']}, {l['max']}]"
              f"  · cumplen {l['pliegues_que_cumplen']}/{summary['evaluables']}")
    if summary.get("pr_auc"):
        pa, ra = summary["pr_auc"], summary.get("roc_auc") or {}
        print(f"  PR-AUC   : mediana {pa['mediana']}  rango [{pa['min']}, {pa['max']}]"
              f"  · base (prevalencia) {pa['base_mediana']}"
              f"  · margen {pa['margen_sobre_base_mediana']:+}")
        if ra:
            print(f"  ROC-AUC  : mediana {ra['mediana']}  rango [{ra['min']}, {ra['max']}]")
    if "accuracy" in summary:
        print(f"  VEREDICTO: {summary['veredicto']}")
