#!/usr/bin/env python3
"""run.py — ORQUESTADOR ÚNICO del sprint de IA de QuantumZIGMA.

Un solo comando corre TODO de una tirada, durante 4 días (2 generaciones de 2):
descarga de datasets → preprocesado → entrenamiento → recalibración → export a edge,
con checkpoints cada ~30 min, auto-guardado por etapa y ProcessView en vivo.

Uso (en el PC Windows + RTX 5090):
    python run.py                      # corre Gen 1 (días 1-2) y Gen 2 (días 3-4)
    python run.py --resume             # retoma tras un corte, desde el último checkpoint
    python run.py --generation 2       # solo la Gen 2 (warm-start desde Gen 1)

Prueba del flujo (en el Mac, sin GPU):
    python run.py --dry-run            # simula todas las fases en segundos
    python run.py --list-data          # lista los datasets del plan y su tamaño

Análisis (en el Mac): ver carpeta analyze/.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_config(path: Path) -> dict:
    try:
        import yaml
    except Exception:
        sys.exit(
            "\nFalta PyYAML → casi seguro NO has activado el entorno virtual.\n"
            "  Windows:  .\\.venv\\Scripts\\Activate.ps1   (el prompt debe empezar por (.venv))\n"
            "  Mac:      source .venv-mac/bin/activate\n"
            "y vuelve a ejecutar. Si aún falta:  pip install -r requirements-train.txt\n"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Sprint de entrenamiento IA QuantumZIGMA (4 días, 2 generaciones, RTX 5090).",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--config", default=str(HERE / "config.yaml"), help="ruta a config.yaml")
    ap.add_argument("--resume", action="store_true", help="retoma desde el último checkpoint")
    ap.add_argument("--generation", type=int, choices=[1, 2], help="corre solo esa generación")
    ap.add_argument("--dry-run", action="store_true", help="simula sin GPU (para probar en Mac)")
    ap.add_argument("--list-data", action="store_true", help="lista los datasets del plan y sale")
    ap.add_argument("--gpu-check", action="store_true", help="solo verifica el stack de GPU y sale")
    ap.add_argument("--train-a", action="store_true",
                    help="ENTRENAMIENTO REAL de la etapa A (Tier A pequeño). Device auto (CUDA/MPS/CPU).")
    ap.add_argument("--dataset", default="skab", help="dataset de la etapa A (por defecto: skab)")
    ap.add_argument("--epochs", type=int, default=60, help="épocas")
    ap.add_argument("--device", choices=["cuda", "mps", "cpu"], help="forzar dispositivo")
    ap.add_argument("--train-rtf", action="store_true",
                    help="RUN-TO-FAILURE real (MetroPT-3) a máxima carga: RUL + lead-time + búsqueda HP.")
    ap.add_argument("--trials", type=int, default=30, help="nº de entrenamientos en la búsqueda HP (máxima carga)")
    ap.add_argument("--cross-validate", metavar="DATASET",
                    help="validación cruzada leave-one-out sobre un dataset run-to-failure "
                         "(un modelo por máquina). Obligatorio cuando hay pocas máquinas: "
                         "con 12 rodamientos, reservar el 20%% da un número que depende del sorteo.")
    ap.add_argument("--cv-only-failed", action="store_true",
                    help="con --cross-validate: usar solo las máquinas que realmente rompieron")
    ap.add_argument("--cv-calls", type=int, default=40, help="bloques de entrenamiento por pliegue")
    ap.add_argument("--cv-group", metavar="CAMPO",
                    help="agrupar los pliegues por un campo del meta (en IMS: 'test'). "
                         "Reserva de golpe las máquinas que comparten banco de ensayo, que "
                         "comparten eje e instante de fallo: sin esto hay fuga entre ellas.")
    ap.add_argument("--train-consumo", action="store_true",
                    help="CONSUMO ELÉCTRICO: previsión de carga + detección de desperdicio + "
                         "línea base contrafactual de ahorro (+ NILM). Es el bloque que "
                         "responde a si se puede optimizar el consumo.")
    ap.add_argument("--consumo-dataset", default="building_data_genome_2",
                    help="dataset de consumo (building_data_genome_2 | "
                         "electricity_load_diagrams | steel_industry_energy)")
    ap.add_argument("--consumo-pasos", type=int, default=3000, help="pasos de entrenamiento")
    ap.add_argument("--sin-nilm", action="store_true", help="omitir la desagregación NILM")
    ap.add_argument("--consumo-perdida", default="huber", choices=["huber", "l1", "l2"],
                    help="pérdida del previsor. L1 optimiza la mediana y empeora el CV(RMSE), "
                         "que es lo que decide la acreditación ASHRAE G14")
    ap.add_argument("--feature-slice", metavar="A:B",
                    help="usar solo las columnas de features [A,B) — para ablaciones")
    a = ap.parse_args(argv)

    cfg = load_config(Path(a.config))
    if a.dry_run:
        cfg["run"]["dry_run"] = True

    if a.gpu_check:
        from src.gpu_check import check
        import json
        print(json.dumps(check(strict=False), ensure_ascii=False, indent=2))
        return 0

    if a.list_data:
        from src.data.registry import resolve, total_gb
        keys = [k for lst in (cfg.get("datasets") or {}).values() for k in (lst or [])]
        specs = resolve(keys)
        for s in specs:
            print(f"  {s.key:24} {s.method:6} ~{s.gb} GB  {s.location[:70]}")
        print(f"\nTotal plan: ~{total_gb(specs)} GB · {len(specs)} datasets")
        return 0

    if a.train_a:
        # Entrenamiento REAL de la etapa A (no la simulación por tiempo).
        from src.logging_utils import RunLogger
        from src.processview import ProcessView
        from src.checkpointing import CheckpointManager
        from src.stage_a import run_stage_a
        p = cfg["paths"]
        logger = RunLogger(HERE / p["logs_dir"])
        pv = ProcessView(HERE / p["processview_dir"], refresh_sec=cfg["run"].get("processview_refresh_sec", 10))
        ckpt = CheckpointManager(HERE / p["checkpoints_dir"], keep_last=6)
        m = run_stage_a(cfg, HERE, logger, pv, ckpt, dataset=a.dataset, epochs=a.epochs, device=a.device)
        print("\n=== RESULTADO ETAPA A ===")
        print(f"  device      : {m['device']} ({m.get('hardware')})")
        print(f"  accuracy    : {m['accuracy']}")
        print(f"  confusión   : {m['confusion']}")
        print(f"  FN / FP     : {m['false_negatives']} / {m['false_positives']}")
        print(f"  PR-AUC      : {m.get('pr_auc')}   ROC-AUC: {m.get('roc_auc')}")
        print(f"  tiempo      : {m['train_seconds']} s")
        print(f"  guardado en : artifacts/stage_a_result_{m['device']}.json")
        return 0

    if a.train_rtf:
        from src.logging_utils import RunLogger
        from src.processview import ProcessView
        from src.checkpointing import CheckpointManager
        from src.stage_rtf import run_stage_rtf
        p = cfg["paths"]
        logger = RunLogger(HERE / p["logs_dir"])
        pv = ProcessView(HERE / p["processview_dir"], refresh_sec=cfg["run"].get("processview_refresh_sec", 10))
        ckpt = CheckpointManager(HERE / p["checkpoints_dir"], keep_last=6)
        m = run_stage_rtf(cfg, HERE, logger, pv, ckpt, n_trials=a.trials, epochs=a.epochs, device=a.device)
        print("\n=== RESULTADO RUN-TO-FAILURE (MetroPT-3) ===")
        print(f"  device        : {m['device']} ({m.get('hardware')})")
        print(f"  datos         : {m['raw_rows']:,} filas → {m['windows']} ventanas")
        print(f"  accuracy      : {m['accuracy']}  (balanced {m['balanced_accuracy']})")
        print(f"  confusión     : {m['confusion']}   FN/FP(ventana): {m['false_negatives']}/{m['false_positives']}")
        print(f"  FALSAS ALARMAS: {m['false_alarm_events']} episodios (operacional, tras antirrebote)")
        print(f"  punto operación: umbral {m['operating_point']['threshold']} · {m['operating_point']['min_consecutive_windows']} ventanas consecutivas")
        print(f"  PR-AUC/ROC    : {m.get('pr_auc')} / {m.get('roc_auc')}")
        print(f"  fallos detect.: {m['failures_detected']}   lead-time medio: {m['mean_lead_days']} días")
        for l in m["lead_time"]:
            print(f"     fallo {l['onset']} → alarma {l['alarm']}  ({l['lead_days']} días antes)")
        print(f"  trials HP     : {m['n_trials']}   tiempo: {m['train_seconds']} s")
        print(f"  guardado en   : artifacts/rtf_result_{m['device']}.json")
        return 0

    if a.train_consumo:
        from src.consumo import correr_todo
        from src.logging_utils import RunLogger
        from src.processview import ProcessView
        p = cfg["paths"]
        logger = RunLogger(HERE / p["logs_dir"])
        pv = ProcessView(HERE / p["processview_dir"],
                         refresh_sec=cfg["run"].get("processview_refresh_sec", 10))
        inf = correr_todo(cfg, HERE, logger, pv, dataset=a.consumo_dataset,
                          pasos=a.consumo_pasos, nilm=not a.sin_nilm,
                          perdida=a.consumo_perdida)
        f = inf["prevision"]
        print()
        print("=== CONSUMO ELÉCTRICO ===")
        print(f"  dataset      : {f['dataset']}  ({f['series']} series)")
        print(f"  previsión {f['horizonte_h']}h : MAE {f['mae']}  ·  base ingenua {f['mae_base_ingenua']}")
        print(f"  SKILL vs base: {f['skill_vs_ingenua']:+.4f}   <- 0 = empatar con no hacer nada")
        print(f"  CV(RMSE)     : {f['cv_rmse_pct']} %   ASHRAE G14 (<25%): {f['cumple_ashrae_g14']}")
        d = inf["desperdicio"]
        print(f"  desperdicio  : {d['pct_ventanas_con_exceso']} % de ventanas con exceso sostenido")
        b = inf["linea_base_ahorro"]
        print(f"  línea base   : ahorro aparente {b['ahorro_aparente_pct']} % (sin intervención "
              f"debería rondar 0) · NMBE {b['nmbe_pct']} %")
        if isinstance(inf.get("nilm"), dict) and "total" in inf["nilm"]:
            n = inf["nilm"]["total"]
            print(f"  NILM         : MAE {n['mae']} W  ·  skill vs media {n['skill_vs_ingenua']:+.4f}")
        print(f"  informe      : {inf['_fichero']}")
        return 0

    if a.cross_validate:
        from src.crossval import leave_one_unit_out, print_report, save_report
        from src.logging_utils import RunLogger
        if a.feature_slice:
            lo, hi = a.feature_slice.split(":")
            cfg.setdefault("train", {})["feature_slice"] = [int(lo), int(hi)]
        logger = RunLogger(HERE / cfg["paths"]["logs_dir"])
        summary = leave_one_unit_out(cfg, HERE, a.cross_validate, logger,
                                     max_calls=a.cv_calls, only_failed=a.cv_only_failed,
                                     group_by=a.cv_group)
        print_report(summary)
        tag = ("solorotos" if a.cv_only_failed else "") + (f"g{a.cv_group}" if a.cv_group else "")
        if a.feature_slice:
            tag += "_f" + a.feature_slice.replace(":", "-")
        print("\n  informe:", save_report(summary, HERE, cfg, tag))
        return 0

    from src.orchestrator import Orchestrator
    orch = Orchestrator(cfg, base_dir=HERE)
    orch.run(resume=a.resume, only_generation=a.generation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
