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
        print(f"  confusión     : {m['confusion']}   FN/FP: {m['false_negatives']}/{m['false_positives']}")
        print(f"  PR-AUC/ROC    : {m.get('pr_auc')} / {m.get('roc_auc')}")
        print(f"  fallos detect.: {m['failures_detected']}   lead-time medio: {m['mean_lead_days']} días")
        for l in m["lead_time"]:
            print(f"     fallo {l['onset']} → alarma {l['alarm']}  ({l['lead_days']} días antes)")
        print(f"  trials HP     : {m['n_trials']}   tiempo: {m['train_seconds']} s")
        print(f"  guardado en   : artifacts/rtf_result_{m['device']}.json")
        return 0

    from src.orchestrator import Orchestrator
    orch = Orchestrator(cfg, base_dir=HERE)
    orch.run(resume=a.resume, only_generation=a.generation)
    return 0


if __name__ == "__main__":
    sys.exit(main())
