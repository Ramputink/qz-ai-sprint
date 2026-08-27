#!/usr/bin/env python3
"""compare.py — compara los resultados de la etapa A entre dispositivos (Mac MPS
vs Windows CUDA vs CPU).

Cada ejecución guarda artifacts/stage_a_result_<device>.json. Copia el del Windows
al Mac (o al revés) y ejecuta esto para verlos lado a lado.

Uso:
    python analyze/compare.py                     # busca todos los stage_a_result_*.json
    python analyze/compare.py a.json b.json       # compara dos ficheros concretos
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def _load(paths):
    out = []
    for p in paths:
        try:
            out.append((Path(p).name, json.loads(Path(p).read_text(encoding="utf-8"))))
        except Exception as e:
            print(f"  (no pude leer {p}: {e})")
    return out


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    files = args or sorted(glob.glob("artifacts/stage_a_result_*.json"))
    if not files:
        sys.exit("No hay resultados. Corre  python run.py --train-a  en cada máquina primero.")
    rows = _load(files)
    if not rows:
        return 1

    cols = ["device", "hardware", "dataset", "accuracy", "balanced_accuracy",
            "false_negatives", "false_positives", "pr_auc", "roc_auc", "train_seconds"]
    w = 18
    print("\n" + "métrica".ljust(22) + "".join(name[:w].ljust(w + 2) for name, _ in rows))
    print("-" * (22 + (w + 2) * len(rows)))
    for c in cols:
        line = c.ljust(22)
        for _, d in rows:
            v = d.get(c)
            if isinstance(v, float):
                v = round(v, 4)
            line += str(v).ljust(w + 2)
        print(line)

    # veredicto rápido de consistencia
    accs = [d.get("accuracy") for _, d in rows if d.get("accuracy") is not None]
    if len(accs) >= 2:
        spread = max(accs) - min(accs)
        print(f"\n  Diferencia de accuracy entre dispositivos: {spread:.4f}")
        print("  " + ("✓ resultados consistentes (misma lógica, distinto hardware)"
                      if spread < 0.05 else
                      "⚠ diferencia notable — revisa seeds/épocas/versión de torch"))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
