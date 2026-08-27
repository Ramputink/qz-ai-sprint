#!/usr/bin/env python3
"""load_checkpoint.py — carga un checkpoint en el Mac (CPU) y resume su contenido.

Los metadatos (paso, métricas, generación) están SIEMPRE en el .json portable, así
que esto funciona en el Mac aunque no tengas torch. Si quieres cargar los pesos
(.pt), instala torch CPU (requirements-analyze.txt).

Uso:
    python analyze/load_checkpoint.py checkpoints/latest.json
    python analyze/load_checkpoint.py checkpoints/gen1_1a_step40_*_best.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    target = args[0] if args else "checkpoints/latest.json"
    matches = glob.glob(target)
    if not matches:
        sys.exit(f"No encuentro: {target}")
    meta_path = Path(sorted(matches)[-1])
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    print(f"\n  Checkpoint  : {meta_path.name}")
    for k in ("generation", "stage", "step", "saved", "weights_file"):
        if k in meta:
            print(f"  {k:12}: {meta[k]}")
    if meta.get("metrics"):
        print("  métricas    :")
        for k, v in meta["metrics"].items():
            print(f"     {k:16} {v}")

    weights = meta.get("weights_file")
    if weights and str(weights).endswith(".pt"):
        wp = meta_path.parent / weights
        try:
            import torch
            sd = torch.load(wp, map_location="cpu")
            n = sum(p.numel() for p in sd.values() if hasattr(p, "numel")) if isinstance(sd, dict) else "?"
            print(f"  pesos       : cargados en CPU ({n} params) desde {wp.name}")
        except Exception as e:
            print(f"  pesos       : {wp.name} (instala torch CPU para cargarlos) — {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
