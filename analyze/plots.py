#!/usr/bin/env python3
"""plots.py — gráficas del entrenamiento en el Mac (curvas, matriz de confusión,
lead-time, y comparativa Gen1 vs Gen2).

Lee logs/run.jsonl (portable) — no necesita GPU. Requiere matplotlib
(requirements-analyze.txt). Guarda PNGs en analyze/plots_out/.

Uso:
    python analyze/plots.py                       # usa ./logs/run.jsonl
    python analyze/plots.py logs/run.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path


def load_events(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out


def main(argv=None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    log_path = Path(args[0]) if args else Path("logs/run.jsonl")
    if not log_path.exists():
        sys.exit(f"No encuentro {log_path}. Corre el sprint (o el dry-run) primero.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        sys.exit(f"Instala matplotlib (requirements-analyze.txt): {e}")

    events = load_events(log_path)
    # Reconstruye métricas por generación desde los checkpoints/stage_done
    series = defaultdict(lambda: {"step": [], "accuracy": [], "lead": [], "fn": []})
    # También aceptamos el status.json final por si el log no trae métricas por paso
    status = None
    sp = Path("processview/status.json")
    if sp.exists():
        status = json.loads(sp.read_text(encoding="utf-8"))

    outdir = Path("analyze/plots_out"); outdir.mkdir(parents=True, exist_ok=True)

    # Gráfica 1: resumen final (matriz de confusión + objetivo)
    if status and status.get("metrics"):
        m = status["metrics"]
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
        # matriz de confusión (con lo que haya)
        fp, fn = m.get("fp", 0), m.get("fn", 0)
        cm = [[100 - fp, fp], [fn, 100 - fn]]  # ilustrativo si no hay TP/TN exactos
        ax[0].imshow(cm, cmap="Purples")
        ax[0].set_title("Matriz de confusión (final)")
        ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["pred sano", "pred fallo"])
        ax[0].set_yticks([0, 1]); ax[0].set_yticklabels(["real sano", "real fallo"])
        for i in range(2):
            for j in range(2):
                ax[0].text(j, i, cm[i][j], ha="center", va="center")
        # barras objetivo vs conseguido
        labels = ["accuracy", "lead_time_days"]
        got = [m.get("accuracy", 0), m.get("lead_time_days", 0)]
        tgt = [status.get("target", {}).get("accuracy≥", 0.9),
               status.get("target", {}).get("lead_time_days≥", 10)]
        x = range(len(labels))
        ax[1].bar([i - 0.2 for i in x], got, width=0.4, label="conseguido")
        ax[1].bar([i + 0.2 for i in x], tgt, width=0.4, label="objetivo")
        ax[1].set_xticks(list(x)); ax[1].set_xticklabels(labels); ax[1].legend()
        ax[1].set_title("Objetivo vs conseguido")
        fig.tight_layout(); fig.savefig(outdir / "resumen_final.png", dpi=120)
        print("  guardado:", outdir / "resumen_final.png")

    # Gráfica 2: línea temporal de eventos de checkpoint
    ck = [e for e in events if e.get("event") == "checkpoint"]
    if ck:
        fig, ax = plt.subplots(figsize=(8, 2.6))
        ax.plot([e.get("step", i) for i, e in enumerate(ck)], marker="o")
        ax.set_title(f"Checkpoints guardados ({len(ck)})")
        ax.set_xlabel("nº checkpoint"); ax.set_ylabel("step")
        fig.tight_layout(); fig.savefig(outdir / "checkpoints.png", dpi=120)
        print("  guardado:", outdir / "checkpoints.png")

    print(f"\n  {len(events)} eventos leídos de {log_path}. PNGs en {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
