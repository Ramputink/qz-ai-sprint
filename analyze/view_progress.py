#!/usr/bin/env python3
"""view_progress.py — abre en el Mac un paquete recibido del PC y muestra el estado.

Uso:
    python analyze/view_progress.py                         # lee ./processview/status.json
    python analyze/view_progress.py artifacts/gen1_etapa1a_*.zip   # abre un paquete recibido
    python analyze/view_progress.py --open                  # además abre el HTML en el navegador

Solo stdlib → corre en cualquier Mac sin instalar nada.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
import webbrowser
import zipfile
from pathlib import Path


def _print_status(status: dict) -> None:
    print(f"\n  Estado      : {status.get('phase_label')}  ({status.get('progress_pct')}%)")
    print(f"  Generación  : {status.get('generation')}  · etapa {status.get('stage')}  · día {status.get('day_of_4')}/4")
    print(f"  Época/paso  : {status.get('epoch')} / {status.get('step')}   ETA {status.get('eta')}")
    m = status.get("metrics") or {}
    if m:
        print("  Métricas    :")
        for k, v in m.items():
            print(f"     {k:16} {v}")
    t = status.get("target") or {}
    if t:
        print("  Objetivo    : " + ", ".join(f"{k}={v}" for k, v in t.items()))
    ck = status.get("checkpoints") or []
    print(f"  Checkpoints : {len(ck)} (últimos: {', '.join(ck[-3:]) if ck else '—'})")


def from_zip(zpath: str) -> tuple[dict | None, Path | None]:
    tmp = Path(tempfile.mkdtemp(prefix="qzview_"))
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(tmp)
    status = None
    html = None
    for p in tmp.rglob("status.json"):
        status = json.loads(p.read_text(encoding="utf-8")); break
    for p in tmp.rglob("index.html"):
        html = p; break
    mani = tmp / "MANIFEST.json"
    if mani.exists():
        print("  Manifiesto  :", json.dumps(json.loads(mani.read_text()), ensure_ascii=False))
    return status, html


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Ver el progreso del sprint en el Mac")
    ap.add_argument("target", nargs="?", help="ruta a un .zip de artefacto (o vacío para ./processview)")
    ap.add_argument("--open", action="store_true", help="abre el dashboard HTML en el navegador")
    a = ap.parse_args(argv)

    html = None
    if a.target:
        matches = glob.glob(a.target)
        if not matches:
            sys.exit(f"No encuentro: {a.target}")
        status, html = from_zip(sorted(matches)[-1])
    else:
        sp = Path("processview/status.json")
        if not sp.exists():
            sys.exit("No hay processview/status.json. Pasa un .zip de artefacto o corre el sprint.")
        status = json.loads(sp.read_text(encoding="utf-8"))
        hp = Path("processview/index.html")
        html = hp if hp.exists() else None

    if status:
        _print_status(status)
    if a.open and html:
        webbrowser.open(html.resolve().as_uri())
        print(f"\n  Abriendo {html} en el navegador…")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
