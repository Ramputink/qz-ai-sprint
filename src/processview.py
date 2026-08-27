"""ProcessView — panel de estado en vivo del sprint.

Escribe dos ficheros que se actualizan durante toda la ejecución:
  * processview/status.json  → estado legible por máquina (lo lee analyze/ en el Mac)
  * processview/index.html   → panel autocontenido con auto-refresco (se abre en el navegador)

El HTML NO depende de nada externo (todo embebido) y usa <meta refresh> para
recargarse solo cada N segundos: al reabrirse relee el fichero, que el entrenamiento
va reescribiendo. Funciona igual abriéndolo por file:// en el PC durante el día, o
en el Mac tras copiar la carpeta.

Sin dependencias de terceros.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ProcessView:
    def __init__(self, pv_dir: str | Path, refresh_sec: int = 10) -> None:
        self.dir = Path(pv_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.refresh_sec = int(refresh_sec)
        self.status_path = self.dir / "status.json"
        self.html_path = self.dir / "index.html"
        self._state: dict[str, Any] = {
            "run_started": datetime.now(timezone.utc).isoformat(),
            "generation": None,
            "stage": None,
            "phase_label": "arranque",
            "progress_pct": 0.0,
            "day_of_4": 0,
            "epoch": None,
            "step": None,
            "metrics": {},
            "gpu": {},
            "eta": None,
            "checkpoints": [],
            "artifacts": [],
            "log_tail": [],
            "target": {},
            "updated": None,
        }

    def update(self, **fields: Any) -> None:
        """Actualiza campos y reescribe status.json + index.html (atómico)."""
        self._state.update(fields)
        self._state["updated"] = datetime.now(timezone.utc).isoformat()
        self._atomic_write(self.status_path, json.dumps(self._state, ensure_ascii=False, indent=2))
        self._atomic_write(self.html_path, self._render_html())

    # --- helpers ---------------------------------------------------------
    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)  # atómico en el mismo volumen

    def _render_html(self) -> str:
        s = self._state
        m = s.get("metrics") or {}
        g = s.get("gpu") or {}
        t = s.get("target") or {}
        pct = float(s.get("progress_pct") or 0.0)

        def rows(d: dict[str, Any]) -> str:
            return "".join(
                f"<tr><td>{k}</td><td class='v'>{v}</td></tr>" for k, v in d.items()
            ) or "<tr><td colspan=2 class='muted'>—</td></tr>"

        ckpts = s.get("checkpoints") or []
        arts = s.get("artifacts") or []
        logtail = s.get("log_tail") or []
        ckpt_list = "".join(f"<li>{c}</li>" for c in ckpts[-8:]) or "<li class='muted'>ninguno aún</li>"
        art_list = "".join(f"<li>{a}</li>" for a in arts[-8:]) or "<li class='muted'>ninguno aún</li>"
        log_lines = "".join(f"<div class='logln'>{self._esc(l)}</div>" for l in logtail) or "<div class='muted'>—</div>"

        return f"""<!doctype html><html lang=es><head><meta charset=utf-8>
<meta http-equiv="refresh" content="{self.refresh_sec}">
<title>ProcessView — QuantumZIGMA sprint</title>
<style>
:root{{--bg:#0d1117;--card:#161b22;--ink:#e6edf3;--muted:#8b949e;--accent:#7c3aed;--ok:#3fb950;--warn:#d29922}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,Segoe UI,Roboto,sans-serif;padding:18px}}
h1{{font-size:18px;margin:0 0 4px}} .sub{{color:var(--muted);font-size:12px;margin-bottom:14px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid #21262d;border-radius:10px;padding:14px}}
.card h2{{font-size:12px;text-transform:uppercase;letter-spacing:.4px;color:var(--muted);margin:0 0 10px}}
.big{{font-size:22px;font-weight:700}} .accent{{color:var(--accent)}}
table{{width:100%;border-collapse:collapse;font-size:13px}} td{{padding:3px 0;border-bottom:1px solid #21262d}}
td.v{{text-align:right;font-variant-numeric:tabular-nums;font-weight:600}} .muted{{color:var(--muted)}}
.bar{{height:12px;background:#21262d;border-radius:6px;overflow:hidden;margin:8px 0}}
.bar>span{{display:block;height:100%;background:linear-gradient(90deg,#7c3aed,#3fb950);width:{pct:.1f}%}}
ul{{margin:0;padding-left:16px}} li{{font-size:12px;margin:2px 0}}
.logbox{{background:#0b0f14;border-radius:8px;padding:8px;max-height:180px;overflow:auto;font:11px/1.4 ui-monospace,Menlo,monospace}}
.logln{{color:#9fb0c3;white-space:pre-wrap}} .pill{{display:inline-block;background:#21262d;border-radius:20px;padding:2px 10px;font-size:12px}}
</style></head><body>
<h1>QuantumZIGMA · Sprint de entrenamiento <span class=accent>ProcessView</span></h1>
<div class=sub>Actualizado {self._esc(s.get('updated') or '')} · auto-refresco {self.refresh_sec}s · inicio {self._esc(s.get('run_started') or '')}</div>
<div class=grid>
  <div class=card><h2>Estado</h2>
    <div class=big>{self._esc(str(s.get('phase_label') or '—'))}</div>
    <div class=sub>Generación {s.get('generation') or '—'} · Etapa {self._esc(str(s.get('stage') or '—'))} · Día {s.get('day_of_4') or 0}/4</div>
    <div class=bar><span></span></div>
    <div class=sub>{pct:.1f}% de la etapa · ETA {self._esc(str(s.get('eta') or '—'))}</div>
    <div><span class=pill>época {s.get('epoch') if s.get('epoch') is not None else '—'}</span>
         <span class=pill>paso {s.get('step') if s.get('step') is not None else '—'}</span></div>
  </div>
  <div class=card><h2>Métricas</h2><table>{rows(m)}</table></div>
  <div class=card><h2>Objetivo</h2><table>{rows(t)}</table></div>
  <div class=card><h2>GPU</h2><table>{rows(g)}</table></div>
  <div class=card><h2>Checkpoints (últimos)</h2><ul>{ckpt_list}</ul></div>
  <div class=card><h2>Paquetes / artefactos</h2><ul>{art_list}</ul></div>
</div>
<div class=card style="margin-top:14px"><h2>Log (últimas líneas)</h2><div class=logbox>{log_lines}</div></div>
</body></html>"""

    @staticmethod
    def _esc(x: Any) -> str:
        return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


if __name__ == "__main__":
    # Demo rápida: genera un ProcessView de ejemplo (útil para probar en Mac).
    import tempfile
    pv = ProcessView(tempfile.mkdtemp(prefix="pv_"), refresh_sec=10)
    pv.update(generation=1, stage="1a", phase_label="Predictivo (demo)", progress_pct=42.0,
              day_of_4=1, epoch=7, step=1234,
              metrics={"loss": 0.031, "accuracy": 0.912, "lead_time_days": 11.4},
              target={"accuracy≥": 0.90, "lead_time_days≥": 10}, gpu={"util%": 98, "temp": 71, "vram_gb": 29.1},
              eta="mañana 06:00", checkpoints=["ckpt_step1000.pt", "ckpt_step1234.pt"],
              log_tail=["[demo] entrenando…", "[demo] checkpoint guardado"])
    print("ProcessView demo escrito en:", pv.html_path)
