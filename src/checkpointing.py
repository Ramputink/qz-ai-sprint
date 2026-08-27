"""Checkpointing atómico + reanudación + auto-save por etapa.

Requisitos del usuario:
  * Backup cada ~30 min (reanudable ante cortes de luz / throttle).
  * Auto-guardado tras cada etapa → paquete .zip que se envía al Mac.
  * Todo sobrevive a un corte: escritura atómica (tmp + os.replace).

Formato portable (Windows entrena → Mac analiza): el state_dict se guarda con
torch.save si torch está disponible; los METADATOS (paso, métricas, generación,
etapa) van SIEMPRE en un .json aparte, legible en el Mac sin torch ni GPU.

torch es opcional aquí para poder probar la lógica en el Mac sin GPU.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch  # noqa
        return torch
    except Exception:
        return None


def _atomic(path: Path, write_fn) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_fn(tmp)
    os.replace(tmp, path)


class CheckpointManager:
    def __init__(self, ckpt_dir: str | Path, keep_last: int = 5) -> None:
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._last_save = 0.0

    # --- guardado periódico ---------------------------------------------
    def maybe_save(self, state: dict[str, Any], meta: dict[str, Any], every_min: int) -> str | None:
        """Guarda si han pasado >= every_min minutos desde el último guardado."""
        if (time.time() - self._last_save) < every_min * 60:
            return None
        return self.save(state, meta)

    def save(self, state: dict[str, Any], meta: dict[str, Any], tag: str = "") -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        gen = meta.get("generation", "x")
        stage = meta.get("stage", "x")
        step = meta.get("step", 0)
        name = f"gen{gen}_{stage}_step{step}_{ts}" + (f"_{tag}" if tag else "")
        base = self.dir / name
        torch = _torch()
        # pesos / estado del modelo
        if torch is not None:
            _atomic(base.with_suffix(".pt"), lambda p: torch.save(state, p))
            weights_file = base.with_suffix(".pt").name
        else:
            # fallback sin torch (solo para pruebas de la lógica en Mac)
            _atomic(base.with_suffix(".pkl"), lambda p: p.write_bytes(pickle.dumps(state)))
            weights_file = base.with_suffix(".pkl").name
        # metadatos SIEMPRE en json portable
        meta_out = {**meta, "saved": ts, "weights_file": weights_file}
        _atomic(base.with_suffix(".json"),
                lambda p: p.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8"))
        # puntero "latest"
        _atomic(self.dir / "latest.json",
                lambda p: p.write_text(json.dumps({"name": name, **meta_out}, ensure_ascii=False, indent=2),
                                       encoding="utf-8"))
        self._last_save = time.time()
        self._rotate()
        return name

    def _rotate(self) -> None:
        metas = sorted(self.dir.glob("gen*_step*.json"), key=lambda p: p.stat().st_mtime)
        # conserva "best*" siempre; rota el resto
        rotatable = [m for m in metas if "_best" not in m.stem]
        excess = len(rotatable) - self.keep_last
        for m in rotatable[:max(0, excess)]:
            for suf in (".json", ".pt", ".pkl"):
                f = m.with_suffix(suf)
                if f.exists():
                    f.unlink()

    def save_best(self, state: dict[str, Any], meta: dict[str, Any]) -> str:
        return self.save(state, meta, tag="best")

    # --- reanudación -----------------------------------------------------
    def find_latest(self) -> dict[str, Any] | None:
        p = self.dir / "latest.json"
        if not p.exists():
            return None
        return json.loads(p.read_text(encoding="utf-8"))

    def load_state(self, name: str, map_location: str = "cpu") -> dict[str, Any] | None:
        torch = _torch()
        pt = self.dir / f"{name}.pt"
        pkl = self.dir / f"{name}.pkl"
        if pt.exists() and torch is not None:
            return torch.load(pt, map_location=map_location)
        if pkl.exists():
            return pickle.loads(pkl.read_bytes())
        return None


class StageArchiver:
    """Auto-guardado por etapa/generación: empaqueta artefactos en un .zip para el Mac."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.dir = Path(artifacts_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def package(self, label: str, include_dirs: list[Path], meta: dict[str, Any]) -> Path:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        zip_path = self.dir / f"{label}_{ts}.zip"
        tmp = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("MANIFEST.json", json.dumps({**meta, "label": label, "packaged": ts},
                                                    ensure_ascii=False, indent=2))
            for d in include_dirs:
                d = Path(d)
                if not d.exists():
                    continue
                for f in d.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=str(Path(d.name) / f.relative_to(d)))
        os.replace(tmp, zip_path)
        return zip_path


if __name__ == "__main__":
    # Test en Mac (sin torch): guarda, rota y reanuda un checkpoint de juguete.
    import tempfile
    d = tempfile.mkdtemp(prefix="ckpt_")
    cm = CheckpointManager(d, keep_last=2)
    for step in (100, 200, 300, 400):
        cm.save({"weights": [0.1] * 10, "step": step}, {"generation": 1, "stage": "1a", "step": step})
    latest = cm.find_latest()
    assert latest and latest["step"] == 400, latest
    st = cm.load_state(latest["name"])
    assert st and st["step"] == 400
    remaining = list(Path(d).glob("gen*_step*.json"))
    print(f"OK checkpointing: reanuda step={latest['step']}, quedan {len(remaining)} checkpoints (rotación keep_last=2)")
    # test archiver
    arc = StageArchiver(d)
    z = arc.package("gen1_etapa1a", [Path(d)], {"generation": 1})
    print("OK archiver: paquete", Path(z).name)
