"""Orquestador — corre las 2 generaciones de 2 días de una sola tirada.

Flujo:
  Fase 0 (una vez): gpu_check → descarga datasets → preprocesa.
  Generación 1 (días 1-2): etapa 1a (predictivo+precisión) → etapa 1b (recal.+edge+eval).
  Generación 2 (días 3-4): etapa 2a (warm-start+HPO+secundarios) → etapa 2b (final+comparativa).

Cada etapa es un BUCLE ACOTADO POR TIEMPO. Durante el bucle:
  * cada `checkpoint_every_min` → guardado atómico reanudable,
  * cada `processview_refresh_sec` → actualiza el panel en vivo,
  al terminar la etapa → auto-guardado en .zip (paquete para el Mac).

Reanudable: guarda el estado (gen/etapa/paso) en state.json; con --resume continúa.
El trabajo real de entrenamiento se delega a work-functions; en --dry-run se SIMULA
(sin torch) para poder verificar todo el flujo en el Mac.
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable

from .checkpointing import CheckpointManager, StageArchiver
from .logging_utils import RunLogger
from .processview import ProcessView


# Etapas: (generation, stage_id, config_hours_key, etiqueta legible, día)
STAGES = [
    (1, "1a", ("generation_1", "stage_1a_hours"), "Gen1·Predictivo+precisión", 1),
    (1, "1b", ("generation_1", "stage_1b_hours"), "Gen1·Recalibración+edge+eval", 2),
    (2, "2a", ("generation_2", "stage_2a_hours"), "Gen2·Warm-start+HPO+secundarios", 3),
    (2, "2b", ("generation_2", "stage_2b_hours"), "Gen2·Final+comparativa Gen1/Gen2", 4),
]


class Orchestrator:
    def __init__(self, cfg: dict[str, Any], base_dir: Path):
        self.cfg = cfg
        self.base = Path(base_dir)
        p = cfg["paths"]
        self.logger = RunLogger(self.base / p["logs_dir"])
        self.pv = ProcessView(self.base / p["processview_dir"],
                              refresh_sec=cfg["run"].get("processview_refresh_sec", 10))
        self.ckpt = CheckpointManager(self.base / p["checkpoints_dir"], keep_last=6)
        self.archiver = StageArchiver(self.base / p["artifacts_dir"])
        self.state_path = self.base / "state.json"
        self.dry = bool(cfg["run"].get("dry_run", False))
        self._t0 = time.time()

    # --- persistencia de estado (reanudación) ---------------------------
    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"phase0_done": False, "completed_stages": []}

    def _save_state(self, st: dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    # --- ejecución principal --------------------------------------------
    def run(self, resume: bool = False, only_generation: int | None = None) -> None:
        st = self._load_state() if resume else {"phase0_done": False, "completed_stages": []}
        self.logger.info("run_start", dry_run=self.dry, resume=resume, only_generation=only_generation)

        if not st.get("phase0_done"):
            self._phase0()
            st["phase0_done"] = True
            self._save_state(st)

        for gen, sid, hours_key, label, day in STAGES:
            if only_generation and gen != only_generation:
                continue
            if sid in st.get("completed_stages", []):
                self.logger.info("stage_skip", stage=sid, reason="ya completada")
                continue
            hours = self._hours(hours_key)
            self._run_stage(gen, sid, label, day, hours)
            st.setdefault("completed_stages", []).append(sid)
            self._save_state(st)

        self.pv.update(phase_label="COMPLETADO", progress_pct=100.0)
        self.logger.info("run_done", elapsed_min=round((time.time() - self._t0) / 60, 1))

    def _hours(self, key) -> float:
        return float(self.cfg["schedule"][key[0]][key[1]])

    # --- fase 0 ----------------------------------------------------------
    def _phase0(self) -> None:
        self.pv.update(phase_label="Fase 0: verificación GPU", progress_pct=0.0, day_of_4=0)
        self.logger.info("phase0_start")
        if not self.dry:
            from .gpu_check import assert_ready
            rep = assert_ready(strict=True)
            self.logger.info("gpu_check", **{k: rep["details"].get(k) for k in ("device", "capability")})
            self._download()
            self._preprocess()
        else:
            self.logger.info("phase0_dry", note="simulado: sin GPU, sin descargas")
        self.pv.update(phase_label="Fase 0 completa", progress_pct=100.0)

    def _download(self) -> None:
        from .data.download import download_all
        from .data.registry import resolve
        keys: list[str] = []
        for _, lst in (self.cfg.get("datasets") or {}).items():
            keys.extend(lst or [])
        specs = resolve(keys)
        self.logger.info("download_start", n=len(specs), keys=keys)

        def cb(k, pct, msg):
            self.pv.update(phase_label=f"Descargando {k}", progress_pct=pct,
                           log_tail=self.logger.tail(12))
            self.logger.info("download_progress", dataset=k, pct=round(pct, 1), msg=msg)
        download_all(specs, self.base / self.cfg["paths"]["data_dir"], cb)

    def _preprocess(self) -> None:
        self.logger.info("preprocess_start")
        self.pv.update(phase_label="Preprocesado (features de vibración)", progress_pct=50.0)
        # El preprocesado concreto depende del layout de cada dataset; se ejecuta en
        # el PC con los datos ya descargados (src/data/preprocess.py provee las features).
        self.logger.info("preprocess_note", note="usa src/data/preprocess.py sobre data/*")

    # --- una etapa (bucle acotado por tiempo) ---------------------------
    def _run_stage(self, gen: int, sid: str, label: str, day: int, hours: float) -> None:
        self.logger.info("stage_start", generation=gen, stage=sid, label=label, hours=hours, day=day)
        deadline = time.time() + hours * 3600
        every_min = int(self.cfg["run"].get("checkpoint_every_min", 30))
        step = 0
        ckpt_names: list[str] = []
        work = self._simulate_step if self.dry else self._real_step
        ctx: dict[str, Any] = {"gen": gen, "stage": sid}

        while time.time() < deadline:
            metrics = work(ctx, step)
            step += 1
            frac = 1.0 - max(0.0, (deadline - time.time()) / (hours * 3600))
            self.pv.update(
                generation=gen, stage=sid, phase_label=label, day_of_4=day,
                progress_pct=round(frac * 100, 1), epoch=metrics.get("epoch"), step=step,
                metrics=metrics, target=self._target_view(),
                gpu=self._gpu_view(), eta=self._eta(deadline),
                checkpoints=[Path(c).name for c in ckpt_names[-8:]],
                log_tail=self.logger.tail(12),
            )
            name = self.ckpt.maybe_save(
                {"stage": sid, "step": step, "metrics": metrics, "model": ctx.get("model_state")},
                {"generation": gen, "stage": sid, "step": step, "metrics": metrics}, every_min)
            if name:
                ckpt_names.append(name)
                self.logger.info("checkpoint", name=name, step=step)
            # en dry-run avanzamos rápido; en real, el propio work bloquea lo suyo
            if self.dry:
                time.sleep(0.01)
                if step >= 40:   # simulación corta: no esperar horas reales
                    break

        # cierre de etapa: checkpoint 'best' + auto-guardado (.zip para el Mac)
        final_metrics = ctx.get("last_metrics", {})
        self.ckpt.save_best({"stage": sid, "step": step, "model": ctx.get("model_state")},
                            {"generation": gen, "stage": sid, "step": step, "metrics": final_metrics})
        zip_path = self.archiver.package(
            f"gen{gen}_etapa{sid}",
            [self.base / self.cfg["paths"]["checkpoints_dir"],
             self.base / self.cfg["paths"]["processview_dir"],
             self.base / self.cfg["paths"]["logs_dir"]],
            {"generation": gen, "stage": sid, "steps": step, "metrics": final_metrics})
        self.pv.update(artifacts=[Path(zip_path).name])
        self.logger.info("stage_done", stage=sid, steps=step, package=Path(zip_path).name)

    # --- work functions -------------------------------------------------
    def _simulate_step(self, ctx: dict[str, Any], step: int) -> dict[str, Any]:
        """Simula progreso de entrenamiento sin torch (para verificar el flujo en Mac)."""
        base_acc = 0.80 + 0.12 * (1 - math.exp(-step / 15))  # sube y se estabiliza
        gen_boost = 0.03 if ctx["gen"] == 2 else 0.0          # Gen2 mejora sobre Gen1
        acc = min(0.985, base_acc + gen_boost)
        lead = 8.0 + 4.0 * (1 - math.exp(-step / 12)) + (1.5 if ctx["gen"] == 2 else 0)
        m = {"epoch": step // 10, "loss": round(0.5 * math.exp(-step / 10) + 0.01, 4),
             "accuracy": round(acc, 4), "lead_time_days": round(lead, 2),
             "fp": max(0, 12 - step // 4), "fn": max(0, 8 - step // 3)}
        ctx["last_metrics"] = m
        ctx["model_state"] = {"sim": True, "step": step}
        return m

    def _real_step(self, ctx: dict[str, Any], step: int) -> dict[str, Any]:
        """Un paso de entrenamiento real en el PC. Wiring de modelos según etapa.
        Aquí se conectan RUL/AE/clasificador/foundation/recalibración/edge. El
        detalle fino se ajusta con los datos ya descargados; este es el punto de
        integración (documentado en README)."""
        # El bucle real de entrenamiento (dataloaders, optimizador, HPO Optuna) se
        # engancha aquí usando src/models/*, src/recalibration.py y src/edge_export.py.
        raise NotImplementedError(
            "Ejecución real: engancha aquí el trainer de la etapa. En Mac usa dry_run:true. "
            "En el PC, tras descargar/preprocesar, completa el wiring por etapa (ver README §Integración).")

    # --- vistas auxiliares ----------------------------------------------
    def _target_view(self) -> dict[str, Any]:
        t = self.cfg.get("target", {})
        return {"accuracy≥": t.get("min_accuracy"), "lead_time_days≥": t.get("lead_time_days"),
                "peso_FN": t.get("fn_weight")}

    def _gpu_view(self) -> dict[str, Any]:
        if self.dry:
            return {"modo": "dry-run (sin GPU)"}
        try:
            import torch
            if torch.cuda.is_available():
                free, total = torch.cuda.mem_get_info()
                return {"device": torch.cuda.get_device_name(0),
                        "vram_libre_gb": round(free / 1e9, 1), "vram_total_gb": round(total / 1e9, 1)}
        except Exception:
            pass
        return {}

    def _eta(self, deadline: float) -> str:
        rem = max(0, deadline - time.time())
        h = int(rem // 3600); m = int((rem % 3600) // 60)
        return f"{h}h{m:02d}m restantes en la etapa"
