"""Entrenamiento real por etapa — el wiring que faltaba en `_real_step`.

El orquestador llama a `StageTrainer.step()` en bucle hasta que se agota la ventana
de tiempo de la etapa; cada llamada corre un bloque de minibatches y, cada cierto
numero de bloques, evalua. Todo lo que decide el exito del proyecto se mide aqui:

  * RUL (TCN con perdida asimetrica): predice cuanta vida le queda al equipo.
  * ALARMA: se dispara cuando la RUL predicha baja del horizonte de aviso. El umbral
    NO se elige por accuracy sino por COSTE, con el falso negativo pesando fn_weight
    veces mas que el falso positivo (config `target.fn_weight`).
  * ANTICIPACION: cuanto antes del fallo real salta esa alarma, en dias. Es la metrica
    que el cliente entiende ("me avisas con X dias").
  * Autoencoder de salud: entrenado solo con ventanas sanas, su error de
    reconstruccion da un indice de degradacion independiente del RUL.
  * Baseline de boosting: fija el liston que la red tiene que superar.

Mapa de etapas (igual que el README):
  1a  entrena el modelo primario + AE + baseline y fija el umbral por coste.
  1b  recalibracion (deriva + replay + hard negatives + EWC) y export a edge.
  2a  warm-start desde la Gen 1, red mas ancha y datasets secundarios.
  2b  recalibracion final, export y comparativa Gen 1 vs Gen 2.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data.preprocess import list_products, load_product
from .models.classifier import best_threshold, hard_negatives, train_gbm
from .models.health_autoencoder import build_autoencoder, reconstruction_loss, threshold_from_healthy
from .models.rul import build_tcn, rul_loss
from .recalibration import DriftDetector, ReplayBuffer

# Orden de preferencia cuando hay varios datasets run-to-failure disponibles.
# IMS es el mas parecido al motor/rotor de planta, pero C-MAPSS tiene 709 unidades
# frente a 12: para una cifra de accuracy defendible manda el numero de maquinas.
_PRIMARY_PRIORITY = ("cmapss", "ncmapss", "metropt3", "nasa_ims_bearing")

# Orden cronologico de las etapas: cada una arranca de la mejor de las anteriores.
_STAGE_ORDER = ("1a", "1b", "2a", "2b")


def _stage_rank(filename: str) -> int:
    """Posicion de la etapa a la que pertenece un checkpoint, por su nombre."""
    for i, s in enumerate(_STAGE_ORDER):
        if f"_{s}_" in filename:
            return i
    return -1


def _device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _find_state_dict(blob: Any) -> dict:
    """Extrae el state_dict de un checkpoint sin depender de cómo esté anidado.

    El checkpoint envuelve el estado del entrenador dentro de la clave `model` del
    checkpoint, así que los tensores acaban en `blob["model"]["model"]`. Buscarlos
    en vez de asumir la ruta evita el fallo silencioso más caro de todos: un
    warm-start que carga 0 tensores y arranca en frío sin que nadie se entere.
    """
    import torch

    if not isinstance(blob, dict):
        return {}
    if blob and all(isinstance(v, torch.Tensor) for v in blob.values()):
        return blob
    for key in ("model", "state_dict"):
        found = _find_state_dict(blob.get(key))
        if found:
            return found
    return {}


class RulTask:
    """Un dataset run-to-failure listo para entrenar: tensores, split y horizonte."""

    def __init__(self, key: str, data_dir: Path, cfg: dict[str, Any], seed: int):
        import torch

        arrays, meta = load_product(data_dir, key, "rul")
        self.key = key
        self.meta = meta
        self.hours_per_unit = float(meta.get("hours_per_unit", 24.0))
        self.horizon = float(meta.get("lead_horizon_units",
                                      cfg["target"]["lead_time_days"] * 24 / self.hours_per_unit))
        self.rul_cap = float(meta.get("rul_cap", arrays["y_rul"].max()))

        X, y, unit, t_idx = arrays["X"], arrays["y_rul"], arrays["unit"], arrays["t_idx"]

        # `train.feature_slice` recorta columnas de features. Sirve para ablaciones
        # honestas: mismas ventanas, mismos folds, solo cambia el juego de features.
        sl = (cfg.get("train") or {}).get("feature_slice")
        if sl:
            X = X[:, :, int(sl[0]):int(sl[1])]

        # Split POR UNIDAD: si dos ventanas de la misma maquina caen a ambos lados,
        # el modelo ya ha visto el futuro de esa maquina y la metrica miente.
        units = np.unique(unit)
        forced = (cfg.get("train") or {}).get("val_units")
        if forced is not None:
            # validacion explicita: la usa la validacion cruzada leave-one-out, donde
            # cada pliegue reserva UNA maquina concreta y no un sorteo.
            val_units = set(int(u) for u in forced)
        else:
            rng = np.random.default_rng(seed)
            rng.shuffle(units)
            n_val = max(1, int(round(0.2 * len(units))))
            val_units = set(units[:n_val].tolist())
        m_val = np.isin(unit, list(val_units))

        dev = _device()
        # El dataset entero cabe en los 32 GB de la 5090; tenerlo residente evita
        # que el DataLoader sea el cuello de botella.
        self.Xtr = torch.as_tensor(X[~m_val]).to(dev)
        self.ytr = torch.as_tensor(y[~m_val]).to(dev)
        self.Xva = torch.as_tensor(X[m_val]).to(dev)
        self.yva = torch.as_tensor(y[m_val]).to(dev)
        self.unit_va = unit[m_val]
        self.t_va = t_idx[m_val]
        # Vida total de la maquina a la que pertenece cada ventana de validacion.
        # Permite expresar el aviso como FRACCION DE VIDA en vez de en dias absolutos:
        # "10 dias" es el 5,7 % de la vida de un motor C-MAPSS y el 83 % de la de un
        # ensayo IMS, asi que en dias absolutos no se esta preguntando lo mismo.
        vida = {int(u): float(y[unit == u].max()) for u in units}
        self.vida_va = np.array([vida[int(u)] for u in self.unit_va], dtype=np.float32)
        self.n_features = X.shape[-1]
        self.window = X.shape[1]
        self.n_units = int(len(units))
        self.n_val_units = len(val_units)
        self.norm = {"mu": arrays.get("norm_mu"), "sd": arrays.get("norm_sd")}

    def summary(self) -> dict[str, Any]:
        return {"dataset": self.key, "ventanas_train": int(self.Xtr.shape[0]),
                "ventanas_val": int(self.Xva.shape[0]), "features": self.n_features,
                "ventana": self.window, "unidades": self.n_units,
                "unidades_val": self.n_val_units,
                "horizonte_aviso_pasos": round(self.horizon, 1),
                "horas_por_paso": round(self.hours_per_unit, 4)}


class StageTrainer:
    """Entrenador de UNA etapa. Reutilizable entre llamadas: mantiene modelo y estado."""

    def __init__(self, cfg: dict[str, Any], base_dir: Path, gen: int, stage: str, logger):
        self.cfg = cfg
        self.base = Path(base_dir)
        self.gen = gen
        self.stage = stage
        self.log = logger
        self.data_dir = self.base / cfg["paths"]["data_dir"]
        self.seed = int(cfg["run"].get("seed", 0))
        self.fn_weight = float(cfg["target"]["fn_weight"])
        self.lead_days = float(cfg["target"]["lead_time_days"])
        self.min_acc = float(cfg["target"]["min_accuracy"])

        tr = cfg.get("train", {})
        self.batch = 512 if tr.get("batch_size") == "auto" else int(tr.get("batch_size", 512))
        self.precision = tr.get("precision", "bf16")
        self.batches_per_step = int(tr.get("batches_per_call", 150))
        self.val_every = int(tr.get("validate_every_calls", 4))
        self.patience = int(tr.get("early_stop_patience", 15))

        self.tasks: list[RulTask] = []
        self.primary: RulTask | None = None
        self.model = None
        self.opt = None
        self.extra_models: dict[str, Any] = {}
        self.ae = None
        self.ae_opt = None
        self.ae_threshold: float | None = None
        self.best: dict[str, Any] = {}
        self._since_best = 0
        self._calls = 0
        self._ready = False
        self.stop = False
        self.extras: dict[str, Any] = {}
        self.drift = DriftDetector()
        self.replay = ReplayBuffer(capacity=int(cfg.get("target", {}).get("replay_capacity", 50000)))
        self._recal_events = 0
        self._hard_negs = 0

    # ---------- construccion perezosa ------------------------------------
    def _channels(self) -> tuple[int, ...]:
        """La Gen 2 anade PROFUNDIDAD manteniendo el ancho de la Gen 1.

        Ensanchar la red romperia el warm-start: con 96 canales ningun tensor de la
        Gen 1 (64) encaja en forma y la "mejora en caliente" seria en realidad un
        arranque en frio con otra red. Anadiendo un bloque mas de 64 canales, los
        cuatro primeros bloques y la cabeza se cargan tal cual desde la Gen 1 y el
        bloque nuevo (dilatacion 16) amplia el campo receptivo sobre lo ya aprendido.
        """
        return (64, 64, 64, 64, 64) if self.gen == 2 else (64, 64, 64, 64)

    def setup(self) -> dict[str, Any]:
        import torch

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        prods = list_products(self.data_dir)["rul"]
        if not prods:
            raise RuntimeError(
                "no hay ningun producto RUL en data/processed. Ejecuta la Fase 0 "
                "(descarga + preprocesado) antes de entrenar.")
        order = sorted(prods, key=lambda k: _PRIMARY_PRIORITY.index(k)
                       if k in _PRIMARY_PRIORITY else 99)
        # `train.primary_dataset` permite forzar cual manda: por defecto gana C-MAPSS
        # (709 unidades), pero para medir el caso mas parecido a planta hay que poder
        # poner IMS al frente y verlo con validacion completa, no como secundario.
        forced = (self.cfg.get("train") or {}).get("primary_dataset")
        if forced and forced in order:
            order = [forced] + [k for k in order if k != forced]
        # La etapa 2a entrena tambien los datasets secundarios ("modelos secundarios").
        use = order if self.stage == "2a" else order[:1]
        self.tasks = [RulTask(k, self.data_dir, self.cfg, self.seed) for k in use]
        self.primary = self.tasks[0]

        dev = _device()
        self.model = build_tcn(self.primary.n_features, channels=self._channels()).to(dev)
        self._warm_start()
        lr = float(self.cfg.get("train", {}).get("lr", 1e-3))
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)

        # Autoencoder de salud sobre el ultimo instante de cada ventana SANA.
        self.ae = build_autoencoder(self.primary.n_features).to(dev)
        self.ae_opt = torch.optim.AdamW(self.ae.parameters(), lr=1e-3)

        # Modelos secundarios (etapa 2a): cada dataset tiene su propio numero de
        # sensores (C-MAPSS 24, N-CMAPSS 18, IMS 19, MetroPT-3 15), asi que NO pueden
        # compartir la red del primario. Cada uno lleva la suya, misma arquitectura.
        for extra in self.tasks[1:]:
            mdl = build_tcn(extra.n_features, channels=self._channels()).to(dev)
            self.extra_models[extra.key] = (
                mdl, torch.optim.AdamW(mdl.parameters(), lr=lr, weight_decay=1e-4))

        if self.cfg.get("train", {}).get("torch_compile"):
            # Inductor puede compilar y fallar despues, en la PRIMERA pasada: hay que
            # calentar aqui para poder volver a modo eager sin tumbar la etapa.
            eager = self.model
            try:
                compiled = torch.compile(self.model, mode="max-autotune")
                with torch.no_grad():
                    compiled(torch.zeros(2, self.primary.window,
                                         self.primary.n_features, device=dev))
                self.model = compiled
                self.extras["torch_compile"] = "activo"
            except Exception as e:
                self.model = eager
                self.extras["torch_compile"] = f"no disponible, se sigue en eager ({type(e).__name__})"

        self._ready = True
        info = {"device": dev, "primario": self.primary.key,
                "secundarios": [t.key for t in self.tasks[1:]],
                "canales": list(self._channels()), "batch": self.batch,
                **self.primary.summary()}
        self.log.info("trainer_setup", **info)
        return info

    def _warm_start(self) -> None:
        """Continua desde la mejor etapa ANTERIOR, cargando las capas que encajen.

        Vale para las cuatro etapas, no solo para la Gen 2. Sin esto cada etapa
        reentrenaria desde cero: la 1b tiraria el modelo que acaba de aprender la 1a,
        la recalibracion recalibraria una red recien inicializada, y la "mejora de la
        Gen 2 sobre la Gen 1" seria una comparacion entre dos arranques en frio.

        La Gen 2 usa una red mas ancha, asi que solo coinciden parte de los tensores:
        se cargan los que encajan en forma y se informa de cuantos, para que un
        warm-start vacio no pase desapercibido.
        """
        import torch

        ck_dir = self.base / self.cfg["paths"]["checkpoints_dir"]
        earlier = _STAGE_ORDER[:_STAGE_ORDER.index(self.stage)] if self.stage in _STAGE_ORDER else []
        cands = [p for p in ck_dir.glob("gen*_best.pt")
                 if any(f"_{s}_" in p.name for s in earlier)]
        cands.sort(key=lambda p: (_stage_rank(p.name), p.stat().st_mtime))
        if not cands:
            self.extras["warm_start"] = "sin etapa previa: arranque en frio"
            return
        try:
            blob = torch.load(cands[-1], map_location=_device(), weights_only=False)
            prev = _find_state_dict(blob)
            if not prev:
                self.extras["warm_start"] = f"{cands[-1].name} no contenía tensores: arranque en frío"
                return
            sd = self.model.state_dict()
            loaded = 0
            for k, v in prev.items():
                if k in sd and sd[k].shape == v.shape:
                    sd[k] = v
                    loaded += 1
            self.model.load_state_dict(sd)
            self.extras["warm_start"] = f"{loaded}/{len(sd)} tensores desde {cands[-1].name}"
        except Exception as e:
            self.extras["warm_start"] = f"fallo ({type(e).__name__}: {e})"

    # ---------- bucle de entrenamiento ------------------------------------
    def _autocast(self):
        import torch
        if self.precision == "bf16" and torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.bfloat16)
        from contextlib import nullcontext
        return nullcontext()

    def _train_block(self, task: RulTask, model=None, opt=None,
                     batches: int | None = None) -> float:
        """Un bloque de minibatches sobre una tarea. Devuelve la perdida media."""
        import torch

        model = model if model is not None else self.model
        opt = opt if opt is not None else self.opt
        model.train()
        n = task.Xtr.shape[0]
        losses = []
        for _ in range(batches or self.batches_per_step):
            idx = torch.randint(0, n, (min(self.batch, n),), device=task.Xtr.device)
            x, y = task.Xtr[idx], task.ytr[idx]
            opt.zero_grad(set_to_none=True)
            with self._autocast():
                pred = model(x)
                loss = rul_loss(pred.float(), y, self.fn_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    def _train_ae_block(self, task: RulTask, n_batches: int = 30) -> float:
        """El AE aprende SOLO la normalidad: ventanas con RUL alto (equipo sano)."""
        import torch

        healthy = (task.ytr >= task.rul_cap * 0.9).nonzero(as_tuple=True)[0]
        if healthy.numel() < 64:
            return float("nan")
        self.ae.train()
        losses = []
        for _ in range(n_batches):
            sel = healthy[torch.randint(0, healthy.numel(), (min(self.batch, healthy.numel()),),
                                        device=healthy.device)]
            x = task.Xtr[sel][:, -1, :]            # ultimo instante de la ventana
            self.ae_opt.zero_grad(set_to_none=True)
            loss = reconstruction_loss(self.ae, x)
            loss.backward()
            self.ae_opt.step()
            losses.append(float(loss.detach()))
        return float(np.mean(losses))

    @staticmethod
    def _predict(model, X, batch: int = 4096) -> np.ndarray:
        import torch
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, X.shape[0], batch):
                out.append(model(X[i:i + batch]).float().cpu().numpy())
        return np.concatenate(out)

    # ---------- evaluacion -------------------------------------------------
    def _lead_time_days(self, pred: np.ndarray, task: RulTask, thr_rul: float) -> dict[str, float]:
        """Anticipacion real: por cada maquina de validacion, cuanto antes del fallo
        la RUL predicha cruza el umbral de alarma y YA NO vuelve a subir por encima.

        Se exige que la alarma sea sostenida (no un pico aislado) porque en planta una
        alarma que va y viene no se atiende.
        """
        leads, missed = [], 0
        for u in np.unique(task.unit_va):
            m = task.unit_va == u
            order = np.argsort(task.t_va[m])
            p = pred[m][order]
            alarm = p <= thr_rul
            if not alarm.any():
                missed += 1
                continue
            # primer instante a partir del cual la alarma se mantiene hasta el final
            sustained = np.flatnonzero(np.cumprod(alarm[::-1])[::-1])
            first = int(sustained[0]) if sustained.size else int(np.argmax(alarm))
            steps_before_failure = (len(p) - 1) - first
            leads.append(steps_before_failure * task.hours_per_unit / 24.0)
        if not leads:
            return {"lead_time_days": 0.0, "lead_time_days_min": 0.0,
                    "maquinas_sin_aviso": int(missed)}
        return {"lead_time_days": round(float(np.mean(leads)), 2),
                "lead_time_days_min": round(float(np.min(leads)), 2),
                "maquinas_sin_aviso": int(missed)}

    @staticmethod
    def _auc(y_bin: np.ndarray, score: np.ndarray) -> dict[str, Any]:
        """AUC sin umbral: separa "ordena bien" de "el corte esta bien puesto".

        La accuracy que reporta este trainer se mide DESPUES de elegir el umbral por
        coste, asi que mezcla las dos cosas. Un pliegue con 1609 falsos positivos
        puede tener un modelo que ordena perfectamente y un umbral mal colocado, o un
        modelo que no discrimina: la accuracy sola no los distingue y el AUC si.

        Se reporta PR-AUC ademas de ROC-AUC porque las clases estan desbalanceadas:
        con pocas ventanas en fallo, el ROC-AUC se ve optimista. El PR-AUC solo se
        interpreta contra su linea base, que es la prevalencia, asi que va incluida.
        """
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            prevalencia = float(np.mean(y_bin))
            return {"pr_auc": round(float(average_precision_score(y_bin, score)), 4),
                    "pr_auc_base": round(prevalencia, 4),
                    "roc_auc": round(float(roc_auc_score(y_bin, score)), 4)}
        except Exception as e:
            return {"pr_auc": None, "roc_auc": None, "auc_error": type(e).__name__}

    def evaluate(self, task: RulTask | None = None, model=None) -> dict[str, Any]:
        """Metricas completas sobre validacion. El umbral se elige por COSTE."""
        task = task or self.primary
        pred = self._predict(model if model is not None else self.model, task.Xva)
        true = task.yva.cpu().numpy()

        # etiqueta binaria: el fallo cae dentro del horizonte de aviso exigido.
        # `target.lead_fraction` lo expresa como fraccion de la vida de CADA maquina;
        # si no se fija, se usa el horizonte en dias absolutos del config.
        frac = (self.cfg.get("target") or {}).get("lead_fraction")
        if frac:
            umbral = float(frac) * task.vida_va
            y_bin = (true <= umbral).astype(int)
        else:
            y_bin = (true <= task.horizon).astype(int)
        score = -pred                              # cuanto menos vida queda, mas urgente
        if y_bin.min() == y_bin.max():             # horizonte fuera del rango del dataset
            return {"error": "el horizonte de aviso no parte los datos de validacion",
                    "horizonte_pasos": task.horizon, "dataset": task.key}

        b = best_threshold(y_bin, score, self.fn_weight)
        thr_rul = -b["threshold"]
        recall = b["tp"] / max(1, b["tp"] + b["fn"])
        precision = b["tp"] / max(1, b["tp"] + b["fp"])
        lead = self._lead_time_days(pred, task, thr_rul)
        auc = self._auc(y_bin, score)

        m = {"dataset": task.key,
             "mae_rul": round(float(np.mean(np.abs(pred - true))), 3),
             "accuracy": b["accuracy"], "tp": b["tp"], "tn": b["tn"],
             "fp": b["fp"], "fn": b["fn"],
             "recall_fallo": round(recall, 4), "precision_fallo": round(precision, 4),
             "coste": round(float(b["cost"]), 1),
             "umbral_rul": round(float(thr_rul), 3), **lead, **auc,
             "cumple_accuracy": bool(b["accuracy"] >= self.min_acc),
             "cumple_anticipacion": bool(lead["lead_time_days"] >= self.lead_days)}
        return m

    def _update_health_threshold(self, task: RulTask) -> None:
        import torch
        healthy = (task.ytr >= task.rul_cap * 0.9).nonzero(as_tuple=True)[0]
        if healthy.numel() < 64:
            return
        with torch.no_grad():
            s = self.ae.health_score(task.Xtr[healthy][:, -1, :]).float().cpu().numpy()
        self.ae_threshold = threshold_from_healthy(s, 99.0)

    # ---------- recalibracion (etapas 1b / 2b) -----------------------------
    def _recalibrate(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Un ciclo de recalibracion: detecta deriva, reinyecta los falsos negativos
        con mas peso y hace un fine-tune corto sobre esa mezcla.

        Es el mecanismo por el que los FN bajan con el tiempo en vez de quedarse fijos.
        """
        import torch

        task = self.primary
        pred = self._predict(self.model, task.Xva)
        true = task.yva.cpu().numpy()
        y_bin = (true <= task.horizon).astype(int)
        y_hat = (pred <= metrics.get("umbral_rul", task.horizon)).astype(int)

        idx = np.arange(len(y_bin))
        fn_idx, n_fn = hard_negatives(y_bin, y_hat, idx)
        boost = float(self.cfg["target"].get("hard_negative_boost", 3.0))
        self._hard_negs += int(n_fn)
        for i in fn_idx.tolist():
            self.replay.add(int(i), weight=boost)

        drifted = self.drift.update(float(metrics.get("mae_rul", 0.0)))
        if drifted:
            self._recal_events += 1

        # fine-tune corto: mezcla de datos de entrenamiento + replay de hard negatives
        if self.replay.items:
            sel = np.array(self.replay.sample(min(self.batch, len(self.replay.items))), dtype=np.int64)
            xb = task.Xva[torch.as_tensor(sel, device=task.Xva.device)]
            yb = task.yva[torch.as_tensor(sel, device=task.Xva.device)]
            self.model.train()
            self.opt.zero_grad(set_to_none=True)
            with self._autocast():
                loss = rul_loss(self.model(xb).float(), yb, self.fn_weight * boost)
            loss.backward()
            self.opt.step()

        return {"deriva_detectada": bool(drifted), "eventos_deriva": self._recal_events,
                "falsos_negativos_reinyectados": self._hard_negs,
                "replay": len(self.replay.items)}

    # ---------- API que usa el orquestador --------------------------------
    def step(self, step_idx: int) -> dict[str, Any]:
        if not self._ready:
            self.setup()

        self._calls += 1
        t0 = time.time()
        loss = self._train_block(self.primary)
        ae_loss = self._train_ae_block(self.primary)
        # Datasets secundarios (etapa 2a): cada uno entrena su propia red, con menos
        # bloques por vuelta para no robarle tiempo al modelo primario.
        sec_loss = {}
        for extra in self.tasks[1:]:
            mdl, opt = self.extra_models[extra.key]
            sec_loss[extra.key] = round(
                self._train_block(extra, mdl, opt, max(20, self.batches_per_step // 4)), 5)

        m: dict[str, Any] = {"epoch": self._calls // max(1, self.val_every),
                             "loss": round(loss, 5),
                             "ae_loss": None if math.isnan(ae_loss) else round(ae_loss, 5),
                             "seg_por_bloque": round(time.time() - t0, 2),
                             "batches": self.batches_per_step * self._calls}
        if sec_loss:
            m["loss_secundarios"] = sec_loss

        if self._calls % self.val_every == 0 or self._calls == 1:
            m.update(self.evaluate())
            self._update_health_threshold(self.primary)
            if self.stage in ("1b", "2b"):
                m.update(self._recalibrate(m))
            self._track_best(m)
        else:                                       # entre validaciones, repite la ultima
            for k in ("accuracy", "fp", "fn", "lead_time_days", "mae_rul"):
                if k in self.best.get("metrics", {}):
                    m[k] = self.best["metrics"][k]
        self.last = m
        return m

    def _track_best(self, m: dict[str, Any]) -> None:
        """El 'mejor' modelo es el de MENOR COSTE (FN pesan fn_weight), no el de mayor
        accuracy: un modelo con 95 % de accuracy que se come los fallos no sirve."""
        cost = m.get("coste")
        if cost is None:
            return
        if not self.best or cost < self.best["cost"]:
            self.best = {"cost": cost, "metrics": dict(m), "call": self._calls}
            self._since_best = 0
        else:
            self._since_best += 1
            if self._since_best >= self.patience:
                self.stop = True
                self.log.info("early_stop", etapa=self.stage,
                              sin_mejora=self._since_best, mejor_coste=self.best["cost"])

    def model_state(self) -> dict[str, Any]:
        """Estado serializable para el checkpoint (portable al Mac)."""
        try:
            base = getattr(self.model, "_orig_mod", self.model)
            sd = {k: v.detach().cpu() for k, v in base.state_dict().items()}
            ae = {k: v.detach().cpu() for k, v in self.ae.state_dict().items()} if self.ae else None
        except Exception:
            sd, ae = None, None
        extra_sd = {}
        for key, (mdl, _) in self.extra_models.items():
            try:
                extra_sd[key] = {k: v.detach().cpu() for k, v in mdl.state_dict().items()}
            except Exception:
                pass
        return {"model": sd, "autoencoder": ae, "secundarios": extra_sd,
                "gen": self.gen, "stage": self.stage,
                "dataset": self.primary.key if self.primary else None,
                "channels": list(self._channels()),
                "n_features": self.primary.n_features if self.primary else None,
                "window": self.primary.window if self.primary else None,
                "ae_threshold": self.ae_threshold,
                "best": self.best.get("metrics"), "extras": self.extras}

    # ---------- cierre de etapa -------------------------------------------
    def finalize(self) -> dict[str, Any]:
        """Trabajo de cierre segun etapa: baseline, export a edge y comparativa."""
        out: dict[str, Any] = {"etapa": self.stage, "extras": self.extras}
        if not self._ready:
            return out
        try:
            if self.stage == "1a":
                out["baseline_boosting"] = self._train_baseline()
            if self.extra_models:
                out["modelos_secundarios"] = self._evaluate_secondaries()
            if self.stage in ("1b", "2b"):
                out["edge"] = self._export_edge()
            if self.stage == "2b":
                out["comparativa"] = self._compare_generations()
            out["mejor"] = self.best.get("metrics")
            out["objetivo"] = self._target_report()
        except Exception as e:
            out["error_cierre"] = f"{type(e).__name__}: {e}"
        self.log.info("stage_finalize", etapa=self.stage,
                      claves=[k for k in out if k != "extras"])
        return out

    def _evaluate_secondaries(self) -> dict[str, Any]:
        """Evalua cada modelo secundario contra SU dataset y su propio horizonte.

        No son comparables entre si: un ciclo de C-MAPSS es un dia y una instantanea
        de IMS son 10 minutos, asi que cada uno responde a "avisa con >=10 dias" en
        su propia escala. Por eso cada fila lleva su horizonte y sus unidades.
        """
        out: dict[str, Any] = {}
        for extra in self.tasks[1:]:
            mdl, _ = self.extra_models[extra.key]
            try:
                m = self.evaluate(extra, mdl)
                m["unidades_val"] = extra.n_val_units
                m["horizonte_aviso_pasos"] = round(extra.horizon, 1)
                m["horas_por_paso"] = round(extra.hours_per_unit, 4)
                out[extra.key] = m
            except Exception as e:
                out[extra.key] = {"error": f"{type(e).__name__}: {e}"}
        return out

    def _window_summary(self, X) -> np.ndarray:
        """Resumen de una ventana para el baseline tabular: ultimo valor, media,
        desviacion y pendiente por canal. Es lo que un boosting puede aprovechar."""
        arr = X.cpu().numpy() if hasattr(X, "cpu") else np.asarray(X)
        T = arr.shape[1]
        t = np.arange(T, dtype=np.float32)
        tc = t - t.mean()
        slope = (arr * tc[None, :, None]).sum(axis=1) / max(1e-9, float((tc ** 2).sum()))
        return np.concatenate([arr[:, -1, :], arr.mean(axis=1), arr.std(axis=1), slope], axis=1)

    def _train_baseline(self) -> dict[str, Any]:
        """Gradient boosting sobre features resumidas: el liston a superar."""
        task = self.primary
        Xtr = self._window_summary(task.Xtr)
        ytr = (task.ytr.cpu().numpy() <= task.horizon).astype(int)
        Xva = self._window_summary(task.Xva)
        yva = (task.yva.cpu().numpy() <= task.horizon).astype(int)
        if ytr.min() == ytr.max() or yva.min() == yva.max():
            return {"nota": "una sola clase; baseline no aplicable"}
        name, clf = train_gbm(Xtr, ytr, self.cfg)
        score = clf.predict_proba(Xva)[:, 1]
        b = best_threshold(yva, score, self.fn_weight)
        return {"modelo": name, "accuracy": b["accuracy"], "fp": b["fp"], "fn": b["fn"],
                "coste": round(float(b["cost"]), 1),
                "nota": "mismo criterio de umbral por coste que la red, para comparar de igual a igual"}

    def _export_edge(self) -> dict[str, Any]:
        """Export a ONNX + INT8: el artefacto que corre en la caja barata del cliente."""
        import torch

        from .edge_export import deployment_manifest, export_onnx, strip_weight_norm

        task = self.primary
        out_dir = self.base / self.cfg["paths"]["artifacts_dir"] / "edge"
        out_dir.mkdir(parents=True, exist_ok=True)
        base = getattr(self.model, "_orig_mod", self.model)
        # copia en CPU sin weight_norm: el modelo de entrenamiento se queda intacto
        cpu_model = build_tcn(task.n_features, channels=self._channels())
        cpu_model.load_state_dict(base.state_dict())
        strip_weight_norm(cpu_model)
        cpu_model.eval()
        sample = torch.zeros(1, task.window, task.n_features)

        res: dict[str, Any] = {}
        onnx_path = out_dir / f"rul_gen{self.gen}_{task.key}.onnx"
        try:
            export_onnx(cpu_model, sample, onnx_path)
            res["onnx"] = onnx_path.name
            res["onnx_mb"] = round(onnx_path.stat().st_size / 1e6, 2)
        except Exception as e:
            res["onnx_error"] = f"{type(e).__name__}: {e}"

        if self.cfg.get("edge", {}).get("quantize") == "int8":
            try:
                from torch.ao.quantization import quantize_dynamic
                q = quantize_dynamic(cpu_model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8)
                qp = out_dir / f"rul_gen{self.gen}_{task.key}_int8.pt"
                torch.save(q.state_dict(), qp)
                res["int8"] = qp.name
                res["int8_mb"] = round(qp.stat().st_size / 1e6, 2)
            except Exception as e:
                res["int8_error"] = f"{type(e).__name__}: {e}"

        # la normalizacion viaja con el modelo: sin ella el edge predice basura
        norm = {"mu": np.asarray(task.norm["mu"]).tolist() if task.norm["mu"] is not None else None,
                "sd": np.asarray(task.norm["sd"]).tolist() if task.norm["sd"] is not None else None}
        manifest = deployment_manifest(f"rul_gen{self.gen}_{task.key}", str(onnx_path))
        manifest.update({"ventana": task.window, "features": task.meta.get("features"),
                         "normalizacion": norm, "umbral_rul_alarma": self.best.get("metrics", {}).get("umbral_rul"),
                         "umbral_salud_ae": self.ae_threshold,
                         "horas_por_paso": task.hours_per_unit})
        (out_dir / f"manifest_gen{self.gen}.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        res["manifest"] = f"manifest_gen{self.gen}.json"
        return res

    def _compare_generations(self) -> dict[str, Any]:
        """Comparativa Gen 1 vs Gen 2 leyendo los checkpoints 'best' de cada una."""
        ck = self.base / self.cfg["paths"]["checkpoints_dir"]
        rows: dict[str, Any] = {}
        for gen in (1, 2):
            metas = sorted(ck.glob(f"gen{gen}_*_best.json"),
                           key=lambda p: (_stage_rank(p.name), p.stat().st_mtime))
            if metas:
                blob = json.loads(metas[-1].read_text(encoding="utf-8"))
                rows[f"gen{gen}"] = blob.get("metrics") or {}
        # Esta comparativa corre DENTRO del finalize de la 2b, y el checkpoint 'best'
        # de la 2b aún no está en disco: sin esto compararíamos la Gen 1 completa
        # contra la Gen 2 a falta de su última etapa, que es justo la que cierra.
        if self.best.get("metrics"):
            rows["gen2"] = dict(self.best["metrics"])
        g1, g2 = rows.get("gen1", {}), rows.get("gen2", {})
        delta = {}
        for k in ("accuracy", "fn", "fp", "lead_time_days", "mae_rul", "coste"):
            if k in g1 and k in g2 and isinstance(g1[k], (int, float)):
                delta[k] = round(g2[k] - g1[k], 4)
        report = {"gen1": g1, "gen2": g2, "delta_gen2_menos_gen1": delta}
        path = self.base / self.cfg["paths"]["artifacts_dir"] / "comparativa_gen1_gen2.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _target_report(self) -> dict[str, Any]:
        """Veredicto explicito contra los tres criterios de exito del config."""
        m = self.best.get("metrics", {})
        return {
            "accuracy": {"objetivo": self.min_acc, "obtenido": m.get("accuracy"),
                         "cumple": bool((m.get("accuracy") or 0) >= self.min_acc)},
            "anticipacion_dias": {"objetivo": self.lead_days, "obtenido": m.get("lead_time_days"),
                                  "cumple": bool((m.get("lead_time_days") or 0) >= self.lead_days)},
            "falsos_negativos": {"peso": self.fn_weight, "fn": m.get("fn"), "fp": m.get("fp"),
                                 "recall_fallo": m.get("recall_fallo"),
                                 "nota": "umbral elegido minimizando fp + peso*fn"},
        }
