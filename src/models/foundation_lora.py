"""Fine-tune de un foundation model de series temporales con LoRA.

Objetivo: aprovechar el conocimiento pre-entrenado (Chronos-Bolt / TimesFM / MOMENT)
para previsión + anomalía con pocos datos, y **fine-tunear solo unos MB (LoRA) por
noche** → barato, versionable, sin catastrophic forgetting del backbone.

Se intentan varios backbones por disponibilidad; si ninguno está instalado, se avisa
y el orquestador sigue con los modelos propios (RUL/AE/clasificador). Perezoso.
"""
from __future__ import annotations

from typing import Any, Optional


def load_backbone(prefer: str = "chronos"):
    """Carga un foundation model disponible. Devuelve (nombre, objeto) o (None, None)."""
    order = [prefer, "chronos", "timesfm", "moment"]
    seen = set()
    for name in order:
        if name in seen:
            continue
        seen.add(name)
        try:
            if name == "chronos":
                from chronos import ChronosPipeline  # type: ignore
                import torch
                pipe = ChronosPipeline.from_pretrained(
                    "amazon/chronos-bolt-base",
                    device_map="cuda" if torch.cuda.is_available() else "cpu")
                return ("chronos-bolt-base", pipe)
            if name == "timesfm":
                import timesfm  # type: ignore
                return ("timesfm", timesfm)
            if name == "moment":
                from momentfm import MOMENTPipeline  # type: ignore
                m = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
                                                   model_kwargs={"task_name": "forecasting"})
                return ("moment-1-large", m)
        except Exception:
            continue
    return (None, None)


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.05,
               target_modules: Optional[list[str]] = None):
    """Envuelve el modelo con adaptadores LoRA (peft). Congela el backbone; solo se
    entrenan las matrices de bajo rango (unos MB)."""
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
    except Exception as e:
        raise RuntimeError(f"instala 'peft' en el PC para LoRA ({e})")
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                     target_modules=target_modules or ["q_proj", "v_proj"], bias="none")
    return get_peft_model(model, cfg)


def save_lora_delta(model, out_path: str) -> str:
    """Guarda SOLO los pesos LoRA (unos MB) → lo que se despliega al edge cada noche."""
    model.save_pretrained(out_path)
    return out_path


def status() -> dict[str, Any]:
    name, obj = load_backbone()
    return {"backbone": name, "available": obj is not None}
