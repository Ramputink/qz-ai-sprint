"""De la RTX 5090 al edge barato — el argumento de 'inversión mínima del cliente'.

Cascada: distillation (profesor grande → alumno pequeño) → INT8 (cuantización) →
ONNX (portable) → TensorRT (opcional, Jetson). El artefacto final corre en una caja
de ~250-400 € (Jetson Orin Nano / Mini-PC / Raspberry Pi + Coral) conectada por MQTT
al SCADA. torch/onnx perezosos.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def distill(teacher, student, dataloader, cfg: dict[str, Any], device="cuda"):
    """Entrena el alumno pequeño para imitar al profesor (soft targets)."""
    import torch
    import torch.nn.functional as F
    opt = torch.optim.AdamW(student.parameters(), lr=1e-3)
    student.train(); teacher.eval()
    logs = []
    for batch in dataloader:
        x = batch[0].to(device)
        with torch.no_grad():
            t = teacher(x)
        s = student(x)
        loss = F.mse_loss(s, t)  # imitación (regresión RUL/score)
        opt.zero_grad(); loss.backward(); opt.step()
        logs.append(float(loss.detach()))
    return {"distill_steps": len(logs), "final_loss": logs[-1] if logs else None}


def quantize_int8(model, calibration_loader, device="cpu"):
    """Cuantización dinámica INT8 (universal para edge barato)."""
    import torch
    model.eval().to(device)
    qmodel = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8)
    return qmodel


def strip_weight_norm(model):
    """Deshace la reparametrización weight_norm del TCN, in situ.

    weight_norm guarda el peso como (g, v) y engancha un hook; mientras está puesto,
    el módulo no se puede deepcopy-ar, y eso es exactamente lo que necesitan el
    exportador ONNX y la cuantización. Deshacerlo funde g y v en un único `weight`
    numéricamente idéntico, así que el modelo exportado predice lo mismo.
    """
    import torch
    import torch.nn.utils.parametrize as P

    for mod in model.modules():
        if P.is_parametrized(mod, "weight"):
            P.remove_parametrizations(mod, "weight", leave_parametrized=True)
        elif hasattr(mod, "weight_g") or hasattr(mod, "weight_v"):
            try:
                torch.nn.utils.remove_weight_norm(mod)
            except Exception:
                pass
    return model


def export_onnx(model, sample_input, out_path: str | Path, opset: int = 17) -> str:
    import torch
    strip_weight_norm(model)
    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(model, sample_input, out_path, opset_version=opset,
                      input_names=["input"], output_names=["output"],
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    _inline_external_data(out_path)
    return out_path


def _inline_external_data(out_path: str) -> None:
    """Deja el .onnx AUTOCONTENIDO (pesos dentro del fichero).

    El exportador nuevo de PyTorch saca los tensores a un `<modelo>.onnx.data`
    aparte. En el edge eso es una trampa: se copia el .onnx solo y el modelo carga
    vacío o falla. Un modelo de estos pesa centenares de KB, así que no hay razón
    para trocearlo.
    """
    sidecar = Path(out_path + ".data")
    if not sidecar.exists():
        return
    import onnx
    model = onnx.load(out_path, load_external_data=True)
    onnx.save(model, out_path, save_as_external_data=False)
    sidecar.unlink(missing_ok=True)


def export_tensorrt(onnx_path: str | Path, out_engine: str | Path, int8: bool = True) -> str:
    """Compila a engine TensorRT (solo si TensorRT está instalado en el PC/Jetson)."""
    try:
        import tensorrt  # noqa
    except Exception as e:
        raise RuntimeError(f"TensorRT no instalado ({e}); usa el ONNX con onnxruntime en el edge.")
    # Nota: la compilación real se hace con trtexec o la API; aquí se deja el punto de enganche.
    raise NotImplementedError("Compila con: trtexec --onnx=%s --saveEngine=%s %s" %
                              (onnx_path, out_engine, "--int8" if int8 else ""))


EDGE_TARGETS = {
    "jetson_orin_nano": {"precio_eur": "~250-400", "runtime": "TensorRT/ONNX", "mqtt": True,
                         "nota": "mejor relación potencia/precio para DL real en edge"},
    "minipc_x86": {"precio_eur": "~150-300", "runtime": "ONNX Runtime/OpenVINO", "mqtt": True,
                   "nota": "fácil de mantener por IT industrial; modelos tabulares/pequeños"},
    "rpi_coral": {"precio_eur": "~120", "runtime": "TFLite INT8 (Coral)", "mqtt": True,
                  "nota": "solo el alumno destilado más pequeño; Coral solo INT8"},
}


def deployment_manifest(model_name: str, onnx_path: str, target: str = "jetson_orin_nano") -> dict[str, Any]:
    return {"model": model_name, "onnx": onnx_path, "edge_target": target,
            "target_info": EDGE_TARGETS.get(target, {}),
            "pipeline": "SCADA/PLC --OPC-UA--> gateway --MQTT--> edge (inferencia INT8) --MQTT--> alertas"}
