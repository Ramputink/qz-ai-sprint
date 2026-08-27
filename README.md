# QZ AI Sprint — entrenamiento predictivo (Windows/RTX 5090 → análisis en Mac)

Paquete de scripts para entrenar el motor de IA de QuantumZIGMA en **4 días de una
tirada**, priorizando **mantenimiento predictivo**: predecir la rotura de un motor/rotor
con **≥10 días de antelación**, **≥90 % de accuracy** y **falsos negativos mínimos**, con
**recalibración continua**. Se entrena en un PC (Windows + AMD + **RTX 5090**) y el progreso
se analiza cómodamente en un **Mac**.

---

## 0. Idea en una frase

Un solo `run.py` orquesta descargas → preprocesado → entrenamiento en **2 generaciones de 2
días** (Gen 1 = pipeline completo; Gen 2 = mejora en caliente sobre la Gen 1), guarda un
**checkpoint cada ~30 min**, **auto-empaqueta cada etapa** en un `.zip` para llevártelo al Mac,
y muestra un **ProcessView** (panel HTML en vivo) para ver en qué punto estás.

---

## 1. En el PC de entrenamiento (Windows + RTX 5090)

### 1.1 Requisitos previos (una vez)
- **Driver NVIDIA ≥ 570** y **CUDA ≥ 12.8** (Blackwell). Comprueba con `nvidia-smi`
  (debe decir *CUDA Version: 12.8* o superior).
- Python 3.11 o 3.12.

### 1.2 Instalar
Copia toda la carpeta `qz-ai-sprint/` al PC y ejecuta (doble clic o en PowerShell):
```powershell
.\setup_windows.bat
```
Instala PyTorch **cu128** (imprescindible para la 5090), el resto de dependencias, y
**verifica el stack** (`python run.py --gpu-check` debe salir con `capability 12.0` y `sm_120`).

### 1.3 Probar sin entrenar (2 segundos)
```powershell
python run.py --dry-run       # simula las 2 generaciones y genera un ProcessView de ejemplo
python run.py --list-data     # lista los datasets del plan y su tamaño
```

### 1.4 Lanzar el sprint de 4 días (de una tirada)
```powershell
python run.py                 # Gen 1 (días 1-2) + Gen 2 (días 3-4)
```
- **Panel en vivo:** abre `processview\index.html` en el navegador (se auto-refresca).
- **Si se corta la luz / reinicias:** `python run.py --resume` retoma desde el último checkpoint.
- **Relanzar solo la Gen 2:** `python run.py --generation 2`.

Las **horas por etapa** se editan en `config.yaml` (`schedule:`). Si quieres 2 días en vez de
4, reduce las horas de cada etapa a 12 h.

---

## 2. En el Mac (análisis del progreso)

Copia del PC las carpetas `checkpoints/`, `artifacts/`, `processview/` y `logs/` (USB o red).
Luego:
```bash
bash setup_mac.sh                                         # entorno de análisis (CPU)
source .venv-mac/bin/activate
python analyze/view_progress.py artifacts/gen1_etapa1a_*.zip --open   # abre el panel de un paquete
python analyze/load_checkpoint.py checkpoints/latest.json            # resume un checkpoint
python analyze/plots.py logs/run.jsonl                              # curvas + matriz de confusión
```
Todo lo que produce el PC es **portable** (JSON/CSV/PNG/ONNX): el análisis en Mac **no necesita
GPU** ni CUDA.

---

## 3. Qué hace cada generación

| | Días | Qué |
|---|---|---|
| **Fase 0** | inicio | verifica GPU → descarga datasets → preprocesa (features de vibración) |
| **Gen 1 · etapa 1a** | 1 | RUL (TCN) + health-autoencoder + baselines boosting; umbral para ≥90 % minimizando FN |
| **Gen 1 · etapa 1b** | 2 | recalibración (River+ADWIN+replay+EWC) + export a edge + eval → **artefacto Gen 1 completo** |
| **Gen 2 · etapa 2a** | 3 | warm-start desde Gen 1 + HPO expandido + modelos secundarios (foundation LoRA, GDN, NILM, IDS OT) |
| **Gen 2 · etapa 2b** | 4 | recalibración final + edge + **comparativa Gen 1 vs Gen 2** (demuestra la mejora) |

**El objetivo que decide el éxito** (`config.yaml` → `target`): aviso ≥10 días antes del fallo,
accuracy ≥90 %, y matriz de confusión con **FN priorizados a la baja** (peso de FN = 5×). La
recalibración reinyecta los falsos negativos con más peso (*hard-negative mining*) → los FN bajan
con el tiempo.

---

## 4. Estructura

```
run.py              orquestador único (--dry-run, --resume, --generation, --list-data, --gpu-check)
config.yaml         horas por etapa, cadencia de checkpoint (30min), datasets, objetivo
setup_windows.*     instalador PC (PyTorch cu128 + deps + verificación)
setup_mac.sh        instalador Mac (solo análisis)
src/                orchestrator, processview, checkpointing, gpu_check, data/, models/, recalibration, edge_export
analyze/            scripts para el Mac (view_progress, load_checkpoint, plots)
checkpoints/        (se generan) backups c/30min
artifacts/          (se generan) paquete .zip por etapa/generación → al Mac
processview/        (se genera) status.json + index.html en vivo
data/               (se descargan) los datasets
logs/               (se genera) run.jsonl (log estructurado, legible en Mac)
```

---

## 5. Datasets (≥450 GB disponibles)

`config.yaml → datasets` elige qué descargar. El **núcleo predictivo** (Tier A) son datos
run-to-failure de motor/rodamiento: **NASA IMS, N-CMAPSS, Paderborn, CWRU, MFPT, MetroPT, SKAB**.
Para preentrenar un foundation model está **LOTSA (925 GB)** (desactivado por defecto en el
config; actívalo solo si tienes disco y días). Fuentes: Hugging Face, Kaggle, Zenodo, UCI, NASA
PCoE. Algunos requieren registro (Kaggle: `~/.kaggle/kaggle.json`; SWaT/WADI: solicitud a iTrust).
`src/data/registry.py` tiene todas las URLs.

---

## 6. §Integración — completar el entrenamiento real en el PC

El paquete está **completo y verificado en su orquestación** (descargas, checkpoints, ProcessView,
reanudación, empaquetado, análisis en Mac). El **wiring fino del bucle de entrenamiento real** por
etapa se engancha en `src/orchestrator.py::Orchestrator._real_step`, usando los módulos ya
provistos:
- `src/models/rul.py` (TCN + pérdida asimétrica pro-FN), `health_autoencoder.py`, `classifier.py`
  (boosting GPU + umbral por coste), `foundation_lora.py` (LoRA).
- `src/recalibration.py` (River+ADWIN+replay+EWC+hard-negatives).
- `src/edge_export.py` (distillation → INT8 → ONNX/TensorRT).

Cada dataset tiene un layout distinto (columnas, frecuencia de muestreo), así que el
`_real_step` se ajusta con los datos ya descargados. Mientras tanto, `--dry-run` valida todo el
flujo de principio a fin. **Recomendación:** empieza con `--dry-run`, luego descarga Tier A, y
completa el trainer de la etapa 1a (RUL) primero — es el núcleo del objetivo.

---

## 7. Honestidad

Los datos son **públicos** (para construir y validar los modelos); los datos reales de planta
(Testo + SCADA sub-medidos) llegan después y se transfieren por fine-tuning. Los umbrales de
éxito (≥10 días, ≥90 %) son **objetivos a validar con datos medidos**, no garantías.
