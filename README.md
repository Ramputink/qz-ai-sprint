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

> **En CADA terminal nueva, activa el entorno ANTES de ejecutar** (si no, verás "Falta PyYAML"):
> ```powershell
> .\.venv\Scripts\Activate.ps1     # el prompt debe empezar por (.venv)
> ```
> Si da error de permisos: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` y reintenta.

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

### Resultados de la ejecución del 27-08-2026 (sprint corto, Tier A)

Objetivo: accuracy ≥90 %, aviso ≥10 días, FN penalizados 5×. Validación **por máquina**.

**Modelo primario (C-MAPSS, 142 motores de validación que el modelo nunca ve):**

| | Accuracy | FP | FN | Recall fallo | Coste | Aviso medio |
|---|---|---|---|---|---|---|
| XGBoost (baseline) | 0,9794 | 502 | 81 | — | 907 | — |
| Gen 1 (TCN) | 0,9795 | 501 | 80 | 0,9488 | **901** | 12,06 d |
| Gen 2 (TCN +1 bloque) | 0,9718 | 752 | **47** | **0,9699** | 987 | **13,56 d** |

**Los mismos modelos sobre el resto de datasets Tier A** (etapa 2a, uno por dataset):

| Dataset | Unid. val | Accuracy | FN | FP | Aviso medio | Aviso mínimo |
|---|---|---|---|---|---|---|
| `ncmapss` | 4 | 0,9397 | 1 | 11 | 12,25 d | 9,0 d |
| `metropt3` | 1 | 0,8514 | 121 | 9 | 21,29 d | 21,29 d |
| `nasa_ims_bearing` | 2 | 0,7732 | 16 | 326 | **3,68 d** | 0,09 d |

**Aviso sobre la fila de IMS: esa cifra no es reproducible, y el problema es de datos.**

Con 12 rodamientos, reservar el 20 % deja 2 de validación y el resultado depende del sorteo:
repitiendo el mismo entrenamiento cambiando solo la semilla, la accuracy iba de **0,338 a
0,773**, y un sorteo ni siquiera era evaluable. Por eso se añadió `--cross-validate`, que
entrena un modelo por máquina y reporta la distribución en vez de un número.

Se probaron las tres mejoras de dominio (28-08-2026): **análisis de envolvente** en las
frecuencias de defecto del rodamiento (BPFO 236,4 / BPFI 296,9 / BSF 139,9 / FTF 14,8 Hz,
calculadas de la geometría Rexnord ZA-2115 a 2000 RPM que documenta el propio dataset),
**media horaria con 72 h de contexto** en vez de instantáneas de 10 min con 10,7 h, y el
**flag de qué rodamiento rompió** de verdad.

Leave-one-bearing-out, 12 pliegues, mismas ventanas:

| | Sin envolvente | Con envolvente |
|---|---|---|
| Pliegues evaluables | 8/12 | 8/12 |
| Accuracy mediana | 0,847 | 0,833 |
| Cumplen ≥0,90 | 3/8 | 3/8 |
| Aviso mediana | 10,33 d | 10,04 d |
| Cumplen ≥10 d | 5/8 | 6/8 |

**La envolvente no mejora la mediana**, pero el desglose importa: en el 1er ensayo (fallo de
pista interna y elemento rodante) el rodamiento que rompió pasa de avisar con **1,5 días a
10,2**; en el 3º (pista externa, 741 h de degradación lenta) empeora. Capta la firma de
impacto localizado y no la degradación gradual. No es una mejora general: **ayuda en un modo
de fallo y estorba en otro.**

Y hay una fuga que el leave-one-bearing-out no elimina: los 4 rodamientos de un banco
comparten eje, carga e instante de fallo, y la vibración se transmite por la carcasa. Al
reservar uno solo, los otros tres están en entrenamiento enseñando cuándo para el ensayo.
Reservando el **banco entero** (`--cv-group test`), sin fuga:

| Banco reservado | Accuracy | FN | FP | Aviso |
|---|---|---|---|---|
| 1er ensayo (2 rotos) | 0,8264 | 12 | 188 | 9,43 d |
| 2º ensayo (1 roto) | *no evaluable* | — | — | — |
| 3er ensayo (1 roto) | 0,3993 | 1 | 1609 | 22,4 d |
| **Mediana** | **0,613** | | | 15,91 d |

**0 de 2 pliegues alcanzan el 90 %.** Ésta es la cifra honesta de IMS, y no se arregla con
más features: el 2º ensayo dura 164 h y el objetivo pide avisar con 240, así que la pregunta
ni se puede formular sobre él; y quedan 2 bancos independientes para estimar todo. Lo que
falta no son gigabytes, son **eventos de fallo**: aquí hay 4.

Reproducir: `python run.py --cross-validate nasa_ims_bearing --cv-group test`

### Qué hay que leer de estos números

1. **El objetivo se cumple en C-MAPSS y no se cumple en IMS.** IMS es vibración real de
   rodamiento hasta rotura — el dato que más se parece a un motor de planta — y ahí el modelo
   da 0,77 de accuracy y 3,7 días de aviso, no 10. El titular «≥90 % y ≥10 días» sale de un
   dataset **simulado** de turbinas.
2. **El baseline de boosting casi empata a la red** (907 vs 901 de coste). Con estos datos y
   este presupuesto, la red neuronal no es todavía el diferenciador.
3. **La media del aviso esconde la cola.** `lead_time_days_min` cae a 0 en varias etapas: hay
   máquinas que avisan tarde aunque la media supere los 10 días. Es la métrica que importa
   para un contrato, no la media.
4. **La recalibración no mejora el coste aquí.** 1a→1b subió de 889 a 901 y 2a→2b de 841 a
   987. Reinyectar los FN con peso 15× (`fn_weight` 5 × `hard_negative_boost` 3) sobrecorrige:
   la Gen 2 cambia 33 falsos negativos menos por 251 falsos positivos más. Si el criterio real
   del cliente es «no se me puede escapar un fallo», la Gen 2 es mejor; bajo la función de
   coste del config, es peor. **Conviene decidir cuál de los dos criterios manda.**
5. **`metropt3` tiene 1 sola máquina de validación y `ncmapss` 4.** Sus cifras no son
   estadísticamente sólidas; están para el registro, no para prometer nada.

Reproducir: `python run.py` (10 min en RTX 5090 con los datos ya descargados).

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
run-to-failure de motor/rodamiento. Estado verificado el 27-08-2026 (descarga + preprocesado
reales, sin credenciales):

| Clave | GB | Qué aporta | Resultado |
|---|---|---|---|
| `cmapss` | 0,013 | 709 motores run-to-failure con RUL | 138.380 ventanas · 709 unidades |
| `ncmapss` | 15,8 | Perfiles de vuelo reales (HDF5) | 1.011 ventanas · 21 unidades |
| `nasa_ims_bearing` | 1,1 | 3 ensayos de rodamiento hasta rotura, 20 kHz | 18.556 ventanas · 12 rodamientos |
| `metropt3` | 0,22 | Compresor de metro, 15 meses, 4 fallos fechados | 1.699 ventanas · 3 tramos |
| `cwru_bearing` | 0,12 | Vibración etiquetada por tipo de fallo | 1.045 segmentos |
| `mfpt_bearing` | 0,05 | Baseline + pista interna/externa | 670 segmentos |
| `skab` | 0,006 | Banco con bomba, anomalías etiquetadas | 13.067 registros |

Tier B/C/D están comentados en el config (sprint corto). **LOTSA son 925 GB**: actívalo solo si
tienes ~1 TB libre y días de margen.

Dos avisos sobre las fuentes:
- **`paderborn_bearing` requiere descarga manual** (ficheros `.rar` uno a uno con registro).
  CWRU + MFPT cubren el mismo papel, así que está desactivado.
- **La descarga directa de mfpt.org dejó de servir el zip** (responde con HTML). El registry
  usa el repaquetado de MathWorks de los mismos datos. El descargador ahora detecta este caso
  en vez de dar por bueno un HTML de 650 KB.

`src/data/registry.py` tiene todas las URLs y su fecha de verificación.

---

## 6. Integración — cómo entrena de verdad

El entrenamiento real está **implementado y ejecutado** (27-08-2026). `_real_step` delega en
`src/trainer.py::StageTrainer`, y la capa de datos vive en `src/data/`:

| Módulo | Qué hace |
|---|---|
| `src/data/registry.py` | Una entrada por dataset: URL, método, tamaño y para qué sirve |
| `src/data/download.py` | Descarga **reanudable** (Range HTTP), extrae zip/7z/rar anidados, marca `_DOWNLOAD_OK.json` para no repetir, y **detecta si el servidor devolvió una página web en vez del fichero** |
| `src/data/preprocess.py` | Un adaptador por dataset → dos productos canónicos en `data/processed/` |
| `src/trainer.py` | El bucle real: RUL, autoencoder de salud, umbral por coste, recalibración, edge y comparativa |

**Los dos productos canónicos** (el trainer no sabe de qué dataset vienen):
- `<clave>_rul.npz` → `X (N,T,F)`, `y_rul`, `unit`, `t_idx` + `hours_per_unit` en el meta.
  El split train/val se hace **por máquina**, nunca por ventana: si dos ventanas del mismo
  motor caen a ambos lados, el modelo ya ha visto su futuro y la métrica miente.
- `<clave>_cls.npz` → `Xf (M,F)`, `y` (0 sano / 1 fallo) para el baseline y el autoencoder.

**Cómo se mide el éxito** (lo hace `StageTrainer.evaluate`):
- La etiqueta es «¿el fallo cae dentro de los `lead_time_days` exigidos?».
- El umbral de alarma se elige **minimizando `FP + fn_weight × FN`**, no maximizando accuracy.
- La anticipación se mide por máquina y exige **alarma sostenida** hasta el fallo: una alarma
  que salta y se retira no cuenta, porque en planta no se atiende.
- Se reportan además `maquinas_sin_aviso` y `lead_time_days_min`: la media puede ser buena y
  aun así haber una máquina que nunca avisó.

**Conversión a días.** Cada dataset tiene su cadencia (`HOURS_PER_UNIT` en `preprocess.py`):
IMS graba cada 10 min y MetroPT-3 se remuestrea a 1 h — ambos **medidos**. En C-MAPSS un ciclo
es un vuelo y el dataset no publica su duración física: se asume **1 ciclo = 1 día** y así
consta en el meta. Los días de anticipación sobre C-MAPSS heredan ese supuesto.

---

## 7. Honestidad

Los datos son **públicos** (para construir y validar los modelos); los datos reales de planta
(Testo + SCADA sub-medidos) llegan después y se transfieren por fine-tuning. Los umbrales de
éxito (≥10 días, ≥90 %) son **objetivos a validar con datos medidos**, no garantías.
