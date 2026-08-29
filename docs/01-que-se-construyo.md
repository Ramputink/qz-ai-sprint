# 01 — Qué se construyó

## Estado inicial (27-08-2026)

El paquete estaba **completo en su orquestación y vacío en su ejecución**. Verificado:

| Comprobación | Resultado |
|---|---|
| `run.py --gpu-check` | RTX 5090, torch 2.11+cu128, `sm_120`, bf16 correcto |
| `run.py --dry-run` | Completaba las 4 etapas con métricas **simuladas por una fórmula** |
| `run.py --list-data` | `ModuleNotFoundError: No module named 'src.data'` |
| `run.py` (real) | `NotImplementedError` en el paso 0 de la etapa 1a |

Faltaban dos piezas: el paquete `src/data/` entero —referenciado en `run.py` y en el
orquestador, pero ausente del disco— y `_real_step`, que era un stub que lanzaba
excepción. Los módulos de modelos (`rul.py`, `classifier.py`, `health_autoencoder.py`,
`recalibration.py`, `edge_export.py`) sí estaban implementados y listos para engancharse.

El README describía este hueco como «wiring pendiente». En la práctica significaba que
**ninguna cifra del proyecto procedía de un entrenamiento real**.

## Piezas añadidas

### Capa de datos

| Módulo | Función |
|---|---|
| `src/data/registry.py` | Catálogo de datasets: URL, método, tamaño, para qué sirve. Todas las fuentes verificadas con fecha |
| `src/data/download.py` | Descarga reanudable por Range HTTP, extracción recursiva zip/7z/rar, marcadores idempotentes, y **detección de fuentes que devuelven algo que no son los datos** |
| `src/data/preprocess.py` | Siete adaptadores de mantenimiento predictivo a dos productos canónicos |
| `src/data/consumption.py` | Cinco adaptadores de consumo eléctrico |
| `src/data/oedi.py` | Descarga y preprocesado del catálogo NREL (bucket S3 público) |
| `src/data/features.py` | Primitivas de señal, incluidas las frecuencias de defecto de rodamiento |

### Entrenamiento

| Módulo | Función |
|---|---|
| `src/trainer.py` | Entrenador por etapa del bloque predictivo: TCN de RUL, autoencoder de salud, umbral por coste, recalibración, export a edge y comparativa entre generaciones |
| `src/consumo.py` | Previsor de carga, seq2point para NILM, detector de desperdicio y línea base contrafactual |
| `src/crossval.py` | Validación cruzada dejando fuera una máquina o un grupo de máquinas |
| `src/ablacion.py` | Barrido de variantes de preprocesado con el mismo dato y la misma semilla |
| `scripts/barridos_oedi.py` | Barridos de escala, capacidad y preprocesado sobre OEDI |
| `scripts/esperar_gpu_y_barrer.sh` | Planificador que espera a que se libere la GPU |

### Formatos canónicos

El punto de diseño que más simplifica todo: **el entrenador no sabe de qué dataset
vienen los datos**. Cada adaptador produce uno de tres formatos.

**Producto RUL** (`<clave>_rul.npz`) — trayectorias hasta el fallo:
```
X       (N, T, F)  ventanas deslizantes
y_rul   (N,)       vida útil restante en pasos del dataset
unit    (N,)       id de máquina, para partir train/val POR MÁQUINA
t_idx   (N,)       posición dentro de la trayectoria
```
El meta lleva `hours_per_unit`, que traduce esos pasos a días de anticipación.

**Producto CONSUMO** (`<clave>_consumo.npz`) — series de carga:
```
y             (S, T)     carga por serie
time_feats    (T, Ct)    calendario cíclico
weather       (W, T, Cw) meteorología por emplazamiento
timestamps    (T,)       epoch, imprescindible para el protocolo pre/post
```
Se guardan **las series, no las ventanas**: 1.578 edificios × 17.544 horas caben en
110 MB, mientras que sus ventanas ocuparían gigabytes. El entrenador las trocea en GPU
sobre la marcha, lo que además permite cambiar el horizonte sin reprocesar nada.

**Producto NILM** (`<clave>_nilm.npz`) — acometida y submedidas alineadas.

## Integración con el trabajo paralelo

A mitad del sprint, `origin/master` tenía **cuatro commits con una implementación
paralela de la misma capa de datos**. No era un fast-forward, era una colisión. Se
resolvió conservando lo mejor de cada lado en vez de descartar uno:

- **De `origin/master`**: `device.py` (CUDA/MPS/CPU, permite correr el mismo código en
  el Mac), `stage_a.py` (SKAB) y `stage_rtf.py` (MetroPT-3 con **split temporal**, mejor
  criterio que el nuestro por máquina para un dataset de un solo compresor), sus
  cargadores autocontenidos y `analyze/compare.py`.
- **De esta rama**: la capa de datos, por ser superconjunto (14 datasets frente a 3,
  descarga reanudable, extracción anidada, detección de fuentes rotas).

Las utilidades de señal del remoto se conservaron en `src/data/features.py`, separadas
del pipeline porque son de otro nivel: primitivas sobre una señal, no orquestación de
datasets. Entre ellas venía `bearing_fault_freqs`, que resultó ser exactamente la mejora
que se iba a implementar para IMS.
