# 06 — Datasets

Catálogo con estado **verificado**, no declarado. Cada fila se descargó y se preprocesó
de verdad entre el 27 y el 29 de agosto de 2026.

`python run.py --list-data` lista el plan activo según `config.yaml`.

## Consumo eléctrico (prioritario)

| Clave | Tamaño | Qué es | Resultado del preprocesado |
|---|---|---|---|
| `building_data_genome_2` | 595 MB | 1.636 edificios reales, 2 años horarios **+ meteorología + metadatos** | 1.440 series × 17.544 h |
| `low_carbon_london` | 801 MB → 8,5 GB | 5.567 hogares reales de Londres, media hora, 2011-2014 | 2.113 series × 19.864 h · **167.932.474 lecturas** |
| `oedi_comstock` | 18,13 GB | 4.200 edificios de NREL, 15 min, **con desglose por uso final** | 4.000 series × 8.761 h |
| `electricity_load_diagrams` | 261 MB | 370 clientes, 15 min, 4 años. Referencia clásica | 370 series |
| `steel_industry_energy` | 482 KB | Planta siderúrgica: **potencia reactiva**, factor de potencia, CO₂ | 1 serie con covariables de proceso |
| `ampds2` | 312 MB | 21 medidores a 1 min, 2 años | 1.051.200 pasos × 20 submedidas (NILM) |

**Por qué OEDI aporta algo distinto.** BDG2 y Low Carbon London dan el consumo total. OEDI
da además el reparto por uso final de cada edificio. Ejemplo real del primer edificio
procesado:

| Uso | % |
|---|---|
| Iluminación exterior | 45,8 % |
| Equipamiento interior | 32,4 % |
| Iluminación interior | 21,5 % |
| Refrigeración | 0,0 % |

Con BDG2 ese edificio sería una única curva. Con OEDI se puede decir **dónde actuar**,
que es de lo que va optimizar el consumo.

**Por qué `steel_industry_energy` importa a pesar de su tamaño.** Es el único realmente
industrial y trae la potencia reactiva, que es dinero directo en factura y se corrige con
condensadores **sin tocar la producción**. Es una vía de ahorro que no existe en los
datasets de edificios.

## Mantenimiento predictivo

| Clave | Tamaño | Qué aporta | Resultado |
|---|---|---|---|
| `cmapss` | 13 MB | 709 motores run-to-failure con RUL | 138.380 ventanas × 709 unidades |
| `ncmapss` | 15,8 GB | Perfiles de vuelo reales (HDF5) | 1.011 ventanas × 21 unidades |
| `nasa_ims_bearing` | 1,1 GB | 3 ensayos de rodamiento hasta rotura, 20 kHz | 12 rodamientos, **4 fallos** |
| `metropt3` | 220 MB | Compresor de metro, 15 meses, 4 fallos fechados | 3 tramos |
| `cwru_bearing` | 120 MB | Vibración etiquetada por tipo de fallo | 1.045 segmentos |
| `mfpt_bearing` | 50 MB | Baseline + pista interna/externa | 670 segmentos |
| `skab` | 6 MB | Banco con bomba, anomalías etiquetadas | 13.067 registros |

## Fuentes con trampa

Documentadas porque volverán a fallar:

| Dataset | Trampa | Solución aplicada |
|---|---|---|
| **BDG2** | El zip de GitHub trae **punteros de Git LFS**, no CSV. Se extrae sin error y deja ficheros de 3 líneas | Bajar de **Zenodo** (record 3887306) |
| **MFPT** | La URL de mfpt.org devuelve **HTML** con código 200 | Repaquetado de MathWorks |
| **AMPds2** | El id de Dataverse es opaco; el HDF5 va en **blosc** y lo escribió NILMTK | Consultar la API por el DOI; leer con h5py + `hdf5plugin` |
| **Low Carbon London** | El zip usa una compresión que `zipfile` no abre. CSV en formato largo de 8,5 GB | Extraer con **7z**; volcar por bloques sobre matriz preasignada |
| **IMS** | El 3er ensayo cuelga de `3rd_test/4th_test/txt/` y trae **1.876 ficheros posteriores al fallo documentado** | Buscar carpetas por contenido; truncar en el fichero 4.448 |
| **N-CMAPSS** | Zip anidado dentro del zip | Extracción recursiva |
| **UK-DALE** | La URL directa devuelve HTML desde el 28-08-2026 | Marcado `manual`; AMPds2 cubre el papel |
| **Paderborn** | Ficheros `.rar` uno a uno con registro | Marcado `manual`; CWRU+MFPT cubren el papel |
| **BuildingsBench** | Requiere autenticación en Hugging Face (401) | No descargado |

## Lo que NO merece la pena descargar

**LOTSA (925 GB).** Es un corpus de series temporales genéricas para preentrenar modelos
de previsión. No contiene ni un rodamiento roto ni prácticamente datos de consumo
industrial. Además, para obtener el beneficio de un foundation model **no hace falta el
corpus**: `src/models/foundation_lora.py` carga los pesos ya entrenados de Chronos-Bolt o
MOMENT desde Hugging Face, unos cientos de MB. Bajar 925 GB para preentrenar desde cero
en una sola GPU cuando el checkpoint entrenado es gratis cuesta ~9 h de descarga, ~1 TB
de disco y días de cómputo para llegar a un modelo peor.

## Lo que sí habría que descargar (mantenimiento predictivo)

Si se retoma el bloque predictivo, lo que falta son **eventos de fallo**, no gigabytes:

- **XJTU-SY** — 15 rodamientos hasta rotura
- **FEMTO/PRONOSTIA** — 17 rodamientos
- **Paderborn** — 20 GB, verificado disponible

Pasar de 4 a ~35 eventos con varios modos de fallo. Son decenas de GB.
