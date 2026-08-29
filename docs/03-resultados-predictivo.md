# 03 — Mantenimiento predictivo

Bloque original del repositorio. Ejecutable con `python run.py` (10 min en RTX 5090 con
los datos ya descargados).

## Objetivo declarado

`config.yaml → target`: avisar del fallo con **≥10 días** de antelación, **≥90 % de
accuracy**, y falsos negativos penalizados **5×** frente a los falsos positivos.

El umbral de alarma no se elige maximizando accuracy sino **minimizando el coste**
`FP + 5·FN`. Un modelo con 95 % de accuracy que se come los fallos no sirve.

## C-MAPSS: el objetivo se cumple

709 motores, 142 reservados para validación, split **por máquina**.

| | Accuracy | FP | FN | Recall fallo | Coste | Aviso medio |
|---|---|---|---|---|---|---|
| XGBoost (baseline) | 0,9794 | 502 | 81 | — | 907 | — |
| Gen 1 (TCN) | 0,9795 | 501 | 80 | 0,9488 | **901** | 12,06 d |
| Gen 2 (TCN +1 bloque) | 0,9718 | 752 | **47** | **0,9699** | 987 | **13,56 d** |

Y la evaluación sin umbral:

| | Valor |
|---|---|
| ROC-AUC | **0,996** |
| PR-AUC | 0,931 |
| Base del PR-AUC (prevalencia) | 0,055 |
| Margen sobre la base | **+0,876** |
| Accuracy de no predecir nada | 0,9449 |

**Hay discriminación real.** El titular «97,95 % de accuracy» suena flojo al lado del
94,49 % que saca no predecir nada, pero el ROC-AUC de 0,996 y un PR-AUC diecisiete veces
por encima de su base dicen que el modelo funciona excelentemente. La accuracy le hacía
un flaco favor.

Dos matices honestos:

- **El baseline de boosting casi empata** (907 vs 901 de coste). Con estos datos y este
  presupuesto, la red neuronal no es todavía el diferenciador.
- **C-MAPSS es simulado.** Son turbinas de gas generadas por un modelo de degradación,
  no medidas de campo.

## IMS: no hay señal

IMS es vibración **real** de rodamiento hasta rotura a 20 kHz — el dato más parecido a
un motor de planta. 3 ensayos, 12 rodamientos, **4 fallos**.

### El número depende del sorteo

Con 12 rodamientos, reservar el 20 % deja 2 de validación. Repitiendo el mismo
entrenamiento cambiando **solo la semilla**:

| Semilla | Val | Accuracy | Aviso |
|---|---|---|---|
| 20260827 | [2, 7] | 0,7732 | 3,66 d |
| 7 | [4, 6] | *no evaluable* | — |
| 42 | [0, 7] | 0,7732 | 3,64 d |
| 1234 | [0, 9] | 0,4552 | 11,13 d |
| 99 | [8, 10] | 0,3379 | 7,64 d |

Accuracy entre **0,34 y 0,77**. Un split ni siquiera se puede evaluar: si los dos
rodamientos reservados salen del 2º ensayo (164 h en total), el horizonte de 10 días no
tiene clase negativa.

### Hay fuga entre rodamientos del mismo banco

Los 4 rodamientos de un ensayo van en el mismo eje, con la misma carga, y el ensayo se
detiene para todos en el mismo instante. Al reservar uno solo, los otros tres están en
entrenamiento con **ese mismo instante de fallo** y una vibración correlacionada a través
de la carcasa.

Se detectó por un síntoma raro: **los rodamientos que realmente rompieron daban los
peores avisos** (1,5 y 1,83 días) mientras los sanos conseguían 10-17 días.

Reservando el **banco entero** (`--cv-group test`):

| Banco reservado | Accuracy | PR-AUC | Base | ROC-AUC | Aviso |
|---|---|---|---|---|---|
| 1er ensayo | 0,8264 | 0,8561 | 0,8368 | **0,4952** | 9,43 d |
| 2º ensayo | *no evaluable* | — | — | — | — |
| 3er ensayo | 0,3993 | 0,4023 | 0,3597 | 0,5837 | 22,4 d |
| **Mediana** | 0,6129 | 0,6292 | 0,5982 | **0,5394** | 15,91 d |

**ROC-AUC 0,54 es una moneda al aire**, y el 1er ensayo sale en 0,4952 —por debajo del
azar—. El PR-AUC queda a +0,031 de su línea base.

Aquel `accuracy 0,8264` era **íntegramente prevalencia**: el 83,68 % de las ventanas de
ese ensayo caen dentro de la ventana de alarma, así que decir «alarma» siempre ya
puntúa 0,84. No había modelo debajo del número.

### Cuatro intentos de arreglarlo, cuatro fracasos

**1. Features de envolvente.** Se implementó el análisis estándar de rodamientos: filtro
paso banda 2-9,5 kHz, envolvente de Hilbert, y altura del pico en las frecuencias de
defecto y sus armónicos. Las frecuencias salen de la geometría documentada del banco
(Rexnord ZA-2115, 16 elementos, 2000 RPM): **BPFO 236,4 · BPFI 296,9 · BSF 139,9 ·
FTF 14,8 Hz**.

| Leave-one-bearing-out | Sin envolvente | Con envolvente |
|---|---|---|
| Accuracy mediana | 0,847 | 0,833 |
| Cumplen ≥0,90 | 3/8 | 3/8 |
| Aviso mediana | 10,33 d | 10,04 d |

No mejora la mediana. Sí cambia mucho por ensayo: en el 1º —fallo de pista interna y
elemento rodante— el rodamiento roto pasa de avisar con **1,5 días a 10,21**; en el 3º
—pista externa, 741 h de degradación lenta— empeora. **Capta el impacto localizado y no
la degradación gradual: ayuda en un modo de fallo y estorba en otro.**

**2. Techo de RUL.** Se detectó una asimetría real: a C-MAPSS se le pedía «¿está en los
últimos 40 ciclos?» (79,5 % de sus ventanas colapsadas en el techo) y a IMS «¿cuántas
horas exactas le quedan, desde la hora 1?» (0,0 % en el techo). Lo segundo es irresoluble
— a un rodamiento sano no se le ve en la vibración cuánto le queda. Corregido, el
ROC-AUC pasó de 0,539 a **0,545**. Nada.

**3. Horizonte relativo a la vida útil.** Pedir 10 días absolutos sobre ensayos que duran
6,8 / 15 / 31 días es pedir el 83 % de la vida mediana; en C-MAPSS son 10 ciclos sobre
176, el 5,7 %. Reformulado como fracción de vida:

| Criterio | ROC-AUC | PR-AUC | Base | Margen |
|---|---|---|---|---|
| 10 días absolutos | 0,5453 | 0,6327 | 0,5982 | +0,034 |
| Último 25 % de vida | **0,4357** | 0,1962 | 0,2500 | −0,025 |
| Último 15 % de vida | **0,4491** | 0,1063 | 0,1505 | −0,012 |
| Último 10 % de vida | **0,4066** | 0,0749 | 0,1007 | +0,007 |

**Los tres criterios relativos quedan por debajo de 0,5**: el modelo ordena los casos al
revés del azar. La reformulación no solo no arregla, empeora.

**4. Validación agrupada.** Eliminó la fuga, pero eso reveló el problema en vez de
resolverlo: la accuracy mediana cayó de 0,847 a 0,613.

### Conclusión

No era la especificación, no era el techo de RUL, no eran las features y no era el corte
de validación. **Con 4 eventos de fallo y 2 bancos independientes utilizables no hay
señal que aprender.** Cualquier trabajo adicional sobre IMS es tiempo perdido hasta que
haya más rodamientos con más modos de fallo.

Lo que haría falta: **XJTU-SY** (15 rodamientos hasta rotura), **FEMTO/PRONOSTIA** (17) y
**Paderborn** (20 GB, verificado disponible). Pasar de 4 a ~35 eventos con varios modos
de fallo. Son decenas de GB — el límite no son los gigabytes, son los **eventos**.

## Otros datasets del bloque

Medidos como modelos secundarios en la etapa 2a:

| Dataset | Unid. val | Accuracy | Aviso medio |
|---|---|---|---|
| `ncmapss` | 4 | 0,9397 | 12,25 d |
| `metropt3` | 1 | 0,8514 | 21,29 d |
| `nasa_ims_bearing` | 2 | 0,7732 | 3,68 d |

`metropt3` tiene **una sola máquina de validación** y `ncmapss` cuatro. Sus cifras están
para el registro, no para prometer nada.
