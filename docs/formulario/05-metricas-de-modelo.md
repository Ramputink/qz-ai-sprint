# 05 — Métricas de modelo

Las fórmulas de evaluación. El criterio de cuándo usar cada una está en
[04 — Cómo se mide](../04-como-se-mide.md); aquí están las definiciones.

## Error de regresión

```
err[i] = pred[i] - real[i]

MAE   = mean( |err| )                              misma unidad que y
RMSE  = sqrt( mean( err^2 ) )                      penaliza más los errores grandes
MAPE  = 100 * mean( |err| / |real| )               [%]  inestable si real ~ 0
sMAPE = 200 * mean( |err| / (|real| + |pred|) )    [%]  variante acotada
```

**MAE frente a RMSE no es un detalle de presentación.** El MAE es minimizado por la
mediana y el RMSE por la media, así que un modelo entrenado con L1 afina el caso típico
y se despreocupa de las puntas. Como el RMSE es lo que decide la acreditación ASHRAE,
parecía que cambiar la pérdida ayudaría. **Se midió y no fue así** — ver
[02 — Consumo](../02-resultados-consumo.md).

## Skill score

Ninguna métrica de error significa nada sin su línea base:

```
SS = 1 - Error_modelo / Error_base
```

| Valor | Lectura |
|---|---|
| `SS = 0` | Empatar con no hacer nada |
| `SS < 0` | **Estorbar**: peor que la base trivial |
| `SS = 1` | Perfecto |

Base según el problema:

| Problema | Base |
|---|---|
| Previsión de consumo | `y(t - 168h)` — misma hora, semana pasada |
| Desagregación NILM | El consumo medio de ese circuito |
| Clasificación | Predecir siempre la clase mayoritaria |

```python
# src/consumo.py::metricas   ->  skill_vs_ingenua
```

## Matriz de confusión y coste asimétrico

```
                   Predicho
                sano    fallo
Real  sano       TN       FP
      fallo      FN       TP

accuracy  = (TP + TN) / (TP + TN + FP + FN)
recall    = TP / (TP + FN)      cuántos fallos reales se detectan
precision = TP / (TP + FP)      qué fracción de las alarmas son ciertas
F1        = 2 * P * R / (P + R)
```

**El umbral no se elige maximizando accuracy sino minimizando el coste:**

```
Coste = FP + fn_weight * FN
```

con `fn_weight = 5` en `config.yaml`. **Ese 5 está puesto a ojo** y debería salir de un
número del negocio: cuánto cuesta una parada no prevista frente a una inspección
innecesaria.

```python
# src/models/classifier.py::best_threshold
```

## Prevalencia: la trampa de la accuracy

```
prevalencia = (TP + FN) / N          fracción de positivos

accuracy_trivial = max( prevalencia , 1 - prevalencia )
```

Es la accuracy de no predecir nada. **Una accuracy sin esta cifra al lado es ilegible.**
Dos casos reales del proyecto:

| Caso | Accuracy | Trivial | Lectura real |
|---|---|---|---|
| C-MAPSS | 0,9795 | 0,9449 | Parecía mediocre; el AUC dice que es excelente |
| IMS, 1er ensayo | 0,8264 | **0,8368** | **Peor que no predecir nada** |

## AUC: evaluar sin umbral

La accuracy se mide *después* de elegir el umbral, así que mezcla dos cosas: si el modelo
ordena bien los casos, y si el corte está bien puesto. El AUC separa ambas.

**ROC-AUC** — área bajo la curva TPR frente a FPR:

```
TPR = TP / (TP + FN)          FPR = FP / (FP + TN)
```

Interpretación: probabilidad de que un positivo aleatorio reciba mayor puntuación que un
negativo aleatorio.

| ROC-AUC | Lectura |
|---|---|
| 1,0 | Separación perfecta |
| **0,5** | **Moneda al aire** |
| < 0,5 | Ordena al revés del azar |

**PR-AUC** — área bajo precisión frente a recall. Con clases desbalanceadas el ROC-AUC
se ve optimista porque la clase mayoritaria domina el FPR; el PR-AUC no perdona eso.

**Su línea base es la prevalencia**, y sin ella no se interpreta:

```
PR-AUC_base = prevalencia

margen = PR-AUC - prevalencia       <- esto es lo que aporta el modelo
```

Un PR-AUC de 0,60 es excelente con prevalencia 0,05 y mediocre con prevalencia 0,55.

```python
# src/trainer.py::StageTrainer._auc   ->  pr_auc, pr_auc_base, roc_auc
```

### El caso que lo justificó

El pliegue del 3er ensayo de IMS daba 1.609 falsos positivos. Dos explicaciones posibles:

- El modelo ordena bien y el umbral está mal puesto → **PR-AUC alto** → arreglable
  recalibrando.
- El modelo no discrimina → **PR-AUC pegado a su base** → no arreglable.

Salió a `+0,031` de su base. Segunda explicación.

## Anticipación (lead time)

```
lead_time = ( T_fallo - t_primera_alarma_sostenida ) * horas_por_paso / 24    [días]
```

**Alarma sostenida** significa que se mantiene hasta el fallo, no que salta una vez. Una
alarma que va y viene no se atiende en planta, así que contarla sería medir algo que no
tiene valor operativo.

```python
# src/trainer.py::_lead_time_days
# sustained = np.flatnonzero(np.cumprod(alarm[::-1])[::-1])
```

Se reportan siempre tres cifras, no una:

| Métrica | Por qué |
|---|---|
| `lead_time_days` | La media |
| `lead_time_days_min` | **La máquina peor avisada.** Un aviso medio de 12 días con mínimo de 0 no sirve para un contrato |
| `maquinas_sin_aviso` | Cuántas no dispararon nunca |

## Validación con pocas unidades

Con pocas máquinas, un número suelto depende del sorteo. Se reporta la **distribución**
sobre pliegues leave-one-out:

```
mediana, minimo, maximo, y cuantos pliegues cumplen el objetivo
```

Y se declaran los **pliegues degenerados**: si la máquina reservada dura menos que el
horizonte de aviso, todas sus ventanas caen del mismo lado y no hay clase negativa. Ese
pliegue **no se puede evaluar**, y decirlo es más honesto que promediarlo.

```python
# src/crossval.py::leave_one_unit_out
```

## Diagnóstico de capacidad

```
brecha = MAE_test - MAE_train
```

| Situación | Diagnóstico | Qué hacer |
|---|---|---|
| `train ~= test`, ambos altos | Infraajuste | Más capacidad |
| `train << test` | Sobreajuste | Menos capacidad o más regularización |
| `train ~= test`, y más capacidad no mejora | **Techo de los datos** | Ni una cosa ni la otra: cambiar de enfoque o conseguir mejores datos |

El tercer caso es el del previsor de consumo: la brecha es casi nula en todas las
configuraciones y multiplicar los parámetros por 28 no movió el porcentaje de edificios
acreditables del 72-73 %.

```python
# src/consumo.py::entrenar_previsor  ->  brecha_train_test
```

## Dónde está en el código

| Concepto | Fichero |
|---|---|
| MAE, RMSE, CV(RMSE), NMBE, skill | `src/consumo.py::metricas` |
| Métricas por serie y % acreditables | `src/consumo.py::evaluar_previsor` |
| ROC-AUC, PR-AUC y su base | `src/trainer.py::StageTrainer._auc` |
| Umbral por coste | `src/models/classifier.py::best_threshold` |
| Anticipación sostenida | `src/trainer.py::_lead_time_days` |
| Validación cruzada y pliegues degenerados | `src/crossval.py` |
| Brecha train/test | `src/consumo.py::entrenar_previsor` |
