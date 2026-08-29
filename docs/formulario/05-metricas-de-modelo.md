# 05 — Métricas de modelo

Las fórmulas de evaluación. El criterio de cuándo usar cada una está en
[04 — Cómo se mide](../04-como-se-mide.md); aquí están las definiciones.

## Error de regresión

Con $e_i = \hat{y}_i - y_i$:

$$
\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n} \left|e_i\right|
\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^{2}}
$$

$$
\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^{n} \frac{|e_i|}{|y_i|}
\qquad
\mathrm{sMAPE} = \frac{200}{n}\sum_{i=1}^{n} \frac{|e_i|}{|y_i| + |\hat{y}_i|}
$$

El MAPE es inestable cuando $y_i \approx 0$, que en consumo pasa de madrugada; la
variante simétrica acota ese problema.

**MAE frente a RMSE no es un detalle de presentación.** El MAE es minimizado por la
mediana y el RMSE por la media, así que un modelo entrenado con $L_1$ afina el caso
típico y se despreocupa de las puntas. Como el RMSE es lo que decide la acreditación
ASHRAE, parecía que cambiar la pérdida ayudaría. **Se midió y no fue así** — ver
[02 — Consumo](../02-resultados-consumo.md).

## Skill score

Ninguna métrica de error significa nada sin su línea base:

$$
\mathrm{SS} = 1 - \frac{\mathcal{E}_{\text{modelo}}}{\mathcal{E}_{\text{base}}}
$$

| Valor | Lectura |
|---|---|
| $\mathrm{SS} = 0$ | Empatar con no hacer nada |
| $\mathrm{SS} < 0$ | **Estorbar**: peor que la base trivial |
| $\mathrm{SS} = 1$ | Perfecto |

Base según el problema:

| Problema | Base |
|---|---|
| Previsión de consumo | $y(t - 168\,\text{h})$ — misma hora, semana pasada |
| Desagregación NILM | El consumo medio de ese circuito |
| Clasificación | Predecir siempre la clase mayoritaria |

```python
# src/consumo.py::metricas   ->  skill_vs_ingenua
```

## Matriz de confusión y coste asimétrico

$$
\begin{array}{c|cc}
 & \hat{y}=\text{sano} & \hat{y}=\text{fallo} \\ \hline
y=\text{sano} & \mathrm{TN} & \mathrm{FP} \\
y=\text{fallo} & \mathrm{FN} & \mathrm{TP}
\end{array}
$$

$$
\text{accuracy} = \frac{\mathrm{TP}+\mathrm{TN}}{\mathrm{TP}+\mathrm{TN}+\mathrm{FP}+\mathrm{FN}}
\qquad
\text{recall} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
$$

$$
\text{precision} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}
\qquad
F_1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
$$

**El umbral no se elige maximizando accuracy sino minimizando el coste:**

$$
\mathcal{C}(\theta) = \mathrm{FP}(\theta) + \lambda_{\mathrm{FN}} \cdot \mathrm{FN}(\theta)
\qquad
\theta^{*} = \arg\min_{\theta} \mathcal{C}(\theta)
$$

con $\lambda_{\mathrm{FN}} = 5$ en `config.yaml`. **Ese 5 está puesto a ojo** y debería
salir de un número del negocio: cuánto cuesta una parada no prevista frente a una
inspección innecesaria.

```python
# src/models/classifier.py::best_threshold
```

## Prevalencia: la trampa de la accuracy

$$
\pi = \frac{\mathrm{TP}+\mathrm{FN}}{N}
\qquad
\text{accuracy}_{\text{trivial}} = \max(\pi,\ 1-\pi)
$$

Es la accuracy de no predecir nada. **Una accuracy sin esta cifra al lado es ilegible.**
Dos casos reales del proyecto:

| Caso | Accuracy | Trivial | Lectura real |
|---|---|---|---|
| C-MAPSS | 0,9795 | 0,9449 | Parecía mediocre; el AUC dice que es excelente |
| IMS, 1er ensayo | 0,8264 | **0,8368** | **Peor que no predecir nada** |

## AUC: evaluar sin umbral

La accuracy se mide *después* de elegir el umbral, así que mezcla dos cosas: si el modelo
ordena bien los casos, y si el corte está bien puesto. El AUC separa ambas.

**ROC-AUC** — área bajo la curva de $\mathrm{TPR}$ frente a $\mathrm{FPR}$:

$$
\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}
\qquad
\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP}+\mathrm{TN}}
$$

$$
\mathrm{AUC}_{\mathrm{ROC}} = \int_{0}^{1} \mathrm{TPR}\big(\mathrm{FPR}^{-1}(u)\big)\, du
= \Pr\big(s(x^{+}) > s(x^{-})\big)
$$

Es decir: la probabilidad de que un positivo aleatorio reciba mayor puntuación que un
negativo aleatorio.

| $\mathrm{AUC}_{\mathrm{ROC}}$ | Lectura |
|---|---|
| $1{,}0$ | Separación perfecta |
| $\mathbf{0{,}5}$ | **Moneda al aire** |
| $< 0{,}5$ | Ordena al revés del azar |

**PR-AUC** — área bajo precisión frente a recall. Con clases desbalanceadas el ROC-AUC
se ve optimista porque la clase mayoritaria domina el $\mathrm{FPR}$; el PR-AUC no
perdona eso. **Su línea base es la prevalencia**, y sin ella no se interpreta:

$$
\mathrm{AUC}_{\mathrm{PR}}^{\text{base}} = \pi
\qquad
\text{margen} = \mathrm{AUC}_{\mathrm{PR}} - \pi
$$

Un PR-AUC de 0,60 es excelente con $\pi = 0{,}05$ y mediocre con $\pi = 0{,}55$.

```python
# src/trainer.py::StageTrainer._auc   ->  pr_auc, pr_auc_base, roc_auc
```

### El caso que lo justificó

El pliegue del 3er ensayo de IMS daba 1.609 falsos positivos. Dos explicaciones posibles:

- El modelo ordena bien y el umbral está mal puesto → **PR-AUC alto** → arreglable
  recalibrando.
- El modelo no discrimina → **PR-AUC pegado a su base** → no arreglable.

Salió a $+0{,}031$ de su base. Segunda explicación.

## Anticipación (lead time)

$$
\text{lead} = \frac{\left(T_{\text{fallo}} - t_{\text{alarma}}\right)\cdot h_{\text{paso}}}{24}
\quad [\text{días}]
$$

donde $t_{\text{alarma}}$ es el **primer instante de alarma sostenida**:

$$
t_{\text{alarma}} = \min\Big\{\,t \;\Big|\; \hat{y}(\tau) \le \theta \;\; \forall\, \tau \in [t,\ T_{\text{fallo}}]\Big\}
$$

Alarma sostenida significa que **se mantiene hasta el fallo**, no que salta una vez. Una
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
sobre pliegues leave-one-out: mediana, mínimo, máximo y cuántos pliegues cumplen el
objetivo.

Y se declaran los **pliegues degenerados**: si la máquina reservada dura menos que el
horizonte de aviso, todas sus ventanas caen del mismo lado y no hay clase negativa,

$$
\pi = 0 \quad\text{o}\quad \pi = 1 \;\Longrightarrow\; \text{pliegue no evaluable}
$$

Decirlo es más honesto que promediarlo.

```python
# src/crossval.py::leave_one_unit_out
```

## Diagnóstico de capacidad

$$
\Delta = \mathrm{MAE}_{\text{test}} - \mathrm{MAE}_{\text{train}}
$$

| Situación | Diagnóstico | Qué hacer |
|---|---|---|
| $\Delta \approx 0$, ambos altos | Infraajuste | Más capacidad |
| $\Delta \gg 0$ | Sobreajuste | Menos capacidad o más regularización |
| $\Delta \approx 0$ y más capacidad no mejora | **Techo de los datos** | Ni una cosa ni la otra: cambiar de enfoque o conseguir mejores datos |

El tercer caso es el del previsor de consumo: la brecha es casi nula en todas las
configuraciones y multiplicar los parámetros por 28 no movió el porcentaje de edificios
acreditables del 72–73 %.

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
