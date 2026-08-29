# 02 — Consumo eléctrico

La línea prioritaria del proyecto. Ejecutable con `python run.py --train-consumo`.

## «Optimizar el consumo» no es un modelo, son cuatro

Y solo uno de ellos produce euros:

| # | Pieza | Qué responde | Implementación |
|---|---|---|---|
| 1 | **Previsión de carga** | Cuánto vas a consumir a H horas vista | Previsor por parches con normalización reversible |
| 2 | **Desagregación NILM** | *Dónde* se va la energía sin instrumentar cada máquina | seq2point CNN |
| 3 | **Detección de desperdicio** | Consumes más de lo que tocaría dadas las condiciones | Autoencoder sobre residuos |
| 4 | **Línea base contrafactual** | **Cuánto habrías consumido si no hubieras hecho nada** | El previsor, con protocolo pre/post |

La cuarta es la importante y es la que casi todo el mundo se salta. **Sin contrafactual
no se puede demostrar un ahorro**: si el consumo baja un 8 % pero ese mes hizo más frío
o se produjo menos, no se ha ahorrado nada. Es el protocolo IPMVP Opción C — se entrena
con datos *anteriores* a la intervención, se predice lo que habría pasado *después*, y
el ahorro es la diferencia. Todo lo demás son insumos para esto.

## El previsor

Arquitectura por parches con **normalización reversible por ventana**: se resta la media
y la desviación del propio contexto antes de predecir, y se vuelve a aplicar después.
Eso es lo que permite que **un solo modelo sirva para 400 edificios de escalas muy
distintas** — aprende la forma del perfil, no el nivel absoluto. Es la idea central de
los previsores modernos tipo PatchTST/N-HiTS en su versión mínima.

Contexto: 168 h (una semana, cubre el ciclo semanal completo). Horizonte: 24 h.

## Resultados (BDG2, 400 series, 6.000 pasos)

| Métrica | Valor | Lectura |
|---|---|---|
| MAE previsión 24 h | 10,38 | — |
| MAE base ingenua (misma hora, semana pasada) | 19,15 | — |
| **Skill vs base** | **+0,4798** | Le gana un 48 % a no hacer nada |
| CV(RMSE) agrupado | 30,16 % | Engañoso, ver abajo |
| **CV(RMSE) mediana por edificio** | **16,05 %** | Cumple ASHRAE G14 (<25 %) |
| **Edificios acreditables** | **73,2 %** | p90 en 44,9 % |
| NMBE | 0,01 % | Sin sesgo |
| Ahorro aparente sin intervención | **0,096 %** | La línea base no inventa ahorros |
| NILM, skill medio vs media por circuito | +0,4477 | Con 3 circuitos en negativo |
| Desperdicio | 1,86 % de ventanas con exceso sostenido | — |

**Aquí sí hay señal.** Un +48 % sobre la base ingenua es una mejora sustancial, y en
consumo eléctrico esa base es dura de batir porque los edificios son muy periódicos. Es
el único bloque del proyecto donde el modelo supera claramente a lo trivial.

**El ahorro se puede certificar.** Sobre datos sin ninguna intervención, la línea base
detecta un 0,096 % de ahorro aparente. Eso valida el método —no inventa ahorros— y fija
el suelo: cualquier ahorro reportado por debajo de ~0,2 % es ruido, no una mejora.

## Lo que la media esconde

**NILM funciona de media y falla en concreto.** El skill global es +0,4477, pero por
circuito:

| Circuito | Skill | MAE |
|---|---|---|
| meter14 | +0,749 | 62,3 W |
| meter4 | +0,583 | 26,4 W |
| ... | | |
| meter12 | −0,014 | 17,4 W |
| **meter9** | **−0,141** | 19,7 W |

Skill negativo significa **peor que predecir la media de ese circuito**. Son cargas casi
constantes o muy estocásticas, donde no hay nada que aprender. Reportar solo el +0,4477
lo escondería.

## Dos barridos que descartaron mejoras plausibles

### La función de pérdida da igual

Se comparó con el mismo dato, la misma semilla y 6.000 pasos:

| Pérdida | MAE | Skill | CV(RMSE) | CV mediana | % acreditables |
|---|---|---|---|---|---|
| **L1** | **10,378** | **+0,4798** | 30,16 % | 16,05 % | 73,2 % |
| Huber | 10,781 | +0,4596 | 30,00 % | 16,07 % | 73,8 % |
| L2 | 11,327 | +0,4323 | 30,37 % | 16,38 % | 74,0 % |

Todo dentro del ruido, y Huber además cuesta 0,02 de skill. Se mantiene **L1**.

La hipótesis era que la L1, al optimizar la mediana, afinaba el caso típico y descuidaba
las puntas —justo lo que castiga el RMSE, que es lo que decide la acreditación—. Sonaba
razonable y **no se sostuvo**.

### Más capas tampoco

| Ancho | Bloques | Params | MAE test | MAE train | Brecha | Skill | % acreditables |
|---|---|---|---|---|---|---|---|
| 256 | 3 | **0,9M** | 10,512 | 10,496 | 0,015 | +0,4731 | **72,8 %** |
| 512 | 1 | 1,3M | 10,537 | 10,474 | 0,062 | +0,4650 | 72,5 % |
| 512 | 3 | 3,4M | 10,378 | 10,223 | 0,155 | +0,4798 | 73,2 % |
| 512 | 6 | 6,6M | 10,573 | 9,074 | 1,500 | +0,4477 | 72,5 % |
| 512 | 12 | 12,9M | 9,759 | 9,544 | 0,215 | +0,4944 | 72,8 % |
| 1024 | 3 | 13,1M | 10,590 | 9,340 | 1,250 | +0,4469 | 73,2 % |
| 1024 | 6 | **25,7M** | 9,775 | 9,695 | 0,080 | +0,4936 | **72,2 %** |

Tres lecturas:

1. **De 0,9M a 25,7M de parámetros —28 veces más— el porcentaje de edificios
   acreditables no se mueve del 72-73 %.** La métrica que decide el negocio es
   completamente insensible a la capacidad.
2. **La brecha entre entrenamiento y test es casi nula en todas las filas.** No hay
   sobreajuste ni infraajuste: el modelo está tocando el techo de lo predecible con
   estos datos.
3. **El comportamiento no es monótono** (512×6 es peor que 512×3 y que 512×12). Eso es
   inestabilidad de optimización, no una tendencia de capacidad, y sitúa la banda de
   ruido en unos ±0,4 de MAE — dentro de la cual caen casi todas las diferencias.

**Corolario práctico:** el modelo de 0,9M de parámetros consigue lo mismo que el de
25,7M. Eso significa que **el modelo que cumple la métrica de negocio cabe en un
dispositivo de edge**, lo cual es una ventaja de producto, no solo una curiosidad.

## Pendiente de medir

El barrido de escala sobre OEDI (100 → 4.000 edificios) contestará la pregunta que queda
abierta: si el techo no está en los parámetros, ¿está en los datos o está en el problema?
Ver [07 — Preguntas abiertas](07-preguntas-abiertas.md).
