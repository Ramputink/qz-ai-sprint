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

## OEDI: el techo tampoco está en los datos (29-08-2026)

4.000 edificios de ComStock, 16 entrenamientos, **6,7 minutos** en total.

### Curva de escala — la pregunta que quedaba abierta

| Edificios | Skill | CV(RMSE) mediana | % acreditables | Brecha train/test |
|---|---|---|---|---|
| 100 | +0,3865 | 10,84 % | 90,0 % | 0,965 |
| 400 | +0,3799 | 11,64 % | 90,0 % | 0,527 |
| 1.600 | +0,3642 | 11,79 % | 90,6 % | 0,421 |
| 4.000 | +0,3719 | 11,20 % | 85,7 % | 0,407 |

**Multiplicar por 40 el número de edificios no mejora nada.** El skill se mueve entre
0,364 y 0,387 sin tendencia, y el porcentaje de acreditables tampoco.

> **El MAE de esta tabla NO es comparable entre filas** y por eso no aparece. Cada
> submuestra tiene una mezcla distinta de edificios, y el consumo medio por serie va de
> 0,7 a 4.132 kWh — un factor de 6.000. El skill sí es comparable porque se normaliza
> contra la base ingenua calculada sobre esos mismos datos.

**Conclusión combinada con el barrido de capacidad: el techo no está en los parámetros
ni en el número de edificios.** Está en el problema. Lo que falta no es más de lo mismo,
sino información distinta: condicionar por edificio (metadatos, embedding) o aceptar que
hay una fracción de edificios intrínsecamente poco predecibles.

### Capacidad — confirma lo visto en BDG2

| Ancho | Bloques | Params | Skill | % acreditables |
|---|---|---|---|---|
| 256 | 3 | **0,88M** | **+0,3896** | 90,5 % |
| 512 | 3 | 3,3M | +0,3799 | 90,0 % |
| 512 | 12 | 12,8M | +0,3640 | 90,8 % |
| 1024 | 6 | **25,6M** | **+0,3583** | 90,8 % |

El modelo más pequeño es el **mejor** en skill. 29 veces más parámetros lo empeoran.

### Preprocesado — solo una transformación aporta

| Variante | Skill | Brecha train/test | Lectura |
|---|---|---|---|
| **log1p** | **+0,3958** | **0,175** | **Mejor, y reduce la brecha a un tercio** |
| log1p + robusta | +0,3942 | 0,162 | Equivalente |
| referencia | +0,3799 | 0,527 | — |
| contexto 336 h | +0,3794 | 1,642 | Igual, pero empieza a sobreajustar |
| normalización robusta | +0,3758 | 0,540 | Ligeramente peor |
| contexto 720 h | +0,3429 | 2,824 | **Peor: sobreajusta claramente** |
| horizonte 168 h | **+0,0362** | 2,992 | **Se desploma** |
| ~~sin meteorología~~ | +0,3799 | 0,527 | **No aplica**: OEDI no trae meteorología |

### Ningún preprocesado transfiere entre datasets

Al correr la misma ablación sobre BDG2 los efectos **se invierten**:

| Variante | Skill OEDI | Skill BDG2 |
|---|---|---|
| referencia | +0,3799 | +0,4798 |
| `log1p` | **+0,3958** (mejor) | **+0,4559** (peor) |
| normalización robusta | +0,3758 | +0,4744 |
| contexto 336 h | +0,3794 | +0,4827 |
| contexto 720 h | **+0,3429** (peor) | **+0,4836** (mejor) |
| `log1p` + robusta | +0,3942 | +0,4372 (el peor) |
| horizonte 168 h | +0,0362 | +0,2110 |

`log1p` ayuda en OEDI y estorba en BDG2. Ampliar el contexto estorba en OEDI y ayuda en
BDG2. **Exactamente al revés en cada caso.**

Tiene explicación: OEDI es simulado, sin ruido de contador, y sus series abarcan un
rango de consumo enorme (de 0,7 a 4.132 kWh de media), que es justo donde comprimir la
cola con `log1p` aporta. BDG2 son edificios medidos, ya limpiados, y la normalización
por serie que hay antes ya resuelve la escala. Y OEDI cubre un año frente a los dos de
BDG2, así que una ventana de 720 h se come una fracción mucho mayor del histórico y
sobreajusta.

**La conclusión útil no es «usa log1p», es que no hay un preprocesado universalmente
bueno aquí.** Todas las diferencias son de ±0,02 de skill —del orden de la banda de
ruido que ya habíamos establecido— y cambian de signo según el dataset. Ajustar el
preprocesado al dataset concreto es afinar dentro del ruido; no es la palanca.

**Lo único que se sostiene en los dos:** a una semana vista el modelo casi no aporta
(+0,036 en OEDI, +0,211 en BDG2, frente a ~+0,38 y ~+0,48 a 24 h). La base ingenua es
durísima en horizontes largos, porque a siete días la mejor predicción sigue siendo «lo
mismo que la semana pasada». **El horizonte de 24 h no es una limitación técnica: es
donde el modelo tiene algo que decir.**

> **Precaución con el 90 % de acreditables de OEDI frente al 73 % de BDG2.** ComStock es
> **simulado**: no tiene ruido de contador ni comportamiento humano irregular. Ese 90 %
> mide la predecibilidad de una simulación, no la de una cartera real. **El número que
> hay que llevarse a un contrato es el 73 % de BDG2**, que son edificios medidos.
