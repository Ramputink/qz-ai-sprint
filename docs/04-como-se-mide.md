# 04 — Cómo se mide

El documento más transferible de la bitácora. Casi todos los errores caros del sprint no
fueron de modelado sino **de medida**: números correctamente calculados que no
significaban lo que parecían.

## Regla general: ninguna métrica sola dice nada

Toda cifra va acompañada de **su línea base**. Sin ella no se puede saber si el modelo
aporta algo.

| Contexto | Métrica | Su línea base |
|---|---|---|
| Clasificación desbalanceada | Accuracy | Accuracy de predecir siempre la clase mayoritaria |
| Clasificación desbalanceada | PR-AUC | La prevalencia de la clase positiva |
| Previsión de consumo | MAE | La base ingenua (misma hora, semana pasada) |
| Desagregación NILM | MAE por circuito | El consumo medio de ese circuito |

Dos casos reales del sprint donde la ausencia de la base invirtió la conclusión:

- **C-MAPSS, infravalorado.** «97,95 % de accuracy» parece mediocre cuando no predecir
  nada saca 94,49 %. El ROC-AUC de 0,996 dice que el modelo es excelente.
- **IMS, sobrevalorado.** «82,64 % de accuracy» parece decente. El 83,68 % de las
  ventanas estaban en alarma, así que alarmar siempre ya puntúa 0,84. El ROC-AUC de
  0,4952 dice que no había modelo.

## Accuracy vs AUC

La accuracy de este repositorio se mide **después** de elegir el umbral por coste, así
que mezcla dos cosas distintas: si el modelo ordena bien los casos, y si el corte está
bien puesto. El AUC separa ambas.

Sirvió para resolver una duda concreta: el pliegue del 3er ensayo de IMS daba 1.609
falsos positivos. Si el modelo ordenara bien y solo estuviera mal calibrado, el PR-AUC
sería alto. Estaba a +0,031 de su base — **era ausencia de discriminación, y eso no se
arregla recalibrando**.

Se reporta **PR-AUC además de ROC-AUC** porque con clases desbalanceadas el ROC-AUC se
ve optimista, y el PR-AUC **siempre con su prevalencia al lado**: un PR-AUC de 0,60 es
excelente si la prevalencia es 0,05 y mediocre si es 0,55.

## Cómo se parte train/validación

**Nunca por ventana.** Es el error que infla resultados de forma más silenciosa.

| Dominio | Corte correcto | Por qué |
|---|---|---|
| Run-to-failure | **Por máquina** | Si dos ventanas del mismo motor caen a ambos lados, el modelo ya vio su futuro |
| Consumo | **Temporal (pasado → futuro)** | Con ventanas barajadas se predice el martes habiendo visto el miércoles de esa semana |
| Pocas máquinas | **Leave-one-out, y agrupado** | Ver abajo |

### Agrupar cuando las máquinas comparten destino

Con 12 rodamientos, leave-one-bearing-out **no basta**. Los 4 rodamientos de un ensayo
IMS comparten eje, carga e instante de fallo, y la vibración se transmite por la
carcasa. Al reservar uno, los otros tres le enseñan al modelo cuándo para el ensayo.

El efecto medido: la accuracy mediana pasa de **0,847 a 0,613** al reservar el banco
entero. La mitad del «rendimiento» era fuga.

**Síntoma que lo delata:** las máquinas que realmente fallaron daban peores avisos que
las sanas. Cuando el orden de los resultados contradice el sentido físico, sospechar de
la partición antes que del modelo.

### Declarar los pliegues degenerados

Si la máquina reservada dura menos que el horizonte de aviso exigido, todas sus ventanas
caen del mismo lado y **no hay clase negativa**. Ese pliegue no se puede evaluar. En IMS
pasa con los 4 rodamientos del 2º ensayo (164 h frente a un objetivo de 240 h).

Se reporta como *no evaluable* en vez de colar un número. Antes eso se disolvía en una
media.

## Cuando hay pocas unidades, reportar la distribución

Con 12 rodamientos y 2 de validación, la accuracy medida oscilaba entre **0,338 y 0,773**
cambiando solo la semilla. Un número suelto de ese conjunto no significa nada.

`--cross-validate` entrena un modelo por máquina y reporta **mediana, rango y cuántos
pliegues cumplen el objetivo**. Un pliegue afortunado ya no puede pasar por el resultado
del dataset.

## Las normas del dominio se aplican como dicen las normas

ASHRAE Guideline 14 exige CV(RMSE) < 25 % para aceptar una línea base de ahorro, y se
aplica **a cada emplazamiento**, no a un promedio de cartera.

| | Agrupando 400 series | Por edificio |
|---|---|---|
| CV(RMSE) | 30,16 % | **mediana 16,05 %** |
| Veredicto | Falla | **73,2 % cumplen** |

Medirlo agrupado invirtió el veredicto: con edificios de escalas muy distintas el número
lo dominan unos pocos grandes o erráticos (p90 en 44,9 %). **No hacía falta cambiar el
modelo; hacía falta dejar de promediar cosas que no se promedian.**

## La media esconde la cola

Reportar siempre, junto a la media:

- `lead_time_days_min` — la máquina peor avisada, no la media. Un aviso medio de 12 días
  con un mínimo de 0 no sirve para un contrato.
- `maquinas_sin_aviso` — cuántas no dispararon nunca.
- Skill **por circuito** en NILM — el global de +0,4477 escondía un circuito en −0,141.
- Percentil 90 del CV(RMSE), no solo la mediana.

## La alarma tiene que ser sostenida

La anticipación se mide exigiendo que la alarma **se mantenga hasta el fallo**, no que
salte una vez. Una alarma que va y viene no se atiende en planta. Igual en la detección
de desperdicio: se exigen 3 horas consecutivas de exceso, porque un pico aislado es
ruido de medida.

## El umbral se elige por coste, no por accuracy

`FP + fn_weight·FN`, con `fn_weight = 5`. Un modelo con 95 % de accuracy que se come los
fallos no sirve.

Esto tiene una consecuencia que conviene decidir explícitamente: la Gen 2 de C-MAPSS
cambia **33 falsos negativos menos por 251 falsos positivos más**. Bajo la función de
coste del config es peor; bajo el criterio «no se me puede escapar un fallo» es mejor.
**Hay que decidir cuál manda, y ese número sale del negocio, no del modelo.**

## Cinco hipótesis propias que se cayeron

Todas sonaban razonables. Ninguna sobrevivió al experimento:

| Hipótesis | Predicción | Resultado |
|---|---|---|
| Las features de envolvente arreglarían IMS | Subiría el AUC | Mediana 0,847 → 0,833 |
| La asimetría del techo de RUL era la causa | Subiría al corregirla | 0,539 → 0,545 |
| El horizonte relativo a la vida arreglaría la especificación | Subiría | Los tres criterios **por debajo de 0,5** |
| Huber bajaría el CV(RMSE) bajo el 25 % | Bajaría | 30,16 % → 30,00 %, ruido |
| Más capas mejorarían la previsión | Bajaría el error | 28× parámetros, % acreditables plano |

**En esta máquina un entrenamiento cuesta entre 47 y 228 segundos.** A ese precio, una
rejilla de diez configuraciones es un cuarto de hora. Sale más barato medir que discutir,
y por eso estas cinco hipótesis son datos y no opiniones que seguirían circulando.
