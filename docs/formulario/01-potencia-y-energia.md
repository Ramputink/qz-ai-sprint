# 01 — Potencia y energía

## Las tres potencias

En corriente alterna la potencia se descompone en tres magnitudes que **no son
intercambiables** y que se facturan de forma distinta.

```
P  potencia activa     [W]    trabajo útil: mueve motores, ilumina, calienta
Q  potencia reactiva   [VAr]  magnetiza bobinados; no produce trabajo, pero circula
S  potencia aparente   [VA]   la que dimensiona cables, transformadores y contrato
```

Se relacionan por el **triángulo de potencias**:

```
S^2 = P^2 + Q^2

S = sqrt(P^2 + Q^2)
```

La reactiva es la magnitud contraintuitiva: **no hace trabajo pero ocupa la
instalación**. Un motor con mucha reactiva obliga a dimensionar cables y transformador
para una corriente que no se convierte en producción, y la compañía la penaliza.

## Factor de potencia

```
cos(phi) = P / S = P / sqrt(P^2 + Q^2)          [adimensional, 0..1]
```

Un `cos(phi) = 1` significa que toda la potencia que circula hace trabajo. Un
`cos(phi) = 0,8` significa que el 20 % de la capacidad de la instalación se está
gastando en mover reactiva.

**Dos factores de potencia distintos**, y conviene no confundirlos:

```
DPF  factor de desplazamiento  = cos(phi)  entre las fundamentales de V e I
APF  factor de potencia real   = P / S     incluyendo armónicos
```

Con cargas lineales coinciden. Con variadores de frecuencia, rectificadores o
iluminación LED, el APF es menor que el DPF porque hay armónicos que engordan `S` sin
aportar a `P`. **El dataset `steel_industry_energy` trae los dos por separado**, y su
diferencia es un indicador directo de contaminación armónica.

## Sistemas trifásicos

```
P = sqrt(3) * U_linea * I_linea * cos(phi)      [W]
Q = sqrt(3) * U_linea * I_linea * sin(phi)      [VAr]
S = sqrt(3) * U_linea * I_linea                 [VA]
```

`U_linea` es la tensión entre fases (400 V en baja tensión industrial europea), no la
de fase a neutro (230 V). Confundirlas introduce un factor `sqrt(3) = 1,732` — el error
de cálculo más común en instalaciones.

## Energía

La energía es la integral de la potencia en el tiempo, y es lo que se factura:

```
E = integral( P(t) dt )                          [Wh]

E = sum( P[i] * dt )     para muestreo uniforme   [Wh]
```

Con medidas cada 15 minutos (`dt = 0,25 h`), la energía horaria es la suma de las cuatro
lecturas por su intervalo. **En este proyecto se agregan siempre a hora** para que
datasets de distinta cadencia sean comparables: si uno va a 15 min y otro a hora,
cualquier diferencia de error puede deberse a la resolución y no al modelo.

```python
# src/data/consumption.py — prep_electricity_load, prep_steel
df.resample("1h").sum()    # energía: se SUMAN los intervalos
df.resample("1h").mean()   # potencia o temperatura: se PROMEDIAN
```

**Sumar potencias o promediar energías es un error silencioso** que escala el resultado
por un factor constante y no salta en ninguna validación.

## Energía reactiva

Igual que la activa, pero acumulando `Q`:

```
E_q = sum( Q[i] * dt )                           [VArh]
```

El dataset de la siderúrgica distingue reactiva **inductiva** (retrasada, la de motores
y transformadores) y **capacitiva** (adelantada, la de baterías de condensadores
sobredimensionadas o de cables largos en vacío). Ambas se penalizan, y la segunda
sorprende a mucha gente: **pasarse compensando también cuesta dinero**.

```
Lagging_Current_Reactive.Power_kVarh   reactiva inductiva
Leading_Current_Reactive_Power_kVarh   reactiva capacitiva
```

## Dónde está en el código

| Concepto | Fichero |
|---|---|
| Agregación energía/potencia a hora | `src/data/consumption.py` |
| Reactiva, DPF y APF como covariables de proceso | `src/data/consumption.py::prep_steel` |
| Potencia activa de AMPds2 (columna 5 de 11) | `src/data/consumption.py::prep_ampds2` |

El HDF5 de AMPds2 guarda por circuito, en este orden:

```
0 voltage   1 current   2 frequency   3 pf(d)   4 power factor(apparent)
5 power(active)   6 energy(active)   7 power(reactive)   8 energy(reactive)
9 power(apparent) 10 energy(apparent)
```

Los nombres no se adivinan: se leen del atributo `non_index_axes` del propio fichero.
Suponer el orden por convención habría dado un modelo entrenado sobre la tensión.
