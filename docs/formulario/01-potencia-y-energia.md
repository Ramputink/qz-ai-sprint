# 01 — Potencia y energía

## Las tres potencias

En corriente alterna la potencia se descompone en tres magnitudes que **no son
intercambiables** y que se facturan de forma distinta.

| Símbolo | Nombre | Unidad | Qué es |
|---|---|---|---|
| $P$ | potencia activa | $[\text{W}]$ | trabajo útil: mueve motores, ilumina, calienta |
| $Q$ | potencia reactiva | $[\text{VAr}]$ | magnetiza bobinados; no produce trabajo, pero circula |
| $S$ | potencia aparente | $[\text{VA}]$ | la que dimensiona cables, transformadores y contrato |

Se relacionan por el **triángulo de potencias**:

$$
S^2 = P^2 + Q^2 \qquad\Longrightarrow\qquad S = \sqrt{P^2 + Q^2}
$$

La reactiva es la magnitud contraintuitiva: **no hace trabajo pero ocupa la
instalación**. Un motor con mucha reactiva obliga a dimensionar cables y transformador
para una corriente que no se convierte en producción, y la compañía la penaliza.

## Factor de potencia

$$
\cos\varphi \;=\; \frac{P}{S} \;=\; \frac{P}{\sqrt{P^2 + Q^2}}
\qquad \text{(adimensional, } 0 \le \cos\varphi \le 1)
$$

Un $\cos\varphi = 1$ significa que toda la potencia que circula hace trabajo. Un
$\cos\varphi = 0{,}8$ significa que el 20 % de la capacidad de la instalación se está
gastando en mover reactiva.

**Dos factores de potencia distintos**, y conviene no confundirlos:

$$
\text{DPF} = \cos\varphi_1 \quad\text{(entre las fundamentales de } V \text{ e } I)
\qquad\qquad
\text{APF} = \frac{P}{S} \quad\text{(incluyendo armónicos)}
$$

Con cargas lineales coinciden. Con variadores de frecuencia, rectificadores o
iluminación LED, el APF es menor que el DPF porque hay armónicos que engordan $S$ sin
aportar a $P$. **El dataset `steel_industry_energy` trae los dos por separado**, y su
diferencia es un indicador directo de contaminación armónica.

## Sistemas trifásicos

$$
P = \sqrt{3}\; U_L I_L \cos\varphi
\qquad
Q = \sqrt{3}\; U_L I_L \sin\varphi
\qquad
S = \sqrt{3}\; U_L I_L
$$

$U_L$ es la tensión **entre fases** (400 V en baja tensión industrial europea), no la de
fase a neutro (230 V). Confundirlas introduce un factor $\sqrt{3} \approx 1{,}732$ — el
error de cálculo más común en instalaciones.

## Energía

La energía es la integral de la potencia en el tiempo, y es lo que se factura:

$$
E = \int_{t_0}^{t_1} P(t)\, dt
\qquad\xrightarrow{\text{muestreo uniforme}}\qquad
E = \sum_{i} P_i \,\Delta t \quad [\text{Wh}]
$$

Con medidas cada 15 minutos ($\Delta t = 0{,}25\ \text{h}$), la energía horaria es la
suma de las cuatro lecturas por su intervalo. **En este proyecto se agregan siempre a
hora** para que datasets de distinta cadencia sean comparables: si uno va a 15 min y otro
a hora, cualquier diferencia de error puede deberse a la resolución y no al modelo.

```python
# src/data/consumption.py — prep_electricity_load, prep_steel
df.resample("1h").sum()    # energía: se SUMAN los intervalos
df.resample("1h").mean()   # potencia o temperatura: se PROMEDIAN
```

**Sumar potencias o promediar energías es un error silencioso** que escala el resultado
por un factor constante y no salta en ninguna validación.

## Energía reactiva

Igual que la activa, pero acumulando $Q$:

$$
E_Q = \sum_i Q_i \,\Delta t \quad [\text{VArh}]
$$

El dataset de la siderúrgica distingue reactiva **inductiva** (retrasada, la de motores
y transformadores) y **capacitiva** (adelantada, la de baterías de condensadores
sobredimensionadas o de cables largos en vacío). Ambas se penalizan, y la segunda
sorprende a mucha gente: **pasarse compensando también cuesta dinero**.

| Columna del dataset | Tipo |
|---|---|
| `Lagging_Current_Reactive.Power_kVarh` | reactiva inductiva |
| `Leading_Current_Reactive_Power_kVarh` | reactiva capacitiva |

## Dónde está en el código

| Concepto | Fichero |
|---|---|
| Agregación energía/potencia a hora | `src/data/consumption.py` |
| Reactiva, DPF y APF como covariables de proceso | `src/data/consumption.py::prep_steel` |
| Potencia activa de AMPds2 (columna 5 de 11) | `src/data/consumption.py::prep_ampds2` |

El HDF5 de AMPds2 guarda por circuito, en este orden:

| Índice | Magnitud | | Índice | Magnitud |
|---|---|---|---|---|
| 0 | voltage | | 6 | energy (active) |
| 1 | current | | 7 | power (reactive) |
| 2 | frequency | | 8 | energy (reactive) |
| 3 | pf (displacement) | | 9 | power (apparent) |
| 4 | power factor (apparent) | | 10 | energy (apparent) |
| **5** | **power (active)** ← la que se usa | | | |

Los nombres no se adivinan: se leen del atributo `non_index_axes` del propio fichero.
Suponer el orden por convención habría dado un modelo entrenado sobre la tensión.
