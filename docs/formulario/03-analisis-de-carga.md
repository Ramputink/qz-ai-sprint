# 03 — Análisis de carga

Los indicadores que describen *cómo* consume una instalación, no solo *cuánto*. Son los
que convierten una curva en un diagnóstico.

## Carga base y carga variable

$$
P_{\text{base}} = \mathrm{P}_{5}\big(P(t)\big)
\qquad
P_{\text{var}}(t) = P(t) - P_{\text{base}}
$$

donde $\mathrm{P}_{5}$ es el percentil 5. Se usa un percentil bajo y no el mínimo porque
el mínimo suele ser un hueco de medida. **Una carga base alta en una planta que para los
fines de semana es el hallazgo más rentable de una auditoría**: son equipos encendidos
sin producir.

## Factor de carga

$$
\mathrm{FC} = \frac{E_{\text{periodo}}}{P_{\max} \cdot h_{\text{periodo}}}
\qquad 0 \le \mathrm{FC} \le 1
$$

Relaciona lo consumido con lo que se habría consumido al máximo todo el rato.

| $\mathrm{FC}$ | Lectura |
|---|---|
| Cercano a 1 | Consumo plano; poco margen de desplazamiento de carga |
| Bajo (0,3–0,5) | Muy picudo: **el término de potencia domina la factura** |

Un factor de carga bajo señala que el ahorro está en **aplanar los picos**, no en
reducir la energía total. Son dos intervenciones distintas y se confunden a menudo.

## Curva de duración de carga

Las potencias del periodo ordenadas de mayor a menor:

$$
\mathrm{CDC}(k) = P_{(k)}, \qquad P_{(1)} \ge P_{(2)} \ge \dots \ge P_{(n)}
$$

Responde a «¿cuántas horas al año supero $X$ kW?». Es la herramienta para dimensionar la
potencia contratada: si solo se superan 400 kW durante 20 horas al año, puede salir más
barato pagar los excesos que contratar esa potencia.

## Grados-día

Cuantifican la demanda térmica y son la variable explicativa principal en climatización.

$$
\mathrm{HDD} = \sum_{d} \max\!\left(T_{\text{base}} - \bar{T}_d,\; 0\right)
\qquad
\mathrm{CDD} = \sum_{d} \max\!\left(\bar{T}_d - T_{\text{base}},\; 0\right)
\qquad [^\circ\mathrm{C}\cdot\text{día}]
$$

$T_{\text{base}}$ es la temperatura a partir de la cual el edificio necesita climatizar.
**Es un supuesto, no una medida**: se usa 15–18 °C por convenio, pero el valor correcto
depende del aislamiento y de las cargas internas, y se puede estimar ajustándolo para
maximizar la correlación con el consumo.

En este proyecto no se calculan grados-día explícitamente: se le pasa al modelo la
temperatura horaria cruda y que él aprenda la relación, que además no es lineal (hay un
tramo muerto entre calefacción y refrigeración).

```python
# src/data/consumption.py::prep_bdg2
wx_cols = ["airTemperature", "dewTemperature", "windSpeed",
           "cloudCoverage", "precipDepth1HR"]
```

La **temperatura de rocío** va incluida porque la carga de deshumidificación depende de
la humedad, no solo de la temperatura seca, y en climas húmedos es una fracción grande
del consumo de refrigeración.

## Intensidad de uso energético (EUI)

$$
\mathrm{EUI} = \frac{E_{\text{anual}}}{A} \qquad [\text{kWh}/\text{m}^2/\text{año}]
$$

Es lo que permite **comparar edificios de distinto tamaño**. Sin normalizar por
superficie, un edificio grande siempre parece peor que uno pequeño.

BDG2 trae la superficie en sus metadatos (`sqm`), así que el EUI es calculable y es el
camino natural para investigar el 26,8 % de edificios no acreditables — ver
[07 — Preguntas abiertas](../07-preguntas-abiertas.md).

## Estacionalidad y la base ingenua

Un edificio es fuertemente periódico en dos escalas: diaria y semanal. De ahí que la
base contra la que hay que ganar sea:

$$
\hat{P}_{\text{ingenua}}(t) = P(t - 168\,\text{h})
\qquad\text{«lo mismo que a esta hora la semana pasada»}
$$

$168\ \text{h} = 7$ días. **Se elige la semana y no el día porque el ciclo semanal
incluye el fin de semana**: predecir un lunes con los datos del domingo falla
sistemáticamente.

Esta base es dura de batir. El modelo del proyecto le gana un 48 % en MAE, y ese es el
número que mide si el modelo aporta algo — no el MAE absoluto.

## Codificación cíclica del calendario

La hora 23 y la hora 0 son contiguas, pero un modelo alimentado con los números 23 y 0
las ve máximamente distantes. Se codifican sobre el círculo:

$$
\left(\sin\frac{2\pi h}{24},\; \cos\frac{2\pi h}{24}\right)
\qquad
\left(\sin\frac{2\pi d}{7},\; \cos\frac{2\pi d}{7}\right)
\qquad
\left(\sin\frac{2\pi (m-1)}{12},\; \cos\frac{2\pi (m-1)}{12}\right)
$$

más un indicador binario de fin de semana, $\mathbb{1}[d \ge 5]$.

Hacen falta **dos** componentes por ciclo: con solo el seno, las 6:00 y las 18:00 darían
el mismo valor.

```python
# src/data/consumption.py::calendar_features
```

## Normalización por serie

Para que un solo modelo sirva a cientos de edificios de escalas muy distintas:

$$
z_i = \frac{y_i - \mu}{\sigma}
$$

Con dos precauciones que el código respeta:

1. **$\mu$ y $\sigma$ se calculan solo con el tramo de entrenamiento.** Usar todo el
   histórico filtra información del futuro hacia el pasado.
2. **Variante robusta disponible**, con mediana e intervalo intercuartílico:

$$
z_i = \frac{y_i - \mathrm{mediana}(y)}{\mathrm{IQR}(y) / 1{,}349}
$$

El $1{,}349$ es el rango intercuartílico de una normal estándar, así que la escala
resultante es comparable a una desviación típica. Un contador con lecturas disparatadas
no desplaza la escala de toda la serie.

```python
# src/consumo.py::TareaPrevision   parámetro norma="zscore" | "robusta"
```

## Normalización reversible por ventana

La que permite que **un solo modelo sirva para 400 edificios de escalas distintas**: se
normaliza con los estadísticos del propio contexto y se deshace en la salida, de modo
que la red aprende la *forma* del perfil y no el nivel absoluto.

$$
z = \frac{x - \mu_{\text{ventana}}}{\sigma_{\text{ventana}}}
\qquad\longrightarrow\qquad
\hat{y} = f(z)\cdot \sigma_{\text{ventana}} + \mu_{\text{ventana}}
$$

Es la idea central de los previsores modernos tipo PatchTST o N-HiTS, en su versión
mínima.

```python
# src/consumo.py::construir_previsor
```

## Dónde está en el código

| Concepto | Fichero |
|---|---|
| Codificación cíclica del calendario | `src/data/consumption.py::calendar_features` |
| Normalización z-score y robusta | `src/consumo.py::TareaPrevision` |
| Base ingenua estacional (168 h) | `src/consumo.py` — constante `ESTACIONAL` |
| Meteorología alineada por emplazamiento | `src/data/consumption.py::prep_bdg2` |
| Normalización reversible por ventana | `src/consumo.py::construir_previsor` |
