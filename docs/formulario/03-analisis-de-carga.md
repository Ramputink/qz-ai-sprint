# 03 — Análisis de carga

Los indicadores que describen *cómo* consume una instalación, no solo *cuánto*. Son los
que convierten una curva en un diagnóstico.

## Carga base y carga variable

```
P_base     = percentil_5( P(t) )        [kW]   lo que consume incluso parado
P_variable = P(t) - P_base              [kW]   lo que depende de la actividad
```

La carga base se estima con un percentil bajo, no con el mínimo, porque el mínimo suele
ser un hueco de medida. **Una carga base alta en una planta que para los fines de semana
es el hallazgo más rentable de una auditoría**: son equipos encendidos sin producir.

## Factor de carga

```
FC = E_periodo / ( P_max * h_periodo )         [adimensional, 0..1]
```

Relaciona lo consumido con lo que se habría consumido al máximo todo el rato.

| FC | Lectura |
|---|---|
| Cercano a 1 | Consumo plano; poco margen de desplazamiento de carga |
| Bajo (0,3-0,5) | Muy picudo: **el término de potencia domina la factura** |

Un factor de carga bajo señala que el ahorro está en **aplanar los picos**, no en
reducir la energía total. Son dos intervenciones distintas y se confunden a menudo.

## Curva de duración de carga

Las potencias del periodo ordenadas de mayor a menor:

```
CDC = sort( P(t), descendente )
```

Responde a «¿cuántas horas al año supero X kW?». Es la herramienta para dimensionar la
potencia contratada: si solo se superan 400 kW durante 20 horas al año, puede salir más
barato pagar los excesos que contratar esa potencia.

## Grados-día

Cuantifican la demanda térmica y son la variable explicativa principal en climatización.

```
HDD = sum( max( T_base - T_media_dia , 0 ) )    [°C·día]   calefacción
CDD = sum( max( T_media_dia - T_base , 0 ) )    [°C·día]   refrigeración
```

`T_base` es la temperatura a partir de la cual el edificio necesita climatizar. **Es un
supuesto, no una medida**: se usa 15-18 °C por convenio, pero el valor correcto depende
del aislamiento y de las cargas internas y se puede estimar ajustándolo para maximizar
la correlación con el consumo.

En este proyecto no se calculan grados-día explícitamente: se le pasa al modelo la
temperatura horaria cruda y que él aprenda la relación, que además no es lineal
(hay un tramo muerto entre calefacción y refrigeración).

```python
# src/data/consumption.py::prep_bdg2
wx_cols = ["airTemperature", "dewTemperature", "windSpeed",
           "cloudCoverage", "precipDepth1HR"]
```

La **temperatura de rocío** va incluida porque la carga de deshumidificación depende de
la humedad, no solo de la temperatura seca, y en climas húmedos es una fracción grande
del consumo de refrigeración.

## Intensidad de uso energético (EUI)

```
EUI = E_anual / Superficie                      [kWh / m2 / año]
```

Es lo que permite **comparar edificios de distinto tamaño**. Sin normalizar por
superficie, un edificio grande siempre parece peor que uno pequeño.

BDG2 trae la superficie en sus metadatos (`sqm`), así que el EUI es calculable y es el
camino natural para investigar el 26,8 % de edificios no acreditables — ver
[07 — Preguntas abiertas](../07-preguntas-abiertas.md).

## Estacionalidad y la base ingenua

Un edificio es fuertemente periódico en dos escalas: diaria y semanal. De ahí que la
base contra la que hay que ganar sea:

```
P_ingenua(t) = P(t - 168h)      "lo mismo que a esta hora la semana pasada"
```

168 horas = 7 días. **Se elige la semana y no el día porque el ciclo semanal incluye el
fin de semana**: predecir un lunes con los datos del domingo falla sistemáticamente.

Esta base es dura de batir. El modelo del proyecto le gana un 48 % en MAE, y ese es el
número que mide si el modelo aporta algo — no el MAE absoluto.

## Codificación cíclica del calendario

La hora 23 y la hora 0 son contiguas, pero un modelo alimentado con los números 23 y 0
las ve máximamente distantes. Se codifican en el círculo:

```
hora_sin = sin( 2*pi * h / 24 )
hora_cos = cos( 2*pi * h / 24 )
dow_sin  = sin( 2*pi * d / 7 )
dow_cos  = cos( 2*pi * d / 7 )
mes_sin  = sin( 2*pi * (m-1) / 12 )
mes_cos  = cos( 2*pi * (m-1) / 12 )
finde    = 1 si d >= 5, si no 0
```

Hacen falta **dos** componentes por ciclo: con solo el seno, las 6:00 y las 18:00 darían
el mismo valor.

```python
# src/data/consumption.py::calendar_features
```

## Normalización por serie

Para que un solo modelo sirva a cientos de edificios de escalas muy distintas:

```
z[i] = ( y[i] - mu ) / sigma
```

Con dos precauciones que el código respeta:

1. **`mu` y `sigma` se calculan solo con el tramo de entrenamiento.** Usar todo el
   histórico filtra información del futuro hacia el pasado.
2. **Variante robusta disponible**, con mediana e IQR en vez de media y desviación:

```
z[i] = ( y[i] - mediana ) / ( IQR / 1,349 )
```

El `1,349` es el rango intercuartílico de una normal estándar, así que la escala
resultante es comparable a una desviación típica. Un contador con lecturas disparatadas
no desplaza la escala de toda la serie.

```python
# src/consumo.py::TareaPrevision   parámetro norma="zscore" | "robusta"
```

## Dónde está en el código

| Concepto | Fichero |
|---|---|
| Codificación cíclica del calendario | `src/data/consumption.py::calendar_features` |
| Normalización z-score y robusta | `src/consumo.py::TareaPrevision` |
| Base ingenua estacional (168 h) | `src/consumo.py` — constante `ESTACIONAL` |
| Meteorología alineada por emplazamiento | `src/data/consumption.py::prep_bdg2` |
| Normalización reversible por ventana | `src/consumo.py::construir_previsor` |
