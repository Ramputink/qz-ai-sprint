# 02 — Facturación y ahorro

El documento que conecta las magnitudes eléctricas con euros. Sin esto, «optimizar el
consumo» no tiene una función objetivo.

## De qué se compone una factura industrial

$$
\text{Coste} = \underbrace{C_{\text{pot}}}_{\text{potencia}} +
               \underbrace{C_{\text{ene}}}_{\text{energía}} +
               \underbrace{C_{\text{reac}}}_{\text{reactiva}} +
               \text{impuestos}
$$

**Término de potencia** — se paga por la potencia *contratada* y por los excesos sobre
ella, no por lo consumido:

$$
C_{\text{pot}} = \sum_{p} P_{c,p} \cdot \pi_{p} \cdot d
$$

donde $p$ recorre los periodos tarifarios, $P_{c,p}$ es la potencia contratada en cada
uno, $\pi_p$ su precio y $d$ los días del periodo. La clave es que **se paga aunque no se
consuma**, y que un único pico de 15 minutos puede fijar el coste de todo un año si
dispara el maxímetro.

**Término de energía** — lo consumido, discriminado por periodo horario:

$$
C_{\text{ene}} = \sum_{p} E_{p} \cdot \rho_{p}
$$

Esto es lo que hace que **el previsor tenga valor económico**: si sabes el consumo de
mañana hora a hora, puedes desplazar cargas a periodos baratos. El horizonte de 24 h del
modelo no es arbitrario, es el horizonte de decisión operativa.

## Penalización por reactiva y su corrección

La compañía penaliza cuando el factor de potencia baja de un umbral (típicamente 0,95).
La corrección es una batería de condensadores, y su dimensionado es esta fórmula:

$$
Q_C = P \left( \tan\varphi_1 - \tan\varphi_2 \right) \quad [\text{kVAr}]
$$

$$
\varphi_1 = \arccos(\cos\varphi_{\text{actual}})
\qquad
\varphi_2 = \arccos(\cos\varphi_{\text{objetivo}})
$$

**Ejemplo.** Una planta con $P = 500\ \text{kW}$ y $\cos\varphi = 0{,}80$ que quiere
llegar a $0{,}95$:

$$
\varphi_1 = \arccos(0{,}80) = 36{,}87^\circ \;\Rightarrow\; \tan\varphi_1 = 0{,}7500
$$

$$
\varphi_2 = \arccos(0{,}95) = 18{,}19^\circ \;\Rightarrow\; \tan\varphi_2 = 0{,}3287
$$

$$
Q_C = 500 \cdot (0{,}7500 - 0{,}3287) = 210{,}6\ \text{kVAr}
$$

**Por qué esto importa más de lo que parece:** es una vía de ahorro que **no toca la
producción**. No hay que apagar nada ni cambiar horarios — se instala un equipo y la
penalización desaparece. Es el ahorro más fácil de vender y el más fácil de verificar.
Y es la razón de que `steel_industry_energy`, con sus 482 KB, importe pese a ser el
dataset más pequeño del catálogo: es el único que trae reactiva y factor de potencia.

## Medir un ahorro: el problema del contrafactual

**No se puede demostrar un ahorro comparando meses.** Si el consumo baja un 8 % pero
ese mes hizo más frío, o se produjo menos, o hubo un puente, no se ha ahorrado nada.

$$
\text{Ahorro} = E_{\text{habría habido}} - E_{\text{medido}}
$$

El primer término **no es observable**: es un contrafactual. Hay que estimarlo con un
modelo entrenado en el periodo anterior a la intervención.

## IPMVP Opción C

El protocolo estándar (*International Performance Measurement and Verification
Protocol*) para medir ahorros a nivel de acometida:

1. **Periodo base** — se mide consumo y variables explicativas (clima, producción).
2. **Ajuste** — se entrena $\hat{E} = f(\text{clima},\ \text{calendario},\ \text{producción})$.
3. **Intervención** — se aplica la mejora.
4. **Periodo de reporte** — se predice lo que *habría* consumido y se compara.

$$
\text{Ahorro} = f\!\left(x_{\text{reporte}}\right) - E_{\text{medido, reporte}}
$$

El punto sutil: el modelo se alimenta con **las condiciones del periodo de reporte**
($x_{\text{reporte}}$), no las del base. Así se descuenta que hiciera más frío o que se
produjera más.

```python
# src/consumo.py::medir_ahorro
# El corte es TEMPORAL: se entrena con el pasado y se predice el futuro.
```

**La prueba de que el método no miente:** aplicado a datos **sin ninguna intervención**,
el ahorro aparente debe salir próximo a cero. En este proyecto sale **0,096 %**. Eso
fija el suelo de credibilidad: cualquier ahorro reportado por debajo de $\sim 0{,}2\ \%$
es sesgo del método, no una mejora.

## ASHRAE Guideline 14: cuándo se acepta una línea base

Una línea base solo vale si es lo bastante precisa. Dos criterios:

$$
\mathrm{CV(RMSE)} = \frac{100}{\bar{y}}
\sqrt{\frac{\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)^2}{n - p}} \quad [\%]
$$

$$
\mathrm{NMBE} = \frac{100}{\bar{y}} \cdot
\frac{\sum_{i=1}^{n} \left(y_i - \hat{y}_i\right)}{n - p} \quad [\%]
$$

donde $p$ es el número de parámetros del modelo de regresión.

| Criterio | Umbral en datos horarios | Qué mide |
|---|---|---|
| $\mathrm{CV(RMSE)}$ | $< 25\ \%$ | Dispersión: cuánto se equivoca |
| $\mathrm{NMBE}$ | $\pm 5\ \%$ (a veces $\pm 10\ \%$) | Sesgo: si se equivoca *siempre hacia el mismo lado* |

**El NMBE es el que protege contra el fraude involuntario.** Un modelo que sobreestima
sistemáticamente el consumo esperado inventa ahorros que no existen: la diferencia
$\hat{E} - E$ sale positiva sin que nadie haya hecho nada.

### Se aplica por emplazamiento

Esto costó una conclusión equivocada durante un día. ASHRAE G14 evalúa **cada
emplazamiento por separado**, no un promedio de cartera:

| | Agrupando 400 edificios | Por edificio |
|---|---|---|
| $\mathrm{CV(RMSE)}$ | 30,16 % → suspende | **mediana 16,05 %** → aprueba |
| Veredicto | No acreditable | **73,2 % de los edificios lo son** |

Con edificios de escalas muy distintas, el número agrupado lo dominan unos pocos grandes
o erráticos (percentil 90 en 44,9 %). Ver [05 — Métricas de modelo](05-metricas-de-modelo.md).

### Una simplificación de la implementación

El código calcula el RMSE y el NMBE dividiendo entre $n$, no entre $n - p$:

```python
# src/consumo.py::metricas
rmse = sqrt(mean(err ** 2))                  # usa n, no n - p
nmbe = 100 * mean(err) / abs(mean(real))     # idem
```

Con $n$ del orden de decenas de miles la diferencia es despreciable. **Pero si alguna vez
se evalúa un emplazamiento con pocas muestras, esta simplificación es optimista** y hay
que corregirla antes de presentar el número a un tercero.

## Dónde está en el código

| Concepto | Fichero |
|---|---|
| CV(RMSE), NMBE, umbral ASHRAE | `src/consumo.py::metricas` |
| CV(RMSE) por emplazamiento y % acreditables | `src/consumo.py::evaluar_previsor` |
| Protocolo IPMVP Opción C | `src/consumo.py::medir_ahorro` |
| Reactiva y factor de potencia como covariables | `src/data/consumption.py::prep_steel` |
