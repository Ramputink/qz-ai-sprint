# 04 — Vibración y rodamientos

Formulario del bloque de mantenimiento predictivo. Todas estas features están
implementadas y medidas; su rendimiento sobre IMS está en
[03 — Mantenimiento predictivo](../03-resultados-predictivo.md).

## Estadísticos en el dominio del tiempo

Sobre una señal de vibración $x_i$ de $N$ muestras, previamente centrada
($x \leftarrow x - \bar{x}$):

$$
\mathrm{RMS} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^{2}}
\qquad
\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\left(x_i - \bar{x}\right)^{2}}
$$

$$
x_{\text{pico}} = \max_i |x_i|
\qquad
x_{\text{p2p}} = \max_i x_i - \min_i x_i
$$

$$
\text{curtosis} = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{x_i-\bar{x}}{\sigma}\right)^{4}
\quad (\text{normal} = 3)
\qquad
\text{asimetría} = \frac{1}{N}\sum_{i=1}^{N}\left(\frac{x_i-\bar{x}}{\sigma}\right)^{3}
$$

$$
C_{\text{cresta}} = \frac{x_{\text{pico}}}{\mathrm{RMS}}
\qquad
C_{\text{forma}} = \frac{\mathrm{RMS}}{\overline{|x|}}
\qquad
C_{\text{impulso}} = \frac{x_{\text{pico}}}{\overline{|x|}}
\qquad
C_{\text{holgura}} = \frac{x_{\text{pico}}}{\left(\overline{\sqrt{|x|}}\right)^{2}}
$$

$$
H = -\sum_{k} p_k \log p_k
\qquad\text{con } p_k = \frac{|X_k|^2}{\sum_j |X_j|^2}
$$

**Por qué hay tantos y no basta el RMS.** El RMS crece cuando la vibración global sube,
pero un defecto incipiente no sube el nivel global: introduce **impactos cortos y
repetitivos**. La curtosis, el factor de cresta y el de impulso son sensibles a esos
impactos aunque la energía total apenas cambie. Es la diferencia entre «vibra más» y
«vibra distinto».

El **factor de holgura** usa la media de la raíz al cuadrado, lo que lo hace el más
sensible de todos a impactos aislados — y también el más ruidoso.

```python
# src/data/preprocess.py::vibration_features
# 11 estadísticos + 8 bandas de energía FFT = 19 features por canal
```

## Frecuencias características de defecto

Aquí está el salto de calidad. Un rodamiento con un defecto en una pista **golpea a una
frecuencia exacta**, deducible de su geometría.

Con $f_r$ la frecuencia de giro del eje $[\text{Hz}]$, $n$ el número de elementos
rodantes, $d$ su diámetro, $D$ el diámetro primitivo y $\varphi$ el ángulo de contacto,
definimos:

$$
r = \frac{d}{D}\cos\varphi
$$

$$
\mathrm{FTF} = \frac{f_r}{2}\left(1 - r\right)
\qquad
\mathrm{BPFO} = \frac{n\,f_r}{2}\left(1 - r\right) = n\cdot\mathrm{FTF}
$$

$$
\mathrm{BPFI} = \frac{n\,f_r}{2}\left(1 + r\right)
\qquad
\mathrm{BSF} = \frac{D}{2d}\,f_r\left(1 - r^{2}\right)
$$

| Sigla | Significado |
|---|---|
| $\mathrm{FTF}$ | *Fundamental Train Frequency* — la jaula que separa los elementos |
| $\mathrm{BPFO}$ | *Ball Pass Frequency, Outer race* — pista externa |
| $\mathrm{BPFI}$ | *Ball Pass Frequency, Inner race* — pista interna |
| $\mathrm{BSF}$ | *Ball Spin Frequency* — elemento rodante |

**La BPFI es siempre mayor que la BPFO** porque la pista interna gira con el eje y los
elementos la recorren más veces por vuelta. Si en un espectro aparece un pico a la
frecuencia mayor, el defecto está en la pista interna.

### Valores del banco IMS

Geometría **documentada por la fuente** (readme del propio dataset): rodamiento Rexnord
ZA-2115 de doble hilera, $n = 16$ elementos por hilera, $d = 0{,}331$, $D = 2{,}815$,
$\varphi = 15{,}17^\circ$, eje a 2000 RPM constante, muestreo a 20 kHz.

$$
f_r = \frac{2000}{60} = 33{,}333\ \text{Hz}
\qquad
r = \frac{0{,}331}{2{,}815}\cos(15{,}17^\circ) = 0{,}113489
$$

| Frecuencia | Cálculo | Valor |
|---|---|---|
| $\mathrm{FTF}$ | $16{,}667 \cdot (1 - 0{,}113489)$ | $14{,}78\ \text{Hz}$ |
| $\mathrm{BPFO}$ | $8 \cdot 33{,}333 \cdot (1 - 0{,}113489)$ | $236{,}40\ \text{Hz}$ |
| $\mathrm{BPFI}$ | $8 \cdot 33{,}333 \cdot (1 + 0{,}113489)$ | $296{,}93\ \text{Hz}$ |
| $\mathrm{BSF}$ | $4{,}2523 \cdot 33{,}333 \cdot (1 - 0{,}113489^{2})$ | $139{,}92\ \text{Hz}$ |

Coinciden con los valores publicados para este banco, lo que valida la implementación.

```python
# src/data/features.py::bearing_fault_freqs
# src/data/preprocess.py — IMS_BEARING, IMS_RPM, IMS_FS
```

## Análisis de envolvente

**El problema:** las frecuencias de defecto están entre 14 y 300 Hz, pero en el espectro
crudo esa zona la domina el desequilibrio del eje y sus armónicos. El impacto del defecto
es energéticamente pequeño y queda enterrado.

**La solución:** el impacto excita las **resonancias de alta frecuencia de la carcasa**
(varios kHz). Esa zona está limpia de ruido mecánico de baja frecuencia. Se filtra ahí,
se extrae la envolvente y en *su* espectro aparece el golpeteo a $\mathrm{BPFO}$ limpio.

$$
x_b = \mathrm{bandpass}\big(x,\ 2000\text{–}9500\ \text{Hz}\big)
\qquad
x_a = \mathcal{H}\{x_b\} \quad\text{(señal analítica)}
$$

$$
e(t) = \left|x_a(t)\right|
\qquad
E_k = \left|\mathcal{F}\big\{(e - \bar{e}) \cdot w_{\text{hann}}\big\}_k\right|
$$

$$
\mathrm{SNR}(f) = \frac{\max\limits_{|f_k - f| \le 3\Delta f} E_k}{\mathrm{mediana}(E)}
$$

El último paso normaliza por el **suelo de ruido** (la mediana del espectro), no por el
máximo. Así la cifra es comparable entre instantáneas y entre rodamientos, que es lo que
permite usarla como feature de un modelo.

La ventana de $\pm 3\Delta f$ existe porque el régimen no es perfectamente constante y el
pico se desplaza ligeramente entre instantáneas.

Se extraen **3 armónicos por cada una de las 4 frecuencias**, más el nivel global de la
envolvente: 13 features.

```python
# src/data/preprocess.py::envelope_defect_features
```

### Lo que se midió

Estas features **no mejoraron la mediana** sobre IMS (0,847 → 0,833 de accuracy), pero
el desglose por ensayo es revelador:

| Ensayo | Modo de fallo | Efecto |
|---|---|---|
| 1º | Pista interna + elemento rodante | El rodamiento roto pasa de avisar con **1,5 días a 10,21** |
| 3º | Pista externa, 741 h de degradación lenta | **Empeora** |

**Captan el impacto localizado, no la degradación gradual.** Ayudan en un modo de fallo y
estorban en otro. Es un resultado honesto y limita cuándo merece la pena implementarlas.

## Vida útil restante (RUL)

$$
\mathrm{RUL}(t) = t_{\text{fallo}} - t
$$

Con dos decisiones de diseño que resultaron importantes:

**Techo de RUL.** Al principio de la vida «queda mucho» y el valor exacto no es
aprendible: a un rodamiento sano no se le ve en la vibración cuánto le queda, eso depende
de su variabilidad de fabricación, no de su estado observable.

$$
y(t) = \min\big(t_{\text{fallo}} - t,\; \tau\big)
$$

Si el techo $\tau$ no recorta, se está pidiendo una regresión irresoluble sobre toda la
fase sana. En C-MAPSS el techo colapsa el 79,5 % de las ventanas; en IMS, con la
configuración inicial, el 0,0 %. **Eran tareas distintas sin que nadie lo hubiera
decidido.**

**Conversión a días.**

$$
\text{días de anticipación} = \frac{\text{pasos} \cdot h_{\text{paso}}}{24}
$$

| Dataset | $h_{\text{paso}}$ | Origen |
|---|---|---|
| IMS | 1,0 (tras promediar 6 instantáneas de 10 min) | **Medido** |
| MetroPT-3 | 1,0 (remuestreado desde 1 Hz) | **Medido** |
| C-MAPSS / N-CMAPSS | 24,0 | **SUPUESTO**: 1 ciclo de vuelo = 1 día |

C-MAPSS no publica la duración física del ciclo. **Todos los días de anticipación sobre
C-MAPSS heredan ese supuesto** y así consta en el meta de cada producto.

## Pérdida asimétrica

Un error de RUL no cuesta lo mismo en las dos direcciones: predecir *más* vida de la real
es un falso negativo peligroso.

$$
\mathcal{L} = \frac{1}{N}\sum_{i=1}^{N} w_i \left(\hat{y}_i - y_i\right)^{2}
\qquad
w_i = \begin{cases}
\lambda_{\mathrm{FN}} & \text{si } \hat{y}_i > y_i \quad (\text{optimista}) \\
1 & \text{si } \hat{y}_i \le y_i \quad (\text{pesimista})
\end{cases}
$$

```python
# src/models/rul.py::rul_loss
```

## Dónde está en el código

| Concepto | Fichero |
|---|---|
| Estadísticos temporales + bandas FFT | `src/data/preprocess.py::vibration_features` |
| Frecuencias de defecto | `src/data/features.py::bearing_fault_freqs` |
| Análisis de envolvente | `src/data/preprocess.py::envelope_defect_features` |
| Techo de RUL y horizonte | `src/data/preprocess.py` — `_RUL_CAP_FACTOR`, `lead_horizon_units` |
| Pérdida asimétrica | `src/models/rul.py::rul_loss` |
| Conversión a días | `src/data/preprocess.py::HOURS_PER_UNIT` |
