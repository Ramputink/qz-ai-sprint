# 04 — Vibración y rodamientos

Formulario del bloque de mantenimiento predictivo. Todas estas features están
implementadas y medidas; su rendimiento sobre IMS está en
[03 — Mantenimiento predictivo](../03-resultados-predictivo.md).

## Estadísticos en el dominio del tiempo

Sobre una señal de vibración `x[i]` de `N` muestras, centrada (`x = x - mean(x)`):

```
RMS      = sqrt( mean( x^2 ) )                       energía global de la vibración
std      = sqrt( mean( (x - mean(x))^2 ) )
pico     = max( |x| )
p2p      = max(x) - min(x)                           pico a pico

curtosis = mean( ((x - mu) / sigma)^4 )              "puntiagudez"  (normal = 3)
asimetria= mean( ((x - mu) / sigma)^3 )              (simétrica = 0)

cresta   = pico / RMS                                 cuánto sobresale el pico
forma    = RMS / mean(|x|)
impulso  = pico / mean(|x|)
holgura  = pico / ( mean( sqrt(|x|) ) )^2
entropia = - sum( p * log(p) )    con p = espectro normalizado
```

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
frecuencia exacta**, deducible de su geometría:

```
Datos:
  f_r  frecuencia de giro del eje        [Hz]  = RPM / 60
  n    número de elementos rodantes
  d    diámetro del elemento rodante
  D    diámetro primitivo (pitch)
  phi  ángulo de contacto                [rad]

  ratio = (d / D) * cos(phi)

FTF  = (f_r / 2) * ( 1 - ratio )                    jaula
BPFO = (n / 2) * f_r * ( 1 - ratio )                pista externa   = n * FTF
BPFI = (n / 2) * f_r * ( 1 + ratio )                pista interna
BSF  = (D / (2*d)) * f_r * ( 1 - ratio^2 )          elemento rodante
```

| Sigla | Significado |
|---|---|
| FTF | *Fundamental Train Frequency* — la jaula que separa los elementos |
| BPFO | *Ball Pass Frequency, Outer race* |
| BPFI | *Ball Pass Frequency, Inner race* |
| BSF | *Ball Spin Frequency* |

**La BPFI es siempre mayor que la BPFO** porque la pista interna gira con el eje y los
elementos la recorren más veces por vuelta. Si en un espectro aparece un pico a la
frecuencia mayor, el defecto está en la pista interna.

### Valores del banco IMS

Geometría **documentada por la fuente** (readme del propio dataset): rodamiento Rexnord
ZA-2115 de doble hilera, 16 elementos por hilera, eje a 2000 RPM constante, muestreo a
20 kHz.

```
f_r   = 2000 / 60 = 33,333 Hz
ratio = (0,331 / 2,815) * cos(15,17 deg) = 0,113489

FTF  = 16,667 * (1 - 0,113489)              =  14,78 Hz
BPFO = 8 * 33,333 * (1 - 0,113489)          = 236,40 Hz
BPFI = 8 * 33,333 * (1 + 0,113489)          = 296,93 Hz
BSF  = 4,2523 * 33,333 * (1 - 0,113489^2)   = 139,92 Hz
```

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
se extrae la envolvente y en *su* espectro aparece el golpeteo a BPFO limpio.

```
1. Filtrado paso banda        x_b = bandpass( x, 2000..9500 Hz )
2. Señal analítica            x_a = hilbert( x_b )          (compleja)
3. Envolvente                 env = |x_a|
4. Centrado y ventana         env = (env - mean(env)) * hanning(N)
5. Espectro de envolvente     E   = |rfft(env)|
6. Altura relativa del pico   SNR(f) = max(E en f +- 3 bins) / mediana(E)
```

El paso 6 normaliza por el **suelo de ruido** (la mediana del espectro), no por el
máximo. Así la cifra es comparable entre instantáneas y entre rodamientos, que es lo que
permite usarla como feature de un modelo.

La ventana de `± 3 bins` existe porque el régimen no es perfectamente constante y el pico
se desplaza ligeramente entre instantáneas.

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

```
RUL(t) = t_fallo - t                    [pasos del dataset]
```

Con dos decisiones de diseño que resultaron importantes:

**Techo de RUL.** Al principio de la vida «queda mucho» y el valor exacto no es
aprendible: a un rodamiento sano no se le ve en la vibración cuánto le queda, eso depende
de su variabilidad de fabricación, no de su estado observable.

```
RUL_etiqueta(t) = min( t_fallo - t , techo )
```

Si el techo no recorta, se está pidiendo una regresión irresoluble sobre toda la fase
sana. En C-MAPSS el techo colapsa el 79,5 % de las ventanas; en IMS, con la
configuración inicial, el 0,0 %. **Eran tareas distintas sin que nadie lo hubiera
decidido.**

**Conversión a días.** Cada dataset tiene su cadencia:

```
dias_de_anticipacion = pasos * horas_por_paso / 24
```

| Dataset | horas_por_paso | Origen |
|---|---|---|
| IMS | 1,0 (tras promediar 6 instantáneas de 10 min) | **Medido** |
| MetroPT-3 | 1,0 (remuestreado desde 1 Hz) | **Medido** |
| C-MAPSS / N-CMAPSS | 24,0 | **SUPUESTO**: 1 ciclo de vuelo = 1 día |

C-MAPSS no publica la duración física del ciclo. **Todos los días de anticipación sobre
C-MAPSS heredan ese supuesto** y así consta en el meta de cada producto.

## Pérdida asimétrica

Un error de RUL no cuesta lo mismo en las dos direcciones: predecir *más* vida de la real
es un falso negativo peligroso.

```
err = pred - real
w   = fn_weight  si err > 0  (optimista)
      1          si err <= 0 (pesimista)

L = mean( w * err^2 )
```

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
