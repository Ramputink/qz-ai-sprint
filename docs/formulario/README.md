# Formulario — conceptos matemáticos y eléctricos

Referencia de todas las fórmulas que sostienen el proyecto, con sus **unidades**, su
**significado físico o de negocio** y **dónde está implementada cada una en el código**.

Esa última columna es la razón de ser de estos documentos: una fórmula sin su punto de
implementación se reimplementa mal, y una implementación sin su fórmula no se puede
auditar. Aquí van juntas.

## Índice

| Documento | Contenido |
|---|---|
| [01 — Potencia y energía](01-potencia-y-energia.md) | Activa, reactiva y aparente; factor de potencia; trifásica |
| [02 — Facturación y ahorro](02-facturacion-y-ahorro.md) | Términos de factura, compensación de reactiva, IPMVP y ASHRAE |
| [03 — Análisis de carga](03-analisis-de-carga.md) | Perfil, carga base, factor de carga, grados-día, EUI |
| [04 — Vibración y rodamientos](04-vibracion-y-rodamientos.md) | Estadísticos temporales, frecuencias de defecto, envolvente |
| [05 — Métricas de modelo](05-metricas-de-modelo.md) | Error, skill, AUC, matriz de confusión, RUL |

## Notación

Las fórmulas van en **LaTeX**, que GitHub renderiza de forma nativa tanto en línea
($E = P \cdot t$) como en bloque:

$$
S = \sqrt{P^2 + Q^2}
$$

Convenio general:

| Símbolo | Significado |
|---|---|
| $x_i$ | muestra $i$ de una señal |
| $N$ | número de muestras |
| $\bar{x}$ | media aritmética |
| $\hat{y}$ | valor predicho por un modelo |
| $\sigma$ | desviación típica |
| $[\text{kW}]$, $[\text{kWh}]$ | unidades entre corchetes |

Si abres estos documentos en un editor que no renderice LaTeX, verás el código fuente;
es legible, pero el destino natural de estos ficheros es GitHub o cualquier visor de
Markdown con soporte matemático.

## Criterio sobre las constantes

Cuando una fórmula necesita una constante del equipo —geometría de un rodamiento,
temperatura base de los grados-día, coste de un falso negativo— el documento indica si
ese valor está **medido**, **documentado por la fuente** o **supuesto**. Los supuestos se
heredan a todos los resultados que dependen de ellos, y confundirlos con medidas es la
forma más rápida de prometer algo que no se puede sostener.
