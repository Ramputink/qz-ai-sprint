# Documentación del sprint — QuantumZIGMA

Bitácora técnica del trabajo realizado entre el **27 y el 29 de agosto de 2026** sobre
`qz-ai-sprint`. Documenta lo que se construyó, lo que se midió y —sobre todo— **lo que
no funcionó**, porque esa parte es la que no sobrevive en el código y la que evita
repetir el mismo camino dentro de tres meses.

## Índice

| Documento | Contenido |
|---|---|
| [01 — Qué se construyó](01-que-se-construyo.md) | Estado inicial del paquete y piezas añadidas |
| [02 — Consumo eléctrico](02-resultados-consumo.md) | La línea prioritaria: previsión, NILM, desperdicio y línea base de ahorro |
| [03 — Mantenimiento predictivo](03-resultados-predictivo.md) | RUL, C-MAPSS e IMS, y por qué IMS no da señal |
| [04 — Cómo se mide](04-como-se-mide.md) | Criterios de validación y métricas. **El documento más transferible** |
| [05 — Fallos silenciosos](05-fallos-silenciosos.md) | Los 20 errores que no lanzaban excepción |
| [06 — Datasets](06-datasets.md) | Catálogo, estado verificado y trampas de cada fuente |
| [07 — Preguntas abiertas](07-preguntas-abiertas.md) | Qué queda por decidir y por medir |
| **[Formulario](formulario/README.md)** | **Todas las fórmulas eléctricas y matemáticas, con su implementación** |

## Resumen en una página

**El paquete no entrenaba.** Al empezar, `python run.py` no podía ejecutarse: faltaba
el paquete `src/data/` completo (referenciado pero ausente en disco) y `_real_step` era
un `raise NotImplementedError`. Solo funcionaba `--dry-run`, que simula métricas con una
fórmula. Hoy entrena de verdad en los dos bloques.

**La prioridad cambió a mitad de camino.** El repositorio está escrito alrededor del
mantenimiento predictivo, pero la pregunta que importa es **si se puede optimizar el
consumo eléctrico**. El bloque de consumo se construyó después y es donde están los
resultados buenos.

**Dónde hay señal y dónde no:**

| | Métrica | Veredicto |
|---|---|---|
| Consumo eléctrico (BDG2) | Skill **+0,48** sobre la base ingenua | Señal clara y aprovechable |
| Predicción de fallo (C-MAPSS, simulado) | ROC-AUC **0,996** | Excelente, pero sobre datos simulados |
| Predicción de fallo (IMS, vibración real) | ROC-AUC **0,54** | **Moneda al aire: no hay señal** |

**El resultado más caro fue negativo.** IMS es el dato más parecido a un motor de planta
y el modelo no aprende nada de él: 4 eventos de fallo y 2 bancos de ensayo utilizables no
dan para más. Se probaron cuatro vías para arreglarlo (features de envolvente, techo de
RUL, horizonte relativo a la vida útil, validación agrupada) y ninguna funcionó. Eso
está medido, no supuesto.

**Cinco hipótesis propias se cayeron con datos.** Están todas registradas en el
documento 04, con sus números. Que una explicación suene razonable no la hace cierta, y
en esta máquina un entrenamiento cuesta entre 47 y 228 segundos: **sale más barato medir
que discutir**.

**El ahorro se puede demostrar.** La línea base contrafactual, sobre datos sin ninguna
intervención, detecta un ahorro aparente del **0,096 %**. No inventa ahorros, y fija el
suelo de credibilidad: cualquier ahorro por debajo de ~0,2 % es ruido del método. Medida
como exige ASHRAE Guideline 14 —por emplazamiento, no en agregado— **el 73 % de los
edificios son acreditables**.
