# 05 — Fallos silenciosos

Veinte errores encontrados durante el sprint. **Ninguno lanzaba excepción.** Todos
producían resultados de aspecto normal, y varios habrían llegado hasta una presentación.

Se documentan porque el patrón se repite: el fallo peligroso no es el que rompe, es el
que devuelve un número plausible.

## Los tres graves: la cadena de generaciones

Actuaban juntos y convertían «la Gen 2 mejora en caliente sobre la Gen 1» en dos
entrenamientos independientes comparados entre sí.

**1. Cada etapa reentrenaba desde cero.** El orquestador construye un `StageTrainer` por
etapa y solo la Gen 2 intentaba warm-start. La etapa 1b tiraba el modelo que la 1a
acababa de aprender y la «recalibración» recalibraba una red recién inicializada.
*Síntoma:* coste 2484 en la 1b frente a 889 en la 1a. Corregido: warm-start en las cuatro
etapas desde la mejor anterior → 901.

**2. El warm-start leía metadatos en vez de tensores.** El checkpoint envuelve el estado
del entrenador dentro de su propia clave `model`, así que los tensores acaban en
`blob["model"]["model"]`. El código leía `blob["model"]` y obtenía un diccionario de
metadatos. Habría cargado **0 tensores sin fallar**, reportando un arranque en frío como
si fuera caliente. Corregido con búsqueda del `state_dict` por estructura, no por ruta.

**3. La Gen 2 usaba otra anchura.** 96 canales frente a los 64 de la Gen 1: ningún tensor
encaja en forma, así que el warm-start habría sido cosmético aunque el punto 2 estuviera
resuelto. Corregido dando **profundidad manteniendo el ancho** (un quinto bloque de 64,
dilatación 16), que sí transfiere: 28/28 tensores en 1b, 28/34 en 2a.

## Datos que no eran los datos

**4. El zip de MFPT era una página HTML.** La URL de mfpt.org devolvía 200 con el HTML
del sitio; el descargador lo daba por bueno y el fallo aparecía horas después.
*Corregido:* se detecta el contenido y se falla en la descarga. Fuente cambiada al
repaquetado de MathWorks.

**5. El zip de BDG2 traía punteros de Git LFS.** El archivo de GitHub de un repositorio
con LFS contiene, en lugar de cada CSV grande, un texto de 130 bytes. El zip es válido,
se extrae sin error y deja un CSV de 3 líneas. *Corregido:* se detectan los punteros.
Fuente cambiada a Zenodo, que sirve los datos reales (595 MB frente a 140 MB de punteros).

**6. El id de Dataverse de AMPds2 apuntaba a otro dataset.** Se descargó un fichero de
discursos presidenciales. Lo detectó la validación de «no produjo ningún fichero», pero
el error de origen fue **adivinar un identificador numérico opaco** en vez de consultar
la API del repositorio.

**7. El HDF5 de AMPds2 va comprimido con blosc** y h5py no lo abre sin `hdf5plugin`, ni
`pd.read_hdf` sin `tables`. Además pandas moderno revienta al leer los metadatos que
escribió NILMTK. Se lee con h5py directamente y se localiza la columna por los nombres
guardados en los atributos.

## Preprocesado

**8. El extractor borraba el archivo de origen.** Al buscar contenedores anidados
encontraba también el `.zip` que acababa de descargar, lo re-extraía en una carpeta
duplicada y lo borraba. C-MAPSS quedó marcado como fallido y N-CMAPSS habría duplicado
15 GB. *Corregido:* nunca se toca el archivo de origen.

**9. `np.savez_compressed` añade `.npz` al nombre** si no lo lleva, así que el guardado
atómico buscaba un fichero que numpy había escrito con otro nombre. El temporal ahora
acaba en `.npz`.

**10. El 3er ensayo de IMS no se detectaba.** Cuelga de `3rd_test/4th_test/txt/` y la
búsqueda por nombre de carpeta no llegaba: faltaban **6.324 de 12.464 instantáneas**.
*Corregido:* se buscan por contenido (carpetas con suficientes ficheros cuyo nombre es
una marca de tiempo).

**11. La RUL de IMS estaba desplazada 13 días.** La carpeta del 3er ensayo trae 6.324
ficheros pero el readme del dataset fija el fallo en el **4.448** (2004.04.04 19:01:57);
los 1.876 restantes llegan hasta el 18 de abril, ya después del fallo. Como la RUL se
mide desde el final de la trayectoria, **un tercio de los rodamientos aprendía que al
fallo le quedaban casi dos semanas más de las que le quedaban**.

## Modelo y export

**12. `torch.compile` fallaba fuera del `try`.** Inductor compila sin protestar y revienta
en la **primera pasada**, no al compilar, así que el `try/except` alrededor de
`torch.compile()` no lo cubría. *Corregido:* se calienta con una pasada de prueba y se
vuelve a eager. (En este stack no funciona: `LoweringException` con conv1d.)

**13. `weight_norm` impedía el `deepcopy`** que necesitan el exportador ONNX y la
cuantización. Se deshace la reparametrización antes de exportar; el modelo resultante es
numéricamente idéntico (desviación 6e-8).

**14. El ONNX dejaba los pesos en un fichero aparte.** El exportador nuevo de PyTorch
saca los tensores a un `<modelo>.onnx.data`. En el edge eso es una trampa: se copia el
`.onnx` solo y el modelo carga vacío. *Corregido:* se consolida en un único fichero
autocontenido (0,378 MB).

## Medida

**15. No se calculaba ningún AUC** en el camino RUL. Todas las cifras del bloque salían
solo de accuracy medida tras elegir el umbral. Ver [04](04-como-se-mide.md).

**16. La comparativa Gen1/Gen2 estaba desfasada una etapa.** Se ejecuta dentro del
`finalize` de la 2b, antes de que el checkpoint `best` de esa etapa esté en disco, así
que comparaba la Gen 1 completa contra la Gen 2 a falta de su última etapa.

**17. CV(RMSE) agrupado en vez de por emplazamiento**, invirtiendo el veredicto de
acreditación ASHRAE.

## Infraestructura

**18. `.gitignore` ocultaba `src/data/` entero.** El patrón `data/` sin barra inicial
ignora *cualquier* carpeta llamada `data`, incluido el paquete de código. El paquete
recién escrito no aparecía en `git status`. *Corregido:* anclado con `/` en las cinco
rutas.

**19. La ablación agotaba la VRAM.** Cacheaba una `TareaPrevision` por variante y cada
una reserva su copia del dataset en GPU; a la segunda variante, OOM. *Corregido:* se
libera entre variantes y se tolera el OOM sin tumbar el barrido.

**20. El preprocesado de OEDI agotaba la RAM.** Leía las ~50 columnas de cada parquet en
float64 y `sort_index()` copiaba el bloque entero: ~27 MB por edificio, imposible llegar
a 4.200. *Corregido:* se leen solo las columnas necesarias y en float32, menos de 1 MB
por edificio.

## Qué se aprende de la lista

Tres patrones se repiten:

1. **Una fuente puede devolver 200 y no ser los datos.** HTML, punteros LFS, el dataset
   equivocado. Validar el contenido, no el código de respuesta.
2. **Lo que se rompe en la primera pasada no lo protege un `try` alrededor de la
   construcción.** Vale para `torch.compile` y para cualquier cosa perezosa.
3. **Un número que baja no siempre es un modelo peor.** Varias veces el síntoma
   (coste 2484, avisos de 1,5 días en los rodamientos rotos, 1.609 falsos positivos)
   apuntaba a la partición o a la métrica, no al modelo.
