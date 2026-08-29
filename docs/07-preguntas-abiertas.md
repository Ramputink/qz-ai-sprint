# 07 — Preguntas abiertas

Lo que queda por medir y lo que queda por decidir. Separadas porque no se resuelven igual:
las primeras las contesta la máquina, las segundas las contesta el negocio.

## Contestado (29-08-2026)

**¿El modelo está limitado por datos?** No. El barrido de escala sobre OEDI (100 → 4.000
edificios, 40×) deja el skill entre 0,364 y 0,387 sin tendencia y el porcentaje de
acreditables plano. Junto con el barrido de capacidad —de 0,88 a 25,6 M de parámetros,
también plano— queda establecido que **el techo no está ni en los parámetros ni en el
número de edificios: está en el problema**.

Consecuencia práctica: **no merece la pena descargar más datos del mismo tipo.** La vía
es información *distinta* (condicionar por edificio) o aceptar que hay una fracción de
edificios poco predecibles y excluirla del alcance.

**¿Qué preprocesado aporta?** Ninguno de forma general. `log1p` ayuda en OEDI
(+0,3799 → +0,3958) y estorba en BDG2 (+0,4798 → +0,4559); ampliar el contexto hace lo
contrario en cada uno. Todas las diferencias son de ±0,02 de skill, dentro de la banda de
ruido. **El preprocesado no es la palanca.** Detalle en
[02 — Consumo](02-resultados-consumo.md).

## Por medir

**Qué caracteriza al 26,8 % de edificios no acreditables.** El percentil 90 del CV(RMSE)
está en 44,9 %, así que hay una cola clara. Si resulta ser un subconjunto identificable
—por tamaño, tipo de uso o calidad del contador— se puede **excluir del alcance
contractual desde el principio** en vez de descubrirlo al facturar. Es una consulta sobre
los metadatos de BDG2 más que un problema de modelado, y es barata.

**Si condicionar por edificio mejora.** Darle al modelo los metadatos (uso, superficie,
año) o un embedding aprendido por edificio. Es capacidad puesta donde está el problema
—la heterogeneidad entre edificios— en vez de capacidad a bulto, que ya sabemos que no
sirve.

**Previsión probabilística.** El previsor da un punto. Un contrato necesita intervalos:
«el consumo estará entre X e Y con un 90 % de confianza». Es cambiar a pérdida cuantílica
y volver a medir; la infraestructura ya está.

**Si preentrenar transfiere.** Entrenar en Low Carbon London (2.113 hogares) y afinar
sobre BDG2. Es la versión honesta y en dominio de lo que LOTSA prometía, y ahora los dos
datasets están procesados.

**El bloque predictivo con AUC homogéneo.** Las cifras de IMS y C-MAPSS se midieron con
protocolos distintos (validación agrupada frente al split estándar de la etapa 1a). Los
`.zip` y la comparativa Gen1/Gen2 que hay en `artifacts/` son anteriores al AUC. Un
`python run.py` de 10 minutos los homogeneiza.

## Por decidir (no lo resuelve la máquina)

**Qué criterio manda: coste o falsos negativos.** La Gen 2 de C-MAPSS cambia 33 falsos
negativos menos por 251 falsos positivos más. Bajo la función de coste del `config.yaml`
(`fn_weight: 5`) es peor; bajo «no se me puede escapar un fallo» es mejor. **Ese 5 está
puesto a ojo y debería salir de un número del negocio**: cuánto cuesta una parada no
prevista frente a cuánto cuesta una inspección innecesaria.

**Si se sigue con el mantenimiento predictivo.** El bloque funciona sobre datos simulados
(C-MAPSS, ROC-AUC 0,996) y no funciona sobre vibración real (IMS, ROC-AUC 0,54). Retomarlo
exige descargar XJTU-SY, FEMTO y Paderborn para pasar de 4 a ~35 eventos de fallo. Es una
inversión razonable, pero compite con el bloque de consumo, que ya da resultados.

**Cómo se escribe la promesa de anticipación en un contrato.** «≥10 días» no es portable:
son el 5,7 % de la vida de un motor C-MAPSS y el 83 % de la de un ensayo IMS. Un motor de
planta que dure tres años y un rodamiento de banco acelerado que dure doce días no
admiten el mismo número absoluto. La forma defendible es como **fracción de vida útil** o
como un horizonte calibrado por clase de máquina — y conviene decidirlo antes de firmar,
no después.

**Qué se hace con la recalibración.** Empeora el coste en ambas generaciones (1a→1b de
889 a 901; 2a→2b de 841 a 987). Reinyectar los falsos negativos con peso 15×
(`fn_weight` 5 × `hard_negative_boost` 3) sobrecorrige. O se ajustan esos pesos, o se
acepta que el mecanismo que el README vende como diferenciador no está aportando aquí.

## Deuda técnica conocida

- **Optuna nunca se enganchó.** El `config.yaml` pide 100 trials por etapa y el trainer
  usa hiperparámetros fijos. Es una promesa del config sin cumplir. El trabajo paralelo
  de `stage_rtf.py` sí trae una búsqueda aleatoria.
- **El detector de deriva registró 0 eventos.** Se le pasa el MAE, que es suave. Debería
  recibir los residuos por ventana.
- **`torch.compile` no funciona en este stack** (`LoweringException` con conv1d en
  torch 2.11+cu128 / Windows). Está desactivado en el config con la comprobación hecha.
- **TensorRT no está instalado**, así que el export a edge llega hasta ONNX e INT8.
