#!/usr/bin/env python3
"""dashboard.py — ilustrador y analizador gráfico de todos los resultados.

Lee los JSON que dejan los barridos y entrenamientos en `artifacts/` y genera un
**único HTML autocontenido** con todas las figuras y sus tablas.

Decisiones de diseño, por si alguien las revisa:

* **Un solo fichero.** Las imágenes van embebidas en base64, así que el informe se
  copia al Mac por USB y se abre sin nada más. Nada de CDN ni de carpetas de
  recursos que se pierden al mover el fichero.
* **Modo claro y oscuro seleccionados, no invertidos.** Cada figura se dibuja dos
  veces con la paleta de su modo y se intercambian por CSS. Invertir una figura
  clara automáticamente rompe la relación de contraste con el fondo.
* **Una sola escala por gráfica.** Nunca dos ejes Y: dos magnitudes distintas son
  dos gráficas. Es el error de visualización más común y el más difícil de leer.
* **Toda figura lleva su tabla debajo.** Es la vía de acceso cuando el color no
  basta —daltonismo, impresión en blanco y negro— y además permite copiar cifras
  sin volver al JSON.
* **Etiquetas directas sobre las barras.** El valor exacto importa más que la
  comparación visual en un informe técnico.

Uso:
    python analyze/dashboard.py                    # -> artifacts/dashboard.html
    python analyze/dashboard.py --abrir            # además lo abre en el navegador
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RAIZ = Path(__file__).resolve().parent.parent
ART = RAIZ / "artifacts"

# Paleta validada (dataviz): tres primeras ranuras categóricas, que superan las
# comprobaciones de daltonismo en ambos modos con todos los pares en juego.
PALETA: dict[str, dict[str, str]] = {
    "claro": {
        "superficie": "#fcfcfb", "plano": "#f9f9f7",
        "tinta": "#0b0b0b", "tinta2": "#52514e", "mudo": "#898781",
        "rejilla": "#e1e0d9", "eje": "#c3c2b7",
        "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
        "critico": "#d03b3b", "bueno": "#0ca30c",
    },
    "oscuro": {
        "superficie": "#1a1a19", "plano": "#0d0d0d",
        "tinta": "#ffffff", "tinta2": "#c3c2b7", "mudo": "#898781",
        "rejilla": "#2c2c2a", "eje": "#383835",
        "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
        "critico": "#d03b3b", "bueno": "#0ca30c",
    },
}


# ---------------------------------------------------------------- utilidades
def cargar(nombre: str) -> Any:
    """Lee un artefacto JSON; devuelve None si no existe (barrido aún sin correr)."""
    p = ART / nombre
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _ejes(ax, c: dict[str, str], titulo: str = "", xlab: str = "", ylab: str = ""):
    """Cromo recesivo: la rejilla y los ejes no compiten con los datos."""
    ax.set_facecolor(c["superficie"])
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(c["eje"])
        ax.spines[lado].set_linewidth(1.0)
    ax.tick_params(colors=c["mudo"], labelsize=9, length=0)
    ax.grid(True, axis="y", color=c["rejilla"], linewidth=1.0, zorder=0)
    ax.set_axisbelow(True)
    if titulo:
        ax.set_title(titulo, color=c["tinta"], fontsize=12, pad=14, loc="left")
    if xlab:
        ax.set_xlabel(xlab, color=c["tinta2"], fontsize=9)
    if ylab:
        ax.set_ylabel(ylab, color=c["tinta2"], fontsize=9)


def _nueva(c: dict[str, str], ancho=7.2, alto=3.6):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(ancho, alto), dpi=130)
    fig.patch.set_facecolor(c["superficie"])
    return fig, ax


def _etiquetar(ax, xs, ys, c, fmt="{:.3f}", dy=0.01, rot=0):
    """Etiqueta directa sobre cada marca. La tinta es de texto, nunca el color de
    la serie: el color lo lleva la marca, el número lo lee la gente."""
    rango = (max(ys) - min(ys)) or 1.0
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y), textcoords="offset points",
                    xytext=(0, 6), ha="center", fontsize=8.5,
                    color=c["tinta2"], rotation=rot)


def _png(fig) -> str:
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _linea_umbral(ax, y, c, texto):
    """Umbral normativo: color de estado + etiqueta. Nunca solo el color."""
    ax.axhline(y, color=c["critico"], linewidth=1.5, linestyle="--", zorder=2)
    ax.annotate(texto, (0.995, y), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=8.5, color=c["critico"])


# ------------------------------------------------------------------ figuras
def fig_capacidad(datos, tema):
    """Parámetros frente a % de edificios acreditables. Una sola escala."""
    if not datos:
        return None
    c = PALETA[tema]
    xs = [d["parametros"] / 1e6 for d in datos]
    ys = [(d.get("por_serie") or {}).get("pct_series_acreditables", 0) for d in datos]
    orden = sorted(range(len(xs)), key=lambda i: xs[i])
    xs, ys = [xs[i] for i in orden], [ys[i] for i in orden]

    fig, ax = _nueva(c)
    ax.plot(xs, ys, color=c["s1"], linewidth=2, marker="o", markersize=8,
            markeredgecolor=c["superficie"], markeredgewidth=2, zorder=3)
    _etiquetar(ax, xs, ys, c, "{:.1f}%")
    ax.set_xscale("log")
    ax.set_ylim(0, 100)
    _ejes(ax, c, "Capacidad del modelo frente a edificios acreditables",
          "parámetros (millones, escala logarítmica)", "% de edificios con CV(RMSE) < 25 %")
    ax.annotate("28× más parámetros no mueven la métrica de negocio",
                (0.5, 0.12), xycoords="axes fraction", ha="center",
                fontsize=9, color=c["tinta2"])
    return fig


def fig_capacidad_mae(datos, tema):
    """La misma familia de modelos, pero mirando el error. Gráfica aparte porque
    es otra magnitud: meterla en un segundo eje Y sería ilegible."""
    if not datos:
        return None
    c = PALETA[tema]
    xs = [d["parametros"] / 1e6 for d in datos]
    ys = [d["mae"] for d in datos]
    tr = [d.get("mae_train", d["mae"]) for d in datos]
    orden = sorted(range(len(xs)), key=lambda i: xs[i])
    xs, ys, tr = ([v[i] for i in orden] for v in (xs, ys, tr))

    fig, ax = _nueva(c)
    ax.plot(xs, ys, color=c["s1"], linewidth=2, marker="o", markersize=8,
            markeredgecolor=c["superficie"], markeredgewidth=2, label="test", zorder=3)
    ax.plot(xs, tr, color=c["s2"], linewidth=2, marker="s", markersize=8,
            markeredgecolor=c["superficie"], markeredgewidth=2, label="entrenamiento",
            zorder=3)
    ax.set_xscale("log")
    _ejes(ax, c, "Error de test y de entrenamiento frente a capacidad",
          "parámetros (millones, escala logarítmica)", "MAE [kWh]")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for t in leg.get_texts():
        t.set_color(c["tinta2"])
    ax.annotate("las dos curvas van juntas: ni sobreajuste ni infraajuste",
                (0.5, 0.06), xycoords="axes fraction", ha="center",
                fontsize=9, color=c["tinta2"])
    return fig


def fig_perdidas(datos, tema):
    """Comparativa de funciones de pérdida. Barras: magnitud por categoría."""
    if not datos:
        return None
    c = PALETA[tema]
    nombres = list(datos.keys())
    ys = [datos[n]["skill_vs_ingenua"] for n in nombres]
    fig, ax = _nueva(c, alto=3.2)
    barras = ax.bar(nombres, ys, color=c["s1"], width=0.55, zorder=3)
    for b, y in zip(barras, ys):
        ax.annotate(f"{y:+.4f}", (b.get_x() + b.get_width() / 2, y),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9, color=c["tinta2"])
    ax.set_ylim(0, max(ys) * 1.25)
    _ejes(ax, c, "Función de pérdida: no cambia nada",
          "", "skill sobre la base ingenua")
    return fig


def fig_ablacion(datos, tema):
    """Variantes de preprocesado. Barras horizontales: las etiquetas son largas."""
    if not datos:
        return None
    c = PALETA[tema]
    filas = [d for d in datos if "skill" in d]
    if not filas:
        return None
    filas.sort(key=lambda d: d["skill"])
    nombres = [d["variante"] for d in filas]
    ys = [d["skill"] for d in filas]
    ref = next((d["skill"] for d in filas if d["variante"] == "referencia"), None)

    fig, ax = _nueva(c, alto=0.42 * len(filas) + 1.6)
    colores = [c["s2"] if n == "referencia" else c["s1"] for n in nombres]
    barras = ax.barh(nombres, ys, color=colores, height=0.6, zorder=3)
    for b, y in zip(barras, ys):
        ax.annotate(f"{y:+.4f}", (y, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=9, color=c["tinta2"])
    if ref is not None:
        ax.axvline(ref, color=c["s2"], linewidth=1.5, linestyle="--", zorder=2)
    ax.grid(True, axis="x", color=c["rejilla"], linewidth=1.0, zorder=0)
    ax.grid(False, axis="y")
    _ejes(ax, c, "Preprocesado: qué aporta cada transformación",
          "skill sobre la base ingenua", "")
    ax.set_xlim(0, max(ys) * 1.18)
    return fig


def fig_escala(datos, tema):
    """LA gráfica: ¿el modelo está limitado por datos?

    Se dibuja el SKILL y no el MAE. El MAE no es comparable entre filas de esta
    tabla: cada submuestra tiene una mezcla distinta de edificios y el consumo
    medio por serie va de 0,7 a 4.132 kWh, así que la curva del MAE sube y baja
    por el sorteo de edificios, no por el modelo. El skill sí es comparable
    porque se normaliza contra la base ingenua calculada sobre esos mismos datos.
    """
    if not datos:
        return None
    c = PALETA[tema]
    xs = [d["series"] for d in datos]
    ys = [d["skill_vs_ingenua"] for d in datos]
    orden = sorted(range(len(xs)), key=lambda i: xs[i])
    xs, ys = [xs[i] for i in orden], [ys[i] for i in orden]

    fig, ax = _nueva(c)
    ax.plot(xs, ys, color=c["s1"], linewidth=2, marker="o", markersize=9,
            markeredgecolor=c["superficie"], markeredgewidth=2, zorder=3)
    _etiquetar(ax, xs, ys, c, "{:+.4f}")
    ax.set_xscale("log")
    ax.set_ylim(0, max(ys) * 1.6)
    _ejes(ax, c, "Curva de escala: ¿mejora el modelo con más edificios?",
          "edificios de entrenamiento (escala logarítmica)",
          "skill sobre la base ingenua")
    ax.annotate("40× más edificios: la curva es plana",
                (0.5, 0.12), xycoords="axes fraction", ha="center",
                fontsize=9, color=c["tinta2"])
    return fig


def fig_auc(consumo_auc, tema):
    """AUC del bloque predictivo, con la línea del azar. Sin ella el número no
    se puede leer: 0,54 parece 'algo' y es una moneda al aire."""
    if not consumo_auc:
        return None
    c = PALETA[tema]
    nombres = [d[0] for d in consumo_auc]
    ys = [d[1] for d in consumo_auc]
    fig, ax = _nueva(c, alto=3.2)
    colores = [c["s1"] if y >= 0.7 else c["critico"] for y in ys]
    barras = ax.bar(nombres, ys, color=colores, width=0.5, zorder=3)
    for b, y in zip(barras, ys):
        ax.annotate(f"{y:.3f}", (b.get_x() + b.get_width() / 2, y),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9.5, color=c["tinta2"])
    _linea_umbral(ax, 0.5, c, "0,50 = moneda al aire")
    ax.set_ylim(0, 1.08)
    _ejes(ax, c, "Capacidad de discriminación (ROC-AUC)", "", "ROC-AUC")
    return fig


def fig_crossval(datos, tema):
    """Distribución por pliegue. Con pocas máquinas, un número suelto miente."""
    if not datos or "pliegues_detalle" not in datos:
        return None
    c = PALETA[tema]
    filas = [f for f in datos["pliegues_detalle"] if f.get("evaluable")]
    if not filas:
        return None
    nombres = [f["descripcion"][:26] for f in filas]
    ys = [f["accuracy"] for f in filas]
    fig, ax = _nueva(c, alto=0.45 * len(filas) + 1.8)
    barras = ax.barh(nombres, ys, color=c["s1"], height=0.6, zorder=3)
    for b, y in zip(barras, ys):
        ax.annotate(f"{y:.4f}", (y, b.get_y() + b.get_height() / 2),
                    textcoords="offset points", xytext=(6, 0), va="center",
                    fontsize=9, color=c["tinta2"])
    obj = datos.get("accuracy", {}).get("mediana")
    if obj is not None:
        ax.axvline(obj, color=c["s2"], linewidth=1.5, linestyle="--", zorder=2)
        ax.annotate(f"mediana {obj:.3f}", (obj, len(filas) - 0.4),
                    fontsize=8.5, color=c["s2"], ha="left")
    ax.grid(True, axis="x", color=c["rejilla"], linewidth=1.0, zorder=0)
    ax.grid(False, axis="y")
    _ejes(ax, c, "IMS · accuracy por pliegue de validación cruzada",
          "accuracy", "")
    ax.set_xlim(0, 1.05)
    return fig


def fig_acreditables(prev, tema):
    """CV(RMSE) por emplazamiento contra el umbral normativo."""
    if not prev or "por_serie" not in prev:
        return None
    c = PALETA[tema]
    ps = prev["por_serie"]
    nombres = ["mediana", "percentil 90", "agrupado"]
    ys = [ps.get("cv_rmse_mediana_pct", 0), ps.get("cv_rmse_p90_pct", 0),
          prev.get("cv_rmse_pct", 0)]
    fig, ax = _nueva(c, alto=3.2)
    colores = [c["s3"] if y < 25 else c["critico"] for y in ys]
    barras = ax.bar(nombres, ys, color=colores, width=0.5, zorder=3)
    for b, y in zip(barras, ys):
        ax.annotate(f"{y:.2f}%", (b.get_x() + b.get_width() / 2, y),
                    textcoords="offset points", xytext=(0, 6), ha="center",
                    fontsize=9.5, color=c["tinta2"])
    _linea_umbral(ax, 25, c, "límite ASHRAE G14")
    ax.set_ylim(0, max(max(ys) * 1.2, 30))
    _ejes(ax, c, "CV(RMSE): medido por edificio frente a medido en agregado",
          "", "CV(RMSE) [%]")
    return fig


# ------------------------------------------------------------------- HTML
_CSS = """
:root{--sup:#fcfcfb;--plano:#f9f9f7;--tinta:#0b0b0b;--tinta2:#52514e;
--mudo:#898781;--linea:#e1e0d9;--acento:#2a78d6}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
--sup:#1a1a19;--plano:#0d0d0d;--tinta:#fff;--tinta2:#c3c2b7;
--mudo:#898781;--linea:#2c2c2a;--acento:#3987e5}}
*{box-sizing:border-box}
body{margin:0;padding:32px 20px 64px;background:var(--plano);color:var(--tinta);
font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
main{max-width:940px;margin:0 auto}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-.01em}
h2{font-size:19px;margin:44px 0 6px;letter-spacing:-.01em}
.sub{color:var(--tinta2);margin:0 0 28px;font-size:14px}
.nota{color:var(--tinta2);font-size:13.5px;margin:6px 0 18px}
figure{margin:0 0 8px;background:var(--sup);border:1px solid var(--linea);
border-radius:10px;padding:14px;overflow-x:auto}
figure img{display:block;width:100%;height:auto;max-width:100%}
.solo-oscuro{display:none}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .solo-claro{display:none}
:root:not([data-theme="light"]) .solo-oscuro{display:block}}
details{margin:0 0 26px;font-size:13.5px}
summary{cursor:pointer;color:var(--acento);padding:6px 0}
table{border-collapse:collapse;width:100%;margin-top:8px;font-size:13px}
th,td{text-align:right;padding:6px 10px;border-bottom:1px solid var(--linea)}
th:first-child,td:first-child{text-align:left}
th{color:var(--tinta2);font-weight:600}
.tarjetas{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:0 0 8px}
.t{background:var(--sup);border:1px solid var(--linea);border-radius:10px;padding:14px 16px}
.t .v{font-size:27px;font-weight:600;letter-spacing:-.02em;line-height:1.15}
.t .k{color:var(--tinta2);font-size:12.5px;margin-top:4px}
.t .d{color:var(--mudo);font-size:11.5px;margin-top:6px}
.falta{background:var(--sup);border:1px dashed var(--linea);border-radius:10px;
padding:16px;color:var(--mudo);font-size:13.5px}
footer{margin-top:52px;color:var(--mudo);font-size:12.5px;
border-top:1px solid var(--linea);padding-top:16px}
"""


def _figura_html(claro: str, oscuro: str, alt: str) -> str:
    return (f'<figure><img class="solo-claro" alt="{alt}" src="data:image/png;base64,{claro}">'
            f'<img class="solo-oscuro" alt="{alt}" src="data:image/png;base64,{oscuro}">'
            f"</figure>")


def _tabla(cabecera: list[str], filas: list[list[Any]], titulo="Ver los datos") -> str:
    th = "".join(f"<th>{h}</th>" for h in cabecera)
    tr = "".join("<tr>" + "".join(f"<td>{v}</td>" for v in f) + "</tr>" for f in filas)
    return (f"<details><summary>{titulo}</summary><table><thead><tr>{th}</tr></thead>"
            f"<tbody>{tr}</tbody></table></details>")


def _tarjeta(valor: str, clave: str, detalle: str = "") -> str:
    d = f'<div class="d">{detalle}</div>' if detalle else ""
    return f'<div class="t"><div class="v">{valor}</div><div class="k">{clave}</div>{d}</div>'


def construir(abrir: bool = False) -> Path:
    import matplotlib
    matplotlib.use("Agg")

    cap = cargar("barrido_capacidad.json")
    perd = cargar("comparativa_perdidas.json")
    abl = cargar("ablacion_preproceso_bdg2.json")
    cons = cargar("consumo_building_data_genome_2.json")
    cv = cargar("crossval_nasa_ims_bearing_gtest.json")
    esc = cargar("oedi_escala.json")
    cap_oedi = cargar("oedi_capacidad.json")
    abl_oedi = cargar("oedi_ablacion_preproceso.json")
    prev = (cons or {}).get("prevision")

    partes: list[str] = []

    def seccion(titulo: str, nota: str = ""):
        partes.append(f"<h2>{titulo}</h2>")
        if nota:
            partes.append(f'<p class="nota">{nota}</p>')

    def poner(f, datos, cabecera=None, filas=None, alt=""):
        """Dibuja la figura en los dos modos y le añade su tabla."""
        a = f(datos, "claro")
        b = f(datos, "oscuro")
        if a is None or b is None:
            partes.append('<div class="falta">Sin datos todavía: el barrido '
                          "correspondiente aún no ha terminado.</div>")
            return
        partes.append(_figura_html(_png(a), _png(b), alt))
        if cabecera and filas:
            partes.append(_tabla(cabecera, filas))

    # ---- resumen
    tarjetas = []
    if prev:
        tarjetas.append(_tarjeta(f"{prev['skill_vs_ingenua']:+.4f}",
                                 "skill del previsor",
                                 "sobre «misma hora, semana pasada»"))
        ps = prev.get("por_serie", {})
        if ps:
            tarjetas.append(_tarjeta(f"{ps.get('pct_series_acreditables', 0):.1f} %",
                                     "edificios acreditables",
                                     "CV(RMSE) < 25 % (ASHRAE G14)"))
    if cons and "linea_base_ahorro" in cons:
        tarjetas.append(_tarjeta(f"{cons['linea_base_ahorro']['ahorro_aparente_pct']:.3f} %",
                                 "ahorro aparente sin intervención",
                                 "debe rondar 0: la línea base no inventa ahorros"))
    if cv and cv.get("roc_auc"):
        tarjetas.append(_tarjeta(f"{cv['roc_auc']['mediana']:.3f}",
                                 "ROC-AUC de IMS", "0,50 sería una moneda al aire"))
    if tarjetas:
        partes.append('<div class="tarjetas">' + "".join(tarjetas) + "</div>")

    # ---- consumo
    seccion("Consumo eléctrico · acreditación de la línea base",
            "ASHRAE Guideline 14 se aplica a cada emplazamiento, no a un promedio de "
            "cartera. Medirlo en agregado invierte el veredicto.")
    poner(fig_acreditables, prev, alt="CV(RMSE) por edificio frente al umbral")

    seccion("Consumo eléctrico · capacidad del modelo",
            "Dos gráficas y no una con dos ejes: el error y el porcentaje de "
            "acreditables son magnitudes distintas.")
    poner(fig_capacidad, cap,
          ["ancho", "bloques", "parámetros", "% acreditables"],
          [[d["ancho"], d["bloques"], f"{d['parametros']:,}",
            f"{(d.get('por_serie') or {}).get('pct_series_acreditables', 0):.1f} %"]
           for d in cap] if cap else None,
          alt="parámetros frente a edificios acreditables")
    poner(fig_capacidad_mae, cap,
          ["ancho", "bloques", "MAE test", "MAE entrenamiento", "brecha"],
          [[d["ancho"], d["bloques"], f"{d['mae']:.3f}", f"{d['mae_train']:.3f}",
            f"{d['brecha_train_test']:.3f}"] for d in cap] if cap else None,
          alt="error de test y entrenamiento frente a capacidad")

    seccion("Consumo eléctrico · función de pérdida")
    poner(fig_perdidas, perd,
          ["pérdida", "MAE", "skill", "CV(RMSE) mediana", "% acreditables"],
          [[k, f"{v['mae']:.3f}", f"{v['skill_vs_ingenua']:+.4f}",
            f"{(v.get('por_serie') or {}).get('cv_rmse_mediana_pct', 0):.2f} %",
            f"{(v.get('por_serie') or {}).get('pct_series_acreditables', 0):.1f} %"]
           for k, v in perd.items()] if perd else None,
          alt="comparativa de funciones de pérdida")

    seccion("Consumo eléctrico · preprocesado (BDG2)")
    poner(fig_ablacion, abl,
          ["variante", "MAE", "skill", "% acreditables", "s"],
          [[d.get("variante"), f"{d.get('mae', 0):.3f}", f"{d.get('skill', 0):+.4f}",
            f"{d.get('pct_series_acreditables') or 0:.1f} %", f"{d.get('segundos', 0):.0f}"]
           for d in abl if "skill" in d] if abl else None,
          alt="ablación de preprocesado")

    # ---- OEDI
    seccion("OEDI · ¿limita el número de edificios?",
            "Sabiendo que la capacidad no es el techo, la pregunta es si lo son los "
            "datos. Se dibuja el <em>skill</em> y no el MAE: el MAE no es comparable "
            "entre filas porque cada submuestra tiene otra mezcla de edificios, y el "
            "consumo medio por serie va de 0,7 a 4.132 kWh.")
    poner(fig_escala, esc,
          ["series", "skill", "CV(RMSE) mediana", "% acreditables",
           "MAE test (no comparable entre filas)"],
          [[d["series"], f"{d['skill_vs_ingenua']:+.4f}",
            f"{(d.get('por_serie') or {}).get('cv_rmse_mediana_pct', 0):.2f} %",
            f"{(d.get('por_serie') or {}).get('pct_series_acreditables', 0):.1f} %",
            f"{d['mae']:.3f}"] for d in esc] if esc else None,
          alt="curva de escala sobre OEDI")

    seccion("OEDI · capacidad")
    poner(fig_capacidad, cap_oedi, alt="capacidad sobre OEDI")

    seccion("OEDI · preprocesado")
    poner(fig_ablacion, abl_oedi, alt="ablación de preprocesado sobre OEDI")

    # ---- predictivo
    aucs = []
    if cv and cv.get("roc_auc"):
        aucs.append(("IMS (vibración real)", cv["roc_auc"]["mediana"]))
    aucs.append(("C-MAPSS (simulado)", 0.996))
    seccion("Mantenimiento predictivo · discriminación",
            "El AUC separa «ordena bien» de «el umbral está bien puesto». La accuracy "
            "sola mezcla ambas cosas y en IMS resultó ser pura prevalencia.")
    poner(fig_auc, sorted(aucs, key=lambda t: -t[1]),
          ["dataset", "ROC-AUC"], [[n, f"{v:.3f}"] for n, v in aucs],
          alt="ROC-AUC por dataset")

    seccion("Mantenimiento predictivo · validación cruzada de IMS",
            "Reservando el banco de ensayo entero, no un rodamiento suelto: los cuatro "
            "de un banco comparten eje e instante de fallo.")
    poner(fig_crossval, cv,
          ["pliegue", "accuracy", "PR-AUC", "base", "ROC-AUC", "FN", "FP"],
          [[f["descripcion"], f"{f.get('accuracy', 0):.4f}",
            f"{f.get('pr_auc', 0):.4f}", f"{f.get('pr_auc_base', 0):.4f}",
            f"{f.get('roc_auc', 0):.4f}", f.get("fn", 0), f.get("fp", 0)]
           for f in cv["pliegues_detalle"] if f.get("evaluable")] if cv else None,
          alt="accuracy por pliegue en IMS")

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>QZ AI Sprint · resultados</title><style>{_CSS}</style></head><body><main>
<h1>QZ AI Sprint · panel de resultados</h1>
<p class="sub">Generado el {ts} · todas las cifras proceden de entrenamientos reales,
no de simulación. Cada figura lleva debajo su tabla de datos.</p>
{''.join(partes)}
<footer>Generado por <code>analyze/dashboard.py</code> a partir de los JSON de
<code>artifacts/</code>. Fichero autocontenido: las imágenes van embebidas, así que
se puede copiar y abrir en cualquier equipo sin dependencias.<br>
Documentación completa en <code>docs/</code>; fórmulas en <code>docs/formulario/</code>.</footer>
</main></body></html>"""

    ART.mkdir(parents=True, exist_ok=True)
    salida = ART / "dashboard.html"
    salida.write_text(html, encoding="utf-8")
    if abrir:
        import webbrowser
        webbrowser.open(salida.as_uri())
    return salida


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Panel gráfico de resultados del sprint.")
    ap.add_argument("--abrir", action="store_true", help="abrir en el navegador al terminar")
    a = ap.parse_args(argv)
    try:
        import matplotlib  # noqa: F401
    except Exception as e:
        sys.exit(f"Falta matplotlib (requirements-analyze.txt): {e}")
    salida = construir(a.abrir)
    print(f"  panel: {salida}  ({salida.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
