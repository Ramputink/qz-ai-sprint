"""Catálogo de datasets — la fuente de verdad de QUÉ descargar y DE DÓNDE.

Cada entrada dice: método de descarga, URL/repo, tamaño, licencia, categoría y si
requiere registro. `download.py` lee este catálogo.

Métodos:
  * "hf"      → Hugging Face Hub (huggingface_hub / datasets)
  * "kaggle"  → Kaggle API (necesita ~/.kaggle/kaggle.json)
  * "url"     → descarga directa por HTTP(S) (wget/requests)
  * "manual"  → requiere registro/solicitud web; se documenta la URL y se salta

Tamaños: 'gb' es orientativo; los marcados con verified=True están confirmados en
la fuente. No se inventan cifras: donde no se conoce, gb=None.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    name: str
    method: str                     # hf | kaggle | url | manual
    location: str                   # repo id, dataset slug, o URL
    category: str                   # tier_a_predictive | tier_b_corpus | tier_c_consumption | tier_d_anomaly_ot
    gb: Optional[float] = None
    verified: bool = False
    license: str = "ver fuente"
    requires_auth: bool = False
    files: list[str] = field(default_factory=list)  # para descargas 'url' con varios ficheros
    notes: str = ""


REGISTRY: dict[str, DatasetSpec] = {
    # ---- Tier A · mantenimiento predictivo (NÚCLEO) ----------------------
    "nasa_ims_bearing": DatasetSpec(
        "nasa_ims_bearing", "NASA IMS Bearing (run-to-failure)", "kaggle",
        "vinayak123tyagi/bearing-dataset", "tier_a_predictive", gb=6.0, verified=False,
        license="NASA (dominio público)", notes="Espejo Kaggle del NASA PCoE IMS. Trayectorias de degradación → RUL."),
    "ncmapss": DatasetSpec(
        "ncmapss", "N-CMAPSS (turbofan RUL alta fidelidad)", "kaggle",
        "behrad3d/nasa-cmaps", "tier_a_predictive", gb=27.0, verified=False,
        license="NASA (dominio público)", notes="Regresión de vida útil restante. Alternativa oficial: NASA PCoE."),
    "paderborn_bearing": DatasetSpec(
        "paderborn_bearing", "Paderborn KAt Bearing (vibración+corriente)", "url",
        "https://zenodo.org/records/15845309", "tier_a_predictive", gb=20.8, verified=True,
        license="CC (académico)", requires_auth=False,
        notes="Fallo de rodamiento con corriente de motor. Fuente oficial mb.uni-paderborn.de (registro) o espejo Zenodo."),
    "cwru_bearing": DatasetSpec(
        "cwru_bearing", "CWRU Bearing", "kaggle",
        "brjapon/cwru-bearing-datasets", "tier_a_predictive", gb=1.0, verified=False,
        license="Case Western (libre académico)", notes="Benchmark clásico de diagnóstico de rodamiento."),
    "mfpt_bearing": DatasetSpec(
        "mfpt_bearing", "MFPT Bearing Fault", "url",
        "https://www.mfpt.org/fault-data-sets/", "tier_a_predictive", gb=0.5, verified=False,
        license="MFPT (libre)", requires_auth=False, notes="Fallos de rodamiento etiquetados."),
    "metropt3": DatasetSpec(
        "metropt3", "MetroPT-3 (compresor/motor metro)", "url",
        "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip", "tier_a_predictive",
        gb=1.7, verified=True, license="CC BY 4.0",
        files=["https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"],
        notes="Predictivo real (presión/corriente motor). UCI ML Repository."),
    "skab": DatasetSpec(
        "skab", "SKAB (testbed bomba/motor)", "url",
        "https://github.com/waico/SKAB/archive/refs/heads/master.zip", "tier_a_predictive",
        gb=0.2, verified=False, license="GPL-3.0",
        files=["https://github.com/waico/SKAB/archive/refs/heads/master.zip"],
        notes="Anomalías en máquina rotativa (Skoltech)."),

    # ---- Tier B · corpus grande para foundation model --------------------
    "monash_tsf": DatasetSpec(
        "monash_tsf", "Monash Time Series Forecasting Archive", "hf",
        "Monash-University/monash_tsf", "tier_b_corpus", gb=4.0, verified=False,
        license="CC BY 4.0", notes="30 datasets de previsión (benchmark)."),
    "lotsa": DatasetSpec(
        "lotsa", "LOTSA (Salesforce, corpus foundation)", "hf",
        "Salesforce/lotsa_data", "tier_b_corpus", gb=925.0, verified=True,
        license="Apache-2.0", notes="925 GB. Solo descárgalo si tienes disco y días; ideal para preentrenar."),

    # ---- Tier C · consumo eléctrico --------------------------------------
    "ukdale_csv": DatasetSpec(
        "ukdale_csv", "UK-DALE (disagregado CSV)", "url",
        "https://data.ukedc.rl.ac.uk/simplebrowse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/UK-DALE-disaggregated",
        "tier_c_consumption", gb=3.5, verified=True, license="CC BY 4.0",
        notes="Consumo doméstico desagregado (NILM). El de 16 kHz (7,6 TB) es aparte y opcional."),
    "building_data_genome_2": DatasetSpec(
        "building_data_genome_2", "Building Data Genome Project 2", "url",
        "https://github.com/buds-lab/building-data-genome-project-2/archive/refs/heads/master.zip",
        "tier_c_consumption", gb=1.0, verified=False, license="MIT",
        files=["https://github.com/buds-lab/building-data-genome-project-2/archive/refs/heads/master.zip"],
        notes="3.053 medidores de edificios no residenciales (horario)."),

    # ---- Tier D · anomalía TS + intrusión OT (secundario) ----------------
    "nab": DatasetSpec(
        "nab", "NAB (Numenta Anomaly Benchmark)", "url",
        "https://github.com/numenta/NAB/archive/refs/heads/master.zip", "tier_d_anomaly_ot",
        gb=0.2, verified=False, license="AGPL-3.0",
        files=["https://github.com/numenta/NAB/archive/refs/heads/master.zip"],
        notes="Benchmark de anomalías en streaming."),
    "smd": DatasetSpec(
        "smd", "SMD (Server Machine Dataset)", "url",
        "https://github.com/NetManAIOps/OmniAnomaly/archive/refs/heads/master.zip", "tier_d_anomaly_ot",
        gb=0.5, verified=False, license="ver repo",
        files=["https://github.com/NetManAIOps/OmniAnomaly/archive/refs/heads/master.zip"],
        notes="Anomalías multivariante (28 máquinas)."),
    "swat_wadi": DatasetSpec(
        "swat_wadi", "SWaT / WADI (intrusión OT)", "manual",
        "https://itrust.sutd.edu.sg/itrust-labs_datasets/", "tier_d_anomaly_ot",
        gb=None, verified=False, license="Acuerdo iTrust", requires_auth=True,
        notes="Requiere solicitud a iTrust (~3 días). Descárgalo a mano y colócalo en data/swat_wadi/."),
}


def resolve(keys: list[str]) -> list[DatasetSpec]:
    """Devuelve los specs para las claves pedidas (ignora desconocidas con aviso)."""
    out = []
    for k in keys:
        if k in REGISTRY:
            out.append(REGISTRY[k])
        else:
            print(f"[registry] AVISO: dataset desconocido '{k}' (ignorado)")
    return out


def total_gb(specs: list[DatasetSpec]) -> float:
    return round(sum(s.gb or 0.0 for s in specs), 1)


if __name__ == "__main__":
    # Lista el catálogo y el total (para verificar en Mac).
    by_cat: dict[str, list[DatasetSpec]] = {}
    for s in REGISTRY.values():
        by_cat.setdefault(s.category, []).append(s)
    grand = 0.0
    for cat, specs in by_cat.items():
        gb = total_gb(specs)
        grand += gb
        print(f"\n== {cat} ({gb} GB) ==")
        for s in specs:
            flag = " [auth]" if s.requires_auth else ""
            v = "✓" if s.verified else "≈"
            print(f"  {s.key:24} {v}{(str(s.gb)+' GB'):>10}{flag}  {s.method:6} {s.location[:60]}")
    print(f"\nTOTAL catálogo: ~{round(grand,1)} GB (LOTSA solo aporta 925 GB → meta ≥450 GB holgada)")
