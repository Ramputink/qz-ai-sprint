"""Registro de datasets del plan — una entrada por clave usada en config.yaml.

Cada `DatasetSpec` describe DE DONDE se baja, COMO (http directo, varios ficheros,
Kaggle, Hugging Face o descarga manual con registro) y CUANTO ocupa. El orquestador
llama a `resolve()` con las claves del config y pasa las specs al descargador.

`kind` clasifica para que sirve cada dataset en el objetivo predictivo:
  run_to_failure  -> trayectorias hasta el fallo: de aqui sale la RUL y la anticipacion.
  vibration_fault -> vibracion etiquetada por tipo de fallo: entrena el clasificador.
  timeseries      -> corpus grande de series (preentreno / prevision).
  consumption     -> consumo electrico (prevision + NILM).
  anomaly         -> deteccion de anomalia / intrusion OT.

Las URLs se verificaron el 2026-08-27. Si alguna cae, el descargador lo registra y
sigue con el resto (una fuente muerta no debe tumbar un sprint de 4 dias).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    method: str          # http | http_multi | kaggle | hf | manual
    gb: float            # tamano aproximado en disco (comprimido)
    location: str        # URL principal o descriptor de la fuente
    kind: str = ""
    files: tuple[str, ...] = field(default_factory=tuple)   # para http_multi
    extract: bool = True
    filename: str = ""   # nombre a dar al fichero cuando la URL no lo lleva
    optional: bool = False   # si falla, no se considera error del sprint
    notes: str = ""


# --- CWRU: ficheros .mat sueltos (baseline sano + fallos a 12 kHz drive-end) ------
_CWRU_BASE = "https://engineering.case.edu/sites/default/files"
_CWRU_FILES = (
    # baseline sano (0-3 HP)
    "97.mat", "98.mat", "99.mat", "100.mat",
    # inner race 0.007 / 0.014 / 0.021 pulgadas
    "105.mat", "106.mat", "107.mat", "108.mat",
    "169.mat", "170.mat", "171.mat", "172.mat",
    "209.mat", "210.mat", "211.mat", "212.mat",
    # ball 0.007 / 0.014 / 0.021
    "118.mat", "119.mat", "120.mat", "121.mat",
    "185.mat", "186.mat", "187.mat", "188.mat",
    "222.mat", "223.mat", "224.mat", "225.mat",
    # outer race centrada 0.007 / 0.014 / 0.021
    "130.mat", "131.mat", "132.mat", "133.mat",
    "197.mat", "198.mat", "199.mat", "200.mat",
    "234.mat", "235.mat", "236.mat", "237.mat",
)


REGISTRY: dict[str, DatasetSpec] = {
    # ---------------- Tier A: nucleo predictivo (run-to-failure) ----------------
    "cmapss": DatasetSpec(
        key="cmapss", method="http", gb=0.013, kind="run_to_failure",
        location="https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
        notes="C-MAPSS FD001-FD004: 709 motores run-to-failure con RUL. Pequeno y denso: "
              "es el mejor arranque para validar el objetivo de anticipacion."),
    "ncmapss": DatasetSpec(
        key="ncmapss", method="http", gb=15.8, kind="run_to_failure",
        location="https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip",
        notes="N-CMAPSS (DS01-DS08): degradacion realista con perfiles de vuelo reales, HDF5."),
    "nasa_ims_bearing": DatasetSpec(
        key="nasa_ims_bearing", method="http", gb=1.08, kind="run_to_failure",
        location="https://phm-datasets.s3.amazonaws.com/NASA/4.+Bearings.zip",
        notes="IMS: 3 ensayos de rodamiento hasta rotura, vibracion cruda a 20 kHz. "
              "El caso mas parecido al motor/rotor de planta."),
    "metropt3": DatasetSpec(
        key="metropt3", method="http", gb=0.22, kind="run_to_failure",
        location="https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip",
        notes="MetroPT-3 (UCI): compresor APU de metro, 15 meses, fallos fechados."),
    "cwru_bearing": DatasetSpec(
        key="cwru_bearing", method="http_multi", gb=0.12, kind="vibration_fault",
        location=_CWRU_BASE, files=_CWRU_FILES, extract=False,
        notes="Case Western: vibracion etiquetada por tipo y severidad de fallo."),
    "mfpt_bearing": DatasetSpec(
        key="mfpt_bearing", method="http", gb=0.05, kind="vibration_fault", optional=True,
        location="https://github.com/mathworks/RollingElementBearingFaultDiagnosis-Data/archive/refs/heads/master.zip",
        notes="Baseline + fallo de pista interna/externa. La descarga directa de mfpt.org "
              "dejo de servir el zip (devuelve HTML), asi que se usa el repaquetado de "
              "MathWorks de los mismos datos MFPT. Opcional: CWRU cubre el mismo papel."),
    "skab": DatasetSpec(
        key="skab", method="http", gb=0.006, kind="anomaly",
        location="https://github.com/waico/SKAB/archive/refs/heads/master.zip",
        notes="SKAB: banco con bomba de agua, anomalias etiquetadas (validacion de alarma)."),
    "paderborn_bearing": DatasetSpec(
        key="paderborn_bearing", method="manual", gb=20.0, kind="vibration_fault", optional=True,
        location="https://groups.uni-paderborn.de/kat/BearingDataCenter/",
        notes="KAt-DataCenter: ficheros .rar por rodamiento; requiere unrar/7z y bajarlos "
              "uno a uno. Opcional: CWRU+MFPT ya cubren el clasificador de fallo."),

    # ---------------- Tier B: corpus grande (foundation model) ------------------
    "monash_tsf": DatasetSpec(
        key="monash_tsf", method="hf", gb=2.0, kind="timeseries", optional=True,
        location="Monash-University/monash_tsf",
        notes="Monash Time Series Forecasting Archive via Hugging Face."),
    "lotsa": DatasetSpec(
        key="lotsa", method="hf", gb=925.0, kind="timeseries", optional=True,
        location="Salesforce/lotsa_data",
        notes="925 GB. Solo si hay ~1 TB libre y dias de margen: preentreno del foundation model."),

    # ---------------- Tier C: CONSUMO ELECTRICO (prioritario) -------------------
    # Este es el bloque que responde a "se puede optimizar el consumo". Verificado
    # el 2026-08-28.
    "building_data_genome_2": DatasetSpec(
        key="building_data_genome_2", method="http", gb=0.6, kind="consumption",
        location="https://zenodo.org/api/records/3887306/files/buds-lab/building-data-genome-project-2-v1.0.zip/content",
        filename="bdg2_v1.0.zip",
        notes="BDG2: 1636 edificios reales, 2 anos de consumo horario CON meteorologia y "
              "metadatos (uso, superficie, ano). Es el mejor punto de partida: sirve a la vez "
              "para prevision de carga, deteccion de desperdicio y linea base de ahorro. "
              "OJO: el zip de GitHub NO vale, sus CSV son punteros de Git LFS; hay que bajarlo "
              "de Zenodo, que sirve los datos de verdad."),
    "electricity_load_diagrams": DatasetSpec(
        key="electricity_load_diagrams", method="http", gb=0.26, kind="consumption",
        location="https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        notes="370 clientes, 15 min, 4 anos. Referencia clasica de prevision de carga: "
              "permite comparar contra resultados publicados."),
    "steel_industry_energy": DatasetSpec(
        key="steel_industry_energy", method="http", gb=0.001, kind="consumption",
        location="https://archive.ics.uci.edu/static/public/851/steel+industry+energy+consumption.zip",
        notes="Planta siderurgica: consumo horario con potencia REACTIVA, factor de potencia, "
              "CO2 y tipo de carga. Pequeno, pero es el unico industrial de verdad y trae las "
              "variables sobre las que se actua para optimizar."),
    "low_carbon_london": DatasetSpec(
        key="low_carbon_london", method="http", gb=0.8, kind="consumption",
        location="https://data.london.gov.uk/download/smartmeter-energy-use-data-in-london-households/3527bf39-d93e-4071-8451-df2ade1ea4f2/LCL-June2015v2.zip",
        filename="LCL-June2015v2.zip",
        notes="5.567 hogares reales de Londres, media hora, 2011-2014: ~167 millones de "
              "lecturas (8,5 GB en un solo CSV). Es el salto de escala en datos MEDIDOS. "
              "El zip usa una compresion que zipfile no abre; el extractor cae en 7z."),
    "ampds2": DatasetSpec(
        key="ampds2", method="http", gb=0.31, kind="consumption", optional=True,
        location="https://dataverse.harvard.edu/api/access/datafile/3661112",
        extract=False, filename="AMPds2.h5",
        notes="AMPds2: 2 anos a 1 min con 21 submedidas -> desagregacion NILM (saber DONDE "
              "se va la energia sin instrumentar cada maquina)."),
    "ukdale_csv": DatasetSpec(
        key="ukdale_csv", method="manual", gb=3.5, kind="consumption", optional=True,
        location="https://data.ukedc.rl.ac.uk/simplebrowse/edc/efficiency/residential/EnergyConsumption/Domestic/UK-DALE-2017/",
        notes="NILM residencial. La URL directa dejo de servir el zip (devuelve HTML) el "
              "2026-08-28; hay que navegar el portal. AMPds2 cubre el mismo papel."),

    # ---------------- Tier D: anomalia TS + intrusion OT ------------------------
    "nab": DatasetSpec(
        key="nab", method="http", gb=0.04, kind="anomaly", optional=True,
        location="https://github.com/numenta/NAB/archive/refs/heads/master.zip",
        notes="Numenta Anomaly Benchmark: referencia estandar de anomalia en streaming."),
    "smd": DatasetSpec(
        key="smd", method="http", gb=0.15, kind="anomaly", optional=True,
        location="https://github.com/NetManAIOps/OmniAnomaly/archive/refs/heads/master.zip",
        notes="Server Machine Dataset (dentro de OmniAnomaly): 28 maquinas multivariante."),
}


def resolve(keys: list[str]) -> list[DatasetSpec]:
    """Traduce claves del config a specs. Ignora (avisando) las desconocidas."""
    out: list[DatasetSpec] = []
    seen: set[str] = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        spec = REGISTRY.get(k)
        if spec is None:
            print(f"  [registry] clave desconocida, se ignora: {k}")
            continue
        out.append(spec)
    return out


def total_gb(specs: list[DatasetSpec]) -> float:
    return round(sum(s.gb for s in specs), 2)


def by_kind(specs: list[DatasetSpec], kind: str) -> list[DatasetSpec]:
    return [s for s in specs if s.kind == kind]
