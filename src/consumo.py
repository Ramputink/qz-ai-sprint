"""Modelos de CONSUMO ELECTRICO — las tres piezas, con sus lineas base.

Responde a la pregunta central del proyecto: si se puede optimizar el consumo. Se
descompone en tres modelos entrenables en el PC (RTX 5090), mas el protocolo que
convierte al primero en un numero de ahorro defendible:

  1. PREVISION DE CARGA  -> `entrenar_previsor`
     Predice el consumo H horas por delante. Es el sustrato de todo lo demas.

  2. DESAGREGACION NILM  -> `entrenar_nilm`
     Descompone la potencia agregada en sus consumidores (seq2point). Dice DONDE
     se va la energia sin poner un contador en cada maquina.

  3. DETECCION DE DESPERDICIO -> `detectar_desperdicio`
     Consumo real por encima del esperado dadas las condiciones. Encuentra equipos
     olvidados encendidos, consignas mal puestas y degradacion de rendimiento.

  4. LINEA BASE CONTRAFACTUAL -> `medir_ahorro`
     El unico que produce euros. Entrena con el periodo ANTERIOR a una intervencion
     y predice lo que habria consumido DESPUES; el ahorro es medido menos predicho
     (protocolo IPMVP Opcion C). Sin esto no se puede demostrar un ahorro: si el
     consumo baja un 8 % pero hizo mas frio o se produjo menos, no se ha ahorrado.

REGLA DE MEDICION
-----------------
Toda metrica va acompanada de su LINEA BASE. En consumo electrico la base ingenua
--"lo mismo que a esta hora la semana pasada"-- es brutalmente fuerte porque los
edificios son muy periodicos, y un MAE suelto no dice si el modelo aporta algo.
Se reporta el SKILL SCORE (1 - error_modelo/error_base): 0 es empatar con no hacer
nada, y negativo es estorbar. Se anade CV(RMSE), que es la metrica que exige
ASHRAE Guideline 14 para aceptar una linea base de ahorro (<25 % en datos horarios).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

CONTEXTO_H = 168      # una semana de contexto: cubre el ciclo semanal completo
HORIZONTE_H = 24      # un dia por delante: el horizonte de decision operativa
ESTACIONAL = 168      # el retardo de la base ingenua = misma hora, semana pasada


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# =====================  metricas  ============================================
def metricas(real: np.ndarray, pred: np.ndarray, base: np.ndarray | None = None) -> dict[str, Any]:
    """Error del modelo y, si se da la base ingenua, cuanto le gana.

    CV(RMSE) y NMBE son las dos que pide ASHRAE Guideline 14 para aceptar una linea
    base: la primera mide dispersion, la segunda si el modelo esta sesgado (un
    modelo que sobreestima sistematicamente inventa ahorros que no existen).
    """
    real, pred = np.asarray(real, float).ravel(), np.asarray(pred, float).ravel()
    ok = np.isfinite(real) & np.isfinite(pred)
    real, pred = real[ok], pred[ok]
    if real.size == 0:
        return {"error": "sin datos validos"}
    err = pred - real
    media = float(np.mean(real)) or 1e-9
    rmse = float(np.sqrt(np.mean(err ** 2)))
    out = {"mae": round(float(np.mean(np.abs(err))), 4),
           "rmse": round(rmse, 4),
           "cv_rmse_pct": round(100 * rmse / abs(media), 2),
           "nmbe_pct": round(100 * float(np.mean(err)) / abs(media), 2),
           "n": int(real.size)}
    if base is not None:
        base = np.asarray(base, float).ravel()[ok]
        mae_base = float(np.mean(np.abs(base - real)))
        out["mae_base_ingenua"] = round(mae_base, 4)
        out["skill_vs_ingenua"] = round(1 - out["mae"] / (mae_base or 1e-9), 4)
        out["cumple_ashrae_g14"] = bool(out["cv_rmse_pct"] < 25.0)
    return out


# =====================  datos  ===============================================
class TareaPrevision:
    """Series de consumo listas para entrenar, troceadas en GPU sobre la marcha.

    El corte es TEMPORAL (pasado -> futuro), no aleatorio: un modelo de consumo
    validado con ventanas barajadas se evalua prediciendo el martes habiendo visto
    el miercoles de la misma semana, y el numero resultante no significa nada.
    """

    def __init__(self, data_dir: Path, key: str, contexto: int = CONTEXTO_H,
                 horizonte: int = HORIZONTE_H, frac_train: float = 0.7,
                 max_series: int = 400, seed: int = 0):
        import torch

        from .data.consumption import load_consumo

        arrays, meta = load_consumo(data_dir, key, "consumo")
        y = arrays["y"]                                     # (S, T)
        if y.shape[0] > max_series:                        # muestreo reproducible
            sel = np.random.default_rng(seed).choice(y.shape[0], max_series, replace=False)
            y = y[np.sort(sel)]
            self.series = [meta["series"][i] for i in np.sort(sel)]
            site = arrays["site_of_serie"][np.sort(sel)]
        else:
            self.series = meta.get("series", [])
            site = arrays["site_of_serie"]

        y = np.nan_to_num(y, nan=0.0)
        self.meta, self.key = meta, key
        self.contexto, self.horizonte = contexto, horizonte
        T = y.shape[1]
        self.corte = int(T * frac_train)

        # normalizacion POR SERIE, con estadisticos del tramo de entrenamiento:
        # usar todo el historico filtraria informacion del futuro.
        mu = y[:, :self.corte].mean(axis=1, keepdims=True)
        sd = y[:, :self.corte].std(axis=1, keepdims=True)
        sd[sd < 1e-6] = 1.0
        self.mu, self.sd = mu.astype(np.float32), sd.astype(np.float32)

        dev = _device()
        self.y = torch.as_tensor((y - mu) / sd).to(dev)              # (S, T)
        self.y_real = torch.as_tensor(y).to(dev)
        tf = arrays["time_feats"]                                     # (T, Ct)
        wx = arrays["weather"]                                        # (W, T, Cw)
        if wx.shape[-1]:
            wmu = wx[:, :self.corte].mean(axis=1, keepdims=True)
            wsd = wx[:, :self.corte].std(axis=1, keepdims=True)
            wsd[wsd < 1e-6] = 1.0
            wx = (wx - wmu) / wsd
        self.tf = torch.as_tensor(tf).to(dev)
        self.wx = torch.as_tensor(wx.astype(np.float32)).to(dev)
        self.site = torch.as_tensor(site.astype(np.int64)).to(dev)
        self.n_series, self.T = self.y.shape
        self.n_cov = tf.shape[1] + wx.shape[-1]

    def _cov(self, s_idx, t0, largo: int):
        """Covariables alineadas: calendario (comun) + meteo del emplazamiento."""
        import torch
        idx = t0[:, None] + torch.arange(largo, device=self.y.device)[None, :]
        cal = self.tf[idx]                                             # (B, L, Ct)
        if self.wx.shape[-1] == 0:
            return cal
        met = self.wx[self.site[s_idx][:, None], idx]                  # (B, L, Cw)
        return torch.cat([cal, met], dim=-1)

    def lote(self, n: int, tramo: str = "train", generador=None):
        """Un lote de ventanas. `tramo` decide de que mitad temporal se sacan."""
        import torch

        L, H = self.contexto, self.horizonte
        lo, hi = (L, self.corte - H) if tramo == "train" else (self.corte, self.T - H)
        if hi <= lo:
            raise RuntimeError(f"{self.key}: tramo '{tramo}' demasiado corto")
        dev = self.y.device
        s = torch.randint(0, self.n_series, (n,), device=dev, generator=generador)
        t = torch.randint(lo, hi, (n,), device=dev, generator=generador)
        ctx_idx = t[:, None] - L + torch.arange(L, device=dev)[None, :]
        fut_idx = t[:, None] + torch.arange(H, device=dev)[None, :]
        ctx = self.y[s[:, None], ctx_idx]                              # (B, L)
        fut = self.y[s[:, None], fut_idx]                              # (B, H)
        cov = self._cov(s, t, H)                                       # (B, H, C)
        # base ingenua: exactamente lo mismo que hace ESTACIONAL pasos antes
        base = self.y[s[:, None], fut_idx - ESTACIONAL]
        return ctx, cov, fut, base, s

    def desnormalizar(self, v, s_idx):
        import torch
        mu = torch.as_tensor(self.mu, device=v.device)[s_idx]
        sd = torch.as_tensor(self.sd, device=v.device)[s_idx]
        return v * sd + mu


# =====================  modelo de prevision  =================================
def construir_previsor(contexto: int, horizonte: int, n_cov: int,
                       ancho: int = 512, bloques: int = 3, parche: int = 24):
    """Previsor por parches: normaliza la ventana, la trocea en dias, y mezcla.

    La normalizacion por ventana (restar media y desviacion del propio contexto) es
    lo que permite que UN SOLO modelo sirva para 400 edificios de escalas muy
    distintas: aprende la forma del perfil, no el nivel absoluto. Es la idea
    central de los previsores modernos tipo PatchTST/N-HiTS, en su version minima.
    """
    import torch
    import torch.nn as nn

    n_parches = contexto // parche

    class Bloque(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(2 * d, d))

        def forward(self, x):
            return x + self.ff(self.norm(x))

    class Previsor(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Linear(n_parches * parche + horizonte * n_cov, ancho)
            self.bloques = nn.Sequential(*[Bloque(ancho) for _ in range(bloques)])
            self.salida = nn.Linear(ancho, horizonte)

        def forward(self, ctx, cov):
            # normalizacion reversible por ventana
            mu = ctx.mean(dim=1, keepdim=True)
            sd = ctx.std(dim=1, keepdim=True).clamp_min(1e-5)
            z = (ctx - mu) / sd
            h = torch.cat([z, cov.flatten(1)], dim=1)
            h = self.bloques(self.emb(h))
            return self.salida(h) * sd + mu        # se devuelve la escala original

    return Previsor()


def _fn_perdida(nombre: str, delta: float = 1.0):
    """Funcion de perdida por nombre.

    Importa mas de lo que parece: la L1 optimiza la MEDIANA, asi que afina el caso
    tipico y se despreocupa de los picos -- que es justo lo que castiga el RMSE, y
    el RMSE es lo que decide si una linea base de ahorro es acreditable (ASHRAE
    G14 exige CV(RMSE) < 25 %). Entrenar mas con L1 mejoraba el MAE y EMPEORABA el
    CV(RMSE). La Huber es el punto medio: cuadratica cerca de cero (atiende a los
    picos) y lineal lejos (no se deja arrastrar por lecturas atipicas del contador).
    """
    import torch.nn.functional as F

    if nombre == "l1":
        return lambda p, y: F.l1_loss(p, y)
    if nombre == "l2":
        return lambda p, y: F.mse_loss(p, y)
    if nombre == "huber":
        return lambda p, y: F.huber_loss(p, y, delta=delta)
    raise ValueError(f"perdida desconocida: {nombre}")


def evaluar_previsor(modelo, tarea: TareaPrevision, n_lotes: int = 20,
                     lote: int = 1024) -> dict[str, Any]:
    """Metricas agregadas y POR SERIE, en kWh reales.

    Lo segundo importa: ASHRAE Guideline 14 se aplica a CADA emplazamiento, no a un
    promedio de la cartera. Un CV(RMSE) agrupado sobre 400 edificios de escalas muy
    distintas lo domina un punado de edificios grandes, y puede suspender aunque la
    mayoria de ellos cumplan de sobra. Lo que decide si se puede certificar es
    cuantos edificios pasan el umbral, no la media.
    """
    import torch

    modelo.eval()
    reales, predichos, bases, series = [], [], [], []
    with torch.no_grad():
        for _ in range(n_lotes):
            ctx, cov, fut, base, s = tarea.lote(lote, "test")
            p = modelo(ctx, cov).float()
            reales.append(tarea.desnormalizar(fut, s).cpu().numpy())
            predichos.append(tarea.desnormalizar(p, s).cpu().numpy())
            bases.append(tarea.desnormalizar(base, s).cpu().numpy())
            series.append(s.cpu().numpy())
    R, P, B = np.concatenate(reales), np.concatenate(predichos), np.concatenate(bases)
    S = np.concatenate(series)

    m = metricas(R, P, B)
    cvs, skills = [], []
    for u in np.unique(S):
        k = S == u
        if k.sum() < 10:
            continue
        mi = metricas(R[k], P[k], B[k])
        if "cv_rmse_pct" in mi:
            cvs.append(mi["cv_rmse_pct"])
            skills.append(mi.get("skill_vs_ingenua", 0.0))
    if cvs:
        cvs_a = np.array(cvs)
        m["por_serie"] = {
            "cv_rmse_mediana_pct": round(float(np.median(cvs_a)), 2),
            "cv_rmse_p90_pct": round(float(np.percentile(cvs_a, 90)), 2),
            "series_que_cumplen_ashrae": int((cvs_a < 25.0).sum()),
            "series_evaluadas": int(len(cvs_a)),
            "pct_series_acreditables": round(100 * float((cvs_a < 25.0).mean()), 1),
            "skill_mediano": round(float(np.median(skills)), 4)}
    return m


def entrenar_previsor(tarea: TareaPrevision, pasos: int = 3000, lote: int = 512,
                      lr: float = 1e-3, perdida: str = "huber", huber_delta: float = 1.0,
                      logger=None, cb=None) -> dict[str, Any]:
    """Entrena el previsor y lo mide CONTRA LA BASE INGENUA."""
    import torch

    dev = _device()
    modelo = construir_previsor(tarea.contexto, tarea.horizonte, tarea.n_cov).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=pasos)
    crit = _fn_perdida(perdida, huber_delta)
    t0 = time.time()

    for paso in range(pasos):
        modelo.train()
        ctx, cov, fut, _, _ = tarea.lote(lote, "train")
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16) if dev == "cuda" else _nulo():
            pred = modelo(ctx, cov)
            p_loss = crit(pred.float(), fut)
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        opt.step(); sched.step()
        if cb and paso % 200 == 0:
            cb(paso / pasos * 100,
               f"paso {paso}/{pasos} · {perdida} {float(p_loss.detach()):.4f}")

    m = evaluar_previsor(modelo, tarea)
    m.update({"dataset": tarea.key, "series": tarea.n_series,
              "contexto_h": tarea.contexto, "horizonte_h": tarea.horizonte,
              "perdida": perdida, "huber_delta": huber_delta if perdida == "huber" else None,
              "pasos": pasos, "segundos": round(time.time() - t0, 1),
              "parametros": sum(p.numel() for p in modelo.parameters())})
    if logger:
        logger.info("previsor", **{k: v for k, v in m.items() if k not in ("n", "por_serie")})
    return {"metricas": m, "modelo": modelo}


class _nulo:
    def __enter__(self): return None
    def __exit__(self, *a): return False


# =====================  NILM (seq2point)  ====================================
def construir_seq2point(ventana: int, n_aparatos: int):
    """CNN seq2point: de una ventana de la acometida al consumo de cada aparato en
    su instante central. Es la arquitectura de referencia en desagregacion y cabe
    de sobra en la 5090."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Conv1d(1, 30, 10, padding="same"), nn.ReLU(),
        nn.Conv1d(30, 30, 8, padding="same"), nn.ReLU(),
        nn.Conv1d(30, 40, 6, padding="same"), nn.ReLU(),
        nn.Conv1d(40, 50, 5, padding="same"), nn.ReLU(),
        nn.Dropout(0.2),
        nn.AdaptiveAvgPool1d(16), nn.Flatten(),
        nn.Linear(50 * 16, 1024), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(1024, n_aparatos),
    )


def entrenar_nilm(data_dir: Path, key: str = "ampds2", ventana: int = 599,
                  pasos: int = 2000, lote: int = 256, frac_train: float = 0.7,
                  logger=None, cb=None) -> dict[str, Any]:
    """Desagrega la acometida en sus circuitos. Corte temporal, como en prevision.

    La base contra la que hay que ganar es "cada aparato consume siempre su media":
    en desagregacion esa base es sorprendentemente dura, porque muchos circuitos son
    casi constantes.
    """
    import torch

    from .data.consumption import load_consumo

    arrays, meta = load_consumo(data_dir, key, "nilm")
    mains = arrays["mains"].astype(np.float32)
    apps = arrays["appliances"].astype(np.float32)
    T, A = apps.shape
    corte = int(T * frac_train)

    mu, sd = float(mains[:corte].mean()), float(mains[:corte].std()) or 1.0
    amu = apps[:corte].mean(axis=0)
    asd = apps[:corte].std(axis=0)
    asd[asd < 1e-6] = 1.0

    dev = _device()
    m_t = torch.as_tensor((mains - mu) / sd).to(dev)
    a_t = torch.as_tensor((apps - amu) / asd).to(dev)
    modelo = construir_seq2point(ventana, A).to(dev)
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-5)
    mitad = ventana // 2
    t0 = time.time()

    def muestrear(n, tramo):
        lo, hi = (mitad, corte - mitad) if tramo == "train" else (corte + mitad, T - mitad)
        c = torch.randint(lo, hi, (n,), device=dev)
        idx = c[:, None] + torch.arange(-mitad, mitad + 1, device=dev)[None, :]
        return m_t[idx].unsqueeze(1), a_t[c], c

    for paso in range(pasos):
        modelo.train()
        x, y, _ = muestrear(lote, "train")
        opt.zero_grad(set_to_none=True)
        perdida = torch.nn.functional.l1_loss(modelo(x), y)
        perdida.backward(); opt.step()
        if cb and paso % 200 == 0:
            cb(paso / pasos * 100, f"NILM paso {paso}/{pasos} · L1 {float(perdida.detach()):.4f}")

    modelo.eval()
    reales, predichos = [], []
    with torch.no_grad():
        for _ in range(20):
            x, y, _ = muestrear(512, "test")
            reales.append((y.cpu().numpy() * asd + amu))
            predichos.append((modelo(x).cpu().numpy() * asd + amu))
    R, P = np.concatenate(reales), np.concatenate(predichos)
    # base: cada circuito en su media del tramo de entrenamiento
    B = np.broadcast_to(amu, R.shape)
    por_aparato = {meta["aparatos"][j]: metricas(R[:, j], P[:, j], B[:, j])
                   for j in range(A)}
    total = metricas(R, P, B)
    total.update({"dataset": key, "aparatos": A, "ventana": ventana,
                  "pasos": pasos, "segundos": round(time.time() - t0, 1)})
    if logger:
        logger.info("nilm", **{k: v for k, v in total.items() if k != "n"})
    return {"total": total, "por_aparato": por_aparato}


# =====================  desperdicio  =========================================
def detectar_desperdicio(tarea: TareaPrevision, modelo, umbral_sigma: float = 2.0,
                         n_lotes: int = 40) -> dict[str, Any]:
    """Consumo por encima del esperado dadas hora, dia y clima.

    Se mira el residuo (real - previsto) y se marcan los tramos donde es
    POSITIVO y SOSTENIDO: un pico aislado es ruido de medida, pero varias horas
    seguidas consumiendo de mas es un equipo encendido que no deberia estarlo o una
    consigna mal puesta. Solo cuenta el exceso: consumir de menos no es un problema.
    """
    import torch

    modelo.eval()
    res = []
    with torch.no_grad():
        for _ in range(n_lotes):
            ctx, cov, fut, _, s = tarea.lote(512, "test")
            p = modelo(ctx, cov).float()
            r = (tarea.desnormalizar(fut, s) - tarea.desnormalizar(p, s)).cpu().numpy()
            res.append(r)
    R = np.concatenate(res)                                    # (N, H) en kWh
    sigma = float(R.std()) or 1e-9
    exceso = R > umbral_sigma * sigma
    # sostenido = al menos 3 horas consecutivas de exceso dentro del horizonte
    sostenido = np.zeros(len(R), dtype=bool)
    for k in range(R.shape[1] - 2):
        sostenido |= exceso[:, k] & exceso[:, k + 1] & exceso[:, k + 2]
    energia_exceso = float(R[R > 0].sum())
    return {"sigma_residuo_kwh": round(sigma, 3),
            "umbral_kwh": round(umbral_sigma * sigma, 3),
            "ventanas_analizadas": int(len(R)),
            "ventanas_con_exceso_sostenido": int(sostenido.sum()),
            "pct_ventanas_con_exceso": round(100 * float(sostenido.mean()), 2),
            "energia_por_encima_de_lo_esperado_kwh": round(energia_exceso, 1),
            "nota": "exceso sostenido = 3+ horas seguidas por encima del umbral; "
                    "un pico aislado es ruido de medida, no desperdicio"}


# =====================  linea base contrafactual (ahorro)  ===================
def medir_ahorro(tarea: TareaPrevision, modelo, n_lotes: int = 40) -> dict[str, Any]:
    """Protocolo IPMVP Opcion C: cuanto se habria consumido sin la intervencion.

    El modelo se entrena SOLO con el periodo anterior y predice el posterior. El
    ahorro es (predicho - medido). Aqui no ha habido intervencion, asi que lo
    correcto es que el ahorro salga proximo a cero: **este numero es la prueba de
    que la linea base no inventa ahorros**. Si con datos sin intervencion el
    metodo ya "detecta" un 5 % de ahorro, cualquier ahorro que reporte despues es
    ese sesgo, no una mejora.
    """
    import torch

    modelo.eval()
    medido, esperado = [], []
    with torch.no_grad():
        for _ in range(n_lotes):
            ctx, cov, fut, _, s = tarea.lote(512, "test")
            p = modelo(ctx, cov).float()
            medido.append(tarea.desnormalizar(fut, s).cpu().numpy())
            esperado.append(tarea.desnormalizar(p, s).cpu().numpy())
    M, E = np.concatenate(medido), np.concatenate(esperado)
    m = metricas(M, E)
    ahorro_pct = 100 * (E.sum() - M.sum()) / (E.sum() or 1e-9)
    return {"consumo_medido_kwh": round(float(M.sum()), 1),
            "consumo_esperado_kwh": round(float(E.sum()), 1),
            "ahorro_aparente_pct": round(float(ahorro_pct), 3),
            "cv_rmse_pct": m["cv_rmse_pct"], "nmbe_pct": m["nmbe_pct"],
            "cumple_ashrae_g14": bool(m["cv_rmse_pct"] < 25.0),
            "interpretacion": (
                "sin intervencion, el ahorro aparente deberia rondar 0 %. Lo que se "
                "aleje de 0 es el sesgo del metodo, y marca el ahorro minimo que hay "
                "que superar para que una mejora sea creible.")}


# =====================  orquestacion  ========================================
def correr_todo(cfg: dict[str, Any], base_dir: Path, logger, pv=None,
                dataset: str = "building_data_genome_2", pasos: int = 3000,
                nilm: bool = True, perdida: str = "huber") -> dict[str, Any]:
    """Las tres piezas de una tirada, con sus lineas base."""
    data_dir = Path(base_dir) / cfg["paths"]["data_dir"]
    art = Path(base_dir) / cfg["paths"]["artifacts_dir"]
    art.mkdir(parents=True, exist_ok=True)
    informe: dict[str, Any] = {}

    def cb(pct, msg):
        if pv:
            pv.update(phase_label=f"Consumo · {msg}", progress_pct=round(pct, 1),
                      log_tail=logger.tail(10))

    logger.info("consumo_start", dataset=dataset, pasos=pasos)
    tarea = TareaPrevision(data_dir, dataset)
    logger.info("consumo_datos", series=tarea.n_series, pasos_serie=tarea.T,
                covariables=tarea.n_cov, corte_temporal=tarea.corte)

    r = entrenar_previsor(tarea, pasos=pasos, perdida=perdida, logger=logger, cb=cb)
    informe["prevision"] = r["metricas"]
    informe["desperdicio"] = detectar_desperdicio(tarea, r["modelo"])
    informe["linea_base_ahorro"] = medir_ahorro(tarea, r["modelo"])

    if nilm:
        try:
            informe["nilm"] = entrenar_nilm(data_dir, logger=logger, cb=cb)
        except Exception as e:
            informe["nilm"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

    salida = art / f"consumo_{dataset}.json"
    salida.write_text(json.dumps(informe, ensure_ascii=False, indent=2), encoding="utf-8")
    informe["_fichero"] = str(salida)
    logger.info("consumo_done", fichero=str(salida))
    return informe
