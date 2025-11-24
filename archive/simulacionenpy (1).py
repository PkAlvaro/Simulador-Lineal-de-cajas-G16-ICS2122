# %% [markdown]
# # Simulador caso base lineal de cajas

# %%
# Librerías
import os, csv, datetime, pickle, ast
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Tuple, Optional, Any, List
import numpy as np
import simpy
import math
import json
import pandas as pd
from scipy import stats
from itertools import product

#para que lea los display del jupyter
try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)


# %%
# Constantes globales
OPEN_S  = 8 * 3600
CLOSE_S = 22 * 3600
BIN_SEC = 1

# %%
# Clases base
class CustomerProfile(Enum):
    DEAL_HUNTER = "deal_hunter"
    FAMILY_CART = "family_cart"
    WEEKLY_PLANNER = "weekly_planner"
    SELF_CHECKOUT_FAN = "self_checkout_fan"
    REGULAR = "regular"
    EXPRESS_BASKET = "express_basket"

class PaymentMethod(Enum):
    CARD = "card"
    CASH = "cash"

class PriorityType(Enum):
    NO_PRIORITY = "no_priority"
    SENIOR = "senior"
    PREGNANT = "pregnant"
    REDUCED_MOBILITY = "reduced_mobility"

class DayType(Enum):
    TYPE_1 = "tipo_1"
    TYPE_2 = "tipo_2"
    TYPE_3 = "tipo_3"

class LaneType(Enum):
    REGULAR="regular"
    EXPRESS="express"
    PRIORITY="priority"
    SCO="sco"

# %%
MAX_TOTAL_LANES = 40
EXPRESS_BLOCK = None  # express lanes are free, kept for clarity
SCO_BLOCK = 5
LANE_ORDER = ["regular", "express", "priority", "self_checkout"]
LANE_PREFIXES = {
    LaneType.REGULAR: "REG",
    LaneType.EXPRESS: "EXP",
    LaneType.PRIORITY: "PRI",
    LaneType.SCO: "SCO",
}
DEFAULT_LANE_COUNTS = {
    DayType.TYPE_1: {"regular": 10, "express": 3, "priority": 2, "self_checkout": 5},
    DayType.TYPE_2: {"regular": 10, "express": 3, "priority": 2, "self_checkout": 5},
    DayType.TYPE_3: {"regular": 15, "express": 3, "priority": 2, "self_checkout": 5},
}

def enforce_lane_constraints(raw_counts: dict[str, int]) -> dict[str, int]:
    counts = {name: max(0, int(raw_counts.get(name, 0))) for name in LANE_ORDER}

    sco = counts["self_checkout"] - counts["self_checkout"] % SCO_BLOCK
    max_sco = MAX_TOTAL_LANES - (MAX_TOTAL_LANES % SCO_BLOCK)
    counts["self_checkout"] = min(max_sco, max(0, sco))

    remaining = MAX_TOTAL_LANES - counts["self_checkout"]
    other_keys = [name for name in LANE_ORDER if name != "self_checkout"]
    total_other = sum(counts[name] for name in other_keys)
    if total_other > remaining:
        if remaining <= 0:
            for name in other_keys:
                counts[name] = 0
        else:
            scale = remaining / total_other
            scaled = {name: int(scale * counts[name]) for name in other_keys}
            current = sum(scaled.values())
            idx = 0
            while current < remaining and other_keys:
                name = other_keys[idx % len(other_keys)]
                if scaled[name] < counts[name]:
                    scaled[name] += 1
                    current += 1
                idx += 1
            for name in other_keys:
                counts[name] = scaled[name]
    return counts

def build_uniform_policy(counts: dict[str, int]) -> dict[DayType, dict[LaneType, int]]:
    normalized = enforce_lane_constraints(counts)
    return {
        day_type: {
            LaneType.REGULAR: normalized["regular"],
            LaneType.EXPRESS: normalized["express"],
            LaneType.PRIORITY: normalized["priority"],
            LaneType.SCO: normalized["self_checkout"],
        }
        for day_type in DayType
    }

def update_current_lane_policy(counts: dict[str, int]) -> dict[str, int]:
    normalized = enforce_lane_constraints(counts)
    policy = build_uniform_policy(normalized)
    CURRENT_LANE_POLICY.update(policy)
    return normalized

CURRENT_LANE_POLICY: dict[DayType, dict[LaneType, int]] = build_uniform_policy(DEFAULT_LANE_COUNTS[DayType.TYPE_1])
CURRENT_LANE_POLICY.update({
    DayType.TYPE_2: CURRENT_LANE_POLICY[DayType.TYPE_1].copy(),
    DayType.TYPE_3: build_uniform_policy(DEFAULT_LANE_COUNTS[DayType.TYPE_3])[DayType.TYPE_3],
})


# %%
class _NullCallable:
    def __call__(self, *a, **k): return None
NULL = _NullCallable()

class _CompatUnpickler(pickle.Unpickler):
    _MAP = {"__builtin__": "builtins", "copy_reg": "copyreg", "cPickle": "pickle"}
    def find_class(self, module, name):
        module = self._MAP.get(module, module)
        if module == "__main__" or name == "__main__":
            return NULL
        try:
            return super().find_class(module, name)
        except Exception:
            return NULL

def _safe_load_dict(p: Path):
    data = np.load(p, allow_pickle=False)
    keys = data["keys"]
    lambda_matrix = data["lambdas"]
    bin_left = data["bin_left_s"]
    if bin_left.ndim != 1:
        raise ValueError(f"bin_left_s inválido en {p}")
    clean = {}
    for idx, key_str in enumerate(keys):
        parts = str(key_str).split("|")
        if len(parts) < 3:
            continue
        try:
            day_type = _norm_enum(parts[0], DayType)
            priority = _norm_enum(parts[1], PriorityType)
            payment = _norm_enum(parts[2], PaymentMethod)
        except Exception:
            continue
        lam = np.maximum(np.asarray(lambda_matrix[idx], dtype=float), 0.0)
        clean[(day_type, priority, payment)] = (np.asarray(bin_left, dtype=float), lam)
    return clean


# %% [markdown]
# Cargando tasas de llegada desde archivos npz generados por tools/rebuild_arrivals.py.

# %%
_ARRIVALS_NPZ_BY_PROFILE = {
    CustomerProfile.DEAL_HUNTER:       Path("arrivals_npz/lambda_deal_hunter.npz"),
    CustomerProfile.FAMILY_CART:       Path("arrivals_npz/lambda_family_cart.npz"),
    CustomerProfile.WEEKLY_PLANNER:    Path("arrivals_npz/lambda_weekly_planner.npz"),
    CustomerProfile.SELF_CHECKOUT_FAN: Path("arrivals_npz/lambda_self_checkout_fan.npz"),
    CustomerProfile.REGULAR:           Path("arrivals_npz/lambda_regular.npz"),
    CustomerProfile.EXPRESS_BASKET:    Path("arrivals_npz/lambda_express_basket.npz"),
}

_DAYTYPE_MAP = {DayType.TYPE_1:"Tipo 1", DayType.TYPE_2:"Tipo 2", DayType.TYPE_3:"Tipo 3"}
_PRIORITY_MAP = {
    PriorityType.NO_PRIORITY:"no_priority",
    PriorityType.SENIOR:"senior",
    PriorityType.PREGNANT:"pregnant",
    PriorityType.REDUCED_MOBILITY:"reduced_mobility",
}
_PAYMENT_MAP = {PaymentMethod.CARD:"card", PaymentMethod.CASH:"cash"}

_SERIES_CACHE: Dict[CustomerProfile, Dict[Tuple[DayType,PriorityType,PaymentMethod], Tuple[np.ndarray,np.ndarray]]] = {}

# %%
def _normalize_series(raw: dict):
    series = {}
    JORNADA = CLOSE_S - OPEN_S  # 14*3600
    for dt in _DAYTYPE_MAP:
        for pr in _PRIORITY_MAP:
            for pm in _PAYMENT_MAP:
                k = (_DAYTYPE_MAP[dt], _PRIORITY_MAP[pr], _PAYMENT_MAP[pm])
                if k not in raw: 
                    continue
                t, lam = raw[k]
                t = np.asarray(t, float); lam = np.maximum(np.asarray(lam, float), 0.0)
                if t.ndim!=1 or lam.ndim!=1 or len(t)!=len(lam) or len(t)==0: 
                    continue
                o = np.argsort(t); t, lam = t[o], lam[o]
                # pad hasta cierre manteniendo último valor
                if t[-1] < JORNADA:
                    t = np.append(t, JORNADA)
                    lam = np.append(lam, lam[-1])
                series[(dt, pr, pm)] = (t, lam)
    return series

def _get_series_for_profile(profile: CustomerProfile):
    if profile in _SERIES_CACHE:
        return _SERIES_CACHE[profile]
    path = _ARRIVALS_NPZ_BY_PROFILE.get(profile)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Archivo de llegadas no encontrado para {profile.value}: {path}")
    raw = _safe_load_dict(path)
    series = _normalize_series(raw)
    _SERIES_CACHE[profile] = series
    return series

def _lambda_step_at(seconds: float, times: np.ndarray, lambdas: np.ndarray) -> float:
    if seconds <= times[0]:  return float(lambdas[0])
    if seconds >= times[-1]: return float(lambdas[-1])  # ← antes devolvías 0.0
    i = int(np.searchsorted(times, seconds, side="right") - 1)
    return float(lambdas[i])


# %% [markdown]
# ## Tiempos de llegada de clientes

# %%
@dataclass
class ArrivalDistribution:
    profile: CustomerProfile
    day_type: DayType
    priority: PriorityType
    payment_method: PaymentMethod
    total_customers: int = 0
    distribution_type: str = "poisson_non_homogeneous"
    def get_arrival_rate(self, seconds: float) -> float:
        series = _get_series_for_profile(self.profile)
        pair = series.get((self.day_type, self.priority, self.payment_method))
        if not pair: 
            return 0.0
        times, lambdas = pair
        s = float(max(0.0, min(seconds, times[-1])))
        lambda_por_minuto = _lambda_step_at(s, times, lambdas)
        return lambda_por_minuto / 60.0

# %% [markdown]
# ## Mapeo de tipos de día

# %%
_SEG_DAY_MAP = {"tipo 1": DayType.TYPE_1, "tipo 2": DayType.TYPE_2, "tipo 3": DayType.TYPE_3}
_SEG_PAY_MAP = {"card": PaymentMethod.CARD, "cash": PaymentMethod.CASH}

def _norm_enum(v, enum_cls):
    if isinstance(v, enum_cls): 
        return v
    if isinstance(v, str):
        s = v.strip()
        for e in enum_cls:
            if s.lower() == str(getattr(e, "value", "")).lower():
                return e
        try:
            return enum_cls[s]
        except Exception:
            pass
    raise ValueError(f"No se pudo normalizar {v} a {enum_cls.__name__}")

def _parse_segment(seg_raw: str) -> Tuple[Optional[DayType], Optional[PaymentMethod]]:
    if not seg_raw or str(seg_raw).strip().lower() == "general":
        return None, None
    day: Optional[DayType] = None
    pay: Optional[PaymentMethod] = None
    parts = [p.strip().lower() for p in str(seg_raw).split("|")]
    for p in parts:
        if p in _SEG_DAY_MAP:
            day = _SEG_DAY_MAP[p]
        elif p in _SEG_PAY_MAP:
            pay = _SEG_PAY_MAP[p]
    return day, pay

def _row_to_rule(row: Dict[str, Any]) -> Dict[str, Any]:
    def fget(*names, default=None, cast=float):
        for n in names:
            if n in row and str(row[n]).strip() != "":
                try:
                    return cast(row[n])
                except Exception:
                    pass
        return default
    nb_n = fget("nb_n","n")
    nb_p = fget("nb_p","p")
    if nb_n is not None and nb_p is not None:
        return {"method":"parametric", "model":"negbinom", "params": (float(nb_n), float(nb_p))}
    mu = fget("mean","mu")
    sd = fget("std","sigma","sd")
    if mu is not None and sd is not None:
        return {"method": "kde", "mean": float(mu), "std": float(sd)}
    mode_v = fget("mode","median","items_mean")
    if mode_v is not None:
        return {"method":"fixed", "value": float(mode_v)}
    raise ValueError("Fila sin parámetros reconocibles")

# %% [markdown]
# ## Volumen de compra

# %%
ITEMS_DISTRIBUTION_FILE = Path("items_distribution_summary.csv")
RNG_ITEMS = np.random.default_rng(123)

def _sample_poisson(lam, rng): return int(rng.poisson(float(max(lam, 1e-6))))
def _sample_nbinom(r, p, rng): return int(rng.negative_binomial(float(max(r, 1e-6)), float(np.clip(p, 1e-6, 1 - 1e-6))))


def _coerce_profile_value(value) -> Optional[CustomerProfile]:
    if isinstance(value, CustomerProfile):
        return value
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    if not text or text.upper() == "ALL":
        return None
    try:
        return _norm_enum(text, CustomerProfile)
    except Exception:
        return None


def _coerce_optional_enum(value, enum_cls, allow_all: bool = True):
    if isinstance(value, enum_cls):
        return value
    if value is None:
        return None
    text = str(getattr(value, "value", value)).strip()
    if not text:
        return None
    if allow_all and text.upper() == "ALL":
        return None
    try:
        return _norm_enum(text, enum_cls)
    except Exception:
        return None


def _parse_literal_list(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        data = ast.literal_eval(text)
    except Exception:
        return None
    return data if isinstance(data, list) else None


def _parse_fit_params(raw: str) -> dict:
    if raw is None:
        return {}
    text = str(raw).strip()
    if not text:
        return {}
    candidates = [text, text.replace('""', '"')]
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            try:
                parsed = ast.literal_eval(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
    return {}


@dataclass
class ItemsDistributionModel:
    csv_path: Path
    fallback_mean: float = 12.0
    _records: Dict[Tuple[CustomerProfile, Optional[PriorityType], Optional[PaymentMethod], Optional[DayType]], Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        self._records = self._load_records(Path(self.csv_path))

    def _load_records(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"No existe {path}")
        records: Dict[Tuple[CustomerProfile, Optional[PriorityType], Optional[PaymentMethod], Optional[DayType]], Dict[str, Any]] = {}
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                prof = _coerce_profile_value(row.get("profile"))
                if prof is None:
                    continue
                pr = _coerce_optional_enum(row.get("priority"), PriorityType, allow_all=True)
                pay = _coerce_optional_enum(row.get("payment_method"), PaymentMethod, allow_all=True)
                day = _coerce_optional_enum(row.get("day_type"), DayType, allow_all=True)
                fit_type = (row.get("fit_type") or "").strip().lower()

                if fit_type == "parametric":
                    dist_name = (row.get("fit_distribution") or "").strip().lower()
                    params = _parse_fit_params(row.get("fit_params", ""))
                    if not dist_name:
                        continue
                    spec = {
                        "type": "parametric",
                        "distribution": dist_name,
                        "params": params,
                    }
                elif fit_type == "kde":
                    support = _parse_literal_list(row.get("kde_support"))
                    probs = _parse_literal_list(row.get("kde_probs"))
                    if not support or not probs or len(support) != len(probs):
                        continue
                    support_arr = np.asarray(support, dtype=float)
                    probs_arr = np.asarray(probs, dtype=float)
                    probs_arr = np.clip(probs_arr, 0.0, None)
                    total = probs_arr.sum()
                    if not np.isfinite(total) or total <= 0:
                        continue
                    probs_arr = probs_arr / total
                    spec = {
                        "type": "kde",
                        "support": support_arr,
                        "probs": probs_arr,
                    }
                else:
                    continue

                key = (prof, pr, pay, day)
                records[key] = spec
        return records

    def _candidate_keys(self, profile, priority, payment, day_type):
        dims = [priority, payment, day_type]
        for mask in product((True, False), repeat=3):
            yield (profile,) + tuple(dim if keep else None for dim, keep in zip(dims, mask))
        yield (profile, None, None, None)

    def _match_record(self, profile, priority, payment, day_type):
        for key in self._candidate_keys(profile, priority, payment, day_type):
            if key in self._records:
                return self._records[key]
        return None

    def sample(self,
               profile,
               priority: Optional[PriorityType] = None,
               payment: Optional[PaymentMethod] = None,
               day_type: Optional[DayType] = None,
               rng: Optional[np.random.Generator] = None) -> int:
        rng = rng or np.random.default_rng()
        record = self._match_record(profile, priority, payment, day_type)
        if not record:
            return self._fallback(rng)
        if record["type"] == "parametric":
            return self._sample_parametric(record, rng)
        if record["type"] == "kde":
            return self._sample_kde(record, rng)
        return self._fallback(rng)

    def _sample_parametric(self, record, rng):
        dist = record.get("distribution", "")
        params = record.get("params", {}) or {}
        if dist == "nbinom":
            n = float(params.get("n", params.get("r", 1.0)))
            p = float(params.get("p", params.get("prob", 0.5)))
            return max(1, _sample_nbinom(n, p, rng))
        if dist in {"poisson", "pois"}:
            lam = float(params.get("lambda", params.get("lam", params.get("mean", self.fallback_mean))))
            return max(1, _sample_poisson(lam, rng))
        return self._fallback(rng)

    def _sample_kde(self, record, rng):
        support = record.get("support")
        probs = record.get("probs")
        if support is None or probs is None or len(support) == 0:
            return self._fallback(rng)
        choice = rng.choice(support, p=probs)
        return int(max(1, round(choice)))

    def _fallback(self, rng):
        return max(1, _sample_poisson(self.fallback_mean, rng))


try:
    ITEMS_DISTRIBUTION_MODEL = ItemsDistributionModel(ITEMS_DISTRIBUTION_FILE)
except Exception:
    ITEMS_DISTRIBUTION_MODEL = None


# %%
def draw_items(
    profile_str: str,
    rng=RNG_ITEMS,
    models=None,
    max_resample: int = 50,
    *,
    priority: Optional[PriorityType] = None,
    payment_method: Optional[PaymentMethod] = None,
    day_type: Optional[DayType] = None,
    model: Optional[ItemsDistributionModel] = ITEMS_DISTRIBUTION_MODEL,
) -> int:
    del models, max_resample  # compatibilidad con firma anterior
    profile = _coerce_profile_value(profile_str)
    pr = _coerce_optional_enum(priority, PriorityType, allow_all=False)
    pay = _coerce_optional_enum(payment_method, PaymentMethod, allow_all=False)
    day = _coerce_optional_enum(day_type, DayType, allow_all=False)
    if model is not None and profile is not None:
        try:
            sample = model.sample(profile, pr, pay, day, rng=rng)
            if isinstance(sample, (int, float)) and np.isfinite(sample) and sample > 0:
                return int(max(1, round(sample)))
        except Exception:
            pass
    return max(1, _sample_poisson(10.0, rng))

# %% [markdown]
# ## Paciencia de los clientes

# %%
import re
class _PatienceRuleStore:
    def __init__(self):
        self.rules: Dict[Tuple[Optional[CustomerProfile], PriorityType], Dict[str,Any]] = {}
    def add(self, prof: Optional[CustomerProfile], prio: PriorityType, spec: Dict[str,Any]):
        self.rules[(prof, prio)] = spec
    def find(self, prof: CustomerProfile, prio: PriorityType) -> Optional[Dict[str,Any]]:
        return self.rules.get((prof, prio)) or self.rules.get((None, prio))

def _fget(row, *names, default=None, cast=float):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            try: return cast(row[n])
            except: pass
    return default

def _parse_params_str(s: str) -> dict:
    if s is None: return {}
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    # key=value
    pairs = re.findall(r'([a-zA-Z_áéíóúñ]+)\s*=?\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)', s)
    if pairs:
        out = {}
        for k,v in pairs:
            k = k.lower().replace("á","a").replace("é","e").replace("í","i").replace("ó","o").replace("ú","u")
            out[k] = float(v)
        # alias
        if "shape" in out and "k" not in out: out["k"] = out["shape"]
        if "scale" in out and "theta" not in out: out["theta"] = out["scale"]
        return out
    # tuple-only
    nums = re.findall(r'([-+]?\d*\.?\d+(?:e[-+]?\d+)?)', s)
    if nums:
        return {"tuple": [float(x) for x in nums]}
    return {}

def _row_to_spec(row: Dict[str,Any]) -> Dict[str,Any]:
    # método desde múltiples nombres de columna
    m = (row.get("method") or row.get("dist") or row.get("model") or
         row.get("mejor_dist") or row.get("distribucion") or
         row.get("distribucion_recomendada") or row.get("distribution") or "")
    m = str(m).strip().lower()

    # parámetros desde múltiples nombres de columna
    p_raw = (row.get("parameters") or row.get("parametros") or row.get("parámetros") or row.get("parametros_aprox") or "")
    p = _parse_params_str(p_raw)

    # valores numéricos directos
    mean = _fget(row,"mean","avg")
    std  = _fget(row,"std","sigma","sd")
    lam  = _fget(row,"lambda","lam","rate")
    shape= _fget(row,"shape","alpha","k")
    scale= _fget(row,"scale","theta","beta")
    loc  = _fget(row,"loc", default=0.0)

    # normalizar alias de método
    if m in ("expon","exponencial","exponential","exp"): m = "exponential"
    if m in ("weibull_min","weibull","weibullmin"): m = "weibull_min"
    if m in ("ga","gamma"): m = "gamma"
    if m in ("ln","lognorm","lognormal"): m = "lognormal"
    if m in ("norm","normal","gaussian"): m = "normal"

    # completar parámetros faltantes desde p
    if "loc" in p: loc = float(p["loc"])
    if m == "weibull_min":
        k = float(p.get("k", shape or (p.get("tuple",[None,None])[0])))
        sc= float(p.get("scale", scale or (p.get("tuple",[None,None])[1])))
        if not (k and sc):  # inválido → fallback
            return {"method":"fixed","value":300.0}
        return {"method":"weibull_min","k":max(1e-9,k),"scale":max(1e-9,sc),"loc":float(loc)}

    if m == "exponential" or lam is not None or (mean and not std):
        # expon: lambda o scale; si viene tupla, (loc, scale) o (scale)
        if "tuple" in p:
            if len(p["tuple"])==2: loc = p["tuple"][0]; scale = p["tuple"][1]
            elif len(p["tuple"])==1: scale = p["tuple"][0]
        if lam: 
            scale = 1.0/max(1e-9,lam)
        elif scale is None:
            scale = float(mean) if mean else 600.0
        return {"method":"exponential","scale":float(scale),"loc":float(loc)}

    if m == "gamma" or (shape and scale) or ("tuple" in p and len(p["tuple"])>=2):
        if "tuple" in p and (shape is None or scale is None):
            shape = shape or p["tuple"][0]
            scale = scale or p["tuple"][1]
        return {"method":"gamma","shape":float(shape),"scale":float(scale),"loc":float(loc)}

    if m == "lognormal":
        mu = _fget(row,"mu","lognorm_mu","mu_log")
        sg = _fget(row,"sigma","lognorm_sigma","sigma_log")
        if "tuple" in p and (mu is None or sg is None):
            # si solo viene sigma, no hay mu → imposible estimar: fallback
            if len(p["tuple"])==2:
                mu, sg = p["tuple"][0], p["tuple"][1]
        if mu is None or sg is None:
            return {"method":"fixed","value":300.0}
        return {"method":"lognormal","mu":float(mu),"sigma":float(max(1e-9,sg)),"loc":float(loc)}

    if m == "normal" or (mean is not None and std is not None):
        return {"method":"normal","mean":float(mean),"std":float(max(1e-6,std)),"loc":float(loc)}

    # último recurso: valor fijo
    val = _fget(row,"fixed","value","median","mode", default=300.0)
    return {"method":"fixed","value":float(val)}

def _load_patience_rules_from_files(files_by_profile: Dict[Optional[CustomerProfile], str]) -> _PatienceRuleStore:
    store = _PatienceRuleStore()
    for default_profile, path in files_by_profile.items():
        with open(path,"r",encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    prio = _norm_enum(row.get("priority"), PriorityType)
                    prof_col = row.get("profile", "").strip()
                    prof = _norm_enum(prof_col, CustomerProfile) if prof_col else default_profile
                    spec = _row_to_spec(row)
                    store.add(prof, prio, spec)
                except Exception:
                    continue
    return store

# %%
@dataclass
class PatienceDistributionCSV:
    files_by_profile: Dict[Optional[CustomerProfile], str]
    _store: _PatienceRuleStore = None
    def __post_init__(self):
        self._store = _load_patience_rules_from_files(self.files_by_profile)
    def sample(self, profile: CustomerProfile, priority: PriorityType) -> float:
        spec = self._store.find(profile, priority)
        if not spec:
            return float(max(0, np.random.exponential(600.0)))
        m = spec["method"]
        if m == "exponential":
            val = np.random.exponential(max(1e-9, spec.get("scale", 600.0))) + spec.get("loc",0.0)
        elif m == "gamma":
            val = np.random.gamma(max(1e-9,spec["shape"]), max(1e-9,spec["scale"])) + spec.get("loc",0.0)
        elif m == "lognormal":
            val = np.random.lognormal(spec["mu"], max(1e-9,spec["sigma"])) + spec.get("loc",0.0)
        elif m == "normal":
            val = np.random.normal(spec["mean"], max(1e-6,spec["std"])) + spec.get("loc",0.0)
        elif m == "weibull_min":
            # numpy: np.random.weibull(k) * scale  (loc se suma aparte)
            val = (np.random.weibull(max(1e-9, spec["k"])) * max(1e-9, spec["scale"])) + spec.get("loc",0.0)
        else:
            val = float(spec.get("value", 300.0))
        return float(max(0, val))

# %%
@dataclass
class ProfileConfig:
    profile: CustomerProfile

    # Se elimina items_distribution: ahora usamos draw_items(profile_str)

    patience_distribution: PatienceDistributionCSV = field(
        default_factory=lambda: PatienceDistributionCSV({
            CustomerProfile.REGULAR:        "patience/distribuciones_paciencia_regular.csv",
            CustomerProfile.EXPRESS_BASKET: "patience/distribuciones_prioridad_express_basket.csv",
            CustomerProfile.WEEKLY_PLANNER: "patience/distribuciones_prioridad_weekly_planner.csv",
            CustomerProfile.FAMILY_CART:    "patience/distribuciones_paciencia_family_cart.csv",
            CustomerProfile.SELF_CHECKOUT_FAN:  "patience/paciencia_self_checkout_fan.csv",
            CustomerProfile.DEAL_HUNTER:    "patience/deal_hunter_patience_fit.csv"
        })
    )

    items_sampler: callable = field(default=lambda prof_str, **kwargs: draw_items(prof_str, **kwargs))
    
    def create_arrival_distribution(self, day_type: DayType, priority: PriorityType,
                                    payment_method: PaymentMethod, total_customers: int = 0) -> ArrivalDistribution:
        return ArrivalDistribution(self.profile, day_type, priority, payment_method, total_customers)



# %% [markdown]
# ## Tiempo de servicio por cliente

# %%
class CheckoutLane:
    def __init__(self, env, lane_id, lane_type: LaneType):
        self.env = env
        self.lane_id = lane_id
        self.lane_type = lane_type
        self.servidor = simpy.Resource(env, capacity=1)

def elegible(cliente, lane: CheckoutLane) -> bool:
    pr = cliente['priority']
    pm = cliente['payment_method']
    items = int(cliente['items'])
    if lane.lane_type==LaneType.EXPRESS and items>10: return False
    if lane.lane_type==LaneType.PRIORITY and pr==PriorityType.NO_PRIORITY: return False
    if lane.lane_type==LaneType.SCO and (pm!=PaymentMethod.CARD or pr==PriorityType.REDUCED_MOBILITY or items>15): return False
    return True

# %%
_DIST = {
    "normal": stats.norm,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min
}

def _coerce_bool(x):
    if isinstance(x, bool): return x
    if x is None: return None
    s = str(x).strip().lower()
    return True if s in {"true","1","yes"} else False if s in {"false","0","no"} else None

def _pick_best(df: pd.DataFrame) -> pd.DataFrame:
    # Elige por mayor p-valor KS, luego menor KS. Ignora filas con error o sin params.
    work = df.copy()
    if "reject_H0" in work.columns:
        work["reject_H0"] = work["reject_H0"].map(_coerce_bool)
    for c in ["ks_pvalue","ks_stat","shape","loc","scale"]:
        if c in work.columns: work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[work["distribution"].isin(_DIST.keys())]
    work = work[work["loc"].notna() & work["scale"].notna()]
    # preferir no rechazadas si existen; si no, considerar todas
    subset = work
    if "reject_H0" in work.columns and work["reject_H0"].eq(False).any():
        subset = work[work["reject_H0"].eq(False)]
    subset = subset.sort_values(["ks_pvalue","ks_stat"], ascending=[False,True])
    return subset.head(1)

@dataclass
class ServiceTimeCSV:
    table: pd.DataFrame  # filas “ganadoras” ya agregadas

    @classmethod
    def from_file(cls, path: str | Path) -> "ServiceTimeCSV":
        df = pd.read_csv(path)
        # Si ya es “bestfit” una fila por clave, úsalo directo; si no, elegir mejor por grupo
        key_cols_lane = [c for c in ["profile","lane_type"] if c in df.columns]
        key_cols = key_cols_lane if key_cols_lane else ["profile"]
        best_rows = []
        for key, g in df.groupby(key_cols):
            pick = _pick_best(g)
            if len(pick):
                best = pick.iloc[0].copy()
                if not isinstance(key, tuple): key = (key,)
                for kname, kval in zip(key_cols, key):
                    best[kname] = kval
                best_rows.append(best)
        if not best_rows:
            raise ValueError("No se pudo seleccionar ninguna distribución válida del CSV.")
        best_df = pd.DataFrame(best_rows)
        # Normalizar tipos
        for c in ["shape","loc","scale"]:
            best_df[c] = pd.to_numeric(best_df[c], errors="coerce")
        best_df["distribution"] = best_df["distribution"].str.strip().str.lower()
        return cls(best_df)

    def _row_for(self, profile, lane_type=None) -> Optional[pd.Series]:
        # Usar clave más específica si existe lane_type
        if "lane_type" in self.table.columns and lane_type is not None:
            m = self.table[(self.table["profile"]==getattr(profile,"value",profile)) &
                           (self.table["lane_type"]==getattr(lane_type,"value",lane_type))]
            if len(m): return m.iloc[0]
        m = self.table[self.table["profile"]==getattr(profile,"value",profile)]
        return m.iloc[0] if len(m) else None

    def sample(self, *, profile, items: int, lane_type=None, **_ignored) -> float:
        row = self._row_for(profile, lane_type)
        if row is None:  # fallback simple por ítems
            return float(max(0, 15.0 + 4.0*int(items)))
        dist_name = row["distribution"]
        dist = _DIST.get(dist_name)
        if dist is None:
            return float(max(0, 15.0 + 4.0*int(items)))
        shape = row.get("shape")
        loc = float(row.get("loc", 0.0))
        scale = float(row.get("scale", 1.0))
        args: Tuple[Any, ...] = ()
        if not (pd.isna(shape) or dist_name=="normal"):  # normal no usa shape
            args = (float(shape),)
        x = dist.rvs(*args, loc=loc, scale=scale)
        if not np.isfinite(x) or x <= 0:
            x = max(0, 15.0 + 4.0*int(items))
        return float(max(0, x))

# %%
def _norm_day_type_value(value) -> DayType:
    if isinstance(value, DayType):
        return value
    if value is None:
        raise ValueError("DayType vacío")
    s = str(getattr(value, "value", value)).strip()
    if not s:
        raise ValueError("DayType vacío")
    s_lower = s.lower().replace("-", " ").replace("_", " ")
    if s_lower in _SEG_DAY_MAP:
        return _SEG_DAY_MAP[s_lower]
    for dt in DayType:
        val = str(getattr(dt, "value", "")).lower()
        if s_lower == val or s_lower == val.replace("_", " "):
            return dt
        if s_lower == dt.name.lower() or s_lower == dt.name.lower().replace("_", " "):
            return dt
    raise ValueError(f"No se pudo normalizar {value} a DayType")


def _norm_profile_value(value) -> CustomerProfile:
    if isinstance(value, CustomerProfile):
        return value
    if value is None:
        raise ValueError("Profile vacío")
    s = str(getattr(value, "value", value)).strip()
    if not s:
        raise ValueError("Profile vacío")
    slug = s.lower().replace("-", "_").replace(" ", "_")
    for prof in CustomerProfile:
        if slug == prof.value.lower() or slug == prof.name.lower():
            return prof
    raise ValueError(f"No se pudo normalizar {value} a CustomerProfile")


def _float_or_default(value, default=None):
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        num = float(value)
        if np.isnan(num):
            return default
        return num
    except Exception:
        return default


@dataclass
class ServiceTimeSegment:
    profile: CustomerProfile
    day_type: DayType
    lane_type: LaneType
    priority: PriorityType
    payment_method: PaymentMethod
    intercept: float
    slope: float
    residual_spec: Dict[str, Any]
    tail_spec: Dict[str, Any]


class ServiceTimeRepository:
    def __init__(self, base_dir: Path):
        self._segments: Dict[Tuple, ServiceTimeSegment] = {}
        self._load_segments(Path(base_dir))

    def _load_segments(self, base_dir: Path):
        base_dir = base_dir.expanduser()
        inliers_path = base_dir / "service_time_inliers_summary.csv"
        if not inliers_path.exists():
            raise FileNotFoundError(f"No existe {inliers_path}")
        inliers_df = pd.read_csv(inliers_path)
        outliers_map = self._load_outliers(base_dir / "service_time_outliers_summary.csv")

        for _, row in inliers_df.iterrows():
            try:
                prof = _norm_profile_value(row.get("profile"))
                dt = _norm_day_type_value(row.get("day_type"))
                lt = _norm_enum(row.get("lane_type"), LaneType)
                pr = _norm_enum(row.get("priority"), PriorityType)
                pm = _norm_enum(row.get("payment_method"), PaymentMethod)
            except Exception:
                continue

            intercept = _float_or_default(row.get("alpha"), 20.0)
            slope = _float_or_default(row.get("beta"), 0.0)

            residual_spec = {
                "distribution": str(row.get("residual_distribution", "normal")).strip().lower(),
                "df": _float_or_default(row.get("df")),
                "loc": _float_or_default(row.get("loc"), 0.0),
                "scale": _float_or_default(row.get("scale"), 30.0),
            }
            if not residual_spec["scale"]:
                residual_spec["scale"] = 30.0

            tail_row = outliers_map.get((prof, dt, lt, pr, pm))
            tail_spec = self._build_tail_spec(row, tail_row)

            key = (prof, dt, lt, pr, pm)
            self._segments[key] = ServiceTimeSegment(
                profile=prof,
                day_type=dt,
                lane_type=lt,
                priority=pr,
                payment_method=pm,
                intercept=float(intercept),
                slope=float(slope),
                residual_spec=residual_spec,
                tail_spec=tail_spec,
            )

        if not self._segments:
            raise ValueError("No se pudieron cargar segmentos de servicio.")

    def _load_outliers(self, path: Path) -> Dict[Tuple, pd.Series]:
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        out: Dict[Tuple, pd.Series] = {}
        for _, row in df.iterrows():
            try:
                prof = _norm_profile_value(row.get("profile"))
                dt = _norm_day_type_value(row.get("day_type"))
                lt = _norm_enum(row.get("lane_type"), LaneType)
                pr = _norm_enum(row.get("priority"), PriorityType)
                pm = _norm_enum(row.get("payment_method"), PaymentMethod)
            except Exception:
                continue
            out[(prof, dt, lt, pr, pm)] = row
        return out

    def _build_tail_spec(self, inlier_row: pd.Series, tail_row: Optional[pd.Series]) -> Dict[str, Any]:
        inlier_count = _float_or_default(inlier_row.get("inlier_count"), None)
        if inlier_count is None:
            inlier_count = _float_or_default(inlier_row.get("n_obs"), 0.0)
        tail_count = _float_or_default(inlier_row.get("outlier_count"), 0.0)
        if tail_row is not None:
            tail_count = _float_or_default(tail_row.get("outlier_count"), tail_count)
        total = (inlier_count or 0.0) + (tail_count or 0.0)
        p_tail = 0.0
        if total and tail_count:
            p_tail = max(0.0, min(1.0, tail_count / total))

        spec = {
            "p_tail": float(p_tail),
            "distribution": "",
            "shape": None,
            "df": None,
            "loc": None,
            "scale": None,
        }
        if tail_row is not None:
            spec.update({
                "distribution": str(tail_row.get("residual_distribution", "")).strip().lower(),
                "shape": _float_or_default(tail_row.get("shape")),
                "df": _float_or_default(tail_row.get("df")),
                "loc": _float_or_default(tail_row.get("loc"), 0.0),
                "scale": _float_or_default(tail_row.get("scale")),
            })
        return spec

    def _find_segment(self, profile, day_type, lane_type, priority, payment_method):
        profile = _norm_profile_value(profile)
        day_type = _norm_day_type_value(day_type) if day_type else None
        lane = lane_type
        priority = priority
        payment = payment_method

        exact_key = (profile, day_type, lane, priority, payment)
        if None not in exact_key and exact_key in self._segments:
            return self._segments[exact_key]

        best_seg = None
        best_score = -1
        for seg in self._segments.values():
            if seg.profile != profile:
                continue
            score = 0
            if day_type and seg.day_type == day_type:
                score += 4
            if lane and seg.lane_type == lane:
                score += 3
            if priority and seg.priority == priority:
                score += 2
            if payment and seg.payment_method == payment:
                score += 1
            if score > best_score:
                best_seg = seg
                best_score = score

        if best_seg is not None:
            return best_seg

        for seg in self._segments.values():
            if seg.profile == profile:
                return seg
        return None

    def sample(
        self,
        *,
        profile,
        items: int,
        lane_type=None,
        day_type=None,
        priority=None,
        payment_method=None,
        **_,
    ) -> float:
        seg = self._find_segment(profile, day_type, lane_type, priority, payment_method)
        if seg is None:
            return float(max(0, 15.0 + 4.0 * int(items)))

        baseline = float(seg.intercept) + float(seg.slope) * int(items)
        tail_spec = seg.tail_spec or {}
        p_tail = float(tail_spec.get("p_tail", 0.0) or 0.0)
        if p_tail > 0 and np.random.random() < p_tail:
            residual = self._sample_from_spec(tail_spec, fallback_scale=seg.residual_spec.get("scale", 30.0))
            return float(max(10.0, baseline + residual))

        residual = self._sample_from_spec(seg.residual_spec, fallback_scale=seg.residual_spec.get("scale", 30.0))
        return float(max(10.0, baseline + residual))

    def _sample_from_spec(self, spec: Dict[str, Any], fallback_scale: float) -> float:
        dist = (spec.get("distribution") or "").strip().lower()
        loc = float(_float_or_default(spec.get("loc"), 0.0) or 0.0)
        scale = _float_or_default(spec.get("scale"), fallback_scale) or fallback_scale or 1.0
        shape = _float_or_default(spec.get("shape"), None)
        df = _float_or_default(spec.get("df"), None)

        try:
            if dist in {"t", "student", "student_t"}:
                df = df or shape or 5.0
                df = max(1.0, df)
                return stats.t.rvs(df, loc=loc, scale=scale)
            if dist in {"lognorm", "lognormal"}:
                s = shape or 0.5
                return stats.lognorm.rvs(s=s, loc=loc, scale=max(scale, 1e-3))
            if dist == "gamma":
                a = shape or df or 1.0
                return stats.gamma.rvs(a=max(a, 0.1), loc=loc, scale=max(scale, 1e-3))
            if dist in {"weibull", "weibull_min"}:
                c = shape or df or 1.5
                return stats.weibull_min.rvs(c=max(c, 0.1), loc=loc, scale=max(scale, 1e-3))
            if dist in {"normal", "gaussian"}:
                return np.random.normal(loc, max(scale, 1.0))
        except Exception:
            pass

        return float(np.random.normal(loc, max(scale, fallback_scale, 1.0)))

# %%
try:
    SERVICE_TIME_MODEL = ServiceTimeRepository(Path("service_time"))
except Exception:
    SERVICE_TIME_MODEL = ServiceTimeCSV.from_file("service_time_distributions_by_profile.csv")

# %% [markdown]
# ## Balking

# %%
HIGH_TOLERANCE_PROFILES = {
    CustomerProfile.FAMILY_CART,
    CustomerProfile.WEEKLY_PLANNER,
}
RESTRICTED_QUEUE_MAX = 2.0
REGULAR_QUEUE_MAX = 5.0
SELF_CHECKOUT_QUEUE_MAX = 3.0
PERMISSIVE_QUEUE_MIN = 1e6


def _load_queue_tolerance_overrides(
    csv_path: str,
    column: str = "std_queue_length",
) -> Dict[Tuple[CustomerProfile, PriorityType, Optional[PaymentMethod]], float]:
    """
    Lee métricas empíricas de tolerancia por perfil/prioridad/pago.
    Se usa la columna indicada como largo máximo antes del balk.
    """
    path = Path(csv_path)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}

    overrides: Dict[Tuple[CustomerProfile, PriorityType, Optional[PaymentMethod]], float] = {}
    for _, row in df.iterrows():
        try:
            prof = _norm_enum(row.get("profile"), CustomerProfile)
            prio = _norm_enum(row.get("priority"), PriorityType)
            pay = _norm_enum(row.get("payment_method"), PaymentMethod)
        except Exception:
            continue

        value = row.get(column)
        if pd.isna(value):
            continue
        threshold = float(value)
        overrides[(prof, prio, pay)] = threshold
        gen_key = (prof, prio, None)
        current = overrides.get(gen_key)
        overrides[gen_key] = max(current, threshold) if current is not None else threshold

    return overrides


@dataclass
class BalkingToleranceModel:
    tolerance_file: str
    column: str = "std_queue_length"
    fallback_tolerance: float = 1.0
    buffer_ratio: float = 0.25
    _tolerance_map: Dict[Tuple[CustomerProfile, PriorityType, Optional[PaymentMethod]], float] = field(default_factory=dict)

    def __post_init__(self):
        self._tolerance_map = _load_queue_tolerance_overrides(self.tolerance_file, column=self.column)

    def _raw_threshold(self, profile, priority, payment) -> float:
        keys = [
            (profile, priority, payment),
            (profile, priority, None),
        ]
        for key in keys:
            if key in self._tolerance_map:
                return float(self._tolerance_map[key])
        return float("nan")

    def _effective_threshold(self, profile, priority, payment) -> float:
        raw = self._raw_threshold(profile, priority, payment)
        if not np.isfinite(raw):
            raw = self.fallback_tolerance
        if profile not in HIGH_TOLERANCE_PROFILES:
            allowed = RESTRICTED_QUEUE_MAX
            if profile is CustomerProfile.REGULAR:
                allowed = REGULAR_QUEUE_MAX
            elif profile is CustomerProfile.SELF_CHECKOUT_FAN:
                allowed = SELF_CHECKOUT_QUEUE_MAX
            return min(allowed, max(0.0, raw))
        return max(PERMISSIVE_QUEUE_MIN, raw)

    def prob_balk(
        self,
        profile: CustomerProfile,
        priority: PriorityType,
        *,
        queue_len: int,
        items: int,
        lane_type: LaneType,
        payment: PaymentMethod,
        arrival_hour: int,
        day_type: DayType,
    ) -> float:
        del items, lane_type, arrival_hour, day_type  # no usados en el modelo actual
        threshold = self._effective_threshold(profile, priority, payment)
        queue = float(max(queue_len, 0))
        if queue <= threshold:
            return 0.0
        ramp = max(1.0, self.buffer_ratio * max(threshold, 1.0))
        return float(np.clip((queue - threshold) / ramp, 0.0, 1.0))

    def decide(self, *args, **kwargs) -> bool:
        return np.random.rand() < self.prob_balk(*args, **kwargs)


# %%


# %%
QUEUE_TOLERANCE_FILE = "queue_tolerance_by_profile_payment_priority.csv"
balk_model = BalkingToleranceModel(
    tolerance_file=QUEUE_TOLERANCE_FILE,
    column="std_queue_length",
)
BALK_MODEL = balk_model

# %% [markdown]
# ## Relación de profit

# %%
# === Celda: parámetros de profit con coeficientes por ítems ===
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROFIT_FILE = "total_profit_all_tables.csv"
ITEMS_ANALYSIS_FILE = "profile_items_analysis_summary(in).csv"

# Coeficientes de pendiente por ítem cuando no viene en el CSV (CLP por ítem)
PROFILE_COEF_ITEMS = {
    "deal_hunter": 196.951060,
    "regular": 248.6974,
    "weekly_planner": 248.6974,
    "family_cart": 257.442983,
    "self_checkout_fan": 277.252190,
    "express_basket": 280.162367,
}

# Mapea DayType -> etiqueta de "grupo" usada en el CSV
def _grupo_label(day_type):
    if isinstance(day_type, DayType):
        if day_type is DayType.TYPE_1:
            return "Grupo 1 (Dias 1,2,4)"
        if day_type is DayType.TYPE_2:
            return "Grupo 2 (Dias 3,5,6)"
        if day_type is DayType.TYPE_3:
            return "Grupo 3 (Dia 7)"
    text = str(getattr(day_type, "value", day_type)).strip().lower()
    if "1" in text:
        return "Grupo 1 (Dias 1,2,4)"
    if "2" in text:
        return "Grupo 2 (Dias 3,5,6)"
    if "3" in text:
        return "Grupo 3 (Dia 7)"
    return ""

_DAYTYPE_TO_GRUPO = {
    DayType.TYPE_1: "Grupo 1 (Dias 1,2,4)",
    DayType.TYPE_2: "Grupo 2 (Dias 3,5,6)",
    DayType.TYPE_3: "Grupo 3 (Dia 7)",
}

def _normalize_payment(value: str) -> str:
    text = str(value).strip().lower()
    if "card" in text:
        return "card"
    if "cash" in text:
        return "cash"
    return ""

def _normalize_group(value: str) -> str:
    text = str(value).strip().lower()
    for ch in text:
        if ch.isdigit():
            return f"grupo {ch}"
    return ""

def _load_items_mean(csv_path: str) -> dict[tuple[str, str, str, str], float]:
    data: dict[tuple[str, str, str, str], float] = {}
    path = Path(csv_path)
    if not path.exists():
        return data
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                mean = float(row.get("mean_items", ""))
            except (TypeError, ValueError):
                continue
            profile = row.get("profile", "").strip()
            priority = row.get("priority", "").strip()
            segment = row.get("segment", "")
            pay_key = _normalize_payment(segment)
            grp_key = _normalize_group(segment)
            if pay_key or grp_key:
                data[(profile, priority, pay_key, grp_key)] = mean
                if grp_key and (profile, priority, "", grp_key) not in data:
                    data[(profile, priority, "", grp_key)] = mean
            else:
                if (profile, priority, "", "") not in data:
                    data[(profile, priority, "", "")] = mean
    return data

ITEMS_MEAN_BY_SEGMENT = _load_items_mean(ITEMS_ANALYSIS_FILE)

def _parse_segment_fields(segment: str) -> dict:
    # ej: "payment_method=card & grupo=Grupo 1 (Dias 1,2,4)"
    d = {}
    for part in str(segment).split("&"):
        kv = part.strip().split("=")
        if len(kv) == 2:
            d[kv[0].strip()] = kv[1].strip()
    return d

def _load_profit_distributions(csv_path: str) -> dict:
    dfp = pd.read_csv(csv_path)
    dfp = dfp[(dfp["tabla"] == "distribuciones_parametricas") & (dfp["is_best_aic"] == True)]
    out: dict[tuple[str, str, str, str], dict] = {}
    for _, r in dfp.iterrows():
        seg = _parse_segment_fields(r.get("segment", ""))
        payment_txt = seg.get("payment_method", "")
        grupo_txt = seg.get("grupo", "")
        payment_norm = _normalize_payment(payment_txt if payment_txt else r.get("segment", ""))
        group_norm = _normalize_group(grupo_txt if grupo_txt else r.get("segment", ""))
        key = (
            r["archivo"],
            r["priority"],
            payment_txt,
            grupo_txt,
        )
        try:
            params = json.loads(r.get("params", "{}"))
        except Exception:
            params = {}
        try:
            coef_items = float(r.get("coef_items", np.nan))
        except (TypeError, ValueError):
            coef_items = np.nan
        items_ref = np.nan
        if ITEMS_MEAN_BY_SEGMENT:
            lookup_keys = [
                (r["archivo"], r["priority"], payment_norm, group_norm),
                (r["archivo"], r["priority"], "", group_norm),
                (r["archivo"], r["priority"], payment_norm, ""),
                (r["archivo"], r["priority"], "", ""),
            ]
            for lk in lookup_keys:
                candidate = ITEMS_MEAN_BY_SEGMENT.get(lk, np.nan)
                if np.isfinite(candidate):
                    items_ref = candidate
                    break
        out[key] = {
            "dist_name": str(r["distribution"]),
            "params": params,
            "support_min": r.get("support_min", np.nan),
            "support_max": r.get("support_max", np.nan),
            "coef_items": coef_items,
            "items_ref": items_ref,
        }
    assert len(out) > 0, "No se cargaron distribuciones de profit"
    return out

PROFIT_DISTRIBUTIONS = _load_profit_distributions(PROFIT_FILE)


# %%
# === Celda: muestreo de profit desde distribuciones ===
import numpy as np

_rng_profit_internal = np.random.default_rng()

def _sample_from_dist(dist_name: str, params: dict, rng=None) -> float:
    """Muestrea profit base a partir de distribuciones paramétricas del CSV."""
    rng = rng or _rng_profit_internal
    name = str(dist_name).lower()

    if "mezcla lognormal + exponencial" in name:
        w = float(params.get("weight_lognorm", 0.5))
        if rng.random() < w:
            mu = float(params.get("lognormal_mu", 8.0))
            sigma = float(params.get("lognormal_sigma", 1.0))
            x = rng.lognormal(mean=mu, sigma=sigma)
        else:
            scale = float(params.get("exponential_scale", 2000.0))
            x = rng.exponential(scale=scale)
        shift = float(params.get("shift", 0.0))
        return max(0.0, x + shift)

    if "gamma" in name:
        shape = float(params.get("a", params.get("shape", 1.0)))
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        x = rng.gamma(shape=shape, scale=scale) + loc
        return max(0.0, x)

    if "lognormal" in name:
        sigma = float(params.get("s", params.get("sigma", 1.0)))
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        mu = np.log(scale) if scale > 0 else 0.0
        x = rng.lognormal(mean=mu, sigma=sigma) + loc
        return max(0.0, x)

    if "weibull" in name:
        c = float(params.get("c", params.get("k", 1.5)))
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        x = rng.weibull(a=c) * scale + loc
        return max(0.0, x)

    if "burr" in name:
        c = float(params.get("c", 1.5))
        d = float(params.get("d", 5.0))
        loc = float(params.get("loc", 0.0))
        scale = float(params.get("scale", 1.0))
        u = rng.random()
        x = scale * ((u ** (-1.0 / d) - 1.0) ** (1.0 / c)) + loc
        return max(0.0, x)

    return max(0.0, float(params.get("loc", 0.0)))

_DIST_STATS_CACHE: dict[int, tuple[float, float]] = {}

def _estimate_dist_stats(dist_name: str, params: dict) -> tuple[float, float]:
    key = (dist_name, tuple(sorted(params.items())))
    key_hash = hash(key)
    cached = _DIST_STATS_CACHE.get(key_hash)
    if cached:
        return cached
    rng = np.random.default_rng(12345)
    samples = [_sample_from_dist(dist_name, params, rng=rng) for _ in range(1024)]
    if samples:
        mean_val = float(np.mean(samples))
        std_val = float(np.std(samples, ddof=0))
    else:
        mean_val = 0.0
        std_val = 0.0
    _DIST_STATS_CACHE[key_hash] = (mean_val, std_val)
    return mean_val, std_val

_MIN_CV = 0.01   # coeficiente de variación mínimo deseado
_MAX_CV = 0.05   # coeficiente de variación máximo permitido


def sample_profit(items: int,
                  profile, priority, payment_method: str, day_type,
                  profit_dict: dict = PROFIT_DISTRIBUTIONS,
                  items_ref: int = 30) -> float:
    """
    1) Selecciona distribución por (profile, priority, payment_method, grupo(day_type)).
    2) Centra la distribución en la pendiente por ítems y aplica ruido limitado.
    """
    grupo = _DAYTYPE_TO_GRUPO.get(day_type, _grupo_label(day_type))
    payment_txt = getattr(payment_method, "value", str(payment_method))
    profile_key = getattr(profile, "value", str(profile))
    priority_key = getattr(priority, "value", str(priority))
    key = (profile_key, priority_key, payment_txt, grupo)

    items_count = max(0, int(items))
    if items_count <= 0:
        return 0

    spec = profit_dict.get(key)
    if not spec:
        fallbacks = []
        if payment_txt:
            fallbacks.append((profile_key, priority_key, "", grupo))
        if grupo:
            fallbacks.append((profile_key, priority_key, payment_txt, ""))
        fallbacks.append((profile_key, priority_key, "", ""))
        for alt in fallbacks:
            spec = profit_dict.get(alt)
            if spec:
                break

    if not spec:
        coef_fallback = PROFILE_COEF_ITEMS.get(profile_key, np.nan)
        if np.isfinite(coef_fallback) and coef_fallback > 0:
            return int(round(coef_fallback * items_count))
        ref_items = max(1.0, float(items_ref))
        return int(round(items_count / ref_items))

    coef = spec.get("coef_items", np.nan)
    if not np.isfinite(coef) or coef <= 0:
        coef = PROFILE_COEF_ITEMS.get(profile_key, np.nan)

    base = _sample_from_dist(spec["dist_name"], spec["params"])
    base = max(0.0, float(base))

    if np.isfinite(coef) and coef > 0:
        stats = spec.get("dist_stats")
        if not stats:
            stats = _estimate_dist_stats(spec["dist_name"], spec["params"])
            spec["dist_stats"] = stats
        dist_mean, dist_std = stats
        base_cv = dist_std / max(dist_mean, 1.0) if dist_mean > 0 else _MIN_CV
        target_cv = float(np.clip(base_cv, _MIN_CV, _MAX_CV))
        target_std = target_cv * coef * items_count
        dist_std = max(dist_std, 1e-6)
        scale = target_std / dist_std
        residual = (base - dist_mean) * scale
        clip = 2.0 * target_std
        residual = float(np.clip(residual, -clip, clip))
        profit = coef * items_count + residual
    else:
        ref_items = spec.get("items_ref", np.nan)
        if not np.isfinite(ref_items) or ref_items <= 0:
            ref_items = max(1.0, float(items_ref))
        profit = (base / float(ref_items)) * items_count

    lo = spec.get("support_min", np.nan)
    if np.isfinite(lo):
        profit = max(float(lo), profit)
    hi = spec.get("support_max", np.nan)
    if np.isfinite(hi):
        profit = min(float(hi), profit)
    return int(round(max(0.0, profit)))


# %%
# === Celda: integración en el evento de checkout ===
# supuestos: objeto customer con atributos .profile y .items
if "RNG_PROFIT" not in globals():
    RNG_PROFIT = np.random.default_rng(42)

def finalize_customer_profit(customer, *, profit_dict=PROFIT_DISTRIBUTIONS, default_day_type=DayType.TYPE_1):
    profile = getattr(customer, "profile", "")
    priority = getattr(customer, "priority", PriorityType.NO_PRIORITY)
    payment = getattr(customer, "payment_method", PaymentMethod.CARD)
    day_type = getattr(customer, "day_type", default_day_type)
    items = getattr(customer, "items", 0)
    customer.total_profit_clp = sample_profit(
        items=int(items),
        profile=profile,
        priority=priority,
        payment_method=payment,
        day_type=day_type,
        profit_dict=profit_dict,
    )


# %%
# df_events: columnas mínimas ['profile','items'] y opcionales ['priority','payment_method','day_type']
def compute_profit_column(df_events: pd.DataFrame,
                          *,
                          profit_dict: dict = PROFIT_DISTRIBUTIONS,
                          default_priority=PriorityType.NO_PRIORITY,
                          default_payment=PaymentMethod.CARD,
                          default_day_type=DayType.TYPE_1) -> pd.Series:
    def _row_profit(row: pd.Series) -> float:
        profile = row.get("profile", "")
        priority = row.get("priority", default_priority)
        payment = row.get("payment_method", default_payment)
        day_type = row.get("day_type", default_day_type)
        items = row.get("items", 0)
        return sample_profit(
            items=int(items),
            profile=profile,
            priority=priority,
            payment_method=payment,
            day_type=day_type,
            profit_dict=profit_dict,
        )
    return df_events.apply(_row_profit, axis=1)

# ejemplo:
# df_events["total_profit_clp"] = compute_profit_column(df_events)


# %%
for prof in PROFILE_COEF_ITEMS.keys():
    sample = sample_profit(
        items=5,
        profile=prof,
        priority=PriorityType.NO_PRIORITY,
        payment_method=PaymentMethod.CARD,
        day_type=DayType.TYPE_1,
    )
    print(prof, "items=5 =>", round(sample, 2))


# %% [markdown]
# # Generación de clientes y simulación

# %%
# ===================== CSV schemas =====================
CUSTOMERS_FIELDS = [
    "source_folder","customer_id","profile","priority","items","payment_method",
    "arrival_time_s","service_start_s","service_end_s","wait_time_s","service_time_s",
    "total_time_s","arrival_hour","total_revenue_clp","cart_categories",
    "cart_category_revenue_clp","cart_items","total_profit_clp","cart_category_profit_clp",
    "lane_name","lane_type","queue_request_priority","patience_s","outcome",
    "balk_reason","abandon_reason","effective_queue_length"
]
TIMELOG_FIELDS = [
    "source_folder","timestamp_s","event_type","customer_id","profile","priority","items",
    "payment_method","lane_name","lane_type","effective_queue_length",
    "queue_request_priority","patience_s","service_time_s","revenue_clp",
    "profit_clp","reason"
]
def _fill_missing(rows, fields):
    for r in rows:
        for k in fields: r.setdefault(k, "")

# %%
def spawn_customer(env, lanes, cliente, time_log, customers_rows,
                   balk_model=None, day_type=None):
    """
    arrival_time_s relativo a 08:00 (desde la apertura).
    balk_model: instancia estilo BalkingToleranceModel (o None).
    profit_dict: distribuciones de profit cargadas con _load_profit_distributions().
    day_type: enum DayType del día en curso.
    """
    yield env.timeout(cliente["arrival_time_s"])

    # --- asegurar items antes de cualquier uso ---
    if "items" not in cliente or not isinstance(cliente["items"], (int, float)) or cliente["items"] <= 0:
        prof_str = (getattr(cliente["profile"], "name", str(cliente["profile"])) or "").lower()
        try:
            k = draw_items(
                prof_str,
                priority=cliente.get("priority"),
                payment_method=cliente.get("payment_method"),
                day_type=cliente.get("day_type") or day_type,
                rng=RNG_ITEMS,
            )
        except Exception:
            k = 1
        cliente["items"] = int(max(1, k))
    else:
        cliente["items"] = int(max(1, cliente["items"]))
    # --- fin nuevo ---

    def as_str(value):
        return value.value if hasattr(value, "value") else value

    def lane_type_str(lane):
        lt = lane.lane_type
        return lt.value if hasattr(lt, "value") else lt

    # 1) filtrar cajas elegibles
    eligibles = [lane for lane in lanes if elegible(cliente, lane)]
    if not eligibles:
        reason = "no_eligible_lane"
        time_log.append({
            "source_folder": cliente["source_folder"], "timestamp_s": env.now, "event_type": "balk",
            "customer_id": cliente["customer_id"], "profile": as_str(cliente["profile"]),
            "priority": as_str(cliente["priority"]), "items": cliente["items"],
            "payment_method": as_str(cliente["payment_method"]),
            "lane_name": "", "lane_type": "", "effective_queue_length": 0,
            "queue_request_priority": as_str(cliente["priority"]),
            "patience_s": cliente["patience_s"], "service_time_s": 0,
            "revenue_clp": 0, "profit_clp": 0, "reason": reason
        })
        registro = cliente.copy()
        registro.update({
            "service_start_s": "", "service_end_s": "", "wait_time_s": "", "service_time_s": "",
            "total_time_s": "", "arrival_hour": int((OPEN_S + cliente["arrival_time_s"]) // 3600),
            "lane_name": "", "lane_type": "", "queue_request_priority": as_str(cliente["priority"]),
            "outcome": "balk", "balk_reason": reason, "abandon_reason": "", "effective_queue_length": 0
        })
        customers_rows.append(registro)
        return

    def cola_efectiva(lane: "CheckoutLane"):
        en_queue = len(lane.servidor.queue)
        en_servicio = 1 if getattr(lane.servidor, "count", 0) > 0 else 0
        return en_queue + en_servicio

    # 2) elegir la caja con menor cola efectiva
    for lane in eligibles:
        lane._eff = cola_efectiva(lane)
    lane = min(eligibles, key=lambda ln: ln._eff)
    eff_q_len = max(int(lane._eff), 0)
    lane_type_txt = lane_type_str(lane)

    # 3) loguear solicitud de cola
    time_log.append({
        "source_folder": cliente["source_folder"], "timestamp_s": env.now, "event_type": "queue_request",
        "customer_id": cliente["customer_id"], "profile": as_str(cliente["profile"]),
        "priority": as_str(cliente["priority"]), "items": cliente["items"],
        "payment_method": as_str(cliente["payment_method"]),
        "lane_name": lane.lane_id, "lane_type": lane_type_txt,
        "effective_queue_length": eff_q_len, "queue_request_priority": as_str(cliente["priority"]),
        "patience_s": cliente["patience_s"], "service_time_s": 0,
        "revenue_clp": 0, "profit_clp": 0, "reason": ""
    })

    # 4) balk probabilístico
    if balk_model is not None:
        p_balk = float(balk_model.prob_balk(
            profile=cliente["profile"],
            priority=cliente["priority"],
            queue_len=eff_q_len,
            items=int(cliente["items"]),
            lane_type=lane.lane_type,
            payment=cliente["payment_method"],
            arrival_hour=int((OPEN_S + cliente["arrival_time_s"]) // 3600),
            day_type=cliente.get("day_type") or day_type,
        ))
        p_balk = min(max(p_balk, 0.0), 1.0)
        if np.random.random() < p_balk:
            reason = "balk_queue_tolerance"
            time_log.append({
                "source_folder": cliente["source_folder"], "timestamp_s": env.now, "event_type": "balk",
                "customer_id": cliente["customer_id"], "profile": as_str(cliente["profile"]),
                "priority": as_str(cliente["priority"]), "items": cliente["items"],
                "payment_method": as_str(cliente["payment_method"]),
                "lane_name": lane.lane_id, "lane_type": lane_type_txt,
                "effective_queue_length": eff_q_len, "queue_request_priority": as_str(cliente["priority"]),
                "patience_s": cliente["patience_s"], "service_time_s": 0,
                "revenue_clp": 0, "profit_clp": 0, "reason": reason
            })
            registro = cliente.copy()
            registro.update({
                "service_start_s": "", "service_end_s": "", "wait_time_s": "", "service_time_s": "",
                "total_time_s": "", "arrival_hour": int((OPEN_S + cliente["arrival_time_s"]) // 3600),
                "lane_name": lane.lane_id, "lane_type": lane_type_txt,
                "queue_request_priority": as_str(cliente["priority"]),
                "outcome": "balk", "balk_reason": reason, "abandon_reason": "", "effective_queue_length": eff_q_len
            })
            customers_rows.append(registro)
            return

    # 5) esperar turno (paciencia + cierre)
    with lane.servidor.request() as req:
        jornada_restante = max(0.0, (CLOSE_S - OPEN_S) - env.now)
        max_wait = min(float(cliente["patience_s"]), jornada_restante)

        start_wait = env.now
        resultado = yield req | env.timeout(max_wait)

        if req not in resultado:
            reason = "store_close" if env.now >= (CLOSE_S - OPEN_S) else "patience_timeout"
            time_log.append({
                "source_folder": cliente["source_folder"], "timestamp_s": env.now, "event_type": "abandon",
                "customer_id": cliente["customer_id"], "profile": as_str(cliente["profile"]),
                "priority": as_str(cliente["priority"]), "items": cliente["items"],
                "payment_method": as_str(cliente["payment_method"]),
                "lane_name": lane.lane_id, "lane_type": lane_type_txt,
                "effective_queue_length": cola_efectiva(lane), "queue_request_priority": as_str(cliente["priority"]),
                "patience_s": cliente["patience_s"], "service_time_s": 0,
                "revenue_clp": 0, "profit_clp": 0, "reason": reason
            })
            registro = cliente.copy()
            registro.update({
                "service_start_s": "", "service_end_s": "",
                "wait_time_s": env.now - start_wait, "service_time_s": "",
                "total_time_s": env.now - cliente["arrival_time_s"],
                "arrival_hour": int((OPEN_S + cliente["arrival_time_s"]) // 3600),
                "lane_name": lane.lane_id, "lane_type": lane_type_txt,
                "queue_request_priority": as_str(cliente["priority"]),
                "outcome": "abandon", "balk_reason": "", "abandon_reason": reason,
                "effective_queue_length": cola_efectiva(lane)
            })
            customers_rows.append(registro)
            return

        # 6) servicio
        service_start = env.now
        wait_time = env.now - start_wait

        st = None
        st_cap = None  # <-- evita NameError

        if "SERVICE_TIME_MODEL" in globals():
            try:
                st = float(SERVICE_TIME_MODEL.sample(
                    profile=cliente["profile"],
                    items=int(cliente["items"]),
                    lane_type=lane.lane_type,
                    day_type=cliente.get("day_type") or day_type,
                    priority=cliente["priority"],
                    payment_method=cliente["payment_method"],
                ))
            except Exception:
                st = None

        if not (isinstance(st, (int, float)) and np.isfinite(st) and st > 0):
            items_int = int(cliente["items"])
            st = max(0, 15.0 + 4.0 * items_int)

        st_cap = max(0.0, min(st, (CLOSE_S - OPEN_S) - service_start))  # <-- ahora existe
        yield env.timeout(st_cap)
        service_end = env.now
        outcome = "served" if abs(st_cap - st) <= 1e-6 else "cutoff_close"

        # 7) profit
        items_int = int(cliente["items"])
        profit_clp = sample_profit(
            items=items_int,
            profile=cliente["profile"],
            priority=cliente["priority"],
            payment_method=cliente["payment_method"],
            day_type=cliente.get("day_type") or day_type,
            profit_dict=PROFIT_DISTRIBUTIONS,
        )
        cliente["total_profit_clp"] = profit_clp

        # 8) logs de servicio
        time_log.append({
            "source_folder": cliente["source_folder"], "timestamp_s": service_start, "event_type": "service_start",
            "customer_id": cliente["customer_id"], "profile": as_str(cliente["profile"]),
            "priority": as_str(cliente["priority"]), "items": cliente["items"],
            "payment_method": as_str(cliente["payment_method"]),
            "lane_name": lane.lane_id, "lane_type": lane_type_txt,
            "effective_queue_length": cola_efectiva(lane), "queue_request_priority": as_str(cliente["priority"]),
            "patience_s": cliente["patience_s"], "service_time_s": st_cap,
            "revenue_clp": 0, "profit_clp": profit_clp, "reason": ""
        })
        time_log.append({
            "source_folder": cliente["source_folder"], "timestamp_s": service_end, "event_type": "service_end",
            "customer_id": cliente["customer_id"], "profile": as_str(cliente["profile"]),
            "priority": as_str(cliente["priority"]), "items": cliente["items"],
            "payment_method": as_str(cliente["payment_method"]),
            "lane_name": lane.lane_id, "lane_type": lane_type_txt,
            "effective_queue_length": max(cola_efectiva(lane), 0), "queue_request_priority": as_str(cliente["priority"]),
            "patience_s": cliente["patience_s"], "service_time_s": st_cap,
            "revenue_clp": 0, "profit_clp": profit_clp, "reason": ""
        })

        # 9) registro principal
        registro = cliente.copy()
        registro.update({
            "service_start_s": service_start,
            "service_end_s": service_end,
            "wait_time_s": wait_time,
            "service_time_s": st_cap,
            "total_time_s": service_end - cliente["arrival_time_s"],
            "arrival_hour": int((OPEN_S + cliente["arrival_time_s"]) // 3600),
            "lane_name": lane.lane_id,
            "lane_type": lane_type_txt,
            "queue_request_priority": as_str(cliente["priority"]),
            "outcome": outcome,
            "balk_reason": "",
            "abandon_reason": "",
            "effective_queue_length": eff_q_len,
            "profit_clp": profit_clp
        })
        customers_rows.append(registro)

# %%
from pathlib import Path


# Optimización: precomputar una matriz de lambdas por segundo para todas
# las combinaciones (perfil, prioridad, pago) del día. Esto elimina búsquedas
# repetidas en cada iteración del bucle de llegadas sin cambiar el proceso.
def _build_lambda_matrix(combos: list[tuple], day_len: int) -> np.ndarray:
    """Devuelve matriz (n_combos, day_len+1) con tasas por segundo.

    Para cada combinación usa la serie subyacente (lambda por minuto) y la
    transforma a lambda por segundo constante a trozos en [0, day_len].
    """
    per_profile_series: dict[CustomerProfile, dict] = {}
    rows: list[np.ndarray] = []

    for prof, pr, pm, dist in combos:
        # Cachear series por perfil para evitar I/O repetido
        ser = per_profile_series.get(dist.profile)
        if ser is None:
            ser = _get_series_for_profile(dist.profile)
            per_profile_series[dist.profile] = ser

        pair = ser.get((dist.day_type, dist.priority, dist.payment_method))
        if not pair:
            rows.append(np.zeros(day_len + 1, dtype=float))
            continue

        t, lam = pair
        # t en segundos, lam en "por minuto". Queremos por segundo.
        t = np.asarray(t, dtype=int)
        lam_sec = np.maximum(np.asarray(lam, dtype=float), 0.0) / 60.0

        # Asegurar que cubrimos [0, day_len]
        if t[0] > 0:
            t = np.insert(t, 0, 0)
            lam_sec = np.insert(lam_sec, 0, lam_sec[0])
        if t[-1] < day_len:
            t = np.append(t, day_len)
            lam_sec = np.append(lam_sec, lam_sec[-1])

        # Rellenar por tramos: para s en [t[i], t[i+1]) usar lam_sec[i]
        lengths = np.diff(np.append(t, day_len + 1))
        row = np.repeat(lam_sec, lengths)[: day_len + 1]
        if row.shape[0] < (day_len + 1):
            # Padding defensivo si algo quedó corto por redondeos
            row = np.pad(row, (0, (day_len + 1) - row.shape[0]), mode="edge")
        rows.append(row.astype(float, copy=False))

    return np.vstack(rows) if rows else np.zeros((0, day_len + 1), dtype=float)


def _simular_dia_periodo(
    week_idx: int,
    dia_config: tuple[int, DayType, str],
    perfiles: list[CustomerProfile],
    cfg_by_prof: dict[CustomerProfile, ProfileConfig],
    balk_model,
    output_folder: Path,
) -> dict:
    dia_num, day_type, descripcion = dia_config
    tag = f"Week-{week_idx:02d}-Day-{dia_num:02d}"
    print(f"\n[Semana {week_idx}] {descripcion}\n" + "-" * 50)

    env = simpy.Environment()
    lanes: list[CheckoutLane] = []

    def add(prefix: str, lane_type: LaneType, cantidad: int) -> None:
        for idx in range(cantidad):
            lanes.append(CheckoutLane(env, f"{prefix}-{idx+1}", lane_type))

    policy = CURRENT_LANE_POLICY.get(day_type)
    if not policy:
        policy = CURRENT_LANE_POLICY.get(DayType.TYPE_1, {})

    total_lanes = sum(int(policy.get(lt, 0)) for lt in (LaneType.REGULAR, LaneType.EXPRESS, LaneType.PRIORITY, LaneType.SCO))
    if total_lanes <= 0:
        fallback_counts = DEFAULT_LANE_COUNTS.get(day_type, DEFAULT_LANE_COUNTS.get(DayType.TYPE_1, {}))
        policy = {
            LaneType.REGULAR: int(fallback_counts.get("regular", 0)),
            LaneType.EXPRESS: int(fallback_counts.get("express", 0)),
            LaneType.PRIORITY: int(fallback_counts.get("priority", 0)),
            LaneType.SCO: int(fallback_counts.get("self_checkout", 0)),
        }

    for lane_type in (LaneType.REGULAR, LaneType.EXPRESS, LaneType.PRIORITY, LaneType.SCO):
        count = int(policy.get(lane_type, 0)) if policy else 0
        prefix = LANE_PREFIXES.get(lane_type, lane_type.value.upper())
        add(prefix, lane_type, count)

    customers_rows_dia: list[dict] = []
    time_log_dia: list[dict] = []

    combos = [
        (prof, pr, pm, cfg_by_prof[prof].create_arrival_distribution(day_type, pr, pm))
        for prof in perfiles
        for pr in PriorityType
        for pm in PaymentMethod
    ]

    t0, t1 = OPEN_S, CLOSE_S
    t = float(t0)
    clientes_generados = 0

    # Precompute matriz de tasas por segundo para acelerar el bucle
    JORNADA = int(CLOSE_S - OPEN_S)
    lam_mat = _build_lambda_matrix(combos, JORNADA)

    while t < t1:
        if t > t1:
            break

        t_rel = t - OPEN_S
        sec_idx = int(max(0, min(JORNADA, int(t_rel))))
        # Tasas por segundo para cada combinación en este segundo
        lambdas = lam_mat[:, sec_idx]
        Lambda = float(lambdas.sum())

        resto = t % BIN_SEC
        dt_cap = min((BIN_SEC - resto) if resto != 0 else BIN_SEC, t1 - t)

        if Lambda <= 0.0 or dt_cap <= 0:
            t += max(dt_cap, 1.0)
            continue

        delta = np.random.exponential(1.0 / Lambda)
        if delta > dt_cap:
            t += dt_cap
            continue

        t += delta
        t_rel = t - OPEN_S

        probs = lambdas / lambdas.sum()
        idx = int(np.random.choice(len(combos), p=probs))
        prof, priority, payment_method, _ = combos[idx]
        # --- Nuevo
        prof_str = getattr(prof, "name", str(prof)).lower()

        # usa items_sampler si lo agregaste en ProfileConfig; si no, llama draw_items directo
        if hasattr(cfg_by_prof[prof], "items_sampler"):
            items = cfg_by_prof[prof].items_sampler(
                prof_str,
                priority=priority,
                payment_method=payment_method,
                day_type=day_type,
                rng=RNG_ITEMS,
            )
        else:
            items = draw_items(
                prof_str,
                priority=priority,
                payment_method=payment_method,
                day_type=day_type,
                rng=RNG_ITEMS,
            )

        items = int(max(1, items))

        # ---- Nuevo

        patience = cfg_by_prof[prof].patience_distribution.sample(prof, priority)

        cliente = {
            "source_folder": tag,
            "customer_id": f"{clientes_generados + 1}",
            "profile": prof,
            "priority": priority,
            "items": int(items),
            "payment_method": payment_method,
            "arrival_time_s": float(t_rel),
            "patience_s": float(patience),
            "day_type": day_type,
            "wait_time_s": "",
            "service_start_s": "",
            "service_end_s": "",
            "service_time_s": "",
            "total_time_s": "",
            "lane_name": "",
            "lane_type": "",
            "effective_queue_length": 0,
            "outcome": "",
            "balk_reason": "",
            "abandon_reason": "",
            "total_revenue_clp": 0,
            "cart_categories": "",
            "cart_category_revenue_clp": "",
            "cart_items": int(items),
            "total_profit_clp": 0,
            "cart_category_profit_clp": "",
            "queue_request_priority": priority,
            "arrival_hour": int((OPEN_S + t_rel) // 3600),
        }

        env.process(
            spawn_customer(
                env,
                lanes,
                cliente,
                time_log_dia,
                customers_rows_dia,
                balk_model=balk_model,
                day_type=day_type,
            )
        )

        clientes_generados += 1

    env.run(until=(t1 - t0))

    for row in customers_rows_dia:
        row["arrival_hour"] = int((OPEN_S + float(row["arrival_time_s"])) // 3600)
        if row.get("effective_queue_length", "") in ("", None):
            row["effective_queue_length"] = 0
        row["effective_queue_length"] = max(int(row["effective_queue_length"]), 0)

        if "total_revenue_clp" not in row:
            row["total_revenue_clp"] = 0
        if "total_profit_clp" not in row:
            row["total_profit_clp"] = 0

        row.setdefault("cart_categories", "")
        row.setdefault("cart_category_revenue_clp", "")
        row.setdefault("cart_items", row.get("items", ""))
        row.setdefault("cart_category_profit_clp", "")

        if hasattr(row.get("profile"), "value"):
            row["profile"] = row["profile"].value
        if hasattr(row.get("priority"), "value"):
            row["priority"] = row["priority"].value
        if hasattr(row.get("payment_method"), "value"):
            row["payment_method"] = row["payment_method"].value
        if hasattr(row.get("queue_request_priority"), "value"):
            row["queue_request_priority"] = row["queue_request_priority"].value

    for row in time_log_dia:
        row.setdefault("source_folder", tag)
        row.setdefault("profile", "")
        row.setdefault("priority", "")
        row.setdefault("items", "")
        row.setdefault("payment_method", "")
        row.setdefault("lane_name", "")
        row.setdefault("lane_type", "")
        row.setdefault("effective_queue_length", 0)
        row.setdefault("queue_request_priority", "")
        row.setdefault("patience_s", "")
        row.setdefault("service_time_s", "")
        row.setdefault("revenue_clp", 0)
        row.setdefault("profit_clp", 0)
        row.setdefault("reason", "")

    for rows, fields in ((customers_rows_dia, CUSTOMERS_FIELDS), (time_log_dia, TIMELOG_FIELDS)):
        _fill_missing(rows, fields)

    output_folder.mkdir(parents=True, exist_ok=True)

    with open(output_folder / "customers.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CUSTOMERS_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(customers_rows_dia)

    with open(output_folder / "time_log.csv", "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TIMELOG_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(time_log_dia)

    print(f"[Semana {week_idx}] Dia {dia_num} guardado en {output_folder} ({len(customers_rows_dia)} clientes)")

    items_prom = np.mean([c["items"] for c in customers_rows_dia]) if customers_rows_dia else 0
    paciencia_prom = np.mean([c["patience_s"] for c in customers_rows_dia]) if customers_rows_dia else 0

    return {
        "week": week_idx,
        "dia": dia_num,
        "dia_tipo": day_type.value,
        "total_clientes": len(customers_rows_dia),
        "items_promedio": float(items_prom),
        "paciencia_promedio": float(paciencia_prom),
    }


def simulacion_periodos(
    num_weeks: int = 1,
    output_root: str = "outputs",
    include_timestamp: bool = False,
    start_week_idx: int = 1,
    titulo: str | None = None,
) -> list[dict]:
    header = titulo or "SIMULACION MULTI-SEMANA"
    print(header + "\n" + "=" * 70)

    perfiles = [
        CustomerProfile.DEAL_HUNTER,
        CustomerProfile.FAMILY_CART,
        CustomerProfile.WEEKLY_PLANNER,
        CustomerProfile.SELF_CHECKOUT_FAN,
        CustomerProfile.REGULAR,
        CustomerProfile.EXPRESS_BASKET,
    ]
    cfg_by_prof: dict[CustomerProfile, ProfileConfig] = {perfil: ProfileConfig(profile=perfil) for perfil in perfiles}

    dias_simulacion = [
        (1, DayType.TYPE_1, "Dia 1 - Grupo 1"),
        (2, DayType.TYPE_1, "Dia 2 - Grupo 1"),
        (3, DayType.TYPE_2, "Dia 3 - Ofertas"),
        (4, DayType.TYPE_1, "Dia 4 - Grupo 1"),
        (5, DayType.TYPE_2, "Dia 5 - Ofertas"),
        (6, DayType.TYPE_2, "Dia 6 - Ofertas"),
        (7, DayType.TYPE_3, "Dia 7 - Especial"),
    ]

    

    base_dir = Path(output_root)
    if not base_dir.is_absolute():
        base_dir = Path(os.getcwd()) / base_dir
    if include_timestamp:
        base_dir = base_dir / datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    base_dir.mkdir(parents=True, exist_ok=True)

    stats_por_dia: list[dict] = []

    for offset in range(num_weeks):
        semana = start_week_idx + offset
        for dia_config in dias_simulacion:
            day_folder = base_dir / f"Week-{semana:02d}" / f"Day-{dia_config[0]:02d}"
            stats = _simular_dia_periodo(
                week_idx=semana,
                dia_config=dia_config,
                perfiles=perfiles,
                cfg_by_prof=cfg_by_prof,
                balk_model=BALK_MODEL,
                output_folder=day_folder,
            )
            stats_por_dia.append(stats)

    print("\nSIMULACION COMPLETA")
    print(f"Carpeta base: {base_dir}")
    return stats_por_dia


def simulacion_7_dias_completa() -> list[dict]:
    stats = simulacion_periodos(
        num_weeks=1,
        output_root="outputs",
        include_timestamp=True,
        start_week_idx=1,
        titulo="SIMULACION 7 DIAS MULTI-PERFIL",
    )

    stats_filtradas: list[dict] = []
    for registro in stats:
        stats_filtradas.append(
            {
                "dia": registro["dia"],
                "dia_tipo": registro["dia_tipo"],
                "total_clientes": registro["total_clientes"],
                "items_promedio": registro["items_promedio"],
                "paciencia_promedio": registro["paciencia_promedio"],
            }
        )
    return stats_filtradas


def simulacion_anual_completa(output_root: str = "outputs") -> list[dict]:
    return simulacion_periodos(
        num_weeks=52,
        output_root=output_root,
        include_timestamp=False,
        start_week_idx=1,
        titulo="SIMULACION 52 SEMANAS MULTI-PERFIL",
    )


# %%
stats_anual = simulacion_anual_completa(output_root="outputs")
for registro in stats_anual:
    print(registro)

# %% [markdown]
# # Obtención de KPI's 

# %%
OUTPUT_ROOT = Path("outputs")

def load_customers_year(root: Path) -> pd.DataFrame:
    files = sorted(root.glob("Week-*/Day-*/customers.csv"))
    if not files:
        raise FileNotFoundError(f"No se encontraron customers.csv bajo {root.resolve()}")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        # extrae semana/día y asegura tipos
        df["week"] = df["source_folder"].str.extract(r"Week-(\d+)", expand=False).astype(int)
        df["day"] = df["source_folder"].str.extract(r"Day-(\d+)", expand=False).astype(int)
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    out["month_4w"] = ((out["week"] - 1) // 4) + 1
    # normaliza outcomes a nomenclatura del KPI
    out["outcome_norm"] = out["outcome"].replace({"abandon": "abandoned", "balk": "balked"}).fillna(out["outcome"])
    # asegura numérico por si hay celdas vacías
    out["total_profit_clp"] = pd.to_numeric(out["total_profit_clp"], errors="coerce").fillna(0)
    out["wait_time_s"] = pd.to_numeric(out["wait_time_s"], errors="coerce")
    return out

df = load_customers_year(OUTPUT_ROOT)

# Resumen general
perfiles_presentes = sorted(df["profile"].dropna().unique().tolist())
dias_simulados = df[["week","day"]].drop_duplicates().shape[0]
print(f"Perfiles presentes: {perfiles_presentes}")
print(f"Días simulados en outputs: {dias_simulados}")

# 1) PROF_CLP (clientes servidos, año completo)
prof_clp = df.loc[df["outcome_norm"] == "served", "total_profit_clp"].sum()
print(f"PROF_CLP (servidos, anual): {prof_clp:,.0f} CLP")

# 2) Profit mensual (bloques de 4 semanas)
monthly_prof = (
    df.loc[df["outcome_norm"] == "served"]
      .groupby("month_4w")["total_profit_clp"].sum()
      .reset_index(name="prof_clp")
)
monthly_prof["prof_clp_fmt"] = monthly_prof["prof_clp"].map(lambda x: f"{x:,.0f} CLP")
display(monthly_prof)

# 3) TAC_p (tasa de abandono/balk por perfil, anual)
status_cols = ["served", "abandoned", "balked"]
status_df = df[df["outcome_norm"].isin(status_cols)].copy()
counts = status_df.groupby(["profile", "outcome_norm"]).size().unstack(fill_value=0)
counts["total"] = counts.sum(axis=1)
tac_profile = counts.assign(
    served_pct=(counts.get("served", 0) / counts["total"] * 100).round(2),
    abandoned_pct=(counts.get("abandoned", 0) / counts["total"] * 100).round(2),
    balked_pct=(counts.get("balked", 0) / counts["total"] * 100).round(2),
).reset_index()
display(tac_profile)

# 4) TMC_p (tiempo medio de espera por perfil, solo servidos, anual)
tmc_df = (
    df.loc[df["outcome_norm"] == "served"]
      .groupby("profile")["wait_time_s"].mean()
      .reset_index(name="tmc_wait_time_s")
)
tmc_df["tmc_fmt"] = (tmc_df["tmc_wait_time_s"] / 60).map(lambda m: f"{m:.2f} min")
display(tmc_df)

# 5) TAC_HV (tasa de abandono alto volumen, items >= 40, anual)
hv = df[df["items"] >= 40]
hv_counts = hv[hv["outcome_norm"].isin(status_cols)].groupby(["profile", "outcome_norm"]).size().unstack(fill_value=0)
hv_counts["total"] = hv_counts.sum(axis=1)
hv_stats = hv_counts.assign(
    tac_hv_pct=(((hv_counts.get("abandoned", 0) + hv_counts.get("balked", 0)) / hv_counts["total"]) * 100).round(2)
).reset_index()
display(hv_stats)


