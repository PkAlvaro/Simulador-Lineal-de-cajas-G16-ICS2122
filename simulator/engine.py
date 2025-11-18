# %% [markdown]
# # Simulador caso base lineal de cajas

# %%
# LibrerÃ­as
import os, csv, datetime, pickle, ast, inspect, time, json
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
from scipy import stats as scipy_stats

# para que lea los display del jupyter
try:
    from IPython.display import display
except ImportError:

    def display(x):
        print(x)


# %%
# Constantes globales
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT
OPEN_S = 8 * 3600
CLOSE_S = 22 * 3600
BIN_SEC = 1

_ANNOUNCED_FILES: set[Path] = set()
_DEMAND_SCALING_FACTOR: float = 1.0


def _info(message: str, path: Optional[Path] = None) -> None:
    target = f"{path}" if path is not None else ""
    suffix = f": {target}" if target else ""
    print(f"[INFO] {message}{suffix}")


def _warn(message: str, path: Optional[Path] = None) -> None:
    target = f"{path}" if path is not None else ""
    suffix = f": {target}" if target else ""
    print(f"[WARN] {message}{suffix}")


def _announce_once(label: str, path: Path) -> None:
    if path in _ANNOUNCED_FILES:
        return
    _ANNOUNCED_FILES.add(path)
    _info(label, path)


def set_demand_multiplier(factor: float) -> float:
    """Actualiza el factor multiplicador global de demanda."""
    global _DEMAND_SCALING_FACTOR
    try:
        value = float(factor)
    except (TypeError, ValueError):
        value = 1.0
    _DEMAND_SCALING_FACTOR = max(0.0, value)
    return _DEMAND_SCALING_FACTOR


def get_demand_multiplier() -> float:
    return _DEMAND_SCALING_FACTOR


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
    REGULAR = "regular"
    EXPRESS = "express"
    PRIORITY = "priority"
    SCO = "sco"


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
LANE_NAME_NORMALIZATION = {
    "sco": "self_checkout",
    "self_checkout": "self_checkout",
}
SERVICE_TIME_MULTIPLIERS_PATH = (
    PROJECT_ROOT / "service_time/service_time_multipliers.csv"
)
_SERVICE_TIME_MULTIPLIERS = {}


def _load_service_time_multipliers(path: Path) -> dict[tuple[str, str], float]:
    path = Path(path)
    if not path.exists():
        _warn(
            "No se encontro archivo de multiplicadores de servicio, se usara valor base",
            path,
        )
        return {}
    try:
        _announce_once("Cargando multiplicadores de servicio", path)
        df = pd.read_csv(path)
    except Exception as exc:
        _warn(f"No se pudieron cargar los multiplicadores de servicio ({exc})", path)
        return {}
    required = {"lane_type", "profile", "multiplier"}
    if not required.issubset(df.columns):
        print(
            f"[WARN] {path.name} no contiene columnas {required}, se ignoran multiplicadores."
        )
        return {}
    df["lane_type"] = (
        df["lane_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(LANE_NAME_NORMALIZATION)
        .fillna(df["lane_type"])
    )
    df["profile"] = df["profile"].astype(str).str.strip().str.lower()
    table: dict[tuple[str, str], float] = {}
    for _, row in df.iterrows():
        try:
            factor = float(row["multiplier"])
        except Exception:
            continue
        if not np.isfinite(factor) or factor <= 0:
            continue
        key = (row["lane_type"], row["profile"])
        table[key] = factor
    return table


_SERVICE_TIME_MULTIPLIERS = _load_service_time_multipliers(
    SERVICE_TIME_MULTIPLIERS_PATH
)
PATIENCE_DISTRIBUTION_FILE = (
    PROJECT_ROOT / "patience/patience_distribution_profile_priority_payment_day.csv"
)
PATIENCE_BASE_CSV_FILE = PATIENCE_DISTRIBUTION_FILE
_PATIENCE_TABLE_ANNOUNCED = False
DEFAULT_LANE_COUNTS = {
    DayType.TYPE_1: {"regular": 10, "express": 3, "priority": 2, "self_checkout": 5},
    DayType.TYPE_2: {"regular": 10, "express": 3, "priority": 2, "self_checkout": 5},
    DayType.TYPE_3: {"regular": 15, "express": 3, "priority": 2, "self_checkout": 5},
}
DAY_TYPE_BY_DAYNUM = {
    1: DayType.TYPE_1.value,
    2: DayType.TYPE_1.value,
    3: DayType.TYPE_2.value,
    4: DayType.TYPE_1.value,
    5: DayType.TYPE_2.value,
    6: DayType.TYPE_2.value,
    7: DayType.TYPE_3.value,
}

DAY_TYPE_FREQUENCY_PER_WEEK: dict[DayType, int] = {day_type: 0 for day_type in DayType}
for day_num, type_value in DAY_TYPE_BY_DAYNUM.items():
    try:
        dt = DayType(type_value)
    except ValueError:
        continue
    DAY_TYPE_FREQUENCY_PER_WEEK[dt] += 1


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


CURRENT_LANE_POLICY: dict[DayType, dict[LaneType, int]] = build_uniform_policy(
    DEFAULT_LANE_COUNTS[DayType.TYPE_1]
)
CURRENT_LANE_POLICY.update(
    {
        DayType.TYPE_2: CURRENT_LANE_POLICY[DayType.TYPE_1].copy(),
        DayType.TYPE_3: build_uniform_policy(DEFAULT_LANE_COUNTS[DayType.TYPE_3])[
            DayType.TYPE_3
        ],
    }
)

LANE_COSTS_FILE = PROJECT_ROOT / "costos_cajas.csv"


@dataclass
class LaneCostSpec:
    lane_type: LaneType
    capex_clp: float
    maintenance_clp_per_year: float
    opex_clp_per_hour: float
    wage_total_clp_per_hour: float
    useful_life_years: float

    @property
    def hourly_rate(self) -> float:
        return max(0.0, self.opex_clp_per_hour) + max(0.0, self.wage_total_clp_per_hour)

    @property
    def annual_fixed_cost_per_lane(self) -> float:
        life = max(1.0, float(self.useful_life_years))
        return max(0.0, self.capex_clp / life) + max(0.0, self.maintenance_clp_per_year)


def _normalize_lane_type_name(name: str) -> Optional[LaneType]:
    if name is None:
        return None
    name_norm = str(name).strip().lower()
    if name_norm in ("regular",):
        return LaneType.REGULAR
    if name_norm in ("express",):
        return LaneType.EXPRESS
    if name_norm in ("priority", "preferente", "pref"):
        return LaneType.PRIORITY
    if name_norm in ("self_checkout", "sco", "autocaja"):
        return LaneType.SCO
    return None


def _load_lane_cost_specs(path: Path = LANE_COSTS_FILE) -> dict[LaneType, LaneCostSpec]:
    path = Path(path)
    if not path.exists():
        _warn("No se encontro costos_cajas.csv; se asumira costo cero", path)
        return {}
    specs: dict[LaneType, LaneCostSpec] = {}
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                lt = _normalize_lane_type_name(row.get("lane_type"))
                if lt is None:
                    continue
                try:
                    capex = float(row.get("capex_clp", 0))
                    maintenance = float(row.get("maintenance_clp_per_year", 0))
                    opex = float(row.get("opex_clp_per_hour", 0))
                    wage_total = float(
                        row.get("wage_total_clp_per_hour")
                        or row.get("wage_clp_per_hour")
                        or 0
                    )
                    useful_life = float(row.get("useful_life_years", 1))
                except (TypeError, ValueError):
                    continue
                specs[lt] = LaneCostSpec(
                    lane_type=lt,
                    capex_clp=capex,
                    maintenance_clp_per_year=maintenance,
                    opex_clp_per_hour=opex,
                    wage_total_clp_per_hour=wage_total,
                    useful_life_years=useful_life,
                )
    except Exception as exc:
        _warn(f"No se pudieron leer costos de cajas ({exc})", path)
        return {}
    if specs:
        _announce_once("Cargando costos de infraestructura desde CSV", path)
    return specs


def _compute_lane_cost_summary(
    policy: dict[DayType, dict[LaneType, int]],
    specs: dict[LaneType, LaneCostSpec],
) -> dict[str, Any]:
    jornada_hours = (CLOSE_S - OPEN_S) / 3600.0
    days_per_year = {dt: DAY_TYPE_FREQUENCY_PER_WEEK[dt] * 52 for dt in DayType}
    total_cost = 0.0
    cost_by_day_type = {dt: 0.0 for dt in DayType}
    per_type_details: dict[LaneType, dict[str, float]] = {}

    for lane_type, spec in specs.items():
        installed = max(policy.get(dt, {}).get(lane_type, 0) for dt in DayType)
        if installed <= 0:
            continue
        fixed_cost = installed * spec.annual_fixed_cost_per_lane
        hours_by_daytype: dict[DayType, float] = {}
        total_hours = 0.0
        for dt in DayType:
            lanes = policy.get(dt, {}).get(lane_type, 0)
            hours = lanes * jornada_hours * days_per_year.get(dt, 0)
            hours_by_daytype[dt] = hours
            total_hours += hours
        variable_cost = spec.hourly_rate * total_hours
        lane_cost_total = fixed_cost + variable_cost
        total_cost += lane_cost_total
        per_type_details[lane_type] = {
            "installed_lanes": installed,
            "fixed_cost_clp": fixed_cost,
            "variable_cost_clp": variable_cost,
            "total_cost_clp": lane_cost_total,
        }
        if total_hours <= 0:
            continue
        for dt, hours in hours_by_daytype.items():
            share = hours / total_hours if total_hours else 0.0
            cost_by_day_type[dt] += lane_cost_total * share

    per_week_cost = total_cost / 52.0 if total_cost else 0.0
    return {
        "total_cost": total_cost,
        "per_week_cost": per_week_cost,
        "by_day_type": cost_by_day_type,
        "details": per_type_details,
    }


LANE_COST_SPECS = _load_lane_cost_specs()
LANE_COST_SUMMARY = _compute_lane_cost_summary(CURRENT_LANE_POLICY, LANE_COST_SPECS)
LANE_COST_TOTAL_ANNUAL = float(LANE_COST_SUMMARY.get("total_cost", 0.0))
LANE_COST_PER_WEEK = float(LANE_COST_SUMMARY.get("per_week_cost", 0.0))
LANE_COST_BY_DAYTYPE = LANE_COST_SUMMARY.get("by_day_type", {})
LANE_COST_BY_DAYTYPE_STR = {
    (dt.value if isinstance(dt, DayType) else str(dt)): float(cost)
    for dt, cost in LANE_COST_BY_DAYTYPE.items()
}


# %% [markdown]
# Cargando tasas de llegada desde npz generados por tools/rebuild_arrivals.py.

# %%
ARRIVALS_NPZ_DIR = PROJECT_ROOT / "arrivals_npz"
_ARRIVAL_FILES = {
    CustomerProfile.DEAL_HUNTER: ARRIVALS_NPZ_DIR / "lambda_deal_hunter.npz",
    CustomerProfile.FAMILY_CART: ARRIVALS_NPZ_DIR / "lambda_family_cart.npz",
    CustomerProfile.WEEKLY_PLANNER: ARRIVALS_NPZ_DIR / "lambda_weekly_planner.npz",
    CustomerProfile.SELF_CHECKOUT_FAN: ARRIVALS_NPZ_DIR
    / "lambda_self_checkout_fan.npz",
    CustomerProfile.REGULAR: ARRIVALS_NPZ_DIR / "lambda_regular.npz",
    CustomerProfile.EXPRESS_BASKET: ARRIVALS_NPZ_DIR / "lambda_express_basket.npz",
}

_DAYTYPE_MAP = {
    DayType.TYPE_1: "Tipo 1",
    DayType.TYPE_2: "Tipo 2",
    DayType.TYPE_3: "Tipo 3",
}
_PRIORITY_MAP = {
    PriorityType.NO_PRIORITY: "no_priority",
    PriorityType.SENIOR: "senior",
    PriorityType.PREGNANT: "pregnant",
    PriorityType.REDUCED_MOBILITY: "reduced_mobility",
}
_PAYMENT_MAP = {PaymentMethod.CARD: "card", PaymentMethod.CASH: "cash"}

_SERIES_CACHE: Dict[
    CustomerProfile,
    Dict[Tuple[DayType, PriorityType, PaymentMethod], Tuple[np.ndarray, np.ndarray]],
] = {}
_ARRIVAL_ANNOUNCED: set[Path] = set()


def _load_arrival_series(
    path: Path,
) -> dict[tuple[DayType, PriorityType, PaymentMethod], tuple[np.ndarray, np.ndarray]]:
    data = np.load(path, allow_pickle=False)
    keys = data["keys"]
    lambda_matrix = data["lambdas"]
    bin_left = data["bin_left_s"]
    if bin_left.ndim != 1:
        raise ValueError(f"bin_left_s inválido en {path}")
    series: dict[
        tuple[DayType, PriorityType, PaymentMethod], tuple[np.ndarray, np.ndarray]
    ] = {}
    for idx, key_str in enumerate(keys):
        parts = str(key_str).split("|")
        if len(parts) < 3:
            continue
        day_type = _norm_enum(parts[0], DayType)
        priority = _norm_enum(parts[1], PriorityType)
        payment = _norm_enum(parts[2], PaymentMethod)
        series[(day_type, priority, payment)] = (
            bin_left.copy(),
            lambda_matrix[idx].astype(np.float32, copy=False),
        )
    return series


def _get_series_for_profile(profile: CustomerProfile):
    if profile in _SERIES_CACHE:
        return _SERIES_CACHE[profile]
    path = _ARRIVAL_FILES.get(profile)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"Archivo de llegadas no encontrado para {profile.value}: {path}"
        )
    if path not in _ARRIVAL_ANNOUNCED:
        _announce_once(f"Cargando llegadas para perfil {profile.value}", path)
        _ARRIVAL_ANNOUNCED.add(path)
    series = _load_arrival_series(path)
    _SERIES_CACHE[profile] = series
    return series


def _lambda_step_at(seconds: float, times: np.ndarray, lambdas: np.ndarray) -> float:
    if seconds <= times[0]:
        return float(lambdas[0])
    if seconds >= times[-1]:
        return float(lambdas[-1])  # â† antes devolvÃ­as 0.0
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
        return (_DEMAND_SCALING_FACTOR * lambda_por_minuto) / 60.0


# %% [markdown]
# ## Mapeo de tipos de dÃ­a

# %%
_SEG_DAY_MAP = {
    "tipo 1": DayType.TYPE_1,
    "tipo 2": DayType.TYPE_2,
    "tipo 3": DayType.TYPE_3,
}
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


def _normalize_payment_value(value) -> Optional[PaymentMethod]:
    if value is None:
        return None
    if isinstance(value, PaymentMethod):
        return value
    text = str(value).strip().lower()
    if text in {"", "all"}:
        return None
    return _norm_enum(text, PaymentMethod)


def _normalize_day_type_value(value) -> Optional[DayType]:
    if value is None:
        return None
    if isinstance(value, DayType):
        return value
    text = str(value).strip().lower()
    if text in {"", "all"}:
        return None
    return _norm_enum(text, DayType)


# %% [markdown]
# ## Volumen de compra

# %%
ITEMS_SUMMARY_FILE = PROJECT_ROOT / "items_distribution_summary.csv"
RNG_ITEMS = np.random.default_rng(123)


@dataclass
class ItemDistributionSpec:
    method: str
    dist_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    support: Optional[np.ndarray] = None
    probs: Optional[np.ndarray] = None
    kernel: Optional[str] = None
    bandwidth: Optional[float] = None
    kde_model: Optional[Any] = None

    def sample(self, rng: np.random.Generator) -> int:
        if self.method == "parametric":
            dist = (self.dist_name or "").lower()
            if dist == "nbinom":
                n = float(self.params.get("n", self.params.get("nb_n", 1.0)))
                p = float(self.params.get("p", self.params.get("nb_p", 0.5)))
                val = stats.nbinom.rvs(n, p, random_state=rng)
                return int(max(1, val))
            if dist == "poisson":
                lam = float(self.params.get("lambda", self.params.get("lam", 10.0)))
                return int(max(1, rng.poisson(lam)))
            raise ValueError(f"DistribuciÃ³n de items no soportada: {dist}")

        if self.method == "kde":
            if self.kde_model is not None:
                random_state = None
                if isinstance(rng, np.random.Generator):
                    random_state = int(rng.integers(0, 2**32 - 1))
                try:
                    sample = self.kde_model.sample(1, random_state=random_state)
                    val = float(sample[0][0] if sample.ndim > 1 else sample[0])
                    return int(max(1, round(val)))
                except NotImplementedError:
                    pass
                except Exception:
                    pass
            if self.support is None or self.probs is None or len(self.support) == 0:
                return 1
            idx = int(rng.choice(len(self.support), p=self.probs))
            return int(max(1, self.support[idx]))

        raise ValueError(f"MÃ©todo de distribuciÃ³n desconocido: {self.method}")


def _parse_list_field(raw_value: str, dtype=float) -> Optional[np.ndarray]:
    if raw_value is None or str(raw_value).strip() == "":
        return None
    try:
        values = ast.literal_eval(str(raw_value))
    except Exception:
        return None
    return np.asarray(values, dtype=dtype)


_KDE_CLASS = None
_KDE_SUPPORTS_SAMPLE_WEIGHT = False


def _load_kernel_density_class():
    global _KDE_CLASS, _KDE_SUPPORTS_SAMPLE_WEIGHT
    if _KDE_CLASS is not None:
        return _KDE_CLASS
    try:
        from sklearn.neighbors import KernelDensity as _KD  # type: ignore

        params = inspect.signature(_KD.fit).parameters
        _KDE_SUPPORTS_SAMPLE_WEIGHT = "sample_weight" in params
        _KDE_CLASS = _KD
    except Exception:
        _KDE_CLASS = None
        _KDE_SUPPORTS_SAMPLE_WEIGHT = False
    return _KDE_CLASS


def _build_kde_model(
    support: np.ndarray,
    probs: np.ndarray,
    kernel: Optional[str],
    bandwidth: Optional[float],
):
    kd_cls = _load_kernel_density_class()
    if kd_cls is None:
        return None
    kernel = (kernel or "gaussian").strip().lower()
    bw = float(bandwidth) if bandwidth not in (None, "", np.nan) else 1.0
    bw = max(1e-3, bw)
    support = np.asarray(support, dtype=float).reshape(-1, 1)
    probs = probs / probs.sum()
    model = kd_cls(kernel=kernel, bandwidth=bw)
    if _KDE_SUPPORTS_SAMPLE_WEIGHT:
        model.fit(support, sample_weight=probs)
    else:
        counts = np.maximum(1, np.round(probs * 1000).astype(int))
        expanded = np.repeat(support[:, 0], counts)
        model.fit(expanded.reshape(-1, 1))
    return model


def _row_to_item_spec(row: Dict[str, Any]) -> ItemDistributionSpec:
    fit_type = (row.get("fit_type") or "").strip().lower()
    if fit_type == "parametric":
        dist = (row.get("fit_distribution") or "").strip().lower()
        params_raw = row.get("fit_params") or "{}"
        try:
            params = json.loads(params_raw)
        except Exception:
            params = {}
        return ItemDistributionSpec(method="parametric", dist_name=dist, params=params)

    if fit_type == "kde":
        support = _parse_list_field(row.get("kde_support"), dtype=int)
        probs = _parse_list_field(row.get("kde_probs"), dtype=float)
        if support is None or probs is None or len(support) != len(probs):
            raise ValueError("KDE sin soporte/probabilidades vÃ¡lidas")
        probs = np.maximum(probs.astype(float), 0.0)
        total = probs.sum()
        if total <= 0:
            probs = np.full_like(probs, 1.0 / len(probs))
        else:
            probs = probs / total
        kernel = (row.get("fit_kernel") or "").strip().lower() or "gaussian"
        bw_raw = row.get("fit_bandwidth")
        try:
            bandwidth = float(bw_raw) if bw_raw not in (None, "") else None
        except (TypeError, ValueError):
            bandwidth = None
        kde_model = _build_kde_model(support, probs, kernel, bandwidth)
        return ItemDistributionSpec(
            method="kde",
            support=support.astype(int),
            probs=probs,
            kernel=kernel,
            bandwidth=bandwidth,
            kde_model=kde_model,
        )

    raise ValueError("Tipo de ajuste de items no soportado")


def load_item_distributions(
    path: Path = ITEMS_SUMMARY_FILE,
) -> Dict[
    Tuple[CustomerProfile, PriorityType, Optional[PaymentMethod], Optional[DayType]],
    ItemDistributionSpec,
]:
    store: Dict[
        Tuple[
            CustomerProfile, PriorityType, Optional[PaymentMethod], Optional[DayType]
        ],
        ItemDistributionSpec,
    ] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                profile = _norm_enum(row.get("profile"), CustomerProfile)
                priority = _norm_enum(row.get("priority"), PriorityType)
                payment = _normalize_payment_value(row.get("payment_method"))
                day_type = _normalize_day_type_value(row.get("day_type"))
                spec = _row_to_item_spec(row)
                store[(profile, priority, payment, day_type)] = spec
            except Exception:
                continue
    if not store:
        raise RuntimeError(
            "No se pudieron cargar distribuciones de items desde items_distribution_summary.csv"
        )
    return store


ITEMS_DISTRIBUTIONS = load_item_distributions()


def _resolve_item_spec(
    profile: CustomerProfile,
    priority: Optional[PriorityType],
    payment: Optional[PaymentMethod],
    day_type: Optional[DayType],
) -> Optional[ItemDistributionSpec]:
    pay_candidates = [payment]
    if payment is not None:
        pay_candidates.append(None)
    day_candidates = [day_type]
    if day_type is not None:
        day_candidates.append(None)
    pri_candidates = [priority]
    if priority not in (None, PriorityType.NO_PRIORITY):
        pri_candidates.append(PriorityType.NO_PRIORITY)
    pri_candidates.append(None)

    for pr in pri_candidates:
        for pm in pay_candidates:
            for dy in day_candidates:
                spec = ITEMS_DISTRIBUTIONS.get((profile, pr, pm, dy))
                if spec:
                    return spec

    # Ãºltimo recurso: cualquier distribuciÃ³n del perfil
    for (prof_key, *_), spec in ITEMS_DISTRIBUTIONS.items():
        if prof_key == profile:
            return spec
    return None


# %%
def draw_items(
    profile,
    priority: Optional[PriorityType] = None,
    payment_method: Optional[PaymentMethod] = None,
    day_type: Optional[DayType] = None,
    rng=RNG_ITEMS,
) -> int:
    prof = (
        profile
        if isinstance(profile, CustomerProfile)
        else _norm_enum(profile, CustomerProfile)
    )
    prio = (
        priority
        if isinstance(priority, PriorityType) or priority is None
        else _norm_enum(priority, PriorityType)
    )
    pay = _normalize_payment_value(payment_method)
    day = _normalize_day_type_value(day_type)

    spec = _resolve_item_spec(prof, prio, pay, day)
    if spec is None:
        return int(max(1, rng.poisson(10.0)))

    return int(max(1, spec.sample(rng)))


# %% [markdown]
# ## Paciencia de los clientes

# %%
import re


class _PatienceRuleStore:
    def __init__(self):
        self.rules: Dict[
            Tuple[Optional[CustomerProfile], PriorityType, Optional[PaymentMethod]],
            Dict[str, Any],
        ] = {}

    def add(
        self,
        prof: Optional[CustomerProfile],
        prio: PriorityType,
        payment: Optional[PaymentMethod],
        spec: Dict[str, Any],
    ):
        self.rules[(prof, prio, payment)] = spec

    def find(
        self,
        prof: CustomerProfile,
        prio: PriorityType,
        payment: Optional[PaymentMethod],
    ) -> Optional[Dict[str, Any]]:
        candidates = [
            (prof, prio, payment),
            (prof, prio, None),
            (prof, PriorityType.NO_PRIORITY, payment),
            (prof, PriorityType.NO_PRIORITY, None),
            (None, prio, payment),
            (None, prio, None),
            (None, PriorityType.NO_PRIORITY, payment),
            (None, PriorityType.NO_PRIORITY, None),
        ]
        for key in candidates:
            spec = self.rules.get(key)
            if spec:
                return spec
        return None


def _fget(row, *names, default=None, cast=float):
    for n in names:
        if n in row and str(row[n]).strip() != "":
            try:
                return cast(row[n])
            except:
                pass
    return default


def _parse_params_str(s: str) -> dict:
    if s is None:
        return {}
    s = str(s).strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()
    # key=value
    pairs = re.findall(
        r"([a-zA-Z_Ã¡Ã©Ã­Ã³ÃºÃ±]+)\s*=?\s*([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", s
    )
    if pairs:
        out = {}
        for k, v in pairs:
            k = (
                k.lower()
                .replace("Ã¡", "a")
                .replace("Ã©", "e")
                .replace("Ã­", "i")
                .replace("Ã³", "o")
                .replace("Ãº", "u")
            )
            out[k] = float(v)
        # alias
        if "shape" in out and "k" not in out:
            out["k"] = out["shape"]
        if "scale" in out and "theta" not in out:
            out["theta"] = out["scale"]
        return out
    # tuple-only
    nums = re.findall(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)", s)
    if nums:
        return {"tuple": [float(x) for x in nums]}
    return {}


def _row_to_spec(row: Dict[str, Any]) -> Dict[str, Any]:
    # mÃ©todo desde mÃºltiples nombres de columna
    m = (
        row.get("method")
        or row.get("dist")
        or row.get("model")
        or row.get("mejor_dist")
        or row.get("distribucion")
        or row.get("distribucion_recomendada")
        or row.get("distribution")
        or ""
    )
    m = str(m).strip().lower()

    # parÃ¡metros desde mÃºltiples nombres de columna
    p_raw = (
        row.get("parameters")
        or row.get("parametros")
        or row.get("parÃ¡metros")
        or row.get("parametros_aprox")
        or ""
    )
    p = _parse_params_str(p_raw)

    # valores numÃ©ricos directos
    mean = _fget(row, "mean", "avg")
    std = _fget(row, "std", "sigma", "sd")
    lam = _fget(row, "lambda", "lam", "rate")
    shape = _fget(row, "shape", "alpha", "k")
    scale = _fget(row, "scale", "theta", "beta")
    loc = _fget(row, "loc", default=0.0)

    # normalizar alias de mÃ©todo
    if m in ("expon", "exponencial", "exponential", "exp"):
        m = "exponential"
    if m in ("weibull_min", "weibull", "weibullmin"):
        m = "weibull_min"
    if m in ("ga", "gamma"):
        m = "gamma"
    if m in ("ln", "lognorm", "lognormal"):
        m = "lognormal"
    if m in ("norm", "normal", "gaussian"):
        m = "normal"

    # completar parÃ¡metros faltantes desde p
    if "loc" in p:
        loc = float(p["loc"])
    if m == "weibull_min":
        k = float(p.get("k", shape or (p.get("tuple", [None, None])[0])))
        sc = float(p.get("scale", scale or (p.get("tuple", [None, None])[1])))
        if not (k and sc):  # invÃ¡lido â†’ fallback
            return {"method": "fixed", "value": 300.0}
        return {
            "method": "weibull_min",
            "k": max(1e-9, k),
            "scale": max(1e-9, sc),
            "loc": float(loc),
        }

    if m == "exponential" or lam is not None or (mean and not std):
        # expon: lambda o scale; si viene tupla, (loc, scale) o (scale)
        if "tuple" in p:
            if len(p["tuple"]) == 2:
                loc = p["tuple"][0]
                scale = p["tuple"][1]
            elif len(p["tuple"]) == 1:
                scale = p["tuple"][0]
        if lam:
            scale = 1.0 / max(1e-9, lam)
        elif scale is None:
            scale = float(mean) if mean else 600.0
        return {"method": "exponential", "scale": float(scale), "loc": float(loc)}

    if m == "gamma" or (shape and scale) or ("tuple" in p and len(p["tuple"]) >= 2):
        if "tuple" in p and (shape is None or scale is None):
            shape = shape or p["tuple"][0]
            scale = scale or p["tuple"][1]
        return {
            "method": "gamma",
            "shape": float(shape),
            "scale": float(scale),
            "loc": float(loc),
        }

    if m == "lognormal":
        mu = _fget(row, "mu", "lognorm_mu", "mu_log")
        sg = _fget(row, "sigma", "lognorm_sigma", "sigma_log")
        if "tuple" in p and (mu is None or sg is None):
            # si solo viene sigma, no hay mu â†’ imposible estimar: fallback
            if len(p["tuple"]) == 2:
                mu, sg = p["tuple"][0], p["tuple"][1]
        if mu is None or sg is None:
            return {"method": "fixed", "value": 300.0}
        return {
            "method": "lognormal",
            "mu": float(mu),
            "sigma": float(max(1e-9, sg)),
            "loc": float(loc),
        }

    if m == "normal" or (mean is not None and std is not None):
        return {
            "method": "normal",
            "mean": float(mean),
            "std": float(max(1e-6, std)),
            "loc": float(loc),
        }

    # Ãºltimo recurso: valor fijo
    val = _fget(row, "fixed", "value", "median", "mode", default=300.0)
    return {"method": "fixed", "value": float(val)}


def _load_patience_rules_from_csv(path: str | Path) -> _PatienceRuleStore:
    store = _PatienceRuleStore()
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"No se encuentra {path_obj}")
    _announce_once("Cargando distribuciones de paciencia (CSV)", path_obj)
    with open(path_obj, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                prio = _norm_enum(row.get("priority"), PriorityType)
            except Exception:
                continue
            prof_col = row.get("profile", "").strip()
            prof = None
            if prof_col:
                try:
                    prof = _norm_enum(prof_col, CustomerProfile)
                except Exception:
                    prof = None
            payment = _normalize_payment_value(row.get("payment_method"))
            try:
                spec = _row_to_spec(row)
            except Exception:
                continue
            store.add(prof, prio, payment, spec)
    return store


# %%
@dataclass
class PatienceDistributionExponential:
    def sample(
        self,
        profile: CustomerProfile,
        priority: PriorityType,
        payment: Optional[PaymentMethod],
        day_type: Optional[DayType] = None,
    ) -> float:
        return float(max(0, np.random.exponential(600.0)))


@dataclass
class PatienceDistributionCSV:
    source_file: Path = PATIENCE_BASE_CSV_FILE
    _store: _PatienceRuleStore = None

    def __post_init__(self):
        self._store = _load_patience_rules_from_csv(self.source_file)

    def sample(
        self,
        profile: CustomerProfile,
        priority: PriorityType,
        payment: Optional[PaymentMethod],
        day_type: Optional[DayType] = None,
    ) -> float:
        payment_norm = _normalize_payment_value(payment)
        spec = self._store.find(profile, priority, payment_norm)
        if not spec:
            return float(max(0, np.random.exponential(600.0)))
        m = spec["method"]
        if m == "exponential":
            val = np.random.exponential(max(1e-9, spec.get("scale", 600.0))) + spec.get(
                "loc", 0.0
            )
        elif m == "gamma":
            val = np.random.gamma(
                max(1e-9, spec["shape"]), max(1e-9, spec["scale"])
            ) + spec.get("loc", 0.0)
        elif m == "lognormal":
            val = np.random.lognormal(spec["mu"], max(1e-9, spec["sigma"])) + spec.get(
                "loc", 0.0
            )
        elif m == "normal":
            val = np.random.normal(spec["mean"], max(1e-6, spec["std"])) + spec.get(
                "loc", 0.0
            )
        elif m == "weibull_min":
            # numpy: np.random.weibull(k) * scale  (loc se suma aparte)
            val = (
                np.random.weibull(max(1e-9, spec["k"])) * max(1e-9, spec["scale"])
            ) + spec.get("loc", 0.0)
        else:
            val = float(spec.get("value", 300.0))
        return float(max(0, val))


@dataclass
class PatienceDistributionTable:
    source_file: Path = PATIENCE_DISTRIBUTION_FILE
    fallback: PatienceDistributionCSV | None = None
    rng: np.random.Generator = field(
        default_factory=lambda: np.random.default_rng(2025)
    )
    _entries: dict = None
    _cache: dict = None

    def __post_init__(self):
        if self.fallback is None:
            try:
                self.fallback = PatienceDistributionCSV()
            except FileNotFoundError:
                self.fallback = PatienceDistributionExponential()
        self._entries = self._load_entries(self.source_file)
        self._cache = {}

    @staticmethod
    def _parse_params(value) -> list[float]:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        try:
            data = json.loads(value)
            if isinstance(data, list):
                return [float(p) for p in data]
        except Exception:
            pass
        try:
            return [
                float(part.strip()) for part in str(value).split(",") if part.strip()
            ]
        except Exception:
            return []

    @staticmethod
    def _build_kde_entry(subset: pd.DataFrame) -> Optional[dict]:
        subset = subset.sort_values("x_seconds")
        x = subset["x_seconds"].to_numpy(dtype=float)
        y = subset["density"].to_numpy(dtype=float)
        if len(x) < 2 or np.all(y <= 0):
            return None
        cdf = np.zeros_like(x)
        total = 0.0
        for idx in range(1, len(x)):
            dx = max(0.0, x[idx] - x[idx - 1])
            trap = 0.5 * max(0.0, y[idx] + y[idx - 1]) * dx
            total += trap
            cdf[idx] = total
        if total <= 0:
            return None
        cdf = cdf / total
        cdf[-1] = 1.0
        return {"type": "kde", "x": x, "cdf": cdf}

    @staticmethod
    def _load_entries(path: Path) -> dict:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"No existe el archivo de distribuciones de paciencia: {path}"
            )
        df = pd.read_csv(path)
        required_cols = {"profile", "priority", "payment_method", "day_type", "method"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"El archivo {path} no contiene las columnas requeridas {required_cols}"
            )
        for col in ["profile", "priority", "payment_method", "day_type", "method"]:
            df[col] = df[col].astype(str).str.strip().str.lower()
        df["payment_method"] = df["payment_method"].apply(
            lambda v: _norm_payment(_normalize_payment_value(v))
        )
        df["day_type"] = df["day_type"].apply(_norm_day_type)
        table = {}
        for (prof, prio, pay, day), grp in df.groupby(
            ["profile", "priority", "payment_method", "day_type"]
        ):
            if (grp["method"] == "param").any():
                row = grp[grp["method"] == "param"].iloc[0]
                dist = str(row.get("distribution", "")).strip().lower()
                params = PatienceDistributionTable._parse_params(row.get("params"))
                if dist and params:
                    table[(prof, prio, pay, day)] = {
                        "type": "param",
                        "distribution": dist,
                        "params": params,
                    }
                    continue
            kde_grp = grp[grp["method"] == "kde"]
            if not kde_grp.empty:
                entry = PatienceDistributionTable._build_kde_entry(kde_grp)
                if entry:
                    table[(prof, prio, pay, day)] = entry
        return table

    def sample(
        self,
        profile: CustomerProfile,
        priority: PriorityType,
        payment: Optional[PaymentMethod],
        day_type: Optional[DayType] = None,
    ) -> float:
        prof = _norm_profile(profile)
        prio = _norm_priority(priority)
        pay = _norm_payment(_normalize_payment_value(payment))
        day = _norm_day_type(day_type)
        prio_default = _norm_priority(PriorityType.NO_PRIORITY)
        cache_key = (prof, prio, pay, day)
        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if entry is None:
                return self.fallback.sample(profile, priority, payment, day_type)
            return self._sample_from_entry(entry, profile, priority, payment, day_type)
        candidates = [
            (prof, prio, pay, day),
            (prof, prio, "", day),
            (prof, prio, pay, ""),
            (prof, prio, "", ""),
            (prof, prio_default, pay, day),
            (prof, prio_default, pay, ""),
        ]
        for key in candidates:
            entry = self._entries.get(key)
            if entry:
                break
        else:
            self._cache[cache_key] = None
            return self.fallback.sample(profile, priority, payment, day_type)

        self._cache[cache_key] = entry
        return self._sample_from_entry(entry, profile, priority, payment, day_type)

    def _sample_from_entry(
        self,
        entry: dict,
        profile: CustomerProfile,
        priority: PriorityType,
        payment: Optional[PaymentMethod],
        day_type: Optional[DayType],
    ) -> float:
        if entry["type"] == "param":
            dist = getattr(stats, entry["distribution"], None)
            params = entry.get("params") or []
            if dist and params:
                try:
                    val = dist.rvs(*params, random_state=self.rng)
                    return float(max(0.0, val))
                except Exception:
                    pass
        elif entry["type"] == "kde":
            x = entry["x"]
            cdf = entry["cdf"]
            if (
                isinstance(x, np.ndarray)
                and isinstance(cdf, np.ndarray)
                and x.size >= 2
            ):
                u = self.rng.random()
                val = np.interp(u, cdf, x)
                return float(max(0.0, val))
        return self.fallback.sample(profile, priority, payment, day_type)


def _build_patience_sampler():
    try:
        base_fallback = PatienceDistributionCSV()
        base_csv_path = Path(base_fallback.source_file)
        if base_csv_path.exists():
            _announce_once("Usando CSV base de paciencia", base_csv_path)
    except FileNotFoundError:
        _warn(
            "No se encontro el CSV base de paciencia; se usara exponencial por defecto"
        )
        base_fallback = PatienceDistributionExponential()
    if PATIENCE_DISTRIBUTION_FILE.exists():
        try:
            global _PATIENCE_TABLE_ANNOUNCED
            if not _PATIENCE_TABLE_ANNOUNCED:
                _announce_once(
                    "Usando distribuciones de paciencia reconstruidas",
                    PATIENCE_DISTRIBUTION_FILE,
                )
                _PATIENCE_TABLE_ANNOUNCED = True
            return PatienceDistributionTable(
                PATIENCE_DISTRIBUTION_FILE, fallback=base_fallback
            )
        except Exception as exc:
            _warn(
                f"No se pudo cargar el archivo de paciencia; se usara el CSV tradicional ({exc})",
                PATIENCE_DISTRIBUTION_FILE,
            )
    else:
        _warn(
            "Archivo de paciencia reconstruido no encontrado, se usara el CSV tradicional",
            PATIENCE_DISTRIBUTION_FILE,
        )
    return base_fallback


@dataclass
class ProfileConfig:
    profile: CustomerProfile

    patience_distribution: PatienceDistributionCSV = field(
        default_factory=_build_patience_sampler
    )

    items_sampler: callable = field(
        default=lambda prof, priority, payment, day_type, rng=RNG_ITEMS: draw_items(
            prof, priority, payment, day_type, rng
        )
    )

    def create_arrival_distribution(
        self,
        day_type: DayType,
        priority: PriorityType,
        payment_method: PaymentMethod,
        total_customers: int = 0,
    ) -> ArrivalDistribution:
        return ArrivalDistribution(
            self.profile, day_type, priority, payment_method, total_customers
        )


# %% [markdown]
# ## Tiempo de servicio por cliente

# %%
_PRIORITY_QUEUE_RANK = {
    PriorityType.REDUCED_MOBILITY: 0,
    PriorityType.SENIOR: 0,
    PriorityType.PREGNANT: 0,
    PriorityType.NO_PRIORITY: 5,
}


def _priority_request_value(priority: PriorityType) -> int:
    return _PRIORITY_QUEUE_RANK.get(
        priority, _PRIORITY_QUEUE_RANK[PriorityType.NO_PRIORITY]
    )


class CheckoutLane:
    def __init__(self, env, lane_id, lane_type: LaneType, capacity: int = 1):
        self.env = env
        self.lane_id = lane_id
        self.lane_type = lane_type
        self.capacity = max(1, int(capacity))
        if lane_type is LaneType.PRIORITY:
            self.servidor = simpy.PriorityResource(env, capacity=1)
        else:
            self.servidor = simpy.Resource(env, capacity=self.capacity)


def elegible(cliente, lane: CheckoutLane) -> bool:
    pr = cliente["priority"]
    pm = cliente["payment_method"]
    items = int(cliente["items"])
    if lane.lane_type == LaneType.EXPRESS and items > 10:
        return False
    if lane.lane_type == LaneType.SCO and (
        pm != PaymentMethod.CARD or pr == PriorityType.REDUCED_MOBILITY or items > 15
    ):
        return False
    return True


EXPRESS_PROFILE_WEIGHT = {
    CustomerProfile.EXPRESS_BASKET: 0.9,
    CustomerProfile.DEAL_HUNTER: 0.6,
    CustomerProfile.SELF_CHECKOUT_FAN: 0.4,
}
DEFAULT_EXPRESS_WEIGHT = 0.3
SLOW_PROFILES = {
    CustomerProfile.FAMILY_CART,
    CustomerProfile.WEEKLY_PLANNER,
}

# %%
_DIST = {
    "normal": stats.norm,
    "lognorm": stats.lognorm,
    "gamma": stats.gamma,
    "weibull_min": stats.weibull_min,
}


def _coerce_bool(x):
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    return (
        True
        if s in {"true", "1", "yes"}
        else False if s in {"false", "0", "no"} else None
    )


def _pick_best(df: pd.DataFrame) -> pd.DataFrame:
    # Elige por mayor p-valor KS, luego menor KS. Ignora filas con error o sin params.
    work = df.copy()
    if "reject_H0" in work.columns:
        work["reject_H0"] = work["reject_H0"].map(_coerce_bool)
    for c in ["ks_pvalue", "ks_stat", "shape", "loc", "scale"]:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work[work["distribution"].isin(_DIST.keys())]
    work = work[work["loc"].notna() & work["scale"].notna()]
    # preferir no rechazadas si existen; si no, considerar todas
    subset = work
    if "reject_H0" in work.columns and work["reject_H0"].eq(False).any():
        subset = work[work["reject_H0"].eq(False)]
    subset = subset.sort_values(["ks_pvalue", "ks_stat"], ascending=[False, True])
    return subset.head(1)


def _norm_profile(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower()


def _norm_priority(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower()


def _norm_payment(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower()


def _norm_day_type(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value).strip().lower()


def _norm_lane(value) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    s = str(value).strip().lower()
    if s in {"", "nan"}:
        return ""
    s = LANE_NAME_NORMALIZATION.get(s, s)
    return s


SERVICE_TIME_CATEGORY_COLUMNS = [
    "profile",
    "priority",
    "payment_method",
    "lane_type",
    "day_type",
]


@dataclass
class _ServiceTimeResidualDistribution:
    name: str
    params: tuple[float, ...]

    def sample(self, rng: np.random.Generator) -> float:
        dist = getattr(stats, self.name, stats.norm)
        try:
            return float(dist.rvs(*self.params, random_state=rng))
        except Exception:
            return float(rng.normal(0.0, 1.0))

    def mean(self) -> float:
        dist = getattr(stats, self.name, stats.norm)
        try:
            value = dist.mean(*self.params)
            if isinstance(value, (tuple, list, np.ndarray)):
                value = value[0]
            return float(value)
        except Exception:
            return 0.0


@dataclass
class _ServiceTimeResidualBucket:
    prob_outlier: float
    inlier: _ServiceTimeResidualDistribution
    outlier: Optional[_ServiceTimeResidualDistribution] = None

    def sample(self, rng: np.random.Generator) -> float:
        p = float(np.clip(self.prob_outlier, 0.0, 1.0))
        if self.outlier is not None and rng.random() < p:
            return self.outlier.sample(rng)
        return self.inlier.sample(rng)

    def mean(self) -> float:
        p = float(np.clip(self.prob_outlier, 0.0, 1.0))
        inlier_mean = self.inlier.mean()
        outlier_mean = self.outlier.mean() if self.outlier else inlier_mean
        return (1.0 - p) * inlier_mean + p * outlier_mean


class ServiceTimeFactorModel:
    CATEGORY_NORMALIZERS = {
        "profile": _norm_profile,
        "priority": _norm_priority,
        "payment_method": _norm_payment,
        "lane_type": _norm_lane,
        "day_type": _norm_day_type,
    }

    def __init__(
        self,
        model_path: Path,
        multipliers: Optional[dict[tuple[str, str], float]] = None,
        rng_seed: int = 2025,
    ):
        with open(model_path, encoding="utf-8") as fh:
            data = json.load(fh)
        coeffs = data.get("coefficients", {})
        self.intercept = float(coeffs.get("intercept", 0.0))
        self.items_coef = float(coeffs.get("items", 0.0))
        self.categories: dict[str, dict] = {}
        for col, entry in coeffs.get("categories", {}).items():
            baseline = str(entry.get("baseline", "")).strip().lower()
            coeff_map = {
                str(k).strip().lower(): float(v)
                for k, v in entry.get("coeffs", {}).items()
            }
            self.categories[col] = {"baseline": baseline, "coeffs": coeff_map}
        self.residual_models: dict[tuple[str, ...], _ServiceTimeResidualBucket] = {}
        for key, entry in data.get("residual_models", {}).items():
            combo = tuple(part.strip().lower() for part in key.split("|"))
            self.residual_models[combo] = self._build_bucket(entry)
        self.defaults = self._build_bucket(data.get("defaults"))
        self.rng = np.random.default_rng(rng_seed)
        self.multipliers = multipliers or {}

    def _build_distribution(
        self, entry: Optional[dict]
    ) -> Optional[_ServiceTimeResidualDistribution]:
        if not entry:
            return None
        params = tuple(float(p) for p in entry.get("params", []))
        name = str(entry.get("name", "norm")).strip().lower()
        return _ServiceTimeResidualDistribution(name=name, params=params)

    def _build_bucket(self, entry: Optional[dict]) -> _ServiceTimeResidualBucket:
        prob = 0.0
        inlier = None
        outlier = None
        if entry:
            prob = float(entry.get("prob_outlier", 0.0))
            inlier = self._build_distribution(entry.get("inlier"))
            outlier = self._build_distribution(entry.get("outlier"))
        if inlier is None:
            inlier = _ServiceTimeResidualDistribution("norm", (0.0, 1.0))
        return _ServiceTimeResidualBucket(
            prob_outlier=prob, inlier=inlier, outlier=outlier
        )

    def _combo_key(
        self,
        profile,
        priority,
        payment_method,
        day_type,
        lane_type,
    ) -> tuple[str, ...]:
        return (
            _norm_profile(profile),
            _norm_priority(priority),
            _norm_payment(payment_method),
            _norm_day_type(day_type),
            _norm_lane(lane_type),
        )

    def _lookup_bucket(
        self, profile, priority, payment_method, day_type, lane_type
    ) -> _ServiceTimeResidualBucket:
        key = self._combo_key(profile, priority, payment_method, day_type, lane_type)
        return self.residual_models.get(key, self.defaults)

    def _category_adjustment(self, column: str, value) -> float:
        info = self.categories.get(column)
        if not info:
            return 0.0
        normalizer = self.CATEGORY_NORMALIZERS.get(
            column, lambda x: str(x or "").strip().lower()
        )
        norm_value = normalizer(value)
        return float(info["coeffs"].get(norm_value, 0.0))

    def _base_prediction(
        self, profile, priority, payment_method, day_type, lane_type, items: float
    ) -> float:
        total = self.intercept + self.items_coef * float(max(0.0, items))
        context = {
            "profile": profile,
            "priority": priority,
            "payment_method": payment_method,
            "lane_type": lane_type,
            "day_type": day_type,
        }
        for column in SERVICE_TIME_CATEGORY_COLUMNS:
            total += self._category_adjustment(column, context[column])
        return float(max(0.0, total))

    def _multiplier_for(self, lane_type, profile) -> float:
        if not self.multipliers:
            return 1.0
        lane = _norm_lane(lane_type)
        prof = _norm_profile(profile)
        return float(self.multipliers.get((lane, prof), 1.0))

    def expected_value(
        self, *, profile, priority, payment_method, day_type, lane_type, items: float
    ) -> float:
        base = self._base_prediction(
            profile, priority, payment_method, day_type, lane_type, items
        )
        bucket = self._lookup_bucket(
            profile, priority, payment_method, day_type, lane_type
        )
        value = float(max(0.0, base + bucket.mean()))
        return value * self._multiplier_for(lane_type, profile)

    def sample(
        self,
        *,
        profile,
        priority,
        payment_method,
        day_type,
        lane_type,
        items: float,
        rng: Optional[np.random.Generator] = None,
    ) -> float:
        generator = rng or self.rng
        base = self._base_prediction(
            profile, priority, payment_method, day_type, lane_type, items
        )
        bucket = self._lookup_bucket(
            profile, priority, payment_method, day_type, lane_type
        )
        value = float(max(0.0, base + bucket.sample(generator)))
        return value * self._multiplier_for(lane_type, profile)


SERVICE_TIME_MODEL_JSON = PROJECT_ROOT / "service_time/service_time_model.json"
SERVICE_TIME_MODEL: Optional[ServiceTimeFactorModel] = None
if SERVICE_TIME_MODEL_JSON.exists():
    try:
        SERVICE_TIME_MODEL = ServiceTimeFactorModel(
            SERVICE_TIME_MODEL_JSON,
            multipliers=_SERVICE_TIME_MULTIPLIERS,
        )
        _announce_once("Modelo de tiempos de servicio cargado", SERVICE_TIME_MODEL_JSON)
    except Exception as exc:
        _warn(f"No se pudo cargar {SERVICE_TIME_MODEL_JSON.name}: {exc}")
        SERVICE_TIME_MODEL = None
else:
    _warn(
        "No existe modelo de tiempos de servicio; se usara configuracion por defecto",
        SERVICE_TIME_MODEL_JSON,
    )
    SERVICE_TIME_MODEL = None


# %% [markdown]
# ## Balking


# %%
class QueueCapBalkModel:
    """
    Balking determinÃ­stico basado en un mÃ¡ximo de largo de cola efectivo por perfil.
    Si la cola supera el cap configurado para el perfil, el cliente se va inmediatamente.
    """

    def __init__(self, caps: dict[CustomerProfile, int], default_cap: int = 1):
        self.caps = {prof: int(max(0, cap)) for prof, cap in caps.items()}
        self.default_cap = int(max(0, default_cap))

    def _cap_for(self, profile: CustomerProfile) -> int:
        return self.caps.get(profile, self.default_cap)

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
        cap = self._cap_for(profile)
        eff_queue = max(0, int(queue_len))
        return 1.0 if eff_queue > cap else 0.0

    def decide(self, *args, **kwargs) -> bool:
        return self.prob_balk(*args, **kwargs) >= 1.0


# %%
balk_model = QueueCapBalkModel(
    caps={
        CustomerProfile.FAMILY_CART: 10**6,
        CustomerProfile.WEEKLY_PLANNER: 10**6,
    },
    default_cap=1,
)

BALK_MODEL = balk_model


# %% [markdown]
# ## RelaciÃ³n de profit

# %%
# === Celda: parÃ¡metros de profit con coeficientes por Ã­tems ===
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROFIT_FILE = "profit_analysis_summary.csv"

# Coeficientes de pendiente por Ã­tem cuando no viene en el CSV (CLP por Ã­tem)
PROFILE_COEF_ITEMS = {
    "deal_hunter": 196.951060,
    "regular": 248.6974,
    "weekly_planner": 248.6974,
    "family_cart": 257.442983,
    "self_checkout_fan": 277.252190,
    "express_basket": 280.162367,
}


# Mapea DayType -> etiqueta de "grupo" usada en el CSV
def _normalize_day_label(day_type):
    if isinstance(day_type, DayType):
        return day_type
    text = str(day_type).strip().lower()
    if "tipo 1" in text or text == "1":
        return DayType.TYPE_1
    if "tipo 2" in text or text == "2":
        return DayType.TYPE_2
    if "tipo 3" in text or text == "3":
        return DayType.TYPE_3
    return None


def _safe_float(value, default=np.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_profit_betas(csv_path: str | Path) -> dict:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontro archivo de profit: {path}")
    data: dict[tuple[str, str, str, DayType], dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            profile = row.get("profile", "").strip()
            priority = row.get("priority", "").strip()
            payment = row.get("payment_method", "").strip().lower()
            day_label = _normalize_day_label(row.get("day_type", ""))
            if not (profile and priority and payment and day_label):
                continue
            try:
                beta = float(row.get("beta", ""))
            except (TypeError, ValueError):
                continue
            entry = {
                "beta": beta,
                "beta_ci_low": _safe_float(row.get("beta_ci_low")),
                "beta_ci_high": _safe_float(row.get("beta_ci_high")),
                "r2": _safe_float(row.get("r2")),
                "mae": _safe_float(row.get("mae")),
                "rmse": _safe_float(row.get("rmse")),
                "n_obs": _safe_float(row.get("n_obs"), default=0.0),
            }
            data[(profile, priority, payment, day_label)] = entry
            # fallback sin distinguir pago
            if (profile, priority, "", day_label) not in data:
                data[(profile, priority, "", day_label)] = entry
    if not data:
        raise RuntimeError(
            "No se pudieron cargar betas desde profit_analysis_summary.csv"
        )
    _announce_once("Cargando betas de profit", path)
    return data


PROFIT_BETAS = _load_profit_betas(PROFIT_FILE)


def _resolve_profit_entry(
    profile_key: str,
    priority_key: str,
    payment_key: str,
    day_type: DayType,
    profit_data: dict,
) -> Optional[dict]:
    candidates = []
    if payment_key:
        candidates.append((profile_key, priority_key, payment_key, day_type))
    candidates.append((profile_key, priority_key, "", day_type))
    for key in candidates:
        entry = profit_data.get(key)
        if entry:
            return entry
    return None


def _profit_noise(entry: dict, rng: Optional[np.random.Generator]) -> float:
    std = entry.get("rmse")
    if not (np.isfinite(std) and std > 0):
        std = entry.get("mae")
        if np.isfinite(std) and std > 0:
            std = std * math.sqrt(math.pi / 2.0)
    if not (np.isfinite(std) and std > 0):
        return 0.0
    rng = rng or RNG_PROFIT
    if isinstance(rng, np.random.Generator):
        return float(rng.normal(0.0, std))
    return float(np.random.normal(0.0, std))


def sample_profit(
    items: int,
    profile,
    priority,
    payment_method: str,
    day_type,
    profit_dict: dict = PROFIT_BETAS,
) -> float:
    """
    Estima profit como beta(items) mÃ¡s ruido normal segÃºn el RMSE observado.
    """
    items_count = max(0, int(items))
    if items_count <= 0:
        return 0

    day_norm = _normalize_day_label(day_type) or DayType.TYPE_1
    payment_txt = str(getattr(payment_method, "value", payment_method)).strip().lower()
    profile_key = getattr(profile, "value", str(profile)).strip()
    priority_key = getattr(priority, "value", str(priority)).strip()

    entry = _resolve_profit_entry(
        profile_key, priority_key, payment_txt, day_norm, profit_dict
    )
    beta = entry["beta"] if entry else np.nan
    if not (np.isfinite(beta) and beta > 0):
        beta = PROFILE_COEF_ITEMS.get(profile_key, np.nan)
    if not (np.isfinite(beta) and beta > 0):
        return int(round(max(0, items_count)))

    profit = beta * items_count
    if entry:
        profit += _profit_noise(entry, RNG_PROFIT)
    return int(round(max(0.0, profit)))


# %%
# === Celda: integraciÃ³n en el evento de checkout ===
# supuestos: objeto customer con atributos .profile y .items
if "RNG_PROFIT" not in globals():
    RNG_PROFIT = np.random.default_rng(42)


def finalize_customer_profit(
    customer, *, profit_dict=PROFIT_BETAS, default_day_type=DayType.TYPE_1
):
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
# df_events: columnas mÃ­nimas ['profile','items'] y opcionales ['priority','payment_method','day_type']
def compute_profit_column(
    df_events: pd.DataFrame,
    *,
    profit_dict: dict = PROFIT_BETAS,
    default_priority=PriorityType.NO_PRIORITY,
    default_payment=PaymentMethod.CARD,
    default_day_type=DayType.TYPE_1,
) -> pd.Series:
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


# %% [markdown]
# # GeneraciÃ³n de clientes y simulaciÃ³n

# %%
# ===================== CSV schemas =====================
CUSTOMERS_FIELDS = [
    "source_folder",
    "customer_id",
    "profile",
    "priority",
    "items",
    "payment_method",
    "arrival_time_s",
    "service_start_s",
    "service_end_s",
    "wait_time_s",
    "service_time_s",
    "total_time_s",
    "arrival_hour",
    "total_revenue_clp",
    "cart_categories",
    "cart_category_revenue_clp",
    "cart_items",
    "total_profit_clp",
    "cart_category_profit_clp",
    "lane_name",
    "lane_type",
    "queue_request_priority",
    "patience_s",
    "outcome",
    "balk_reason",
    "abandon_reason",
    "effective_queue_length",
]
TIMELOG_FIELDS = [
    "source_folder",
    "timestamp_s",
    "event_type",
    "customer_id",
    "profile",
    "priority",
    "items",
    "payment_method",
    "lane_name",
    "lane_type",
    "effective_queue_length",
    "queue_request_priority",
    "patience_s",
    "service_time_s",
    "revenue_clp",
    "profit_clp",
    "reason",
]


def _fill_missing(rows, fields):
    for r in rows:
        for k in fields:
            r.setdefault(k, "")


def _norm_generic(value) -> str:
    if hasattr(value, "value"):
        value = value.value
    if value is None:
        return ""
    return str(value).strip().lower()


def _load_balking_thresholds(path: Path) -> dict[tuple[str, str, str, str, str], float]:
    path = Path(path)
    if not path.exists():
        _warn(
            "No se encontro archivo de umbrales de balking; se usara sin limites teoricos",
            path,
        )
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        _warn(f"No se pudieron leer umbrales de balking ({exc})", path)
        return {}
    _announce_once("Cargando umbrales de balking", path)
    thresholds: dict[tuple[str, str, str, str, str], float] = {}
    for row in df.itertuples(index=False):
        limit = getattr(row, "estimated_threshold", None)
        if pd.isna(limit):
            continue
        key = (
            _norm_generic(getattr(row, "profile", "")),
            _norm_generic(getattr(row, "priority", "")),
            _norm_generic(getattr(row, "payment_method", "")),
            _norm_generic(getattr(row, "day_type", "")),
            _norm_generic(getattr(row, "lane_type", "")),
        )
        thresholds[key] = float(limit)
    return thresholds


BALKING_DAYTYPE_FACTORS_FILE = PROJECT_ROOT / "balking/balking_daytype_factors.csv"


def _load_balking_daytype_factors(path: Path) -> dict[str, float]:
    path = Path(path)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        _warn(f"No se pudieron leer factores por tipo de dia ({exc})", path)
        return {}
    factors: dict[str, float] = {}
    for row in df.itertuples(index=False):
        day = _norm_generic(getattr(row, "day_type", ""))
        try:
            factor = float(getattr(row, "multiplier", 1.0))
        except (TypeError, ValueError):
            continue
        if not day:
            continue
        factors[day] = max(0.0, factor)
    if factors:
        _announce_once("Factores de balking por tipo de dia cargados", path)
    return factors


BALKING_THRESHOLDS = _load_balking_thresholds(
    PROJECT_ROOT / "balking/balking_thresholds.csv"
)
BALKING_DAYTYPE_FACTORS = _load_balking_daytype_factors(BALKING_DAYTYPE_FACTORS_FILE)


def lookup_balking_threshold(
    profile,
    priority,
    payment_method,
    day_type,
    lane_type,
) -> Optional[float]:
    if not BALKING_THRESHOLDS:
        return None
    prof = _norm_generic(profile)
    prio = _norm_generic(priority)
    pay = _norm_generic(payment_method)
    day = _norm_generic(day_type)
    lane = _norm_generic(lane_type)
    lane_candidates = []
    if lane:
        lane_candidates.extend([lane, ""])
    else:
        lane_candidates.append("")
    keys: list[tuple[str, str, str, str, str]] = []
    for lane_opt in lane_candidates:
        keys.extend(
            [
                (prof, prio, pay, day, lane_opt),
                (prof, prio, pay, "", lane_opt),
                (prof, prio, "", "", lane_opt),
                (prof, "", "", "", lane_opt),
            ]
        )
    for key in keys:
        limit = BALKING_THRESHOLDS.get(key)
        if limit is not None:
            if day:
                factor = BALKING_DAYTYPE_FACTORS.get(day, 1.0)
                return float(limit) * float(factor or 1.0)
            return limit
    return None


# %%
def spawn_customer(
    env, lanes, cliente, time_log, customers_rows, balk_model=None, day_type=None
):
    """
    arrival_time_s relativo a 08:00 (desde la apertura).
    balk_model: instancia con mÃ©todos prob_balk/decide (p. ej., QueueCapBalkModel) o None.
    profit_dict: tabla de betas cargada con _load_profit_betas().
    day_type: enum DayType del dÃ­a en curso.
    """
    yield env.timeout(cliente["arrival_time_s"])

    # --- NUEVO: asegurar items antes de cualquier uso ---
    if (
        "items" not in cliente
        or not isinstance(cliente["items"], (int, float))
        or cliente["items"] <= 0
    ):
        try:
            k = draw_items(
                cliente.get("profile"),
                cliente.get("priority"),
                cliente.get("payment_method"),
                cliente.get("day_type") or day_type,
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
        time_log.append(
            {
                "source_folder": cliente["source_folder"],
                "timestamp_s": env.now,
                "event_type": "balk",
                "customer_id": cliente["customer_id"],
                "profile": as_str(cliente["profile"]),
                "priority": as_str(cliente["priority"]),
                "items": cliente["items"],
                "payment_method": as_str(cliente["payment_method"]),
                "lane_name": "",
                "lane_type": "",
                "effective_queue_length": 0,
                "queue_request_priority": as_str(cliente["priority"]),
                "patience_s": cliente["patience_s"],
                "service_time_s": 0,
                "revenue_clp": 0,
                "profit_clp": 0,
                "reason": reason,
            }
        )
        registro = cliente.copy()
        registro.update(
            {
                "service_start_s": "",
                "service_end_s": "",
                "wait_time_s": "",
                "service_time_s": "",
                "total_time_s": "",
                "arrival_hour": int((OPEN_S + cliente["arrival_time_s"]) // 3600),
                "lane_name": "",
                "lane_type": "",
                "queue_request_priority": as_str(cliente["priority"]),
                "outcome": "balk",
                "balk_reason": reason,
                "abandon_reason": "",
                "effective_queue_length": 0,
            }
        )
        customers_rows.append(registro)
        return

    def cola_efectiva(lane: "CheckoutLane"):
        en_queue = len(lane.servidor.queue)
        if lane.lane_type is LaneType.SCO:
            cap = getattr(lane, "capacity", 1)
            en_servicio = min(getattr(lane.servidor, "count", 0), cap)
            total = en_queue + en_servicio
            return max(0, total - cap)
        en_servicio = 1 if getattr(lane.servidor, "count", 0) > 0 else 0
        return en_queue + en_servicio

    def estado_lane(lane: "CheckoutLane") -> int:
        cap = getattr(lane, "capacity", 1)
        busy = min(getattr(lane.servidor, "count", 0), cap)
        queue_len = len(lane.servidor.queue)
        if queue_len == 0 and busy == 0:
            return 0  # caja totalmente vacÃ­a
        if queue_len == 0 and 0 < busy < cap:
            return 1  # hay servicio pero queda un puesto libre
        return 2  # resto de casos

    def tiempo_estimado_espera(lane: "CheckoutLane") -> float:
        cap = max(1, getattr(lane, "capacity", 1))
        ahead = max(0.0, float(lane._eff))
        servicio_prom = float(max(1.0, 15.0 + 4.0 * int(cliente["items"])))
        if "SERVICE_TIME_MODEL" in globals():
            try:
                servicio_prom = SERVICE_TIME_MODEL.expected_value(
                    profile=cliente["profile"],
                    items=int(cliente["items"]),
                    lane_type=lane.lane_type,
                    priority=cliente["priority"],
                    payment_method=cliente["payment_method"],
                    day_type=cliente.get("day_type"),
                )
                if not (np.isfinite(servicio_prom) and servicio_prom > 0):
                    servicio_prom = float(max(1.0, 15.0 + 4.0 * int(cliente["items"])))
            except Exception:
                servicio_prom = float(max(1.0, 15.0 + 4.0 * int(cliente["items"])))
        return (ahead / cap) * servicio_prom

    # 2) elegir caja priorizando vacÃ­as, luego con hueco, luego cola efectiva
    for lane in eligibles:
        lane._state = estado_lane(lane)
        lane._eff = cola_efectiva(lane)
    best_state = min(lane._state for lane in eligibles)
    candidatos = [lane for lane in eligibles if lane._state == best_state]

    def _pick_random(lanes: list[CheckoutLane]) -> CheckoutLane:
        if len(lanes) == 1:
            return lanes[0]
        slow_profile = cliente["profile"] in SLOW_PROFILES
        express = [ln for ln in lanes if ln.lane_type is LaneType.EXPRESS]
        others = [ln for ln in lanes if ln.lane_type is not LaneType.EXPRESS]
        if slow_profile:
            preferred = [
                ln
                for ln in others
                if ln.lane_type in (LaneType.REGULAR, LaneType.PRIORITY)
            ]
            if preferred:
                return preferred[np.random.randint(len(preferred))]
        if express:
            weight = EXPRESS_PROFILE_WEIGHT.get(
                cliente["profile"], DEFAULT_EXPRESS_WEIGHT
            )
            if slow_profile:
                weight = min(weight, 0.05)
            if np.random.random() < weight:
                return express[np.random.randint(len(express))]
            if others:
                return others[np.random.randint(len(others))]
        return lanes[np.random.randint(len(lanes))]

    if best_state >= 2:
        for lane in candidatos:
            lane._wait_est = tiempo_estimado_espera(lane)
        min_wait = min(lane._wait_est for lane in candidatos)
        WAIT_TOL = 2.0  # segundos
        mejores = [lane for lane in candidatos if lane._wait_est <= min_wait + WAIT_TOL]
        lane = _pick_random(mejores)
    else:
        lane = min(candidatos, key=lambda ln: ln._eff)
    eff_q_len = max(int(lane._eff), 0)
    lane_type_txt = lane_type_str(lane)

    # 3) loguear solicitud de cola
    time_log.append(
        {
            "source_folder": cliente["source_folder"],
            "timestamp_s": env.now,
            "event_type": "queue_request",
            "customer_id": cliente["customer_id"],
            "profile": as_str(cliente["profile"]),
            "priority": as_str(cliente["priority"]),
            "items": cliente["items"],
            "payment_method": as_str(cliente["payment_method"]),
            "lane_name": lane.lane_id,
            "lane_type": lane_type_txt,
            "effective_queue_length": eff_q_len,
            "queue_request_priority": as_str(cliente["priority"]),
            "patience_s": cliente["patience_s"],
            "service_time_s": 0,
            "revenue_clp": 0,
            "profit_clp": 0,
            "reason": "",
        }
    )

    def _registrar_balk(reason: str) -> None:
        time_log.append(
            {
                "source_folder": cliente["source_folder"],
                "timestamp_s": env.now,
                "event_type": "balk",
                "customer_id": cliente["customer_id"],
                "profile": as_str(cliente["profile"]),
                "priority": as_str(cliente["priority"]),
                "items": cliente["items"],
                "payment_method": as_str(cliente["payment_method"]),
                "lane_name": lane.lane_id,
                "lane_type": lane_type_txt,
                "effective_queue_length": eff_q_len,
                "queue_request_priority": as_str(cliente["priority"]),
                "patience_s": cliente["patience_s"],
                "service_time_s": 0,
                "revenue_clp": 0,
                "profit_clp": 0,
                "reason": reason,
            }
        )
        registro = cliente.copy()
        registro.update(
            {
                "service_start_s": "",
                "service_end_s": "",
                "wait_time_s": "",
                "service_time_s": "",
                "total_time_s": "",
                "arrival_hour": int((OPEN_S + cliente["arrival_time_s"]) // 3600),
                "lane_name": lane.lane_id,
                "lane_type": lane_type_txt,
                "queue_request_priority": as_str(cliente["priority"]),
                "outcome": "balk",
                "balk_reason": reason,
                "abandon_reason": "",
                "effective_queue_length": eff_q_len,
            }
        )
        customers_rows.append(registro)

    day_type_value = cliente.get("day_type") or day_type
    threshold = lookup_balking_threshold(
        cliente["profile"],
        cliente["priority"],
        cliente["payment_method"],
        day_type_value,
        lane.lane_type,
    )
    if threshold is not None and eff_q_len > threshold:
        _registrar_balk("queue_threshold")
        return

    # 4) balk probabilÃ­stico
    if balk_model is not None:
        p_balk = float(
            balk_model.prob_balk(
                profile=cliente["profile"],
                priority=cliente["priority"],
                queue_len=eff_q_len,
                items=int(cliente["items"]),
                lane_type=lane.lane_type,
                payment=cliente["payment_method"],
                arrival_hour=int((OPEN_S + cliente["arrival_time_s"]) // 3600),
                day_type=day_type_value,
            )
        )
        p_balk = min(max(p_balk, 0.0), 1.0)
        if np.random.random() < p_balk:
            _registrar_balk("balk_model")
            return

    # 5) esperar turno (paciencia + cierre)
    request_kwargs = {}
    if lane.lane_type is LaneType.PRIORITY:
        request_kwargs["priority"] = _priority_request_value(cliente["priority"])

    with lane.servidor.request(**request_kwargs) as req:
        jornada_restante = max(0.0, (CLOSE_S - OPEN_S) - env.now)
        max_wait = min(float(cliente["patience_s"]), jornada_restante)

        start_wait = env.now
        resultado = yield req | env.timeout(max_wait)

        if req not in resultado:
            reason = (
                "store_close" if env.now >= (CLOSE_S - OPEN_S) else "patience_timeout"
            )
            time_log.append(
                {
                    "source_folder": cliente["source_folder"],
                    "timestamp_s": env.now,
                    "event_type": "abandon",
                    "customer_id": cliente["customer_id"],
                    "profile": as_str(cliente["profile"]),
                    "priority": as_str(cliente["priority"]),
                    "items": cliente["items"],
                    "payment_method": as_str(cliente["payment_method"]),
                    "lane_name": lane.lane_id,
                    "lane_type": lane_type_txt,
                    "effective_queue_length": cola_efectiva(lane),
                    "queue_request_priority": as_str(cliente["priority"]),
                    "patience_s": cliente["patience_s"],
                    "service_time_s": 0,
                    "revenue_clp": 0,
                    "profit_clp": 0,
                    "reason": reason,
                }
            )
            registro = cliente.copy()
            registro.update(
                {
                    "service_start_s": "",
                    "service_end_s": "",
                    "wait_time_s": env.now - start_wait,
                    "service_time_s": "",
                    "total_time_s": env.now - cliente["arrival_time_s"],
                    "arrival_hour": int((OPEN_S + cliente["arrival_time_s"]) // 3600),
                    "lane_name": lane.lane_id,
                    "lane_type": lane_type_txt,
                    "queue_request_priority": as_str(cliente["priority"]),
                    "outcome": "abandon",
                    "balk_reason": "",
                    "abandon_reason": reason,
                    "effective_queue_length": cola_efectiva(lane),
                }
            )
            customers_rows.append(registro)
            return

        # 6) servicio
        service_start = env.now
        wait_time = env.now - start_wait

        st = None
        st_cap = None

        if "SERVICE_TIME_MODEL" in globals():
            try:
                st = float(
                    SERVICE_TIME_MODEL.sample(
                        profile=cliente["profile"],
                        items=int(cliente["items"]),
                        lane_type=lane.lane_type,
                        priority=cliente["priority"],
                        payment_method=cliente["payment_method"],
                        day_type=cliente.get("day_type") or day_type,
                    )
                )
            except Exception:
                st = None

        if not (isinstance(st, (int, float)) and np.isfinite(st) and st > 0):
            items_int = int(cliente["items"])
            st = max(0, 15.0 + 4.0 * items_int)

        st_cap = max(
            0.0, min(st, (CLOSE_S - OPEN_S) - service_start)
        )  # <-- ahora existe
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
            profit_dict=PROFIT_BETAS,
        )
        cliente["total_profit_clp"] = profit_clp

        # 8) logs de servicio
        time_log.append(
            {
                "source_folder": cliente["source_folder"],
                "timestamp_s": service_start,
                "event_type": "service_start",
                "customer_id": cliente["customer_id"],
                "profile": as_str(cliente["profile"]),
                "priority": as_str(cliente["priority"]),
                "items": cliente["items"],
                "payment_method": as_str(cliente["payment_method"]),
                "lane_name": lane.lane_id,
                "lane_type": lane_type_txt,
                "effective_queue_length": cola_efectiva(lane),
                "queue_request_priority": as_str(cliente["priority"]),
                "patience_s": cliente["patience_s"],
                "service_time_s": st_cap,
                "revenue_clp": 0,
                "profit_clp": profit_clp,
                "reason": "",
            }
        )
        time_log.append(
            {
                "source_folder": cliente["source_folder"],
                "timestamp_s": service_end,
                "event_type": "service_end",
                "customer_id": cliente["customer_id"],
                "profile": as_str(cliente["profile"]),
                "priority": as_str(cliente["priority"]),
                "items": cliente["items"],
                "payment_method": as_str(cliente["payment_method"]),
                "lane_name": lane.lane_id,
                "lane_type": lane_type_txt,
                "effective_queue_length": max(cola_efectiva(lane), 0),
                "queue_request_priority": as_str(cliente["priority"]),
                "patience_s": cliente["patience_s"],
                "service_time_s": st_cap,
                "revenue_clp": 0,
                "profit_clp": profit_clp,
                "reason": "",
            }
        )

        # 9) registro principal
        registro = cliente.copy()
        registro.update(
            {
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
                "profit_clp": profit_clp,
            }
        )
        customers_rows.append(registro)


# %%
from pathlib import Path


LAMBDA_MATRIX_DTYPE = np.float32


# OptimizaciÃ³n: precomputar una matriz de lambdas por segundo para todas
# las combinaciones (perfil, prioridad, pago) del dÃ­a. Esto elimina bÃºsquedas
# repetidas en cada iteraciÃ³n del bucle de llegadas sin cambiar el proceso.
def _build_lambda_matrix(combos: list[tuple], day_len: int) -> np.ndarray:
    """Devuelve matriz (n_combos, day_len+1) con tasas por segundo.

    Para cada combinaciÃ³n usa la serie subyacente (lambda por minuto) y la
    transforma a lambda por segundo constante a trozos en [0, day_len].
    """
    per_profile_series: dict[CustomerProfile, dict] = {}
    n = len(combos)
    if n == 0:
        return np.zeros((0, day_len + 1), dtype=LAMBDA_MATRIX_DTYPE)

    matriz = np.zeros((n, day_len + 1), dtype=LAMBDA_MATRIX_DTYPE)

    for idx, (_, _, _, dist) in enumerate(combos):
        # Cachear series por perfil para evitar I/O repetido
        ser = per_profile_series.get(dist.profile)
        if ser is None:
            ser = _get_series_for_profile(dist.profile)
            per_profile_series[dist.profile] = ser

        pair = ser.get((dist.day_type, dist.priority, dist.payment_method))
        if not pair:
            continue

        t, lam = pair
        # t en segundos, lam en "por minuto". Queremos por segundo.
        t = np.asarray(t, dtype=int)
        lam_sec = np.maximum(
            np.asarray(lam, dtype=LAMBDA_MATRIX_DTYPE), 0.0
        ) / np.float32(60.0)

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
            # Padding defensivo si algo quedÃ³ corto por redondeos
            row = np.pad(row, (0, (day_len + 1) - row.shape[0]), mode="edge")

        matriz[idx, :] = row.astype(LAMBDA_MATRIX_DTYPE, copy=False)

    return matriz


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
        if cantidad <= 0:
            return
        if lane_type is LaneType.SCO:
            idx = 0
            for _ in range(cantidad):
                idx += 1
                lanes.append(
                    CheckoutLane(env, f"{prefix}-{idx}", lane_type, capacity=1)
                )
            return
        for idx in range(cantidad):
            lanes.append(CheckoutLane(env, f"{prefix}-{idx+1}", lane_type))

    policy = CURRENT_LANE_POLICY.get(day_type)
    if not policy:
        policy = CURRENT_LANE_POLICY.get(DayType.TYPE_1, {})

    total_lanes = sum(
        int(policy.get(lt, 0))
        for lt in (LaneType.REGULAR, LaneType.EXPRESS, LaneType.PRIORITY, LaneType.SCO)
    )
    if total_lanes <= 0:
        fallback_counts = DEFAULT_LANE_COUNTS.get(
            day_type, DEFAULT_LANE_COUNTS.get(DayType.TYPE_1, {})
        )
        policy = {
            LaneType.REGULAR: int(fallback_counts.get("regular", 0)),
            LaneType.EXPRESS: int(fallback_counts.get("express", 0)),
            LaneType.PRIORITY: int(fallback_counts.get("priority", 0)),
            LaneType.SCO: int(fallback_counts.get("self_checkout", 0)),
        }

    for lane_type in (
        LaneType.REGULAR,
        LaneType.EXPRESS,
        LaneType.PRIORITY,
        LaneType.SCO,
    ):
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
        # Tasas por segundo para cada combinaciÃ³n en este segundo
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

        sampler = getattr(cfg_by_prof[prof], "items_sampler", None)
        if callable(sampler):
            items = sampler(prof, priority, payment_method, day_type)
        else:
            items = draw_items(prof, priority, payment_method, day_type)

        items = int(max(1, items))

        patience = cfg_by_prof[prof].patience_distribution.sample(
            prof, priority, payment_method, day_type
        )

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

    for rows, fields in (
        (customers_rows_dia, CUSTOMERS_FIELDS),
        (time_log_dia, TIMELOG_FIELDS),
    ):
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

    print(
        f"[Semana {week_idx}] Dia {dia_num} guardado en {output_folder} ({len(customers_rows_dia)} clientes)"
    )

    items_prom = (
        np.mean([c["items"] for c in customers_rows_dia]) if customers_rows_dia else 0
    )
    paciencia_prom = (
        np.mean([c["patience_s"] for c in customers_rows_dia])
        if customers_rows_dia
        else 0
    )

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
    cfg_by_prof: dict[CustomerProfile, ProfileConfig] = {
        perfil: ProfileConfig(profile=perfil) for perfil in perfiles
    }

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
        num_weeks=25,
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
# %%
# EjecuciÃ³n principal: simulamos solo algunas semanas y extrapolamos a un aÃ±o

if __name__ == "__main__":
    from simulator.reporting import run_full_workflow

    run_full_workflow()
