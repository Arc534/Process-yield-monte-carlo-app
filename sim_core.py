
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

UNIT_TO_L = {"L": 1.0, "mL": 1e-3, "uL": 1e-6}

def _as_liters(samples: np.ndarray, unit: Optional[str]) -> np.ndarray:
    if unit is None:
        unit = "L"
    factor = UNIT_TO_L.get(unit, 1.0)
    return samples * factor

def _units_from_spec(spec: Dict[str, Any]) -> Optional[str]:
    return spec.get("units")

def sample_distribution(spec: Dict[str, Any], n: int, rng: np.random.Generator, empirical_data: Optional[pd.Series]=None) -> np.ndarray:
    dtype = spec.get("type", "fixed").lower()
    if dtype == "fixed":
        return np.full(n, float(spec.get("value", 0.0)))
    elif dtype == "normal":
        mu = float(spec["mean"]); sd = float(spec["sd"])
        return rng.normal(mu, sd, size=n)
    elif dtype == "lognormal":
        if "mu_log" in spec and "sigma_log" in spec:
            mu_log = float(spec["mu_log"]); sigma_log = float(spec["sigma_log"])
        else:
            m = float(spec["mean"]); s = float(spec["sd"])
            if m <= 0 or s <= 0:
                raise ValueError("For lognormal with natural-space mean/sd, both must be > 0.")
            sigma_log = np.sqrt(np.log(1 + (s**2)/(m**2)))
            mu_log = np.log(m) - 0.5 * sigma_log**2
        return rng.lognormal(mean=mu_log, sigma=sigma_log, size=n)
    elif dtype == "uniform":
        low = float(spec["low"]); high = float(spec["high"])
        return rng.uniform(low, high, size=n)
    elif dtype == "triangular":
        left = float(spec["low"]); mode = float(spec["mode"]); right = float(spec["high"])
        return rng.triangular(left, mode, right, size=n)
    elif dtype == "beta":
        a = float(spec["alpha"]); b = float(spec["beta"])
        return rng.beta(a, b, size=n)
    elif dtype == "empirical":
        if empirical_data is None or len(empirical_data) == 0:
            raise ValueError("Empirical distribution specified but no empirical data provided.")
        return rng.choice(empirical_data.to_numpy(), size=n, replace=True)
    else:
        raise ValueError(f"Unsupported distribution type: {dtype}")

@dataclass
class VolumeModel:
    concentration_factor: Optional[Dict[str, Any]] = None
    dilution_addition_L: Optional[Dict[str, Any]] = None
    dilution_addition: Optional[Dict[str, Any]] = None
    target_volume_L_dist: Optional[Dict[str, Any]] = None
    target_volume: Optional[Dict[str, Any]] = None

@dataclass
class UnitOperation:
    name: str
    yield_dist: Dict[str, Any]
    volume_model: VolumeModel

@dataclass
class ProcessConfig:
    upstream_titer_dist: Dict[str, Any]
    upstream_volume_L_dist: Dict[str, Any]
    unit_operations: List[UnitOperation]

def simulate(config: ProcessConfig, n_iter: int=10000, seed: Optional[int]=42,
             empirical_inputs: Optional[Dict[str, pd.Series]]=None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)

    titer0 = sample_distribution(config.upstream_titer_dist, n_iter, rng, _maybe_empirical(config.upstream_titer_dist, empirical_inputs))
    vol0_spec = config.upstream_volume_L_dist
    vol0_raw = sample_distribution(vol0_spec, n_iter, rng, _maybe_empirical(vol0_spec, empirical_inputs))
    vol0_L = _as_liters(vol0_raw, _units_from_spec(vol0_spec))
    vol0_mL = vol0_L * 1000.0
    genomes0 = titer0 * vol0_mL

    steps = ["Upstream_Start"]
    titer = [titer0]
    volume_L = [vol0_L]
    genomes = [genomes0]

    for op in config.unit_operations:
        y = sample_distribution(op.yield_dist, n_iter, rng, _maybe_empirical(op.yield_dist, empirical_inputs))
        y = np.clip(y, 0.0, 1.0)
        genomes_i = genomes[-1] * y

        vol_in_L = volume_L[-1]

        if op.volume_model.concentration_factor is not None:
            cf_spec = op.volume_model.concentration_factor
            cf = sample_distribution(cf_spec, n_iter, rng, _maybe_empirical(cf_spec, empirical_inputs))
            cf = np.where(cf < 1.0, 1.0, cf)
            vol_in_L = vol_in_L / cf

        add_spec = op.volume_model.dilution_addition or op.volume_model.dilution_addition_L
        if add_spec is not None:
            add_raw = sample_distribution(add_spec, n_iter, rng, _maybe_empirical(add_spec, empirical_inputs))
            unit = _units_from_spec(add_spec) or ("L" if op.volume_model.dilution_addition_L is not None else None)
            vol_in_L = vol_in_L + _as_liters(add_raw, unit)

        tgt_spec = op.volume_model.target_volume or op.volume_model.target_volume_L_dist
        if tgt_spec is not None:
            tgt_raw = sample_distribution(tgt_spec, n_iter, rng, _maybe_empirical(tgt_spec, empirical_inputs))
            unit = _units_from_spec(tgt_spec) or ("L" if op.volume_model.target_volume_L_dist is not None else None)
            vol_in_L = _as_liters(tgt_raw, unit)

        vol_mL = vol_in_L * 1000.0
        titer_i = np.where(vol_mL > 0, genomes_i / vol_mL, np.nan)

        steps.append(op.name)
        titer.append(titer_i)
        volume_L.append(vol_in_L)
        genomes.append(genomes_i)

    data = []
    for s, ti, vl, ge in zip(steps, titer, volume_L, genomes):
        data.append(pd.DataFrame({"step": s, "titer_vg_per_mL": ti, "volume_L": vl, "genomes_vg": ge}))
    results_long = pd.concat(data, ignore_index=True)

    stats_per_step = {}
    for s in results_long["step"].unique():
        df = results_long[results_long["step"] == s]
        stats_per_step[s] = _describe(df[["titer_vg_per_mL", "volume_L", "genomes_vg"]])

    return results_long, stats_per_step

def _maybe_empirical(spec: Dict[str, Any], empirical_inputs: Optional[Dict[str, pd.Series]]) -> Optional[pd.Series]:
    if spec and spec.get("type","").lower() == "empirical":
        key = spec.get("key")
        if not key:
            raise ValueError("Empirical distribution requires a 'key'.")
        if empirical_inputs is None or key not in empirical_inputs:
            raise ValueError(f"Empirical data for key '{key}' not found.")
        return empirical_inputs[key]
    return None

def _describe(df: pd.DataFrame) -> pd.DataFrame:
    desc = {}
    for col in df.columns:
        series = df[col].replace([np.inf, -np.inf], np.nan).dropna()
        if len(series) == 0:
            desc[col] = {"count": 0, "mean": np.nan, "sd": np.nan, "min": np.nan, "p5": np.nan, "p25": np.nan,
                         "median": np.nan, "p75": np.nan, "p95": np.nan, "max": np.nan}
        else:
            desc[col] = {
                "count": int(series.count()),
                "mean": float(series.mean()),
                "sd": float(series.std(ddof=1)) if series.count() > 1 else 0.0,
                "min": float(series.min()),
                "p5": float(np.percentile(series, 5)),
                "p25": float(np.percentile(series, 25)),
                "median": float(np.percentile(series, 50)),
                "p75": float(np.percentile(series, 75)),
                "p95": float(np.percentile(series, 95)),
                "max": float(series.max())
            }
    out = pd.DataFrame(desc).T
    out.index.name = "metric"
    return out
