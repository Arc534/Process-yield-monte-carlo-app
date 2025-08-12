
import json
import io
from typing import Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import yaml

from sim_core import ProcessConfig, UnitOperation, VolumeModel, simulate

st.set_page_config(page_title="AAV Monte Carlo Simulator", layout="wide")
st.title("AAV Monte Carlo Simulator")

# --- Simple password gate ---
def _check_password():
    def password_entered():
        if st.session_state.get("password") == st.secrets.get("APP_PASSWORD"):
            st.session_state["pwd_ok"] = True
        else:
            st.session_state["pwd_ok"] = False
            st.error("Incorrect password")
    if st.session_state.get("pwd_ok"):
        return True
    st.text_input("Password", type="password", key="password", on_change=password_entered)
    st.stop()
_check_password()
# --- End password gate ---

# ----- Helpers -----
def _num(spec, key, default):
    try:
        return float(spec.get(key, default))
    except Exception:
        return default

def dist_editor(prefix: str, label: str, existing: Dict[str, Any] | None = None, default_type: str = "fixed") -> Dict[str, Any]:
    """Distribution editor that PREFILLS from existing spec."""
    spec_in = existing or {}
    with st.container(border=True):
        st.caption(label)
        opts = ["fixed","normal","lognormal","uniform","triangular","empirical","beta"]
        dtype_default = spec_in.get("type", default_type)
        dtype = st.selectbox(f"{label} — distribution type", opts,
                             key=f"{prefix}_dtype",
                             index=opts.index(dtype_default) if dtype_default in opts else 0)
        spec: Dict[str, Any] = {"type": dtype}
        cols = st.columns(3)
        if dtype == "fixed":
            spec["value"] = cols[0].number_input("Value", key=f"{prefix}_fixed_val",
                                                 value=_num(spec_in, "value", 1.0), format="%.6g")
        elif dtype == "normal":
            spec["mean"] = cols[0].number_input("Mean", key=f"{prefix}_norm_mean",
                                                value=_num(spec_in, "mean", 1.0), format="%.6g")
            spec["sd"]   = cols[1].number_input("SD", key=f"{prefix}_norm_sd",
                                                value=_num(spec_in, "sd", 0.1), format="%.6g", min_value=0.0)
        elif dtype == "lognormal":
            mode = st.radio("Parameterization", ["Natural-space mean/sd","Log-space mu/sigma"],
                            key=f"{prefix}_ln_mode",
                            horizontal=True,
                            index=0 if ("mean" in spec_in or "sd" in spec_in) else 1)
            if mode == "Natural-space mean/sd":
                spec["mean"] = cols[0].number_input("Mean (natural)", key=f"{prefix}_ln_mean",
                                                    value=_num(spec_in, "mean", 1.0), format="%.6g", min_value=1e-12)
                spec["sd"]   = cols[1].number_input("SD (natural)", key=f"{prefix}_ln_sd",
                                                    value=_num(spec_in, "sd", 0.1), format="%.6g", min_value=1e-12)
            else:
                spec["mu_log"]    = cols[0].number_input("mu_log", key=f"{prefix}_ln_mu",
                                                         value=_num(spec_in, "mu_log", 0.0), format="%.6g")
                spec["sigma_log"] = cols[1].number_input("sigma_log", key=f"{prefix}_ln_sigma",
                                                         value=_num(spec_in, "sigma_log", 0.25), format="%.6g", min_value=1e-6)
        elif dtype == "uniform":
            spec["low"]  = cols[0].number_input("Low", key=f"{prefix}_uni_low",
                                                value=_num(spec_in, "low", 0.8), format="%.6g")
            spec["high"] = cols[1].number_input("High", key=f"{prefix}_uni_high",
                                                value=_num(spec_in, "high", 1.0), format="%.6g")
        elif dtype == "triangular":
            spec["low"]  = cols[0].number_input("Low", key=f"{prefix}_tri_low",
                                                value=_num(spec_in, "low", 0.8), format="%.6g")
            spec["mode"] = cols[1].number_input("Mode", key=f"{prefix}_tri_mode",
                                                value=_num(spec_in, "mode", 0.9), format="%.6g")
            spec["high"] = cols[2].number_input("High", key=f"{prefix}_tri_high",
                                                value=_num(spec_in, "high", 0.97), format="%.6g")
        elif dtype == "empirical":
            st.info("Provide a CSV under 'Empirical Inputs' with a column name matching this key.")
            spec["key"] = st.text_input("Empirical column key", key=f"{prefix}_emp_key",
                                        value=spec_in.get("key", f"{prefix}_data"))
        elif dtype == "beta":
            spec["alpha"] = cols[0].number_input("Alpha", key=f"{prefix}_beta_a",
                                                 value=_num(spec_in, "alpha", 31.5), format="%.6g", min_value=0.001)
            spec["beta"]  = cols[1].number_input("Beta",  key=f"{prefix}_beta_b",
                                                 value=_num(spec_in, "beta", 3.5), format="%.6g", min_value=0.001)
        return spec

with st.sidebar:
    st.header("Simulation Settings")
    n_iter = st.number_input("Iterations", min_value=100, max_value=500000, value=20000, step=1000)
    seed = st.number_input("Random Seed", min_value=0, max_value=10_000_000, value=42, step=1)

    st.divider()
    st.subheader("Empirical Inputs (optional)")
    empirical_file = st.file_uploader("CSV with columns referenced by 'key' fields", type=["csv"])

    st.divider()
    mode = st.radio("Configuration Mode", ["Builder (no file)","Upload file"], index=0)

empirical_inputs = None
if empirical_file is not None:
    emp_df = pd.read_csv(empirical_file)
    empirical_inputs = {col: emp_df[col].dropna() for col in emp_df.columns}
    st.sidebar.success(f"Empirical data loaded: {list(emp_df.columns)}")

# ---- Session defaults ----
if "ops" not in st.session_state:
    st.session_state.ops = []
if "upstream_cfg" not in st.session_state:
    st.session_state.upstream_cfg = {
        "titer_vg_per_mL": {"type":"lognormal","mean":2.0e11,"sd":5.0e10},
        "volume_L": {"type":"fixed","value":450.0, "units":"L"}
    }
# result holders
for k in ("results_long","stats_per_step","yields_map"):
    st.session_state.setdefault(k, None)

def add_op():
    st.session_state.ops.append({
        "name": f"Unit Op {len(st.session_state.ops)+1}",
        "yield": {"type":"beta","alpha":31.5,"beta":3.5},
        "volume_model": {"concentration_factor":{"type":"fixed","value":1.0}}
    })

def remove_op(idx: int):
    if 0 <= idx < len(st.session_state.ops):
        st.session_state.ops.pop(idx)

def to_config_dict():
    return {"upstream": st.session_state.upstream_cfg, "unit_operations": st.session_state.ops}

# ---- Builder / Upload ----
if mode == "Builder (no file)":
    st.subheader("Upstream Inputs")
    with st.container(border=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Upstream titer (vg/mL)**")
            st.session_state.upstream_cfg["titer_vg_per_mL"] = dist_editor("up_titer","Upstream titer",
                                                                            existing=st.session_state.upstream_cfg["titer_vg_per_mL"],
                                                                            default_type="lognormal")
        with cols[1]:
            st.markdown("**Upstream volume**")
            st.session_state.upstream_cfg["volume_L"] = dist_editor("up_vol","Upstream volume",
                                                                    existing=st.session_state.upstream_cfg["volume_L"],
                                                                    default_type="fixed")
            up_vol_unit = st.selectbox("Upstream volume units", ["L","mL","uL"],
                                       index=["L","mL","uL"].index(st.session_state.upstream_cfg["volume_L"].get("units","L")),
                                       key="up_vol_unit")
            st.session_state.upstream_cfg["volume_L"]["units"] = up_vol_unit

    st.subheader("Unit Operations")
    c1, c2 = st.columns([1,3])
    with c1:
        if st.button("➕ Add unit operation", use_container_width=True):
            add_op()
    with c2:
        if st.button("🗑️ Remove last", use_container_width=True, disabled=(len(st.session_state.ops)==0)):
            remove_op(len(st.session_state.ops)-1)

    for i, op in enumerate(st.session_state.ops):
        with st.expander(f"{i+1}. {op['name']}", expanded=True):
            op["name"] = st.text_input("Name", value=op["name"], key=f"op_{i}_name")
            st.markdown("**Yield (fractional recovery 0–1)**")
            op["yield"] = dist_editor(f"op_{i}_yield", f"{op['name']} — Yield",
                                      existing=op["yield"], default_type="beta")

            st.markdown("**Volume model (choose any that apply)**")
            vm = op.get("volume_model",{})

            use_cf = st.checkbox("Concentration factor (>=1)", key=f"op_{i}_use_cf", value=("concentration_factor" in vm))
            if use_cf:
                vm["concentration_factor"] = dist_editor(f"op_{i}_cf","Concentration factor",
                                                         existing=vm.get("concentration_factor"),
                                                         default_type="fixed")
            else:
                vm.pop("concentration_factor", None)

            use_add = st.checkbox("Dilution addition", key=f"op_{i}_use_add", value=("dilution_addition" in vm) or ("dilution_addition_L" in vm))
            if use_add:
                vm["dilution_addition"] = dist_editor(f"op_{i}_add","Dilution addition",
                                                      existing=vm.get("dilution_addition") or vm.get("dilution_addition_L"))
                vm["dilution_addition"]["units"] = st.selectbox("Units for dilution addition", ["L","mL","uL"], index=0, key=f"op_{i}_add_units")
                vm.pop("dilution_addition_L", None)
            else:
                vm.pop("dilution_addition", None); vm.pop("dilution_addition_L", None)

            use_tgt = st.checkbox("Target volume — overrides the above", key=f"op_{i}_use_tgt", value=("target_volume" in vm) or ("target_volume_L_dist" in vm))
            if use_tgt:
                vm["target_volume"] = dist_editor(f"op_{i}_tgt","Target volume",
                                                  existing=vm.get("target_volume") or vm.get("target_volume_L_dist"))
                vm["target_volume"]["units"] = st.selectbox("Units for target volume", ["L","mL","uL"], index=0, key=f"op_{i}_tgt_units")
                vm.pop("target_volume_L_dist", None)
            else:
                vm.pop("target_volume", None); vm.pop("target_volume_L_dist", None)

            op["volume_model"] = vm

    cfg_dict = to_config_dict()
    with st.expander("Show current config JSON"):
        st.code(json.dumps(cfg_dict, indent=2))

else:
    st.subheader("Upload a YAML/JSON config")
    cfg_file = st.file_uploader("Process Config (YAML or JSON)", type=["yaml","yml","json"], key="cfg_file_upload")
    if cfg_file is None:
        st.info("No file uploaded yet.")
    else:
        content = cfg_file.read()
        try:
            if cfg_file.name.lower().endswith((".yaml",".yml")):
                cfg_dict = yaml.safe_load(content)
            else:
                cfg_dict = json.loads(content)
            # Load into session-state to EDIT after upload
            st.session_state.upstream_cfg = cfg_dict["upstream"]
            st.session_state.ops = cfg_dict["unit_operations"]
            st.success("Config loaded into Builder. You can tweak and run.")
        except Exception as e:
            st.error(f"Failed to parse config: {e}")

# ---- Run / Preserve results ----
c_run, c_clear = st.columns([1,1])
run_clicked = c_run.button("Run Simulation", type="primary", key="run_btn")
clear_clicked = c_clear.button("Clear results", key="clear_btn")

if clear_clicked:
    st.session_state["results_long"] = None
    st.session_state["stats_per_step"] = None
    st.session_state["yields_map"] = None

if run_clicked:
    try:
        proc_cfg = _build_config = None
        # Build ProcessConfig from current session config
        upstream_titer = st.session_state.upstream_cfg["titer_vg_per_mL"]
        upstream_vol = st.session_state.upstream_cfg["volume_L"]
        ops_objs = []
        for op in st.session_state.ops:
            vm = VolumeModel(
                concentration_factor = op.get("volume_model",{}).get("concentration_factor"),
                dilution_addition_L  = op.get("volume_model",{}).get("dilution_addition_L"),
                dilution_addition    = op.get("volume_model",{}).get("dilution_addition"),
                target_volume_L_dist = op.get("volume_model",{}).get("target_volume_L_dist"),
                target_volume        = op.get("volume_model",{}).get("target_volume"),
            )
            ops_objs.append(UnitOperation(name=op["name"], yield_dist=op["yield"], volume_model=vm))
        proc_cfg = ProcessConfig(upstream_titer_dist=upstream_titer, upstream_volume_L_dist=upstream_vol, unit_operations=ops_objs)

        with st.spinner("Simulating..."):
            results_long, stats_per_step, yields_map = simulate(proc_cfg, n_iter=int(st.session_state.get("n_iter", 20000) or 20000),
                                                                seed=int(st.session_state.get("seed", 42) or 42),
                                                                empirical_inputs=None)
        st.session_state["results_long"] = results_long
        st.session_state["stats_per_step"] = stats_per_step
        st.session_state["yields_map"] = yields_map
        st.success("Done! Results are cached — you can change visualization controls without re-running.")
    except Exception as e:
        st.error(f"Simulation error: {e}")

# Use cached results if present
results_long = st.session_state.get("results_long")
stats_per_step = st.session_state.get("stats_per_step")
yields_map = st.session_state.get("yields_map")

if results_long is not None and stats_per_step is not None:
    # Downloads (raw numeric)
    csv_buf = io.StringIO(); results_long.to_csv(csv_buf, index=False)
    st.download_button("Download raw samples (CSV)", data=csv_buf.getvalue(), file_name="samples.csv", mime="text/csv")

    # Stats aggregation
    all_stats = []
    for step, df in stats_per_step.items():
        tmp = df.copy(); tmp.insert(0, "step", step)
        all_stats.append(tmp.reset_index().rename(columns={"index":"metric"}))
    stats_table = pd.concat(all_stats, ignore_index=True)

    # Scientific notation for display (titer & genomes)
    def _fmt_sci(x):
        try:
            return f"{float(x):.3e}"
        except Exception:
            return x
    sci_cols = ["titer_vg_per_mL","genomes_vg"]
    display_stats = stats_table.copy()
    mask = display_stats["metric"].isin(sci_cols)
    num_cols = ["mean","sd","min","p5","p25","median","p75","p95","max"]
    display_stats.loc[mask, num_cols] = display_stats.loc[mask, num_cols].applymap(_fmt_sci)

    st.subheader("Summary statistics")
    st.dataframe(display_stats, use_container_width=True)

    # ---- Bounds controls ----
    st.markdown("### Chart bounds / regions")
    bound_mode = st.radio(
        "How do you want to define bounds?",
        ["Central %", "± k·SD around mean", "Manual bounds"],
        horizontal=True,
        key="bounds_mode"
    )
    central_pct = st.slider("Central percent", 50, 99, 68, step=1, key="central_pct")
    k_sd = st.slider("k (SDs)", 0.1, 3.0, 1.0, step=0.1, key="k_sd")
    manual_low = st.text_input("Manual lower bound (number, scientific ok, e.g. 1e11)", value="", key="manual_low")
    manual_high = st.text_input("Manual upper bound (number, scientific ok, e.g. 9e11)", value="", key="manual_high")

    def _parse_float(s):
        try:
            return float(s)
        except Exception:
            return None

    def _compute_bounds(series: pd.Series):
        s = series.replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
        if s.size == 0:
            return (None, None)
        if st.session_state["bounds_mode"] == "Central %":
            alpha = (100 - st.session_state["central_pct"]) / 2
            lo = float(np.percentile(s, alpha))
            hi = float(np.percentile(s, 100 - alpha))
            return lo, hi
        elif st.session_state["bounds_mode"] == "± k·SD around mean":
            m = float(np.mean(s)); sd = float(np.std(s, ddof=1)) if s.size > 1 else 0.0
            return m - st.session_state["k_sd"]*sd, m + st.session_state["k_sd"]*sd
        else:
            lo = _parse_float(st.session_state["manual_low"])
            hi = _parse_float(st.session_state["manual_high"])
            return lo, hi

    # ---- Visualizer ----
    st.header("Visualizer")
    metric = st.selectbox("Metric", ["titer_vg_per_mL","volume_L","genomes_vg","yield_frac"], index=0, key="vis_metric")

    steps_available = list(results_long["step"].unique())
    if metric == "yield_frac":
        # exclude Upstream_Start (no yield)
        steps_for_yield = [s for s in st.session_state["yields_map"].keys() if s != "Upstream_Start"]
        step = st.selectbox("Step", steps_for_yield, key="vis_step")
        data_series = pd.Series(st.session_state["yields_map"][step])
    else:
        step = st.selectbox("Step", steps_available, key="vis_step")
        data_series = results_long.loc[results_long["step"]==step, metric]

    view = st.selectbox("View", ["Histogram+KDE","CDF","Box"], index=0, key="vis_view")

    s = data_series.replace([np.inf,-np.inf], np.nan).dropna().to_numpy()

    axis_fmt = alt.Axis(format=".2e") if metric in ["titer_vg_per_mL","genomes_vg"] else alt.Axis()

    if st.session_state["vis_view"] == "Histogram+KDE":
        hist = alt.Chart(pd.DataFrame({"value": s})).mark_bar(opacity=0.5).encode(
            x=alt.X("value:Q", bin=alt.Bin(maxbins=50), axis=axis_fmt, title=metric),
            y=alt.Y("count()", title="count")
        )
        kde = alt.Chart(pd.DataFrame({"value": s})).transform_density(
            "value", as_=["value","density"]
        ).mark_area(opacity=0.4).encode(
            x=alt.X("value:Q", axis=axis_fmt, title=metric),
            y="density:Q"
        )
        lo, hi = _compute_bounds(pd.Series(s))
        layers = [hist, kde]
        if lo is not None:
            layers.append(alt.Chart(pd.DataFrame({"x":[lo]})).mark_rule(strokeDash=[4,4]).encode(x="x:Q"))
        if hi is not None:
            layers.append(alt.Chart(pd.DataFrame({"x":[hi]})).mark_rule(strokeDash=[4,4]).encode(x="x:Q"))
        if (lo is not None) and (hi is not None) and (hi > lo):
            layers.append(alt.Chart(pd.DataFrame({"x":[lo], "x2":[hi]})).mark_rect(opacity=0.08).encode(x="x:Q", x2="x2:Q"))
        st.altair_chart(alt.layer(*layers).resolve_scale(y='independent'), use_container_width=True)

    elif st.session_state["vis_view"] == "CDF":
        xs = np.sort(s)
        ys = np.arange(1, len(xs)+1)/len(xs) if len(xs)>0 else np.array([])
        chart = alt.Chart(pd.DataFrame({"x": xs, "F": ys})).mark_line().encode(
            x=alt.X("x:Q", axis=axis_fmt, title=metric),
            y=alt.Y("F:Q", title="CDF")
        )
        st.altair_chart(chart, use_container_width=True)

    else:
        chart = alt.Chart(pd.DataFrame({"metric":[metric]*len(s), "value": s})).mark_boxplot().encode(
            x=alt.X("metric:N", title=""),
            y=alt.Y("value:Q", axis=axis_fmt, title=metric)
        )
        st.altair_chart(chart, use_container_width=True)

    st.caption(f"Step: {step}  |  Metric: {metric}  |  View: {st.session_state['vis_view']}")

else:
    st.info("No results yet. Configure your process and click **Run Simulation**. Results will persist while you explore visuals.")
