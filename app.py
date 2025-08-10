
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

def dist_editor(prefix: str, label: str, default_type: str = "fixed") -> Dict[str, Any]:
    with st.container(border=True):
        st.caption(label)
        dtype = st.selectbox(f"{label} — distribution type", ["fixed","normal","lognormal","uniform","triangular","empirical"],
                             key=f"{prefix}_dtype", index=["fixed","normal","lognormal","uniform","triangular","empirical"].index(default_type) if default_type in ["fixed","normal","lognormal","uniform","triangular","empirical"] else 0)
        spec: Dict[str, Any] = {"type": dtype}
        cols = st.columns(3)
        if dtype == "fixed":
            spec["value"] = cols[0].number_input("Value", key=f"{prefix}_fixed_val", value=1.0, format="%.6g")
        elif dtype == "normal":
            spec["mean"] = cols[0].number_input("Mean", key=f"{prefix}_norm_mean", value=1.0, format="%.6g")
            spec["sd"]   = cols[1].number_input("SD", key=f"{prefix}_norm_sd", value=0.1, format="%.6g", min_value=0.0)
        elif dtype == "lognormal":
            mode = st.radio("Parameterization", ["Natural-space mean/sd","Log-space mu/sigma"], key=f"{prefix}_ln_mode", horizontal=True)
            if mode == "Natural-space mean/sd":
                spec["mean"] = cols[0].number_input("Mean (natural)", key=f"{prefix}_ln_mean", value=1.0, format="%.6g", min_value=1e-12)
                spec["sd"]   = cols[1].number_input("SD (natural)", key=f"{prefix}_ln_sd", value=0.1, format="%.6g", min_value=1e-12)
            else:
                spec["mu_log"]    = cols[0].number_input("mu_log", key=f"{prefix}_ln_mu", value=0.0, format="%.6g")
                spec["sigma_log"] = cols[1].number_input("sigma_log", key=f"{prefix}_ln_sigma", value=0.25, format="%.6g", min_value=1e-6)
        elif dtype == "uniform":
            spec["low"]  = cols[0].number_input("Low", key=f"{prefix}_uni_low", value=0.8, format="%.6g")
            spec["high"] = cols[1].number_input("High", key=f"{prefix}_uni_high", value=1.0, format="%.6g")
        elif dtype == "triangular":
            spec["low"]  = cols[0].number_input("Low", key=f"{prefix}_tri_low", value=0.8, format="%.6g")
            spec["mode"] = cols[1].number_input("Mode", key=f"{prefix}_tri_mode", value=0.9, format="%.6g")
            spec["high"] = cols[2].number_input("High", key=f"{prefix}_tri_high", value=0.97, format="%.6g")
        elif dtype == "empirical":
            st.info("Provide a CSV under 'Empirical Inputs' with a column name matching this key.")
            spec["key"] = st.text_input("Empirical column key", key=f"{prefix}_emp_key", value=f"{prefix}_data")
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

if "ops" not in st.session_state:
    st.session_state.ops = []
if "upstream_cfg" not in st.session_state:
    st.session_state.upstream_cfg = {
        "titer_vg_per_mL": {"type":"lognormal","mean":5.0e11,"sd":1.0e11},
        "volume_L": {"type":"uniform","low":190,"high":210}
    }

def add_op():
    st.session_state.ops.append({
        "name": f"Unit Op {len(st.session_state.ops)+1}",
        "yield": {"type":"triangular","low":0.85,"mode":0.92,"high":0.98},
        "volume_model": {"concentration_factor":{"type":"fixed","value":1.0}}
    })

def remove_op(idx: int):
    if 0 <= idx < len(st.session_state.ops):
        st.session_state.ops.pop(idx)

def to_config_dict():
    return {"upstream": st.session_state.upstream_cfg, "unit_operations": st.session_state.ops}

if mode == "Builder (no file)":
    st.subheader("Upstream Inputs")
    with st.container(border=True):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Upstream titer (vg/mL)**")
            st.session_state.upstream_cfg["titer_vg_per_mL"] = dist_editor("up_titer","Upstream titer", default_type=st.session_state.upstream_cfg["titer_vg_per_mL"]["type"])
        with cols[1]:
            st.markdown("**Upstream volume (L)**")
            st.session_state.upstream_cfg["volume_L"] = dist_editor("up_vol","Upstream volume", default_type=st.session_state.upstream_cfg["volume_L"]["type"])

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
            op["yield"] = dist_editor(f"op_{i}_yield", f"{op['name']} — Yield", default_type=op["yield"]["type"])

            st.markdown("**Volume model (choose any that apply)**")
            vm = op.get("volume_model",{})

            use_cf = st.checkbox("Concentration factor (>=1)", key=f"op_{i}_use_cf", value=("concentration_factor" in vm))
            if use_cf:
                vm["concentration_factor"] = dist_editor(f"op_{i}_cf","Concentration factor", default_type=vm.get("concentration_factor",{"type":"fixed"}).get("type","fixed"))
            else:
                vm.pop("concentration_factor", None)

            use_add = st.checkbox("Dilution addition (L)", key=f"op_{i}_use_add", value=("dilution_addition_L" in vm))
            if use_add:
                vm["dilution_addition_L"] = dist_editor(f"op_{i}_add","Dilution addition (L)", default_type=vm.get("dilution_addition_L",{"type":"fixed"}).get("type","fixed"))
            else:
                vm.pop("dilution_addition_L", None)

            use_tgt = st.checkbox("Target volume (L) — overrides the above", key=f"op_{i}_use_tgt", value=("target_volume_L_dist" in vm))
            if use_tgt:
                vm["target_volume_L_dist"] = dist_editor(f"op_{i}_tgt","Target volume (L)", default_type=vm.get("target_volume_L_dist",{"type":"fixed"}).get("type","fixed"))
            else:
                vm.pop("target_volume_L_dist", None)

            op["volume_model"] = vm

    cfg_dict = to_config_dict()
    with st.expander("Show current config JSON"):
        st.code(json.dumps(cfg_dict, indent=2))

else:
    st.subheader("Upload a YAML/JSON config")
    cfg_file = st.file_uploader("Process Config (YAML or JSON)", type=["yaml","yml","json"], key="cfg_file_upload")
    if cfg_file is None:
        st.info("No file uploaded yet."); st.stop()
    content = cfg_file.read()
    try:
        if cfg_file.name.lower().endswith((".yaml",".yml")):
            cfg_dict = yaml.safe_load(content)
        else:
            cfg_dict = json.loads(content)
    except Exception as e:
        st.error(f"Failed to parse config: {e}"); st.stop()

def load_config_from_obj(obj: Dict[str, Any]) -> ProcessConfig:
    upstream_titer = obj["upstream"]["titer_vg_per_mL"]
    upstream_vol = obj["upstream"]["volume_L"]
    ops = []
    for op in obj["unit_operations"]:
        vm = VolumeModel(
            concentration_factor = op.get("volume_model",{}).get("concentration_factor"),
            dilution_addition_L  = op.get("volume_model",{}).get("dilution_addition_L"),
            target_volume_L_dist = op.get("volume_model",{}).get("target_volume_L_dist"),
        )
        ops.append(UnitOperation(name=op["name"], yield_dist=op["yield"], volume_model=vm))
    return ProcessConfig(upstream_titer_dist=upstream_titer, upstream_volume_L_dist=upstream_vol, unit_operations=ops)

try:
    proc_cfg = load_config_from_obj(cfg_dict)
except Exception as e:
    st.error(f"Config error: {e}"); st.stop()

run = st.button("Run Simulation", type="primary")
if run:
    with st.spinner("Simulating..."):
        results_long, stats_per_step = simulate(proc_cfg, n_iter=int(n_iter), seed=int(seed), empirical_inputs=empirical_inputs)
    st.success("Done!")

    csv_buf = io.StringIO(); results_long.to_csv(csv_buf, index=False)
    st.download_button("Download raw samples (CSV)", data=csv_buf.getvalue(), file_name="samples.csv", mime="text/csv")

    all_stats = []
    for step, df in stats_per_step.items():
        tmp = df.copy(); tmp.insert(0, "step", step)
        all_stats.append(tmp.reset_index().rename(columns={"index":"metric"}))
    stats_table = pd.concat(all_stats, ignore_index=True)
    csv_buf2 = io.StringIO(); stats_table.to_csv(csv_buf2, index=False)
    st.download_button("Download summary stats (CSV)", data=csv_buf2.getvalue(), file_name="summary_stats.csv", mime="text/csv")

    st.subheader("Summary statistics")
    st.dataframe(stats_table, use_container_width=True)

    st.subheader("Distributions by step")
    metrics = ["titer_vg_per_mL", "volume_L", "genomes_vg"]
    for step in results_long["step"].unique():
        st.markdown(f"### {step}")
        df = results_long[results_long["step"] == step].copy()
        for metric in metrics:
            st.markdown(f"**{metric}**")
            chart = alt.Chart(df).transform_density(metric, as_=[metric, 'density']).mark_area(opacity=0.5).encode(
                x=alt.X(f"{metric}:Q", title=metric), y='density:Q'
            )
            hist = alt.Chart(df).mark_bar(opacity=0.5).encode(
                x=alt.X(f"{metric}:Q", bin=alt.Bin(maxbins=50), title=metric), y=alt.Y('count()', title='count')
            )
            st.altair_chart((hist & chart).resolve_scale(y='independent'), use_container_width=True)

    st.subheader("Compounding relationship")
    st.markdown("""
- **Total_vg_A = Titer_A × Volume_A**  
- **Total_vg_B = Yield_B × Total_vg_A**  
- After volume changes at B: **Titer_B = Total_vg_B / Volume_B**
""")
