
# AAV Monte Carlo Simulator (Streamlit)

This app runs a Monte Carlo simulation for AAV processes with a **process-agnostic Builder**. Add unit operations, define distributions, and visualize outputs and stats per step.

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push these files to a GitHub repo (keep `app.py` at repo root).
2. Go to https://share.streamlit.io → **New app**.
3. Select your repo/branch and `app.py` as the entry point.
4. Deploy → share the URL.

## Notes
- Supported distributions: `fixed`, `normal`, `lognormal` (natural or log params), `uniform`, `triangular`, `empirical` (requires a CSV column `key`).
- Volume model: concentration factor (>=1), dilution addition (L), or target volume (overrides others).
- Compounding:
  - Total_vg_A = Titer_A × Volume_A
  - Total_vg_B = Yield_B × Total_vg_A
  - Titer_B = Total_vg_B / Volume_B
