
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from typing import Dict, List, Tuple, Literal, Optional

st.set_page_config(page_title="Klimatdeklarationer – Energiklass vs klimatpåverkan", layout="wide")

# -----------------------------
# Helpers
# -----------------------------

COL_ENERGY = "Energiklass"
COL_BASELINE = "Total klimatpåverkan per bruttoarea [kg CO₂e/m² BTA]"

TYPE_PREFIX = "Byggnadens användning "
TYPE_SUFFIX = " (BTA) m²"

# Note: G utesluts enligt beställning
CLASS_FACTOR = {
    "A": 0.50,
    "B": 0.75,
    "C": 1.00,
    "D": 1.35,
    "E": 1.80,
    "F": 2.35,
    # "G": 2.50,  # Utesluts
}

def detect_header_row(file_like) -> int:
    """Find row index that contains expected headers; default to 0."""
    tmp = BytesIO(file_like.getvalue()) if hasattr(file_like, "getvalue") else file_like
    peek = pd.read_excel(tmp, sheet_name=0, header=None, nrows=10)
    header_row = 0
    for i in range(min(10, len(peek))):
        vals = [str(x) for x in list(peek.iloc[i].values)]
        line = " ".join(vals)
        if ("Energiklass" in line) or ("Total klimatpåverkan per bruttoarea" in line):
            header_row = i
            break
    return header_row

def read_excel_auto_header(uploaded_file) -> pd.DataFrame:
    """Read first sheet, auto-detect header row by scanning for known columns."""
    header_row = detect_header_row(uploaded_file)
    tmp2 = BytesIO(uploaded_file.getvalue())
    df = pd.read_excel(tmp2, sheet_name=0, header=header_row)
    df.columns = df.columns.astype(str).str.strip()
    return df

def find_bta_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith(TYPE_PREFIX) and c.endswith(TYPE_SUFFIX)]

def type_from_col(col: str) -> str:
    return col.replace(TYPE_PREFIX, "").replace(TYPE_SUFFIX, "").strip()

def infer_type_row(row: pd.Series, bta_cols: List[str]) -> Optional[str]:
    present = []
    for c in bta_cols:
        try:
            v = float(row.get(c, np.nan))
        except Exception:
            v = np.nan
        if pd.notnull(v) and v > 0:
            present.append(type_from_col(c))
    if len(present) == 1:
        return present[0]
    elif len(present) == 0:
        return None
    else:
        return "Flera typer"

def weighted_median(values: pd.Series, weights: pd.Series) -> float:
    """Compute a weighted median aligned to values' index. Ignores NaN weights/values."""
    s = pd.Series(values).astype(float)
    w = pd.Series(weights).astype(float)
    mask = s.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float(np.nanmedian(s.values))
    s = s[mask]
    w = w[mask]
    order = np.argsort(s.values)
    sv = s.values[order]
    wv = w.values[order]
    cumw = np.cumsum(wv)
    cutoff = 0.5 * wv.sum()
    idx = np.searchsorted(cumw, cutoff)
    idx = min(idx, len(sv) - 1)
    return float(sv[idx])

def filter_outliers(df: pd.DataFrame, method: str, q_low: float, q_high: float, std_sigmas: float, exclude_zero: bool) -> pd.DataFrame:
    s = pd.to_numeric(df[COL_BASELINE], errors="coerce")
    work = df.copy()
    if exclude_zero:
        work = work[s > 0]
        s = pd.to_numeric(work[COL_BASELINE], errors="coerce")
    if method == "Kvantil (P-låg–P-hög)":
        lo, hi = s.quantile(q_low), s.quantile(q_high)
        return work[(s >= lo) & (s <= hi)]
    elif method == "Std (μ ± σ)":
        mu, sigma = s.mean(), s.std(ddof=0)
        lo, hi = mu - std_sigmas * sigma, mu + std_sigmas * sigma
        return work[(s >= lo) & (s <= hi)]
    else:
        return work

def build_grid_curve(points_df: pd.DataFrame, start_year: int, end_year: int) -> pd.Series:
    """Interpolate g CO2e/kWh over years, then convert to kg CO2e/kWh. Hold last value constant."""
    pts = points_df.dropna().sort_values("År")
    if len(pts) < 2:
        raise ValueError("Minst två punkter krävs i elintensitetskurvan.")
    xp = pts["År"].astype(float).values
    fp = pts["g CO₂e/kWh (el)"].astype(float).values
    years = np.arange(start_year, end_year + 1)
    interp_g = np.interp(years, xp, fp, left=fp[0], right=fp[-1])
    return pd.Series(interp_g / 1000.0, index=years, name="kg_CO2e_per_kWh_el")

def build_other_curve_const(start_year: int, end_year: int, g_co2e_per_kwh_other: float) -> pd.Series:
    years = np.arange(start_year, end_year + 1)
    return pd.Series((g_co2e_per_kwh_other or 0.0) / 1000.0, index=years, name="kg_CO2e_per_kWh_other")

def aggregate_L0(df_sub: pd.DataFrame, agg: str, weighted: bool, selected_type: str) -> float:
    s = pd.to_numeric(df_sub[COL_BASELINE], errors="coerce").dropna()
    if s.empty:
        return np.nan
    if not weighted:
        return (s.median() if agg == "Median" else s.mean())
    # Weighted by BTA of the selected type column
    area_col = f"{TYPE_PREFIX}{selected_type}{TYPE_SUFFIX}"
    if area_col not in df_sub.columns:
        return (s.median() if agg == "Median" else s.mean())
    w = pd.to_numeric(df_sub.loc[s.index, area_col], errors="coerce").fillna(0.0)
    if agg == "Median":
        return weighted_median(s, w)
    else:
        w_nonzero = w.replace(0, np.nan)
        if w_nonzero.notna().any():
            return float(np.average(s, weights=w_nonzero.fillna(0.0)))
        return float(s.mean())

def compute_curves(df_all: pd.DataFrame,
                   selected_type: str,
                   classes: List[str],
                   c_ref_map: Dict[str, float],
                   start_year: int,
                   horizon_years: int,
                   grid_curve_kg: pd.Series,
                   share_el: float,
                   other_curve_kg: Optional[pd.Series],
                   agg: str,
                   weighted: bool) -> pd.DataFrame:
    # Filter to selected type
    df_type = df_all[df_all["Byggnadstyp"] == selected_type].copy()
    years = np.arange(start_year, start_year + horizon_years + 1)
    # Build blended intensity curve (kg/kWh)
    if share_el >= 1.0 or other_curve_kg is None:
        mix_curve = grid_curve_kg.loc[years]
    else:
        mix_curve = share_el * grid_curve_kg.loc[years] + (1.0 - share_el) * other_curve_kg.loc[years]
    curves = {}
    c_ref = c_ref_map.get(selected_type, c_ref_map.get("default", 90.0))
    for k in classes:
        if k not in CLASS_FACTOR:
            continue
        df_k = df_type[df_type[COL_ENERGY] == k]
        if df_k.empty:
            continue
        L0 = aggregate_L0(df_k, agg=agg, weighted=weighted, selected_type=selected_type)
        if np.isnan(L0):
            continue
        EP = CLASS_FACTOR[k] * c_ref  # kWh/m²·år
        annual = EP * mix_curve
        cum = L0 + annual.cumsum()
        curves[k] = cum
    out = pd.DataFrame(curves, index=years)
    out.index.name = "År"
    return out

def baseline_stats(df_type: pd.DataFrame, classes: List[str]) -> pd.DataFrame:
    rows = []
    for k in classes:
        s = pd.to_numeric(df_type.loc[df_type[COL_ENERGY] == k, COL_BASELINE], errors="coerce")
        s = s[s > 0]
        if len(s) == 0:
            rows.append({"Energiklass": k, "Antal": 0, "P10": np.nan, "Median": np.nan, "P90": np.nan})
        else:
            rows.append({
                "Energiklass": k,
                "Antal": int(s.count()),
                "P10": float(s.quantile(0.10)),
                "Median": float(s.median()),
                "P90": float(s.quantile(0.90)),
            })
    return pd.DataFrame(rows)

# -----------------------------
# Sidebar – Inputs
# -----------------------------

st.sidebar.header("1) Ladda upp data")
uploaded = st.sidebar.file_uploader("Excel (xlsx/xls) med Klimatdeklarationer", type=["xlsx", "xls"])

if uploaded is None:
    st.info("Ladda upp din Excel-fil för att börja.")
    st.stop()

df = read_excel_auto_header(uploaded)

# Guard: basic columns
missing_cols = [c for c in [COL_ENERGY, COL_BASELINE] if c not in df.columns]
if missing_cols:
    st.error(f"Saknar kolumner: {missing_cols}")
    st.stop()

# Infer building type
bta_cols = find_bta_columns(df)
df["Byggnadstyp"] = df.apply(lambda r: infer_type_row(r, bta_cols), axis=1)

# Optionally exclude ambiguous/unknown types
st.sidebar.header("2) Dataval & filter")
exclude_ambiguous = st.sidebar.checkbox("Exkludera rader med 'Flera typer' och okänd typ", value=True)
if exclude_ambiguous:
    df = df[df["Byggnadstyp"].isin([type_from_col(c) for c in bta_cols])]

# Available types after exclusion
available_types = sorted([t for t in df["Byggnadstyp"].dropna().unique().tolist()])

if not available_types:
    st.error("Hittade inga tydliga byggnadstyper efter filtrering.")
    st.stop()

selected_type = st.sidebar.selectbox("Välj byggnadstyp", available_types)

# Outlier filter settings
st.sidebar.subheader("Outlier-filter (baseline)")
method = st.sidebar.radio("Metod", ["Kvantil (P-låg–P-hög)", "Std (μ ± σ)", "Ingen"], index=0, horizontal=False)
q_low = st.sidebar.slider("P-låg", 0.0, 0.2, 0.05, 0.01)
q_high = st.sidebar.slider("P-hög", 0.8, 1.0, 0.95, 0.01)
std_sigmas = st.sidebar.slider("σ (std)", 0.5, 4.0, 2.0, 0.1)
exclude_zero = st.sidebar.checkbox("Exkludera baseline = 0", value=True)

# Apply outlier filtering (on the whole df so klass-statistik matchar)
df = filter_outliers(df, method, q_low, q_high, std_sigmas, exclude_zero)

# Classes available for the selected type
classes_in_type = sorted([c for c in df.loc[df["Byggnadstyp"] == selected_type, COL_ENERGY].dropna().unique().tolist() if c in CLASS_FACTOR])
default_classes = [c for c in ["A", "B", "C"] if c in classes_in_type] or classes_in_type
selected_classes = st.sidebar.multiselect("Energiklasser att visa", classes_in_type, default=default_classes)

# Startår & horisont
st.sidebar.header("3) Tid")
start_year = st.sidebar.number_input("Startår", value=2025, min_value=1990, max_value=2100, step=1)
horizon_years = st.sidebar.slider("Tidshorisont (år)", min_value=1, max_value=100, value=50, step=1)

# Elintensitetskurva (5 punkter – men vi accepterar ≥2)
st.sidebar.header("4) Elintensitet (g CO₂e/kWh)")
default_curve = pd.DataFrame({
    "År": [2025, 2030, 2040, 2050, 2060],
    "g CO₂e/kWh (el)": [70.0, 65.0, 60.0, 50.0, 45.0],
})
curve_df = st.sidebar.data_editor(default_curve, num_rows="dynamic", use_container_width=True, key="curve_editor")
                                  #help="Minst två punkter. Interpolation görs linjärt per år. Efter sista året hålls värdet konstant.")

# Andel el
st.sidebar.header("5) Energimix i drift")
el_share_pct = st.sidebar.slider("Andel el (%)", 0, 100, 100, 1)
other_g = None
if el_share_pct < 100:
    other_g = st.sidebar.number_input("Övrig energi: g CO₂e/kWh (konstant)", value=0.0, min_value=0.0, step=1.0)

# C-krav
st.sidebar.header("6) C‑krav (kWh/m²·år)")
c_default = st.sidebar.number_input("Globalt C‑krav (default för alla typer)", value=90.0, min_value=1.0, step=1.0)
per_type = st.sidebar.checkbox("Specificera C‑krav per byggnadstyp", value=False)
c_ref_map: Dict[str, float] = {"default": c_default}
if per_type:
    st.sidebar.markdown("Ange C‑värden per typ:")
    for t in available_types:
        c_ref_map[t] = st.sidebar.number_input(f"{t}", value=float(c_default), min_value=1.0, step=1.0, key=f"c_{t}")

# Aggregation
st.sidebar.header("7) Aggregat för startnivå (L₀)")
agg = st.sidebar.radio("Sammanfattningsmått", ["Median", "Medel"], index=0, horizontal=True)
weighted = st.sidebar.checkbox("BTA‑vikta L₀", value=False) #, help="Viktning använder BTA för vald byggnadstyp på varje rad som vikt.")

# -----------------------------
# Main – Results
# -----------------------------

st.title("Klimatpåverkan per m² – Energiklasser över tid")
st.caption("Startnivån L₀ hämtas från 'Total klimatpåverkan per bruttoarea [kg CO₂e/m² BTA]' (efter outlier‑filter). Driftpåverkan läggs på årligen utifrån energiklass (EP) och vald klimatintensitet för energimixen.")

# Quick overview
colA, colB, colC = st.columns(3)
with colA:
    st.metric("Antal rader (efter filter)", len(df))
with colB:
    st.metric("Tillgängliga byggnadstyper", len(available_types))
with colC:
    st.metric("Vald typ", selected_type)

# Stats table for baseline per class
df_type_now = df[df["Byggnadstyp"] == selected_type]
stats_tbl = baseline_stats(df_type_now, selected_classes)
st.subheader("Baseline – översikt per energiklass (efter filter)")
st.dataframe(stats_tbl, use_container_width=True)

# Distribution plots
show_box = st.checkbox("Visa boxplot för baseline (kg CO₂e/m² BTA)", value=True)
show_violin = st.checkbox("Visa violindiagram för baseline (kg CO₂e/m² BTA)", value=True)

if show_box and len(selected_classes) > 0:
    data = [pd.to_numeric(df_type_now.loc[df_type_now[COL_ENERGY]==k, COL_BASELINE], errors="coerce").dropna() for k in selected_classes]
    fig1, ax1 = plt.subplots(figsize=(8,4))
    ax1.boxplot(data, showmeans=True)
    ax1.set_xticks(np.arange(1, len(selected_classes)+1))
    ax1.set_xticklabels(selected_classes)
    ax1.set_ylabel("kg CO₂e/m² BTA")
    ax1.set_title(f"{selected_type} – Baselinefördelning per energiklass (boxplot)")
    ax1.grid(True, linestyle=":")
    st.pyplot(fig1)

if show_violin and len(selected_classes) > 0:
    data = [pd.to_numeric(df_type_now.loc[df_type_now[COL_ENERGY]==k, COL_BASELINE], errors="coerce").dropna() for k in selected_classes]
    fig2, ax2 = plt.subplots(figsize=(8,4))
    ax2.violinplot(data, showmeans=False, showmedians=True)
    ax2.set_xticks(np.arange(1, len(selected_classes)+1))
    ax2.set_xticklabels(selected_classes)
    ax2.set_ylabel("kg CO₂e/m² BTA")
    ax2.set_title(f"{selected_type} – Baselinefördelning per energiklass (violin)")
    ax2.grid(True, linestyle=":")
    st.pyplot(fig2)

# Curves
try:
    grid_curve_kg = build_grid_curve(curve_df, start_year, start_year + horizon_years)
except Exception as e:
    st.error(f"Fel i elintensitetskurvan: {e}")
    st.stop()

other_curve_kg = None
if el_share_pct < 100:
    other_curve_kg = build_other_curve_const(start_year, start_year + horizon_years, other_g or 0.0)

share_el = float(el_share_pct) / 100.0

curves = compute_curves(
    df_all=df,
    selected_type=selected_type,
    classes=selected_classes,
    c_ref_map=c_ref_map,
    start_year=start_year,
    horizon_years=horizon_years,
    grid_curve_kg=grid_curve_kg,
    share_el=share_el,
    other_curve_kg=other_curve_kg,
    agg=agg,
    weighted=weighted,
)

st.subheader("Ackumulerad klimatpåverkan per m² (L₀ + drift över tid)")
if curves.empty:
    st.info("Inga kurvor att visa – kontrollera att valda energiklasser finns i datat efter filtrering.")
else:
    fig3, ax3 = plt.subplots(figsize=(9,5))
    for col in curves.columns:
        ax3.plot(curves.index, curves[col], label=str(col))
    ax3.set_xlabel("År")
    ax3.set_ylabel("Ackumulerad klimatpåverkan (kg CO₂e/m² BTA)")
    ax3.set_title(f"{selected_type} – A/B/C m.fl. över {horizon_years} år (start {start_year})")
    ax3.grid(True, linestyle=":")
    ax3.legend()
    st.pyplot(fig3)

    st.markdown("**Kurvdata (för vidare analys i notebook om du vill):**")
    st.dataframe(curves, use_container_width=True)

# Footnotes
with st.expander("Antaganden & noter"):
    st.markdown("""
- **C‑krav**: Globalt default 90 kWh/m²·år, valfritt per byggnadstyp.
- **Energiklasser**: A=0,50×C, B=0,75×C, C=1,00×C, D=1,35×C, E=1,80×C, F=2,35×C. **G utesluts.**
- **L₀** (startnivå) beräknas som vald sammanfattning (median/medel) av baseline per klass efter outlier‑filtrering.
  Om **BTA‑viktning** är aktiverad vägs varje rad med BTA för *vald byggnadstyp*.
- **Driftpåverkan**: EP (kWh/m²·år) × vald klimatintensitet (kg CO₂e/kWh).
  Elintensitet ges som en kurva (interpolerad), övrig energi som konstant nivå om andel el < 100 %.
- **Outliers**: Standard är P5–P95 och att baseline=0 tas bort.
- **Datakvalitet**: Rader med **Flera typer** eller **okänd** typ exkluderas i standardläget.
""")
