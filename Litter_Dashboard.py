import json
from pathlib import Path
# =========================
# COUNTY-LEVEL HELPERS
# =========================
def get_county_engagement(df, county):
    d = df[df["county"] == county].sort_values("year")
    return d if not d.empty else None

def get_county_cumulative(df, county):
    d = df[df["county"] == county].sort_values("year").copy()
    if d.empty:
        return None
    d["cum_litter"] = d["litter"].cumsum()
    d["cum_recycled"] = d["recycled"].cumsum()
    return d


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st

# =====================================================
# GLOBAL CONFIG
# =====================================================
pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="TN Litter Dashboard",
    page_icon="🗺️",
    layout="wide",
)

# =====================================================
# PATHS
# =====================================================
BASE = Path(__file__).parent
DATA = BASE / "data"

STATE_FILE = DATA / "state_year_kpis.csv"
MAP_FILE = DATA / "TN_Litter_Map_County_Year.csv"
COUNTY_METRICS_FILE = DATA / "county_year_metrics.csv"
GEOJSON_FILE = DATA / "tn_counties.geojson"
GEOJSON_KEY = "NAME"  # property in geojson: properties.NAME

# =====================================================
# STYLING (from gpt.py)
# =====================================================
st.markdown(
    """
<style>
.block-container { padding-top: 0.8rem; }

.header {
  background: #1fb6a6;
  padding: 16px 24px;
  border-radius: 14px;
  color: white;
  margin-bottom: 10px;
}

.header h1 {
  margin: 0;
  font-size: 22px;
  font-weight: 800;
}

.panel {
  background: white;
  border-radius: 14px;
  padding: 20px;
  border: 1px solid #e5e7eb;
}

.panel h2, .panel h3 { margin-top: 0; }

.note {
  font-size: 0.92rem;
  color: #374151;
}

.kpi-grid {
  background: white;
  border-radius: 14px;
  padding: 14px 16px;
  border: 1px solid #e5e7eb;
}

hr.soft {
  border: none;
  border-top: 1px solid #e5e7eb;
  margin: 12px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================
# HELPERS (from gpt.py)
# =====================================================
def safe_div(n, d):
    return np.where((d is None) | (pd.isna(d)) | (d == 0), np.nan, n / d)

def fmt_num(x):
    if pd.isna(x):
        return "—"
    x = float(x)
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.2f}K"
    return f"{x:,.0f}"

def fmt_ratio(x, pct=False):
    if pd.isna(x):
        return "—"
    x = float(x)
    return f"{x*100:.1f}%" if pct else f"{x:,.2f}"

def has_cols(df, cols):
    return all(c in df.columns for c in cols)

def fmt(x):
    """Format numbers with K/M suffixes - from dashboard_fixed.py"""
    if pd.isna(x):
        return "N/A"
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{int(x)}"

@st.cache_data(show_spinner=False)
def load_data():
    df_state = pd.read_csv(STATE_FILE)
    df_map = pd.read_csv(MAP_FILE)
    df_county = pd.read_csv(COUNTY_METRICS_FILE) if COUNTY_METRICS_FILE.exists() else pd.DataFrame()
    with open(GEOJSON_FILE, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # Normalize column names for both dataframes
    def normalize_cols(df):
        rename_map = {}
        for c in df.columns:
            cl = c.lower().strip()
            if cl in ["litter_lbs", "litter lbs", "litter (lbs)"]:
                rename_map[c] = "litter"
            elif cl in ["recycled_lbs", "recycled lbs", "recycled (lbs)"]:
                rename_map[c] = "recycled"
            elif cl in ["dump_sites", "dump sites"]:
                rename_map[c] = "dumps"
            elif cl in ["partners_helped", "partners # helped", "partners helped"]:
                rename_map[c] = "partners"
            elif cl in ["vol_hr", "vol_hrs", "vol'r hours", "vol hours", "volunteer_hours", "volunteer hours"]:
                rename_map[c] = "vol_hours"
            elif cl in ["pers_use", "pers'l # use", "personal use", "person use", "persons"]:
                rename_map[c] = "pers_use"
            elif cl in ["county_rd_miles", "county road miles", "county rd. miles", "county rd miles"]:
                rename_map[c] = "county_rd_miles"
            elif cl in ["state_rd_miles", "state road miles", "state rd. miles", "state rd miles"]:
                rename_map[c] = "state_rd_miles"
        if rename_map:
            df = df.rename(columns=rename_map)
        return df
    
    df_state = normalize_cols(df_state)
    df_map = normalize_cols(df_map)
    if not df_county.empty:
        df_county = normalize_cols(df_county)

    # normalize county names
    if "county" in df_map.columns:
        df_map["county"] = df_map["county"].astype(str).str.strip()
    if "county" in df_state.columns:
        df_state["county"] = df_state["county"].astype(str).str.strip()
    if not df_county.empty and "county" in df_county.columns:
        df_county["county"] = df_county["county"].astype(str).str.strip()

    return df_state, df_map, df_county, geojson

def add_derived_metrics(df_map_in: pd.DataFrame) -> pd.DataFrame:
    df = df_map_in.copy()

    # Standardize likely column names
    rename_map = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["litter_lbs", "litter lbs", "litter (lbs)"]:
            rename_map[c] = "litter"
        elif cl in ["recycled_lbs", "recycled lbs", "recycled (lbs)"]:
            rename_map[c] = "recycled"
        elif cl in ["dump_sites", "dump sites"]:
            rename_map[c] = "dumps"
        elif cl in ["partners_helped", "partners # helped", "partners helped"]:
            rename_map[c] = "partners"
        elif cl in ["vol_hr", "vol_hrs", "vol'r hours", "vol hours", "volunteer_hours", "volunteer hours"]:
            rename_map[c] = "vol_hours"
        elif cl in ["pers_use", "pers'l # use", "personal use", "person use", "persons"]:
            rename_map[c] = "pers_use"
        elif cl in ["county_rd_miles", "county road miles", "county rd. miles", "county rd miles"]:
            rename_map[c] = "county_rd_miles"
        elif cl in ["state_rd_miles", "state road miles", "state rd. miles", "state rd miles"]:
            rename_map[c] = "state_rd_miles"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Derived metrics (only compute if the needed columns exist)
    if has_cols(df, ["litter", "county_rd_miles"]):
        df["litter_per_county_mile"] = df["litter"] / df["county_rd_miles"].replace(0, np.nan)

    if has_cols(df, ["litter", "state_rd_miles"]):
        df["litter_per_state_mile"] = df["litter"] / df["state_rd_miles"].replace(0, np.nan)

    if has_cols(df, ["litter", "vol_hours"]):
        df["litter_per_vol_hour"] = df["litter"] / df["vol_hours"].replace(0, np.nan)

    if has_cols(df, ["litter", "pers_use"]):
        df["litter_per_person_use"] = df["litter"] / df["pers_use"].replace(0, np.nan)

    if has_cols(df, ["recycled", "litter"]):
        denom = (df["recycled"] + df["litter"]).replace(0, np.nan)
        df["recycling_rate"] = df["recycled"] / denom

    if has_cols(df, ["dumps", "county_rd_miles"]):
        df["dumps_per_100_miles"] = 100 * df["dumps"] / df["county_rd_miles"].replace(0, np.nan)

    return df

def metric_catalog(df_map: pd.DataFrame):
    """
    Returns a list of metric definitions available in df_map.
    Each item: dict(label=..., col=..., fmt=..., palette=..., family=...)
    """
    cats = []

    # Core (likely always present)
    if "litter" in df_map.columns:
        cats.append(dict(label="Litter (lbs)", col="litter", fmt="num", family="litter"))
    if "recycled" in df_map.columns:
        cats.append(dict(label="Recycled (lbs)", col="recycled", fmt="num", family="recycled"))
    if "dumps" in df_map.columns:
        cats.append(dict(label="Dump Sites", col="dumps", fmt="num", family="dumps"))
    if "partners" in df_map.columns:
        cats.append(dict(label="Partners Helped", col="partners", fmt="num", family="engagement"))
    if "vol_hours" in df_map.columns:
        cats.append(dict(label="Volunteer Hours", col="vol_hours", fmt="num", family="engagement"))
    if "county_rd_miles" in df_map.columns:
        cats.append(dict(label="County Road Miles", col="county_rd_miles", fmt="num", family="miles"))
    if "state_rd_miles" in df_map.columns:
        cats.append(dict(label="State Road Miles", col="state_rd_miles", fmt="num", family="miles"))
    if "pers_use" in df_map.columns:
        cats.append(dict(label="Person Use", col="pers_use", fmt="num", family="engagement"))

    # Derived
    if "litter_per_county_mile" in df_map.columns:
        cats.append(dict(label="Litter per County Mile (lbs/mi)", col="litter_per_county_mile", fmt="ratio", family="litter"))
    if "litter_per_vol_hour" in df_map.columns:
        cats.append(dict(label="Litter per Vol. Hour (lbs/hr)", col="litter_per_vol_hour", fmt="ratio", family="litter"))
    if "litter_per_person_use" in df_map.columns:
        cats.append(dict(label="Litter per Person Use (lbs/use)", col="litter_per_person_use", fmt="ratio", family="litter"))
    if "recycling_rate" in df_map.columns:
        cats.append(dict(label="Recycling Rate (%)", col="recycling_rate", fmt="pct", family="recycled"))
    if "dumps_per_100_miles" in df_map.columns:
        cats.append(dict(label="Dump Sites per 100 Miles", col="dumps_per_100_miles", fmt="ratio", family="dumps"))

    return cats

def family_palette(family: str):
    # Discrete palettes (ArcGIS-ish) for binned intensity
    if family == "litter":
        return ["#fff5eb", "#fdd0a2", "#fdae6b", "#e6550d", "#a63603"]
    if family == "recycled":
        return ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"]
    if family == "dumps":
        return ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]
    if family == "engagement":
        return ["#f7fbff", "#deebf7", "#9ecae1", "#3182bd", "#08519c"]
    if family == "miles":
        return ["#f7f7f7", "#cccccc", "#969696", "#636363", "#252525"]
    return ["#f7f7f7", "#d9d9d9", "#bdbdbd", "#969696", "#636363"]

def bin_series(values: pd.Series, n_bins=5):
    # Stable bins based on max, with protection for all-zeros / all-nan
    v = values.astype(float)
    vmax = np.nanmax(v.values) if np.isfinite(v).any() else 0
    if vmax <= 0 or np.isnan(vmax):
        # all zeros -> single bin
        bins = [-1, 0, 1, 2, 3, 4]
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        return pd.cut(v.fillna(0), bins=bins, labels=labels, include_lowest=True), labels
    bins = [0, 0.2*vmax, 0.4*vmax, 0.6*vmax, 0.8*vmax, vmax]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    # if duplicates in bins (can happen with small vmax), use quantiles fallback
    if len(set([round(b, 8) for b in bins])) < len(bins):
        try:
            out = pd.qcut(v.rank(method="first"), q=5, labels=labels)
            return out, labels
        except Exception:
            out = pd.cut(v.fillna(0), bins=5, labels=labels, include_lowest=True)
            return out, labels
    return pd.cut(v, bins=bins, labels=labels, include_lowest=True), labels

# =====================================================
# LOAD & PROCESS DATA
# =====================================================
df_state, df_map, df_county_eng, geojson = load_data()
df_map = add_derived_metrics(df_map)
metrics = metric_catalog(df_map)

# Prepare for UI
years = sorted(df_map["year"].dropna().unique())
metric_labels = [m["label"] for m in metrics]
metric_label_to_def = {m["label"]: m for m in metrics}

# =====================================================
# HEADER
# =====================================================
st.markdown(
    """
<div class="header">
  <h1>🌲 Tennessee Statewide Litter, Recycling & Community Impact Dashboard</h1>
</div>
""",
    unsafe_allow_html=True,
)

# =====================================================
# GLOBAL CONTROLS (from gpt.py)
# =====================================================
c1, c2, c3 = st.columns([1.2, 1.8, 1])

with c1:
    year = st.selectbox("Select Year", years, index=len(years) - 1)

with c2:
    metric_label = st.selectbox("Map Metric", metric_labels, index=0)
    metric_def = metric_label_to_def[metric_label]

with c3:
    # This is useful for "priority view" quickly
    mode = st.radio("View Mode", ["All Counties", "Top 10", "Bottom 10"], horizontal=False)

# =====================================================
# TABS
# =====================================================
tab_overview, tab_trends, tab_compare, tab_regional, tab_priorities, tab_summary = st.tabs(
    ["🗺️ Overview", "📈 Trends", "📊 Comparisons", "🌍 Regional", "🎯 Priorities", "📌 Summary"]
)

# =====================================================
# TAB 1 — OVERVIEW (from gpt.py)
# =====================================================
with tab_overview:
    # Filtered data for selected year
    d_year = df_map[df_map["year"] == year].copy()

    # Optional filtering for Top/Bottom 10 (based on selected metric)
    mcol = metric_def["col"]
    if mcol in d_year.columns:
        d_year_sorted = d_year.sort_values(mcol, ascending=False)
        if mode == "Top 10":
            d_year = d_year_sorted.head(10)
        elif mode == "Bottom 10":
            d_year = d_year_sorted.tail(10)

    left, right = st.columns([1, 2.3], gap="large")

    # ---- LEFT PANEL: TEXT + KPIs
    with left:
        st.markdown('<div class="panel overview-text">', unsafe_allow_html=True)
        st.markdown("## Overview")
        st.write(
            """
This dashboard provides a statewide view of litter collection, recycling, dump site activity,
and (where available) engagement/operations metrics such as partners helped and volunteer hours.

Use the map to identify spatial patterns for the selected year, and switch metrics to view
**volume**, **efficiency**, and **engagement** perspectives.
"""
        )

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        st.markdown("### Year Snapshot")

        # Prefer df_state for statewide totals if present; otherwise compute from df_map
        row_state = None
        if "year" in df_state.columns and (df_state["year"] == year).any():
            row_state = df_state[df_state["year"] == year].iloc[0].to_dict()
        else:
            row_state = {"year": year}
            for c in ["litter", "recycled", "dumps", "partners", "vol_hours", "pers_use", "county_rd_miles", "state_rd_miles"]:
                if c in d_year.columns:
                    row_state[c] = d_year[c].sum(skipna=True)

        # KPIs (show what exists)
        kpis = []
        if "litter" in row_state:
            kpis.append(("Total Litter (lbs)", fmt_num(row_state["litter"])))
        if "recycled" in row_state:
            kpis.append(("Recycled (lbs)", fmt_num(row_state["recycled"])))
        if "dumps" in row_state:
            kpis.append(("Dump Sites", fmt_num(row_state["dumps"])))
        if "partners" in row_state:
            kpis.append(("Partners Helped", fmt_num(row_state["partners"])))
        if "vol_hours" in row_state:
            kpis.append(("Volunteer Hours", fmt_num(row_state["vol_hours"])))

        # Derived statewide KPIs
        if ("recycled" in row_state) and ("litter" in row_state):
            denom = (row_state["recycled"] + row_state["litter"])
            rr = (row_state["recycled"] / denom) if denom else np.nan
            kpis.append(("Recycling Rate", fmt_ratio(rr, pct=True)))

        if ("litter" in row_state) and ("vol_hours" in row_state) and row_state["vol_hours"]:
            kpis.append(("Lbs per Volunteer Hour", fmt_ratio(row_state["litter"] / row_state["vol_hours"])))

        # Display KPI tiles (2 per row)
        for i in range(0, len(kpis), 2):
            a, b = st.columns(2)
            with a:
                st.metric(kpis[i][0], kpis[i][1])
            if i + 1 < len(kpis):
                with b:
                    st.metric(kpis[i + 1][0], kpis[i + 1][1])

        st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
        st.markdown(
            "<div class='note'>Tip: Switch to an efficiency metric like <b>Litter per County Mile</b> to compare counties more fairly.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- RIGHT: MAP
    with right:
        if mcol not in d_year.columns:
            st.warning(f"Selected metric column '{mcol}' not found in the map dataset.")
        else:
            values = d_year[mcol].astype(float)

            d_year["Intensity"], labels = bin_series(values)
            palette = family_palette(metric_def["family"])


            # Keep original metric for hover and use a stabilized (log) version for binning
            values_for_bins = np.log10(values + 1)
            d_year["Intensity"], labels = bin_series(values_for_bins)
            d_year["_metric_value"] = values

            # Hover config: always show the real metric value + helpful fields if present
            hover_data = {"_metric_value": ":,.2f"}

            for c in ["litter", "recycled", "dumps", "partners", "vol_hours", "county_rd_miles", "pers_use",
                      "litter_per_county_mile", "litter_per_vol_hour", "litter_per_person_use", "recycling_rate",
                      "dumps_per_100_miles"]:
                if c in d_year.columns:
                    if c == "recycling_rate":
                        hover_data[c] = ":.2%"
                    elif c in ["litter_per_county_mile", "litter_per_vol_hour", "litter_per_person_use", "dumps_per_100_miles"]:
                        hover_data[c] = ":.2f"
                    else:
                        hover_data[c] = ":,.0f"

            fig_map = px.choropleth_mapbox(
                d_year,
                geojson=geojson,
                locations="county",
                featureidkey=f"properties.{GEOJSON_KEY}",
                color="Intensity",
                category_orders={"Intensity": labels},
                hover_name="county",
                hover_data=hover_data,
                mapbox_style="carto-positron",
                zoom=5.8,
                center={"lat": 35.75, "lon": -86.4},
                color_discrete_sequence=palette,
            )

            fig_map.update_layout(
                height=720,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    title="Intensity",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#d1d5db",
                    borderwidth=1,
                    x=0.98,
                    y=0.02,
                    xanchor="right",
                    yanchor="bottom",
                ),
            )

            st.plotly_chart(fig_map, use_container_width=True)

# =====================================================
# TAB 2 — TRENDS
# =====================================================
with tab_trends:
    st.markdown("## Trends Over Time")
    
    # Controls: County and Year selection
    # Allow users to pick a county (or statewide) and optionally focus on a specific year for detailed inspection.
    ctl_col1, ctl_col2 = st.columns(2)

    with ctl_col1:
        counties_list = ["Statewide"] + sorted(df_map["county"].dropna().unique())
        selected_county = st.selectbox("Select County", counties_list, index=0, key="trend_county")

    with ctl_col2:
        # For context, allow the user to highlight a particular year in the trend charts
        selected_year = st.selectbox("Highlight Year", years, index=len(years) - 1, key="trend_year")

    # Initialize dataframes used in this tab prior to branching so they always exist
    df_trend_litter = pd.DataFrame()
    df_trend_eng = pd.DataFrame()
    df_trend_cum = pd.DataFrame()

    # Populate dataframes based on county/statewide selection
    if selected_county == "Statewide":
        # Use statewide aggregates for all three datasets
        df_trend_litter = df_state.copy()
        df_trend_eng = df_state.copy()
        df_trend_cum = df_state.copy()
    else:
        # Slice the map for county-specific litter/recycling trends
        df_trend_litter = df_map[df_map["county"] == selected_county].copy()
        # Engagement comes from the county-level dataset if available, otherwise fall back to map slice
        if not df_county_eng.empty and "county" in df_county_eng.columns:
            df_trend_eng = df_county_eng[df_county_eng["county"] == selected_county].copy()
        else:
            df_trend_eng = df_trend_litter.copy()
        # For cumulative trends, reuse the litter/recycled slice
        df_trend_cum = df_trend_litter.copy()

    # Sort all trend datasets by year for proper chronological plotting
    for _df in [df_trend_litter, df_trend_eng, df_trend_cum]:
        if not _df.empty and "year" in _df.columns:
            _df.sort_values("year", inplace=True)

    # Prepare cumulative values if possible
    if not df_trend_cum.empty and {'litter', 'recycled'}.issubset(df_trend_cum.columns):
        df_trend_cum = df_trend_cum.copy()
        df_trend_cum["cum_litter"] = df_trend_cum["litter"].cumsum()
        df_trend_cum["cum_recycled"] = df_trend_cum["recycled"].cumsum()

    # Lay out three equal-width columns for the three trend charts
    gcol1, gcol2, gcol3 = st.columns(3)

    # --- Graph 1: Litter & Recycling Trends (stacked bar) ---
    with gcol1:
        if not df_trend_litter.empty and 'litter' in df_trend_litter.columns:
            fig_lr = go.Figure()
            # X-axis categories are the fiscal years as provided
            x_vals = df_trend_litter["year"]
            fig_lr.add_trace(go.Bar(
                x=x_vals,
                y=df_trend_litter["litter"],
                name="Litter Collected",
                marker_color="#e6550d"
            ))
            if 'recycled' in df_trend_litter.columns:
                fig_lr.add_trace(go.Bar(
                    x=x_vals,
                    y=df_trend_litter['recycled'],
                    name="Recycled",
                    marker_color="#31a354"
                ))
            fig_lr.update_layout(
                title=f"Litter & Recycling Trends — {selected_county}",
                xaxis_title="Year",
                yaxis_title="Pounds (lbs)",
                height=380,
                barmode='stack',
                hovermode='x unified'
            )
            st.plotly_chart(fig_lr, use_container_width=True)
        else:
            st.info("Litter and recycling data not available for this selection.")

    # --- Graph 2: Community Engagement Trends (Volunteer Hours & Partners) ---
    with gcol2:
        # Identify available columns for volunteer hours and partners
        vol_col = None
        if 'vol_hours' in df_trend_eng.columns:
            vol_col = 'vol_hours'
        elif 'volunteer_hours' in df_trend_eng.columns:
            vol_col = 'volunteer_hours'
        partners_col = 'partners' if 'partners' in df_trend_eng.columns else None

        if (vol_col or partners_col) and not df_trend_eng.empty:
            fig_eng = make_subplots(specs=[[{"secondary_y": True}]])
            x_vals = df_trend_eng["year"]
            if vol_col:
                fig_eng.add_trace(
                    go.Bar(
                        x=x_vals,
                        y=df_trend_eng[vol_col],
                        name="Volunteer Hours",
                        marker_color="#667eea"
                    ),
                    secondary_y=False
                )
            if partners_col:
                fig_eng.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=df_trend_eng[partners_col],
                        name="Partners",
                        mode='lines+markers',
                        line=dict(color="#f59e0b", width=3),
                        marker=dict(size=6)
                    ),
                    secondary_y=True
                )
            fig_eng.update_layout(
                title=f"Community Engagement Trends — {selected_county}",
                height=380,
                hovermode='x unified'
            )
            fig_eng.update_xaxes(title_text="Year")
            if vol_col:
                fig_eng.update_yaxes(title_text="Volunteer Hours", secondary_y=False)
            if partners_col:
                fig_eng.update_yaxes(title_text="Number of Partners", secondary_y=True)
            st.plotly_chart(fig_eng, use_container_width=True)
        else:
            st.info("Community engagement metrics are not available for this selection.")

    # --- Graph 3: Cumulative Impact Trends ---
    with gcol3:
        if not df_trend_cum.empty and {'cum_litter', 'cum_recycled'}.issubset(df_trend_cum.columns):
            x_vals = df_trend_cum["year"]
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=x_vals,
                y=df_trend_cum['cum_litter'],
                name="Cumulative Litter",
                mode='lines+markers',
                line=dict(color="#e6550d", width=3)
            ))
            fig_cum.add_trace(go.Scatter(
                x=x_vals,
                y=df_trend_cum['cum_recycled'],
                name="Cumulative Recycled",
                mode='lines+markers',
                line=dict(color="#31a354", width=3)
            ))
            fig_cum.update_layout(
                title=f"Cumulative Impact Trends — {selected_county}",
                xaxis_title="Year",
                yaxis_title="Cumulative Pounds (lbs)",
                height=380,
                hovermode='x unified'
            )
            st.plotly_chart(fig_cum, use_container_width=True)
        else:
            st.info("Cumulative impact data not available for this selection.")

# =====================================================
# TAB 3 — COMPARISONS
# =====================================================
with tab_compare:
    st.markdown("## County Comparisons")

    col1, col2 = st.columns([1.3, 2.7], gap="large")

    # Controls from gpt.py
    with col1:
        year_cmp = st.selectbox("Year (for comparisons)", years, index=len(years) - 1, key="year_cmp")
        metric_cmp_label = st.selectbox("Comparison Metric", metric_labels, index=0, key="metric_cmp")
        metric_cmp = metric_label_to_def[metric_cmp_label]
        n = st.slider("How many counties to show?", 5, 30, 10, step=1)
        order = st.radio("Order", ["Top", "Bottom"], horizontal=True)

    # Plots from dashboard_fixed.py
    with col2:
        d = df_map[df_map["year"] == year_cmp].copy()
        mcol = metric_cmp["col"]
        if mcol not in d.columns:
            st.warning(f"Selected metric '{metric_cmp_label}' not found for comparisons.")
        else:
            d = d.sort_values(mcol, ascending=(order == "Bottom")).head(n)

            fig_bar = px.bar(
                d,
                x="county",
                y=mcol,
                title=f"{order} {n} Counties — {metric_cmp_label} ({year_cmp})",
            )
            fig_bar.update_layout(height=520, xaxis_title="", yaxis_title="")
            st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")
    st.markdown("### Efficiency Scatter (if available)")
    
    d = df_map[df_map["year"] == year].copy()

    scatter_opts = []
    if has_cols(d, ["litter", "vol_hours"]):
        scatter_opts.append(("Litter vs Volunteer Hours", "vol_hours", "litter"))
    if has_cols(d, ["litter", "county_rd_miles"]):
        scatter_opts.append(("Litter vs County Road Miles", "county_rd_miles", "litter"))
    if has_cols(d, ["litter_per_county_mile", "vol_hours"]):
        scatter_opts.append(("Litter per Mile vs Volunteer Hours", "vol_hours", "litter_per_county_mile"))

    if not scatter_opts:
        st.info("Scatter views require columns like litter, vol_hours, county_rd_miles, or derived efficiency metrics.")
    else:
        opt_label = st.selectbox("Scatter View", [o[0] for o in scatter_opts])
        xcol, ycol = [(o[1], o[2]) for o in scatter_opts if o[0] == opt_label][0]

        fig_sc = px.scatter(
            d,
            x=xcol,
            y=ycol,
            hover_name="county",
            title=f"{opt_label} ({year})",
        )
        fig_sc.update_layout(height=520, xaxis_title=xcol, yaxis_title=ycol)
        st.plotly_chart(fig_sc, use_container_width=True)

# =====================================================
# TAB 4 — REGIONAL (entire tab from dashboard_fixed.py)
# =====================================================
with tab_regional:
    st.markdown("## Regional Analysis")
    
    st.markdown("""
    Tennessee is divided into three regions: East, Middle, and West.  
    This view compares performance and trends across these regions.
    """)
    
    # Define regional groupings
    east_tn = [
        'Anderson', 'Blount', 'Bradley', 'Campbell', 'Carter', 'Claiborne',
        'Cocke', 'Grainger', 'Greene', 'Hamblen', 'Hamilton', 'Hancock',
        'Hawkins', 'Jefferson', 'Johnson', 'Knox', 'Loudon', 'McMinn',
        'Meigs', 'Monroe', 'Morgan', 'Polk', 'Rhea', 'Roane', 'Scott',
        'Sevier', 'Sullivan', 'Unicoi', 'Union', 'Washington'
    ]
    middle_tn = [
        'Bedford', 'Cannon', 'Cheatham', 'Clay', 'Coffee', 'Davidson',
        'DeKalb', 'Dickson', 'Fentress', 'Franklin', 'Giles', 'Grundy',
        'Hickman', 'Houston', 'Humphreys', 'Jackson', 'Lawrence', 'Lewis',
        'Lincoln', 'Macon', 'Marshall', 'Maury', 'Montgomery', 'Moore',
        'Overton', 'Perry', 'Pickett', 'Putnam', 'Robertson', 'Rutherford',
        'Sequatchie', 'Smith', 'Stewart', 'Sumner', 'Trousdale', 'Van Buren',
        'Warren', 'Wayne', 'White', 'Williamson', 'Wilson'
    ]
    west_tn = [
        'Benton', 'Carroll', 'Chester', 'Crockett', 'Decatur', 'Dyer',
        'Fayette', 'Gibson', 'Hardeman', 'Hardin', 'Haywood', 'Henderson',
        'Henry', 'Lake', 'Lauderdale', 'Madison', 'McNairy', 'Obion',
        'Shelby', 'Tipton', 'Weakley'
    ]
    
    df_regional = df_map[df_map["year"] == year].copy()
    
    def assign_region(county):
        if any(e in county for e in east_tn):
            return 'East Tennessee'
        elif any(m in county for m in middle_tn):
            return 'Middle Tennessee'
        elif any(w in county for w in west_tn):
            return 'West Tennessee'
        else:
            return 'Other'
    
    df_regional['region'] = df_regional['county'].apply(assign_region)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Regional totals
        if 'litter' in df_regional.columns:
            agg_dict = {'litter': 'sum'}
            if 'recycled' in df_regional.columns:
                agg_dict['recycled'] = 'sum'
            if 'dumps' in df_regional.columns:
                agg_dict['dumps'] = 'sum'
                
            regional_totals = df_regional.groupby('region').agg(agg_dict).reset_index()
            
            y_cols = ['litter']
            if 'recycled' in regional_totals.columns:
                y_cols.append('recycled')
            
            fig_regional = px.bar(
                regional_totals,
                x='region',
                y=y_cols,
                title='Regional Comparison - Litter & Recycling',
                barmode='group',
                color_discrete_map={'litter': '#e6550d', 'recycled': '#31a354'}
            )
            fig_regional.update_layout(height=400)
            fig_regional.update_xaxes(title="Region")
            fig_regional.update_yaxes(title="Pounds (lbs)")
            
            st.plotly_chart(fig_regional, use_container_width=True)
    
    with col2:
        # Regional averages
        if 'litter' in df_regional.columns:
            regional_avg = df_regional.groupby('region')['litter'].mean().reset_index()
            
            fig_avg = px.bar(
                regional_avg,
                x='region',
                y='litter',
                title='Average Litter per County by Region',
                color='litter',
                color_continuous_scale='Reds'
            )
            fig_avg.update_layout(height=400, showlegend=False)
            fig_avg.update_xaxes(title="Region")
            fig_avg.update_yaxes(title="Average Lbs per County")
            
            st.plotly_chart(fig_avg, use_container_width=True)
    
    # Year-over-year regional trends
    if 'litter' in df_map.columns:
        st.markdown("### Regional Trends Over Time")
        
        df_all_years = df_map.copy()
        df_all_years['region'] = df_all_years['county'].apply(assign_region)
        
        regional_yearly = df_all_years.groupby(['year', 'region'])['litter'].sum().reset_index()
        
        fig_regional_trend = px.line(
            regional_yearly,
            x='year',
            y='litter',
            color='region',
            markers=True,
            title='Litter Collection Trends by Region',
            line_shape='linear'
        )
        fig_regional_trend.update_layout(height=400)
        fig_regional_trend.update_xaxes(title="Year")
        fig_regional_trend.update_yaxes(title="Total Litter (lbs)")
        
        st.plotly_chart(fig_regional_trend, use_container_width=True)

# =====================================================
# TAB 5 — PRIORITIES (entire tab from gpt.py)
# =====================================================
with tab_priorities:
    st.markdown("## Priority Counties (Decision Support)")

    st.write(
        """
This section flags counties using simple, transparent rules.
You can tune thresholds to match TDOT's definition of "priority."
"""
    )

    d = df_map[df_map["year"] == year].copy()

    # Build a scoring frame only using available columns
    score_parts = []

    # Default thresholds (can be adjusted via UI)
    c1, c2, c3 = st.columns(3)
    with c1:
        use_litter_intensity = st.checkbox("Use litter intensity", value=("litter_per_county_mile" in d.columns))
        t_lpm = st.number_input("Threshold: Litter per county mile (lbs/mi)", value=float(np.nanmedian(d["litter_per_county_mile"])) if "litter_per_county_mile" in d.columns else 0.0, step=0.1)
    with c2:
        use_low_engagement = st.checkbox("Use low engagement", value=("vol_hours" in d.columns))
        t_vh = st.number_input("Threshold: Volunteer hours (below)", value=float(np.nanmedian(d["vol_hours"])) if "vol_hours" in d.columns else 0.0, step=1.0)
    with c3:
        use_dumps = st.checkbox("Use dump sites", value=("dumps" in d.columns))
        t_dump = st.number_input("Threshold: Dump sites (above)", value=float(np.nanmedian(d["dumps"])) if "dumps" in d.columns else 0.0, step=1.0)

    # Compute flags
    d["flag_litter_intensity"] = False
    d["flag_low_engagement"] = False
    d["flag_dumps"] = False

    if use_litter_intensity and "litter_per_county_mile" in d.columns:
        d["flag_litter_intensity"] = d["litter_per_county_mile"] >= t_lpm
        score_parts.append("flag_litter_intensity")

    if use_low_engagement and "vol_hours" in d.columns:
        d["flag_low_engagement"] = d["vol_hours"] <= t_vh
        score_parts.append("flag_low_engagement")

    if use_dumps and "dumps" in d.columns:
        d["flag_dumps"] = d["dumps"] >= t_dump
        score_parts.append("flag_dumps")

    if not score_parts:
        st.info("No priority rules could be applied because required columns are missing.")
    else:
        d["priority_score"] = d[score_parts].sum(axis=1)

        topn = st.slider("Show top N priority counties", 5, 30, 12, step=1)
        d_top = d.sort_values(["priority_score", "county"], ascending=[False, True]).head(topn)

        # Table
        show_cols = ["county", "priority_score"]
        for c in ["litter", "litter_per_county_mile", "recycling_rate", "dumps", "vol_hours", "partners"]:
            if c in d_top.columns:
                show_cols.append(c)

        st.dataframe(d_top[show_cols], use_container_width=True, height=380)

        # Bar chart of score
        fig_p = px.bar(
            d_top.sort_values("priority_score", ascending=True),
            x="priority_score",
            y="county",
            orientation="h",
            title=f"Priority Score — Top {topn} Counties ({year})",
        )
        fig_p.update_layout(height=520, xaxis_title="Priority score (count of flags)", yaxis_title="")
        st.plotly_chart(fig_p, use_container_width=True)

# =====================================================
# TAB 6 — SUMMARY (entire tab from dashboard_fixed.py)
# =====================================================
with tab_summary:
    st.markdown("## Key Performance Indicators")

    if 'litter' not in df_state.columns:
        st.error("Required metrics not found in state data")
    else:
        row = df_state[df_state["year"] == year].iloc[0]
        
        # Top KPIs
        k1, k2, k3, k4 = st.columns(4)

        k1.metric("Total Litter (lbs)", fmt(row["litter"]) if 'litter' in row else "N/A")
        k2.metric("Recycled (lbs)", fmt(row["recycled"]) if 'recycled' in row else "N/A")
        k3.metric("Dump Sites", fmt(row["dumps"]) if 'dumps' in row else "N/A")
        k4.metric("Partners", fmt(row["partners"]) if 'partners' in row else "N/A")
        
        st.markdown("---")
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 5-Year Summary Statistics")
            
            total_litter = df_state["litter"].sum() if 'litter' in df_state.columns else 0
            total_recycled = df_state["recycled"].sum() if 'recycled' in df_state.columns else 0
            avg_partners = df_state["partners"].mean() if 'partners' in df_state.columns else 0
            
            st.markdown(f"""
            - **Total Litter Collected**: {fmt(total_litter)} lbs
            - **Total Recycled**: {fmt(total_recycled)} lbs  
            - **Average Partners/Year**: {int(avg_partners)}
            - **Years of Data**: {len(years)}
            - **Counties Covered**: {df_map['county'].nunique() if 'county' in df_map.columns else 'N/A'}
            """)
            
            # Environmental impact
            st.markdown("### 🌱 Environmental Impact")
            trucks_filled = total_litter / 4000  # Assuming 4000 lbs per truck
            recycle_rate = (total_recycled/total_litter*100) if total_litter > 0 else 0
            st.markdown(f"""
            - **Equivalent Garbage Trucks**: ~{int(trucks_filled)} trucks filled
            - **Recycling Rate**: {recycle_rate:.1f}% of collected litter recycled
            """)
        
        with col2:
            st.markdown("### 🎯 Program Highlights")
            
            if 'litter' in df_state.columns:
                # Best performing year
                best_year = df_state.loc[df_state["litter"].idxmax()]
                
                # Best performing county
                best_county = df_map.loc[df_map["litter"].idxmax()]
                
                st.markdown(f"""
                **Best Year for Collection**  
                {best_year['year']} - {fmt(best_year['litter'])} lbs collected
                
                **Top Performing County (All-Time)**  
                {best_county['county']} - {fmt(best_county['litter'])} lbs in {best_county['year']}
                """)
                
                # Growth metrics
                if len(years) > 1:
                    first_year_data = df_state[df_state["year"] == years[0]].iloc[0]
                    last_year_data = df_state[df_state["year"] == years[-1]].iloc[0]
                    growth = ((last_year_data["litter"] - first_year_data["litter"]) / first_year_data["litter"] * 100) if first_year_data["litter"] > 0 else 0
                    
                    st.markdown(f"""
                    **Program Growth**  
                    {growth:+.1f}% change from {years[0]} to {years[-1]}
                    """)
        
        # Data table
        st.markdown("### 📋 Detailed County Data")
        
        # Build display columns list based on what exists
        possible_cols = ['county', 'litter', 'recycled', 'dumps', 'vol_hours', 'county_rd_miles', 'state_rd_miles', 'partners']
        display_cols = [col for col in possible_cols if col in df_map.columns]
        
        if len(display_cols) > 0:
            df_table = df_map[df_map["year"] == year][display_cols].copy()
            if 'litter' in df_table.columns:
                df_table = df_table.sort_values('litter', ascending=False)
            
            # Format numeric columns
            format_dict = {}
            for col in df_table.columns:
                if col != 'county' and pd.api.types.is_numeric_dtype(df_table[col]):
                    if 'mile' in col.lower():
                        format_dict[col] = '{:,.1f}'
                    else:
                        format_dict[col] = '{:,.0f}'
            
            st.dataframe(
                df_table.style.format(format_dict),
                use_container_width=True,
                height=400
            )
            
            # Download option
            csv = df_table.to_csv(index=False)
            st.download_button(
                label="📥 Download County Data (CSV)",
                data=csv,
                file_name=f"tn_litter_data_{year}.csv",
                mime="text/csv"
            )
        else:
            st.error("No displayable columns found in dataset")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption("Tennessee Department of Transportation | Litter & Community Impact Dashboard | Data reflects volunteer-reported activities")
