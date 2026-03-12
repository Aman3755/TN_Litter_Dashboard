import json
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import streamlit as st
import numpy as np

# =====================================================
# GLOBAL CONFIG
# =====================================================


pio.templates.default = "plotly_dark"

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
GEOJSON_FILE = DATA / "tn_counties.geojson"
GEOJSON_KEY = "NAME"

# =====================================================
# STYLING
# =====================================================
st.markdown("""
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

.overview-text h2 {
  margin-top: 0;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA & NORMALIZE COLUMN NAMES
# =====================================================
df_state = pd.read_csv(STATE_FILE)
df_map = pd.read_csv(MAP_FILE)
geojson = json.load(open(GEOJSON_FILE, "r", encoding="utf-8"))

# Normalize column names - create a mapping for common variations
def normalize_columns(df):
    """Normalize column names to standard format"""
    df = df.copy()
    col_mapping = {}
    
    for col in df.columns:
        col_lower = col.lower().strip()
        
        # Litter columns
        if 'litter' in col_lower and 'lb' in col_lower:
            col_mapping[col] = 'litter'
        # Recycled columns  
        elif 'recycl' in col_lower and 'lb' in col_lower:
            col_mapping[col] = 'recycled'
        # Dump sites
        elif 'dump' in col_lower:
            col_mapping[col] = 'dumps'
        # Volunteer hours
        elif 'vol' in col_lower and 'hour' in col_lower:
            col_mapping[col] = 'volunteer_hours'
        # Partners
        elif 'partner' in col_lower:
            col_mapping[col] = 'partners'
        # County road miles
        elif 'county' in col_lower and ('road' in col_lower or 'rd' in col_lower) and 'mile' in col_lower:
            col_mapping[col] = 'county_road_miles'
        # State road miles
        elif 'state' in col_lower and ('road' in col_lower or 'rd' in col_lower) and 'mile' in col_lower:
            col_mapping[col] = 'state_road_miles'
        # County name
        elif 'county' in col_lower and 'mile' not in col_lower and 'road' not in col_lower:
            col_mapping[col] = 'county'
        # Year
        elif col_lower in ['year', 'yr']:
            col_mapping[col] = 'year'
    
    df = df.rename(columns=col_mapping)
    return df

# Apply normalization
df_state = normalize_columns(df_state)
df_map = normalize_columns(df_map)

# Clean county names
if 'county' in df_map.columns:
    df_map["county"] = df_map["county"].str.strip()
    # Remove any "statewide total" rows accidentally included in county-year file
    bad_county_labels = {"state", "tennessee", "tn", "statewide", "all counties", "total"}
    df_map = df_map[~df_map["county"].str.strip().str.lower().isin(bad_county_labels)]

# Get years
if 'year' in df_state.columns:
    years = sorted(df_state["year"].unique())
else:
    st.error("Year column not found in state data. Please check your CSV file.")
    st.stop()

# =====================================================
# HELPER FUNCTIONS
# =====================================================
def fmt(x):
    """Format numbers with K/M suffixes"""
    if pd.isna(x):
        return "N/A"
    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if x >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{int(x)}"

def calculate_efficiency_metrics(df):
    """Calculate efficiency metrics for counties"""
    df = df.copy()
    
    # Calculate lbs per hour if volunteer hours exist
    if 'volunteer_hours' in df.columns:
        df['lbs_per_hour'] = df['litter'] / df['volunteer_hours'].replace(0, np.nan)
        df['volunteers_estimated'] = df['volunteer_hours'] / 4
    
    # Calculate lbs per mile if road miles exist
    total_miles = pd.Series(0, index=df.index)
    if 'county_road_miles' in df.columns:
        total_miles = total_miles + df['county_road_miles'].fillna(0)
    if 'state_road_miles' in df.columns:
        total_miles = total_miles + df['state_road_miles'].fillna(0)
    
    if total_miles.sum() > 0:
        df['lbs_per_mile'] = df['litter'] / total_miles.replace(0, np.nan)
    
    return df

# =====================================================
# HEADER
# =====================================================
st.markdown("""
<div class="header">
  <h1>🌲 Tennessee Statewide Litter, Recycling & Community Impact Dashboard</h1>
</div>
""", unsafe_allow_html=True)

# =====================================================
# GLOBAL CONTROLS
# =====================================================
c1, c2 = st.columns([1.2, 1.8])

with c1:
    year = st.selectbox("Select Year", years, index=len(years)-1)

with c2:
    metric = st.radio(
        "Map Metric",
        ["litter", "recycled", "dumps"],
        horizontal=True
    )

# =====================================================
# TABS
# =====================================================
tab_overview, tab_trends, tab_compare, tab_efficiency, tab_regional, tab_summary = st.tabs(
    ["🗺️ Overview", "📈 Trends", "📊 Comparisons", "⚡ Efficiency", "🌍 Regional", "📌 Summary"]
)

# =====================================================
# TAB 1 — OVERVIEW (COUNTY MAP HERO)
# =====================================================
with tab_overview:
    left, right = st.columns([1, 2.3], gap="large")

    # ---- LEFT TEXT
    with left:
        st.markdown('<div class="panel overview-text">', unsafe_allow_html=True)
        st.markdown("## Overview")
        st.write("""
This dashboard provides a statewide view of litter collection,
recycling efforts, and dump site activity across Tennessee.

The **county map** is the primary overview, allowing users to visually
identify spatial patterns and regional differences for the selected year.
""")
        
        # Year-over-year change
        if len(years) > 1:
            try:
                current_idx = years.index(year)
                if current_idx > 0:
                    prev_year = years[current_idx - 1]
                    current = df_state[df_state["year"] == year].iloc[0]
                    previous = df_state[df_state["year"] == prev_year].iloc[0]
                    
                    if 'litter' in df_state.columns and 'recycled' in df_state.columns:
                        litter_change = ((current["litter"] - previous["litter"]) / previous["litter"] * 100) if previous["litter"] > 0 else 0
                        recycle_change = ((current["recycled"] - previous["recycled"]) / previous["recycled"] * 100) if previous["recycled"] > 0 else 0
                        
                        st.markdown("### Year-over-Year Changes")
                        col1, col2 = st.columns(2)
                        col1.metric("Litter Collection", fmt(current["litter"]), f"{litter_change:+.1f}%")
                        col2.metric("Recycling", fmt(current["recycled"]), f"{recycle_change:+.1f}%")
            except (ValueError, IndexError, KeyError):
                pass
        
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- RIGHT MAP
    with right:
        if 'county' not in df_map.columns or metric not in df_map.columns:
            st.error(f"Required columns not found. Looking for: county and {metric}")
            st.info(f"Available columns: {', '.join(df_map.columns)}")
        else:
            d = df_map[df_map["year"] == year].copy()
            values = d[metric]

            # Stable bins
            max_val = values.max() if len(values) > 0 and values.max() > 0 else 1
            bins = [0, 0.2*max_val, 0.4*max_val, 0.6*max_val, 0.8*max_val, max_val]
            labels = ["Very Low", "Low", "Medium", "High", "Very High"]
            d["Intensity"] = pd.cut(values, bins=bins, labels=labels, include_lowest=True)

            palette = (
                ["#fff5eb", "#fdd0a2", "#fdae6b", "#e6550d", "#a63603"]
                if metric == "litter" else
                ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"]
                if metric == "recycled" else
                ["#eff3ff", "#bdd7e7", "#6baed6", "#3182bd", "#08519c"]
            )

            hover_data_dict = {"Intensity": False}
            if 'litter' in d.columns:
                hover_data_dict['litter'] = ":,.0f"
            if 'recycled' in d.columns:
                hover_data_dict['recycled'] = ":,.0f"
            if 'dumps' in d.columns:
                hover_data_dict['dumps'] = True

            fig_map = px.choropleth_mapbox(
                d,
                geojson=geojson,
                locations="county",
                featureidkey=f"properties.{GEOJSON_KEY}",
                color="Intensity",
                hover_name="county",
                hover_data=hover_data_dict,
                mapbox_style="carto-positron",
                zoom=5.8,
                center={"lat": 35.75, "lon": -86.4},
                color_discrete_sequence=palette,
                category_orders={"Intensity": labels}
            )

            fig_map.update_layout(
                height=650,
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    title="Intensity",
                    bgcolor="rgba(255,255,255,0.8)",
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
    st.markdown("## Statewide Trends Over Time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Multi-line trend
        if 'litter' in df_state.columns:
            fig_trends = go.Figure()
            
            fig_trends.add_trace(go.Scatter(
                x=df_state["year"],
                y=df_state["litter"],
                name="Litter Collected",
                mode='lines+markers',
                line=dict(color="#e6550d", width=3),
                marker=dict(size=8)
            ))
            
            if 'recycled' in df_state.columns:
                fig_trends.add_trace(go.Scatter(
                    x=df_state["year"],
                    y=df_state["recycled"],
                    name="Recycled",
                    mode='lines+markers',
                    line=dict(color="#31a354", width=3),
                    marker=dict(size=8)
                ))
            
            fig_trends.update_layout(
                title="Litter Collection & Recycling Trends",
                xaxis_title="Year",
                yaxis_title="Pounds (lbs)",
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.info("Litter data not available for trend analysis")
    
    with col2:
        # Volunteer and partner trends
        if 'volunteer_hours' in df_state.columns or 'partners' in df_state.columns:
            fig_vol = make_subplots(specs=[[{"secondary_y": True}]])
            
            if 'volunteer_hours' in df_state.columns:
                fig_vol.add_trace(
                    go.Bar(x=df_state["year"], y=df_state["volunteer_hours"], 
                           name="Volunteer Hours", marker_color="#667eea"),
                    secondary_y=False
                )
            
            if 'partners' in df_state.columns:
                fig_vol.add_trace(
                    go.Scatter(x=df_state["year"], y=df_state["partners"], 
                              name="Partners", mode='lines+markers',
                              line=dict(color="#f59e0b", width=3), marker=dict(size=8)),
                    secondary_y=True
                )
            
            fig_vol.update_layout(
                title="Community Engagement Trends",
                height=400,
                hovermode='x unified'
            )
            fig_vol.update_xaxes(title_text="Year")
            fig_vol.update_yaxes(title_text="Volunteer Hours", secondary_y=False)
            fig_vol.update_yaxes(title_text="Number of Partners", secondary_y=True)
            
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("Volunteer/Partner data not available")
    
    # Cumulative Impact
    if 'litter' in df_state.columns:
        st.markdown("### Cumulative Impact")
        df_cumulative = df_state.sort_values("year").copy()
        df_cumulative["cumulative_litter"] = df_cumulative["litter"].cumsum()
        if 'recycled' in df_cumulative.columns:
            df_cumulative["cumulative_recycled"] = df_cumulative["recycled"].cumsum()
        
        fig_cumulative = go.Figure()
        
        fig_cumulative.add_trace(go.Scatter(
            x=df_cumulative["year"],
            y=df_cumulative["cumulative_litter"],
            name="Cumulative Litter",
            fill='tozeroy',
            line=dict(color="#e6550d", width=2)
        ))
        
        if 'cumulative_recycled' in df_cumulative.columns:
            fig_cumulative.add_trace(go.Scatter(
                x=df_cumulative["year"],
                y=df_cumulative["cumulative_recycled"],
                name="Cumulative Recycled",
                fill='tozeroy',
                line=dict(color="#31a354", width=2)
            ))
        
        fig_cumulative.update_layout(
            title="Cumulative Environmental Impact (5-Year Period)",
            xaxis_title="Year",
            yaxis_title="Total Pounds (lbs)",
            height=350,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_cumulative, use_container_width=True)

# =====================================================
# TAB 3 — COMPARISONS
# =====================================================
with tab_compare:
    st.markdown("## County Comparison")
    
    if metric not in df_map.columns:
        st.error(f"Metric '{metric}' not found in data")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Top performers
            top = (
                df_map[df_map["year"] == year]
                .sort_values(metric, ascending=False)
                .head(15)
            )

            fig_top = px.bar(
                top,
                x=metric,
                y="county",
                orientation='h',
                title=f"Top 15 Counties by {metric.title()} ({year})",
                color=metric,
                color_continuous_scale="Viridis"
            )
            fig_top.update_layout(height=500, showlegend=False)
            fig_top.update_xaxes(title=f"{metric.title()} (lbs)" if metric != "dumps" else "Number of Dumps")
            fig_top.update_yaxes(title="")
            
            st.plotly_chart(fig_top, use_container_width=True)
        
        with col2:
            st.markdown("### Bottom Performers")
            st.caption("Counties with lowest activity may need additional support")
            
            bottom = (
                df_map[df_map["year"] == year]
                .sort_values(metric, ascending=True)
                .head(10)
            )
            
            for idx, row in bottom.iterrows():
                st.markdown(f"**{row['county']}**: {fmt(row[metric])}")
        
        # Multi-metric comparison
        if 'litter' in df_map.columns and 'recycled' in df_map.columns:
            st.markdown("### Multi-Metric County Performance")
            
            top_multi = df_map[df_map["year"] == year].sort_values("litter", ascending=False).head(10)
            
            fig_multi = go.Figure()
            
            fig_multi.add_trace(go.Bar(
                name='Litter',
                x=top_multi['county'],
                y=top_multi['litter'],
                marker_color='#e6550d'
            ))
            
            fig_multi.add_trace(go.Bar(
                name='Recycled',
                x=top_multi['county'],
                y=top_multi['recycled'],
                marker_color='#31a354'
            ))
            
            fig_multi.update_layout(
                barmode='group',
                height=400,
                xaxis_title="County",
                yaxis_title="Pounds (lbs)",
                title="Litter vs Recycling - Top 10 Counties"
            )
            
            st.plotly_chart(fig_multi, use_container_width=True)

# =====================================================
# TAB 4 — EFFICIENCY METRICS
# =====================================================
with tab_efficiency:
    st.markdown("## Efficiency & Productivity Analysis")
    
    # Calculate efficiency metrics
    df_eff = calculate_efficiency_metrics(df_map[df_map["year"] == year])
    
    # Check if we have the necessary columns
    has_hour_efficiency = 'lbs_per_hour' in df_eff.columns
    has_mile_efficiency = 'lbs_per_mile' in df_eff.columns
    
    if not has_hour_efficiency and not has_mile_efficiency:
        st.warning("⚠️ Efficiency metrics require volunteer hours and/or road miles data.")
        st.info("Available columns: " + ", ".join(df_eff.columns.tolist()))
        st.write("To enable efficiency metrics, ensure your data includes:")
        st.write("- Volunteer Hours (for lbs/hour calculations)")
        st.write("- County Road Miles and/or State Road Miles (for lbs/mile calculations)")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            # Lbs per volunteer hour
            if has_hour_efficiency:
                df_plot = df_eff.dropna(subset=['lbs_per_hour']).sort_values('lbs_per_hour', ascending=False).head(15)
                
                if len(df_plot) > 0:
                    fig_eff1 = px.bar(
                        df_plot,
                        x='lbs_per_hour',
                        y='county',
                        orientation='h',
                        title='Most Efficient Counties (Lbs per Volunteer Hour)',
                        color='lbs_per_hour',
                        color_continuous_scale='RdYlGn'
                    )
                    fig_eff1.update_layout(height=500, showlegend=False)
                    fig_eff1.update_xaxes(title="Lbs per Hour")
                    fig_eff1.update_yaxes(title="")
                    
                    st.plotly_chart(fig_eff1, use_container_width=True)
                else:
                    st.info("No volunteer hour data available.")
            else:
                st.info("💡 Add volunteer hours data to see efficiency metrics")
        
        with col2:
            # Lbs per mile
            if has_mile_efficiency:
                df_plot2 = df_eff.dropna(subset=['lbs_per_mile']).sort_values('lbs_per_mile', ascending=False).head(15)
                
                if len(df_plot2) > 0:
                    fig_eff2 = px.bar(
                        df_plot2,
                        x='lbs_per_mile',
                        y='county',
                        orientation='h',
                        title='Highest Litter Density (Lbs per Mile Covered)',
                        color='lbs_per_mile',
                        color_continuous_scale='Oranges'
                    )
                    fig_eff2.update_layout(height=500, showlegend=False)
                    fig_eff2.update_xaxes(title="Lbs per Mile")
                    fig_eff2.update_yaxes(title="")
                    
                    st.plotly_chart(fig_eff2, use_container_width=True)
                else:
                    st.info("No road miles data available.")
            else:
                st.info("💡 Add road miles data to see density metrics")
        
        # Scatter plot: effort vs impact
        if has_hour_efficiency and 'volunteer_hours' in df_eff.columns:
            st.markdown("### Effort vs Impact Analysis")
            
            df_scatter = df_eff.dropna(subset=['volunteer_hours', 'litter'])
            
            if len(df_scatter) > 0:
                fig_scatter = px.scatter(
                    df_scatter,
                    x='volunteer_hours',
                    y='litter',
                    size='litter',
                    color='lbs_per_hour',
                    hover_name='county',
                    title='Volunteer Hours vs Litter Collected',
                    color_continuous_scale='Viridis',
                    labels={
                        'volunteer_hours': 'Total Volunteer Hours',
                        'litter': 'Litter Collected (lbs)',
                        'lbs_per_hour': 'Efficiency (lbs/hr)'
                    }
                )
                
                fig_scatter.update_layout(height=450)
                st.plotly_chart(fig_scatter, use_container_width=True)

# =====================================================
# TAB 5 — REGIONAL ANALYSIS
# =====================================================
with tab_regional:
    st.markdown("## Regional Analysis")
    st.info("💡 Counties are grouped by region to identify geographic patterns")
    
    if 'county' not in df_map.columns:
        st.error("County column not found in data")
    else:
        # Define TN regions
        east_tn = [
            'Anderson', 'Bledsoe', 'Blount', 'Bradley', 'Campbell', 'Carter',
            'Claiborne', 'Cocke', 'Cumberland', 'Grainger', 'Greene', 'Hamblen',
            'Hamilton', 'Hancock', 'Hawkins', 'Jefferson', 'Johnson', 'Knox',
            'Loudon', 'Marion', 'McMinn', 'Meigs', 'Monroe', 'Morgan', 'Polk',
            'Rhea', 'Roane', 'Scott', 'Sevier', 'Sullivan', 'Unicoi', 'Union',
            'Washington']
        middle_tn = [
            'Bedford', 'Cannon', 'Cheatham', 'Clay', 'Coffee', 'Davidson',
            'DeKalb', 'Dickson', 'Fentress', 'Franklin', 'Giles', 'Grundy',
            'Hickman', 'Houston', 'Humphreys', 'Jackson', 'Lawrence', 'Lewis',
            'Lincoln', 'Macon', 'Marshall', 'Maury', 'Montgomery', 'Moore',
            'Overton', 'Perry', 'Pickett', 'Putnam', 'Robertson', 'Rutherford',
            'Sequatchie', 'Smith', 'Stewart', 'Sumner', 'Trousdale', 'Van Buren',
            'Warren', 'Wayne', 'White', 'Williamson', 'Wilson']
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
# TAB 6 — SUMMARY / KPIs
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
        possible_cols = ['county', 'litter', 'recycled', 'dumps', 'volunteer_hours', 'county_road_miles', 'state_road_miles', 'partners']
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
