from __future__ import annotations

from pathlib import Path
import base64

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pressure_proxy import analyze_all


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
FORMAT_NAMES = ("IPL", "T20I")


def load_image_b64(filename: str) -> str:
    """Load an image file and return a base64 data URI."""
    path = ROOT / filename
    if not path.exists():
        return ""
    data = path.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"


RR_LOGO_B64 = load_image_b64("images/Rajasthan-Royals-Logo-768x432.png")
IPL_LOGO_B64 = load_image_b64("images/ipl-logo.png")

st.set_page_config(
    page_title="T20 Pressure Performance",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1rem; padding-bottom: 2.5rem; max-width: 1600px;}
    h1, h2, h3 {color: #ffffff;}
    .stMetric {background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155; padding: 1rem 1.25rem; border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .stMetric [data-testid="stMetricValue"] {color: #fbbf24 !important; font-size: 1.75rem;}
    .stMetric [data-testid="stMetricLabel"] {color: #94a3b8 !important;}
    .top-banner {background: linear-gradient(90deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        border: 2px solid #4f46e5; border-radius: 16px; padding: 2.5rem 3rem;
        margin: 1rem 0 3rem 0; box-shadow: 0 20px 60px rgba(15, 23, 42, 0.6);
        display: flex; align-items: center; justify-content: space-between; gap: 2rem;}
    .top-logo {width: 160px; height: auto; filter: drop-shadow(0 6px 12px rgba(0,0,0,0.5));}
    .banner-content h1 {color: #fbbf24; margin: 0; font-size: 3rem; letter-spacing: 3px;}
    .banner-content .subtitle {color: #cbd5e1; margin-top: 0.75rem; font-size: 1.25rem;
        letter-spacing: 1.5px; text-transform: uppercase;}
    .rr-badge {display: inline-block; background: #fbbf24; color: #1e1b4b;
        font-weight: 700; padding: 4px 12px; border-radius: 6px; font-size: 1rem;
        margin-left: 12px; letter-spacing: 2px;}
    .tab-card {background: #0f172a; border: 1px solid #334155; border-radius: 10px;
        padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.3);}
    .section-title {color: #fbbf24; font-size: 1.25rem; font-weight: 600;
        margin-bottom: 1rem; letter-spacing: 0.5px; text-transform: uppercase;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Load analysis (computed dynamically from raw event data)
# ---------------------------------------------------------------------------
try:
    rankings_by_format, sensitivity_data = analyze_all(str(DATA_DIR))
except (OSError, pd.errors.ParserError, KeyError, ValueError) as error:
    st.error(f"Could not build the analysis from the input data: {error}")
    st.stop()


def make_batter_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "rank", "name", "pressure_deliveries", "pressure_matches",
        "survival_ordinary", "survival_pressure", "survival_delta",
        "boundary_ordinary", "boundary_pressure", "boundary_delta",
    ]
    table = frame[columns].copy()
    percentage_columns = [
        "survival_ordinary", "survival_pressure", "survival_delta",
        "boundary_ordinary", "boundary_pressure", "boundary_delta",
    ]
    table[percentage_columns] = table[percentage_columns] * 100
    return table.rename(
        columns={
            "rank": "Rank",
            "name": "Batter",
            "pressure_deliveries": "Pressure rows",
            "pressure_matches": "Pressure matches",
            "survival_ordinary": "Survival ordinary (%)",
            "survival_pressure": "Survival pressure (%)",
            "survival_delta": "Survival change (pp)",
            "boundary_ordinary": "Boundary ordinary (%)",
            "boundary_pressure": "Boundary pressure (%)",
            "boundary_delta": "Boundary change (pp)",
        }
    )


def make_bowler_table(frame: pd.DataFrame) -> pd.DataFrame:
    table = frame[
        [
            "rank", "bowler_name", "pressure_deliveries", "pressure_matches",
            "pressure_wickets", "ordinary_wickets_per_100",
            "pressure_wickets_per_100", "delta_per_100",
        ]
    ].copy()
    return table.rename(
        columns={
            "rank": "Rank",
            "bowler_name": "Bowler",
            "pressure_deliveries": "Pressure rows",
            "pressure_matches": "Pressure matches",
            "pressure_wickets": "Pressure wickets",
            "ordinary_wickets_per_100": "Ordinary rate (/100 rows)",
            "pressure_wickets_per_100": "Pressure rate (/100 rows)",
            "delta_per_100": "Change (/100 rows)",
        }
    )


def batter_figure(frame: pd.DataFrame) -> go.Figure:
    chart = frame.copy()
    chart["survival_delta_pp"] = chart["survival_delta"] * 100
    chart["boundary_delta_pp"] = chart["boundary_delta"] * 100
    chart["selection"] = chart["rank"].le(10).map({True: "Top 10", False: "Other eligible"})
    chart["label"] = chart["name"].where(chart["rank"].le(10), "")
    figure = px.scatter(
        chart,
        x="survival_delta_pp",
        y="boundary_delta_pp",
        color="selection",
        text="label",
        hover_name="name",
        hover_data={
            "rank": True,
            "pressure_deliveries": True,
            "pressure_matches": True,
            "survival_delta_pp": ":.2f",
            "boundary_delta_pp": ":.2f",
            "selection": False,
            "label": False,
        },
        category_orders={"selection": ["Other eligible", "Top 10"]},
        color_discrete_map={"Other eligible": "#9aafbd", "Top 10": "#e07a36"},
        labels={
            "survival_delta_pp": "Survival change (percentage points)",
            "boundary_delta_pp": "Boundary change (percentage points)",
            "selection": "Selection",
            "rank": "Rank",
            "pressure_deliveries": "Pressure rows",
            "pressure_matches": "Pressure matches",
        },
    )
    figure.update_traces(marker={"size": 11, "line": {"width": 1, "color": "white"}})
    figure.update_traces(textposition="top center", selector={"mode": "markers+text"})
    figure.add_hline(y=0, line_dash="dot", line_color="#667788", opacity=0.7)
    figure.add_vline(x=0, line_dash="dot", line_color="#667788", opacity=0.7)
    figure.update_layout(
        height=560,
        margin={"l": 15, "r": 20, "t": 25, "b": 20},
        legend_title_text="",
        hovermode="closest",
    )
    return figure


def bowler_figures(frame: pd.DataFrame, display_rows: int) -> tuple[go.Figure, go.Figure]:
    visible = frame.head(display_rows).copy()
    ordered = visible.sort_values("delta_per_100", ascending=True)
    differential = px.bar(
        ordered,
        x="delta_per_100",
        y="bowler_name",
        orientation="h",
        color="delta_per_100",
        color_continuous_scale=["#b84d4d", "#e5b85c", "#2c8c78"],
        hover_data={
            "rank": True,
            "pressure_deliveries": True,
            "pressure_matches": True,
            "pressure_wickets": True,
            "delta_per_100": ":.2f",
        },
        labels={
            "delta_per_100": "Credited-wicket change per 100 rows",
            "bowler_name": "Bowler",
        },
    )
    differential.update_traces(texttemplate="%{x:.2f}", textposition="outside", cliponaxis=False)
    differential.update_layout(
        height=max(440, 38 * len(ordered)),
        margin={"l": 15, "r": 55, "t": 25, "b": 20},
        coloraxis_showscale=False,
        yaxis_title="",
        xaxis_zeroline=True,
        xaxis_zerolinecolor="#667788",
    )

    rates = go.Figure()
    rates.add_trace(
        go.Bar(
            name="Ordinary",
            x=ordered["ordinary_wickets_per_100"],
            y=ordered["bowler_name"],
            orientation="h",
            marker_color="#7393a7",
            hovertemplate="%{y}<br>Ordinary: %{x:.2f} per 100 rows<extra></extra>",
        )
    )
    rates.add_trace(
        go.Bar(
            name="Pressure",
            x=ordered["pressure_wickets_per_100"],
            y=ordered["bowler_name"],
            orientation="h",
            marker_color="#e07a36",
            hovertemplate="%{y}<br>Pressure: %{x:.2f} per 100 rows<extra></extra>",
        )
    )
    rates.update_layout(
        barmode="group",
        height=max(440, 38 * len(ordered)),
        margin={"l": 15, "r": 20, "t": 25, "b": 20},
        xaxis_title="Credited wickets per 100 delivery rows",
        yaxis_title="",
        legend_title_text="",
    )
    return differential, rates


def methodology_flow_figure() -> go.Figure:
    """Detailed analysis architecture figure with batter/bowler split,
    key parameters, color-coded stages, and sensitivity feedback loop."""
    fig = go.Figure()

    # ---------- Row 0: Data ----------
    fig.add_trace(
        go.Scatter(
            x=[0], y=[0], mode="markers+text",
            marker=dict(size=42, color="#1f3a5f"),
            text="Raw event data", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="ball_by_ball.csv<br>matches.csv<br>competitions.csv",
            hoverinfo="text", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[1.6], y=[0], mode="markers+text",
            marker=dict(size=42, color="#1f3a5f"),
            text="Prepare deliveries", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="Filter innings 1–2<br>Exclude revised-overs matches<br>86,668 IPL / 102,887 T20I rows",
            hoverinfo="text", showlegend=False,
        )
    )

    # ---------- Row 1: Pressure states (batter / bowler split) ----------
    fig.add_trace(
        go.Scatter(
            x=[0.6], y=[-1.6], mode="markers+text",
            marker=dict(size=42, color="#e07a36"),
            text="Batter pressure", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="Overs 1–6 with ≥2 wickets lost<br>OR overs 16–20 (death)",
            hoverinfo="text", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[2.6], y=[-1.6], mode="markers+text",
            marker=dict(size=42, color="#e07a36"),
            text="Bowler pressure", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="Partnership event-row index > 25<br>Any wicket resets the counter",
            hoverinfo="text", showlegend=False,
        )
    )

    # ---------- Row 2: Outcome metrics ----------
    fig.add_trace(
        go.Scatter(
            x=[0.6], y=[-3.1], mode="markers+text",
            marker=dict(size=42, color="#f0ad4e"),
            text="Batter outcomes", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="Survival rate (non-dismissal)<br>Boundary rate (4s & 6s)",
            hoverinfo="text", showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[2.6], y=[-3.1], mode="markers+text",
            marker=dict(size=42, color="#f0ad4e"),
            text="Bowler outcomes", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="Credited wickets per phase<br>(powerplay / middle / death)",
            hoverinfo="text", showlegend=False,
        )
    )

    # ---------- Row 3: Shrinkage ----------
    fig.add_trace(
        go.Scatter(
            x=[1.6], y=[-4.6], mode="markers+text",
            marker=dict(size=42, color="#2c8c78"),
            text="Shrinkage (κ=30)", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="Shrink toward state/phase pooled rates<br>30 pseudo-rows prior",
            hoverinfo="text", showlegend=False,
        )
    )

    # ---------- Row 4: Eligibility ----------
    fig.add_trace(
        go.Scatter(
            x=[1.6], y=[-6.1], mode="markers+text",
            marker=dict(size=42, color="#8e44ad"),
            text="Eligibility", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="≥30 pressure rows across ≥3 matches<br>+ ≥30 ordinary rows<br>(IPL: 126 batters / 129 bowlers<br>T20I: 198 batters / 186 bowlers)",
            hoverinfo="text", showlegend=False,
        )
    )

    # ---------- Row 5: Ranking ----------
    fig.add_trace(
        go.Scatter(
            x=[1.6], y=[-7.6], mode="markers+text",
            marker=dict(size=42, color="#34495e"),
            text="Pareto ranking", textposition="bottom center",
            textfont=dict(color="white", size=10),
            hovertext="ΔSurvival & ΔBoundary (batters)<br>ΔW per 100 rows (bowlers)<br>Positive-both → Pareto front order",
            hoverinfo="text", showlegend=False,
        )
    )

    # ---------- Arrows ----------
    arrows = [
        # Data row
        (0.0, 0, 1.6, 0),
        # Split to batter & bowler
        (1.6, 0, 0.6, -1.6),
        (1.6, 0, 2.6, -1.6),
        # Batter & bowler to outcomes
        (0.6, -1.6, 0.6, -3.1),
        (2.6, -1.6, 2.6, -3.1),
        # Outcomes to shrinkage
        (0.6, -3.1, 1.6, -4.6),
        (2.6, -3.1, 1.6, -4.6),
        # Shrinkage to eligibility to ranking
        (1.6, -4.6, 1.6, -6.1),
        (1.6, -6.1, 1.6, -7.6),
        # Sensitivity feedback loop (ranking back to pressure states)
        (1.6, -7.6, 1.6, -1.6),
    ]
    for x0, y0, x1, y1 in arrows:
        fig.add_annotation(
            x=x1, y=y1, ax=x0, ay=y0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor="#667788",
        )

    # Sensitivity loop label (curved dashed arrow text)
    fig.add_annotation(
        x=0.3, y=-4.6,
        text="Sensitivity<br>feedback",
        showarrow=False,
        font=dict(color="#c0392b", size=9),
        bgcolor="rgba(192,57,43,0.1)",
        bordercolor="#c0392b",
    )

    # Stage band labels
    stage_labels = [
        (0.8, 0.7, "DATA", "#1f3a5f"),
        (1.6, -0.9, "PRESSURE STATES", "#e07a36"),
        (1.6, -2.4, "OUTCOME METRICS", "#f0ad4e"),
        (1.6, -3.9, "SHRINKAGE", "#2c8c78"),
        (1.6, -5.4, "ELIGIBILITY", "#8e44ad"),
        (1.6, -6.9, "RANKING", "#34495e"),
    ]
    for sx, sy, slabel, scolor in stage_labels:
        fig.add_annotation(
            x=sx, y=sy, text=slabel, showarrow=False,
            font=dict(color=scolor, size=8, family="Arial Black"),
            bgcolor="rgba(255,255,255,0.05)",
            bordercolor=scolor,
            borderwidth=1,
            borderpad=2,
        )

    fig.update_xaxes(range=[-0.9, 3.5], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-8.4, 1.2], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        height=620,
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


st.markdown(
    f"""
    <div class="top-banner">
        <img src="{RR_LOGO_B64}" class="top-logo" alt="RR Logo">
        <div class="banner-content">
            <h1>T20 Pressure Performance Dashboard <span class="rr-badge">RAJASTHAN ROYALS</span></h1>
            <div class="subtitle">Pressure-Proxy Scouting System · IPL &amp; T20I · Rajasthan Royals Analytics</div>
        </div>
        <img src="{IPL_LOGO_B64}" class="top-logo" alt="IPL Logo">
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    f"🔗 **Source code:** [Aathish14/superr-pressure-dashboard](https://github.com/Aathish14/superr-pressure-dashboard) "
    f"(GitHub repository)"
)
st.markdown("**Author:** Aathish Rao A B — Rajasthan Royals")
st.caption(
    "A transparent scouting view of pressure-versus-ordinary outcomes. IPL and T20I are analyzed separately."
)

# ---------------------------------------------------------------------------
# Data scope & integrity notes
# ---------------------------------------------------------------------------
with st.expander("Data scope & integrity decisions", expanded=False):
    st.markdown(
        """
- **Matches:** 371 IPL + 482 T20I matches supplied. Matches with revised-overs or
  revised-target metadata are excluded (5 IPL, 29 T20I). Only regulation innings 1
  and 2 are retained; Super Overs and other T20 competitions are omitted.
- **Delivery rows:** 86,668 IPL and 102,887 T20I delivery-event rows. These are
  event-row counts, **not** official legal-ball totals.
- **Integrity decisions:**
  - `extras1_runs` behaves as an event-category code, not an extras-run amount; it is
    never summed as runs.
  - 36 nonzero `extras_2_runs` values are unresolved; that field is not used as a run
    total.
  - The `runs` field does not reproduce innings totals in reviewed scorecard examples,
    and no innings totals or provider-documented legal-ball flag are available. Exact
    score, required run rate, and economy are therefore **not reconstructed**.
  - Bowler wicket credit is limited to Bowled, Caught, Caught and Bowled, Hit Wicket,
    LBW, and Stump. A non-striker run-out is not counted as a striker dismissal.
    """
    )

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Dashboard controls")
    selected_format = st.selectbox("Competition format", list(FORMAT_NAMES))

try:
    batter_data, bowler_data = rankings_by_format[selected_format]
except KeyError:
    st.error(f"No ranking data available for {selected_format}.")
    st.stop()

batter_data = batter_data[batter_data["eligible"]].copy()
bowler_data = bowler_data[bowler_data["eligible"]].copy()
if batter_data.empty or bowler_data.empty:
    st.error(f"No eligible ranking rows were found for {selected_format}.")
    st.stop()

maximum_rows = max(10, len(batter_data), len(bowler_data))
with st.sidebar:
    display_rows = st.slider(
        "Rows in ranking tables and bowler charts",
        min_value=10,
        max_value=maximum_rows,
        value=10,
        step=1,
    )

    st.markdown("---")
    st.markdown("### 👑 RR Scout View")
    rr_mode = st.toggle("Show RR scout targets only", value=False)
    if rr_mode:
        st.caption("High pressure-delivery players with positive deltas — prime auction targets.")
        rr_min_deliveries = st.slider("Min pressure deliveries", 30, 200, 50)
        batter_data = batter_data[
            (batter_data["pressure_deliveries"] >= rr_min_deliveries) &
            (batter_data["survival_delta"] > 0) &
            (batter_data["boundary_delta"] > 0)
        ]
        bowler_data = bowler_data[
            (bowler_data["pressure_deliveries"] >= rr_min_deliveries) &
            (bowler_data["delta_per_100"] > 0)
        ]

top_batters = batter_data.head(10)
top_bowlers = bowler_data.head(10)

ipl_batters, ipl_bowlers = rankings_by_format["IPL"]
t20i_batters, t20i_bowlers = rankings_by_format["T20I"]
batter_overlap = len(set(ipl_batters.head(10)["name"]) & set(t20i_batters.head(10)["name"]))
bowler_overlap = len(set(ipl_bowlers.head(10)["bowler_name"]) & set(t20i_bowlers.head(10)["bowler_name"]))

st.info(
    "Rates use delivery-event rows, not verified legal balls. The available fields do not support exact score-state, required-run-rate, or economy calculations."
)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
overview_tab, batting_tab, bowling_tab, comparison_tab, novelty_tab, methodology_tab = st.tabs(
    ["Overview", "Batting", "Bowling", "Compare Players", "Novelty & Method", "Methodology & reliability"]
)

with overview_tab:
    st.subheader(f"{selected_format} overview")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Eligible batters", f"{len(batter_data):,}")
    metric_columns[1].metric("Eligible bowlers", f"{len(bowler_data):,}")
    metric_columns[2].metric("Top batter", top_batters.iloc[0]["name"])
    metric_columns[3].metric("Top bowler", top_bowlers.iloc[0]["bowler_name"])

    st.markdown("#### Leading selections")
    batter_column, bowler_column = st.columns(2)
    with batter_column:
        st.markdown("**Top batters**")
        overview_batters = make_batter_table(top_batters)[
            ["Rank", "Batter", "Pressure rows", "Pressure matches", "Survival change (pp)", "Boundary change (pp)"]
        ]
        st.dataframe(
            overview_batters.style.format(
                {"Survival change (pp)": "+.2f", "Boundary change (pp)": "+.2f"}
            ),
            hide_index=True,
            width="stretch",
        )
    with bowler_column:
        st.markdown("**Top bowlers**")
        st.dataframe(
            make_bowler_table(top_bowlers).style.format(
                {
                    "Ordinary rate (/100 rows)": ".2f",
                    "Pressure rate (/100 rows)": ".2f",
                    "Change (/100 rows)": "+.2f",
                }
            ),
            hide_index=True,
            width="stretch",
        )

    st.markdown("#### Cross-format top-ten overlap")
    overlap_columns = st.columns(2)
    overlap_columns[0].metric("Batters shared by IPL and T20I", f"{batter_overlap}/10")
    overlap_columns[1].metric("Bowlers shared by IPL and T20I", f"{bowler_overlap}/10")
    st.caption(
        "Low overlap is a warning against treating performance rankings as transferable across formats. "
        "IPL and T20I state pools, estimates, and candidate filters are calculated independently; "
        "the two formats are not pooled because their roles and match environments differ."
    )

    if selected_format == "IPL":
        st.markdown("#### Sensitivity checks (IPL)")
        sens_ipl = sensitivity_data[sensitivity_data["format_group"].eq("IPL")].copy()
        sens_ipl["top10_jaccard"] = sens_ipl["top10_jaccard"].round(3)
        sens_display = sens_ipl.rename(
            columns={
                "analysis": "Analysis",
                "variant": "Alternative pressure definition",
                "qualified_candidates": "Qualified candidates",
                "top10_overlap_with_base": "Overlap with base top-10",
                "top10_jaccard": "Jaccard index",
            }
        )
        st.dataframe(sens_display, hide_index=True, width="stretch")
        st.caption(
            "Jaccard is the intersection divided by the union of the base and alternative top-ten lists. "
            "Lower overlap indicates greater sensitivity to the analyst-defined pressure cutoff."
        )

    download_columns = st.columns(2)
    download_columns[0].download_button(
        "Download batter rankings CSV",
        data=batter_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_format.lower()}_batter_rankings.csv",
        mime="text/csv",
    )
    download_columns[1].download_button(
        "Download bowler rankings CSV",
        data=bowler_data.to_csv(index=False).encode("utf-8"),
        file_name=f"{selected_format.lower()}_bowler_rankings.csv",
        mime="text/csv",
    )

with batting_tab:
    st.subheader(f"{selected_format} batting performance")
    st.markdown(
        "The scatterplot compares each eligible batter’s pressure-minus-ordinary survival and boundary-rate changes. "
        "The highlighted labels are the reported top ten."
    )
    st.plotly_chart(batter_figure(batter_data), width="stretch")
    st.markdown(f"#### Ranked batters (showing {min(display_rows, len(batter_data))} of {len(batter_data)})")
    st.dataframe(
        make_batter_table(batter_data.head(display_rows)).style.format(
            {
                "Survival ordinary (%)": ".2f",
                "Survival pressure (%)": ".2f",
                "Survival change (pp)": "+.2f",
                "Boundary ordinary (%)": ".2f",
                "Boundary pressure (%)": ".2f",
                "Boundary change (pp)": "+.2f",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with bowling_tab:
    st.subheader(f"{selected_format} bowling performance")
    st.markdown(
        "The differential chart ranks credited wickets per 100 event rows; the paired bars compare ordinary and pressure rates."
    )
    differential_figure, rate_figure = bowler_figures(bowler_data, display_rows)
    st.plotly_chart(differential_figure, width="stretch")
    st.plotly_chart(rate_figure, width="stretch")
    st.markdown(f"#### Ranked bowlers (showing {min(display_rows, len(bowler_data))} of {len(bowler_data)})")
    st.dataframe(
        make_bowler_table(bowler_data.head(display_rows)).style.format(
            {
                "Ordinary rate (/100 rows)": ".2f",
                "Pressure rate (/100 rows)": ".2f",
                "Change (/100 rows)": "+.2f",
            }
        ),
        hide_index=True,
        width="stretch",
    )

with comparison_tab:
    st.subheader("Head-to-Head Player Comparison")
    st.markdown(
        "Select any two eligible players to compare their pressure-versus-ordinary metrics side by side."
    )

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        player_a = st.selectbox("Player A", batter_data["name"].tolist(), key="comp_a")
    with comp_col2:
        player_b = st.selectbox("Player B", batter_data["name"].tolist(), key="comp_b")

    if player_a and player_b:
        row_a = batter_data[batter_data["name"] == player_a].iloc[0]
        row_b = batter_data[batter_data["name"] == player_b].iloc[0]

        # Comparison metrics radar
        categories = ["Survival Δ (pp)", "Boundary Δ (pp)", "Pressure Deliveries", "Pressure Matches"]
        values_a = [
            row_a["survival_delta"] * 100,
            row_a["boundary_delta"] * 100,
            min(row_a["pressure_deliveries"] / max(batter_data["pressure_deliveries"].max(), 1) * 100, 100),
            min(row_a["pressure_matches"] / max(batter_data["pressure_matches"].max(), 1) * 100, 100),
        ]
        values_b = [
            row_b["survival_delta"] * 100,
            row_b["boundary_delta"] * 100,
            min(row_b["pressure_deliveries"] / max(batter_data["pressure_deliveries"].max(), 1) * 100, 100),
            min(row_b["pressure_matches"] / max(batter_data["pressure_matches"].max(), 1) * 100, 100),
        ]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatterpolar(
            r=values_a + [values_a[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=player_a,
            line=dict(color="#fbbf24", width=3),
            fillcolor="rgba(251, 191, 36, 0.25)",
        ))
        fig_comp.add_trace(go.Scatterpolar(
            r=values_b + [values_b[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name=player_b,
            line=dict(color="#38bdf8", width=3),
            fillcolor="rgba(56, 189, 248, 0.25)",
        ))
        fig_comp.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155"),
                angularaxis=dict(gridcolor="#334155"),
                bgcolor="#0f172a",
            ),
            paper_bgcolor="#0f172a",
            plot_bgcolor="#0f172a",
            font=dict(color="#ffffff", size=11),
            legend=dict(bgcolor="#0f172a", font=dict(color="#cbd5e1")),
            height=450,
            margin=dict(l=60, r=60, t=40, b=40),
        )
        st.plotly_chart(fig_comp, width="stretch")

        # Side-by-side metrics table (all values as strings for Arrow compatibility)
        comp_table = pd.DataFrame({
            "Metric": [
                "Rank", "Pressure rows", "Pressure matches", "Survival ordinary (%)",
                "Survival pressure (%)", "Survival Δ (pp)", "Boundary ordinary (%)",
                "Boundary pressure (%)", "Boundary Δ (pp)",
            ],
            player_a: [
                str(int(row_a["rank"])), str(int(row_a["pressure_deliveries"])), str(int(row_a["pressure_matches"])),
                f"{row_a['survival_ordinary']*100:.2f}", f"{row_a['survival_pressure']*100:.2f}",
                f"{row_a['survival_delta']*100:+.2f}", f"{row_a['boundary_ordinary']*100:.2f}",
                f"{row_a['boundary_pressure']*100:.2f}", f"{row_a['boundary_delta']*100:+.2f}",
            ],
            player_b: [
                str(int(row_b["rank"])), str(int(row_b["pressure_deliveries"])), str(int(row_b["pressure_matches"])),
                f"{row_b['survival_ordinary']*100:.2f}", f"{row_b['survival_pressure']*100:.2f}",
                f"{row_b['survival_delta']*100:+.2f}", f"{row_b['boundary_ordinary']*100:.2f}",
                f"{row_b['boundary_pressure']*100:.2f}", f"{row_b['boundary_delta']*100:+.2f}",
            ],
        })
        st.dataframe(
            comp_table.style.set_properties(**{"background-color": "#0f172a", "color": "#ffffff"}),
            hide_index=True,
            width="stretch",
        )

with novelty_tab:
    st.subheader("Novelty & Methodological Contribution")
    st.markdown(
        """
        > **Methodological contribution.** This work does not claim to introduce a new statistical theory of pressure.
        > Its contribution is an auditable, role-aware scouting workflow suited to the limitations of the supplied
        > delivery data. It defines separate batting and bowling pressure proxies, compares pressure outcomes with
        > player-specific ordinary baselines, applies shrinkage and multi-match eligibility rules, adjusts bowling
        > comparisons for over phase, and reports threshold sensitivity and cross-format results. The method therefore
        > offers a transparent alternative to rankings based only on aggregate performance or an opaque single score,
        > while treating its results as descriptive scouting signals rather than a definitive measure of clutch ability.
        """
    )

    st.markdown("---")

    st.markdown("### What distinguishes this approach")

    novelty_col1, novelty_col2 = st.columns(2)

    with novelty_col1:
        st.markdown("#### Role-aware pressure states")
        st.markdown(
            "Pressure is not treated as one identical situation. Batters are assessed in "
            "**early-innings collapse** (overs 1–6 with ≥2 wickets lost) or **death overs** "
            "(overs 16–20); bowlers are assessed during **established partnerships** "
            "(partnership event-row index > 25)."
        )

        st.markdown("#### Player-specific baselines")
        st.markdown(
            "The question is not who has the most boundaries or wickets overall, but whether "
            "a player's selected outcomes **improve under pressure compared with their own "
            "ordinary rows** — survival and boundary production for batters, credited-wicket "
            "rate for bowlers."
        )

    with novelty_col2:
        st.markdown("#### Distinct batting dimensions")
        st.markdown(
            "Survival (non-dismissal rate) and boundary production remain **separate ranking "
            "dimensions** rather than being collapsed into one opaque 'clutch' score."
        )

        st.markdown("#### Shrinkage & eligibility")
        st.markdown(
            "Small samples are pulled toward state/phase averages (κ=30 prior). Eligibility "
            "requires ≥30 pressure rows across ≥3 matches, and bowling comparisons are "
            "matched by over phase (powerplay / middle / death)."
        )

    st.markdown("---")

    st.markdown("### Honest scope & contrast")

    st.info(
        """
        **This is a pragmatic, auditable alternative — not a claim of invention.**

        Published T20I studies define pressure using required run rate and remaining resources,
        then use pressure indices to predict match outcomes and weight batter contributions.
        More sophisticated published approaches exist. This method does not compete with them
        on match-outcome prediction or win-probability modelling.

        **Where this method fits:** The supplied data do not reliably support exact score-state,
        required-run-rate, or legal-ball economy calculations. This workflow is designed for
        those constraints — it reconstructs what *can* be audited from the available delivery
        events, and it is transparent about where confidence is limited.

        **Confidence limits shown:**
        - IPL and T20I are analyzed **separately** (not pooled)
        - Threshold sensitivity reveals whether top-ten membership changes under alternative
          pressure definitions
        - Results are **descriptive associations**, not causal evidence of a universal clutch trait
        """
    )

    st.markdown("---")

    st.markdown("### Reference")
    st.markdown(
        "Published pressure-index approach: "
        "[Sage Open: A pressure index for T20 cricket](https://journals.sagepub.com/doi/full/10.3233/JSA-240792)"
    )

with methodology_tab:
    definitions_column, outcomes_column = st.columns(2)
    with definitions_column:
        st.markdown(
            """
            **Batting pressure**
            - Overs 1–6 after at least two wickets have fallen before the event, or overs 16–20.
            - Ordinary state: the batter’s other faced-event rows.

            **Bowling pressure**
            - Partnership event-row index greater than 25; any wicket resets the counter.
            - Ordinary outcomes are compared within powerplay, middle, and death phases.
            """
        )
    with outcomes_column:
        st.markdown(
            """
            **Batting outcomes:** non-dismissal-on-striker-event rate and flagged boundary rate. Both are compared with the same player’s ordinary-state rates.

            **Bowling outcome:** credited wickets per event row, phase-adjusted against ordinary rows.

            **Eligibility:** at least 30 pressure rows across at least 3 distinct matches and at least 30 ordinary rows.
            """
        )

    st.markdown("### Outcome metrics and ranking formulas")

    st.markdown("#### Batting shrinkage (survival & boundary)")
    st.latex(
        r"\widehat{p}_{i,s}^{(m)}=\frac{x_{i,s}^{(m)}+\kappa\,p_s^{(m)}}{n_{i,s}+\kappa},\qquad \kappa=30"
    )
    st.caption(
        r"For batter $i$ and state $s\in\{P,O\}$: $x_{i,s}^{(m)}$ is the outcome count, $n_{i,s}$ its row exposure, "
        r"and $p_s^{(m)}$ the pooled rate for that state within the format. Ranking axes are the pressure-minus-ordinary "
        r"differences $\Delta S_i=\widehat p_{i,P}^{(S)}-\widehat p_{i,O}^{(S)}$ and "
        r"$\Delta B_i=\widehat p_{i,P}^{(B)}-\widehat p_{i,O}^{(B)}$. Players positive on both axes are ordered by Pareto front."
    )

    st.markdown("#### Bowling shrinkage (credited wickets per phase)")
    st.latex(
        r"\widehat{r}_{i,g,s}=\frac{w_{i,g,s}+\kappa\,\lambda_{g,s}}{n_{i,g,s}+\kappa},\qquad \kappa=30"
    )
    st.caption(
        r"For bowler $i$, over phase $g$, and state $s$: $w_{i,g,s}$ credited wickets, $n_{i,g,s}$ delivery-row exposure, "
        r"and $\lambda_{g,s}$ the pooled phase/state rate."
    )

    st.markdown("#### Bowler weighted delta")
    st.latex(
        r"\Delta W_i=\frac{\sum_g n_{i,g,P}(\widehat{r}_{i,g,P}-\widehat{r}_{i,g,O})}{\sum_g n_{i,g,P}}"
    )
    st.caption(
        r"Phase-specific pressure-minus-ordinary changes are aggregated using each bowler’s pressure-row exposure. "
        r"Bowler rates and $\Delta W_i$ are reported as credited wickets per 100 delivery rows, not per legal ball."
    )

    st.markdown("### Analysis architecture")
    st.plotly_chart(methodology_flow_figure(), width="stretch")

    st.markdown("### Sensitivity and format check")
    st.markdown(
        "Threshold sensitivity compares each alternative top-ten list with the primary IPL list. "
        "Jaccard is the intersection divided by the union; lower overlap indicates greater sensitivity to "
        "the analyst-defined cutoff. The separately ranked T20I lists overlap the IPL top ten by "
        f"{batter_overlap}/10 batters and {bowler_overlap}/10 bowlers — limited cross-format transfer, "
        "not independent confirmation of the IPL ordering."
    )

    st.markdown("### Reliability and limitations")
    st.warning(
        r"""
        **These are observable proxies, not exact match leverage.** The pressure states capture innings phase and
        wicket/partnership context, not exact contest closeness. Across-match support is checked by requiring pressure
        exposure in at least 3 distinct matches, but no leave-one-match-out analysis or bootstrap uncertainty interval
        was calculated, so a player’s rate change may still be influenced by a small number of innings.

        Unverified extras encoding and missing legal-ball metadata mean the denominators are delivery rows;
        score, chase pressure, and economy are not inferred. The shrinkage strength ($\kappa=30$) and eligibility
        cutoffs are design choices, and the sensitivity results show that list membership changes when thresholds change.
        IPL and T20I rankings differ substantially, so recommendations are not pooled across formats.

        Results are **descriptive associations, not causal evidence** of a universal clutch trait, and should be treated
        as scouting evidence rather than guaranteed forecasts.
        """
    )

st.caption("Source: computed from supplied event data. Rankings are descriptive evidence, not causal estimates or guaranteed forecasts.")
