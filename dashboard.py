from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "submission_outputs"
FIGURES = OUTPUTS / "figures"
FORMAT_PREFIXES = {"IPL": "ipl", "T20I": "t20i"}

# Detect which formats have both batter and bowler ranking files available.
AVAILABLE_FORMATS = {}
for _fmt in FORMAT_PREFIXES:
    _prefix = FORMAT_PREFIXES[_fmt]
    if (OUTPUTS / f"{_prefix}_batter_rankings.csv").exists() and (OUTPUTS / f"{_prefix}_bowler_rankings.csv").exists():
        AVAILABLE_FORMATS[_fmt] = True

# Try to load sensitivity checks; hide the section if the file is absent.
try:
    sensitivity_data = load_sensitivity()
    SENSITIVITY_AVAILABLE = True
except (OSError, pd.errors.ParserError):
    sensitivity_data = None
    SENSITIVITY_AVAILABLE = False

st.set_page_config(
    page_title="T20 Pressure Performance",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 2.5rem; max-width: 1500px;}
    [data-testid="stMetric"] {background: #f4f7fa; border: 1px solid #e1e8ef;
        padding: 0.9rem 1rem; border-radius: 0.8rem;}
    [data-testid="stMetricLabel"] {color: #536579;}
    h1, h2, h3 {color: #183b56;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_rankings(format_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    prefix = FORMAT_PREFIXES[format_name]
    batters = pd.read_csv(OUTPUTS / f"{prefix}_batter_rankings.csv")
    bowlers = pd.read_csv(OUTPUTS / f"{prefix}_bowler_rankings.csv")
    for frame in (batters, bowlers):
        if "eligible" in frame:
            frame["eligible"] = frame["eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
        frame.sort_values("rank", inplace=True)
    return batters, bowlers


@st.cache_data
def load_sensitivity() -> pd.DataFrame:
    return pd.read_csv(OUTPUTS / "sensitivity_checks.csv")


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


st.title("T20 Pressure Performance Dashboard")
st.caption(
    "A transparent scouting view of pressure-versus-ordinary outcomes. IPL and T20I are analyzed separately."
)

with st.sidebar:
    st.header("Dashboard controls")
    _format_options = list(AVAILABLE_FORMATS.keys()) if AVAILABLE_FORMATS else ["IPL"]
    selected_format = st.selectbox("Competition format", _format_options)

try:
    batter_data, bowler_data = load_rankings(selected_format)
except (OSError, pd.errors.ParserError) as error:
    st.error(f"Could not load the analysis outputs: {error}")
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

top_batters = batter_data.head(10)
top_bowlers = bowler_data.head(10)
ipl_available = "IPL" in AVAILABLE_FORMATS
t20i_available = "T20I" in AVAILABLE_FORMATS
if ipl_available:
    ipl_batters, ipl_bowlers = load_rankings("IPL")
if t20i_available:
    t20i_batters, t20i_bowlers = load_rankings("T20I")

if ipl_available and t20i_available:
    batter_overlap = len(set(ipl_batters.head(10)["name"]) & set(t20i_batters.head(10)["name"]))
    bowler_overlap = len(set(ipl_bowlers.head(10)["bowler_name"]) & set(t20i_bowlers.head(10)["bowler_name"]))
else:
    batter_overlap = 0
    bowler_overlap = 0

st.info(
    "Rates use delivery-event rows, not verified legal balls. The available fields do not support exact score-state, required-run-rate, or economy calculations."
)

overview_tab, batting_tab, bowling_tab, methodology_tab = st.tabs(
    ["Overview", "Batting", "Bowling", "Methodology & reliability"]
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
    st.caption("Low overlap is a warning against treating performance rankings as transferable across formats.")

    if selected_format == "IPL" and SENSITIVITY_AVAILABLE:
        st.markdown("#### Sensitivity checks")
        selected_sensitivity = sensitivity_data[sensitivity_data["format_group"].eq("IPL")].copy()
        selected_sensitivity["top10_jaccard"] = selected_sensitivity["top10_jaccard"].round(2)
        st.dataframe(selected_sensitivity, hide_index=True, width="stretch")

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

with methodology_tab:
    st.subheader("How to read the rankings")
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

            **Shrinkage:** rates use a 30-row prior toward state- or phase-level averages. Eligibility requires at least 30 pressure rows across 3 or more matches and at least 30 ordinary rows.
            """
        )
    st.warning(
        "These are observable proxies, not exact match leverage. The source data lack reliable innings totals and a documented legal-ball indicator; interpret denominators as event rows."
    )

    flow_image = FIGURES / "asp_methodology_flow.png"
    states_image = FIGURES / "asp_pressure_states.png"
    if flow_image.exists():
        st.image(str(flow_image), caption="Analysis workflow", width="stretch")
    if states_image.exists():
        st.image(str(states_image), caption="Operational pressure states", width="stretch")

    if selected_format == "IPL" and SENSITIVITY_AVAILABLE:
        st.markdown("#### IPL threshold sensitivity")
        st.dataframe(
            sensitivity_data[sensitivity_data["format_group"].eq("IPL")],
            hide_index=True,
            width="stretch",
        )

st.caption("Source: supplied analysis outputs. Treat rankings as descriptive evidence, not causal estimates or guaranteed forecasts.")