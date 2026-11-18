from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pressure_proxy import analyze_all


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
FORMAT_NAMES = ("IPL", "T20I")

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
    """Generated analysis-sequence figure (replaces missing asp_methodology_flow.png)."""
    nodes = [
        ("Raw event data", "ball_by_ball.csv\nmatches.csv\ncompetitions.csv", 0, 0),
        ("Prepare deliveries", "Filter innings 1–2,\nexclude revised-overs matches", 1, 0),
        ("Pressure states", "Batter: collapse / death\nBowler: partnership > 25", 2, 0),
        ("Outcome metrics", "Survival, boundary,\ncredited wickets per phase", 3, 0),
        ("Shrinkage (κ=30)", "Shrink toward state/phase\npooled rates", 4, 0),
        ("Ranking & filters", "Pareto front,\neligibility ≥30 rows / 3 matches", 5, 0),
    ]
    fig = go.Figure()
    for i, (label, desc, x, y) in enumerate(nodes):
        fig.add_trace(
            go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=40, color="#183b56" if i in (0, 5) else "#e07a36"),
                text=label,
                textposition="bottom center",
                textfont=dict(color="white", size=11),
                hoverinfo="text",
                hovertext=f"{label}<br>{desc}",
                showlegend=False,
            )
        )
        if i < len(nodes) - 1:
            fig.add_annotation(
                x=x + 0.5, y=y,
                ax=x, ay=y,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="#667788",
            )
    fig.update_xaxes(range=[-0.7, 5.7], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(range=[-0.6, 0.6], showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(
        height=260,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


st.title("T20 Pressure Performance Dashboard")
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
