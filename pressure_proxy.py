from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path(".")
FORMATS = ("IPL", "T20I")
PRIOR_ROWS = 30
MIN_PRESSURE_ROWS = 30
MIN_MATCHES = 3
CREDITED_DISMISSALS = {
    "Bowled", "Caught", "Caught and Bowled", "Hit Wicket", "LBW", "Stump",
}



def load_data(data_dir: str | Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir)
    balls = pd.read_csv(data_dir / "ball_by_ball.csv")
    matches = pd.read_csv(data_dir / "matches.csv")
    competitions = pd.read_csv(data_dir / "competitions.csv")
    match_info = matches.merge(competitions, on="comp_id", how="left")
    match_info["format_group"] = np.select(
        [match_info["comp_name"].str.startswith("IPL", na=False), match_info["match_format"].eq("T20I")],
        ["IPL", "T20I"],
        default="Other",
    )
    dls_columns = ["first_inn_revised_overs", "second_inn_revised_overs", "revised_target"]
    match_info["dls_metadata_present"] = match_info[dls_columns].notna().any(axis=1)
    return balls, matches, match_info


def prepare_deliveries(balls: pd.DataFrame, match_info: pd.DataFrame) -> pd.DataFrame:
    included = match_info[
        match_info["format_group"].isin(FORMATS)
        & ~match_info["dls_metadata_present"]
    ][["match_id", "format_group", "comp_name", "season"]]
    data = balls.merge(included, on="match_id", how="inner", validate="many_to_one")
    data = data[data["innings"].isin([1, 2])].copy()
    data = data.sort_values(["match_id", "innings", "innings_ball_num"], kind="stable").reset_index(drop=True)
    innings_keys = ["match_id", "innings"]
    data["wickets_before"] = (
        data.groupby(innings_keys, sort=False)["is_wicket"].cumsum().astype(int)
        - data["is_wicket"].astype(int)
    )
    data["wide_event"] = data["extras_type"].fillna("").str.strip().str.lower().eq("wide")
    data["batter_faced_proxy"] = data["striker_id"].notna() & ~data["wide_event"]
    data["batter_dismissed_on_strike"] = (
        data["is_wicket"] & data["dismissed_player_id"].eq(data["striker_id"])
    )
    data["boundary_flag"] = data["is_four"] | data["is_six"]
    data["pressure_batting"] = (
        (data["overid"].le(6) & data["wickets_before"].ge(2))
        | data["overid"].ge(16)
    )
    previous_wickets = (
        data.groupby(innings_keys, sort=False)["is_wicket"].cumsum().astype(int)
        - data["is_wicket"].astype(int)
    )
    partnership_groups = [data["match_id"], data["innings"], previous_wickets]
    data["partnership_delivery_index"] = data.groupby(partnership_groups, sort=False).cumcount()
    data["pressure_bowling_25"] = data["partnership_delivery_index"].gt(25)
    data["credited_bowler_wicket"] = data["dismissal_type"].isin(CREDITED_DISMISSALS)
    data["bowling_phase"] = pd.cut(
        data["overid"],
        bins=[0, 6, 15, 20],
        labels=["Powerplay", "Middle", "Death"],
        include_lowest=True,
    ).astype("object")
    return data


def zscore(values: pd.Series) -> pd.Series:
    spread = values.std(ddof=0)
    if pd.isna(spread) or spread == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / spread


def pareto_front_ranks(frame: pd.DataFrame, metrics: list[str]) -> pd.Series:
    labels = frame.index.to_list()
    values = frame[metrics].to_numpy(dtype=float)
    remaining = np.arange(len(labels))
    ranks = pd.Series(index=frame.index, dtype=int)
    rank = 1
    while len(remaining):
        front = []
        for candidate in remaining:
            others = remaining[remaining != candidate]
            if len(others) == 0:
                front.append(candidate)
                continue
            dominates = (values[others] >= values[candidate]).all(axis=1) & (
                values[others] > values[candidate]
            ).any(axis=1)
            if not dominates.any():
                front.append(candidate)
        ranks.loc[[labels[index] for index in front]] = rank
        remaining = np.array([index for index in remaining if index not in front], dtype=int)
        rank += 1
    return ranks


def rank_batters(
    data: pd.DataFrame,
    format_group: str,
    pressure_mask: pd.Series,
    compute_ci: bool = False,
) -> pd.DataFrame:
    scoped = data[data["format_group"].eq(format_group)].copy()
    bat = scoped[scoped["batter_faced_proxy"]].copy()
    bat["pressure"] = pressure_mask.loc[bat.index].astype(bool)
    bat["survive_event"] = ~bat["batter_dismissed_on_strike"]
    grouped = bat.groupby(["striker_id", "striker_name", "pressure"], dropna=False).agg(
        deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        survived=("survive_event", "sum"),
        boundaries=("boundary_flag", "sum"),
    )
    pressure = grouped[grouped.index.get_level_values("pressure")].droplevel("pressure")
    ordinary = grouped[~grouped.index.get_level_values("pressure")].droplevel("pressure")
    pressure = pressure.add_prefix("pressure_")
    ordinary = ordinary.add_prefix("ordinary_")
    result = pressure.join(ordinary, how="outer").fillna(0)
    result["name"] = result.index.get_level_values("striker_name")
    result["player_id"] = result.index.get_level_values("striker_id")

    pressure_pool = bat[bat["pressure"]]
    ordinary_pool = bat[~bat["pressure"]]
    prior_pressure_survival = pressure_pool["survive_event"].mean()
    prior_ordinary_survival = ordinary_pool["survive_event"].mean()
    prior_pressure_boundary = pressure_pool["boundary_flag"].mean()
    prior_ordinary_boundary = ordinary_pool["boundary_flag"].mean()

    def shrunk(successes: pd.Series, count: pd.Series, prior: float) -> pd.Series:
        return (successes + PRIOR_ROWS * prior) / (count + PRIOR_ROWS)

    result["survival_pressure"] = shrunk(
        result["pressure_survived"], result["pressure_deliveries"], prior_pressure_survival
    )
    result["survival_ordinary"] = shrunk(
        result["ordinary_survived"], result["ordinary_deliveries"], prior_ordinary_survival
    )
    result["boundary_pressure"] = shrunk(
        result["pressure_boundaries"], result["pressure_deliveries"], prior_pressure_boundary
    )
    result["boundary_ordinary"] = shrunk(
        result["ordinary_boundaries"], result["ordinary_deliveries"], prior_ordinary_boundary
    )
    result["survival_delta"] = result["survival_pressure"] - result["survival_ordinary"]
    result["boundary_delta"] = result["boundary_pressure"] - result["boundary_ordinary"]
    result["pressure_matches"] = result["pressure_matches"].astype(int)
    result["eligible"] = (
        result["pressure_deliveries"].ge(MIN_PRESSURE_ROWS)
        & result["pressure_matches"].ge(MIN_MATCHES)
        & result["ordinary_deliveries"].ge(MIN_PRESSURE_ROWS)
    )
    eligible = result[result["eligible"]].copy()
    eligible["pareto_front"] = pareto_front_ranks(eligible, ["survival_delta", "boundary_delta"])
    eligible["positive_both"] = eligible["survival_delta"].gt(0) & eligible["boundary_delta"].gt(0)
    eligible["survival_delta_z"] = zscore(eligible["survival_delta"])
    eligible["boundary_delta_z"] = zscore(eligible["boundary_delta"])
    eligible["balanced_delta_z"] = eligible[["survival_delta_z", "boundary_delta_z"]].min(axis=1)
    eligible = eligible.sort_values(
        ["positive_both", "pareto_front", "balanced_delta_z", "pressure_deliveries", "survival_pressure", "boundary_pressure", "name"],
        ascending=[False, True, False, False, False, False, True],
    )
    if compute_ci:
        ci_rows = []
        for _, row in eligible.iterrows():
            ci = batter_bootstrap_ci(
                data, format_group, pressure_mask,
                row["player_id"], row["name"], n_bootstrap=400,
            )
            ci_rows.append({
                "survival_delta_ci_low": ci.get("survival_delta_ci", (np.nan, np.nan))[0],
                "survival_delta_ci_high": ci.get("survival_delta_ci", (np.nan, np.nan))[1],
                "boundary_delta_ci_low": ci.get("boundary_delta_ci", (np.nan, np.nan))[0],
                "boundary_delta_ci_high": ci.get("boundary_delta_ci", (np.nan, np.nan))[1],
            })
        ci_df = pd.DataFrame(ci_rows, index=eligible.index)
        eligible = pd.concat([eligible, ci_df], axis=1)
    eligible.insert(0, "rank", range(1, len(eligible) + 1))
    return eligible.reset_index(drop=True)


def rank_bowlers(
    data: pd.DataFrame,
    format_group: str,
    partner_threshold: int,
    compute_ci: bool = False,
) -> pd.DataFrame:
    scoped = data[data["format_group"].eq(format_group)].copy()
    scoped["pressure"] = scoped["partnership_delivery_index"].gt(partner_threshold)
    scoped = scoped[scoped["bowler_id"].notna() & scoped["bowling_phase"].notna()].copy()
    grouped = scoped.groupby(["bowler_id", "bowler_name", "bowling_phase", "pressure"], dropna=False).agg(
        deliveries=("match_id", "size"),
        matches=("match_id", "nunique"),
        wickets=("credited_bowler_wicket", "sum"),
    )
    pressure = grouped[grouped.index.get_level_values("pressure")].droplevel("pressure")
    ordinary = grouped[~grouped.index.get_level_values("pressure")].droplevel("pressure")
    pressure = pressure.add_prefix("pressure_")
    ordinary = ordinary.add_prefix("ordinary_")
    player_phase = pressure.join(ordinary, how="outer").fillna(0).reset_index()
    prior_table = scoped.groupby(["bowling_phase", "pressure"]).agg(
        total_wickets=("credited_bowler_wicket", "sum"), total_deliveries=("match_id", "size")
    )
    prior_rates = (prior_table["total_wickets"] / prior_table["total_deliveries"].replace(0, np.nan)).to_dict()
    player_phase["pressure_rate_prior"] = player_phase["bowling_phase"].map(
        lambda phase: prior_rates.get((phase, True), 0.0)
    )
    player_phase["ordinary_rate_prior"] = player_phase["bowling_phase"].map(
        lambda phase: prior_rates.get((phase, False), 0.0)
    )
    player_phase["rate_pressure"] = (
        player_phase["pressure_wickets"]
        + PRIOR_ROWS * player_phase["pressure_rate_prior"]
    ) / (player_phase["pressure_deliveries"] + PRIOR_ROWS)
    player_phase["rate_ordinary"] = (
        player_phase["ordinary_wickets"]
        + PRIOR_ROWS * player_phase["ordinary_rate_prior"]
    ) / (player_phase["ordinary_deliveries"] + PRIOR_ROWS)
    player_phase["rate_delta"] = player_phase["rate_pressure"] - player_phase["rate_ordinary"]

    player_phase["weighted_delta"] = player_phase["rate_delta"] * player_phase["pressure_deliveries"]
    player_phase["weighted_pressure_rate"] = player_phase["rate_pressure"] * player_phase["pressure_deliveries"]
    player_phase["weighted_ordinary_rate"] = player_phase["rate_ordinary"] * player_phase["pressure_deliveries"]
    result = player_phase.groupby(["bowler_id", "bowler_name"], as_index=False).agg(
        pressure_deliveries=("pressure_deliveries", "sum"),
        pressure_wickets=("pressure_wickets", "sum"),
        ordinary_deliveries=("ordinary_deliveries", "sum"),
        weighted_delta=("weighted_delta", "sum"),
        weighted_pressure_rate=("weighted_pressure_rate", "sum"),
        weighted_ordinary_rate=("weighted_ordinary_rate", "sum"),
    )
    pressure_match_counts = (
        scoped[scoped["pressure"]]
        .groupby(["bowler_id", "bowler_name"])["match_id"]
        .nunique()
        .rename("pressure_matches")
    )
    result = result.join(pressure_match_counts, on=["bowler_id", "bowler_name"])
    result["pressure_matches"] = result["pressure_matches"].fillna(0).astype(int)
    denom = result["pressure_deliveries"].replace(0, np.nan)
    result["wicket_rate_pressure"] = result["weighted_pressure_rate"] / denom
    result["wicket_rate_ordinary"] = result["weighted_ordinary_rate"] / denom
    result["wicket_rate_delta"] = result["weighted_delta"] / denom
    result["pressure_wickets_per_100"] = 100 * result["wicket_rate_pressure"]
    result["ordinary_wickets_per_100"] = 100 * result["wicket_rate_ordinary"]
    result["delta_per_100"] = 100 * result["wicket_rate_delta"]
    result["eligible"] = (
        result["pressure_deliveries"].ge(MIN_PRESSURE_ROWS)
        & result["pressure_matches"].ge(MIN_MATCHES)
        & result["ordinary_deliveries"].ge(MIN_PRESSURE_ROWS)
    )
    eligible = result[result["eligible"]].copy().sort_values(
        ["delta_per_100", "pressure_deliveries", "pressure_wickets", "bowler_name"],
        ascending=[False, False, False, True],
    )
    if compute_ci:
        ci_rows = []
        for _, row in eligible.iterrows():
            ci = bowler_bootstrap_ci(
                data, format_group, partner_threshold,
                row["bowler_id"], row["bowler_name"], n_bootstrap=400,
            )
            ci_rows.append({
                "delta_per_100_ci_low": ci.get("delta_per_100_ci", (np.nan, np.nan))[0],
                "delta_per_100_ci_high": ci.get("delta_per_100_ci", (np.nan, np.nan))[1],
            })
        ci_df = pd.DataFrame(ci_rows, index=eligible.index)
        eligible = pd.concat([eligible, ci_df], axis=1)
    eligible.insert(0, "rank", range(1, len(eligible) + 1))
    return eligible.reset_index(drop=True)


def top_names(frame: pd.DataFrame, name_col: str, count: int = 10) -> list[str]:
    return frame.head(count)[name_col].astype(str).tolist()


def sensitivity(data: pd.DataFrame, format_group: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_bat_mask = data["pressure_batting"]
    base_bat = rank_batters(data, format_group, base_bat_mask)
    base_bowl = rank_bowlers(data, format_group, 25)
    base_bat_names = set(top_names(base_bat, "name"))
    base_bowl_names = set(top_names(base_bowl, "bowler_name"))
    for label, mask in [
        ("Death from over 15", ((data.overid.ge(15)) | (data.overid.le(6) & data.wickets_before.ge(2)))),
        ("Collapse at 3+ wickets", ((data.overid.ge(16)) | (data.overid.le(6) & data.wickets_before.ge(3)))),
    ]:
        alt = rank_batters(data, format_group, mask)
        names = set(top_names(alt, "name"))
        rows.append({
            "analysis": "Batter pressure state",
            "variant": label,
            "qualified_candidates": len(alt),
            "top10_overlap_with_base": len(base_bat_names & names),
            "top10_jaccard": len(base_bat_names & names) / max(1, len(base_bat_names | names)),
        })
    alt_bowl = rank_bowlers(data, format_group, 30)
    names = set(top_names(alt_bowl, "bowler_name"))
    rows.append({
        "analysis": "Bowler partnership threshold",
        "variant": "More than 30 delivery rows",
        "qualified_candidates": len(alt_bowl),
        "top10_overlap_with_base": len(base_bowl_names & names),
        "top10_jaccard": len(base_bowl_names & names) / max(1, len(base_bowl_names | names)),
    })
    return pd.DataFrame(rows)




def analyze_all(
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]:
    balls, _, match_info = load_data(data_dir)
    deliveries = prepare_deliveries(balls, match_info)
    rankings: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    sensitivity_frames = []
    for format_group in FORMATS:
        batters = rank_batters(deliveries, format_group, deliveries["pressure_batting"])
        bowlers = rank_bowlers(deliveries, format_group, 25)
        rankings[format_group] = (batters, bowlers)
        sensitivity_frames.append(sensitivity(deliveries, format_group).assign(format_group=format_group))
    return rankings, pd.concat(sensitivity_frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Bootstrap / leave-one-match-out uncertainty
# ---------------------------------------------------------------------------
def _match_pressure_agg(
    bat: pd.DataFrame,
    pressure_mask: pd.Series,
) -> pd.DataFrame:
    """Per match x player aggregates for survival & boundary, split by pressure."""
    bat = bat.copy()
    bat["pressure"] = pressure_mask.loc[bat.index].astype(bool)
    bat["survive_event"] = ~bat["batter_dismissed_on_strike"]
    agg = bat.groupby(["striker_id", "striker_name", "match_id", "pressure"]).agg(
        deliveries=("match_id", "size"),
        survived=("survive_event", "sum"),
        boundaries=("boundary_flag", "sum"),
    ).reset_index()
    return agg


def batter_bootstrap_ci(
    data: pd.DataFrame,
    format_group: str,
    pressure_mask: pd.Series,
    player_id: object,
    player_name: str,
    n_bootstrap: int = 400,
    seed: int = 42,
    prior_rows: int = PRIOR_ROWS,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs over matches for a batter's survival & boundary delta."""
    scoped = data[data["format_group"].eq(format_group)].copy()
    bat = scoped[scoped["striker_id"].eq(player_id)].copy()
    if bat.empty:
        return {}
    bat["pressure"] = pressure_mask.loc[bat.index].astype(bool)
    bat["survive_event"] = ~bat["batter_dismissed_on_strike"]

    # Pooled priors for this format/state (same as rank_batters)
    pressure_pool = scoped[scoped["batter_faced_proxy"] & pressure_mask.loc[scoped.index]]
    ordinary_pool = scoped[scoped["batter_faced_proxy"] & ~pressure_mask.loc[scoped.index]]
    prior_survival_p = pressure_pool["survive_event"].mean() if len(pressure_pool) else 0.5
    prior_survival_o = ordinary_pool["survive_event"].mean() if len(ordinary_pool) else 0.5
    prior_boundary_p = pressure_pool["boundary_flag"].mean() if len(pressure_pool) else 0.1
    prior_boundary_o = ordinary_pool["boundary_flag"].mean() if len(ordinary_pool) else 0.1

    agg = _match_pressure_agg(bat, pressure_mask)
    pressure_matches = agg[agg["pressure"]]["match_id"].unique()
    ordinary_matches = agg[~agg["pressure"]]["match_id"].unique()

    rng = np.random.default_rng(seed)
    surv_deltas = []
    bound_deltas = []
    for _ in range(n_bootstrap):
        p_sample = rng.choice(pressure_matches, size=len(pressure_matches), replace=True)
        o_sample = rng.choice(ordinary_matches, size=len(ordinary_matches), replace=True)
        psub = agg[agg["match_id"].isin(p_sample)]
        osub = agg[agg["match_id"].isin(o_sample)]
        p_surv = psub[psub["pressure"]]["survived"].sum()
        p_del = psub[psub["pressure"]]["deliveries"].sum()
        o_surv = osub[~osub["pressure"]]["survived"].sum()
        o_del = osub[~osub["pressure"]]["deliveries"].sum()
        p_b = psub[psub["pressure"]]["boundaries"].sum()
        o_b = osub[~osub["pressure"]]["boundaries"].sum()
        s_p = (p_surv + prior_rows * prior_survival_p) / (p_del + prior_rows)
        s_o = (o_surv + prior_rows * prior_survival_o) / (o_del + prior_rows)
        b_p = (p_b + prior_rows * prior_boundary_p) / (p_del + prior_rows)
        b_o = (o_b + prior_rows * prior_boundary_o) / (o_del + prior_rows)
        surv_deltas.append(s_p - s_o)
        bound_deltas.append(b_p - b_o)

    def ci(vals: list[float]) -> tuple[float, float]:
        return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))

    return {
        "survival_delta_ci": ci(surv_deltas),
        "boundary_delta_ci": ci(bound_deltas),
    }


def _bowler_match_phase_agg(
    scoped: pd.DataFrame,
    partner_threshold: int,
) -> pd.DataFrame:
    """Per match x bowler x phase x pressure aggregates for credited wickets."""
    scoped = scoped.copy()
    scoped["pressure"] = scoped["partnership_delivery_index"].gt(partner_threshold)
    scoped = scoped[scoped["bowler_id"].notna() & scoped["bowling_phase"].notna()].copy()
    agg = scoped.groupby(["bowler_id", "bowler_name", "match_id", "bowling_phase", "pressure"]).agg(
        deliveries=("match_id", "size"),
        wickets=("credited_bowler_wicket", "sum"),
    ).reset_index()
    return agg


def bowler_bootstrap_ci(
    data: pd.DataFrame,
    format_group: str,
    partner_threshold: int,
    player_id: object,
    player_name: str,
    n_bootstrap: int = 400,
    seed: int = 42,
    prior_rows: int = PRIOR_ROWS,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CI over matches for a bowler's delta_per_100."""
    scoped = data[data["format_group"].eq(format_group)].copy()
    sub = scoped[scoped["bowler_id"].eq(player_id)].copy()
    if sub.empty:
        return {}
    sub = sub[sub["bowling_phase"].notna()].copy()
    sub["pressure"] = sub["partnership_delivery_index"].gt(partner_threshold)

    # Pooled priors per phase/state (same as rank_bowlers)
    prior_rates = {}
    for phase in sub["bowling_phase"].dropna().unique():
        for press in (True, False):
            pool = scoped[scoped["bowling_phase"].eq(phase)]
            press_pool = pool[pool["pressure"] == press]
            if len(press_pool):
                prior_rates[(phase, press)] = press_pool["credited_bowler_wicket"].mean() / max(len(press_pool), 1)
            else:
                prior_rates[(phase, press)] = 0.0

    agg = _bowler_match_phase_agg(scoped, partner_threshold)
    sub_agg = agg[agg["bowler_id"].eq(player_id)].copy()
    pressure_matches = sub_agg[sub_agg["pressure"]]["match_id"].unique()
    ordinary_matches = sub_agg[~sub_agg["pressure"]]["match_id"].unique()

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        p_sample = rng.choice(pressure_matches, size=len(pressure_matches), replace=True)
        o_sample = rng.choice(ordinary_matches, size=len(ordinary_matches), replace=True)
        psub = sub_agg[sub_agg["match_id"].isin(p_sample)]
        osub = sub_agg[sub_agg["match_id"].isin(o_sample)]
        num = 0.0
        den = 0.0
        for phase in psub["bowling_phase"].unique():
            p_row = psub[psub["bowling_phase"].eq(phase) & psub["pressure"]]
            o_row = osub[osub["bowling_phase"].eq(phase) & ~osub["pressure"]]
            p_w = int(p_row["wickets"].sum()); p_d = int(p_row["deliveries"].sum())
            o_w = int(o_row["wickets"].sum()); o_d = int(o_row["deliveries"].sum())
            p_prior = prior_rates.get((phase, True), 0.0)
            o_prior = prior_rates.get((phase, False), 0.0)
            r_p = (p_w + prior_rows * p_prior) / (p_d + prior_rows)
            r_o = (o_w + prior_rows * o_prior) / (o_d + prior_rows)
            num += (r_p - r_o) * p_d
            den += p_d
        delta = num / den if den else 0.0
        deltas.append(delta)

    return {"delta_per_100_ci": (
        float(np.percentile(deltas, 2.5) * 100),
        float(np.percentile(deltas, 97.5) * 100),
    )}


# ---------------------------------------------------------------------------
# Match-level provenance & team affiliation
# ---------------------------------------------------------------------------
def batter_match_provenance(
    data: pd.DataFrame,
    format_group: str,
    pressure_mask: pd.Series,
    player_id: object,
) -> pd.DataFrame:
    """Matches in which a batter had pressure rows, with pressure deliveries per match."""
    scoped = data[data["format_group"].eq(format_group)].copy()
    bat = scoped[scoped["striker_id"].eq(player_id)].copy()
    if bat.empty:
        return pd.DataFrame(columns=["match_id", "pressure_deliveries", "pressure_survived", "pressure_boundaries"])
    bat = bat.copy()
    bat["pressure"] = pressure_mask.loc[bat.index].astype(bool)
    bat["survive_event"] = ~bat["batter_dismissed_on_strike"]
    prov = bat[bat["pressure"]].groupby("match_id").agg(
        pressure_deliveries=("match_id", "size"),
        pressure_survived=("survive_event", "sum"),
        pressure_boundaries=("boundary_flag", "sum"),
    ).reset_index()
    return prov.sort_values("pressure_deliveries", ascending=False).reset_index(drop=True)


def bowler_match_provenance(
    data: pd.DataFrame,
    format_group: str,
    partner_threshold: int,
    player_id: object,
) -> pd.DataFrame:
    """Matches in which a bowler had pressure rows, with pressure deliveries/wickets per match."""
    scoped = data[data["format_group"].eq(format_group)].copy()
    sub = scoped[scoped["bowler_id"].eq(player_id)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["match_id", "pressure_deliveries", "pressure_wickets"])
    sub = sub.copy()
    sub["pressure"] = sub["partnership_delivery_index"].gt(partner_threshold)
    prov = sub[sub["pressure"]].groupby("match_id").agg(
        pressure_deliveries=("match_id", "size"),
        pressure_wickets=("credited_bowler_wicket", "sum"),
    ).reset_index()
    return prov.sort_values("pressure_deliveries", ascending=False).reset_index(drop=True)


def detect_rr_team_id(data: pd.DataFrame) -> object | None:
    """Infer the Rajasthan Royals batting_team_id from match/team names."""
    matches = data[["match_id", "format_group"]].drop_duplicates()
    # Lazy import to avoid circular dependency; load matches.csv fresh
    from pathlib import Path
    data_dir = Path.cwd()
    try:
        matches_full = pd.read_csv(data_dir / "matches.csv")
    except Exception:
        return None
    rr_rows = matches_full[
        matches_full["home_team"].str.contains("Rajasthan", case=False, na=False)
        | matches_full["away_team"].str.contains("Rajasthan", case=False, na=False)
    ]
    if rr_rows.empty:
        return None
    return set(rr_rows["home_team_id"]).union(set(rr_rows["away_team_id"]))


def player_is_rr(
    data: pd.DataFrame,
    format_group: str,
    player_id: object,
    role: str = "batter",
    rr_team_ids: set | None = None,
) -> bool:
    """Whether a player appeared for RR in IPL (batting or bowling team)."""
    if rr_team_ids is None:
        rr_team_ids = detect_rr_team_id(data)
    if rr_team_ids is None:
        return False
    scoped = data[data["format_group"].eq(format_group)].copy()
    if role == "batter":
        sub = scoped[scoped["striker_id"].eq(player_id)]
    else:
        sub = scoped[scoped["bowler_id"].eq(player_id)]
    if sub.empty:
        return False
    teams = set(sub["batting_team_id"].dropna()).union(set(sub["bowling_team_id"].dropna()))
    return bool(teams & rr_team_ids)


# ---------------------------------------------------------------------------
# Per-phase bowler breakdown
# ---------------------------------------------------------------------------
def bowler_phase_breakdown(
    data: pd.DataFrame,
    format_group: str,
    player_id: object,
    partner_threshold: int = 25,
) -> pd.DataFrame:
    """Per-phase ordinary/pressure credited-wicket rates for a bowler."""
    scoped = data[data["format_group"].eq(format_group)].copy()
    sub = scoped[scoped["bowler_id"].eq(player_id)].copy()
    if sub.empty:
        return pd.DataFrame()
    sub = sub[sub["bowling_phase"].notna()].copy()
    sub["pressure"] = sub["partnership_delivery_index"].gt(partner_threshold)

    grouped = sub.groupby(["bowling_phase", "pressure"]).agg(
        deliveries=("match_id", "size"),
        wickets=("credited_bowler_wicket", "sum"),
    ).reset_index()

    result = grouped.pivot_table(
        index="bowling_phase", columns="pressure",
        values=["deliveries", "wickets"], aggfunc="sum",
    ).reset_index()
    result.columns = [
        "phase",
        "ordinary_deliveries", "pressure_deliveries",
        "ordinary_wickets", "pressure_wickets",
    ]
    result["ordinary_rate_per_100"] = 100 * result["ordinary_wickets"] / result["ordinary_deliveries"].replace(0, np.nan)
    result["pressure_rate_per_100"] = 100 * result["pressure_wickets"] / result["pressure_deliveries"].replace(0, np.nan)
    result["delta_per_100"] = result["pressure_rate_per_100"] - result["ordinary_rate_per_100"]
    return result