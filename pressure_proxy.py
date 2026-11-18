from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("prism-uploads")
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


def rank_batters(data: pd.DataFrame, format_group: str, pressure_mask: pd.Series) -> pd.DataFrame:
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
    eligible.insert(0, "rank", range(1, len(eligible) + 1))
    return eligible.reset_index(drop=True)


def rank_bowlers(data: pd.DataFrame, format_group: str, partner_threshold: int) -> pd.DataFrame:
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