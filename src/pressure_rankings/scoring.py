import numpy as np
import pandas as pd

from .config import (
    BASELINE_WEIGHTS,
    CREDITED_DISMISSALS,
    MIN_NORMAL_OPPORTUNITIES,
    MIN_PRESSURE_MATCHES,
    MIN_PRESSURE_OPPORTUNITIES,
    PRESSURE_CUTOFF,
    SHRINKAGE_OPPORTUNITIES,
)


def calculate_pressure(deliveries, weights=BASELINE_WEIGHTS):
    """Return bounded, pre-delivery pressure components and composite index."""
    data = deliveries.copy()
    state_valid = (
        data["state_valid"].fillna(False)
        if "state_valid" in data
        else pd.Series(True, index=data.index)
    )
    if "expected_future_team_runs" in data:
        state_valid &= data["expected_future_team_runs"].notna()
    first_innings_valid = data["innings"].ne(1) | data["training_par_total"].notna()
    state_valid &= first_innings_valid
    stage = (
        data["legal_balls_before"] / data["scheduled_balls"].replace(0, np.nan)
    ).fillna(0).clip(0, 1)
    resource = (data["wickets_lost_before"] / 9).clip(0, 1)

    projected_first_innings_score = (
        data["current_score_before"] + data["expected_future_team_runs"]
    )
    par_gap = data["training_par_total"] - projected_first_innings_score
    first_run_rate_pressure = (
        par_gap / (0.20 * data["training_par_total"].clip(lower=1))
    ).clip(0, 1)

    expected_chase_runs = data["expected_future_team_runs"].clip(lower=1)
    second_run_rate_pressure = (
        (data["runs_required_before"] / expected_chase_runs - 1) / 0.50
    ).clip(0, 1)
    leverage_denominator = np.maximum(
        data["runs_required_before"], data["expected_future_team_runs"]
    ).clip(lower=1)
    leverage = (
        1
        - (
            data["runs_required_before"]
            - data["expected_future_team_runs"]
        ).abs()
        / leverage_denominator
    ).clip(0, 1)

    first_weights = weights["first"]
    second_weights = weights["second"]
    data["pressure_run_rate"] = np.where(
        data["innings"].eq(1),
        first_run_rate_pressure,
        second_run_rate_pressure,
    )
    data["pressure_resource"] = resource
    data["pressure_stage"] = stage
    data["pressure_leverage"] = np.where(data["innings"].eq(2), leverage, 0)
    first_pressure = (
        first_weights["run_rate"] * first_run_rate_pressure
        + first_weights["resource"] * resource
        + first_weights["stage"] * stage
    )
    second_pressure = (
        second_weights["run_rate"] * second_run_rate_pressure
        + second_weights["resource"] * resource
        + second_weights["leverage"] * leverage
        + second_weights["stage"] * stage
    )
    data["pressure_index"] = np.where(
        data["innings"].eq(1), first_pressure, second_pressure
    ).clip(0, 1)
    data["pressure_index"] = data["pressure_index"].where(state_valid)
    data["pressure_state_valid"] = state_valid
    data["is_pressure"] = (
        data["pressure_state_valid"] & data["pressure_index"].ge(PRESSURE_CUTOFF)
    )
    return data


def _batting_role(order):
    if pd.isna(order):
        return "Unknown"
    if order <= 3:
        return "Top order"
    if order <= 6:
        return "Middle order"
    return "Lower order"


def build_player_events(deliveries, player_roster):
    """Create player-delivery events with correct batter and bowler attribution."""
    common = [
        "delivery_id",
        "match_id",
        "competition_format",
        "season",
        "phase",
        "pressure_index",
        "is_pressure",
    ]

    batter_rows = deliveries.loc[deliveries["is_batter_opportunity"]]
    batter_events = batter_rows[
        common + ["striker_id", "striker_name", "batting_order"]
    ].copy()
    batter_events = batter_events.rename(
        columns={"striker_id": "player_id", "striker_name": "player_name"}
    )
    batter_events["role"] = batter_events["batting_order"].map(_batting_role)
    batter_events["contribution"] = (
        batter_rows["batter_runs"].to_numpy()
        - batter_rows["expected_batter_runs"].to_numpy()
    )
    batter_events["opportunity"] = 1
    batter_events["dismissal_cost"] = 0.0
    batter_events["event_type"] = "batted_delivery"

    roster = player_roster.dropna(subset=["batting_order"]).drop_duplicates(
        ["match_id", "team_id", "player_id"]
    )
    order_lookup = {
        (int(row.match_id), int(row.team_id), int(row.player_id)): row.batting_order
        for row in roster.itertuples(index=False)
    }
    source_order = (
        deliveries[
            ["match_id", "batting_team_id", "striker_id", "batting_order"]
        ]
        .dropna(subset=["batting_order"])
        .drop_duplicates(["match_id", "batting_team_id", "striker_id"])
    )
    for row in source_order.itertuples(index=False):
        order_lookup.setdefault(
            (int(row.match_id), int(row.batting_team_id), int(row.striker_id)),
            row.batting_order,
        )

    dismissal_events = deliveries.loc[
        deliveries["is_wicket"] & deliveries["dismissed_player_id"].notna(),
        common
        + [
            "dismissed_player_id",
            "striker_id",
            "striker_name",
            "non_striker_id",
            "non_striker_name",
            "batting_team_id",
            "wicket_run_value",
        ],
    ].copy()
    dismissal_events["player_id"] = dismissal_events[
        "dismissed_player_id"
    ].astype(int)
    dismissal_events["player_name"] = np.select(
        [
            dismissal_events["player_id"].eq(dismissal_events["striker_id"]),
            dismissal_events["player_id"].eq(dismissal_events["non_striker_id"]),
        ],
        [
            dismissal_events["striker_name"],
            dismissal_events["non_striker_name"],
        ],
        default=None,
    )
    name_lookup = pd.concat(
        [
            deliveries[["striker_id", "striker_name"]].rename(
                columns={"striker_id": "player_id", "striker_name": "player_name"}
            ),
            deliveries[["non_striker_id", "non_striker_name"]].rename(
                columns={
                    "non_striker_id": "player_id",
                    "non_striker_name": "player_name",
                }
            ),
        ],
        ignore_index=True,
    ).dropna(subset=["player_id", "player_name"])
    name_lookup = (
        name_lookup.drop_duplicates("player_id")
        .set_index("player_id")["player_name"]
        .to_dict()
    )
    missing_names = dismissal_events["player_name"].isna()
    dismissal_events.loc[missing_names, "player_name"] = dismissal_events.loc[
        missing_names, "player_id"
    ].map(name_lookup)
    dismissal_events["role"] = [
        _batting_role(
            order_lookup.get((int(match_id), int(team_id), int(player_id)), np.nan)
        )
        for match_id, team_id, player_id in dismissal_events[
            ["match_id", "batting_team_id", "player_id"]
        ].itertuples(index=False, name=None)
    ]
    dismissal_events["contribution"] = -dismissal_events["wicket_run_value"]
    dismissal_events["opportunity"] = 0
    dismissal_events["dismissal_cost"] = dismissal_events["wicket_run_value"]
    dismissal_events["event_type"] = "dismissal"

    batters = pd.concat(
        [
            batter_events[
                common
                + [
                    "player_id",
                    "player_name",
                    "role",
                    "contribution",
                    "opportunity",
                    "dismissal_cost",
                    "event_type",
                ]
            ],
            dismissal_events[
                common
                + [
                    "player_id",
                    "player_name",
                    "role",
                    "contribution",
                    "opportunity",
                    "dismissal_cost",
                    "event_type",
                ]
            ],
        ],
        ignore_index=True,
    )

    credited = deliveries["dismissal_type"].astype("string").str.strip().str.lower().isin(
        CREDITED_DISMISSALS
    )
    bowler_events = deliveries[common + ["bowler_id", "bowler_name"]].copy()
    bowler_events = bowler_events.rename(
        columns={"bowler_id": "player_id", "bowler_name": "player_name"}
    )
    bowler_events["role"] = bowler_events["phase"].astype("string").fillna("Unknown")
    bowler_events["credited_wicket"] = (
        deliveries["is_wicket"].to_numpy() & credited.to_numpy()
    ).astype(int)
    bowler_events["wicket_credit_value"] = (
        bowler_events["credited_wicket"].to_numpy()
        * deliveries["wicket_run_value"].to_numpy()
    )
    bowler_events["contribution"] = (
        deliveries["expected_bowler_runs"].to_numpy()
        - deliveries["bowler_runs_conceded"].to_numpy()
        + bowler_events["wicket_credit_value"].to_numpy()
    )
    bowler_events["opportunity"] = 1
    bowler_events["dismissal_cost"] = 0.0
    bowler_events["event_type"] = "bowled_delivery"
    bowler_events = bowler_events.loc[bowler_events["player_id"].notna()].copy()

    return batters, bowler_events


def _stability(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 1.0
    return float(1 / (1 + values.std(ddof=0) / (abs(values.mean()) + 1)))


def _summarize_events(events, group_keys):
    event_data = events.copy()
    event_data["pressure_contribution"] = np.where(
        event_data["is_pressure"], event_data["contribution"], 0.0
    )
    event_data["normal_contribution"] = np.where(
        ~event_data["is_pressure"], event_data["contribution"], 0.0
    )
    event_data["pressure_opportunities"] = (
        event_data["opportunity"] * event_data["is_pressure"].astype(int)
    )
    event_data["normal_opportunities"] = (
        event_data["opportunity"] * (~event_data["is_pressure"]).astype(int)
    )

    match_keys = list(
        dict.fromkeys(group_keys + ["competition_format", "match_id", "season"])
    )
    match_metrics = (
        event_data.groupby(match_keys, as_index=False, dropna=False, observed=True)
        .agg(
            pressure_contribution=("pressure_contribution", "sum"),
            normal_contribution=("normal_contribution", "sum"),
            pressure_opportunities=("pressure_opportunities", "sum"),
            normal_opportunities=("normal_opportunities", "sum"),
        )
    )
    if "wpa_added" in event_data:
        win_probability = (
            event_data.groupby(
                match_keys, as_index=False, dropna=False, observed=True
            )["wpa_added"]
            .sum()
            .rename(columns={"wpa_added": "win_probability_added"})
        )
        match_metrics = match_metrics.merge(
            win_probability,
            on=match_keys,
            how="left",
            validate="one_to_one",
        )
    match_metrics["pressure_rate_per100"] = np.where(
        match_metrics["pressure_opportunities"].gt(0),
        100
        * match_metrics["pressure_contribution"]
        / match_metrics["pressure_opportunities"].replace(0, np.nan),
        np.nan,
    )
    match_metrics["has_pressure"] = match_metrics["pressure_opportunities"].gt(0)
    match_metrics["positive_pressure_match"] = (
        match_metrics["has_pressure"]
        & match_metrics["pressure_contribution"].gt(0)
    )

    metrics = (
        match_metrics.groupby(group_keys, as_index=False, dropna=False, observed=True)
        .agg(
            pressure_contribution=("pressure_contribution", "sum"),
            normal_contribution=("normal_contribution", "sum"),
            pressure_opportunities=("pressure_opportunities", "sum"),
            normal_opportunities=("normal_opportunities", "sum"),
            pressure_matches=("has_pressure", "sum"),
            positive_pressure_matches=("positive_pressure_match", "sum"),
            normal_matches=("normal_opportunities", lambda x: int(x.gt(0).sum())),
        )
    )
    metrics["positive_pressure_match_share"] = np.where(
        metrics["pressure_matches"].gt(0),
        metrics["positive_pressure_matches"] / metrics["pressure_matches"],
        np.nan,
    )
    metrics["pressure_per100"] = (
        100
        * metrics["pressure_contribution"]
        / metrics["pressure_opportunities"].replace(0, np.nan)
    )
    metrics["normal_per100"] = (
        100
        * metrics["normal_contribution"]
        / metrics["normal_opportunities"].replace(0, np.nan)
    )
    metrics["clutch_uplift_per100"] = (
        metrics["pressure_per100"] - metrics["normal_per100"]
    )
    metrics["base_performance"] = (
        metrics["pressure_per100"] + 0.5 * metrics["clutch_uplift_per100"]
    )

    season_keys = list(dict.fromkeys(group_keys + ["competition_format", "season"]))
    season_metrics = (
        match_metrics.groupby(
            season_keys, as_index=False, dropna=False, observed=True
        )
        .agg(
            season_pressure_contribution=("pressure_contribution", "sum"),
            season_pressure_opportunities=("pressure_opportunities", "sum"),
        )
    )
    season_metrics["season_pressure_per100"] = (
        100
        * season_metrics["season_pressure_contribution"]
        / season_metrics["season_pressure_opportunities"].replace(0, np.nan)
    )
    season_summary_keys = group_keys
    season_summary = (
        season_metrics.groupby(
            season_summary_keys, as_index=False, dropna=False, observed=True
        )
        .agg(
            seasons=("season", "nunique"),
            positive_season_share=(
                "season_pressure_per100",
                lambda x: float(x.gt(0).mean()),
            ),
            season_stability=("season_pressure_per100", _stability),
        )
    )
    metrics = metrics.merge(
        season_summary,
        on=season_summary_keys,
        how="left",
        validate="many_to_one",
    )
    metrics["sample_shrinkage"] = (
        metrics["pressure_opportunities"]
        / (metrics["pressure_opportunities"] + SHRINKAGE_OPPORTUNITIES)
        * np.minimum(1, metrics["pressure_matches"] / MIN_PRESSURE_MATCHES)
    )
    metrics["consistency_factor"] = (
        0.5 + 0.5 * metrics["positive_pressure_match_share"].fillna(0)
    )
    metrics["season_stability"] = metrics["season_stability"].fillna(0)
    metrics["positive_season_share"] = metrics["positive_season_share"].fillna(0)
    metrics["stability_factor"] = 0.75 + 0.25 * metrics["season_stability"]
    metrics["metric_scope"] = "player_context"
    return metrics, match_metrics


def rank_players(
    events,
    cutoff=PRESSURE_CUTOFF,
    min_pressure_opportunities=MIN_PRESSURE_OPPORTUNITIES,
    min_pressure_matches=MIN_PRESSURE_MATCHES,
    min_normal_opportunities=MIN_NORMAL_OPPORTUNITIES,
):
    """Build role/format-normalized player rankings and detailed metrics."""
    data = events.copy()
    data["is_pressure"] = data["pressure_index"].ge(cutoff)
    context_keys = [
        "player_id",
        "player_name",
        "competition_format",
        "role",
    ]
    context_metrics, match_metrics = _summarize_events(data, context_keys)
    context_metrics["context_mean"] = context_metrics.groupby(
        ["competition_format", "role"], dropna=False
    )["base_performance"].transform("mean")
    context_metrics["context_std"] = context_metrics.groupby(
        ["competition_format", "role"], dropna=False
    )["base_performance"].transform("std").fillna(0)
    context_metrics["context_z"] = np.where(
        context_metrics["context_std"].gt(0),
        (context_metrics["base_performance"] - context_metrics["context_mean"])
        / context_metrics["context_std"],
        0,
    )
    context_metrics["adjusted_context_score"] = (
        context_metrics["context_z"]
        * context_metrics["sample_shrinkage"]
        * context_metrics["consistency_factor"]
        * context_metrics["stability_factor"]
    )

    overall_keys = ["player_id", "player_name"]
    overall_metrics, _ = _summarize_events(data, overall_keys)
    overall_metrics["metric_scope"] = "overall"
    overall_metrics["competition_format"] = "All"
    overall_metrics["role"] = "All"
    overall_metrics["context_mean"] = np.nan
    overall_metrics["context_std"] = np.nan
    overall_metrics["context_z"] = np.nan

    context_metrics["score_weight"] = np.where(
        context_metrics["adjusted_context_score"].notna(),
        np.sqrt(context_metrics["pressure_opportunities"].clip(upper=400)),
        0,
    )
    context_score = (
        context_metrics.assign(
            weighted_score=context_metrics["adjusted_context_score"]
            * context_metrics["score_weight"]
        )
        .groupby(overall_keys, as_index=False)
        .agg(
            weighted_score=("weighted_score", "sum"),
            score_weight=("score_weight", "sum"),
            formats=("competition_format", lambda x: ", ".join(sorted(set(x)))),
            roles=("role", lambda x: ", ".join(sorted(set(x)))),
        )
    )
    context_score["ranking_score"] = (
        context_score["weighted_score"]
        / context_score["score_weight"].replace(0, np.nan)
    )
    overall_metrics = overall_metrics.merge(
        context_score[overall_keys + ["ranking_score", "formats", "roles"]],
        on=overall_keys,
        how="left",
        validate="one_to_one",
    )
    overall_metrics["eligible"] = (
        overall_metrics["pressure_opportunities"].ge(min_pressure_opportunities)
        & overall_metrics["pressure_matches"].ge(min_pressure_matches)
        & overall_metrics["normal_opportunities"].ge(min_normal_opportunities)
        & overall_metrics["ranking_score"].notna()
    )
    overall_metrics["metric_scope"] = "overall"

    eligible_ids = set(
        overall_metrics.loc[overall_metrics["eligible"], "player_id"]
    )
    context_metrics["eligible"] = context_metrics["player_id"].isin(eligible_ids)
    context_metrics["ranking_score"] = context_metrics["adjusted_context_score"]

    metrics = pd.concat(
        [overall_metrics, context_metrics], ignore_index=True, sort=False
    )
    eligible_overall = overall_metrics.loc[overall_metrics["eligible"]].copy()
    eligible_overall = eligible_overall.sort_values(
        ["ranking_score", "pressure_opportunities"],
        ascending=[False, False],
        kind="stable",
    )
    eligible_overall["rank"] = np.arange(1, len(eligible_overall) + 1)
    metrics = metrics.merge(
        eligible_overall[overall_keys + ["rank"]],
        on=overall_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_overall"),
    )

    format_metrics, match_metrics_by_format = _summarize_events(
        data, ["player_id", "player_name", "competition_format"]
    )
    return (
        metrics,
        eligible_overall,
        match_metrics,
        format_metrics,
        match_metrics_by_format,
    )


def compare_rankings(baseline, scenario, label, player_type, scenario_info):
    baseline_top = set(baseline.head(10)["player_id"])
    scenario_top = set(scenario.head(10)["player_id"])
    union = baseline_top | scenario_top
    common = baseline[["player_id", "rank"]].rename(
        columns={"rank": "rank_baseline"}
    ).merge(
        scenario[["player_id", "rank"]].rename(
            columns={"rank": "rank_scenario"}
        ),
        on="player_id",
    )
    correlation = (
        common["rank_baseline"].corr(common["rank_scenario"], method="spearman")
        if len(common) >= 2
        else np.nan
    )
    return {
        "player_type": player_type,
        "scenario": label,
        **scenario_info,
        "eligible_players": len(scenario),
        "baseline_top10_overlap": len(baseline_top & scenario_top),
        "top10_jaccard": len(baseline_top & scenario_top) / len(union) if union else 1.0,
        "spearman_rank_correlation": correlation,
    }


def make_top_ten(eligible_players):
    return eligible_players.head(10).copy()
