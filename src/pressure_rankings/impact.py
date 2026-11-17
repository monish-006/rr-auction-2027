import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import (
    MIN_NORMAL_OPPORTUNITIES,
    MIN_PRESSURE_MATCHES,
    MIN_PRESSURE_OPPORTUNITIES,
)
from .scoring import _summarize_events


_BATTER_SUMS = [
    "opportunity",
    "runs_above_expected",
    "actual_runs",
    "expected_runs",
    "actual_fours",
    "expected_fours",
    "actual_sixes",
    "expected_sixes",
    "actual_boundaries",
    "expected_boundaries",
    "actual_wickets",
    "expected_wickets",
    "wicket_preservation",
    "contribution",
]
_BOWLER_SUMS = [
    "opportunity",
    "runs_saved",
    "actual_dots",
    "expected_dots",
    "actual_boundaries",
    "expected_boundaries",
    "actual_credited_wickets",
    "expected_credited_wickets",
    "wicket_impact",
    "contribution",
]
_CONTEXT = ["competition_format", "role", "phase"]


def _batting_role(order):
    if pd.isna(order):
        return "Unknown"
    if order <= 3:
        return "Top order"
    if order <= 6:
        return "Middle order"
    return "Lower order"


def _empty_batter_event(rows, player_ids, player_names, roles, event_type):
    frame = rows[
        [
            "delivery_id",
            "match_id",
            "competition_format",
            "season",
            "phase",
            "pressure_index",
        ]
    ].copy()
    frame["player_id"] = player_ids
    frame["player_name"] = player_names
    frame["role"] = roles
    frame["event_type"] = event_type
    frame["opportunity"] = 0
    for column in _BATTER_SUMS:
        if column != "opportunity":
            frame[column] = 0.0
    return frame


def build_player_events(deliveries, player_roster):
    """Build batter and bowler events with expected and outcome-based impacts."""
    if "pressure_state_valid" in deliveries:
        deliveries = deliveries.loc[deliveries["pressure_state_valid"]].copy()
    else:
        deliveries = deliveries.loc[deliveries["pressure_index"].notna()].copy()
    common = [
        "delivery_id",
        "match_id",
        "competition_format",
        "season",
        "phase",
        "pressure_index",
    ]
    batting_order = (
        deliveries[["match_id", "batting_team_id", "striker_id", "batting_order"]]
        .dropna(subset=["striker_id", "batting_order"])
        .drop_duplicates(["match_id", "batting_team_id", "striker_id"])
    )
    order_lookup = {
        (int(row.match_id), int(row.batting_team_id), int(row.striker_id)): row.batting_order
        for row in batting_order.itertuples(index=False)
    }
    roster = player_roster.dropna(subset=["batting_order"]).drop_duplicates(
        ["match_id", "team_id", "player_id"]
    )
    for row in roster.itertuples(index=False):
        order_lookup.setdefault(
            (int(row.match_id), int(row.team_id), int(row.player_id)),
            row.batting_order,
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

    batter_rows = deliveries.loc[deliveries["is_batter_opportunity"]].copy()
    batters = batter_rows[
        common
        + [
            "striker_id",
            "striker_name",
            "batting_order",
            "batter_runs",
            "expected_batter_runs",
            "is_four",
            "is_six",
            "is_boundary",
            "expected_batter_four_probability",
            "expected_batter_six_probability",
            "expected_batter_boundary_probability",
            "expected_any_wicket_probability",
            "wicket_run_value",
        ]
    ].copy()
    batters = batters.rename(
        columns={"striker_id": "player_id", "striker_name": "player_name"}
    )
    batters["role"] = batters["batting_order"].map(_batting_role)
    batters["event_type"] = "batted_delivery"
    batters["opportunity"] = 1
    batters["runs_above_expected"] = (
        batters["batter_runs"] - batters["expected_batter_runs"]
    )
    batters["actual_runs"] = batters["batter_runs"]
    batters["expected_runs"] = batters["expected_batter_runs"]
    batters["actual_fours"] = batters["is_four"].astype(int)
    batters["expected_fours"] = batters["expected_batter_four_probability"]
    batters["actual_sixes"] = batters["is_six"].astype(int)
    batters["expected_sixes"] = batters["expected_batter_six_probability"]
    batters["actual_boundaries"] = batters["is_boundary"].astype(int)
    batters["expected_boundaries"] = batters[
        "expected_batter_boundary_probability"
    ]
    batters["actual_wickets"] = 0
    batters["expected_wickets"] = batters["expected_any_wicket_probability"]
    batters["wicket_preservation"] = (
        batters["expected_any_wicket_probability"] * batters["wicket_run_value"]
    )
    batters["contribution"] = batters["runs_above_expected"]
    batter_events = batters[
        common
        + [
            "player_id",
            "player_name",
            "role",
            "event_type",
            *_BATTER_SUMS,
        ]
    ]

    dismissal_rows = deliveries.loc[
        deliveries["is_wicket"] & deliveries["dismissed_player_id"].notna()
    ].copy()
    if not dismissal_rows.empty:
        dismissal_ids = dismissal_rows["dismissed_player_id"].astype(int)
        dismissal_roles = [
            _batting_role(
                order_lookup.get((int(match), int(team), int(player)), np.nan)
            )
            for match, team, player in dismissal_rows[
                ["match_id", "batting_team_id", "dismissed_player_id"]
            ].itertuples(index=False, name=None)
        ]
        dismissal_events = _empty_batter_event(
            dismissal_rows,
            dismissal_ids,
            dismissal_ids.map(name_lookup),
            dismissal_roles,
            "dismissal",
        )
        dismissal_events["actual_wickets"] = 1
        dismissal_events["wicket_preservation"] = -dismissal_rows[
            "wicket_run_value"
        ].to_numpy()
        dismissal_events["contribution"] = dismissal_events["wicket_preservation"]
        batter_events = pd.concat(
            [batter_events, dismissal_events], ignore_index=True, sort=False
        )

    bowler_rows = deliveries.loc[deliveries["is_bowler_opportunity"]].copy()
    bowlers = bowler_rows[
        common
        + [
            "bowler_id",
            "bowler_name",
            "expected_bowler_runs",
            "bowler_runs_conceded",
            "is_dot_ball",
            "expected_dot_probability",
            "is_boundary",
            "expected_bowler_boundary_probability",
            "is_credited_wicket",
            "expected_credited_wicket_probability",
            "wicket_run_value",
        ]
    ].copy()
    bowlers = bowlers.rename(
        columns={"bowler_id": "player_id", "bowler_name": "player_name"}
    )
    bowlers = bowlers.loc[bowlers["player_id"].notna()].copy()
    bowlers["role"] = bowlers["phase"].astype("string").fillna("Unknown")
    bowlers["event_type"] = "bowled_delivery"
    bowlers["opportunity"] = 1
    bowlers["runs_saved"] = (
        bowlers["expected_bowler_runs"] - bowlers["bowler_runs_conceded"]
    )
    bowlers["actual_dots"] = bowlers["is_dot_ball"].astype(int)
    bowlers["expected_dots"] = bowlers["expected_dot_probability"]
    bowlers["actual_boundaries"] = bowlers["is_boundary"].astype(int)
    bowlers["expected_boundaries"] = bowlers[
        "expected_bowler_boundary_probability"
    ]
    bowlers["actual_credited_wickets"] = bowlers["is_credited_wicket"].astype(int)
    bowlers["expected_credited_wickets"] = bowlers[
        "expected_credited_wicket_probability"
    ]
    bowlers["wicket_impact"] = (
        bowlers["actual_credited_wickets"] - bowlers["expected_credited_wickets"]
    ) * bowlers["wicket_run_value"]
    bowlers["contribution"] = (
        bowlers["runs_saved"]
        + bowlers["actual_credited_wickets"] * bowlers["wicket_run_value"]
    )
    bowler_events = bowlers[
        common
        + [
            "player_id",
            "player_name",
            "role",
            "event_type",
            *_BOWLER_SUMS,
        ]
    ]
    return batter_events, bowler_events


def _rate(numerator, denominator):
    return 100 * numerator / denominator.where(denominator.gt(0))


def _aggregate_metrics(events, group_keys, player_type):
    sums = _BATTER_SUMS if player_type == "batter" else _BOWLER_SUMS
    grouped = (
        events.assign(
            condition=np.where(events["is_pressure"], "pressure", "normal")
        )
        .groupby(
            list(group_keys) + ["condition"],
            as_index=False,
            dropna=False,
            observed=True,
        )[sums]
        .sum()
    )
    opportunities = grouped["opportunity"]
    if player_type == "batter":
        grouped["opportunities"] = opportunities
        grouped["runs_above_expected_per100"] = _rate(
            grouped["runs_above_expected"], opportunities
        )
        grouped["actual_strike_rate"] = _rate(grouped["actual_runs"], opportunities)
        grouped["expected_strike_rate"] = _rate(
            grouped["expected_runs"], opportunities
        )
        grouped["strike_rate_impact"] = (
            grouped["actual_strike_rate"] - grouped["expected_strike_rate"]
        )
        grouped["actual_four_rate"] = _rate(grouped["actual_fours"], opportunities)
        grouped["expected_four_rate"] = _rate(
            grouped["expected_fours"], opportunities
        )
        grouped["four_rate_delta"] = (
            grouped["actual_four_rate"] - grouped["expected_four_rate"]
        )
        grouped["actual_six_rate"] = _rate(grouped["actual_sixes"], opportunities)
        grouped["expected_six_rate"] = _rate(
            grouped["expected_sixes"], opportunities
        )
        grouped["six_rate_delta"] = (
            grouped["actual_six_rate"] - grouped["expected_six_rate"]
        )
        grouped["actual_boundary_percentage"] = _rate(
            grouped["actual_boundaries"], opportunities
        )
        grouped["expected_boundary_percentage"] = _rate(
            grouped["expected_boundaries"], opportunities
        )
        grouped["boundary_percentage_delta_per100"] = (
            grouped["actual_boundary_percentage"]
            - grouped["expected_boundary_percentage"]
        )
        grouped["actual_dismissal_probability_per100"] = _rate(
            grouped["actual_wickets"], opportunities
        )
        grouped["expected_dismissal_probability_per100"] = _rate(
            grouped["expected_wickets"], opportunities
        )
        grouped["wicket_preservation_per100"] = _rate(
            grouped["wicket_preservation"], opportunities
        )
        metrics = [
            "opportunities",
            "runs_above_expected_per100",
            "actual_strike_rate",
            "expected_strike_rate",
            "strike_rate_impact",
            "actual_four_rate",
            "expected_four_rate",
            "four_rate_delta",
            "actual_six_rate",
            "expected_six_rate",
            "six_rate_delta",
            "actual_boundary_percentage",
            "expected_boundary_percentage",
            "boundary_percentage_delta_per100",
            "actual_dismissal_probability_per100",
            "expected_dismissal_probability_per100",
            "wicket_preservation_per100",
            "contribution",
        ]
    else:
        grouped["opportunities"] = opportunities
        grouped["runs_saved_per100"] = _rate(grouped["runs_saved"], opportunities)
        grouped["actual_dot_percentage"] = _rate(grouped["actual_dots"], opportunities)
        grouped["expected_dot_percentage"] = _rate(
            grouped["expected_dots"], opportunities
        )
        grouped["dot_percentage_delta_per100"] = (
            grouped["actual_dot_percentage"] - grouped["expected_dot_percentage"]
        )
        grouped["actual_boundary_percentage"] = _rate(
            grouped["actual_boundaries"], opportunities
        )
        grouped["expected_boundary_percentage"] = _rate(
            grouped["expected_boundaries"], opportunities
        )
        grouped["boundary_prevention_per100"] = (
            grouped["expected_boundary_percentage"]
            - grouped["actual_boundary_percentage"]
        )
        grouped["actual_credited_wicket_rate_per100"] = _rate(
            grouped["actual_credited_wickets"], opportunities
        )
        grouped["expected_credited_wicket_rate_per100"] = _rate(
            grouped["expected_credited_wickets"], opportunities
        )
        grouped["wicket_impact_per100"] = _rate(
            grouped["wicket_impact"], opportunities
        )
        metrics = [
            "opportunities",
            "runs_saved_per100",
            "actual_dot_percentage",
            "expected_dot_percentage",
            "dot_percentage_delta_per100",
            "actual_boundary_percentage",
            "expected_boundary_percentage",
            "boundary_prevention_per100",
            "actual_credited_wicket_rate_per100",
            "expected_credited_wicket_rate_per100",
            "wicket_impact_per100",
            "contribution",
        ]

    wide = grouped.pivot(
        index=list(group_keys), columns="condition", values=metrics
    )
    expected_columns = pd.MultiIndex.from_product(
        [metrics, ["pressure", "normal"]], names=[None, "condition"]
    )
    wide = wide.reindex(columns=expected_columns)
    wide.columns = [f"{condition}_{metric}" for metric, condition in wide.columns]
    return wide.reset_index()


def _merge_impact_metrics(summary, impact_metrics, keys):
    duplicate_columns = sorted(
        (set(summary.columns) & set(impact_metrics.columns)) - set(keys)
    )
    return summary.merge(
        impact_metrics.drop(columns=duplicate_columns),
        on=keys,
        how="left",
        validate="one_to_one",
    )


def _learn_weights(matrix, component_names, mode):
    if mode == "equal":
        return (
            {name: 1 / len(component_names) for name in component_names},
            np.nan,
        )
    valid = np.isfinite(matrix).all(axis=1)
    usable = matrix[valid]
    if len(usable) < 2 or len(component_names) == 1:
        return (
            {name: 1 / len(component_names) for name in component_names},
            np.nan,
        )
    pca = PCA(n_components=1, svd_solver="full")
    pca.fit(usable)
    loadings = np.abs(pca.components_[0])
    if not np.isfinite(loadings).all() or loadings.sum() == 0:
        weights = np.full(len(component_names), 1 / len(component_names))
    else:
        weights = loadings / loadings.sum()
    return dict(zip(component_names, weights)), float(pca.explained_variance_ratio_[0])


def _zscore_by_context(frame, column, output_column):
    grouped = frame.groupby(_CONTEXT, dropna=False, observed=True)[column]
    means = grouped.transform("mean")
    standard_deviations = grouped.transform("std").fillna(0)
    zscore = np.where(
        standard_deviations.gt(0),
        (frame[column] - means) / standard_deviations.replace(0, np.nan),
        0,
    )
    frame[output_column] = np.where(frame[column].notna(), zscore, np.nan)
    return frame


def _score_components(metrics, player_type, weight_mode):
    frame = metrics.copy()
    conditions = ("pressure", "normal")
    if player_type == "batter":
        scoring_metrics = [
            "runs_above_expected_per100",
            "boundary_percentage_delta_per100",
        ]
        wicket_metric = "wicket_preservation_per100"
        impact_name = "scoring_impact"
    else:
        scoring_metrics = [
            "runs_saved_per100",
            "dot_percentage_delta_per100",
            "boundary_prevention_per100",
        ]
        wicket_metric = "wicket_impact_per100"
        impact_name = "bowling_control"

    for condition in conditions:
        for metric in scoring_metrics + [wicket_metric]:
            _zscore_by_context(
                frame,
                f"{condition}_{metric}",
                f"{condition}_z_{metric}",
            )

    inner_names = [f"z_{metric}" for metric in scoring_metrics]
    inner_values = np.concatenate(
        [
            frame[[f"{condition}_{name}" for name in inner_names]].to_numpy()
            for condition in conditions
        ],
        axis=0,
    )
    inner_weights, inner_variance = _learn_weights(
        inner_values, inner_names, weight_mode
    )
    for condition in conditions:
        frame[f"{condition}_{impact_name}"] = sum(
            frame[f"{condition}_z_{metric}"] * inner_weights[f"z_{metric}"]
            for metric in scoring_metrics
        )
        _zscore_by_context(
            frame,
            f"{condition}_{impact_name}",
            f"{condition}_z_{impact_name}",
        )
        frame[f"{condition}_z_wicket_dimension"] = frame[
            f"{condition}_z_{wicket_metric}"
        ]

    outer_names = [
        impact_name,
        "wicket_preservation" if player_type == "batter" else "wicket_taking",
    ]
    outer_values = np.concatenate(
        [
            frame[
                [
                    f"{condition}_z_{impact_name}",
                    f"{condition}_z_wicket_dimension",
                ]
            ].to_numpy()
            for condition in conditions
        ],
        axis=0,
    )
    outer_weights, outer_variance = _learn_weights(
        outer_values, outer_names, weight_mode
    )
    for condition in conditions:
        dimensions = [
            frame[f"{condition}_z_{impact_name}"],
            frame[f"{condition}_z_wicket_dimension"],
        ]
        frame[f"{condition}_composite"] = (
            dimensions[0] * outer_weights[outer_names[0]]
            + dimensions[1] * outer_weights[outer_names[1]]
        )
        frame.loc[
            frame[f"{condition}_opportunities"].fillna(0).eq(0),
            f"{condition}_composite",
        ] = np.nan
    frame["clutch_uplift"] = (
        frame["pressure_composite"] - frame["normal_composite"]
    )

    weight_rows = []
    for component, weight in inner_weights.items():
        weight_rows.append(
            {
                "level": impact_name,
                "component": component.removeprefix("z_"),
                "weight": weight,
                "variance_explained": inner_variance,
                "weight_method": weight_mode,
            }
        )
    for component, weight in outer_weights.items():
        weight_rows.append(
            {
                "level": "overall_impact",
                "component": component,
                "weight": weight,
                "variance_explained": outer_variance,
                "weight_method": weight_mode,
            }
        )
    return frame, pd.DataFrame(weight_rows)


def _stability(values):
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return 1.0
    return float(1 / (1 + values.std(ddof=0) / (abs(values.mean()) + 1)))


def _phase_leaderboards(events, context_metrics, player_type):
    phase_keys = ["player_id", "player_name", "phase"]
    context_metrics = context_metrics.copy()
    context_metrics["phase_weight"] = np.sqrt(
        context_metrics["pressure_opportunities"].clip(upper=400)
    )
    context_metrics["normal_phase_weight"] = np.sqrt(
        context_metrics["normal_opportunities"].clip(upper=400)
    )
    phase_scores = (
        context_metrics.assign(
            weighted_score=context_metrics["adjusted_context_score"]
            * context_metrics["phase_weight"],
            weighted_pressure=context_metrics["pressure_composite"]
            * context_metrics["phase_weight"],
            weighted_normal=context_metrics["normal_composite"]
            * context_metrics["normal_phase_weight"],
        )
        .groupby(phase_keys, as_index=False, dropna=False, observed=True)
        .agg(
            weighted_score=("weighted_score", "sum"),
            score_weight=("phase_weight", "sum"),
            weighted_pressure=("weighted_pressure", "sum"),
            normal_score_weight=("normal_phase_weight", "sum"),
            weighted_normal=("weighted_normal", "sum"),
        )
    )
    phase_scores["ranking_score"] = (
        phase_scores["weighted_score"]
        / phase_scores["score_weight"].replace(0, np.nan)
    )
    phase_scores["pressure_composite"] = (
        phase_scores["weighted_pressure"]
        / phase_scores["score_weight"].replace(0, np.nan)
    )
    phase_scores["normal_composite"] = (
        phase_scores["weighted_normal"]
        / phase_scores["normal_score_weight"].replace(0, np.nan)
    )
    phase_scores["clutch_uplift"] = (
        phase_scores["pressure_composite"] - phase_scores["normal_composite"]
    )
    summary, _ = _summarize_events(events, phase_keys)
    phase_scores = phase_scores.merge(
        summary[
            phase_keys
            + [
                "pressure_opportunities",
                "normal_opportunities",
                "pressure_matches",
                "positive_pressure_match_share",
                "season_stability",
                "sample_shrinkage",
                "consistency_factor",
                "stability_factor",
            ]
        ],
        on=phase_keys,
        how="left",
        validate="one_to_one",
    )
    phase_impact = _aggregate_metrics(events, phase_keys, player_type)
    phase_scores = _merge_impact_metrics(phase_scores, phase_impact, phase_keys)
    phase_scores["eligible"] = (
        phase_scores["pressure_opportunities"].ge(MIN_PRESSURE_OPPORTUNITIES)
        & phase_scores["pressure_matches"].ge(MIN_PRESSURE_MATCHES)
        & phase_scores["normal_opportunities"].ge(MIN_NORMAL_OPPORTUNITIES)
        & phase_scores["ranking_score"].notna()
    )
    phase_scores = phase_scores.loc[phase_scores["eligible"]].copy()
    phase_scores["player_type"] = player_type
    phase_scores["rank"] = (
        phase_scores.groupby("phase", observed=True)["ranking_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    return phase_scores.sort_values(["phase", "rank"], kind="stable")


def rank_players(
    events,
    cutoff=0.55,
    min_pressure_opportunities=MIN_PRESSURE_OPPORTUNITIES,
    min_pressure_matches=MIN_PRESSURE_MATCHES,
    min_normal_opportunities=MIN_NORMAL_OPPORTUNITIES,
    weight_mode="pca",
):
    """Build multimetric pressure rankings and phase-specific leaderboards."""
    data = events.copy()
    data["is_pressure"] = data["pressure_index"].ge(cutoff)
    player_type = "batter" if "runs_above_expected" in data.columns else "bowler"
    context_keys = [
        "player_id",
        "player_name",
        "competition_format",
        "role",
        "phase",
    ]
    context_metrics = _aggregate_metrics(data, context_keys, player_type)
    context_metrics, component_weights = _score_components(
        context_metrics, player_type, weight_mode
    )
    component_weights["player_type"] = player_type

    reliability, match_metrics = _summarize_events(data, context_keys)
    reliability_columns = [
        "pressure_matches",
        "positive_pressure_matches",
        "positive_pressure_match_share",
        "normal_matches",
        "season_stability",
        "positive_season_share",
        "sample_shrinkage",
        "consistency_factor",
        "stability_factor",
    ]
    context_metrics = context_metrics.merge(
        reliability[context_keys + reliability_columns],
        on=context_keys,
        how="left",
        validate="one_to_one",
    )
    match_keys = context_keys + ["match_id", "season"]
    match_player_metrics = _aggregate_metrics(data, match_keys, player_type)
    match_metrics = _merge_impact_metrics(
        match_metrics, match_player_metrics, match_keys
    )
    context_metrics["adjusted_context_score"] = (
        context_metrics["pressure_composite"]
        * context_metrics["sample_shrinkage"]
        * context_metrics["consistency_factor"]
        * context_metrics["stability_factor"]
    )
    context_metrics["ranking_score"] = context_metrics["adjusted_context_score"]
    context_metrics["metric_scope"] = "player_context"

    overall_keys = ["player_id", "player_name"]
    overall_metrics, _ = _summarize_events(data, overall_keys)
    overall_impact = _aggregate_metrics(data, overall_keys, player_type)
    overall_metrics = _merge_impact_metrics(
        overall_metrics, overall_impact, overall_keys
    )
    context_metrics["score_weight"] = np.sqrt(
        context_metrics["pressure_opportunities"].clip(upper=400)
    )
    context_metrics["normal_score_weight"] = np.sqrt(
        context_metrics["normal_opportunities"].clip(upper=400)
    )
    player_scores = (
        context_metrics.assign(
            weighted_score=context_metrics["adjusted_context_score"]
            * context_metrics["score_weight"],
            weighted_pressure=context_metrics["pressure_composite"]
            * context_metrics["score_weight"],
            weighted_normal=context_metrics["normal_composite"]
            * context_metrics["normal_score_weight"],
        )
        .groupby(overall_keys, as_index=False)
        .agg(
            weighted_score=("weighted_score", "sum"),
            score_weight=("score_weight", "sum"),
            weighted_pressure=("weighted_pressure", "sum"),
            normal_score_weight=("normal_score_weight", "sum"),
            weighted_normal=("weighted_normal", "sum"),
            formats=("competition_format", lambda values: ", ".join(sorted(set(values)))),
            roles=("role", lambda values: ", ".join(sorted(set(values)))),
            phases=("phase", lambda values: ", ".join(sorted(set(values)))),
        )
    )
    player_scores["ranking_score"] = (
        player_scores["weighted_score"]
        / player_scores["score_weight"].replace(0, np.nan)
    )
    player_scores["pressure_composite"] = (
        player_scores["weighted_pressure"]
        / player_scores["score_weight"].replace(0, np.nan)
    )
    player_scores["normal_composite"] = (
        player_scores["weighted_normal"]
        / player_scores["normal_score_weight"].replace(0, np.nan)
    )
    player_scores["clutch_uplift"] = (
        player_scores["pressure_composite"] - player_scores["normal_composite"]
    )
    overall_metrics = overall_metrics.merge(
        player_scores[
            overall_keys
            + [
                "ranking_score",
                "pressure_composite",
                "normal_composite",
                "clutch_uplift",
                "formats",
                "roles",
                "phases",
            ]
        ],
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
    overall_metrics["competition_format"] = "All"
    overall_metrics["role"] = "All"
    overall_metrics["phase"] = "All"
    overall_metrics["context_mean"] = np.nan
    overall_metrics["context_std"] = np.nan
    overall_metrics["context_z"] = np.nan

    eligible_overall = overall_metrics.loc[overall_metrics["eligible"]].copy()
    eligible_overall = eligible_overall.sort_values(
        ["ranking_score", "pressure_opportunities"],
        ascending=[False, False],
        kind="stable",
    )
    eligible_overall["rank"] = np.arange(1, len(eligible_overall) + 1)
    context_metrics["eligible"] = context_metrics["player_id"].isin(
        eligible_overall["player_id"]
    )
    metrics = pd.concat(
        [overall_metrics, context_metrics], ignore_index=True, sort=False
    )
    metrics = metrics.merge(
        eligible_overall[overall_keys + ["rank"]],
        on=overall_keys,
        how="left",
        validate="many_to_one",
        suffixes=("", "_overall"),
    )

    format_metrics, format_match_metrics = _summarize_events(
        data, ["player_id", "player_name", "competition_format"]
    )
    format_impact = _aggregate_metrics(
        data, ["player_id", "player_name", "competition_format"], player_type
    )
    format_metrics = _merge_impact_metrics(
        format_metrics,
        format_impact,
        ["player_id", "player_name", "competition_format"],
    )
    phase_leaderboards = _phase_leaderboards(
        data, context_metrics, player_type
    )
    return (
        metrics,
        eligible_overall,
        match_metrics,
        format_metrics,
        format_match_metrics,
        phase_leaderboards,
        component_weights,
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
