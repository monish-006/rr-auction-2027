import numpy as np
import pandas as pd


BATTER = "BATTER"
BOWLER = "BOWLER"
UNCLASSIFIED = "UNCLASSIFIED"
MIN_ROLE_INNINGS = 5
ROLE_THRESHOLD = 0.70

_DISCIPLINE_METRICS = (
    "eligible",
    "rank",
    "ranking_score",
    "pressure_composite",
    "normal_composite",
    "clutch_uplift",
    "pressure_opportunities",
    "pressure_matches",
    "normal_opportunities",
    "sample_shrinkage",
    "consistency_factor",
    "stability_factor",
)
_OPTIONAL_CONTRIBUTION_METRICS = (
    "pressure_runs_above_expected_per100",
    "pressure_strike_rate_impact",
    "pressure_boundary_percentage_delta_per100",
    "pressure_wicket_preservation_per100",
    "pressure_runs_saved_per100",
    "pressure_dot_percentage_delta_per100",
    "pressure_boundary_prevention_per100",
    "pressure_wicket_impact_per100",
)


def _overall_discipline_metrics(metrics, prefix):
    overall = metrics.loc[metrics["metric_scope"].eq("overall")].copy()
    optional_metrics = [
        column
        for column in _OPTIONAL_CONTRIBUTION_METRICS
        if column in overall.columns
    ]
    columns = [
        "player_id",
        "player_name",
        *_DISCIPLINE_METRICS,
        *optional_metrics,
    ]
    missing = set(columns) - set(overall.columns)
    if missing:
        raise ValueError(
            f"Overall {prefix} metrics are missing required columns: "
            f"{', '.join(sorted(missing))}"
        )
    column_names = {
        "player_name": f"{prefix}_player_name",
        **{column: f"{prefix}_{column}" for column in _DISCIPLINE_METRICS},
    }
    column_names.update(
        {column: f"{prefix}_{column}" for column in optional_metrics}
    )
    return overall[columns].rename(columns=column_names)


def _build_batting_order_history(deliveries):
    required = {"match_id", "innings", "striker_id", "batting_order"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(
            "Deliveries are missing batting-order columns: "
            + ", ".join(sorted(missing))
        )

    columns = ["match_id", "innings", "striker_id", "batting_order"]
    if "striker_name" in deliveries:
        columns.append("striker_name")
    rows = deliveries.loc[deliveries["striker_id"].notna(), columns].copy()
    if "striker_name" not in rows:
        rows["striker_name"] = pd.NA
    rows["batting_order"] = pd.to_numeric(
        rows["batting_order"], errors="coerce"
    )

    innings = (
        rows.groupby(
            ["striker_id", "match_id", "innings"],
            as_index=False,
            sort=False,
            dropna=False,
        )
        .agg(
            history_player_name=("striker_name", "first"),
            batting_order=("batting_order", "first"),
            distinct_order_values=("batting_order", "nunique"),
            order_observations=("batting_order", "count"),
        )
        .rename(columns={"striker_id": "player_id"})
    )
    valid = (
        innings["distinct_order_values"].eq(1)
        & innings["batting_order"].between(1, 11)
    )
    innings["valid_order_innings"] = valid
    innings["top_order_innings"] = (
        valid & innings["batting_order"].between(1, 7)
    )
    innings["lower_order_innings"] = (
        valid & innings["batting_order"].between(8, 11)
    )
    innings["position_12_innings"] = (
        innings["distinct_order_values"].eq(1)
        & innings["batting_order"].eq(12)
    )
    innings["conflicting_order_innings"] = innings[
        "distinct_order_values"
    ].gt(1)
    innings["missing_order_innings"] = innings["order_observations"].eq(0)
    innings["other_invalid_order_innings"] = (
        innings["distinct_order_values"].eq(1)
        & innings["batting_order"].notna()
        & ~innings["batting_order"].between(1, 11)
        & ~innings["batting_order"].eq(12)
    )

    history = (
        innings.groupby("player_id", as_index=False, sort=False)
        .agg(
            history_player_name=("history_player_name", "first"),
            total_innings=("match_id", "size"),
            valid_batting_order_innings=("valid_order_innings", "sum"),
            batting_order_1_7_innings=("top_order_innings", "sum"),
            batting_order_8_11_innings=("lower_order_innings", "sum"),
            position_12_innings=("position_12_innings", "sum"),
            conflicting_order_innings=("conflicting_order_innings", "sum"),
            missing_order_innings=("missing_order_innings", "sum"),
            other_invalid_order_innings=("other_invalid_order_innings", "sum"),
        )
    )
    history["invalid_batting_order_innings"] = (
        history["total_innings"] - history["valid_batting_order_innings"]
    )
    denominator = history["valid_batting_order_innings"].replace(0, np.nan)
    history["batting_order_1_7_pct"] = (
        history["batting_order_1_7_innings"] / denominator
    )
    history["batting_order_8_11_pct"] = (
        history["batting_order_8_11_innings"] / denominator
    )
    history["minimum_history_met"] = history[
        "valid_batting_order_innings"
    ].ge(MIN_ROLE_INNINGS)

    history["assigned_role"] = UNCLASSIFIED
    enough_history = history["minimum_history_met"]
    batter_role = (
        enough_history
        & history["batting_order_1_7_pct"].ge(ROLE_THRESHOLD)
    )
    bowler_role = (
        enough_history
        & history["batting_order_8_11_pct"].ge(ROLE_THRESHOLD)
    )
    history.loc[batter_role, "assigned_role"] = BATTER
    history.loc[bowler_role, "assigned_role"] = BOWLER
    history["role_resolution"] = np.select(
        [
            ~enough_history,
            batter_role,
            bowler_role,
        ],
        [
            "insufficient_batting_order_history",
            "at_least_70_percent_in_positions_1_7",
            "at_least_70_percent_in_positions_8_11",
        ],
        default="mixed_batting_order_distribution",
    )
    return history


def _rank_category(role_classification, assigned_role, discipline):
    role_column = f"{discipline}_ranking_score"
    opportunities_column = f"{discipline}_pressure_opportunities"
    eligible_column = f"{discipline}_eligible"
    rank_column = f"{discipline}_rank"
    ranked = role_classification.loc[
        role_classification["assigned_role"].eq(assigned_role)
        & role_classification[eligible_column]
        & role_classification[role_column].notna()
    ].copy()
    ranked = ranked.sort_values(
        [role_column, opportunities_column],
        ascending=[False, False],
        kind="stable",
    )
    ranked["previous_rank"] = ranked[rank_column]
    ranked["ranking_score"] = ranked[role_column]
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _leaderboard_change(previous, updated, discipline):
    old = previous.head(10).copy()
    new = updated.head(10).copy()
    old_ids = set(old["player_id"])
    new_ids = set(new["player_id"])

    def labels(frame, ids):
        selected = frame.loc[frame["player_id"].isin(ids)]
        return "; ".join(
            f"{row.player_name} ({row.player_id})"
            for row in selected.itertuples()
        )

    moved_out = old_ids - new_ids
    moved_in = new_ids - old_ids
    return {
        "leaderboard": discipline,
        "previous_top_10_count": len(old_ids),
        "updated_top_10_count": len(new_ids),
        "retained_count": len(old_ids & new_ids),
        "moved_out_count": len(moved_out),
        "moved_in_count": len(moved_in),
        "moved_out_players": labels(old, moved_out),
        "moved_in_players": labels(new, moved_in),
    }


def build_role_leaderboards(
    batter_metrics,
    bowler_metrics,
    deliveries,
    previous_batters,
    previous_bowlers,
):
    """Classify by unique batting innings and rank eligible role-specific players."""
    batting = _overall_discipline_metrics(batter_metrics, "batting")
    bowling = _overall_discipline_metrics(bowler_metrics, "bowling")
    players = batting.merge(
        bowling,
        on="player_id",
        how="outer",
        validate="one_to_one",
    )
    history = _build_batting_order_history(deliveries)
    role_classification = history.merge(
        players,
        on="player_id",
        how="outer",
        validate="one_to_one",
    )
    role_classification["player_name"] = (
        role_classification["batting_player_name"]
        .combine_first(role_classification["bowling_player_name"])
        .combine_first(role_classification["history_player_name"])
    )
    for discipline in ("batting", "bowling"):
        eligible_column = f"{discipline}_eligible"
        role_classification[eligible_column] = (
            pd.array(role_classification[eligible_column], dtype="boolean")
            .fillna(False)
            .to_numpy(dtype=bool)
        )
    count_columns = [
        "total_innings",
        "valid_batting_order_innings",
        "batting_order_1_7_innings",
        "batting_order_8_11_innings",
        "position_12_innings",
        "conflicting_order_innings",
        "missing_order_innings",
        "other_invalid_order_innings",
        "invalid_batting_order_innings",
    ]
    for column in count_columns:
        role_classification[column] = (
            role_classification[column].fillna(0).astype(int)
        )
    role_classification["minimum_history_met"] = (
        pd.array(role_classification["minimum_history_met"], dtype="boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    role_classification["assigned_role"] = (
        role_classification["assigned_role"].fillna(UNCLASSIFIED)
    )
    role_classification["role_resolution"] = (
        role_classification["role_resolution"].fillna(
            "no_batting_order_history"
        )
    )

    batters = _rank_category(role_classification, BATTER, "batting")
    bowlers = _rank_category(role_classification, BOWLER, "bowling")
    unclassified = role_classification.loc[
        role_classification["assigned_role"].eq(UNCLASSIFIED)
    ].sort_values(
        ["total_innings", "player_name"],
        ascending=[False, True],
        kind="stable",
    )
    changes = pd.DataFrame(
        [
            _leaderboard_change(previous_batters, batters, "BATTER"),
            _leaderboard_change(previous_bowlers, bowlers, "BOWLER"),
        ]
    )
    return {
        "role_classification": role_classification,
        "unclassified_players": unclassified,
        "batters": batters,
        "bowlers": bowlers,
        "leaderboard_changes": changes,
    }
