from pathlib import Path

import numpy as np
import pandas as pd

from .config import DELIVERY_KEY
from .config import CREDITED_DISMISSALS

SOURCE_FILES = (
    "ball_by_ball.csv",
    "competition_team_player_mapping.csv",
    "competitions.csv",
    "matches.csv",
    "playing_eleven.csv",
    "teams.csv",
)


def _source_audit(tables):
    rows = []
    for name, table in tables.items():
        missing = table.isna().sum()
        missing_columns = {
            column: int(count) for column, count in missing.items() if count
        }
        rows.append(
            {
                "file": name,
                "rows": len(table),
                "columns": len(table.columns),
                "column_names": ", ".join(table.columns),
                "missing_by_column": repr(missing_columns),
            }
        )
    return pd.DataFrame(rows)


def _quality_row(check, value, detail=""):
    return {"check": check, "value": value, "detail": detail}


def _as_bool(values):
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    return pd.to_numeric(values, errors="coerce").fillna(0).ne(0)


def load_and_prepare(data_dir):
    """Load all supplied tables and create pre-delivery match states."""
    data_dir = Path(data_dir).expanduser()
    missing_files = [name for name in SOURCE_FILES if not (data_dir / name).is_file()]
    if missing_files:
        raise FileNotFoundError(
            f"Missing required CSV file(s) in {data_dir}: {', '.join(missing_files)}"
        )

    tables = {
        name: pd.read_csv(data_dir / name, low_memory=False)
        for name in SOURCE_FILES
    }
    audit = _source_audit(tables)
    ball = tables["ball_by_ball.csv"].copy()
    matches = tables["matches.csv"].copy()
    competitions = tables["competitions.csv"].copy()
    playing_eleven = tables["playing_eleven.csv"].copy()

    matches["match_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    match_metadata = matches.merge(
        competitions[["comp_id", "comp_name", "match_format", "season"]],
        on="comp_id",
        how="left",
        validate="many_to_one",
    )
    season_year = pd.to_numeric(
        match_metadata["season"].astype(str).str.slice(0, 4), errors="coerce"
    )
    is_ipl = (
        match_metadata["match_format"].astype(str).str.upper().eq("T20")
        & match_metadata["comp_name"].astype(str).str.match(r"^IPL\s+\d{4}$")
    )
    is_t20i = match_metadata["match_format"].astype(str).str.upper().eq("T20I")
    selected_matches = match_metadata.loc[
        (is_ipl | is_t20i)
        & season_year.ge(2022)
        & match_metadata["match_date"].dt.year.ge(2022)
    ].copy()
    selected_matches["competition_format"] = np.where(
        selected_matches["comp_name"].astype(str).str.match(r"^IPL\s+\d{4}$"),
        "IPL",
        "T20I",
    )

    selected_ids = set(selected_matches["match_id"])
    ball_rows_without_metadata = int((~ball["match_id"].isin(matches["match_id"])).sum())
    selected_ball = ball.loc[ball["match_id"].isin(selected_ids)].copy()
    duplicate_delivery_keys = int(selected_ball.duplicated(DELIVERY_KEY).sum())

    selected_ball = selected_ball.merge(
        selected_matches[
            [
                "match_id",
                "comp_id",
                "comp_name",
                "match_format",
                "season",
                "match_date",
                "competition_format",
                "first_inn_revised_overs",
                "second_inn_revised_overs",
                "revised_target",
                "win_team_id",
                "result",
                "home_team_id",
                "away_team_id",
            ]
        ],
        on="match_id",
        how="inner",
        validate="many_to_one",
    )

    super_over_rows = int(selected_ball["innings"].gt(2).sum())
    selected_ball = selected_ball.loc[
        selected_ball["innings"].isin([1, 2])
    ].copy()
    selected_ball = selected_ball.sort_values(
        ["match_id", "innings", "innings_ball_num"], kind="stable"
    ).reset_index(drop=True)
    selected_ball["delivery_id"] = np.arange(len(selected_ball), dtype=np.int64)

    extra_type = selected_ball["extras_type"].astype("string").str.strip().str.lower()
    known_extra_types = {"wide", "overstep", "other", "byes", "legbyes", "penalty"}
    observed_extra_types = set(extra_type.dropna().unique())
    unknown_extra_types = sorted(observed_extra_types - known_extra_types)
    if unknown_extra_types:
        raise ValueError(
            "Unrecognized extras_type value(s); review bowler-run accounting before "
            f"continuing: {unknown_extra_types}"
        )

    # In this feed, "other" rows almost always add a delivery beyond six balls in
    # the over, matching the no-ball pattern of "Overstep".
    selected_ball["extras_type_normalized"] = extra_type
    selected_ball["is_legal_ball"] = ~extra_type.isin(
        {"wide", "overstep", "other", "penalty"}
    )
    selected_ball["is_batter_opportunity"] = ~extra_type.isin({"wide", "penalty"})
    selected_ball["is_bowler_opportunity"] = ~extra_type.eq("penalty").fillna(False)
    selected_ball["batter_runs"] = pd.to_numeric(
        selected_ball["runs"], errors="coerce"
    ).fillna(0)
    for column in ("extras1_runs", "extras_2_runs"):
        selected_ball[column] = pd.to_numeric(
            selected_ball[column], errors="coerce"
        ).fillna(0)
    selected_ball["delivery_runs"] = (
        selected_ball["batter_runs"]
        + selected_ball["extras1_runs"]
        + selected_ball["extras_2_runs"]
    )
    selected_ball["bowler_runs_conceded"] = selected_ball["batter_runs"] + np.where(
        extra_type.isin({"wide", "overstep", "other"}),
        selected_ball["extras1_runs"],
        0,
    )
    selected_ball["is_wicket"] = _as_bool(selected_ball["is_wicket"])
    selected_ball["is_four"] = _as_bool(selected_ball["is_four"])
    selected_ball["is_six"] = _as_bool(selected_ball["is_six"])
    dismissal_type = (
        selected_ball["dismissal_type"].astype("string").str.strip().str.lower()
    )
    selected_ball["is_credited_wicket"] = (
        selected_ball["is_wicket"] & dismissal_type.isin(CREDITED_DISMISSALS)
    )
    selected_ball["is_boundary"] = selected_ball["is_four"] | selected_ball["is_six"]
    selected_ball["is_dot_ball"] = selected_ball["delivery_runs"].eq(0)
    selected_ball["batting_team_won"] = np.where(
        selected_ball["win_team_id"].notna(),
        selected_ball["batting_team_id"].eq(selected_ball["win_team_id"]).astype(float),
        np.nan,
    )

    group_keys = ["match_id", "innings"]
    grouped = selected_ball.groupby(group_keys, sort=False)
    selected_ball["current_score_before"] = (
        grouped["delivery_runs"].cumsum() - selected_ball["delivery_runs"]
    )
    selected_ball["wickets_lost_before"] = (
        grouped["is_wicket"].cumsum().astype(int)
        - selected_ball["is_wicket"].astype(int)
    )
    selected_ball["wickets_in_hand_before"] = (
        10 - selected_ball["wickets_lost_before"]
    ).clip(lower=0)
    selected_ball["legal_balls_before"] = (
        grouped["is_legal_ball"].cumsum().astype(int)
        - selected_ball["is_legal_ball"].astype(int)
    )
    innings_totals = selected_ball.groupby(group_keys)["delivery_runs"].sum()
    selected_ball["innings_total"] = pd.MultiIndex.from_frame(
        selected_ball[group_keys]
    ).map(innings_totals)

    selected_ball["is_dls_match"] = (
        selected_ball["result"]
        .astype("string")
        .str.contains("DLS", case=False, na=False)
    )
    first_innings_totals = (
        selected_ball.loc[selected_ball["innings"].eq(1)]
        .groupby("match_id")["innings_total"]
        .first()
    )
    is_chase = selected_ball["innings"].eq(2)
    original_chase_target = selected_ball["match_id"].map(first_innings_totals).add(1)
    revised_targets = pd.to_numeric(
        selected_ball["revised_target"], errors="coerce"
    ).where(lambda values: values.gt(0))
    selected_ball["target_runs"] = np.where(
        is_chase,
        np.where(
            selected_ball["is_dls_match"],
            revised_targets,
            revised_targets.fillna(original_chase_target),
        ),
        0,
    )
    selected_ball["target_state_valid"] = (
        ~is_chase | selected_ball["target_runs"].notna()
    )
    selected_ball["dls_target_missing"] = (
        selected_ball["is_dls_match"] & revised_targets.isna()
    )

    revised_limit = pd.Series(
        np.where(
            selected_ball["innings"].eq(1),
            selected_ball["first_inn_revised_overs"],
            selected_ball["second_inn_revised_overs"],
        ),
        index=selected_ball.index,
    )
    revised_limit = pd.to_numeric(revised_limit, errors="coerce")
    schedule_is_dls_unknown = selected_ball["is_dls_match"] & revised_limit.isna()
    selected_ball["scheduled_balls"] = revised_limit.where(
        revised_limit.notna(),
        np.where(selected_ball["is_dls_match"], np.nan, 120.0),
    )
    selected_ball["schedule_state_valid"] = ~schedule_is_dls_unknown
    known_schedule = selected_ball["scheduled_balls"].notna()
    if (selected_ball.loc[known_schedule, "scheduled_balls"] <= 0).any():
        raise ValueError("A match has a non-positive scheduled-ball limit.")
    if (selected_ball.loc[known_schedule, "scheduled_balls"] % 6 != 0).any():
        raise ValueError(
            "Revised-over metadata is not expressed in whole six-ball units; "
            "review schedule interpretation."
        )
    selected_ball["state_valid"] = (
        selected_ball["schedule_state_valid"]
        & selected_ball["target_state_valid"]
    )
    state_reasons = np.where(
        selected_ball["schedule_state_valid"],
        "",
        "missing_revised_schedule",
    )
    state_reasons = np.where(
        selected_ball["target_state_valid"],
        state_reasons,
        state_reasons
        + np.where(state_reasons == "", "", ";")
        + "missing_revised_target",
    )
    selected_ball["state_invalid_reason"] = state_reasons

    selected_ball["balls_remaining"] = (
        selected_ball["scheduled_balls"] - selected_ball["legal_balls_before"]
    ).clip(lower=0)
    selected_ball["current_run_rate_before"] = np.where(
        selected_ball["legal_balls_before"].gt(0),
        selected_ball["current_score_before"]
        / selected_ball["legal_balls_before"].replace(0, np.nan)
        * 6,
        0,
    )
    selected_ball["runs_required_before"] = np.where(
        is_chase,
        (
            selected_ball["target_runs"] - selected_ball["current_score_before"]
        ).clip(lower=0),
        0,
    )
    selected_ball["required_run_rate_before"] = np.where(
        is_chase & selected_ball["balls_remaining"].gt(0),
        selected_ball["runs_required_before"]
        / selected_ball["balls_remaining"].replace(0, np.nan)
        * 6,
        0,
    )
    selected_ball["is_second_innings"] = selected_ball["innings"].eq(2).astype(int)
    selected_ball["competition_is_ipl"] = selected_ball[
        "competition_format"
    ].eq("IPL").astype(int)
    phase_labels = ["Powerplay (1-6)", "Middle (7-15)", "Slog (16-20)"]
    selected_ball["phase"] = pd.cut(
        selected_ball["overid"],
        bins=[0, 6, 15, 20],
        labels=phase_labels,
    )
    selected_ball["phase_code"] = selected_ball["overid"].map(
        lambda over: 0 if over <= 6 else (1 if over <= 15 else 2)
    )
    selected_ball["season"] = selected_ball["season"].astype(str)
    selected_ball["future_team_runs"] = (
        selected_ball["innings_total"] - selected_ball["current_score_before"]
    )

    selected_ball["wickets_after"] = (
        selected_ball["wickets_lost_before"] + selected_ball["is_wicket"].astype(int)
    )
    selected_ball["legal_balls_after"] = (
        selected_ball["legal_balls_before"] + selected_ball["is_legal_ball"].astype(int)
    )
    innings_ends = selected_ball.groupby(group_keys, sort=False).tail(1).copy()
    all_out = innings_ends["wickets_after"].ge(10)
    scheduled_end = (
        innings_ends["scheduled_balls"].notna()
        & innings_ends["legal_balls_after"].ge(innings_ends["scheduled_balls"])
    )
    target_reached = (
        innings_ends["innings"].eq(2)
        & innings_ends["target_runs"].notna()
        & innings_ends["innings_total"].ge(innings_ends["target_runs"])
    )
    chasing_team_won = (
        innings_ends["innings"].eq(2)
        & innings_ends["win_team_id"].notna()
        & innings_ends["batting_team_id"].eq(innings_ends["win_team_id"])
    )
    confirmed_complete = all_out | scheduled_end | target_reached | chasing_team_won
    no_result_or_abandoned = innings_ends["result"].astype("string").str.contains(
        r"no result|abandon", case=False, na=False
    )
    innings_ends["innings_completion_status"] = np.select(
        [confirmed_complete, no_result_or_abandoned],
        ["complete", "incomplete"],
        default="unknown",
    )
    completion_lookup = innings_ends.set_index(group_keys)[
        "innings_completion_status"
    ]
    selected_ball["innings_completion_status"] = pd.MultiIndex.from_frame(
        selected_ball[group_keys]
    ).map(completion_lookup)
    selected_ball["innings_completed"] = pd.Series(
        selected_ball["innings_completion_status"].eq("complete").to_numpy(),
        index=selected_ball.index,
        dtype="boolean",
    )
    selected_ball.loc[
        selected_ball["innings_completion_status"].eq("unknown"),
        "innings_completed",
    ] = pd.NA
    selected_ball["future_run_label_valid"] = (
        selected_ball["innings_completed"].fillna(False)
        & selected_ball["state_valid"]
    )
    selected_ball["par_model_eligible"] = (
        selected_ball["innings"].eq(1)
        & selected_ball["innings_completed"].fillna(False)
        & selected_ball["state_valid"]
    )

    roster_columns = ["match_id", "team_id", "player_id", "batting_order"]
    roster_columns.extend(
        column
        for column in ("is_played", "is_wk")
        if column in playing_eleven.columns
    )
    player_roster = playing_eleven[roster_columns].drop_duplicates()
    batters = selected_ball[
        ["match_id", "batting_team_id", "striker_id"]
    ].drop_duplicates()
    batters = batters.merge(
        player_roster,
        left_on=["match_id", "batting_team_id", "striker_id"],
        right_on=["match_id", "team_id", "player_id"],
        how="left",
        indicator=True,
    )
    bowlers = selected_ball[
        ["match_id", "bowling_team_id", "bowler_id"]
    ].drop_duplicates()
    bowlers = bowlers.merge(
        player_roster[["match_id", "team_id", "player_id"]].drop_duplicates(),
        left_on=["match_id", "bowling_team_id", "bowler_id"],
        right_on=["match_id", "team_id", "player_id"],
        how="left",
        indicator=True,
    )
    team_pair = selected_ball[
        ["match_id", "batting_team_id", "home_team_id", "away_team_id"]
    ].drop_duplicates()
    non_match_team_rows = int(
        (
            team_pair["batting_team_id"].ne(team_pair["home_team_id"])
            & team_pair["batting_team_id"].ne(team_pair["away_team_id"])
        ).sum()
    )
    unknown_extras_rows = selected_ball.loc[
        selected_ball["extras_type"].isna()
        & (selected_ball["extras1_runs"].ne(0) | selected_ball["extras_2_runs"].ne(0))
    ]
    other_type_rows = selected_ball["extras_type_normalized"].eq("other")
    overs_with_more_than_six_rows = selected_ball.groupby(
        ["match_id", "innings", "overid"]
    )["over_ball_num"].transform("max").gt(6)
    wicket_rows = selected_ball.loc[
        selected_ball["is_wicket"] & selected_ball["dismissed_player_id"].notna()
    ]
    dismissal_id_matches_delivery = (
        wicket_rows["dismissed_player_id"].eq(wicket_rows["striker_id"])
        | wicket_rows["dismissed_player_id"].eq(wicket_rows["non_striker_id"])
    )
    credited_wicket_rows = selected_ball.loc[
        selected_ball["is_wicket"]
        & selected_ball["dismissal_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .isin(CREDITED_DISMISSALS)
    ]

    quality = pd.DataFrame(
        [
            _quality_row("selected_metadata_matches", selected_matches["match_id"].nunique()),
            _quality_row(
                "selected_matches_without_decisive_winner",
                int(selected_matches["win_team_id"].isna().sum()),
                "No-result, abandoned, and tied matches are excluded from win-probability model training.",
            ),
            _quality_row("selected_matches_with_deliveries", selected_ball["match_id"].nunique()),
            _quality_row(
                "selected_matches_without_deliveries",
                selected_matches["match_id"].nunique()
                - selected_ball["match_id"].nunique(),
            ),
            _quality_row("selected_delivery_rows_after_super_over_exclusion", len(selected_ball)),
            _quality_row("duplicate_delivery_keys", duplicate_delivery_keys),
            _quality_row("source_delivery_rows_without_match_metadata", ball_rows_without_metadata),
            _quality_row("super_over_rows_excluded", super_over_rows),
            _quality_row(
                "duplicate_playing_eleven_keys",
                int(
                    tables["playing_eleven.csv"].duplicated(
                        ["match_id", "team_id", "player_id"]
                    ).sum()
                ),
                "Deduplicated for roster validation; delivery records remain the scoring source.",
            ),
            _quality_row(
                "duplicate_match_ids",
                int(matches["match_id"].duplicated().sum()),
            ),
            _quality_row(
                "duplicate_competition_ids",
                int(competitions["comp_id"].duplicated().sum()),
            ),
            _quality_row(
                "duplicate_competition_team_keys",
                int(tables["teams.csv"].duplicated(["comp_id", "team_id"]).sum()),
            ),
            _quality_row(
                "duplicate_competition_team_player_keys",
                int(
                    tables["competition_team_player_mapping.csv"].duplicated(
                        ["comp_id", "team_id", "player_id"]
                    ).sum()
                ),
            ),
            _quality_row(
                "batter_delivery_players_not_in_roster",
                int(batters["_merge"].eq("left_only").sum()),
                "Retained from ball-by-ball IDs (likely impact-substitution coverage).",
            ),
            _quality_row(
                "bowler_delivery_players_not_in_roster",
                int(bowlers["_merge"].eq("left_only").sum()),
            ),
            _quality_row("batting_team_not_home_or_away_rows", non_match_team_rows),
            _quality_row(
                "wickets_with_missing_dismissed_player_id",
                int(
                    selected_ball.loc[
                        selected_ball["is_wicket"], "dismissed_player_id"
                    ].isna().sum()
                ),
            ),
            _quality_row(
                "dismissed_player_not_striker_or_non_striker",
                int((~dismissal_id_matches_delivery).sum()),
                "Non-striker run-outs are retained and attributed by dismissed_player_id.",
            ),
            _quality_row(
                "credited_wickets_with_missing_bowler_id",
                int(credited_wicket_rows["bowler_id"].isna().sum()),
            ),
            _quality_row(
                "unclassified_extra_rows_with_extra_runs",
                len(unknown_extras_rows),
                "Extra amounts on rows with no extras_type are included in team totals but not bowler-charged runs.",
            ),
            _quality_row(
                "other_extra_rows",
                int(other_type_rows.sum()),
                "Treated as no-balls based on delivery-number patterns.",
            ),
            _quality_row(
                "other_extra_rows_in_overs_with_more_than_six_balls",
                int((other_type_rows & overs_with_more_than_six_rows).sum()),
            ),
            _quality_row(
                "wide_and_penalty_rows_excluded_from_batter_opportunities",
                int((~selected_ball["is_batter_opportunity"]).sum()),
                "No-balls count as batter opportunities; wides and administrative penalty rows do not.",
            ),
            _quality_row(
                "second_innings_rows_without_target",
                int(
                    (
                        selected_ball["innings"].eq(2)
                        & selected_ball["target_runs"].isna()
                    ).sum()
                ),
            ),
            _quality_row(
                "scheduled_ball_limits_not_default_120",
                int(selected_ball["scheduled_balls"].ne(120).groupby(selected_ball["match_id"]).any().sum()),
                "Revised-limit metadata values are multiples of six and are used as ball counts.",
            ),
        ]
    )
    dls_matches = selected_matches.loc[
        selected_matches["result"].astype("string").str.contains(
            "DLS", case=False, na=False
        )
    ]
    dls_targets = pd.to_numeric(dls_matches["revised_target"], errors="coerce")
    dls_missing_target = dls_targets.isna() | dls_targets.le(0)
    first_innings_records = selected_ball.loc[
        selected_ball["innings"].eq(1),
        ["match_id", "par_model_eligible", "innings_completion_status"],
    ].drop_duplicates("match_id")
    valid_delivery_outcomes = (
        selected_ball["batter_runs"].notna()
        & selected_ball["bowler_runs_conceded"].notna()
        & (selected_ball["is_batter_opportunity"] | selected_ball["is_bowler_opportunity"])
    )
    summary_quality = [
        _quality_row("selected_dls_matches", dls_matches["match_id"].nunique()),
        _quality_row(
            "dls_matches_missing_revised_target",
            int(dls_missing_target.sum()),
            "A DLS chase without a revised target has target_runs=NaN and state_valid=False; no first-innings fallback is used.",
        ),
        _quality_row(
            "deliveries_excluded_from_state_dependent_analysis",
            int((~selected_ball["state_valid"]).sum()),
            "Rows remain in pressure_deliveries.csv for observed outcomes, but are excluded from state-dependent models, pressure scores, and player events.",
        ),
        _quality_row(
            "incomplete_innings",
            int(
                selected_ball.loc[
                    selected_ball["innings_completion_status"].eq("incomplete"),
                    ["match_id", "innings"],
                ]
                .drop_duplicates()
                .shape[0]
            ),
            "Confirmed incomplete when a no-result/abandoned innings has no evidence of reaching all-out, its known schedule, or its target.",
        ),
        _quality_row(
            "innings_completion_unknown",
            int(
                selected_ball.loc[
                    selected_ball["innings_completion_status"].eq("unknown"),
                    ["match_id", "innings"],
                ].drop_duplicates().shape[0]
            ),
            "Completion cannot be established from the available result, wickets, target, and schedule fields.",
        ),
        _quality_row(
            "future_run_labels_excluded",
            int((~selected_ball["future_run_label_valid"]).sum()),
            "Future-innings-run labels are training-eligible only for completed innings with valid state metadata.",
        ),
        _quality_row(
            "future_run_labels_excluded_incomplete_innings",
            int(
                selected_ball["innings_completion_status"]
                .eq("incomplete")
                .sum()
            ),
            "Observed delivery outcomes remain available; only the truncated innings-total-derived future-run target is excluded.",
        ),
        _quality_row(
            "future_run_labels_excluded_unknown_completion",
            int(
                selected_ball["innings_completion_status"]
                .eq("unknown")
                .sum()
            ),
            "Completion uncertainty is not treated as a completed innings.",
        ),
        _quality_row(
            "par_model_observations_excluded",
            int((~first_innings_records["par_model_eligible"]).sum()),
            "Par estimation uses only first innings with confirmed completion and valid match-state metadata.",
        ),
        _quality_row(
            "par_model_observations_retained",
            int(first_innings_records["par_model_eligible"].sum()),
        ),
        _quality_row(
            "valid_delivery_level_run_outcomes_retained",
            int(valid_delivery_outcomes.sum()),
            "Actual batter runs and bowler runs remain in the delivery data, including no-result matches; no-result status alone does not remove a delivery target.",
        ),
        _quality_row(
            "scheduled_ball_limit_unknown_rows",
            int(selected_ball["scheduled_balls"].isna().sum()),
            "DLS rows without the corresponding revised schedule are not assigned a 120-ball fallback.",
        ),
    ]
    quality = pd.concat(
        [quality, pd.DataFrame(summary_quality)], ignore_index=True
    )
    match_rows = []
    for match in dls_matches.itertuples(index=False):
        match_id = match.match_id
        match_deliveries = selected_ball.loc[
            selected_ball["match_id"].eq(match_id)
        ]
        missing_fields = []
        revised_target = pd.to_numeric(
            pd.Series([match.revised_target]), errors="coerce"
        ).iloc[0]
        if pd.isna(revised_target) or revised_target <= 0:
            missing_fields.append("revised_target")
        for innings, field in (
            (1, "first_inn_revised_overs"),
            (2, "second_inn_revised_overs"),
        ):
            if (
                match_deliveries["innings"].eq(innings).any()
                and pd.isna(getattr(match, field))
            ):
                missing_fields.append(field)
        if missing_fields:
            affected_rows = int((~match_deliveries["state_valid"]).sum())
            match_rows.append(
                _quality_row(
                    "dls_match_state_treatment",
                    match_id,
                    "Missing fields: "
                    + ", ".join(missing_fields)
                    + f"; affected state-invalid deliveries: {affected_rows}; "
                    "missing target/schedule is kept unknown, not imputed; "
                    "observed delivery outcomes are retained.",
                )
            )
    if match_rows:
        quality = pd.concat(
            [quality, pd.DataFrame(match_rows)], ignore_index=True
        )
    return selected_ball, player_roster, quality, audit
