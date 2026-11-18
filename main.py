import argparse
from pathlib import Path

import pandas as pd

from src.pressure_rankings.config import (
    BASELINE_WEIGHTS,
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    MIN_NORMAL_OPPORTUNITIES,
    MIN_PRESSURE_MATCHES,
    MIN_PRESSURE_OPPORTUNITIES,
    PRESSURE_CUTOFF,
    PRESSURE_WEIGHT_SCENARIOS,
    SENSITIVITY_CUTOFFS,
    SENSITIVITY_ELIGIBILITY,
)
from src.pressure_rankings.data import load_and_prepare
from src.pressure_rankings.models import fit_oof_expectations
from src.pressure_rankings.reporting import write_outputs
from src.pressure_rankings.roles import build_role_leaderboards
from src.pressure_rankings.impact import (
    build_player_events,
    compare_rankings,
    rank_players,
)
from src.pressure_rankings.scoring import calculate_pressure


def _arguments():
    parser = argparse.ArgumentParser(
        description="Build IPL and T20I pressure-player rankings."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the six supplied CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for CSV, PNG and methodology outputs.",
    )
    return parser.parse_args()


def _run_sensitivity(deliveries, batter_events, bowler_events, base_rankings):
    scenarios = [
        {
            "scenario": "weights:equal",
            "weights": PRESSURE_WEIGHT_SCENARIOS["equal"],
            "cutoff": PRESSURE_CUTOFF,
            "min_opportunities": MIN_PRESSURE_OPPORTUNITIES,
            "min_matches": MIN_PRESSURE_MATCHES,
            "weight_mode": "pca",
        },
        {
            "scenario": "weights:rate_focused",
            "weights": PRESSURE_WEIGHT_SCENARIOS["rate_focused"],
            "cutoff": PRESSURE_CUTOFF,
            "min_opportunities": MIN_PRESSURE_OPPORTUNITIES,
            "min_matches": MIN_PRESSURE_MATCHES,
            "weight_mode": "pca",
        },
        {
            "scenario": "weights:resource_focused",
            "weights": PRESSURE_WEIGHT_SCENARIOS["resource_focused"],
            "cutoff": PRESSURE_CUTOFF,
            "min_opportunities": MIN_PRESSURE_OPPORTUNITIES,
            "min_matches": MIN_PRESSURE_MATCHES,
            "weight_mode": "pca",
        },
        {
            "scenario": "component_weights:equal",
            "weights": BASELINE_WEIGHTS,
            "cutoff": PRESSURE_CUTOFF,
            "min_opportunities": MIN_PRESSURE_OPPORTUNITIES,
            "min_matches": MIN_PRESSURE_MATCHES,
            "weight_mode": "equal",
        },
    ]
    for cutoff in SENSITIVITY_CUTOFFS:
        scenarios.append(
            {
                "scenario": f"cutoff:{cutoff:.2f}",
                "weights": BASELINE_WEIGHTS,
                "cutoff": cutoff,
                "min_opportunities": MIN_PRESSURE_OPPORTUNITIES,
                "min_matches": MIN_PRESSURE_MATCHES,
                "weight_mode": "pca",
            }
        )
    for opportunities, matches in SENSITIVITY_ELIGIBILITY:
        scenarios.append(
            {
                "scenario": f"eligibility:{opportunities}/{matches}",
                "weights": BASELINE_WEIGHTS,
                "cutoff": PRESSURE_CUTOFF,
                "min_opportunities": opportunities,
                "min_matches": matches,
                "weight_mode": "pca",
            }
        )

    top_rows = []
    stability_rows = []
    for scenario in scenarios:
        pressure = calculate_pressure(deliveries, scenario["weights"])
        pressure_values = pressure["pressure_index"].to_numpy()
        for player_type, events in (
            ("batter", batter_events),
            ("bowler", bowler_events),
        ):
            scenario_events = events.copy()
            scenario_events["pressure_index"] = pressure_values[
                scenario_events["delivery_id"].to_numpy(dtype=int)
            ]
            _, eligible, _, _, _, _, _ = rank_players(
                scenario_events,
                cutoff=scenario["cutoff"],
                min_pressure_opportunities=scenario["min_opportunities"],
                min_pressure_matches=scenario["min_matches"],
                weight_mode=scenario["weight_mode"],
            )
            scenario_rankings = eligible.copy()
            if not scenario_rankings.empty:
                scenario_rankings["player_type"] = player_type
                scenario_rankings["scenario"] = scenario["scenario"]
                top_rows.append(scenario_rankings)

            stability_rows.append(
                compare_rankings(
                    base_rankings[player_type],
                    eligible,
                    scenario["scenario"],
                    player_type,
                    {
                        "pressure_cutoff": scenario["cutoff"],
                        "min_pressure_opportunities": scenario[
                            "min_opportunities"
                        ],
                        "min_pressure_matches": scenario["min_matches"],
                        "component_weight_mode": scenario["weight_mode"],
                    },
                )
            )

    rankings = (
        pd.concat(top_rows, ignore_index=True, sort=False)
        if top_rows
        else pd.DataFrame()
    )
    return rankings, pd.DataFrame(stability_rows)


def _append_model_quality(quality, modeled, pressured):
    rows = []
    no_result_wpa = (
        modeled["wpa_attribution_status"].eq("excluded_no_decisive_result")
        & modeled["win_probability_added"].notna()
    )
    ambiguous_wpa = (
        modeled["wpa_attribution_status"].eq(
            "team_state_proxy_not_individually_attributed"
        )
        & modeled["win_probability_added"].notna()
    )
    rows.extend(
        [
            {
                "check": "deliveries_excluded_from_state_dependent_analysis",
                "value": int((~pressured["pressure_state_valid"]).sum()),
                "detail": (
                    "Includes unknown targets/schedules and rows without an OOF "
                    "future-run/par estimate; excluded from pressure and player-event scoring."
                ),
            },
            {
                "check": "no_result_wpa_events_excluded_from_individual_rankings",
                "value": int(no_result_wpa.sum()),
                "detail": (
                    "WPA estimates may remain in pressure_deliveries.csv for diagnosis, "
                    "but no-result matches contribute no player WPA score."
                ),
            },
            {
                "check": "ambiguous_individual_wpa_attributions",
                "value": int(ambiguous_wpa.sum()),
                "detail": (
                    "Decisive-match delivery-level team-state WPA proxies retained "
                    "for diagnostics but deliberately not assigned to a player or "
                    "included in the composite."
                ),
            },
        ]
    )
    for context, count in (
        modeled.loc[ambiguous_wpa, "wpa_proxy_event_context"]
        .value_counts()
        .items()
    ):
        rows.append(
            {
                "check": f"ambiguous_wpa_proxy_context_{context}",
                "value": int(count),
                "detail": "Team-state WPA proxy is not individually attributed.",
            }
        )
    retained = quality.loc[
        ~quality["check"].eq("deliveries_excluded_from_state_dependent_analysis")
    ]
    return pd.concat(
        [retained, pd.DataFrame(rows)], ignore_index=True
    )


def run_pipeline(data_dir=DEFAULT_DATA_DIR, output_dir=DEFAULT_OUTPUT_DIR):
    prepared, player_roster, quality, schema_audit = load_and_prepare(data_dir)
    modeled, validation = fit_oof_expectations(prepared)
    pressured = calculate_pressure(modeled)
    quality = _append_model_quality(quality, modeled, pressured)
    batter_events, bowler_events = build_player_events(pressured, player_roster)

    (
        batter_metrics,
        batter_eligible,
        batter_match_metrics,
        batter_format_metrics,
        batter_format_match_metrics,
        batter_phase_leaderboards,
        batter_component_weights,
    ) = rank_players(batter_events)
    (
        bowler_metrics,
        bowler_eligible,
        bowler_match_metrics,
        bowler_format_metrics,
        bowler_format_match_metrics,
        bowler_phase_leaderboards,
        bowler_component_weights,
    ) = rank_players(bowler_events)

    role_leaderboards = build_role_leaderboards(
        batter_metrics,
        bowler_metrics,
        pressured,
        batter_eligible,
        bowler_eligible,
    )

    sensitivity_rankings, ranking_stability = _run_sensitivity(
        pressured,
        batter_events,
        bowler_events,
        {"batter": batter_eligible, "bowler": bowler_eligible},
    )
    format_comparison = write_outputs(
        output_dir=output_dir,
        prepared=pressured,
        quality=quality,
        schema_audit=schema_audit,
        validation=validation,
        batter_metrics=batter_metrics,
        batter_top=role_leaderboards["batters"].head(10),
        batter_match_metrics=batter_match_metrics,
        batter_format_metrics=batter_format_metrics,
        bowler_metrics=bowler_metrics,
        bowler_top=role_leaderboards["bowlers"].head(10),
        bowler_match_metrics=bowler_match_metrics,
        bowler_format_metrics=bowler_format_metrics,
        sensitivity_rankings=sensitivity_rankings,
        stability=ranking_stability,
        batter_phase_leaderboards=batter_phase_leaderboards,
        bowler_phase_leaderboards=bowler_phase_leaderboards,
        component_weights=pd.concat(
            [batter_component_weights, bowler_component_weights],
            ignore_index=True,
        ),
        role_classification=role_leaderboards["role_classification"],
        unclassified_players=role_leaderboards["unclassified_players"],
        leaderboard_changes=role_leaderboards["leaderboard_changes"],
    )

    output_dir = Path(output_dir)
    print(
        f"Analyzed {len(pressured):,} deliveries from "
        f"{pressured['match_id'].nunique():,} IPL/T20I matches."
    )
    print("\nModel validation (match-grouped out-of-fold):")
    print(
        validation[
            [
                "target",
                "observations",
                "mae",
                "rmse",
                "r2",
                "brier_score",
                "log_loss",
                "roc_auc",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )
    role_counts = role_leaderboards["role_classification"][
        "assigned_role"
    ].value_counts()
    print(
        "\nRole counts: "
        f"BATTER={int(role_counts.get('BATTER', 0))}, "
        f"BOWLER={int(role_counts.get('BOWLER', 0))}, "
        f"UNCLASSIFIED={int(role_counts.get('UNCLASSIFIED', 0))}"
    )
    print(
        "Invalid position-12 player-innings excluded from role percentages: "
        f"{int(role_leaderboards['role_classification']['position_12_innings'].sum())}"
    )
    for label, top in (
        ("Top batters", role_leaderboards["batters"].head(10)),
        ("Top bowlers", role_leaderboards["bowlers"].head(10)),
    ):
        print(f"\n{label} ({len(top)} eligible):")
        if top.empty:
            print("No player met the evidence thresholds.")
        else:
            print(
                top[
                    [
                        "rank",
                        "player_name",
                        "ranking_score",
                        "batting_pressure_opportunities"
                        if label == "Top batters"
                        else "bowling_pressure_opportunities",
                        "batting_pressure_matches"
                        if label == "Top batters"
                        else "bowling_pressure_matches",
                    ]
                ]
                .round(3)
                .to_string(index=False)
            )
    print("\nLeaderboard changes from previous top 10:")
    print(
        role_leaderboards["leaderboard_changes"][
            [
                "leaderboard",
                "retained_count",
                "moved_out_count",
                "moved_in_count",
            ]
        ].to_string(index=False)
    )
    print(f"\nOutputs written to {output_dir.resolve()}")
    return {
        "batter_rankings": batter_eligible,
        "bowler_rankings": bowler_eligible,
        "role_classification": role_leaderboards["role_classification"],
        "top_batters": role_leaderboards["batters"],
        "top_bowlers": role_leaderboards["bowlers"],
        "unclassified_players": role_leaderboards["unclassified_players"],
        "leaderboard_changes": role_leaderboards["leaderboard_changes"],
        "validation": validation,
        "format_comparison": format_comparison,
        "ranking_stability": ranking_stability,
    }


if __name__ == "__main__":
    args = _arguments()
    run_pipeline(data_dir=args.data_dir, output_dir=args.output_dir)
