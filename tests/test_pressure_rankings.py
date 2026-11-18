import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.pressure_rankings.data import load_and_prepare
from src.pressure_rankings.models import (
    _future_run_label_mask,
    _wpa_diagnostic_labels,
)
from src.pressure_rankings.scoring import (
    build_player_events,
    calculate_pressure,
)
from src.pressure_rankings.config import PRESSURE_CUTOFF
from src.pressure_rankings.impact import (
    build_player_events as build_impact_events,
    rank_players,
)
from src.pressure_rankings.roles import (
    BATTER,
    BOWLER,
    UNCLASSIFIED,
    _build_batting_order_history,
    build_role_leaderboards,
)


class PlayerRoleLeaderboardTests(unittest.TestCase):
    @staticmethod
    def _metrics(player_ids, scores, eligible=None):
        if eligible is None:
            eligible = [True] * len(player_ids)
        return pd.DataFrame(
            {
                "metric_scope": ["overall"] * len(player_ids),
                "player_id": player_ids,
                "player_name": [f"Player {player_id}" for player_id in player_ids],
                "eligible": eligible,
                "rank": range(1, len(player_ids) + 1),
                "ranking_score": scores,
                "pressure_composite": scores,
                "normal_composite": [score - 0.1 for score in scores],
                "clutch_uplift": [0.1] * len(player_ids),
                "pressure_opportunities": [40] * len(player_ids),
                "pressure_matches": [9] * len(player_ids),
                "normal_opportunities": [40] * len(player_ids),
                "sample_shrinkage": [0.5] * len(player_ids),
                "consistency_factor": [0.8] * len(player_ids),
                "stability_factor": [0.9] * len(player_ids),
            }
        )

    @staticmethod
    def _deliveries():
        orders = {
            1: [4] * 7 + [8] * 3,
            2: [11] * 4 + [1],
            3: [1] * 3 + [8] * 2,
            4: [2] * 4,
            5: [1] * 4 + [12],
        }
        rows = []
        match_id = 1
        for player_id, player_orders in orders.items():
            for order in player_orders:
                for _ in range(2):
                    rows.append(
                        {
                            "match_id": match_id,
                            "innings": 1,
                            "striker_id": player_id,
                            "striker_name": f"Player {player_id}",
                            "batting_order": order,
                        }
                    )
                match_id += 1
        return pd.DataFrame(rows)

    def _build(self, batting_eligible=None, bowling_eligible=None):
        player_ids = [1, 2, 3, 4, 5, 6]
        batter_metrics = self._metrics(
            player_ids,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            batting_eligible,
        )
        bowler_metrics = self._metrics(
            player_ids,
            [0.8, 0.9, 0.7, 0.6, 0.5, 0.4],
            bowling_eligible,
        )
        previous = pd.DataFrame(
            {
                "player_id": [1, 2],
                "player_name": ["Player 1", "Player 2"],
            }
        )
        result = build_role_leaderboards(
            batter_metrics,
            bowler_metrics,
            self._deliveries(),
            previous,
            previous,
        )
        return result, batter_metrics, bowler_metrics

    def test_player_innings_are_counted_once_not_per_delivery(self):
        history = _build_batting_order_history(self._deliveries())
        player = history.set_index("player_id").loc[1]
        self.assertEqual(player["total_innings"], 10)
        self.assertEqual(player["valid_batting_order_innings"], 10)
        self.assertEqual(player["batting_order_1_7_innings"], 7)
        self.assertAlmostEqual(player["batting_order_1_7_pct"], 0.70)

    def test_seventy_percent_threshold_and_five_innings_minimum(self):
        result, _, _ = self._build()
        roles = result["role_classification"].set_index("player_id")
        self.assertEqual(roles.loc[1, "assigned_role"], BATTER)
        self.assertEqual(roles.loc[2, "assigned_role"], BOWLER)
        self.assertEqual(roles.loc[3, "assigned_role"], UNCLASSIFIED)
        self.assertEqual(roles.loc[3, "role_resolution"], "mixed_batting_order_distribution")
        self.assertEqual(roles.loc[4, "role_resolution"], "insufficient_batting_order_history")

    def test_position_twelve_is_reported_but_not_used_for_history_or_percentages(self):
        result, _, _ = self._build()
        player = result["role_classification"].set_index("player_id").loc[5]
        self.assertEqual(player["total_innings"], 5)
        self.assertEqual(player["valid_batting_order_innings"], 4)
        self.assertEqual(player["position_12_innings"], 1)
        self.assertEqual(player["invalid_batting_order_innings"], 1)
        self.assertEqual(player["assigned_role"], UNCLASSIFIED)

    def test_role_filtering_is_exclusive_and_retains_unclassified_metrics(self):
        result, batter_metrics, bowler_metrics = self._build()
        self.assertEqual(result["batters"]["player_id"].tolist(), [1])
        self.assertEqual(result["bowlers"]["player_id"].tolist(), [2])
        self.assertTrue(
            set(result["batters"]["player_id"]).isdisjoint(
                result["bowlers"]["player_id"]
            )
        )
        unresolved = result["unclassified_players"].set_index("player_id")
        self.assertAlmostEqual(unresolved.loc[3, "batting_ranking_score"], 0.7)
        self.assertIn(3, set(batter_metrics["player_id"]))
        self.assertIn(3, set(bowler_metrics["player_id"]))

    def test_existing_eligibility_and_scores_are_unchanged(self):
        self.assertEqual(PRESSURE_CUTOFF, 0.55)
        player_ids = [1, 2, 3, 4, 5, 6]
        batter_metrics = self._metrics(
            player_ids,
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            [False, True, True, True, True, True],
        )
        bowler_metrics = self._metrics(
            player_ids,
            [0.8, 0.9, 0.7, 0.6, 0.5, 0.4],
            [True, False, True, True, True, True],
        )
        original_batter_metrics = batter_metrics.copy(deep=True)
        original_bowler_metrics = bowler_metrics.copy(deep=True)
        previous = pd.DataFrame(
            {"player_id": [1, 2], "player_name": ["Player 1", "Player 2"]}
        )
        result = build_role_leaderboards(
            batter_metrics,
            bowler_metrics,
            self._deliveries(),
            previous,
            previous,
        )
        self.assertTrue(result["batters"].empty)
        self.assertTrue(result["bowlers"].empty)
        roles = result["role_classification"].set_index("player_id")
        self.assertEqual(roles.loc[1, "batting_ranking_score"], 0.9)
        self.assertEqual(roles.loc[2, "bowling_ranking_score"], 0.9)
        self.assertEqual(PRESSURE_CUTOFF, 0.55)
        pd.testing.assert_frame_equal(batter_metrics, original_batter_metrics)
        pd.testing.assert_frame_equal(bowler_metrics, original_bowler_metrics)

    def test_previous_leaderboard_changes_are_counted(self):
        result, _, _ = self._build()
        changes = result["leaderboard_changes"].set_index("leaderboard")
        self.assertEqual(changes.loc["BATTER", "moved_out_count"], 1)
        self.assertEqual(changes.loc["BATTER", "moved_in_count"], 0)
        self.assertEqual(changes.loc["BOWLER", "moved_out_count"], 1)
        self.assertEqual(changes.loc["BOWLER", "moved_in_count"], 0)

class PressureScoringTests(unittest.TestCase):
    def test_pressure_index_is_bounded_and_first_innings_has_no_leverage(self):
        deliveries = pd.DataFrame(
            {
                "innings": [1, 2],
                "legal_balls_before": [60, 60],
                "scheduled_balls": [120, 120],
                "wickets_lost_before": [2, 5],
                "current_score_before": [70, 120],
                "expected_future_team_runs": [90.0, 40.0],
                "training_par_total": [175.0, 175.0],
                "runs_required_before": [0, 40],
            }
        )
        scored = calculate_pressure(deliveries)
        self.assertTrue(scored["pressure_index"].between(0, 1).all())
        self.assertEqual(scored.loc[0, "pressure_leverage"], 0)
        self.assertGreater(scored.loc[1, "pressure_leverage"], 0)

    def test_dismissal_and_bowler_credit_use_the_recorded_player_and_type(self):
        deliveries = pd.DataFrame(
            {
                "delivery_id": [0, 1],
                "match_id": [7, 7],
                "competition_format": ["IPL", "IPL"],
                "season": ["2025", "2025"],
                "phase": ["Powerplay (1-6)", "Powerplay (1-6)"],
                "pressure_index": [0.8, 0.2],
                "is_pressure": [True, False],
                "striker_id": [10, 10],
                "striker_name": ["Batter A", "Batter A"],
                "non_striker_id": [11, 11],
                "non_striker_name": ["Batter B", "Batter B"],
                "batting_order": [1, 1],
                "is_batter_opportunity": [True, True],
                "batter_runs": [1, 0],
                "expected_batter_runs": [0.5, 0.5],
                "is_wicket": [True, True],
                "dismissed_player_id": [11, 10],
                "wicket_run_value": [4.0, 5.0],
                "batting_team_id": [1, 1],
                "dismissal_type": ["RunOut", "Caught"],
                "bowler_id": [20, 20],
                "bowler_name": ["Bowler A", "Bowler A"],
                "expected_bowler_runs": [1.0, 1.0],
                "bowler_runs_conceded": [1.0, 1.0],
            }
        )
        roster = pd.DataFrame(
            {
                "match_id": [7, 7],
                "team_id": [1, 1],
                "player_id": [10, 11],
                "batting_order": [1, 5],
            }
        )
        batters, bowlers = build_player_events(deliveries, roster)

        non_striker_dismissal = batters.loc[
            batters["player_id"].eq(11) & batters["event_type"].eq("dismissal")
        ].iloc[0]
        self.assertEqual(non_striker_dismissal["dismissal_cost"], 4)
        self.assertEqual(non_striker_dismissal["opportunity"], 0)
        self.assertEqual(non_striker_dismissal["role"], "Middle order")

        self.assertEqual(bowlers["credited_wicket"].tolist(), [0, 1])
        self.assertEqual(bowlers.loc[0, "contribution"], 0.0)
        self.assertEqual(bowlers.loc[1, "contribution"], 5.0)

    def test_expanded_events_keep_dismissal_but_do_not_assign_team_wpa(self):
        deliveries = pd.DataFrame(
            {
                "delivery_id": [0, 1],
                "match_id": [7, 7],
                "competition_format": ["IPL", "IPL"],
                "season": ["2025", "2025"],
                "phase": ["Powerplay (1-6)", "Powerplay (1-6)"],
                "pressure_index": [0.8, 0.2],
                "is_batter_opportunity": [True, True],
                "is_bowler_opportunity": [True, True],
                "striker_id": [10, 10],
                "striker_name": ["Batter A", "Batter A"],
                "non_striker_id": [11, 11],
                "non_striker_name": ["Batter B", "Batter B"],
                "batting_team_id": [1, 1],
                "batting_order": [1, 1],
                "batter_runs": [1, 4],
                "expected_batter_runs": [0.5, 1.5],
                "is_four": [False, True],
                "is_six": [False, False],
                "is_boundary": [False, True],
                "expected_batter_four_probability": [0.1, 0.2],
                "expected_batter_six_probability": [0.05, 0.1],
                "expected_batter_boundary_probability": [0.15, 0.3],
                "is_wicket": [True, True],
                "is_credited_wicket": [False, True],
                "dismissed_player_id": [11, 10],
                "dismissal_type": ["RunOut", "Caught"],
                "expected_any_wicket_probability": [0.1, 0.1],
                "wicket_run_value": [4.0, 5.0],
                "bowler_id": [20, 20],
                "bowler_name": ["Bowler A", "Bowler A"],
                "expected_bowler_runs": [1.0, 2.0],
                "bowler_runs_conceded": [1.0, 4.0],
                "is_dot_ball": [False, False],
                "expected_dot_probability": [0.3, 0.2],
                "expected_bowler_boundary_probability": [0.1, 0.2],
                "expected_credited_wicket_probability": [0.05, 0.1],
                "win_probability_added": [0.1, -0.2],
            }
        )
        roster = pd.DataFrame(
            {
                "match_id": [7, 7],
                "team_id": [1, 1],
                "player_id": [10, 11],
                "batting_order": [1, 5],
            }
        )
        batters, bowlers = build_impact_events(deliveries, roster)

        run_out = batters.loc[
            batters["event_type"].eq("dismissal") & batters["player_id"].eq(11)
        ].iloc[0]
        self.assertEqual(run_out["role"], "Middle order")
        self.assertEqual(run_out["wicket_preservation"], -4.0)
        self.assertFalse(batters["event_type"].eq("win_probability").any())
        self.assertNotIn("wpa_added", batters.columns)
        self.assertEqual(bowlers["actual_credited_wickets"].tolist(), [0, 1])
        self.assertNotIn("wpa_added", bowlers.columns)

    def test_no_result_delivery_runs_are_retained_without_individual_wpa(self):
        deliveries = pd.DataFrame(
            {
                "delivery_id": [0],
                "match_id": [8],
                "result": ["No Result"],
                "win_team_id": [np.nan],
                "batting_team_won": [np.nan],
                "competition_format": ["T20I"],
                "season": ["2025"],
                "phase": ["Powerplay (1-6)"],
                "pressure_index": [0.7],
                "is_batter_opportunity": [True],
                "is_bowler_opportunity": [True],
                "striker_id": [10],
                "striker_name": ["Batter A"],
                "non_striker_id": [11],
                "non_striker_name": ["Batter B"],
                "batting_team_id": [1],
                "batting_order": [1],
                "batter_runs": [3],
                "expected_batter_runs": [1.0],
                "is_four": [False],
                "is_six": [False],
                "is_boundary": [False],
                "expected_batter_four_probability": [0.1],
                "expected_batter_six_probability": [0.05],
                "expected_batter_boundary_probability": [0.15],
                "expected_any_wicket_probability": [0.1],
                "is_wicket": [False],
                "is_credited_wicket": [False],
                "dismissed_player_id": [np.nan],
                "dismissal_type": [None],
                "wicket_run_value": [4.0],
                "bowler_id": [20],
                "bowler_name": ["Bowler A"],
                "expected_bowler_runs": [1.0],
                "bowler_runs_conceded": [3.0],
                "is_dot_ball": [False],
                "expected_dot_probability": [0.3],
                "expected_bowler_boundary_probability": [0.1],
                "expected_credited_wicket_probability": [0.05],
                "win_probability_added": [0.2],
            }
        )
        roster = pd.DataFrame(
            columns=["match_id", "team_id", "player_id", "batting_order"]
        )
        batters, bowlers = build_impact_events(deliveries, roster)
        batter_delivery = batters.loc[batters["event_type"].eq("batted_delivery")].iloc[0]
        self.assertEqual(batter_delivery["actual_runs"], 3)
        self.assertEqual(batter_delivery["runs_above_expected"], 2)
        self.assertFalse(batters["event_type"].eq("win_probability").any())
        self.assertNotIn("wpa_added", bowlers.columns)

    def test_wpa_diagnostics_mark_extras_and_no_result_without_player_credit(self):
        deliveries = pd.DataFrame(
            {
                "batting_team_won": [1.0, np.nan, 1.0],
                "extras_type": ["wide", "legbyes", "overstep"],
                "is_wicket": [False, False, False],
                "dismissed_player_id": [np.nan, np.nan, np.nan],
                "dismissal_type": [None, None, None],
            }
        )
        status, context = _wpa_diagnostic_labels(
            deliveries, np.array([True, True, True])
        )
        self.assertEqual(status[0], "team_state_proxy_not_individually_attributed")
        self.assertEqual(status[1], "excluded_no_decisive_result")
        self.assertEqual(context.tolist(), ["wide", "leg_bye", "no_ball"])

    def test_wide_keeps_bowler_outcome_but_has_no_individual_wpa(self):
        deliveries = pd.DataFrame(
            {
                "delivery_id": [0],
                "match_id": [9],
                "competition_format": ["IPL"],
                "season": ["2025"],
                "phase": ["Powerplay (1-6)"],
                "extras_type": ["wide"],
                "pressure_index": [0.8],
                "is_batter_opportunity": [False],
                "is_bowler_opportunity": [True],
                "striker_id": [10],
                "striker_name": ["Batter A"],
                "non_striker_id": [11],
                "non_striker_name": ["Batter B"],
                "batting_team_id": [1],
                "batting_order": [1],
                "batter_runs": [0],
                "expected_batter_runs": [1.0],
                "is_four": [False],
                "is_six": [False],
                "is_boundary": [False],
                "expected_batter_four_probability": [0.1],
                "expected_batter_six_probability": [0.05],
                "expected_batter_boundary_probability": [0.15],
                "expected_any_wicket_probability": [0.1],
                "is_wicket": [False],
                "is_credited_wicket": [False],
                "dismissed_player_id": [np.nan],
                "dismissal_type": [None],
                "wicket_run_value": [4.0],
                "bowler_id": [20],
                "bowler_name": ["Bowler A"],
                "expected_bowler_runs": [1.0],
                "bowler_runs_conceded": [1.0],
                "is_dot_ball": [False],
                "expected_dot_probability": [0.3],
                "expected_bowler_boundary_probability": [0.1],
                "expected_credited_wicket_probability": [0.05],
                "win_probability_added": [0.1],
            }
        )
        roster = pd.DataFrame(
            columns=["match_id", "team_id", "player_id", "batting_order"]
        )
        batters, bowlers = build_impact_events(deliveries, roster)
        self.assertTrue(batters.empty)
        self.assertEqual(bowlers.iloc[0]["runs_saved"], 0)
        self.assertNotIn("wpa_added", bowlers.columns)

    def test_multimetric_rankings_include_clutch_and_phase_scores(self):
        deliveries = pd.DataFrame(
            {
                "delivery_id": [0, 1],
                "match_id": [7, 7],
                "competition_format": ["IPL", "IPL"],
                "season": ["2025", "2025"],
                "phase": ["Powerplay (1-6)", "Powerplay (1-6)"],
                "pressure_index": [0.8, 0.2],
                "is_batter_opportunity": [True, True],
                "is_bowler_opportunity": [True, True],
                "striker_id": [10, 10],
                "striker_name": ["Batter A", "Batter A"],
                "non_striker_id": [11, 11],
                "non_striker_name": ["Batter B", "Batter B"],
                "batting_team_id": [1, 1],
                "batting_order": [1, 1],
                "batter_runs": [1, 4],
                "expected_batter_runs": [0.5, 1.5],
                "is_four": [False, True],
                "is_six": [False, False],
                "is_boundary": [False, True],
                "expected_batter_four_probability": [0.1, 0.2],
                "expected_batter_six_probability": [0.05, 0.1],
                "expected_batter_boundary_probability": [0.15, 0.3],
                "is_wicket": [False, False],
                "is_credited_wicket": [False, False],
                "dismissed_player_id": [None, None],
                "dismissal_type": [None, None],
                "expected_any_wicket_probability": [0.1, 0.1],
                "wicket_run_value": [4.0, 5.0],
                "bowler_id": [20, 20],
                "bowler_name": ["Bowler A", "Bowler A"],
                "expected_bowler_runs": [1.0, 2.0],
                "bowler_runs_conceded": [1.0, 4.0],
                "is_dot_ball": [False, False],
                "expected_dot_probability": [0.3, 0.2],
                "expected_bowler_boundary_probability": [0.1, 0.2],
                "expected_credited_wicket_probability": [0.05, 0.1],
                "win_probability_added": [0.1, -0.2],
            }
        )
        roster = pd.DataFrame(
            {
                "match_id": [7],
                "team_id": [1],
                "player_id": [10],
                "batting_order": [1],
            }
        )
        batters, _ = build_impact_events(deliveries, roster)
        metrics, ranked, _, _, _, phases, weights = rank_players(
            batters,
            min_pressure_opportunities=1,
            min_pressure_matches=1,
            min_normal_opportunities=1,
            weight_mode="equal",
        )
        self.assertIn("clutch_uplift", metrics.columns)
        self.assertIn("pressure_composite", ranked.columns)
        self.assertIn("phase", phases.columns)
        self.assertTrue(weights["weight"].between(0, 1).all())


class MatchStateDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls.temp_dir.name)
        cls._write_datapack(cls.data_dir)
        cls.prepared, _, cls.quality, _ = load_and_prepare(cls.data_dir)

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    @staticmethod
    def _write_datapack(data_dir):
        matches = []
        ball_rows = []
        specs = [
            {
                "match_id": 1,
                "result": "Team B won by 1 run",
                "win_team_id": 2,
                "revised_target": np.nan,
                "first_schedule": 6,
                "second_schedule": 6,
                "first_runs": [1] * 6,
                "second_runs": [1] * 6,
            },
            {
                "match_id": 2,
                "result": "Team B won by DLS method",
                "win_team_id": 2,
                "revised_target": 4,
                "first_schedule": 6,
                "second_schedule": 6,
                "first_runs": [1] * 6,
                "second_runs": [1] * 4,
            },
            {
                "match_id": 3,
                "result": "Team B won by DLS method",
                "win_team_id": 2,
                "revised_target": np.nan,
                "first_schedule": 6,
                "second_schedule": 6,
                "first_runs": [1] * 6,
                "second_runs": [1] * 6,
            },
            {
                "match_id": 4,
                "result": "Team B won by DLS method",
                "win_team_id": 2,
                "revised_target": 4,
                "first_schedule": 6,
                "second_schedule": np.nan,
                "first_runs": [1] * 6,
                "second_runs": [1] * 4,
            },
            {
                "match_id": 5,
                "result": "No Result",
                "win_team_id": np.nan,
                "revised_target": np.nan,
                "first_schedule": np.nan,
                "second_schedule": np.nan,
                "first_runs": [2, 1],
                "second_runs": [],
            },
        ]
        roster_rows = []
        for spec in specs:
            match_id = spec["match_id"]
            matches.append(
                {
                    "match_id": match_id,
                    "comp_id": 10,
                    "match_date": "2023-01-01",
                    "first_inn_revised_overs": spec["first_schedule"],
                    "second_inn_revised_overs": spec["second_schedule"],
                    "revised_target": spec["revised_target"],
                    "home_team_id": 1,
                    "away_team_id": 2,
                    "win_team_id": spec["win_team_id"],
                    "result": spec["result"],
                }
            )
            for team_id in (1, 2):
                for batting_order in (1, 2):
                    roster_rows.append(
                        {
                            "match_id": match_id,
                            "team_id": team_id,
                            "player_id": team_id * 100 + batting_order,
                            "batting_order": batting_order,
                        }
                    )
            for innings, runs in (
                (1, spec["first_runs"]),
                (2, spec["second_runs"]),
            ):
                if not runs:
                    continue
                batting_team_id = innings
                bowling_team_id = 3 - innings
                for ball_number, batter_runs in enumerate(runs, start=1):
                    ball_rows.append(
                        {
                            "match_id": match_id,
                            "innings": innings,
                            "overid": (ball_number - 1) // 6 + 1,
                            "over_ball_num": (ball_number - 1) % 6 + 1,
                            "innings_ball_num": ball_number,
                            "batting_team_id": batting_team_id,
                            "bowling_team_id": bowling_team_id,
                            "striker_id": batting_team_id * 100 + 1,
                            "striker_name": f"Player {batting_team_id}A",
                            "non_striker_id": batting_team_id * 100 + 2,
                            "non_striker_name": f"Player {batting_team_id}B",
                            "bowler_id": bowling_team_id * 100 + 1,
                            "bowler_name": f"Player {bowling_team_id}A",
                            "batting_order": 1,
                            "runs": batter_runs,
                            "is_four": False,
                            "is_six": False,
                            "is_wicket": False,
                            "dismissal_type": np.nan,
                            "dismissed_player_id": np.nan,
                            "extras_type": np.nan,
                            "extras1_runs": 0,
                            "extras_2_runs": 0,
                        }
                    )
        pd.DataFrame(matches).to_csv(data_dir / "matches.csv", index=False)
        pd.DataFrame(ball_rows).to_csv(data_dir / "ball_by_ball.csv", index=False)
        pd.DataFrame(
            [{"comp_id": 10, "comp_name": "T20I Series 2023", "match_format": "T20I", "season": "2023"}]
        ).to_csv(data_dir / "competitions.csv", index=False)
        pd.DataFrame(roster_rows).to_csv(
            data_dir / "playing_eleven.csv", index=False
        )
        pd.DataFrame(
            [{"comp_id": 10, "team_id": team_id} for team_id in (1, 2)]
        ).to_csv(data_dir / "teams.csv", index=False)
        pd.DataFrame(
            [
                {"comp_id": 10, "team_id": team_id, "player_id": team_id * 100 + 1}
                for team_id in (1, 2)
            ]
        ).to_csv(
            data_dir / "competition_team_player_mapping.csv", index=False
        )

    def _match_rows(self, match_id, innings):
        return self.prepared.loc[
            self.prepared["match_id"].eq(match_id)
            & self.prepared["innings"].eq(innings)
        ]

    def test_normal_chase_uses_first_innings_total_plus_one(self):
        chase = self._match_rows(1, 2)
        self.assertEqual(chase["target_runs"].unique().tolist(), [7])
        self.assertTrue(chase["state_valid"].all())

    def test_dls_chase_uses_valid_revised_target(self):
        chase = self._match_rows(2, 2)
        self.assertEqual(chase["target_runs"].unique().tolist(), [4])
        self.assertTrue(chase["state_valid"].all())

    def test_dls_chase_with_missing_target_is_unknown_not_imputed(self):
        chase = self._match_rows(3, 2)
        self.assertTrue(chase["target_runs"].isna().all())
        self.assertFalse(chase["state_valid"].any())
        self.assertTrue(chase["runs_required_before"].isna().all())

    def test_dls_missing_schedule_is_not_defaulted_to_120(self):
        chase = self._match_rows(4, 2)
        self.assertTrue(chase["scheduled_balls"].isna().all())
        self.assertFalse(chase["state_valid"].any())
        self.assertIn("second_inn_revised_overs", " ".join(
            self.quality.loc[
                self.quality["check"].eq("dls_match_state_treatment"),
                "detail",
            ]
        ))

    def test_incomplete_no_result_innings_future_labels_are_excluded(self):
        first_innings = self._match_rows(5, 1)
        self.assertTrue(first_innings["state_valid"].all())
        self.assertEqual(
            first_innings["innings_completion_status"].unique().tolist(),
            ["incomplete"],
        )
        self.assertFalse(first_innings["future_run_label_valid"].any())
        self.assertFalse(_future_run_label_mask(first_innings).any())
        self.assertEqual(first_innings["batter_runs"].sum(), 3)
        self.assertEqual(first_innings["bowler_runs_conceded"].sum(), 3)
        completed_first_innings = self._match_rows(1, 1)
        self.assertTrue(_future_run_label_mask(completed_first_innings).all())


if __name__ == "__main__":
    unittest.main()
