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
from src.pressure_rankings.impact import (
    build_player_events as build_impact_events,
    rank_players,
)


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
