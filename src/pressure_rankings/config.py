from pathlib import Path

DEFAULT_DATA_DIR = Path("/Users/monish_006/Documents/RR/Datapack")
DEFAULT_OUTPUT_DIR = Path("output")
RANDOM_SEED = 20261116
CV_FOLDS = 5

PRESSURE_CUTOFF = 0.55
MIN_PRESSURE_OPPORTUNITIES = 30
MIN_PRESSURE_MATCHES = 8
MIN_NORMAL_OPPORTUNITIES = 30
SHRINKAGE_OPPORTUNITIES = 50

BASELINE_WEIGHTS = {
    "first": {"run_rate": 0.50, "resource": 0.30, "stage": 0.20},
    "second": {
        "run_rate": 0.35,
        "resource": 0.20,
        "leverage": 0.25,
        "stage": 0.20,
    },
}

PRESSURE_WEIGHT_SCENARIOS = {
    "baseline": BASELINE_WEIGHTS,
    "equal": {
        "first": {"run_rate": 1 / 3, "resource": 1 / 3, "stage": 1 / 3},
        "second": {
            "run_rate": 0.25,
            "resource": 0.25,
            "leverage": 0.25,
            "stage": 0.25,
        },
    },
    "rate_focused": {
        "first": {"run_rate": 0.60, "resource": 0.25, "stage": 0.15},
        "second": {
            "run_rate": 0.50,
            "resource": 0.15,
            "leverage": 0.20,
            "stage": 0.15,
        },
    },
    "resource_focused": {
        "first": {"run_rate": 0.35, "resource": 0.40, "stage": 0.25},
        "second": {
            "run_rate": 0.25,
            "resource": 0.35,
            "leverage": 0.15,
            "stage": 0.25,
        },
    },
}

SENSITIVITY_CUTOFFS = (0.50, 0.65)
SENSITIVITY_ELIGIBILITY = ((20, 5), (50, 12))

DELIVERY_KEY = ["match_id", "innings", "overid", "over_ball_num"]
MODEL_FEATURES = [
    "competition_is_ipl",
    "innings",
    "is_second_innings",
    "overid",
    "phase_code",
    "scheduled_balls",
    "legal_balls_before",
    "balls_remaining",
    "wickets_in_hand_before",
    "current_score_before",
    "current_run_rate_before",
    "target_runs",
    "runs_required_before",
    "required_run_rate_before",
    "batting_order",
]
WIN_PROB_FEATURES = [
    "competition_is_ipl",
    "innings",
    "is_second_innings",
    "overid",
    "phase_code",
    "scheduled_balls",
    "legal_balls_before",
    "balls_remaining",
    "wickets_in_hand_before",
    "current_score_before",
    "current_run_rate_before",
    "target_runs",
    "runs_required_before",
    "required_run_rate_before",
]

CREDITED_DISMISSALS = {
    "caught",
    "bowled",
    "lbw",
    "stump",
    "hit wicket",
}
