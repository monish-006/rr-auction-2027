import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from .config import CV_FOLDS, MODEL_FEATURES, RANDOM_SEED, WIN_PROB_FEATURES


def _new_model(seed):
    return HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _new_classifier(seed):
    return HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=100,
        max_leaf_nodes=15,
        min_samples_leaf=100,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=seed,
    )


def _class_probabilities(model, features, class_count):
    predicted = model.predict_proba(features)
    probabilities = np.zeros((len(features), class_count), dtype=float)
    for column, label in enumerate(model.classes_):
        probabilities[:, int(label)] = predicted[:, column]
    return probabilities


def _post_delivery_features(data, features):
    post = features.copy()
    legal_balls = data["legal_balls_before"].to_numpy() + data[
        "is_legal_ball"
    ].to_numpy(dtype=int)
    score = data["current_score_before"].to_numpy() + data["delivery_runs"].to_numpy()
    wickets = (
        data["wickets_lost_before"].to_numpy()
        + data["is_wicket"].to_numpy(dtype=int)
    )
    post["legal_balls_before"] = legal_balls
    post["balls_remaining"] = np.maximum(
        data["scheduled_balls"].to_numpy() - legal_balls, 0
    )
    post["current_score_before"] = score
    post["wickets_in_hand_before"] = np.maximum(10 - wickets, 0)
    post["current_run_rate_before"] = np.divide(
        score * 6,
        legal_balls,
        out=np.zeros(len(data), dtype=float),
        where=legal_balls > 0,
    )
    is_chase = data["innings"].eq(2).to_numpy()
    runs_required = np.maximum(data["target_runs"].to_numpy() - score, 0)
    post["runs_required_before"] = np.where(is_chase, runs_required, 0)
    post["required_run_rate_before"] = np.divide(
        runs_required * 6,
        post["balls_remaining"].to_numpy(),
        out=np.zeros(len(data), dtype=float),
        where=is_chase & post["balls_remaining"].to_numpy().astype(bool),
    )
    max_overs = np.maximum(data["scheduled_balls"].to_numpy() // 6, 1)
    post_overs = np.minimum(legal_balls // 6 + 1, max_overs)
    post["overid"] = post_overs
    post["phase_code"] = np.where(post_overs <= 6, 0, np.where(post_overs <= 15, 1, 2))
    return post


def _clean_features(frame):
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0)


def _future_run_label_mask(data):
    return data["future_run_label_valid"].fillna(False).to_numpy(dtype=bool)


def _wpa_diagnostic_labels(data, state_valid):
    winner_labels = data["batting_team_won"].to_numpy(dtype=float)
    attribution_status = np.select(
        [~state_valid, ~np.isfinite(winner_labels)],
        [
            "excluded_invalid_match_state",
            "excluded_no_decisive_result",
        ],
        default="team_state_proxy_not_individually_attributed",
    )
    extras = (
        data["extras_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )
    dismissal = (
        data["dismissal_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .fillna("")
    )
    event_context = np.select(
        [
            extras.eq("wide"),
            extras.isin(["overstep", "other"]),
            extras.eq("byes"),
            extras.eq("legbyes"),
            extras.eq("penalty"),
            data["is_wicket"] & dismissal.eq("runout"),
            data["is_wicket"] & data["dismissed_player_id"].isna(),
            data["is_wicket"],
        ],
        [
            "wide",
            "no_ball",
            "bye",
            "leg_bye",
            "penalty",
            "run_out",
            "dismissal_missing_player_id",
            "other_dismissal",
        ],
        default="ordinary_delivery",
    )
    return attribution_status, event_context


def _regression_diagnostics(name, target, prediction, mask, data, folds):
    observed = target[mask]
    predicted = prediction[mask]
    brier = logloss = auc = np.nan
    if np.isin(observed, [0, 1]).all():
        clipped = np.clip(predicted, 1e-8, 1 - 1e-8)
        brier = mean_squared_error(observed, predicted)
        logloss = log_loss(observed, clipped, labels=[0, 1])
        try:
            auc = roc_auc_score(observed, predicted)
        except ValueError:
            auc = np.nan
    return {
        "target": name,
        "observations": len(observed),
        "matches": data.loc[mask, "match_id"].nunique(),
        "folds": folds,
        "mae": mean_absolute_error(observed, predicted),
        "rmse": mean_squared_error(observed, predicted) ** 0.5,
        "r2": r2_score(observed, predicted),
        "brier_score": brier,
        "log_loss": logloss,
        "accuracy": np.nan,
        "roc_auc": auc,
        "split_unit": "match_id",
        "train_validation_match_overlap": 0,
    }


def _classification_diagnostics(
    name, target, probabilities, mask, data, folds, class_count
):
    observed = target[mask]
    predicted = probabilities[mask]
    one_hot = np.eye(class_count)[observed.astype(int)]
    try:
        auc = roc_auc_score(
            observed,
            predicted,
            labels=np.arange(class_count),
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        auc = np.nan
    return {
        "target": name,
        "observations": len(observed),
        "matches": data.loc[mask, "match_id"].nunique(),
        "folds": folds,
        "mae": np.nan,
        "rmse": np.nan,
        "r2": np.nan,
        "brier_score": np.mean(np.sum((one_hot - predicted) ** 2, axis=1)),
        "log_loss": log_loss(
            observed, predicted, labels=np.arange(class_count)
        ),
        "accuracy": accuracy_score(observed, predicted.argmax(axis=1)),
        "roc_auc": auc,
        "split_unit": "match_id",
        "train_validation_match_overlap": 0,
    }


def fit_oof_expectations(deliveries):
    """Create match-grouped expectations and diagnostics for all impact features."""
    data = deliveries.reset_index(drop=True)
    if data["match_id"].nunique() < 2:
        raise ValueError("At least two matches are required for grouped validation.")

    features = data[MODEL_FEATURES].copy()
    features["batting_order"] = features["batting_order"].fillna(11)
    features = _clean_features(features)
    win_features = _clean_features(data[WIN_PROB_FEATURES].copy())
    state_valid = data["state_valid"].fillna(False).to_numpy(dtype=bool)
    targets = {
        "expected_batter_runs": (
            data["batter_runs"].to_numpy(dtype=float),
            data["is_batter_opportunity"].to_numpy(dtype=bool) & state_valid,
        ),
        "expected_bowler_runs": (
            data["bowler_runs_conceded"].to_numpy(dtype=float),
            data["is_bowler_opportunity"].to_numpy(dtype=bool) & state_valid,
        ),
        "expected_future_team_runs": (
            data["future_team_runs"].to_numpy(dtype=float),
            _future_run_label_mask(data),
        ),
        "expected_dot_probability": (
            data["is_dot_ball"].to_numpy(dtype=float),
            data["is_bowler_opportunity"].to_numpy(dtype=bool) & state_valid,
        ),
        "expected_bowler_boundary_probability": (
            data["is_boundary"].to_numpy(dtype=float),
            data["is_bowler_opportunity"].to_numpy(dtype=bool) & state_valid,
        ),
    }
    categorical_targets = {
        "batter_boundary": (
            np.where(data["is_six"], 2, np.where(data["is_four"], 1, 0)).astype(int),
            data["is_batter_opportunity"].to_numpy(dtype=bool) & state_valid,
            3,
        ),
        "dismissal_type": (
            np.where(
                data["is_credited_wicket"],
                2,
                np.where(data["is_wicket"], 1, 0),
            ).astype(int),
            data["is_bowler_opportunity"].to_numpy(dtype=bool) & state_valid,
            3,
        ),
    }
    predictions = {
        name: np.full(len(data), np.nan, dtype=float) for name in targets
    }
    categorical_predictions = {
        name: np.full((len(data), class_count), np.nan, dtype=float)
        for name, (_, _, class_count) in categorical_targets.items()
    }
    wicket_values = np.full(len(data), np.nan, dtype=float)
    par_totals = np.full(len(data), np.nan, dtype=float)
    fold_ids = np.full(len(data), -1, dtype=int)
    wp_before = np.full(len(data), np.nan, dtype=float)
    wp_after = np.full(len(data), np.nan, dtype=float)

    n_splits = min(CV_FOLDS, data["match_id"].nunique())
    splitter = GroupKFold(n_splits=n_splits)
    post_win_features = _post_delivery_features(data, win_features)
    win_labels = data["batting_team_won"].to_numpy(dtype=float)
    known_winner = np.isfinite(win_labels) & state_valid
    valid_state_counts = (
        data.assign(_state_valid=state_valid)
        .groupby("match_id")["_state_valid"]
        .transform("sum")
        .to_numpy(dtype=float)
    )
    for fold, (train_idx, valid_idx) in enumerate(
        splitter.split(features, groups=data["match_id"]), start=1
    ):
        train_matches = set(data.iloc[train_idx]["match_id"].unique())
        valid_matches = set(data.iloc[valid_idx]["match_id"].unique())
        if train_matches.intersection(valid_matches):
            raise AssertionError("Match leakage detected between train and validation.")

        for target_name, (target, target_mask) in targets.items():
            target_train_idx = train_idx[target_mask[train_idx]]
            prediction_valid_idx = valid_idx[state_valid[valid_idx]]
            target_valid_idx = prediction_valid_idx[
                target_mask[prediction_valid_idx]
            ]
            if not len(target_train_idx) or not len(target_valid_idx):
                raise ValueError(f"No train or validation rows for {target_name}.")
            model = _new_model(RANDOM_SEED + fold)
            model.fit(features.iloc[target_train_idx], target[target_train_idx])
            predicted = model.predict(features.iloc[prediction_valid_idx])
            if target_name in {
                "expected_dot_probability",
                "expected_bowler_boundary_probability",
            }:
                predicted = np.clip(predicted, 0, 1)
            else:
                predicted = np.maximum(predicted, 0)
            predictions[target_name][prediction_valid_idx] = predicted

            if target_name == "expected_future_team_runs":
                counterfactual = features.iloc[prediction_valid_idx].copy()
                counterfactual["wickets_in_hand_before"] = np.maximum(
                    counterfactual["wickets_in_hand_before"] - 1, 0
                )
                counterfactual_runs = np.maximum(
                    model.predict(counterfactual), 0
                )
                wicket_values[prediction_valid_idx] = np.maximum(
                    predictions[target_name][prediction_valid_idx] - counterfactual_runs,
                    0,
                )

        for target_name, (target, target_mask, class_count) in categorical_targets.items():
            target_train_idx = train_idx[target_mask[train_idx]]
            prediction_valid_idx = valid_idx[state_valid[valid_idx]]
            target_valid_idx = prediction_valid_idx[
                target_mask[prediction_valid_idx]
            ]
            if not len(target_train_idx) or not len(target_valid_idx):
                raise ValueError(f"No train or validation rows for {target_name}.")
            model = _new_classifier(RANDOM_SEED + 100 + fold)
            model.fit(features.iloc[target_train_idx], target[target_train_idx])
            categorical_predictions[target_name][target_valid_idx] = (
                _class_probabilities(
                    model, features.iloc[target_valid_idx], class_count
                )
            )

        known_train_idx = train_idx[known_winner[train_idx]]
        if len(np.unique(win_labels[known_train_idx])) < 2:
            raise ValueError("Win-probability training fold must contain both outcomes.")
        training_states = pd.concat(
            [
                win_features.iloc[known_train_idx],
                post_win_features.iloc[known_train_idx],
            ],
            ignore_index=True,
        )
        training_labels = np.tile(win_labels[known_train_idx].astype(int), 2)
        state_weights = np.tile(
            1 / (2 * valid_state_counts[known_train_idx]), 2
        )
        win_model = _new_classifier(RANDOM_SEED + 200 + fold)
        win_model.fit(
            training_states,
            training_labels,
            sample_weight=state_weights,
        )
        prediction_valid_idx = valid_idx[state_valid[valid_idx]]
        valid_states = pd.concat(
            [
                win_features.iloc[prediction_valid_idx],
                post_win_features.iloc[prediction_valid_idx],
            ],
            ignore_index=True,
        )
        valid_probabilities = _class_probabilities(win_model, valid_states, 2)[:, 1]
        state_count = len(prediction_valid_idx)
        wp_before[prediction_valid_idx] = valid_probabilities[:state_count]
        wp_after[prediction_valid_idx] = valid_probabilities[state_count:]

        train_innings = data.iloc[train_idx].loc[
            data.iloc[train_idx]["innings"].eq(1)
            & data.iloc[train_idx]["par_model_eligible"],
            ["match_id", "competition_format", "innings_total"],
        ].drop_duplicates("match_id")
        par_by_format = train_innings.groupby("competition_format")[
            "innings_total"
        ].mean()
        valid_formats = data.iloc[valid_idx]["competition_format"]
        par_totals[valid_idx] = valid_formats.map(par_by_format).to_numpy(dtype=float)
        fold_ids[valid_idx] = fold

    if (fold_ids < 1).any() or any(
        np.isnan(predictions[name][mask]).any()
        for name, (_, mask) in targets.items()
    ) or any(
        np.isnan(categorical_predictions[name][mask]).any()
        for name, (_, mask, _) in categorical_targets.items()
    ):
        raise AssertionError("Out-of-fold predictions are incomplete.")

    output = data.copy()
    for name, values in predictions.items():
        output[name] = values
    boundary_probabilities = categorical_predictions["batter_boundary"]
    output["expected_batter_four_probability"] = boundary_probabilities[:, 1]
    output["expected_batter_six_probability"] = boundary_probabilities[:, 2]
    output["expected_batter_boundary_probability"] = (
        boundary_probabilities[:, 1] + boundary_probabilities[:, 2]
    )
    dismissal_probabilities = categorical_predictions["dismissal_type"]
    output["expected_any_wicket_probability"] = (
        dismissal_probabilities[:, 1] + dismissal_probabilities[:, 2]
    )
    output["expected_credited_wicket_probability"] = dismissal_probabilities[:, 2]
    output["wicket_run_value"] = wicket_values
    output["training_par_total"] = par_totals
    output["validation_fold"] = fold_ids
    output["batting_win_probability_before"] = wp_before
    output["batting_win_probability_after"] = wp_after
    output["win_probability_added"] = wp_after - wp_before
    (
        output["wpa_attribution_status"],
        output["wpa_proxy_event_context"],
    ) = _wpa_diagnostic_labels(data, state_valid)

    validation_rows = []
    for target_name, (target, mask) in targets.items():
        validation_rows.append(
            _regression_diagnostics(
                target_name, target, predictions[target_name], mask, data, n_splits
            )
        )
    for target_name, (target, mask, class_count) in categorical_targets.items():
        validation_rows.append(
            _classification_diagnostics(
                target_name,
                target,
                categorical_predictions[target_name],
                mask,
                data,
                n_splits,
                class_count,
            )
        )

    win_validation_mask = known_winner
    if win_validation_mask.sum() and len(np.unique(win_labels[win_validation_mask])) == 2:
        win_probabilities = np.clip(wp_before[win_validation_mask], 1e-8, 1 - 1e-8)
        win_observed = win_labels[win_validation_mask].astype(int)
        win_sample_weights = 1 / valid_state_counts[win_validation_mask]
        try:
            win_auc = roc_auc_score(
                win_observed,
                win_probabilities,
                sample_weight=win_sample_weights,
            )
        except ValueError:
            win_auc = np.nan
        validation_rows.append(
            {
                "target": "batting_team_win_probability",
                "observations": int(win_validation_mask.sum()),
                "matches": data.loc[win_validation_mask, "match_id"].nunique(),
                "folds": n_splits,
                "mae": np.nan,
                "rmse": np.nan,
                "r2": np.nan,
                "brier_score": brier_score_loss(
                    win_observed,
                    win_probabilities,
                    sample_weight=win_sample_weights,
                ),
                "log_loss": log_loss(
                    win_observed,
                    win_probabilities,
                    sample_weight=win_sample_weights,
                    labels=[0, 1],
                ),
                "accuracy": np.nan,
                "roc_auc": win_auc,
                "split_unit": "match_id",
                "train_validation_match_overlap": 0,
            }
        )
    return output, pd.DataFrame(validation_rows)
