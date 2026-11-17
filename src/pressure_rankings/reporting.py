from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .config import (
    MIN_NORMAL_OPPORTUNITIES,
    MIN_PRESSURE_MATCHES,
    MIN_PRESSURE_OPPORTUNITIES,
    PRESSURE_CUTOFF,
)


PRESSURE_OUTPUT_COLUMNS = [
    "delivery_id",
    "match_id",
    "comp_id",
    "comp_name",
    "competition_format",
    "season",
    "match_date",
    "win_team_id",
    "result",
    "batting_team_won",
    "innings",
    "overid",
    "over_ball_num",
    "innings_ball_num",
    "phase",
    "batting_team_id",
    "bowling_team_id",
    "extras_type",
    "extras1_runs",
    "extras_2_runs",
    "striker_id",
    "striker_name",
    "non_striker_id",
    "non_striker_name",
    "bowler_id",
    "bowler_name",
    "batting_order",
    "is_legal_ball",
    "is_dls_match",
    "state_valid",
    "state_invalid_reason",
    "target_state_valid",
    "schedule_state_valid",
    "scheduled_balls",
    "innings_completion_status",
    "innings_completed",
    "future_run_label_valid",
    "is_batter_opportunity",
    "is_bowler_opportunity",
    "is_four",
    "is_six",
    "is_boundary",
    "is_dot_ball",
    "legal_balls_before",
    "balls_remaining",
    "current_score_before",
    "current_run_rate_before",
    "wickets_lost_before",
    "wickets_in_hand_before",
    "target_runs",
    "runs_required_before",
    "required_run_rate_before",
    "batter_runs",
    "delivery_runs",
    "bowler_runs_conceded",
    "is_wicket",
    "is_credited_wicket",
    "dismissal_type",
    "dismissed_player_id",
    "expected_batter_runs",
    "expected_batter_four_probability",
    "expected_batter_six_probability",
    "expected_batter_boundary_probability",
    "expected_any_wicket_probability",
    "expected_bowler_runs",
    "expected_dot_probability",
    "expected_bowler_boundary_probability",
    "expected_credited_wicket_probability",
    "expected_future_team_runs",
    "future_team_runs",
    "wicket_run_value",
    "batting_win_probability_before",
    "batting_win_probability_after",
    "win_probability_added",
    "wpa_attribution_status",
    "wpa_proxy_event_context",
    "training_par_total",
    "pressure_run_rate",
    "pressure_resource",
    "pressure_leverage",
    "pressure_stage",
    "pressure_index",
    "is_pressure",
    "validation_fold",
]


def _safe_csv(frame, path):
    frame.to_csv(path, index=False)


def _eligible_format_rows(format_metrics):
    return format_metrics.loc[
        format_metrics["pressure_opportunities"].ge(MIN_PRESSURE_OPPORTUNITIES)
        & format_metrics["pressure_matches"].ge(MIN_PRESSURE_MATCHES)
        & format_metrics["normal_opportunities"].ge(MIN_NORMAL_OPPORTUNITIES)
    ]


def _format_comparison(batter_formats, bowler_formats):
    rows = []
    for player_type, frame in (
        ("batter", batter_formats),
        ("bowler", bowler_formats),
    ):
        eligible = _eligible_format_rows(frame)
        for competition_format, group in eligible.groupby("competition_format"):
            rows.append(
                {
                    "player_type": player_type,
                    "competition_format": competition_format,
                    "eligible_players": group["player_id"].nunique(),
                    "pressure_opportunities": int(group["pressure_opportunities"].sum()),
                    "normal_opportunities": int(group["normal_opportunities"].sum()),
                    "mean_pressure_per100": group["pressure_per100"].mean(),
                    "mean_normal_per100": group["normal_per100"].mean(),
                    "mean_clutch_uplift_per100": group["clutch_uplift_per100"].mean(),
                    "median_pressure_matches": group["pressure_matches"].median(),
                }
            )
    return pd.DataFrame(rows)


def _plot_rankings(batters, bowlers, path):
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    for axis, data, title in (
        (axes[0], batters, "Pressure-adjusted batter ranking"),
        (axes[1], bowlers, "Pressure-adjusted bowler ranking"),
    ):
        if data.empty:
            axis.text(0.5, 0.5, "No players meet eligibility thresholds", ha="center")
            axis.set_axis_off()
            continue
        ordered = data.sort_values("ranking_score", ascending=True)
        axis.barh(ordered["player_name"], ordered["ranking_score"], color="#176B87")
        axis.set_title(title)
        axis.set_xlabel("Reliability-adjusted normalized score")
        axis.set_ylabel("")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_pressure_vs_normal(batter_formats, bowler_formats, path):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, data, title in (
        (axes[0], _eligible_format_rows(batter_formats), "Batters"),
        (axes[1], _eligible_format_rows(bowler_formats), "Bowlers"),
    ):
        if data.empty:
            axis.text(0.5, 0.5, "No eligible players", ha="center")
            axis.set_axis_off()
            continue
        sns.scatterplot(
            data=data,
            x="normal_per100",
            y="pressure_per100",
            hue="competition_format",
            alpha=0.75,
            s=55,
            ax=axis,
        )
        low = min(data["normal_per100"].min(), data["pressure_per100"].min())
        high = max(data["normal_per100"].max(), data["pressure_per100"].max())
        axis.plot([low, high], [low, high], linestyle="--", color="gray", linewidth=1)
        axis.set_title(title)
        axis.set_xlabel("Normal-condition contribution per 100 opportunities (runs)")
        axis.set_ylabel("Pressure contribution per 100 opportunities (runs)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_reliability(overall_batters, overall_bowlers, path):
    figure, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    for axis, data, title in (
        (axes[0], overall_batters, "Batters"),
        (axes[1], overall_bowlers, "Bowlers"),
    ):
        eligible = data.loc[data["eligible"]]
        if eligible.empty:
            axis.text(0.5, 0.5, "No eligible players", ha="center")
            axis.set_axis_off()
            continue
        sns.scatterplot(
            data=eligible,
            x="pressure_opportunities",
            y="ranking_score",
            size="pressure_matches",
            sizes=(30, 180),
            alpha=0.75,
            ax=axis,
        )
        axis.set_title(title)
        axis.set_xlabel("Pressure opportunities")
        axis.set_ylabel("Reliability-adjusted normalized score")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_pressure_heatmap(deliveries, path):
    data = deliveries.copy()
    data["innings_context"] = data["competition_format"] + " innings " + data[
        "innings"
    ].astype(str)
    heat = data.pivot_table(
        index="innings_context",
        columns="phase",
        values="pressure_index",
        aggfunc="mean",
        observed=True,
    )
    heat = heat.reindex(
        columns=["Powerplay (1-6)", "Middle (7-15)", "Slog (16-20)"]
    )
    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    sns.heatmap(
        heat,
        annot=True,
        fmt=".2f",
        cmap="mako",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        ax=axis,
    )
    axis.set_title("Mean pre-delivery pressure by format, innings and phase")
    axis.set_xlabel("Innings phase")
    axis.set_ylabel("")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _markdown_table(frame, columns, limit=10):
    if frame.empty:
        return "_No eligible records._"
    view = frame.loc[:, columns].head(limit).copy()
    headers = [str(column) for column in view.columns]

    def format_value(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in view.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(format_value(value) for value in row) + " |")
    return "\n".join(lines)


def _write_methodology_report(
    path,
    prepared,
    quality,
    validation,
    batter_top,
    bowler_top,
    format_comparison,
    stability,
    sensitivity_rankings,
    component_weights,
    batter_phase_leaderboards,
    bowler_phase_leaderboards,
):
    format_counts = (
        prepared.groupby("competition_format")["match_id"].nunique().to_dict()
    )
    rows = int(len(prepared))
    high_pressure_rows = int(prepared["is_pressure"].sum())
    missing_matches = quality.loc[
        quality["check"].eq("selected_matches_without_deliveries"), "value"
    ].iloc[0]
    duplicate_keys = quality.loc[
        quality["check"].eq("duplicate_delivery_keys"), "value"
    ].iloc[0]
    unmatched_dismissals = quality.loc[
        quality["check"].eq("wickets_with_missing_dismissed_player_id"), "value"
    ].iloc[0]
    attribution_mismatches = quality.loc[
        quality["check"].eq("dismissed_player_not_striker_or_non_striker"), "value"
    ].iloc[0]
    def quality_value(check):
        values = quality.loc[quality["check"].eq(check), "value"]
        return values.iloc[-1] if not values.empty else 0

    dls_treatments = quality.loc[
        quality["check"].eq("dls_match_state_treatment"),
        ["value", "detail"],
    ].rename(columns={"value": "match_id", "detail": "missing_fields_and_treatment"})
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    phase_tops = pd.concat(
        [batter_phase_leaderboards, bowler_phase_leaderboards],
        ignore_index=True,
        sort=False,
    ).sort_values(["player_type", "phase", "rank"], kind="stable")
    report = f"""# IPL and T20I Pressure Player Rankings (2022 onward)

Generated: {generated}

## Data scope and audit

The pipeline inspected all six supplied CSV files and their columns, missing values, keys and joins. It selected IPL competitions named `IPL YYYY` and T20I competitions with seasons beginning in 2022 or later; other `T20` competitions were excluded. The delivery data contains **{rows:,}** regular-innings deliveries across **{format_counts.get('IPL', 0)} IPL** and **{format_counts.get('T20I', 0)} T20I** matches with deliveries. Super Overs were excluded because their six-ball format is not comparable to the standard innings model. {missing_matches} selected match metadata rows have no ball-by-ball records. Duplicate delivery keys: {duplicate_keys}. Wickets with no dismissed-player ID: {unmatched_dismissals}; dismissal IDs not matching either listed batter on that delivery: {attribution_mismatches}.

### Data quality checks

{_markdown_table(quality, ['check', 'value', 'detail'], limit=100)}

`runs` is treated as striker runs. Team totals add both extras columns. Revised-over metadata values are multiples of six and are interpreted as ball limits. A missing schedule defaults to 120 balls only for matches not marked DLS; DLS innings with missing revised schedules have unknown balls remaining and are excluded from state-dependent analysis. A DLS chase without a revised target has an unknown target and is not assigned first-innings total + 1. The feed labels many no-balls as `Overstep`; `other` extras are treated as no-balls because {int(quality.loc[quality.check.eq('other_extra_rows_in_overs_with_more_than_six_balls'), 'value'].iloc[0])} of those rows occur in overs with more than six recorded balls (out of {int(quality.loc[quality.check.eq('other_extra_rows'), 'value'].iloc[0])} `other` rows). Wide and no-ball primary extras are charged to the bowler; byes, leg-byes and penalty extras are not. Secondary extras on no-balls and two rows with missing extras labels are excluded from bowler-charged extras because their attribution is not explicit in the source.

### DLS state metadata

Selected DLS matches: **{int(quality_value('selected_dls_matches'))}**; missing revised targets: **{int(quality_value('dls_matches_missing_revised_target'))}**. Match-specific missing fields and treatment:

{_markdown_table(dls_treatments, ['match_id', 'missing_fields_and_treatment'], limit=100)}

### State and label exclusions

- Deliveries excluded from state-dependent analysis: **{int(quality_value('deliveries_excluded_from_state_dependent_analysis'))}**.
- Confirmed incomplete innings: **{int(quality_value('incomplete_innings'))}**; completion unknown: **{int(quality_value('innings_completion_unknown'))}**.
- Future-run labels excluded: **{int(quality_value('future_run_labels_excluded'))}**; first-innings par observations excluded: **{int(quality_value('par_model_observations_excluded'))}**.
- No-result WPA events excluded from individual rankings: **{int(quality_value('no_result_wpa_events_excluded_from_individual_rankings'))}**.
- Ambiguous individual WPA attributions not assigned to players: **{int(quality_value('ambiguous_individual_wpa_attributions'))}**.
- Valid delivery-level run outcomes retained: **{int(quality_value('valid_delivery_level_run_outcomes_retained'))}**.

An innings is marked complete only when the feed supports all-out, the known schedule is exhausted, a known chase target is reached, or the chasing team is the recorded winner. No-result/abandoned innings without such evidence are incomplete; other unresolved cases are marked unknown. Future-run labels derived from an innings total are used for training only when completion and state metadata are valid. Individual-delivery run outcomes are not excluded solely because their match had no result.

## Pre-delivery state and pressure index

Every feature is computed before the current delivery: innings, over/phase, scheduled and remaining legal balls, score, current run rate, wickets lost and in hand, target, runs required and required run rate. At the baseline 0.55 cutoff, **{high_pressure_rows:,}** valid-state deliveries are labeled high pressure. Invalid DLS target/schedule rows have `pressure_index=NaN`, remain visible in the delivery output, and do not enter pressure/normal player opportunities or any state-dependent model. Wides, `Overstep`, `other` and penalty rows do not consume a legal ball. Wides and administrative penalty rows are excluded from batter opportunities; no-balls count as batter opportunities. First-innings targets are not available before the innings ends, so first-innings pressure uses a training-fold mean of completed first innings; second-innings pressure uses a known target.

All components are bounded to [0, 1]:

- **Run-rate pressure, first innings:** `clip((training par - projected innings total) / (0.20 × training par), 0, 1)`. The projection is current score plus expected remaining team runs.
- **Run-rate pressure, chase:** `clip((runs required / expected remaining runs - 1) / 0.50, 0, 1)`.
- **Resource pressure:** `wickets lost / 9`.
- **Stage:** legal balls elapsed / scheduled legal balls.
- **Match leverage, chase:** `clip(1 - abs(runs required - expected remaining runs) / max(runs required, expected remaining runs, 1), 0, 1)`. It is unavailable in the first innings and is not imputed.

Baseline first-innings weights are run rate **0.50**, resources **0.30**, stage **0.20**. Chase weights are run rate **0.35**, resources **0.20**, leverage **0.25**, stage **0.20**. The first-innings weights sum to one without an unobservable match-leverage term. This is a transparent starting specification, not a learned causal truth. The pipeline tests equal, rate-focused and resource-focused weights, as well as pressure cutoffs of 0.50, 0.55 and 0.65.

## Expected performance and leakage controls

Match-grouped out-of-fold `HistGradientBoosting` models estimate striker runs, bowler-charged runs, future innings runs, batter four/six probabilities, dismissal probabilities, bowler dot-ball probability, and bowler boundary probability from valid pre-delivery state. Valid individual batter/bowler outcome labels remain trainable in no-result matches. Future-innings-run models train only on completed innings; incomplete or unknown innings totals are not treated as final labels. First-innings par is estimated only from completed first innings in the training fold. A separate state model estimates batting-team win probability from valid match state and trains only on matches with a decisive winner. Five-fold `GroupKFold` validation keeps every match out of its own training fold. `model_validation.csv` reports run/rate errors plus classification Brier/log-loss/AUC where applicable.

The delivery-level team-state WPA proxy is `estimated batting-team win probability after - before`. Before/after probabilities and the proxy are retained for valid-state diagnostics. No-result WPA events are excluded from individual ranking metrics. For decisive matches the proxy is still a team-state change, not identifiable individual causal value: wides, no-balls, byes, leg-byes, penalties, run-outs, dismissals and ordinary deliveries are all explicitly labeled by delivery context but are not credited to a batter or bowler. WPA is excluded from the PCA composite.

## Player contributions and reliability score

- **Batting diagnostics:** runs above expectation and strike-rate impact are both reported; with the same batter-opportunity denominator, strike-rate impact is the runs-above-expectation rate expressed per 100 balls, so it is not counted a second time. Four/six counts and rates are compared with state-based expected probabilities. Wicket preservation is expected dismissal probability times wicket run value, minus actual dismissal cost. Actual dismissal cost is attributed by `dismissed_player_id`, including non-striker run-outs.
- **Bowling diagnostics:** runs saved are expected bowler-charged runs minus actual; dot impact is actual minus expected dot percentage; boundary prevention is expected minus actual boundary percentage. Credited wicket impact is `(actual credited wicket - expected credited wicket probability) × wicket run value`. Only caught, bowled, LBW, stumped (`Stump`) and hit-wicket dismissals receive bowler credit; run-outs, retire-outs, and obstructing-the-field dismissals do not.
- **Scoring/control dimensions:** correlated indicators are not added as separate bonuses. Standardized runs-above-expectation and boundary-rate residuals form one batting scoring dimension. Standardized runs saved, dot-rate delta, and boundary-prevention measures form one bowling control dimension. The first principal-component loading magnitudes from the observed player-context data, normalized to sum to one, provide the component weights.
- **Final composite:** scoring/control and wicket dimensions are standardized within format, role, and phase. A second first-component PCA produces the data-driven overall weights. The primary ranking uses these PCA weights; an equal-component alternative is tested for stability. `component_weights.csv` records the learned weights and variance explained. WPA is a delivery-level diagnostic only, not a player-level component. This unsupervised weighting uses observed covariance, not match-result labels.
- **Pressure and normal scores:** the same learned weights are applied separately to pressure and normal-condition performance. Clutch uplift is `pressure composite - normal composite`; it is reported directly and is not added as an extra bonus to the pressure ranking.
- The pressure composite is multiplied by sample shrinkage `n/(n+50) × min(1, pressure_matches/8)`, by match consistency `0.5 + 0.5 × positive-pressure-match share`, and by season stability `0.75 + 0.25 × season stability`. Season stability is `1/(1 + SD(season rates)/(abs(mean season rate)+1))`. The overall player score is the square-root-opportunity-weighted mean of format/role/phase scores, with context opportunities capped at 400.

Baseline eligibility requires at least **{MIN_PRESSURE_OPPORTUNITIES} pressure deliveries**, **{MIN_PRESSURE_MATCHES} pressure matches**, and **{MIN_NORMAL_OPPORTUNITIES} normal deliveries**. DNBs are not assigned artificial zeroes; they have no delivery opportunities and are ineligible. Rankings are relative among eligible players, not claims that a player with a negative score performed above expectation.

## Results

### Top batters

{_markdown_table(batter_top, ['rank', 'player_name', 'ranking_score', 'pressure_runs_above_expected_per100', 'pressure_strike_rate_impact', 'pressure_boundary_percentage_delta_per100', 'pressure_wicket_preservation_per100', 'pressure_composite', 'normal_composite', 'clutch_uplift', 'pressure_opportunities', 'pressure_matches'])}

### Top bowlers

{_markdown_table(bowler_top, ['rank', 'player_name', 'ranking_score', 'pressure_runs_saved_per100', 'pressure_dot_percentage_delta_per100', 'pressure_boundary_prevention_per100', 'pressure_wicket_impact_per100', 'pressure_composite', 'normal_composite', 'clutch_uplift', 'pressure_opportunities', 'pressure_matches'])}

### Phase leaderboards

The separate phase table uses the same format/role normalization, learned component weights, and reliability adjustment. It applies the baseline evidence thresholds within each phase; players without enough pressure and normal opportunities are omitted.

{_markdown_table(phase_tops, ['player_type', 'phase', 'rank', 'player_name', 'ranking_score', 'pressure_composite', 'normal_composite', 'clutch_uplift', 'pressure_opportunities', 'pressure_matches'], limit=30)}

### Learned component weights

{_markdown_table(component_weights, ['player_type', 'level', 'component', 'weight', 'variance_explained', 'weight_method'], limit=30)}

### IPL vs T20I

{_markdown_table(format_comparison, ['player_type', 'competition_format', 'eligible_players', 'mean_pressure_per100', 'mean_normal_per100', 'mean_clutch_uplift_per100'])}

### Out-of-fold model diagnostics

{_markdown_table(validation, ['target', 'observations', 'matches', 'folds', 'mae', 'rmse', 'r2', 'brier_score', 'log_loss', 'roc_auc', 'train_validation_match_overlap'], limit=20)}

### Ranking stability

{_markdown_table(stability, ['player_type', 'scenario', 'eligible_players', 'baseline_top10_overlap', 'top10_jaccard', 'spearman_rank_correlation'], limit=30)}

## Sensitivity

`sensitivity_rankings.csv` contains all eligible ranked players under alternative pressure weights, pressure cutoffs, eligibility thresholds (20/5 and 50/12, against baseline 30/8), and equal instead of PCA component weights. `ranking_stability.csv` compares each scenario's top-10 overlap/Jaccard and eligible-player rank correlation with the baseline. Cutoff and minimum-sample changes alter which deliveries and players qualify; treat these as robustness checks, not independent validation.

## Limitations

The source contains no published ball-by-ball win probabilities, venue conditions, batter intent, or explicit no-ball subtype for `other`. The fitted win-probability model is an estimate based on match winners and may be imperfectly calibrated. Expected models estimate league-average context rather than player-specific ability, and delivery outcomes are noisy. PCA weights are data-driven but unsupervised; equal-weight sensitivity results should be reviewed before relying on close ranks. Pressure weights, pressure scales, cutoff, role bins, and eligibility thresholds remain explicit analyst choices. IPL/T20I comparisons use format- and role-standardized scores, while raw run contributions remain in each format's own units. Results are descriptive, not causal.

## Reproducibility

Install dependencies from `requirements.txt`, then run `python main.py --data-dir /path/to/Datapack --output-dir output`. The default data directory is the location inspected for this run; both paths can be overridden.

## Output files

`top_10_batters.csv`, `top_10_bowlers.csv`, `phase_leaderboards.csv`, `player_metrics.csv`, `component_weights.csv`, `pressure_deliveries.csv`, `match_performance.csv`, `consistency.csv`, `format_comparison.csv`, `model_validation.csv`, `data_quality.csv`, `source_schema_audit.csv`, `sensitivity_rankings.csv`, `ranking_stability.csv`, and four PNG files under `plots/`.
"""
    path.write_text(report, encoding="utf-8")


def write_outputs(
    output_dir,
    prepared,
    quality,
    schema_audit,
    validation,
    batter_metrics,
    batter_top,
    batter_match_metrics,
    batter_format_metrics,
    bowler_metrics,
    bowler_top,
    bowler_match_metrics,
    bowler_format_metrics,
    sensitivity_rankings,
    stability,
    batter_phase_leaderboards,
    bowler_phase_leaderboards,
    component_weights,
):
    output_dir = Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    batter_top = batter_top.head(10)
    bowler_top = bowler_top.head(10)
    _safe_csv(batter_top, output_dir / "top_10_batters.csv")
    _safe_csv(bowler_top, output_dir / "top_10_bowlers.csv")
    _safe_csv(
        pd.concat(
            [
                batter_metrics.assign(player_type="batter"),
                bowler_metrics.assign(player_type="bowler"),
            ],
            ignore_index=True,
            sort=False,
        ),
        output_dir / "player_metrics.csv",
    )
    _safe_csv(
        pd.concat(
            [batter_phase_leaderboards, bowler_phase_leaderboards],
            ignore_index=True,
            sort=False,
        ),
        output_dir / "phase_leaderboards.csv",
    )
    _safe_csv(component_weights, output_dir / "component_weights.csv")
    pressure_columns = [
        column for column in PRESSURE_OUTPUT_COLUMNS if column in prepared.columns
    ]
    _safe_csv(
        prepared[pressure_columns], output_dir / "pressure_deliveries.csv"
    )
    match_performance = pd.concat(
        [
            batter_match_metrics.assign(player_type="batter"),
            bowler_match_metrics.assign(player_type="bowler"),
        ],
        ignore_index=True,
        sort=False,
    )
    _safe_csv(match_performance, output_dir / "match_performance.csv")

    consistency = pd.concat(
        [
            batter_metrics.loc[
                batter_metrics["metric_scope"].eq("overall")
            ].assign(player_type="batter"),
            bowler_metrics.loc[
                bowler_metrics["metric_scope"].eq("overall")
            ].assign(player_type="bowler"),
        ],
        ignore_index=True,
        sort=False,
    )
    _safe_csv(consistency, output_dir / "consistency.csv")
    _safe_csv(quality, output_dir / "data_quality.csv")
    _safe_csv(schema_audit, output_dir / "source_schema_audit.csv")
    _safe_csv(validation, output_dir / "model_validation.csv")
    _safe_csv(sensitivity_rankings, output_dir / "sensitivity_rankings.csv")
    _safe_csv(stability, output_dir / "ranking_stability.csv")

    format_comparison = _format_comparison(
        batter_format_metrics, bowler_format_metrics
    )
    _safe_csv(format_comparison, output_dir / "format_comparison.csv")

    overall_batters = batter_metrics.loc[
        batter_metrics["metric_scope"].eq("overall")
    ]
    overall_bowlers = bowler_metrics.loc[
        bowler_metrics["metric_scope"].eq("overall")
    ]
    _plot_rankings(
        batter_top, bowler_top, plots_dir / "top_rankings.png"
    )
    _plot_pressure_vs_normal(
        batter_format_metrics,
        bowler_format_metrics,
        plots_dir / "pressure_vs_normal.png",
    )
    _plot_reliability(
        overall_batters,
        overall_bowlers,
        plots_dir / "reliability_vs_performance.png",
    )
    _plot_pressure_heatmap(prepared, plots_dir / "pressure_heatmap.png")
    _write_methodology_report(
        output_dir / "methodology_report.md",
        prepared,
        quality,
        validation,
        batter_top,
        bowler_top,
        format_comparison,
        stability,
        sensitivity_rankings,
        component_weights,
        batter_phase_leaderboards,
        bowler_phase_leaderboards,
    )
    return format_comparison
