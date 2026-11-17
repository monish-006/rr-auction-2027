# IPL and T20I Pressure Player Rankings (2022 onward)

Generated: 2026-11-16 17:17 UTC

## Data scope and audit

The pipeline inspected all six supplied CSV files and their columns, missing values, keys and joins. It selected IPL competitions named `IPL YYYY` and T20I competitions with seasons beginning in 2022 or later; other `T20` competitions were excluded. The delivery data contains **193,988** regular-innings deliveries across **367 IPL** and **470 T20I** matches with deliveries. Super Overs were excluded because their six-ball format is not comparable to the standard innings model. 16 selected match metadata rows have no ball-by-ball records. Duplicate delivery keys: 0. Wickets with no dismissed-player ID: 1; dismissal IDs not matching either listed batter on that delivery: 0.

### Data quality checks

| check | value | detail |
| --- | --- | --- |
| selected_metadata_matches | 853 |  |
| selected_matches_without_decisive_winner | 41 | No-result, abandoned, and tied matches are excluded from win-probability model training. |
| selected_matches_with_deliveries | 837 |  |
| selected_matches_without_deliveries | 16 |  |
| selected_delivery_rows_after_super_over_exclusion | 193988 |  |
| duplicate_delivery_keys | 0 |  |
| source_delivery_rows_without_match_metadata | 0 |  |
| super_over_rows_excluded | 26 |  |
| duplicate_playing_eleven_keys | 40 | Deduplicated for roster validation; delivery records remain the scoring source. |
| duplicate_match_ids | 0 |  |
| duplicate_competition_ids | 0 |  |
| duplicate_competition_team_keys | 0 |  |
| duplicate_competition_team_player_keys | 0 |  |
| batter_delivery_players_not_in_roster | 0 | Retained from ball-by-ball IDs (likely impact-substitution coverage). |
| bowler_delivery_players_not_in_roster | 0 |  |
| batting_team_not_home_or_away_rows | 0 |  |
| wickets_with_missing_dismissed_player_id | 1 |  |
| dismissed_player_not_striker_or_non_striker | 0 | Non-striker run-outs are retained and attributed by dismissed_player_id. |
| credited_wickets_with_missing_bowler_id | 0 |  |
| unclassified_extra_rows_with_extra_runs | 2 | Extra amounts on rows with no extras_type are included in team totals but not bowler-charged runs. |
| other_extra_rows | 259 | Treated as no-balls based on delivery-number patterns. |
| other_extra_rows_in_overs_with_more_than_six_balls | 251 |  |
| wide_and_penalty_rows_excluded_from_batter_opportunities | 7473 | No-balls count as batter opportunities; wides and administrative penalty rows do not. |
| second_innings_rows_without_target | 0 |  |
| scheduled_ball_limits_not_default_120 | 34 | Revised-limit metadata values are multiples of six and are used as ball counts. |

`runs` is treated as striker runs. Team totals add both extras columns. Revised-over metadata values are multiples of six and are interpreted as ball limits; otherwise the limit is 120 balls. The feed labels many no-balls as `Overstep`; `other` extras are treated as no-balls because 251 of those rows occur in overs with more than six recorded balls (out of 259 `other` rows). Wide and no-ball primary extras are charged to the bowler; byes, leg-byes and penalty extras are not. Secondary extras on no-balls and two rows with missing extras labels are excluded from bowler-charged extras because their attribution is not explicit in the source.

## Pre-delivery state and pressure index

Every feature is computed before the current delivery: innings, over/phase, scheduled and remaining legal balls, score, current run rate, wickets lost and in hand, target, runs required and required run rate. At the baseline 0.55 cutoff, **45,200** deliveries are labeled high pressure. Wides, `Overstep`, `other` and penalty rows do not consume a legal ball. Wides and administrative penalty rows are excluded from batter opportunities; no-balls count as batter opportunities. First-innings targets are not available before the innings ends, so first-innings pressure uses the training-fold mean first-innings score as par; second-innings pressure uses the actual or revised target.

All components are bounded to [0, 1]:

- **Run-rate pressure, first innings:** `clip((training par - projected innings total) / (0.20 × training par), 0, 1)`. The projection is current score plus expected remaining team runs.
- **Run-rate pressure, chase:** `clip((runs required / expected remaining runs - 1) / 0.50, 0, 1)`.
- **Resource pressure:** `wickets lost / 9`.
- **Stage:** legal balls elapsed / scheduled legal balls.
- **Match leverage, chase:** `clip(1 - abs(runs required - expected remaining runs) / max(runs required, expected remaining runs, 1), 0, 1)`. It is unavailable in the first innings and is not imputed.

Baseline first-innings weights are run rate **0.50**, resources **0.30**, stage **0.20**. Chase weights are run rate **0.35**, resources **0.20**, leverage **0.25**, stage **0.20**. The first-innings weights sum to one without an unobservable match-leverage term. This is a transparent starting specification, not a learned causal truth. The pipeline tests equal, rate-focused and resource-focused weights, as well as pressure cutoffs of 0.50, 0.55 and 0.65.

## Expected performance and leakage controls

Match-grouped out-of-fold `HistGradientBoosting` models estimate striker runs, bowler-charged runs, future innings runs, batter four/six probabilities, dismissal probabilities, bowler dot-ball probability, and bowler boundary probability from pre-delivery state. A separate state model estimates batting-team win probability from match state. The latter is trained against `win_team_id` from other matches; the final result is a training label, never a state feature. Each match has equal total training weight so long matches do not dominate. Pre- and post-delivery states are both scored by the held-out model; five-fold `GroupKFold` validation keeps every delivery from a match out of that fold's training data. `model_validation.csv` reports run/rate errors plus classification Brier/log-loss/AUC where applicable. A model's low predictive performance should be treated as uncertainty in its expected values, not hidden by the ranking.

The win-probability-added (WPA) value for a delivery is `estimated batting-team win probability after - before`. It is contextualized by the match state, not the raw outcome. Batter impact uses WPA added; bowler impact uses the negative of that change (win probability reduced). The model is learned from match outcomes because the CSVs do not contain published ball-by-ball probabilities; WPA is an estimate, not a causal or official win-probability value.

## Player contributions and reliability score

- **Batting diagnostics:** runs above expectation and strike-rate impact are both reported; with the same batter-opportunity denominator, strike-rate impact is the runs-above-expectation rate expressed per 100 balls, so it is not counted a second time. Four/six counts and rates are compared with state-based expected probabilities. Wicket preservation is expected dismissal probability times wicket run value, minus actual dismissal cost. Actual dismissal cost is attributed by `dismissed_player_id`, including non-striker run-outs.
- **Bowling diagnostics:** runs saved are expected bowler-charged runs minus actual; dot impact is actual minus expected dot percentage; boundary prevention is expected minus actual boundary percentage. Credited wicket impact is `(actual credited wicket - expected credited wicket probability) × wicket run value`. Only caught, bowled, LBW, stumped (`Stump`) and hit-wicket dismissals receive bowler credit; run-outs, retire-outs, and obstructing-the-field dismissals do not.
- **Scoring/control dimensions:** correlated indicators are not added as separate bonuses. Standardized runs-above-expectation and boundary-rate residuals form one batting scoring dimension. Standardized runs saved, dot-rate delta, and boundary-prevention measures form one bowling control dimension. The first principal-component loading magnitudes from the observed player-context data, normalized to sum to one, provide the component weights. Wicket impact and model-based WPA are separate dimensions.
- **Final composite:** the scoring/control, wicket, and match-impact dimensions are standardized within format, role, and phase. A second first-component PCA produces the data-driven overall weights. The primary ranking uses these PCA weights; an equal-component alternative is tested for stability. `component_weights.csv` records the learned weights and variance explained. This unsupervised weighting uses observed covariance, not match-result labels, and is a transparent, reproducible compromise rather than a claim that covariance is the only valid value system.
- **Pressure and normal scores:** the same learned weights are applied separately to pressure and normal-condition performance. Clutch uplift is `pressure composite - normal composite`; it is reported directly and is not added as an extra bonus to the pressure ranking.
- The pressure composite is multiplied by sample shrinkage `n/(n+50) × min(1, pressure_matches/8)`, by match consistency `0.5 + 0.5 × positive-pressure-match share`, and by season stability `0.75 + 0.25 × season stability`. Season stability is `1/(1 + SD(season rates)/(abs(mean season rate)+1))`. The overall player score is the square-root-opportunity-weighted mean of format/role/phase scores, with context opportunities capped at 400.

Baseline eligibility requires at least **30 pressure deliveries**, **8 pressure matches**, and **30 normal deliveries**. DNBs are not assigned artificial zeroes; they have no delivery opportunities and are ineligible. Rankings are relative among eligible players, not claims that a player with a negative score performed above expectation.

## Results

### Top batters

| rank | player_name | ranking_score | pressure_runs_above_expected_per100 | pressure_strike_rate_impact | pressure_boundary_percentage_delta_per100 | pressure_wicket_preservation_per100 | pressure_win_probability_added_per100 | pressure_composite | normal_composite | clutch_uplift | pressure_opportunities | pressure_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Ashutosh Sharma | 0.172 | 33.201 | 33.201 | 6.996 | 9.872 | 0.307 | 0.327 | 0.103 | 0.224 | 251 | 19 |
| 2 | Mark Chapman | 0.138 | 18.291 | 18.291 | 4.331 | 12.921 | 0.654 | 0.324 | 0.225 | 0.099 | 253 | 15 |
| 3 | Rashid Khan | 0.123 | 23.322 | 23.322 | 4.933 | 3.840 | 0.838 | 0.274 | 0.045 | 0.229 | 399 | 48 |
| 4 | Heinrich Klaasen | 0.110 | 17.071 | 17.071 | 2.154 | 6.755 | 0.049 | 0.199 | 0.445 | -0.247 | 404 | 28 |
| 5 | Tim David | 0.110 | 24.016 | 24.016 | 4.306 | 12.167 | 0.789 | 0.367 | 0.529 | -0.162 | 406 | 41 |
| 6 | Jitesh Sharma | 0.110 | 37.619 | 37.619 | 7.869 | 4.203 | 0.600 | 0.349 | 0.171 | 0.178 | 245 | 20 |
| 7 | Riyan Parag | 0.108 | 15.663 | 15.663 | 2.086 | 8.279 | 0.024 | 0.173 | 0.249 | -0.076 | 248 | 20 |
| 8 | Suryakumar Yadav | 0.103 | 20.817 | 20.817 | 5.787 | 8.956 | 0.257 | 0.392 | 0.484 | -0.092 | 439 | 32 |
| 9 | David Miller | 0.098 | 13.448 | 13.448 | 2.696 | 9.138 | 0.286 | 0.250 | 0.249 | 0.001 | 508 | 37 |
| 10 | Romario Shepherd | 0.093 | 12.343 | 12.343 | 2.994 | 16.654 | 0.212 | 0.186 | 0.233 | -0.047 | 403 | 38 |

### Top bowlers

| rank | player_name | ranking_score | pressure_runs_saved_per100 | pressure_dot_percentage_delta_per100 | pressure_boundary_prevention_per100 | pressure_wicket_impact_per100 | pressure_win_probability_reduced_per100 | pressure_composite | normal_composite | clutch_uplift | pressure_opportunities | pressure_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Sunil Narine | 0.272 | 34.069 | 6.379 | 6.606 | -3.301 | 0.182 | 0.424 | 0.501 | -0.077 | 400 | 33 |
| 2 | Varun Chakravarthy | 0.256 | 14.090 | 8.112 | 2.151 | 10.955 | 0.288 | 0.489 | 0.471 | 0.018 | 782 | 56 |
| 3 | Mohsin Khan | 0.241 | 18.348 | 4.497 | 4.904 | 28.405 | 0.061 | 0.741 | 0.365 | 0.376 | 131 | 14 |
| 4 | Sufiyan Muqeem | 0.236 | 29.327 | 17.469 | 3.229 | 29.671 | 0.748 | 0.649 | 0.214 | 0.435 | 131 | 10 |
| 5 | Sai Kishore | 0.232 | 21.792 | 0.204 | 6.174 | 12.466 | -0.138 | 0.540 | 0.137 | 0.403 | 132 | 10 |
| 6 | Yuzvendra Chahal | 0.229 | 11.639 | 1.666 | 3.523 | 14.237 | 0.272 | 0.516 | 0.093 | 0.423 | 571 | 49 |
| 7 | Rashid Khan | 0.227 | 6.678 | 2.309 | 2.085 | 8.106 | 0.099 | 0.383 | 0.362 | 0.021 | 883 | 67 |
| 8 | Mohit Sharma | 0.223 | 14.570 | 1.798 | 3.315 | 9.555 | -0.143 | 0.413 | -0.143 | 0.556 | 238 | 18 |
| 9 | Kuldeep Yadav | 0.201 | 4.991 | 1.119 | 2.139 | 16.084 | -0.052 | 0.383 | 0.422 | -0.039 | 532 | 41 |
| 10 | Abrar Ahmed | 0.191 | 29.611 | 10.003 | 5.002 | 2.717 | 0.227 | 0.335 | 0.217 | 0.119 | 269 | 22 |

### Phase leaderboards

The separate phase table uses the same format/role normalization, learned component weights, and reliability adjustment. It applies the baseline evidence thresholds within each phase; players without enough pressure and normal opportunities are omitted.

| player_type | phase | rank | player_name | ranking_score | pressure_composite | normal_composite | clutch_uplift | pressure_opportunities | pressure_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter | Powerplay (1-6) | 1 | Sikandar Raza | 0.063 | -0.117 | 0.030 | -0.147 | 59 | 12 |
| batter | Powerplay (1-6) | 2 | Rovman Powell | 0.033 | 0.215 | 0.582 | -0.367 | 32 | 8 |
| batter | Powerplay (1-6) | 3 | Aiden Markram | 0.032 | 0.307 | 0.262 | 0.045 | 75 | 11 |
| batter | Powerplay (1-6) | 4 | Suryakumar Yadav | 0.028 | 0.480 | 0.388 | 0.092 | 57 | 11 |
| batter | Powerplay (1-6) | 5 | Heinrich Klaasen | 0.016 | -0.190 | 0.493 | -0.683 | 35 | 10 |
| batter | Powerplay (1-6) | 6 | Salman Agha | 0.013 | 0.198 | 0.558 | -0.360 | 48 | 9 |
| batter | Powerplay (1-6) | 7 | Dewald Brevis | 0.012 | 0.325 | -0.238 | 0.563 | 39 | 8 |
| batter | Powerplay (1-6) | 8 | Shimron Hetmyer | 0.010 | -0.016 | 0.520 | -0.536 | 54 | 9 |
| batter | Powerplay (1-6) | 9 | Tristan Stubbs | 0.005 | -0.012 | -0.391 | 0.378 | 43 | 10 |
| batter | Powerplay (1-6) | 10 | Axar Patel | 0.002 | -0.052 | -0.281 | 0.230 | 44 | 9 |
| batter | Powerplay (1-6) | 11 | Namboori Tilak Varma | -0.002 | 0.031 | 0.338 | -0.307 | 61 | 14 |
| batter | Powerplay (1-6) | 12 | Towhid Hridoy | -0.042 | -0.218 | -0.035 | -0.183 | 35 | 9 |
| batter | Middle (7-15) | 1 | Mark Chapman | 0.214 | 0.436 | 0.411 | 0.025 | 195 | 13 |
| batter | Middle (7-15) | 2 | Riyan Parag | 0.207 | 0.592 | 0.284 | 0.308 | 137 | 12 |
| batter | Middle (7-15) | 3 | Heinrich Klaasen | 0.188 | 0.422 | 0.562 | -0.139 | 290 | 24 |
| batter | Middle (7-15) | 4 | Suryakumar Yadav | 0.167 | 0.446 | 0.585 | -0.139 | 294 | 25 |
| batter | Middle (7-15) | 5 | Jitesh Sharma | 0.166 | 0.534 | 0.401 | 0.133 | 160 | 16 |
| batter | Middle (7-15) | 6 | David Miller | 0.152 | 0.377 | 0.114 | 0.264 | 345 | 26 |
| batter | Middle (7-15) | 7 | Romario Shepherd | 0.144 | 0.292 | -0.033 | 0.325 | 195 | 23 |
| batter | Middle (7-15) | 8 | Jimmy Neesham | 0.121 | 0.404 | 0.380 | 0.024 | 147 | 17 |
| batter | Middle (7-15) | 9 | Andre Russell | 0.121 | 0.309 | 0.604 | -0.295 | 205 | 24 |
| batter | Middle (7-15) | 10 | Towhid Hridoy | 0.108 | 0.235 | 0.138 | 0.098 | 145 | 15 |
| batter | Middle (7-15) | 11 | Shivam Dube | 0.107 | 0.567 | 0.193 | 0.374 | 307 | 32 |
| batter | Middle (7-15) | 12 | Shashank Singh | 0.104 | 0.316 | 0.028 | 0.288 | 141 | 17 |
| batter | Middle (7-15) | 13 | Nicholas Pooran | 0.104 | 0.455 | 0.238 | 0.218 | 257 | 30 |
| batter | Middle (7-15) | 14 | Tim David | 0.103 | 0.409 | 0.473 | -0.065 | 163 | 25 |
| batter | Middle (7-15) | 15 | Nitish Rana | 0.092 | 0.428 | 0.287 | 0.140 | 79 | 8 |
| batter | Middle (7-15) | 16 | Nehal Wadhera | 0.091 | 0.353 | 0.380 | -0.027 | 168 | 15 |
| batter | Middle (7-15) | 17 | Glenn Maxwell | 0.090 | 0.386 | 0.291 | 0.094 | 89 | 11 |
| batter | Middle (7-15) | 18 | Rovman Powell | 0.083 | 0.218 | 0.149 | 0.070 | 370 | 37 |

### Learned component weights

| player_type | level | component | weight | variance_explained | weight_method |
| --- | --- | --- | --- | --- | --- |
| batter | scoring_impact | runs_above_expected_per100 | 0.500 | 0.946 | pca |
| batter | scoring_impact | boundary_percentage_delta_per100 | 0.500 | 0.946 | pca |
| batter | overall_impact | scoring_impact | 0.302 | 0.577 | pca |
| batter | overall_impact | wicket_preservation | 0.333 | 0.577 | pca |
| batter | overall_impact | match_impact | 0.365 | 0.577 | pca |
| bowler | bowling_control | runs_saved_per100 | 0.373 | 0.769 | pca |
| bowler | bowling_control | dot_percentage_delta_per100 | 0.277 | 0.769 | pca |
| bowler | bowling_control | boundary_prevention_per100 | 0.350 | 0.769 | pca |
| bowler | overall_impact | bowling_control | 0.344 | 0.575 | pca |
| bowler | overall_impact | wicket_taking | 0.293 | 0.575 | pca |
| bowler | overall_impact | match_impact | 0.363 | 0.575 | pca |

### IPL vs T20I

| player_type | competition_format | eligible_players | mean_pressure_per100 | mean_normal_per100 | mean_clutch_uplift_per100 |
| --- | --- | --- | --- | --- | --- |
| batter | IPL | 69 | -28.378 | -24.603 | -3.775 |
| batter | T20I | 78 | -30.321 | -24.102 | -6.219 |
| bowler | IPL | 81 | 23.498 | 21.693 | 1.805 |
| bowler | T20I | 107 | 31.301 | 26.114 | 5.186 |

### Out-of-fold model diagnostics

| target | observations | matches | folds | mae | rmse | r2 | brier_score | log_loss | roc_auc | train_validation_match_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expected_batter_runs | 186515 | 837 | 5 | 1.305 | 1.722 | 0.026 |  |  |  | 0 |
| expected_bowler_runs | 193987 | 837 | 5 | 1.275 | 1.699 | 0.024 |  |  |  | 0 |
| expected_future_team_runs | 193988 | 837 | 5 | 15.059 | 21.998 | 0.835 |  |  |  | 0 |
| expected_dot_probability | 193987 | 837 | 5 | 0.431 | 0.464 | 0.037 | 0.215 | 0.621 | 0.614 | 0 |
| expected_bowler_boundary_probability | 193987 | 837 | 5 | 0.290 | 0.381 | 0.020 | 0.145 | 0.463 | 0.604 | 0 |
| batter_boundary | 186515 | 837 | 5 |  |  |  | 0.316 | 0.591 | 0.613 | 0 |
| dismissal_type | 193987 | 837 | 5 |  |  |  | 0.102 | 0.218 | 0.629 | 0 |
| batting_team_win_probability | 192056 | 812 | 5 |  |  |  | 0.180 | 0.529 | 0.803 | 0 |

### Ranking stability

| player_type | scenario | eligible_players | baseline_top10_overlap | top10_jaccard | spearman_rank_correlation |
| --- | --- | --- | --- | --- | --- |
| batter | weights:equal | 124 | 8 | 0.667 | 0.915 |
| bowler | weights:equal | 177 | 10 | 1.000 | 0.920 |
| batter | weights:rate_focused | 147 | 9 | 0.818 | 0.939 |
| bowler | weights:rate_focused | 162 | 9 | 0.818 | 0.923 |
| batter | weights:resource_focused | 117 | 7 | 0.538 | 0.938 |
| bowler | weights:resource_focused | 166 | 9 | 0.818 | 0.927 |
| batter | component_weights:equal | 135 | 10 | 1.000 | 0.998 |
| bowler | component_weights:equal | 165 | 10 | 1.000 | 0.995 |
| batter | cutoff:0.50 | 160 | 6 | 0.429 | 0.940 |
| bowler | cutoff:0.50 | 181 | 9 | 0.818 | 0.913 |
| batter | cutoff:0.65 | 112 | 5 | 0.333 | 0.841 |
| bowler | cutoff:0.65 | 135 | 8 | 0.667 | 0.872 |
| batter | eligibility:20/5 | 185 | 10 | 1.000 | 1.000 |
| bowler | eligibility:20/5 | 222 | 10 | 1.000 | 1.000 |
| batter | eligibility:50/12 | 101 | 10 | 1.000 | 1.000 |
| bowler | eligibility:50/12 | 118 | 8 | 0.667 | 1.000 |

## Sensitivity

`sensitivity_rankings.csv` contains all eligible ranked players under alternative pressure weights, pressure cutoffs, eligibility thresholds (20/5 and 50/12, against baseline 30/8), and equal instead of PCA component weights. `ranking_stability.csv` compares each scenario's top-10 overlap/Jaccard and eligible-player rank correlation with the baseline. Cutoff and minimum-sample changes alter which deliveries and players qualify; treat these as robustness checks, not independent validation.

## Limitations

The source contains no published ball-by-ball win probabilities, venue conditions, batter intent, or explicit no-ball subtype for `other`. The fitted win-probability model is an estimate based on match winners and may be imperfectly calibrated. Expected models estimate league-average context rather than player-specific ability, and delivery outcomes are noisy. PCA weights are data-driven but unsupervised; equal-weight sensitivity results should be reviewed before relying on close ranks. Pressure weights, pressure scales, cutoff, role bins, and eligibility thresholds remain explicit analyst choices. IPL/T20I comparisons use format- and role-standardized scores, while raw run contributions remain in each format's own units. Results are descriptive, not causal.

## Reproducibility

Install dependencies from `requirements.txt`, then run `python main.py --data-dir /path/to/Datapack --output-dir output`. The default data directory is the location inspected for this run; both paths can be overridden.

## Output files

`top_10_batters.csv`, `top_10_bowlers.csv`, `phase_leaderboards.csv`, `player_metrics.csv`, `component_weights.csv`, `pressure_deliveries.csv`, `match_performance.csv`, `consistency.csv`, `format_comparison.csv`, `model_validation.csv`, `data_quality.csv`, `source_schema_audit.csv`, `sensitivity_rankings.csv`, `ranking_stability.csv`, and four PNG files under `plots/`.
