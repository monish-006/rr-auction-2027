# IPL and T20I Pressure Player Rankings (2022 onward)

Generated: 2026-11-17 19:01 UTC

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
| second_innings_rows_without_target | 136 |  |
| scheduled_ball_limits_not_default_120 | 36 | Revised-limit metadata values are multiples of six and are used as ball counts. |
| selected_dls_matches | 14 |  |
| dls_matches_missing_revised_target | 2 | A DLS chase without a revised target has target_runs=NaN and state_valid=False; no first-innings fallback is used. |
| incomplete_innings | 18 | Confirmed incomplete when a no-result/abandoned innings has no evidence of reaching all-out, its known schedule, or its target. |
| innings_completion_unknown | 26 | Completion cannot be established from the available result, wickets, target, and schedule fields. |
| future_run_labels_excluded | 3851 | Future-innings-run labels are training-eligible only for completed innings with valid state metadata. |
| future_run_labels_excluded_incomplete_innings | 892 | Observed delivery outcomes remain available; only the truncated innings-total-derived future-run target is excluded. |
| future_run_labels_excluded_unknown_completion | 2698 | Completion uncertainty is not treated as a completed innings. |
| par_model_observations_excluded | 32 | Par estimation uses only first innings with confirmed completion and valid match-state metadata. |
| par_model_observations_retained | 805 |  |
| valid_delivery_level_run_outcomes_retained | 193987 | Actual batter runs and bowler runs remain in the delivery data, including no-result matches; no-result status alone does not remove a delivery target. |
| scheduled_ball_limit_unknown_rows | 1070 | DLS rows without the corresponding revised schedule are not assigned a 120-ball fallback. |
| dls_match_state_treatment | 21586 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 122; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| dls_match_state_treatment | 21070 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 123; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| dls_match_state_treatment | 24822 | Missing fields: revised_target, first_inn_revised_overs, second_inn_revised_overs; affected state-invalid deliveries: 170; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| dls_match_state_treatment | 28046 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 123; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| dls_match_state_treatment | 28334 | Missing fields: revised_target, first_inn_revised_overs, second_inn_revised_overs; affected state-invalid deliveries: 218; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| dls_match_state_treatment | 29984 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 66; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| dls_match_state_treatment | 40749 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 121; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| dls_match_state_treatment | 48611 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 127; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| deliveries_excluded_from_state_dependent_analysis | 1070 | Includes unknown targets/schedules and rows without an OOF future-run/par estimate; excluded from pressure and player-event scoring. |
| no_result_wpa_events_excluded_from_individual_rankings | 1932 | WPA estimates may remain in pressure_deliveries.csv for diagnosis, but no-result matches contribute no player WPA score. |
| ambiguous_individual_wpa_attributions | 190986 | Decisive-match delivery-level team-state WPA proxies retained for diagnostics but deliberately not assigned to a player or included in the composite. |
| ambiguous_wpa_proxy_context_ordinary_delivery | 169416 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_other_dismissal | 9731 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_wide | 7367 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_leg_bye | 2617 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_no_ball | 810 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_run_out | 542 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_bye | 501 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_penalty | 1 | Team-state WPA proxy is not individually attributed. |
| ambiguous_wpa_proxy_context_dismissal_missing_player_id | 1 | Team-state WPA proxy is not individually attributed. |

`runs` is treated as striker runs. Team totals add both extras columns. Revised-over metadata values are multiples of six and are interpreted as ball limits. A missing schedule defaults to 120 balls only for matches not marked DLS; DLS innings with missing revised schedules have unknown balls remaining and are excluded from state-dependent analysis. A DLS chase without a revised target has an unknown target and is not assigned first-innings total + 1. The feed labels many no-balls as `Overstep`; `other` extras are treated as no-balls because 251 of those rows occur in overs with more than six recorded balls (out of 259 `other` rows). Wide and no-ball primary extras are charged to the bowler; byes, leg-byes and penalty extras are not. Secondary extras on no-balls and two rows with missing extras labels are excluded from bowler-charged extras because their attribution is not explicit in the source.

### DLS state metadata

Selected DLS matches: **14**; missing revised targets: **2**. Match-specific missing fields and treatment:

| match_id | missing_fields_and_treatment |
| --- | --- |
| 21586 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 122; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| 21070 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 123; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| 24822 | Missing fields: revised_target, first_inn_revised_overs, second_inn_revised_overs; affected state-invalid deliveries: 170; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| 28046 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 123; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| 28334 | Missing fields: revised_target, first_inn_revised_overs, second_inn_revised_overs; affected state-invalid deliveries: 218; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| 29984 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 66; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| 40749 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 121; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |
| 48611 | Missing fields: first_inn_revised_overs; affected state-invalid deliveries: 127; missing target/schedule is kept unknown, not imputed; observed delivery outcomes are retained. |

### State and label exclusions

- Deliveries excluded from state-dependent analysis: **1070**.
- Confirmed incomplete innings: **18**; completion unknown: **26**.
- Future-run labels excluded: **3851**; first-innings par observations excluded: **32**.
- No-result WPA events excluded from individual rankings: **1932**.
- Ambiguous individual WPA attributions not assigned to players: **190986**.
- Valid delivery-level run outcomes retained: **193987**.

An innings is marked complete only when the feed supports all-out, the known schedule is exhausted, a known chase target is reached, or the chasing team is the recorded winner. No-result/abandoned innings without such evidence are incomplete; other unresolved cases are marked unknown. Future-run labels derived from an innings total are used for training only when completion and state metadata are valid. Individual-delivery run outcomes are not excluded solely because their match had no result.

## Pre-delivery state and pressure index

Every feature is computed before the current delivery: innings, over/phase, scheduled and remaining legal balls, score, current run rate, wickets lost and in hand, target, runs required and required run rate. At the baseline 0.55 cutoff, **45,555** valid-state deliveries are labeled high pressure. Invalid DLS target/schedule rows have `pressure_index=NaN`, remain visible in the delivery output, and do not enter pressure/normal player opportunities or any state-dependent model. Wides, `Overstep`, `other` and penalty rows do not consume a legal ball. Wides and administrative penalty rows are excluded from batter opportunities; no-balls count as batter opportunities. First-innings targets are not available before the innings ends, so first-innings pressure uses a training-fold mean of completed first innings; second-innings pressure uses a known target.

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

Baseline eligibility requires at least **30 pressure deliveries**, **8 pressure matches**, and **30 normal deliveries**. DNBs are not assigned artificial zeroes; they have no delivery opportunities and are ineligible. Rankings are relative among eligible players, not claims that a player with a negative score performed above expectation.

## Results

### Top batters

| rank | player_name | ranking_score | pressure_runs_above_expected_per100 | pressure_strike_rate_impact | pressure_boundary_percentage_delta_per100 | pressure_wicket_preservation_per100 | pressure_composite | normal_composite | clutch_uplift | pressure_opportunities | pressure_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Ashutosh Sharma | 0.289 | 32.224 | 32.224 | 6.745 | 11.747 | 0.536 | 0.167 | 0.369 | 250 | 18 |
| 2 | Rashid Khan | 0.163 | 20.553 | 20.553 | 4.587 | 4.287 | 0.387 | 0.207 | 0.179 | 417 | 49 |
| 3 | Mark Chapman | 0.151 | 18.289 | 18.289 | 3.773 | 15.321 | 0.325 | 0.223 | 0.102 | 257 | 15 |
| 4 | Mark Adair | 0.137 | 11.528 | 11.528 | 3.161 | 3.803 | 0.368 | 0.096 | 0.272 | 198 | 21 |
| 5 | Romario Shepherd | 0.132 | 13.322 | 13.322 | 3.510 | 14.098 | 0.327 | 0.221 | 0.107 | 422 | 38 |
| 6 | Heinrich Klaasen | 0.123 | 18.617 | 18.617 | 2.739 | 10.608 | 0.268 | 0.443 | -0.175 | 404 | 30 |
| 7 | Tim David | 0.119 | 25.475 | 25.475 | 4.256 | 12.038 | 0.429 | 0.586 | -0.157 | 402 | 40 |
| 8 | Andre Russell | 0.118 | 17.976 | 17.976 | 5.497 | 2.435 | 0.350 | 0.600 | -0.250 | 330 | 29 |
| 9 | Akeal Hosein | 0.117 | 14.951 | 14.951 | 2.739 | 3.961 | 0.374 | 0.117 | 0.257 | 158 | 20 |
| 10 | Mahendra Singh Dhoni | 0.112 | 2.701 | 2.701 | 1.197 | 17.431 | 0.253 | 0.008 | 0.246 | 336 | 30 |

### Top bowlers

| rank | player_name | ranking_score | pressure_runs_saved_per100 | pressure_dot_percentage_delta_per100 | pressure_boundary_prevention_per100 | pressure_wicket_impact_per100 | pressure_composite | normal_composite | clutch_uplift | pressure_opportunities | pressure_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Mohsin Khan | 0.368 | 19.054 | 4.688 | 5.128 | 36.496 | 1.069 | 0.306 | 0.763 | 133 | 15 |
| 2 | Sunil Narine | 0.318 | 36.525 | 6.251 | 7.219 | -2.097 | 0.483 | 0.456 | 0.027 | 386 | 31 |
| 3 | Sai Kishore | 0.306 | 16.990 | 0.142 | 4.987 | 16.730 | 0.789 | 0.172 | 0.618 | 136 | 11 |
| 4 | Varun Chakravarthy | 0.299 | 14.796 | 8.771 | 2.167 | 11.562 | 0.520 | 0.487 | 0.034 | 778 | 56 |
| 5 | Mohit Sharma | 0.286 | 17.015 | 2.320 | 3.580 | 10.279 | 0.528 | -0.151 | 0.679 | 242 | 18 |
| 6 | Sufiyan Muqeem | 0.273 | 30.779 | 18.111 | 3.280 | 27.407 | 0.691 | 0.166 | 0.525 | 133 | 10 |
| 7 | Kuldeep Yadav | 0.271 | 6.248 | 1.992 | 2.071 | 20.602 | 0.560 | 0.400 | 0.159 | 533 | 43 |
| 8 | Rashid Khan | 0.258 | 7.819 | 3.122 | 2.044 | 5.889 | 0.451 | 0.285 | 0.166 | 899 | 67 |
| 9 | Yuzvendra Chahal | 0.248 | 11.221 | 0.914 | 3.599 | 13.512 | 0.500 | 0.083 | 0.416 | 573 | 49 |
| 10 | Adil Rashid | 0.225 | 5.089 | 4.927 | 0.583 | 18.188 | 0.381 | -0.003 | 0.384 | 403 | 27 |

### Phase leaderboards

The separate phase table uses the same format/role normalization, learned component weights, and reliability adjustment. It applies the baseline evidence thresholds within each phase; players without enough pressure and normal opportunities are omitted.

| player_type | phase | rank | player_name | ranking_score | pressure_composite | normal_composite | clutch_uplift | pressure_opportunities | pressure_matches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| batter | Powerplay (1-6) | 1 | Sikandar Raza | 0.088 | 0.192 | 0.064 | 0.128 | 52 | 11 |
| batter | Powerplay (1-6) | 2 | Rovman Powell | 0.064 | 0.370 | 0.507 | -0.136 | 37 | 8 |
| batter | Powerplay (1-6) | 3 | Aiden Markram | 0.046 | 0.437 | 0.261 | 0.176 | 74 | 12 |
| batter | Powerplay (1-6) | 4 | Tristan Stubbs | 0.023 | 0.183 | -0.127 | 0.310 | 42 | 10 |
| batter | Powerplay (1-6) | 5 | Heinrich Klaasen | 0.018 | -0.041 | 0.299 | -0.340 | 36 | 10 |
| batter | Powerplay (1-6) | 6 | Shimron Hetmyer | 0.018 | 0.100 | 0.669 | -0.569 | 54 | 9 |
| batter | Powerplay (1-6) | 7 | Salman Agha | 0.017 | 0.226 | 0.594 | -0.368 | 46 | 10 |
| batter | Powerplay (1-6) | 8 | Suryakumar Yadav | 0.016 | 0.258 | 0.387 | -0.129 | 63 | 10 |
| batter | Powerplay (1-6) | 9 | Dewald Brevis | 0.015 | 0.309 | -0.220 | 0.530 | 42 | 8 |
| batter | Powerplay (1-6) | 10 | Axar Patel | 0.011 | 0.104 | 0.010 | 0.095 | 40 | 9 |
| batter | Powerplay (1-6) | 11 | Namboori Tilak Varma | 0.001 | 0.054 | 0.319 | -0.265 | 65 | 14 |
| batter | Powerplay (1-6) | 12 | Towhid Hridoy | -0.056 | -0.315 | -0.019 | -0.297 | 36 | 8 |
| batter | Middle (7-15) | 1 | Mark Chapman | 0.237 | 0.436 | 0.475 | -0.039 | 195 | 13 |
| batter | Middle (7-15) | 2 | Romario Shepherd | 0.203 | 0.461 | -0.055 | 0.516 | 195 | 23 |
| batter | Middle (7-15) | 3 | Heinrich Klaasen | 0.203 | 0.471 | 0.591 | -0.120 | 289 | 25 |
| batter | Middle (7-15) | 4 | David Miller | 0.167 | 0.436 | -0.002 | 0.438 | 345 | 28 |
| batter | Middle (7-15) | 5 | Jitesh Sharma | 0.163 | 0.574 | 0.499 | 0.075 | 168 | 15 |
| batter | Middle (7-15) | 6 | Riyan Parag | 0.160 | 0.567 | 0.251 | 0.316 | 142 | 12 |
| batter | Middle (7-15) | 7 | Suryakumar Yadav | 0.152 | 0.438 | 0.617 | -0.179 | 291 | 23 |
| batter | Middle (7-15) | 8 | Andre Russell | 0.143 | 0.433 | 0.780 | -0.348 | 206 | 25 |
| batter | Middle (7-15) | 9 | Jimmy Neesham | 0.142 | 0.511 | 0.181 | 0.330 | 155 | 17 |
| batter | Middle (7-15) | 10 | Tim David | 0.136 | 0.531 | 0.553 | -0.022 | 155 | 24 |
| batter | Middle (7-15) | 11 | Shivam Dube | 0.130 | 0.588 | 0.232 | 0.356 | 302 | 30 |
| batter | Middle (7-15) | 12 | Shashank Singh | 0.120 | 0.378 | 0.068 | 0.309 | 141 | 17 |
| batter | Middle (7-15) | 13 | Moeen Ali | 0.119 | 0.431 | 0.052 | 0.379 | 143 | 15 |
| batter | Middle (7-15) | 14 | Towhid Hridoy | 0.110 | 0.251 | 0.063 | 0.188 | 138 | 12 |
| batter | Middle (7-15) | 15 | Rovman Powell | 0.109 | 0.296 | 0.094 | 0.202 | 369 | 38 |
| batter | Middle (7-15) | 16 | Glenn Phillips | 0.106 | 0.203 | 0.252 | -0.048 | 130 | 14 |
| batter | Middle (7-15) | 17 | Nehal Wadhera | 0.105 | 0.449 | 0.287 | 0.162 | 171 | 15 |
| batter | Middle (7-15) | 18 | Azmatullah Omarzai | 0.103 | 0.364 | 0.152 | 0.212 | 213 | 23 |

### Learned component weights

| player_type | level | component | weight | variance_explained | weight_method |
| --- | --- | --- | --- | --- | --- |
| batter | scoring_impact | runs_above_expected_per100 | 0.500 | 0.944 | pca |
| batter | scoring_impact | boundary_percentage_delta_per100 | 0.500 | 0.944 | pca |
| batter | overall_impact | scoring_impact | 0.500 | 0.621 | pca |
| batter | overall_impact | wicket_preservation | 0.500 | 0.621 | pca |
| bowler | bowling_control | runs_saved_per100 | 0.374 | 0.769 | pca |
| bowler | bowling_control | dot_percentage_delta_per100 | 0.277 | 0.769 | pca |
| bowler | bowling_control | boundary_prevention_per100 | 0.350 | 0.769 | pca |
| bowler | overall_impact | bowling_control | 0.500 | 0.636 | pca |
| bowler | overall_impact | wicket_taking | 0.500 | 0.636 | pca |

### IPL vs T20I

| player_type | competition_format | eligible_players | mean_pressure_per100 | mean_normal_per100 | mean_clutch_uplift_per100 |
| --- | --- | --- | --- | --- | --- |
| batter | IPL | 71 | -29.102 | -23.261 | -5.841 |
| batter | T20I | 76 | -30.163 | -23.802 | -6.361 |
| bowler | IPL | 80 | 22.856 | 22.158 | 0.698 |
| bowler | T20I | 106 | 31.807 | 25.329 | 6.479 |

### Out-of-fold model diagnostics

| target | observations | matches | folds | mae | rmse | r2 | brier_score | log_loss | roc_auc | train_validation_match_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expected_batter_runs | 185491 | 835 | 5 | 1.306 | 1.723 | 0.026 |  |  |  | 0 |
| expected_bowler_runs | 192917 | 835 | 5 | 1.276 | 1.699 | 0.024 |  |  |  | 0 |
| expected_future_team_runs | 190137 | 816 | 5 | 14.627 | 21.195 | 0.847 |  |  |  | 0 |
| expected_dot_probability | 192917 | 835 | 5 | 0.431 | 0.464 | 0.037 | 0.215 | 0.621 | 0.615 | 0 |
| expected_bowler_boundary_probability | 192917 | 835 | 5 | 0.290 | 0.381 | 0.020 | 0.145 | 0.463 | 0.604 | 0 |
| batter_boundary | 185491 | 835 | 5 |  |  |  | 0.316 | 0.591 | 0.613 | 0 |
| dismissal_type | 192917 | 835 | 5 |  |  |  | 0.102 | 0.218 | 0.629 | 0 |
| batting_team_win_probability | 190986 | 810 | 5 |  |  |  | 0.180 | 0.528 | 0.804 | 0 |

### Ranking stability

| player_type | scenario | eligible_players | baseline_top10_overlap | top10_jaccard | spearman_rank_correlation |
| --- | --- | --- | --- | --- | --- |
| batter | weights:equal | 131 | 7 | 0.538 | 0.887 |
| bowler | weights:equal | 178 | 8 | 0.667 | 0.948 |
| batter | weights:rate_focused | 151 | 8 | 0.667 | 0.964 |
| bowler | weights:rate_focused | 159 | 8 | 0.667 | 0.955 |
| batter | weights:resource_focused | 118 | 8 | 0.667 | 0.944 |
| bowler | weights:resource_focused | 167 | 8 | 0.667 | 0.943 |
| batter | component_weights:equal | 137 | 10 | 1.000 | 1.000 |
| bowler | component_weights:equal | 165 | 10 | 1.000 | 0.999 |
| batter | cutoff:0.50 | 158 | 8 | 0.667 | 0.934 |
| bowler | cutoff:0.50 | 175 | 9 | 0.818 | 0.919 |
| batter | cutoff:0.65 | 109 | 7 | 0.538 | 0.913 |
| bowler | cutoff:0.65 | 136 | 9 | 0.818 | 0.901 |
| batter | eligibility:20/5 | 191 | 10 | 1.000 | 1.000 |
| bowler | eligibility:20/5 | 227 | 10 | 1.000 | 1.000 |
| batter | eligibility:50/12 | 100 | 10 | 1.000 | 1.000 |
| bowler | eligibility:50/12 | 118 | 8 | 0.667 | 1.000 |

## Sensitivity

`sensitivity_rankings.csv` contains all eligible ranked players under alternative pressure weights, pressure cutoffs, eligibility thresholds (20/5 and 50/12, against baseline 30/8), and equal instead of PCA component weights. `ranking_stability.csv` compares each scenario's top-10 overlap/Jaccard and eligible-player rank correlation with the baseline. Cutoff and minimum-sample changes alter which deliveries and players qualify; treat these as robustness checks, not independent validation.

## Limitations

The source contains no published ball-by-ball win probabilities, venue conditions, batter intent, or explicit no-ball subtype for `other`. The fitted win-probability model is an estimate based on match winners and may be imperfectly calibrated. Expected models estimate league-average context rather than player-specific ability, and delivery outcomes are noisy. PCA weights are data-driven but unsupervised; equal-weight sensitivity results should be reviewed before relying on close ranks. Pressure weights, pressure scales, cutoff, role bins, and eligibility thresholds remain explicit analyst choices. IPL/T20I comparisons use format- and role-standardized scores, while raw run contributions remain in each format's own units. Results are descriptive, not causal.

## Reproducibility

Install dependencies from `requirements.txt`, then run `python main.py --data-dir /path/to/Datapack --output-dir output`. The default data directory is the location inspected for this run; both paths can be overridden.

## Output files

`top_10_batters.csv`, `top_10_bowlers.csv`, `phase_leaderboards.csv`, `player_metrics.csv`, `component_weights.csv`, `pressure_deliveries.csv`, `match_performance.csv`, `consistency.csv`, `format_comparison.csv`, `model_validation.csv`, `data_quality.csv`, `source_schema_audit.csv`, `sensitivity_rankings.csv`, `ranking_stability.csv`, and four PNG files under `plots/`.
