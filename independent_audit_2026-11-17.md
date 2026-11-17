# Independent Audit: Players Under Pressure

**Audit date:** 2026-11-17  
**Project:** IPL/T20I pressure-player rankings, 2022 onward  
**Audit scope:** Current working-tree code and available CSV inputs/generated outputs.  
**Change control:** No existing source, data, or output file was changed. This report is the sole audit deliverable. No full pipeline rerun was performed because the normal run overwrites files in `output/`; supplemental cutoff rerankings were calculated in memory from the saved out-of-fold delivery data.

The worktree was already dirty before the audit (branch `main`, HEAD `947d9e6`, with modified/untracked project files). Findings therefore describe the inspected working tree, not a clean commit or a before/after diff.

## 1. Files inspected and execution flow

### Source and project files

| File | Role in the implementation |
|---|---|
| `main.py` | Command-line entry point; orchestrates the current end-to-end pipeline and sensitivity scenarios. |
| `src/pressure_rankings/config.py` | Default paths, model feature lists, baseline pressure weights, cutoff, eligibility and sensitivity settings, credited-dismissal types. |
| `src/pressure_rankings/data.py` | Loads all six CSVs, selects matches, audits data, interprets extras, and builds pre-delivery states and opportunity flags. |
| `src/pressure_rankings/models.py` | Fits match-grouped out-of-fold run, event-probability, future-run and win-probability models; emits diagnostics and delivery predictions. |
| `src/pressure_rankings/scoring.py` | The current path uses `calculate_pressure`. It also contains older event, summary and ranking helpers that the current `main.py` does not call. |
| `src/pressure_rankings/impact.py` | Current player-event attribution, component metrics, PCA/equal weights, reliability adjustments, overall and phase rankings. |
| `src/pressure_rankings/reporting.py` | Writes CSV outputs, plots, and the generated methodology report. |
| `src/pressure_rankings/__init__.py` | Package docstring only. |
| `tests/test_pressure_rankings.py` | Four focused unit tests for pressure bounds/leverage, dismissal credit, WPA attribution, and ranking output shape. It does not test the full data/model pipeline, DLS cases, model calibration, or the sensitivity grid. |
| `README.md` | User-facing methodology and run instructions; useful documentation, but not treated as ground truth for this audit. |
| `requirements.txt` | Python dependency constraints. |
| `src/testFile.py`, `src/phase_performance.py` | Earlier exploratory scripts. Neither is imported by `main.py` and neither produces the current pressure rankings. |

### Input data inspected

All six input files were present under `/Users/monish_006/Documents/RR/Datapack`. The row/column counts below are from `source_schema_audit.csv`.

| Input | Rows × columns | Role in this pipeline |
|---|---:|---|
| `ball_by_ball.csv` | 200,999 × 33 | Delivery, player, scoring, extras, wicket, batting-order, and over-order source. |
| `matches.csv` | 883 × 22 | Match metadata, results, targets, revised-innings fields, team IDs and dates. |
| `competitions.csv` | 141 × 4 | Competition name, format and season. |
| `playing_eleven.csv` | 28,381 × 9 | Participation/roster check and fallback batting order for dismissal attribution. |
| `teams.csv` | 353 × 3 | Audited for duplicate competition/team keys; not used to create player events. |
| `competition_team_player_mapping.csv` | 6,413 × 6 | Audited for duplicate competition/team/player keys; not used to create player events. |

The ball-by-ball source itself contains `batting_order`. `impact.py` uses that order first and the roster table as a fallback; the README's description that the roster supplies batting order is incomplete.

### Generated outputs present

The `output/` directory contained `source_schema_audit.csv`, `data_quality.csv`, `pressure_deliveries.csv`, `model_validation.csv`, `component_weights.csv`, `player_metrics.csv`, `match_performance.csv`, `consistency.csv`, `phase_leaderboards.csv`, `format_comparison.csv`, `sensitivity_rankings.csv`, `ranking_stability.csv`, top-10 batter/bowler CSVs, `methodology_report.md`, and four plots under `output/plots/`.

The audit directly used `source_schema_audit.csv`, `data_quality.csv`, `pressure_deliveries.csv`, `model_validation.csv`, `component_weights.csv`, `player_metrics.csv`, `sensitivity_rankings.csv`, `ranking_stability.csv`, the top-10 CSVs, and the generated methodology report. The generated methodology report is dated 2026-11-16 17:17 UTC.

### Actual execution flow

`main.py::run_pipeline` calls:

1. `data.load_and_prepare`: select and order deliveries; calculate runs, wicket and legal-ball state; infer chase target and scheduled-ball limit.
2. `models.fit_oof_expectations`: generate out-of-fold expected outcomes and win probabilities.
3. `scoring.calculate_pressure`: add pressure components and index.
4. `impact.build_player_events`: create batter and bowler events.
5. `impact.rank_players`: create each role's player metrics, PCA-weighted composite and phase boards.
6. `main._run_sensitivity`: rerun rankings for configured pressure weights, cutoffs and eligibility thresholds; compare with baseline.
7. `reporting.write_outputs`: write existing output files and plots.

`main.py` calls `scoring.calculate_pressure` but calls `impact.build_player_events` and `impact.rank_players` for current rankings. The similarly named older ranking helpers in `scoring.py` are not the current ranking path. This duplicate implementation is a maintenance/documentation risk.

## 2. What could and could not be verified

**Verifiable from the inspected code/data:** actual pressure formulas and weights; event attribution; model features, targets and cross-validation; run-model bias/calibration; saved PCA metrics and rankings; additional cutoff sensitivity; opportunity and eligibility counts; extras and wicket labels; source duplicates; target/schedule fields and observed DLS results.

**Not independently verifiable with the available materials:**

- The CSVs provide no codebook establishing the exact semantics of `extras_type="other"`, `extras1_runs` versus `extras_2_runs`, or every revised-target field.
- There is no complete DLS/resource timeline or official revised target for two DLS-result matches; those match states cannot be reconstructed from the supplied data.
- There is no external expected-runs benchmark or published delivery-level win-probability series. Model calibration can be checked against held-out outcomes, but not against an official “true” expectation.
- The source does not identify which fielder or runner caused most delivery-level win-probability changes. Individual causal WPA cannot be verified from these columns.
- It does not provide authoritative powerplay phase adjustments for shortened innings, weather/interruption timelines, or a full designation of an innings as complete/incomplete.
- Saved diagnostics do not contain the actual train-match IDs for every fold. The code's `GroupKFold` and overlap assertion, together with the saved validation-fold IDs, support the no-match-overlap conclusion; the exported `train_validation_match_overlap` field itself is hard-coded as zero rather than independently calculated.
- A fresh complete run was intentionally not performed because it would replace existing outputs. The saved OOF dataset and current source were used for read-only reanalysis instead.

## 3. Executive findings

The core pipeline is genuinely match-grouped out-of-fold, computes delivery states before the delivery, and implements the documented baseline pressure weights and reliability multipliers. Several important qualifications or defects remain:

1. **DLS and shortened-match states can be wrong.** The code substitutes first-innings total plus one whenever `revised_target` is null, including DLS results. Two DLS wins in the supplied matches have no revised target, and the reconstructed target exceeds the chasing team's actual final score.
2. **No-result/abandoned delivery records are included in rankings.** Twenty-five non-decisive matches have 1,932 delivery rows; the win-probability model has no labels for these matches, but the code still produces WPA values for them and uses those values in player metrics. Partial innings also contribute to future-team-run labels and first-innings par estimates.
3. **WPA is a team-state change assigned in full to an individual.** The striker (or recorded dismissed player) receives the full batting-team probability change and the bowler receives the full opposite value. This is a contextual proxy, not an individual causal contribution.
4. **PCA weights reflect observed covariance/variation, not cricket value or outcome importance.** The equal-component alternative leaves the same top-ten membership, but pressure weights and cutoffs can materially change batter top tens.
5. **The requested threshold grid is not in the pipeline.** `config.py` has only 0.50 and 0.65 as sensitivity cutoffs in addition to the 0.55 baseline. The 0.45 and 0.60 results below were recomputed from saved OOF data for this audit.
6. **Run models have low discrimination but little global calibration bias.** Their R² values are about 0.02. Bias and prediction-decile calibration are good in aggregate, but the models only modestly improve over a match-fold training-mean baseline.
7. **Reliability multipliers overlap and are not all aligned to the full composite.** Consistency uses a narrower run/wicket contribution than the final multimetric score; season stability returns 1.0 for a player with one season, which treats unmeasured stability as maximal stability.

These are not all equal in severity: the DLS/partial-match state defects directly affect the state and WPA being ranked; PCA and reliability are modeling-design limitations; the run model is weakly predictive but reasonably calibrated.

## 4. PCA and composite scoring

### PCA inputs, signed loadings and normalized weights

I reproduced the PCA fits from the standardized player-context metrics persisted in `player_metrics.csv`. Each fit stacks pressure and normal-condition rows for the same metric group. PCA uses the first component's signed loadings; the ranking code then takes absolute loadings and normalizes them to sum to one:

`weight_j = abs(PC1_loading_j) / sum_k(abs(PC1_loading_k))`

In this run all reproduced PC1 loadings are positive, so taking absolute values does not change their signs. The output `component_weights.csv` stores normalized weights and explained variance, but not signed loadings or the PCA input-row counts; the signed values below were reconstructed.

| Player type | PCA level | Input metric/component | Signed PC1 loading | Final normalized weight | PC1 variance explained | Valid condition rows |
|---|---|---|---:|---:|---:|---:|
| Batter | Scoring dimension | Runs above expected per 100 | 0.707107 | 0.500000 | 0.945565 | 4,448 |
| Batter | Scoring dimension | Boundary-percentage delta per 100 | 0.707107 | 0.500000 | 0.945565 | 4,448 |
| Batter | Overall composite | Scoring impact | 0.521447 | 0.301956 | 0.576985 | 4,448 |
| Batter | Overall composite | Wicket-preservation dimension | 0.574993 | 0.332963 | 0.576985 | 4,448 |
| Batter | Overall composite | Match-impact dimension | 0.630457 | 0.365081 | 0.576985 | 4,448 |
| Bowler | Bowling-control dimension | Runs saved per 100 | 0.641885 | 0.373421 | 0.768810 | 3,042 |
| Bowler | Bowling-control dimension | Dot-percentage delta per 100 | 0.475386 | 0.276559 | 0.768810 | 3,042 |
| Bowler | Bowling-control dimension | Boundary prevention per 100 | 0.601658 | 0.350019 | 0.768810 | 3,042 |
| Bowler | Overall composite | Bowling-control dimension | 0.592784 | 0.343587 | 0.574636 | 3,042 |
| Bowler | Overall composite | Wicket-taking dimension | 0.505701 | 0.293112 | 0.574636 | 3,042 |
| Bowler | Overall composite | Match-impact dimension | 0.626796 | 0.363301 | 0.574636 | 3,042 |

The batter scoring PC explains 94.6% of variation in its two highly related standardized inputs; the bowler control PC explains 76.9%. Those percentages describe how much **input variance** the PC captures. They do not say how important the metric is to winning or how much player value it explains.

The PCA inputs are standardized player-format-role-phase rates, not individual delivery rows. The fit is global across available player contexts; it is not restricted to eligible players or well-supported contexts. Using the baseline 30-opportunity/8-pressure-match/30-normal-opportunity thresholds as a **context-level support screen**, only 243 of 4,448 batter condition rows and 450 of 3,042 bowler condition rows meet all three criteria. The other rows still influence the PCA weights. This is an important small-sample sensitivity risk; the thresholds themselves are player-level eligibility rules, so this context-level comparison is diagnostic rather than a code requirement.

### How the weights become the score

For each player context `c = format × role × phase` and condition `q = pressure or normal`, each listed rate is standardized within that context across players. The inner weighted dimension is then standardized again:

- **Batting:** `S(c,q) = z_c(0.5000 × z_c(runs_above_expected_per100) + 0.5000 × z_c(boundary_delta_per100))`.
- **Bowling:** `C(c,q) = z_c(0.3734 × z_c(runs_saved_per100) + 0.2766 × z_c(dot_delta_per100) + 0.3500 × z_c(boundary_prevention_per100))`.

The final context composites are:

- `BatterComposite(c,q) = 0.3020 × S(c,q) + 0.3330 × z_c(wicket_preservation_per100) + 0.3651 × z_c(WPA_added_per100)`.
- `BowlerComposite(c,q) = 0.3436 × C(c,q) + 0.2931 × z_c(wicket_impact_per100) + 0.3633 × z_c(WPA_reduced_per100)`.

`z_c` here means a z-score over players within the same format, role and phase, separately for pressure and normal conditions. The same learned weights are used for both conditions. Context scores are combined into the overall ranking using the reliability formula in Section 8.

PCA is **not measuring importance**: it finds a direction of high covariance/variation, and the code uses normalized absolute loadings as positive weights rather than PCA scores. Grouping correlated measures into one dimension reduces the appearance of an additive list of bonuses, but does not make the metrics independent. Batting boundary rate overlaps with runs scored; bowling dots/boundary prevention overlap with runs saved. Four-rate and six-rate residuals are computed and reported but are not direct composite inputs; the composite uses overall boundary percentage.

### Ranking comparison

`top-10 overlap` is the number of baseline top-ten players still in that scenario's top ten. Jaccard is overlap divided by the union of the two top tens. Rank correlation is Spearman correlation among all eligible players common to baseline and scenario, not just the top ten. The 0.45 and 0.60 rows were calculated in this audit from `pressure_deliveries.csv`; other scenarios are in the existing sensitivity outputs.

| Scenario | Batter eligible | Batter top-10 overlap / Jaccard / Spearman | Bowler eligible | Bowler top-10 overlap / Jaccard / Spearman |
|---|---:|---:|---:|---:|
| Baseline PCA, cutoff 0.55 | 135 | 10 / 1.000 / 1.000 | 165 | 10 / 1.000 / 1.000 |
| Equal pressure weights | 124 | 8 / 0.667 / 0.915 | 177 | 10 / 1.000 / 0.920 |
| Rate-focused pressure weights | 147 | 9 / 0.818 / 0.939 | 162 | 9 / 0.818 / 0.923 |
| Resource-focused pressure weights | 117 | 7 / 0.538 / 0.938 | 166 | 9 / 0.818 / 0.927 |
| Equal component weights, cutoff 0.55 | 135 | 10 / 1.000 / 0.998 | 165 | 10 / 1.000 / 0.995 |
| Cutoff 0.45 | 184 | 6 / 0.429 / 0.832 | 202 | 7 / 0.538 / 0.847 |
| Cutoff 0.50 | 160 | 6 / 0.429 / 0.940 | 181 | 9 / 0.818 / 0.913 |
| Cutoff 0.55 | 135 | 10 / 1.000 / 1.000 | 165 | 10 / 1.000 / 1.000 |
| Cutoff 0.60 | 125 | 8 / 0.667 / 0.911 | 151 | 9 / 0.818 / 0.907 |
| Cutoff 0.65 | 112 | 5 / 0.333 / 0.841 | 135 | 8 / 0.667 / 0.872 |
| Eligibility 20 opportunities / 5 matches | 185 | 10 / 1.000 / 1.000 | 222 | 10 / 1.000 / 1.000 |
| Eligibility 50 opportunities / 12 matches | 101 | 10 / 1.000 / 1.000 | 118 | 8 / 0.667 / 1.000 |

Equal component weights preserve every baseline top-ten player for both roles, although they reorder players. The pressure cutoff is much more consequential for batter membership: only 5 of the baseline ten remain at 0.65; at 0.45, only 6 remain. At 0.45, the batter replacements include Andre Russell, Fakhar Zaman, Rovman Powell and Shashank Singh; at 0.60, Ayush Badoni and Nicholas Pooran enter. At 0.45, three bowler replacements are Krunal Pandya, Ravindra Jadeja and Sakib Hussain; at 0.60, Mohammad Nawaz enters and Abrar Ahmed leaves. These are sensitivity effects, not evidence that one cutoff is objectively correct.

## 5. Expected-performance models

### Leakage and model specification

`fit_oof_expectations` uses `GroupKFold` with `match_id` groups, up to five folds. It explicitly asserts that each fold's train and validation match sets do not overlap. The saved `validation_fold` assigns every match to one validation fold; the maximum number of validation folds per match in the saved file is one. The run, event, future-run and WP predictions used downstream are assigned to validation rows, not in-sample predictions.

The run models are `HistGradientBoostingRegressor` models with squared-error loss. That loss is an appropriate objective for estimating the **conditional mean** of runs, but it does not make the conditional mean easy to predict. Features include format, innings, over/phase, scheduled and remaining balls, wickets, score/run rate, target/required rate and batting order. They do not include batter/bowler identity, bowling style, venue, or weather. The model is therefore a league-average state expectation, not a player-specific expectation.

### Out-of-fold run-model performance and calibration

The training-fold mean comparison below is computed separately for each validation fold using only rows assigned to the other folds.

| Model | OOF observations | Actual mean | Predicted mean | Bias (predicted − actual) | MAE: model vs fold mean | RMSE: model vs fold mean | OOF R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| Batter runs | 186,515 | 1.3998 | 1.4006 | +0.0008 | 1.3054 vs 1.3338 | 1.7225 vs 1.7453 | 0.026 |
| Bowler runs conceded | 193,987 | 1.3959 | 1.3969 | +0.0010 | 1.2754 vs 1.3004 | 1.6991 vs 1.7199 | 0.024 |

The aggregate calibration slopes/intercepts from actual versus predicted runs are approximately 1.006 / -0.009 for batters and 1.005 / -0.008 for bowlers. Prediction-decile means are close to observed-decile means. Across the 12 format × innings × phase groups, mean prediction bias ranges from -0.024 to +0.016 runs for batters and -0.024 to +0.015 for bowlers; group MAEs range about 1.11–1.51 and 1.09–1.48 runs respectively.

**Interpretation:** the run expectations are nearly unbiased in aggregate and reasonably calibrated across prediction deciles, but explain only about 2–3% of delivery-level run variance and improve MAE over a fold-specific constant mean by only about 2%. This is a weak-signal model for individual deliveries. It may still provide a marginal baseline when residuals are aggregated over many opportunities, but close player differences should not be overinterpreted. The observed R² does not by itself prove a coding error; it does show the current model has limited ability to distinguish expected runs by state.

Other OOF diagnostics are also modest for event probabilities: dot probability Brier 0.215/AUC 0.614; bowler boundary probability Brier 0.145/AUC 0.604; batter boundary multiclass Brier 0.316/AUC 0.613; dismissal-type multiclass Brier 0.102/AUC 0.629. Future-team-runs R² is 0.835, but its targets include truncated recorded innings as discussed in Section 9.

### Win-probability model

The win-probability model uses only pre-delivery state features and `win_team_id` as its training label. The code builds a post-delivery state by updating team runs, wickets, legal balls, required runs/rate and phase; the held-out model predicts before and after, then subtracts before from after. Its match-grouped validation has 192,056 delivery observations from 812 decisive matches; match-balanced Brier score is 0.180, log loss 0.529 and AUC 0.803. Match-balanced mean predicted probability is 0.4881 against observed 0.4891 (bias -0.0010). This supports useful ordering of match states, but does not validate player-level causal attribution or DLS states.

The diagnostics report `train_validation_match_overlap=0` as a literal constant. The source assertion and group split establish no overlap in this run; the exported field is not itself a computed audit statistic. `GroupKFold` is also not a forward-in-time split, so it establishes unseen-match performance, not future-season generalization.

## 6. Pressure index

Let `b` be legal balls bowled before the delivery, `L` the scheduled legal balls, `w` wickets lost, `s` the current score, `T` the chase target, `R = max(T-s, 0)` runs required, `F` expected future team runs from the current state, and `P` the training-fold mean first-innings total.

The implementation is:

- `Stage = clip(b / L, 0, 1)`.
- `Resource = clip(w / 9, 0, 1)`.
- First innings: `RRpressure = clip((P - (s + F)) / (0.20 × max(P, 1)), 0, 1)`.
- Chase: `RRpressure = clip((R / max(F, 1) - 1) / 0.50, 0, 1)`.
- Chase leverage: `Leverage = clip(1 - abs(R-F) / max(R, F, 1), 0, 1)`. First-innings leverage is set to zero.
- First-innings index: `0.50 × RRpressure + 0.30 × Resource + 0.20 × Stage`.
- Chase index: `0.35 × RRpressure + 0.20 × Resource + 0.25 × Leverage + 0.20 × Stage`.
- A pressure opportunity is `pressure_index >= cutoff`; baseline cutoff is 0.55.

All weights sum to one. In controlled states, raising stage from 0 to 1 adds 0.20 to either index; raising lost wickets from 0 to 9 adds 0.30 in an opening innings and 0.20 in a chase. Raising first-innings run-rate pressure from 0 to 1 adds 0.50; raising chase run-rate pressure or leverage from 0 to 1 adds 0.35 or 0.25 respectively. Thus each component is monotonic in isolation. In the actual data, mean index rises with over band and wickets lost; chase mean is higher than first-innings mean, as expected from the additional target/leverage state. These observed patterns are descriptive and mix different match situations.

At cutoff 0.55, 45,200 of 193,988 deliveries are pressure deliveries. The requested cutoffs select 67,314 (0.45), 53,918 (0.50), 45,200 (0.55), 38,525 (0.60), and 32,340 (0.65).

**Redundancy risk:** the future-runs model already conditions on wickets and balls/phase, so `RRpressure`/leverage partly encode resource and stage effects before the explicit resource and stage weights are added. Chase run-rate pressure and leverage also share `R` and `F`. This is not an algebraic duplicate, but the components are correlated and their weights do not represent independent causes. The pipeline does not report component correlations or perform an ablation that removes one component at a time.

## 7. Win-probability attribution

The before state uses pre-delivery fields. The after state updates score with `delivery_runs`, wickets with `is_wicket`, legal balls with `is_legal_ball`, and chase state from the target. `win_team_id` is not a feature. For valid decisive-match rows this is a coherent, match-held-out state transition.

Attribution in `impact.build_player_events` is broader than the evidence supports:

- A batter receives the full `batting-team WP after - before` on a batter-opportunity row, attributed to the striker unless a wicket with a dismissed-player ID occurred, in which case that player receives it.
- The bowler receives the full negative of that same change for each bowler-opportunity row.
- A batter does not receive a WPA event for a wide/penalty row because those are not batter opportunities. A striker can receive full WPA on byes/leg-byes or no-balls even where the change was not caused by the striker. A bowler receives the whole team-state change even if the outcome was largely due to batting, a run-out/fielder, or extras.
- This is not double-counting inside either separate role leaderboard, but it is a full team-state proxy credited to each role's player separately. It cannot establish an individual's causal contribution.
- No-winner matches have missing `batting_team_won` labels and are excluded from WP training/evaluation, but their 1,932 delivery rows still receive predicted WPA and enter rankings; 610 of those are pressure rows. Their “win probability added” is hypothetical and should not be scored as observed decisive-match WPA.

## 8. Reliability, eligibility and exact final score

For each player-context, the code computes:

- `n = pressure opportunities`, `m = matches with at least one pressure opportunity`.
- `shrinkage = n/(n+50) × min(1, m/8)`.
- `consistency = 0.5 + 0.5 × positive_pressure_match_share`.
- `season_stability = 1/(1 + SD(season_pressure_rates)/(abs(mean(season_pressure_rates))+1))`, using population SD; one or zero observed seasons returns 1.0.
- `stability_factor = 0.75 + 0.25 × season_stability`.
- `adjusted_context_score = pressure_composite × shrinkage × consistency × stability_factor`.
- `overall ranking_score = sum_c(adjusted_context_score_c × sqrt(min(n_c,400))) / sum_c(sqrt(min(n_c,400)))`, across contexts with pressure opportunities.

Eligibility is evaluated at player level: at least 30 pressure opportunities, at least 8 pressure matches, and at least 30 normal opportunities. The current run has 135 eligible batters and 165 eligible bowlers.

The consistency factor's `positive_pressure_match_share` is **not computed from the final multimetric composite**. Its match contribution is batter runs-above-expectation minus actual wicket cost, or bowler runs-saved plus actual credited-wicket run value. It omits the expected wicket-risk adjustment, dot/boundary dimensions and WPA. Therefore the reliability adjustment can disagree with the composite it multiplies.

Consistency and season stability are distinct but overlapping temporal-stability measures; sample size also enters shrinkage and the square-root context weight. This may over-penalize or reweight sample size. More importantly, one-season players get season stability 1.0 and stability factor 1.0: the code treats insufficient evidence of multi-season stability as no penalty, rather than unknown stability.

Among current top bowlers, Sufiyan Muqeem has 131 pressure opportunities in 10 pressure matches (shrinkage 0.724), Sai Kishore 132/10 (0.725), and Mohsin Khan 131/14 (0.724). Under the stricter 50/12 eligibility scenario, Sufiyan and Sai drop out of the top ten; the other eight baseline bowlers remain. The top-ten batter list is unchanged by 50/12. Thus the current threshold admits a few top-ranked bowlers whose evidence is close to the match-count floor.

`clutch_uplift` is pressure composite minus normal composite, but is not included in the ranking score. Pressure and normal scores are standardized separately, then aggregated with different weights (`sqrt(n_pressure)` versus `sqrt(n_normal)`); the resulting difference is a descriptive contrast, not a matched-state causal uplift.

## 9. Cricket-data correctness

### Correctly implemented or supported by checks

- The selected set has 193,988 regular-innings deliveries across 837 matches (367 IPL, 470 T20I); 26 innings-3+ Super Over rows are excluded. Deliveries are sorted by match, innings and `innings_ball_num`; no duplicate delivery keys or duplicate innings-ball numbers were found, and the within-innings sequence is monotonic.
- The regular target is first-innings total + 1; a non-null `revised_target` overrides it for innings two. `runs_required_before` is clipped at zero and the required rate uses remaining legal balls.
- Revised-over values used as ball limits are multiples of six. Thirty-four selected matches have a non-120 scheduled-ball limit; limits range from 30 to 114 balls. `balls_remaining` is clipped at zero.
- Team runs are batter runs plus both extras columns. Wides, `Overstep`, `other` and penalty are treated as illegal; byes/leg-byes are legal. Batter opportunities exclude wides and penalty but include no-balls, byes and leg-byes. Bowler opportunities exclude penalty but include wides/no-balls. This is broadly consistent with standard balls-faced/balls-bowled accounting.
- The source has 10,478 wicket-flag rows: 9,899 credited to bowlers (`Caught`, `Bowled`, `LBW`, `Stump`, `Hit Wicket`) and 579 uncredited (`RunOut`, `Retire Out`, and the two spellings of obstructing the field). The observed types are covered by the configured credit set; there is no separate `Caught and Bowled` label in these rows.
- A dismissal with an ID is assigned to that dismissed player, including a non-striker. One wicket lacks a dismissed-player ID; no dismissal ID disagrees with striker/non-striker IDs. The missing ID cannot receive the correct batter cost.

### Confirmed or potential data defects

- **DLS fallback is wrong in two confirmed cases.** Among 14 selected match results containing “DLS”, two lack `revised_target`. Code falls back to original first-innings total + 1. Match 24822 (India in Ireland T20I Series 2023) is a DLS win by India with innings-two score 47 after 41 legal balls, while the pipeline assigns target 140. Match 28334 (Bangladesh in New Zealand T20I Series 2023) is a DLS win by New Zealand with score 95 after 88 legal balls, while the pipeline assigns target 111. Both chasing winners finish below the pipeline's target. In these rows, target, required runs/rate, leverage, pressure and WP features are not the actual DLS state.
- **Scheduled length can also be unavailable for DLS.** Of the 14 DLS-result matches, first-innings revised-ball metadata is missing in 8 and second-innings revised-ball metadata in 2. The code defaults missing values to 120; for an interrupted chase this can leave the model believing there are balls remaining after the actual interruption. The exact intended schedule cannot be restored without source metadata.
- **No-result/abandoned partial innings influence future-runs/par labels.** There are 41 selected matches without a decisive winner: 21 `No Result`, 19 `Match Abandoned`, 1 `Match Tied`. Sixteen have no deliveries; the remaining 25 contain 1,932 delivery rows. Nineteen of their 28 recorded innings end before ten wickets, the configured scheduled limit, or the reconstructed target. Immediate per-delivery batter/bowler run targets from these deliveries are valid observed outcomes, but the `future_team_runs = recorded innings total - current score` target and first-innings par can reflect truncated innings. The run-model target mask and par calculation do not exclude or identify them. WPA from these rows is also included as detailed above.
- **Quality check misses DLS target fallback.** `second_innings_rows_without_target` is zero because every missing DLS target is filled with first-innings total + 1. This verifies that a value was created, not that it is the correct target. DLS result strings and revised-target availability should be cross-checked.
- **Extras are inferred and two rows are unclassified.** Source counts are 7,472 wides, 565 `Overstep`, 259 `other`, 2,661 leg-byes, 505 byes and 1 penalty row; `extras_type` is missing for most ordinary balls. All 259 `other` rows are treated as no-balls because 251 occur in overs with more than six recorded balls; 8 do not have that supporting pattern. Two rows have nonzero extra runs but missing `extras_type`; they are treated as legal batter opportunities, and their extras are not charged to bowlers. These rows are surfaced in the audit but not excluded or resolved.
- **Roster duplicates have conflicting batting order.** The input has 40 duplicate `(match, team, player)` keys overall; 35 are in selected matches and have conflicting `batting_order` values. All 35 affected players also occur as strikers in the ball feed, so source-delivery batting order takes precedence in current roles. The output quality check counts duplicates but does not detect conflicts; the roster fallback could be ambiguous for future/non-striker-only cases.
- **Fixed phase boundaries on shortened games.** `phase` is based on `overid` bands 1–6, 7–15, 16–20 even when scheduled innings are 5, 7, 9 or 15 overs. This follows the project's stated phase definition, but it does not represent a rule-adjusted powerplay or comparable phases for shortened matches. No official adjusted phase data is present.

## 10. Confirmed issues, potential issues and correctly implemented components

### Confirmed issues

1. Two DLS-result matches without revised targets use an incorrect original-total target; pressure and win-state features are consequently wrong.
2. No-winner delivery rows receive WPA and affect rankings despite having no decisive outcome label; partial innings can contaminate future-run/par training targets.
3. Team-level WPA is credited in full to an individual batter and bowler without action/fielder decomposition; the measure is not causal individual value.
4. PCA is unsupervised variance weighting, not importance weighting; most PCA condition rows fail the same context-level evidence screen used to define minimum eligible samples.
5. The code's sensitivity cutoff set omits 0.45 and 0.60 requested in this audit.
6. Match-consistency reliability is computed from a narrower contribution than the final score; single-season evidence is treated as maximal season stability.

### Potential issues / limitations

1. Run expectations are calibrated in aggregate but weakly discriminative; key omitted context (player/bowler style, venue/weather) may matter, but the current evidence does not establish which feature or model change would fix it.
2. Pressure components overlap through shared state variables and the state-aware expected-future-run model.
3. Four/six rate residuals are produced but do not enter the batting composite directly; the combined boundary rate is included and overlaps with run residuals.
4. Bowler role comparison is phase-based; available `bowling_type`/`pace_or_spin` fields are not used in expectations or role normalization.
5. Fixed phase bands do not adapt to shortened innings.
6. The current normal-vs-pressure uplift is not a matched-state or causal comparison and uses different pressure/normal context weights.
7. `component_weights.csv` omits signed PCA loadings, PCA sample counts and fitting details; the hard-coded overlap diagnostic also limits auditability from exported CSVs alone.

### Correctly implemented components

- Match-level `GroupKFold` separation, explicit train/validation match-set assertion, complete OOF assignments, and no observed match appearing in more than one validation fold.
- Pre-delivery state construction for score, wickets, legal balls, target and required rate; post-delivery updates are used only for the WP-after estimate.
- Baseline pressure formula, bounds and weights match the code and output; first-innings leverage is zero.
- Opportunity handling for common wide/no-ball/bye/leg-bye/penalty labels is consistent with the stated assumptions; bowler-credited dismissal labels in the actual dataset match the credit set.
- Dismissal cost and batter WPA use `dismissed_player_id` when present, including non-striker dismissals.
- DNBs are not assigned artificial zero-event rows; eligibility requires actual pressure/normal opportunities.
- Separate batter/bowler and phase rankings, format/role/phase standardization, equal-component sensitivity, match/season summaries, and output generation are present.

## 11. Recommended corrections, ranked by importance

1. **Repair match-state labels before interpreting rankings.** Require valid revised targets/schedules for DLS states; do not fall back to original target when result indicates DLS. Flag incomplete/abandoned innings and exclude their truncated final-total labels from future-runs/par training. Keep valid per-delivery run outcomes where appropriate. Exclude no-decisive-result rows from WPA ranking contributions unless a draw/no-result model is explicitly defined.
2. **Reframe or revise WPA attribution.** Keep WPA as a team-state contextual feature, but do not describe the entire delta as caused by the striker/bowler. Exclude it from individual composite scores unless there is a defensible allocation rule, or explicitly label it a full-delivery proxy and report team-state impact separately. Handle extras, run-outs, fielder actions and unknown dismissed-player IDs explicitly.
3. **Validate expected runs against decision-relevant baselines.** Retain match-grouped OOF predictions, report training-fold mean comparisons and error/calibration by role, phase, format and innings, and compare a small number of plausible conditional-mean models. Preserve a clear interpretation that R² is low. Avoid selecting features/models on the same held-out matches used to report rank quality.
4. **Complete the pressure sensitivity grid.** Add 0.45 and 0.60 to the saved pipeline; report the number of pressure deliveries/opportunities and eligible players for each. Add component correlations/ablations so the chosen pressure dimensions are not presumed independent.
5. **Treat PCA as a covariance summary, not importance.** If retained, save signed loadings, valid-row counts and fit inputs; assess weight stability under bootstrap or eligibility filtering. Compare with equal, predeclared cricket-value weights. Do not claim PCA selected weights by predictive validation.
6. **Align reliability to the scored outcome.** Calculate match consistency from the same multimetric contribution used in the composite, define how one-season players are handled, and quantify the combined effect of sample shrinkage plus square-root context weighting. Check whether season and match consistency are redundant before multiplying both.
7. **Make data assumptions explicit and tested.** Resolve or exclude untyped extra rows; validate `other` subtype with a source codebook; detect conflicting roster orders; and either apply shortened-match phase rules or mark shortened-game phases as non-comparable.

## 12. Accurate presentation description of the current code

The current pipeline creates pre-delivery match states and uses match-grouped out-of-fold models to estimate expected runs, event probabilities and batting-team win probability. A bounded pressure index combines run-rate pressure, wickets lost, innings stage and (in a chase) leverage. At a 0.55 cutoff, batter and bowler delivery outcomes are compared with context-model expectations, converted into per-100-opportunity metrics, standardized within format/role/phase and combined using PCA-derived positive loading weights. The pressure composite is multiplied by opportunity/match shrinkage, positive-pressure-match consistency and season-stability factors, then averaged across contexts with square-root opportunity weights capped at 400. Eligibility is 30 pressure opportunities, 8 pressure matches and 30 normal opportunities. Clutch uplift is reported as pressure composite minus normal composite but is not an added ranking bonus.

This accurately describes the implemented score, with two qualifications for a presentation: PCA weights represent covariance/variation, not player-value importance; and expected-run models are close to aggregate-calibrated but have low delivery-level R². DLS states with missing revised targets, incomplete no-result states and player-level WPA attribution require correction or explicit limitation before the rankings are presented as definitive.
