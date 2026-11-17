# IPL and T20I Pressure Player Rankings

This project builds pressure-adjusted rankings for batters and bowlers using the supplied ball-by-ball data. The entry point is `main.py`; it runs the data checks, feature engineering, match-separated expectation models, rankings, sensitivity analysis, and report/plot generation.

## 1. Requirements

- Python 3.9 or newer
- The six CSV files listed below
- Packages in `requirements.txt` (Pandas, NumPy, scikit-learn, Matplotlib, and Seaborn)

From the project root, create and activate a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

## 2. Input files

Put all six files in one directory:

| File | How the pipeline uses it |
| --- | --- |
| `ball_by_ball.csv` | Delivery outcomes, player IDs, extras, wickets, and over/ball order. |
| `matches.csv` | Connects `match_id` to competitions, teams, match dates, revised targets/innings limits, and the final winning team used as the win-probability model's training label. |
| `competitions.csv` | Supplies competition names, formats, and seasons via `comp_id`. |
| `playing_eleven.csv` | Checks match/team/player participation and supplies batting order for dismissal attribution. |
| `teams.csv` | Included in the source audit; duplicate `(comp_id, team_id)` keys are counted. |
| `competition_team_player_mapping.csv` | Included in the source audit; duplicate `(comp_id, team_id, player_id)` keys are counted. |

The important joins are `matches.comp_id → competitions.comp_id`, `ball_by_ball.match_id → matches.match_id`, and `playing_eleven.match_id/team_id/player_id → match participants`. The teams and mapping files are audited for their keys but are not used to create player performance events. Player attribution for deliveries and dismissals uses the IDs in `ball_by_ball.csv`; names are labels, not join keys.

By default the code looks in `/Users/monish_006/Documents/RR/Datapack`, the data directory used during development. For another location, pass `--data-dir` as shown below. The pipeline checks that all six files exist and stops with an error if it encounters an unrecognized extras type or an invalid revised innings limit.

## 3. Run the pipeline

Run from the repository root:

```bash
python main.py
```

To specify the data and output directories:

```bash
python main.py --data-dir "/path/to/Datapack" --output-dir output
```

Run the focused correctness tests with:

```bash
./.venv/bin/python -m unittest discover -s tests -v
```

If using the virtual environment directly:

```bash
./.venv/bin/python main.py --data-dir "/path/to/Datapack" --output-dir output
```

The pipeline prints the number of analyzed matches, cross-validation diagnostics, and the final eligible top-10 lists. All generated files go into the selected output directory; plots go in its `plots/` subdirectory. Running it again replaces those output files.

## 4. What the code does

The code is split into small modules:

| File | Responsibility |
| --- | --- |
| `main.py` | Command-line interface and orchestration of the complete run. |
| `src/pressure_rankings/config.py` | Default paths, cross-validation settings, pressure weights, and eligibility thresholds. |
| `src/pressure_rankings/data.py` | Loads all CSVs, selects matches, audits data quality, accounts for runs/extras, and builds pre-delivery state. |
| `src/pressure_rankings/models.py` | Fits context-based models and generates match-grouped out-of-fold predictions. |
| `src/pressure_rankings/scoring.py` | Computes the pre-delivery pressure index and retains the original run/wicket scoring helpers. |
| `src/pressure_rankings/impact.py` | Attributes batting/bowling event metrics, learns PCA component weights, and creates overall and phase leaderboards. |
| `src/pressure_rankings/reporting.py` | Writes CSVs, charts, and the generated methodology report. |

`src/testFile.py` and `src/phase_performance.py` are separate earlier exploratory scripts; `main.py` does not use them.

### 4.1 Competition and innings selection

- T20I matches are selected from competitions whose `match_format` is `T20I`.
- IPL matches are selected where the format is `T20` and the competition name matches `IPL YYYY`. Other T20 competitions are deliberately excluded.
- The competition season and match date must be 2022 or later.
- Innings 1 and 2 are included. Innings 3 and 4 (Super Overs in this data) are excluded because they do not have the same innings structure as a regular T20 innings.
- Matches listed in `matches.csv` without delivery rows cannot contribute to player performance.

### 4.2 Runs, extras, balls, and wickets

The inspected feed stores batter runs in `runs` and extras in `extras1_runs` and `extras_2_runs`.

- **Team runs on a delivery:** `runs + extras1_runs + extras_2_runs`.
- **Batter runs:** `runs`.
- **Bowler-charged runs:** batter runs plus the primary extra for `wide`, `Overstep`, and `other`. Byes, leg-byes, penalty extras, and secondary extras on no-balls are not charged to the bowler by this pipeline.
- `Overstep` is treated as a no-ball. `other` is also treated as a no-ball because its delivery-number pattern usually shows an extra ball in the over. This is a data-specific inference, not an explicit subtype in the CSV.
- Wide and penalty rows are not batter opportunities; no-balls are. Byes and leg-byes still represent a batter opportunity.
- Wides and no-balls are bowling opportunities; administrative penalty rows are excluded from batter and bowler opportunity counts.
- Wides, no-balls, `other`, and penalty rows do not consume a legal ball. Revised innings limits in `matches.csv` are stored as ball counts (multiples of six); a missing schedule defaults to 120 balls only for matches not marked DLS. A DLS innings with missing revised schedule is marked unknown and excluded from state-dependent analysis.
- All `is_wicket` events reduce wickets in hand. For bowler wicket credit, only `Caught`, `Bowled`, `LBW`, `Stump`, and `Hit Wicket` count. Run-outs, retire-outs, and obstructing-the-field dismissals are not credited to the bowler.
- Batter dismissal cost is assigned to `dismissed_player_id`, including when the dismissed player is the non-striker. A wicket with no dismissed-player ID cannot be assigned a batter dismissal cost.

The `data_quality.csv` output records duplicate delivery keys, roster mismatches, missing dismissal IDs, team-ID mismatches, and other checks. The current data has a small number of unclassified extras; their team runs are retained, but their extras are not charged to a bowler.

### 4.3 Pre-delivery match state

Rows are ordered within each match and innings by `innings_ball_num`. For each delivery, state is calculated from earlier deliveries only:

- innings, over, and phase (powerplay 1–6, middle 7–15, slog 16–20);
- legal balls bowled and remaining, using the scheduled innings limit;
- score, current run rate, wickets lost, and wickets in hand;
- for a chase, target, runs required, and required run rate.

The score and wickets shown for a delivery are the values **before** that delivery. A non-DLS chase uses `revised_target` when present and otherwise first-innings total plus one. A DLS chase requires a supplied revised target; if it is missing, the target and chase state are marked unknown rather than imputed. Invalid state rows stay in the delivery output for observed outcomes but do not enter state-dependent models, pressure or player rankings.

### 4.4 Expected-performance models

Context models estimate striker runs, bowler-charged runs, future innings runs, batter four/six probabilities, batter dismissal probability, bowler-credited-wicket probability, dot-ball probability, and boundary probability. The individual run/event outcome targets remain usable in no-result matches when the delivery state is valid. Future innings runs are trained only where innings completion is supported by all-out, the known schedule, a known chase target, or the chasing side being the recorded winner. First-innings par estimates use completed first innings only; unresolved completion is flagged rather than assumed.

Inputs include competition format, innings, over/phase, scheduled and remaining balls, wickets in hand, current score/run rate, and chase target/runs required/rate; batting order is also used for batter outcome models. Player IDs and final match results are not state features. The win-probability model uses `win_team_id` only as its training label, excludes matches without a decisive winner, and trains with equal total weight per match. To estimate a delivery-level team-state WPA proxy, the held-out model predicts the batting team's probability immediately before and after the delivery using the corresponding valid states.

Predictions use up to five-fold `GroupKFold` validation grouped by `match_id` (five folds when at least five matches are available). Each model's prediction for a delivery is made by a fold that did not train on any delivery from that match. The fold's training matches also provide the first-innings par score used in first-innings pressure calculations. `model_validation.csv` reports out-of-fold MAE/RMSE/R² for continuous expectations and Brier score/log loss/AUC for probability models.

The expected remaining-runs model also estimates wicket value. It predicts future team runs from the observed state and from a counterfactual state with one fewer wicket in hand. The non-negative difference is the run value assigned to a dismissal.

### 4.5 Pressure index

Each component is clipped to the range 0–1:

- **Stage:** legal balls already bowled divided by scheduled legal balls.
- **Resource pressure:** wickets already lost divided by nine.
- **First-innings run-rate pressure:**  
  `clip((training-fold par - projected innings total) / (0.20 × training-fold par), 0, 1)`  
  Projected total is current score plus expected remaining team runs.
- **Chase run-rate pressure:**  
  `clip((runs required / max(expected remaining runs, 1) - 1) / 0.50, 0, 1)`
- **Chase leverage:**  
  `clip(1 - abs(runs required - expected remaining runs) / max(runs required, expected remaining runs, 1), 0, 1)`

Leverage is not available in the first innings, so it is not imputed there. Baseline weights are:

| Innings | Run-rate pressure | Resource pressure | Leverage | Stage |
| --- | ---: | ---: | ---: | ---: |
| First | 0.50 | 0.30 | Not available | 0.20 |
| Chase | 0.35 | 0.20 | 0.25 | 0.20 |

The baseline pressure cutoff is **0.55**. A delivery at or above the cutoff is a pressure opportunity. These are explicit modeling choices, not official cricket definitions; alternate weights and cutoffs are tested in the sensitivity outputs.

Sensitivity scenarios use equal, rate-focused, and resource-focused weights; cutoffs of 0.50 and 0.65; and pressure-opportunity/match eligibility thresholds of 20/5 and 50/12. Each scenario is compared with the baseline.

### 4.6 Player contributions and final ranking

Metrics are calculated separately for pressure and normal deliveries and normalized per 100 batter opportunities or bowling deliveries:

- **Batting scoring:** runs above expected, actual/expected strike rate, fours/sixes, and boundary percentage versus expected. Strike-rate impact is the same runs residual expressed per 100 opportunities, so it is reported but not counted again in the composite.
- **Wicket preservation:** expected dismissal probability × wicket run value, minus the actual dismissal cost. The actual dismissed player receives the cost, including a non-striker in a run-out.
- **Bowling control:** runs saved versus expectation, dot-ball percentage versus expectation, and boundary prevention versus expectation.
- **Wicket impact:** `(actual credited wicket - expected credited-wicket probability) × wicket run value`. Uncredited dismissals do not count as bowler wickets.
- **WPA diagnostic:** model-based batting-team `win probability after - before` is retained as a delivery-level team-state proxy. It is not assigned to a batter or bowler and is not included in the individual composite. This avoids treating a team-state change as identifiable individual causal value; no-result WPA events are also excluded from individual rankings.

The code standardizes metrics within format, role, and phase. Related metrics are grouped: batting runs and boundary-rate residuals form one scoring dimension; bowling run-saved, dot-rate, and boundary-prevention metrics form one control dimension. PCA first-component loading magnitudes, normalized to sum to one, learn weights within those correlated groups and across scoring/control and wicket dimensions. The pressure and normal scores use the same weights. `component_weights.csv` records the resulting weights and explained variance. An equal-weight version is run as a robustness scenario.

**Clutch uplift** is the pressure composite minus the normal-condition composite. It is reported as its own measure and is not added as an extra ranking bonus. The pressure composite is reliability-adjusted using:

```text
pressure opportunities / (pressure opportunities + 50)
× min(1, pressure matches / 8)
```

It is further adjusted by match consistency (`0.5 + 0.5 × share of pressure matches with positive net contribution`) and season stability (`0.75 + 0.25 × season stability`). Overall score is the square-root-opportunity-weighted average of format/role/phase scores, with each context's opportunity weight capped at 400.

Baseline eligibility requires at least **30 pressure opportunities**, **8 pressure matches**, and **30 normal opportunities**. Batters' opportunities are batter opportunities as defined above; bowlers' opportunities are deliveries bowled. Players who did not bat or bowl have no opportunities and are not given artificial zeroes. If fewer than ten players meet the criteria, the output contains only eligible players.

## 5. Output files

| File | Contents |
| --- | --- |
| `top_10_batters.csv` | Up to ten eligible batters, ranked by the reliability-adjusted composite, with pressure/normal component summaries. |
| `top_10_bowlers.csv` | Up to ten eligible bowlers with pressure/normal run, dot, boundary, and wicket summaries. |
| `phase_leaderboards.csv` | Eligible batter and bowler leaderboards by powerplay, middle, and slog phase, including component metrics. |
| `player_metrics.csv` | Overall and format/role/phase metrics, including actual/expected rates, pressure/normal composite scores, clutch uplift, sample counts, consistency, and stability. It does not contain individual WPA contributions. |
| `component_weights.csv` | Learned PCA loading weights and explained variance; equal-weight results are available in the sensitivity rankings. |
| `pressure_deliveries.csv` | One row per selected regular-innings delivery, with state-validity and innings-completion flags, observed/expected event metrics, before/after win probabilities, delivery-level team-state WPA proxy and attribution status, pressure components/index, outcome, and fold number. |
| `match_performance.csv` | Player/context performance aggregated by match, useful for checking match-to-match consistency. |
| `consistency.csv` | Overall player-level sample, match consistency, season stability, and eligibility metrics. |
| `format_comparison.csv` | Reliable-player summary comparisons between IPL and T20I, kept in format-specific units. |
| `sensitivity_rankings.csv` | All eligible rankings under alternative weights, pressure cutoffs, and sample thresholds. |
| `ranking_stability.csv` | Top-10 overlap/Jaccard and Spearman rank correlation versus the baseline for each sensitivity scenario. |
| `model_validation.csv` | Match-grouped out-of-fold regression and probability-model diagnostics. |
| `data_quality.csv` | Data-join, duplicate, extras, wicket-attribution, DLS target/schedule gaps, state exclusions, incomplete innings, excluded future-run/par labels, retained delivery outcomes, and WPA attribution diagnostics. |
| `source_schema_audit.csv` | Row/column counts, column names, and missing-value summary for each supplied CSV. |
| `methodology_report.md` | Generated results, formulas, assumptions, validation, sensitivity, and limitations. |
| `plots/top_rankings.png` | Batter and bowler ranking charts. |
| `plots/pressure_vs_normal.png` | Pressure versus normal-condition contribution by format. |
| `plots/reliability_vs_performance.png` | Pressure opportunity count versus final score. |
| `plots/pressure_heatmap.png` | Mean pressure by format, innings, and phase. |

## 6. Interpreting results and limitations

- Scores are relative, normalized rankings—not probabilities of success or causal effects.
- Pressure weights, component scales, cutoff, role groups, and eligibility thresholds are choices. Check `ranking_stability.csv` and `sensitivity_rankings.csv` before relying on a narrow rank difference.
- Individual delivery outcomes are noisy. The report includes diagnostics for each run, event-probability, and win-probability model. Wicket values and win-probability changes are estimates.
- There are no published ball-by-ball win probabilities; the delivery-level WPA proxy is inferred from decisive match outcomes and is not attributed to individuals. The source also lacks official revised DLS targets/schedules for some matches, explicit no-ball subtypes for `other`, and contextual factors such as weather and player intent. Affected DLS states and unresolved innings completion are explicitly excluded from the relevant state-dependent calculations.
- IPL and T20I raw contribution rates are not directly interchangeable. The overall ranking normalizes by format and role before combining contexts; `format_comparison.csv` reports each format separately.
- Sixteen selected match records in the inspected data have no ball-by-ball rows, so no player performance can be calculated for those matches. One wicket lacks a dismissed-player ID and cannot incur a batter dismissal cost.

For the run-specific counts, rankings, validation metrics, and audit results, see `output/methodology_report.md` and the CSV files it references.
