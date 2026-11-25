# Summary: Do NFL Combine drills predict which rookies out-perform their draft position?

**Research question.** Do NFL Combine drills predict which rookies out-perform their draft position, and does the answer
differ by position?

**Goal.** Sort a player into "likely over-performer" or "likely under-performer" **before the draft**, using only what a
team knows then. That helps a team plan its later-round picks and target players who can grow into the roster.

**Short answer.** At **wide receiver**, yes. Using combine results and age at the draft, the forecast sorted **81%** of
the unseen 2025 receivers correctly, against 51% for guessing. Receivers who jump better than their class are the
over-performers. **Age matters at every position:** over-performers were younger on draft day. Offensive line, defensive
line and tight end beat guessing by 6–16 points, but that could still be luck. Defensive backs are at guessing level.

The full report, with every visual, is [PERFORMANCE_CLASSIFIER.md](PERFORMANCE_CLASSIFIER.md).

---

## 1. Dataset

NFL Big Data Bowl 2027 (Kaggle competition `nfl-big-data-bowl-2027`).

| File | Used for |
|---|---|
| `combine_results.csv` | **Forecast input:** 40-yard dash, 10-yard split, vertical, broad jump, three-cone, short shuttle, bench, weight, arm length |
| `players.csv` | **Forecast input:** birth date (age at the draft). Also college conference (tested, not kept), position, draft year, round and pick (undrafted = pick 260; tests and charts only) |
| `player_play.csv` | First-season snaps and production. **Used only to decide who over-performed** |
| `player_career_successes.csv` | Career games started. **Used only to decide who over-performed** |
| `games.csv` | Season and team games, for snaps per team game |

**Scope:** 510 players from the 2023, 2024 and 2025 draft classes at five combine positions:
WR (107), TE (42), OL (121), DL (118), DB (122).

**Data limits**
- In `player_play.csv`, receivers and tight ends appear only on plays where they ran a route. Tight-end blocking
  therefore counts only through the career-starts half of the success score.
- Many players skip drills. Only 22 tight ends ran the three-cone or shuttle, and only 11 benched.
- 2025 rookies have one season, so their starts per season equal their 2025 starts.

## 2. Method

**Combine traits.** Each drill becomes a z-score against the player's own position and draft class (positive = better
than his class; timed drills are flipped). Drills are averaged into six traits: speed, jumping, speed + jumping, change
of direction, strength and size.

**Age at the draft.** Years from birth to 27 April of the draft year.

**Success score (the outcome, never an input).**

`success = 0.5 × z(first-season play score) + 0.5 × z(career starts per season)`

- First-season play score = 0.5 × z(snaps per team game) + 0.5 × the average z of the position's production rates
  (for example targets and yards per route for receivers, pressure and sack rates for defensive linemen).
- Career starts per season = games started ÷ seasons since the draft, which keeps the three classes comparable.
- Each half is compared within position and draft class, so 0 is a typical player.

**Over/under label.** Over-performer = success score above the **median of the 2023–24 classes** at his position. 2025
rookies are measured against that same earlier median.

**Hypothesis tests** (per position × trait): Spearman ρ with bootstrap range; Mann-Whitney U, Welch t and permutation
tests; ANOVA and Kruskal-Wallis across tester thirds; logistic regression over ~ trait + log(pick); Holm correction.
Verdict: **Clear** / **Partial** / **No**.

**Pre-draft forecast.** Logistic regression (standardised inputs, C = 1) on the six combine traits, a missing-drill count
and age at the draft. No draft pick, play data or career starts.
- **Choosing the model:** logistic regression, random forest and XGBoost, each with four input sets (combine; + college
  conference; + age; + both), were scored by repeated 5-fold cross-validation inside 2023–24. Logistic regression scored
  best. Age was kept because it raised the score from 0.60 to 0.70; conference was dropped because it did not help.
- **Train:** 2023 and 2024 classes (25–85 players per position). **Test:** the 2025 class (17–44 per position).
- **Reported:** accuracy against a guess-the-bigger-group baseline, AUC, precision, recall, F1 and bootstrap ranges.

## 3. Results

### Which combine traits mark an over-performer?

| Trait | WR | TE | OL | DL | DB |
|---|---|---|---|---|---|
| Speed | Partial | No | No | Partial | Partial |
| **Jumping** | **Partial** | No | No | Partial | Partial |
| **Speed + jumping** | **Partial** | No | No | Partial | Partial |
| Change of direction | Partial | No | No | Partial | No |
| Strength | Partial (28 tested) | Too few tested | No | No | No |
| Size | No | No | Partial | No | No |

- **WR jumping:** ρ = +0.42. At the same draft spot, each step above the class average multiplies the odds of
  over-performing by **2.2** (Holm p = .054, just missing the bar).
- **DL jumping:** same direction, weaker (ρ = +0.28, odds ratio 1.9, Holm p = .059).
- **DB:** athletic players do better (ρ = +0.26), but teams already draft them earlier; at the same draft spot the odds
  ratio is 0.6.
- **TE:** every link is between −0.09 and +0.10.

### Age at the draft

Over-performers were younger at every position: WR 22.5 vs 23.2, TE 22.2 vs 23.5, OL 22.6 vs 23.4, DL 22.8 vs 23.3,
DB 22.6 vs 23.2 (median age). The link with success runs from −0.28 (DL) to −0.52 (TE).

### How well does the forecast sort the 2025 rookies?

Trained on 2023–24 combine results and age, tested on the 2025 class:

| Position | 2025 players | Guessing | **Accuracy** (range) | AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| **WR** | 37 | 51% | **81%** (68–92%) | 0.85 | 82% (14 of 17) | 78% (14 of 18) | 0.80 |
| OL | 38 | 53% | 68% (53–82%) | 0.81 | 70% | 70% | 0.70 |
| TE | 17 | 59% | 65% (41–88%) | 0.71 | 55% | 86% | 0.67 |
| DL | 44 | 52% | 64% (48–77%) | 0.73 | 73% | 48% | 0.58 |
| DB | 37 | 51% | 54% (38–70%) | 0.63 | 53% | 50% | 0.51 |

- **WR is the one position where the whole likely range sits above guessing.**
- **OL, TE and DL** beat guessing, but their ranges reach a coin flip.
- **DB** sits at guessing level. On combine results alone DB scored 68%, so adding age hurt there on 2025 (a swing of
  five players).

### Checks

- **Algorithms.** Average 2023–24 score (AUC): logistic regression 0.60, random forest 0.58, XGBoost 0.58 on combine
  results. Trees edged ahead at DL, DB and OL by less than 0.05; logistic regression led clearly at WR and TE.
- **College conference.** On its own it changed the score by less than 0.01; on top of age it lowered 2025 accuracy
  (66% → 64%). The SEC supplies the most players at every position except tight end (Big Ten).
- **Success label.** With the same forecast, starts only and play only give similar results. The best result is WR with
  the kept label (81% vs 51%); the highest AUC is WR with starts only (0.90).

### After the draft: late-round receivers

This check uses the real draft round, which is known only after the draft. Among 2025 receivers taken in round 4 or
later, or undrafted, the forecast flagged 10 as likely over-performers and **8 of them were** (80%). Only 11 of those 29
late receivers over-performed overall (38%). Examples it flagged correctly: Tez Johnson (round 7), Jaylin Lane and
Dont'e Thornton (round 4), Isaiah Bond (undrafted).

## 4. Conclusion

1. **Wide receiver is where the combine pays off before the draft.** Jumping, alone or with speed, separates receivers
   who out-perform their draft slot. With age, the forecast sorted 81% of the unseen 2025 class correctly.
2. **Age is the strongest single pre-draft signal.** Younger prospects over-performed at every position.
3. **The answer depends on position.** Offensive line, defensive line and tight end show gains that could be luck.
   Defensive backs show none.
4. **Draft position already reflects much of what the combine shows.** At defensive back, athletic players do better,
   but they are already drafted earlier.
5. **For later rounds:** use the forecast to rank late-round receivers. At the other positions, treat it as a
   tie-breaker at most.

## 5. Limitations

- Small samples: 17–44 players per position in the 2025 test, so accuracy ranges are wide.
- One test year, and 2025 rookies have one season.
- Tight-end blocking is invisible in the play data.
- The input rule ("must not lower 2025 accuracy") looks at the test class, so 2025 numbers are slightly optimistic.

## 6. Future directions

- **Tight end study:** add run-blocking snaps, or the share of plays with two or more tight ends, to test whether teams
  value blocking over athleticism.
- **More classes:** add the 2026 class as a second test year.
- **Tracking data:** replace stopwatch times with top speed and acceleration from `combine_tracking.csv`.
- **Defensive line:** split edge rushers from interior linemen.
- **Age:** separate early entrants from fifth-year seniors.

## 7. Reproduce

```
pip install -r requirements.txt
python src/combine_rookie_classifier.py            # downloads the data with kagglehub
python src/compare_labels.py                       # the success-label check
```

Outputs in `results/`: `trait_tests.csv`, `model_comparison.csv`, `classifier_results.csv`, `predictions_2025.csv`,
`conference_report.csv`, `age_summary.csv`, `player_labels.csv`, `label_comparison.csv`.
