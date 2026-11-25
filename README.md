# Combine Drills and Rookie Out-Performance (NFL Big Data Bowl 2027)

**Research question:** Do NFL Combine drills predict which rookies out-perform their draft position, and does the
answer differ by position?

**Goal:** sort a player into "likely over-performer" or "likely under-performer" **before the draft**, using only what
a team knows then: combine results and age. That helps a team plan its later-round picks and find players who can grow into the roster.

This repository is a Python code sample for a research project. It covers the 2023–25 draft classes at five positions:
WR, TE, OL, DL and DB (510 players). The pipeline:
1. Turn each combine drill into a trait score against the player's own position and draft class.
2. Score each player's success from first-season play data plus career starts per season. This is used only to label
   players.
3. Label each player as an over- or under-performer against the 2023–24 median at his position.
4. Test each combine trait with several hypothesis tests.
5. Choose a forecast (algorithm and inputs) by cross-validation on the 2023–24 classes, then test it once on the
   unseen 2025 class.

**Headline:** on the unseen 2025 class, combine results plus age at the draft sorted **81% of wide receivers**
correctly, against 51% for guessing (likely range 68–92%). Of the 17 receivers it flagged as over-performers, 14 were.
Receivers who jump better than their class are the over-performers, and **younger prospects over-performed at every
position**. Offensive line (68%), tight end (65%) and defensive line (64%) beat guessing but could still be luck;
defensive backs (54% vs 51%) did not. See [SUMMARY.md](SUMMARY.md) for the short write-up and
[PERFORMANCE_CLASSIFIER.md](PERFORMANCE_CLASSIFIER.md) for the full report.

## Repository contents

| Path | What it is |
|---|---|
| [`src/combine_rookie_classifier.py`](src/combine_rookie_classifier.py) | The full analysis in one script: Kaggle download → combine traits → success score → label → hypothesis tests → model choice → 2025 forecast → conference and age reports |
| [`src/compare_labels.py`](src/compare_labels.py) | Check: does the choice of success label (play + starts, starts only, play only) change the forecast? |
| [`SUMMARY.md`](SUMMARY.md) | Short write-up: dataset, method, results, conclusion |
| [`PERFORMANCE_CLASSIFIER.md`](PERFORMANCE_CLASSIFIER.md) | Full report: question, data, method, every result with screenshots and hover notes, checks, limitations, future directions |
| [`aichatrecap.md`](aichatrecap.md) | How AI (Claude) was used in this project, and which decisions were mine |
| `results/` | Output tables written by the script |
| `docs/screenshots/` | Dashboard screenshots used in `PERFORMANCE_CLASSIFIER.md` |
| `requirements.txt` | Python packages |

**Output tables in `results/`**
- `trait_tests.csv`: every hypothesis test for each position and trait, with the verdict.
- `model_comparison.csv`: each algorithm and input set, with its 2023–24 cross-validated score and 2025 results.
- `classifier_results.csv`: the kept forecast's 2025 accuracy, guessing baseline, AUC, precision, recall, F1 and the
  confusion-matrix counts by position.
- `predictions_2025.csv`: each 2025 rookie's forecast chance of over-performing, and whether he did.
- `conference_report.csv`: players, average starts and over-performer share by college conference and position.
- `age_summary.csv`: age at the draft by position, for over- and under-performers.
- `player_labels.csv`: each player's traits, age, success score and over/under label.
- `label_comparison.csv`: the forecast under three success labels (from `src/compare_labels.py`).

## Dashboards

Both are hosted on claude.ai.
- **Performance Classifier:** this repository's question. The approach kept, over-performer signals, the 2025 forecast
  with confusion grids, the algorithm comparison, draft pick against
  success score, trait links with career starts, how each round tested, and college conferences.
  https://claude.ai/artifact/FZMvRW3nZQHZsP9oYSzZnN. A guide with screenshots is in
  [PERFORMANCE_CLASSIFIER.md](PERFORMANCE_CLASSIFIER.md).
- **BDB 2027 Position Scout:** the wider exploration that led to the question.
  https://claude.ai/artifact/KAkF3erffbWJx1eHDmrhWC

## Getting the data

The data comes from the Kaggle competition [`nfl-big-data-bowl-2027`](https://www.kaggle.com/competitions/nfl-big-data-bowl-2027).
It isn't included here. You need a Kaggle account, the competition rules accepted on its page, and an API token
(`~/.kaggle/kaggle.json`, or the `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables).

**Option A: kagglehub (default).** The script downloads the data itself:

```bash
pip install -r requirements.txt
python src/combine_rookie_classifier.py
```

It calls `kagglehub.competition_download("nfl-big-data-bowl-2027")` and finds the folder holding `players.csv`.

**Option B: Kaggle CLI.** Download once, then point the script at the folder:

```bash
kaggle competitions download -c nfl-big-data-bowl-2027 -p data
unzip data/nfl-big-data-bowl-2027.zip -d data
python src/combine_rookie_classifier.py --data-dir data
```

## Running

```bash
python src/combine_rookie_classifier.py [--data-dir FOLDER] [--out results]
```

It takes a few minutes, mostly for the bootstrap and permutation tests. It prints the over/under counts, a verdict table
for each position and trait, the model comparison and the 2025 forecast results, then writes the CSVs to `results/`.
Random seeds are fixed, so reruns give the same numbers.

## Method in brief

- **Forecast inputs (pre-draft only):** each drill becomes a z-score against the player's own position and draft
  class, averaged into speed, jumping, speed + jumping, change of direction, strength and size; plus a missing-drill
  count and age at the draft.
- **Success score (outcome only):** 0.5 × first-season play score (snaps per team game and production rates) + 0.5 ×
  career starts per season. Both halves are compared within position and draft class.
- **Label:** over-performer = above the 2023–24 classes' median at his position. 2025 is measured against the same median.
- **Hypothesis tests:** Spearman with bootstrap range, Mann-Whitney U, Welch t, permutation test, ANOVA and
  Kruskal-Wallis across tester thirds, and logistic regression with draft pick held fixed, all with Holm correction.
- **Model choice:** logistic regression, random forest and XGBoost (algorithms used in NFL draft and Big Data Bowl
  projects), each with and without college conference and age, scored by repeated 5-fold cross-validation on 2023–24.
  Logistic regression with combine + age was kept; conference did not help.
- **Forecast:** no draft pick, play data or career starts. Trained on 2023–24 and tested once on 2025.

## Limitations

- Small samples: 17–44 players per position in the 2025 test, so accuracy ranges are wide.
- 2025 rookies have one season, so their starts per season come from one year.
- Receivers and tight ends appear in `player_play.csv` only on route-running plays, so tight-end blocking counts only
  through career starts.
- Many players skip drills. Missing drills are set to the class average and counted as a feature.
- One test year. The input rule ("must not lower 2025 accuracy") looks at the 2025 class, so those numbers are slightly
  optimistic.
- Most of the gain over combine results alone comes from age. Defensive backs did better on 2025 without age.

## How AI was used

The analysis was done with Claude (Anthropic) as a coding and statistics assistant. I set the question, the scope, the
success definition, the over/under rule, the pre-draft design, and asked for the conference, age and algorithm tests.
Claude wrote and ran the code, chose the tests and model settings, and drafted the dashboards and documents for my review. Details are in [aichatrecap.md](aichatrecap.md).
