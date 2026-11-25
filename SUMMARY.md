# Summary

**Question.** Can NFL Combine drills predict which rookies out-perform the typical rookie at their position? Which other
factors help?

**Answer.** At **wide receiver**, yes: 14 of the 17 receivers the forecast flagged over-performed (precision 82%). ACC
was 81% against 51% for guessing. **Age** is the other factor that helps: over-performers were younger at every
position. At the other positions a flag is right 53–73% of the time, so use the forecast as a tie-breaker at most.

The full report with every visual is [PERFORMANCE_CLASSIFIER.md](PERFORMANCE_CLASSIFIER.md).

## Terms

- **Over-performer:** success score above the typical rookie at his position (the 2023–24 median). The success score is
  half first-season play and half career starts per season.
- **Sorting score:** how often the forecast rates a random over-performer above a random under-performer. 0.5 = coin
  flip, 0.7 = fair, 0.8+ = good.
- **Precision:** of the rookies flagged as over-performers, the share who were. The main measure.
- **ACC:** accuracy on the 2025 class, which the forecast never saw.

## Data and method

- 510 players, 2023–25 drafts: WR, TE, OL, DL, DB (NFL Big Data Bowl 2027).
- **Inputs (pre-draft only):** combine traits (speed, jumping, change of direction, strength, size), missing-drill count
  and age at the draft. Play data and career starts are used only for the label.
- **Model:** logistic regression. It had a higher sorting score than random forest and XGBoost (0.60 vs 0.58 on combine
  results). Adding age raised it to 0.70; college conference did not help.
- **Test:** trained on 2023–24, tested once on 2025. A rookie is flagged when his chance is above 50%. A stricter
  precision-first cut-off was tested and did not raise precision.

## Results (2025 class)

| Position | Precision | Flagged and over | ACC | Guessing | Recall |
|---|---|---|---|---|---|
| **WR** | **82%** | 14 of 17 | **81%** | 51% | 78% |
| DL | 73% | 11 of 15 | 64% | 52% | 48% |
| OL | 70% | 14 of 20 | 68% | 53% | 70% |
| TE | 55% | 6 of 11 | 65% | 59% | 86% |
| DB | 53% | 9 of 17 | 54% | 51% | 50% |

- **Combine signal:** receivers who jump better than their class over-perform (odds ×2.2 at the same draft spot). No
  drill separates tight ends or offensive linemen.
- **Age:** median age of over- vs under-performers is 22.5 vs 23.2 for WR, and the gap is similar at every position.
- **Draft position already reflects much of the combine:** first-round picks test above their class.

## Limitations

- Small 2025 groups (17–44 per position) and one test year.
- Tight-end blocking is invisible in the play data.
- The rule for keeping an input looked at 2025 ACC, so the 2025 results are slightly optimistic.

## Reproduce

```
pip install -r requirements.txt
python src/combine_rookie_classifier.py
```
