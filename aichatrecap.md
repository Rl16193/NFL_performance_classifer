# AI chat recap: how the analysis was built with Claude

This project was done in conversation with Claude (Anthropic), running in the Claude desktop app with access to a local copy of the
competition data. This file records what was asked, what the AI did, and where the human set the direction.

## Roles

| Who | What |
|---|---|
| **Analyst (me)** | Chose the positions and the research question, set the success definition and the over/under rule, rejected or redirected approaches, decided what goes in the report and the dashboard. |
| **Claude** | Explored the data, wrote and ran the Python, chose and ran the statistical tests, built the dashboards, flagged data limitations, and drafted the write-up for my review. |

## How the work moved, step by step

1. **Scope.** I limited the study to five combine positions: WR, TE, OL, DL and DB (510 players, 2023–25 draft classes),
   and told Claude not to add other roles.
2. **First pass: combine vs career starts.** Claude compared each drill with the player's own position and draft class
   (z-scores), then correlated them with starts per season, both on their own and with draft pick held constant.
   The first results focused on "explosiveness".
3. **My correction: look at every trait, write for managers.** I asked why only explosiveness, and asked for plain English
   instead of terms like "raw p" or "partial correlation". Claude reran the tests for speed, jumping, change of direction,
   strength and size, and rewrote the dashboard for non-technical readers.
4. **My challenge: is there enough data for a prediction model?** Claude agreed the sample is small (17–122 players per
   position) and suggested analyses that fit small samples, including over- and under-performers.
5. **Choosing the question.** I picked over/under-performers. My first brief was tight-end specific. I then rejected
   Claude's first plan, which used an "expected starts" model, and set the rules myself:
   - over-performer = above the median, using real values, not an expected-value model;
   - 2025 rookies are labelled against the earlier classes' results;
   - test all positions and see which gives a clear answer;
   - **no career starts as the outcome**, only combine and play-by-play data.
6. **Building a success measure without career starts.** Claude built a first-season score from `player_play.csv`
   (playing time + production per opportunity). Along the way it found and reported three data issues:
   - team play totals are incomplete (a play appears only if a study player was on the field), so it switched from snap
     share to snaps per team game;
   - WR and TE rows exist only on route-running plays, so tight-end run blocking is invisible;
   - some statistical tests crashed on constant groups, which it fixed.
7. **Hypothesis tests.** Claude ran Spearman correlations with bootstrap ranges, Mann-Whitney U with rank-biserial effect
   sizes, Welch t-tests, permutation tests, ANOVA and Kruskal-Wallis on tester thirds, logistic regression with draft pick
   held fixed, Fisher exact tests and Holm correction.
8. **My challenge: what is the accuracy? I can't rely on hypothesis tests alone.** Claude pointed to the leave-one-out
   classifier it had already run, then added a stricter test: train on the 2023–24 classes and predict the unseen 2025 class.
9. **Packaging for GitHub.** I set the final question. Claude consolidated the work into one script
   (`src/combine_rookie_classifier.py`) that downloads the data with kagglehub and reproduces every result. It also
   wrote this recap, the README and the summary, and built the Performance Classifier dashboard.
10. **My redirect: a pre-draft forecast.** I made the goal explicit: classify a player as an over- or under-performer
    *before* the draft, to help with later-round picks and find players who can grow the team. That changed the design:
    - I removed the leave-one-out checks. The focus is the 2025 predictions.
    - I removed a late-round add-on analysis Claude had started without being asked. Only what I asked for stays in.
    - The forecast uses **combine results only**. No draft pick and no play data are inputs.
11. **Choosing the label.** I asked what career stats were used for. Then I decided play data and career starts are
    used only to label who over-performed, never as forecast inputs. I chose a label that combines first-season play
    data with **career starts per season**, which keeps the 2023, 2024 and 2025 classes comparable.
12. **Rebuild.** Claude reran the study with the new label and forecast, rebuilt the dashboard and took screenshots of
    each visual.
13. **A projected draft tier, tried and dropped.** I asked whether a projected draft position could be used before the
    draft. Claude tried a projected tier built from combine results. It predicted the real pick only weakly (link
    0.25–0.39) and did not improve the forecast, so I asked to revert to the best pre-draft forecast without it.
14. **Removing what I hadn't asked for.** A "draft-position forecast" verdict section had stayed on the dashboard after I
    asked for it to be removed. I pointed this out and it was taken out.
15. **Does the label matter?** I asked which success label gives the best result. Claude wrote `src/compare_labels.py`
    to compare play + starts, starts only and play only. Wide receiver was strongest under all three; I kept play +
    starts.
16. **Algorithms from other projects.** I asked to compare algorithms, using ones that NFL draft and Big Data Bowl
    projects found best. Claude searched for published work (NFL Next Gen Stats' XGBoost draft model, a Wharton
    gradient-boosting receiver study, a Berkeley combine classifier, arXiv 2303.05774 and Big Data Bowl entries) and
    compared logistic regression, random forest and XGBoost by cross-validation on 2023–24. Logistic regression won.
17. **College conference.** I asked whether conference should be considered. Claude tested it as an input; a first run
    put every player in "Other" because the data uses full conference names, which it found and fixed. Conference did
    not help, so it is reported (players and average starts by conference and position) but not used.
18. **Age.** I asked whether age was considered. It was not, so Claude tested age at the draft as an input. It was the
    biggest improvement found (average score 0.60 → 0.70) and was kept. Claude flagged that defensive backs did worse
    on 2025 with age.
19. **Clearer results and a full report.** I found the forecast table confusing, so it became confusion grids with
    accuracy, precision, recall, F1 and guessing, each with a plain meaning. I asked for a short overview of only what
    was kept and why. Claude then rewrote [PERFORMANCE_CLASSIFIER.md](PERFORMANCE_CLASSIFIER.md) as the full report.

20. **A sharper question, precision first.** I reworded the question to "Can NFL Combine drills predict which rookies
    out-perform the typical rookie at their position? Which other factors help?" The label line stayed the 2023–24
    median, now called the typical rookie. I asked for precision to get more weight. Claude pointed out that the error I
    described (missing a real over-performer) is measured by recall, and tested a precision-first cut-off on 2023–24.
    It didn't raise precision, so the forecast stayed as it was and precision now leads the reporting. I also asked
    for sorting score to be explained, "ACC" in place of "2025", a filter note before every visual and a trimmed report.

## What I checked myself

- The research question, the over/under rule, the combined label and the choice to keep play data and career starts
  out of the forecast were my decisions.
- I reviewed each set of results and asked for changes when the focus or the language was wrong.
- The final scripts were rerun from scratch, and the numbers in the documents were checked against `results/`.

## Limits of AI use to keep in mind

- Claude chose the specific statistical tests and model settings. Logistic regression was Claude's starting default,
  not copied from past Big Data Bowl projects; it was later checked against random forest and XGBoost and kept.
- The rule for keeping an input (raise the 2023–24 score by more than 0.01 and not lower 2025 accuracy) was Claude's.
  Because it looks at 2025, the 2025 results are slightly optimistic.
- The play + starts label was my choice. The median split was agreed between us.
- The AI's write-ups were drafts. Claims in `SUMMARY.md` were checked against the output tables in `results/`.
- No competition data is stored in this repository.
