"""
Do NFL Combine drills predict which rookies out-perform their draft position, and does the answer differ by position?

Data: NFL Big Data Bowl 2027 (Kaggle competition "nfl-big-data-bowl-2027"), 2023-25 draft classes,
positions WR, TE, OL, DL, DB.

Pipeline
  1. Get the data      kagglehub download (or --data-dir pointing at an already-downloaded copy).
  2. Combine traits    each drill as a z-score within combine position x draft year (positive = better than the class),
                       grouped into speed, jumping, speed + jumping, change of direction, strength and size.
  3. First-season play score (player_play.csv): 0.5 * z(snaps per team game) + 0.5 * mean z(position production
                       rates), z within position x season.
  4. Success label     success = 0.5 * z(first-season play score) + 0.5 * z(career starts per season), z within
                       position x draft class. Over-performer = success above the 2023-24 classes' median at his
                       position; the 2025 class is labelled against that same median. Outcome only, never an input.
  5. Hypothesis tests  per position x trait: Spearman (+ bootstrap range), Mann-Whitney (+ rank-biserial), permutation
                       test, ANOVA / Kruskal-Wallis across tester thirds, and logistic regression over ~ trait + log(pick)
                       ("does the trait still matter at the same draft spot?"). Holm correction over the 3 main traits.
  6. Forecast         pre-draft forecast on information known before the draft: combine results, and optionally the
                       college conference group and age at the draft (no draft pick, no play data, no career stats).
                       Logistic regression, random forest and XGBoost are compared. The algorithm and the inputs are
                       chosen by cross-validated AUC inside the 2023-24 classes, then tested once on the unseen 2025 class.
  7. Reports           players, average starts and over-performer share by conference group and position; age summary.

Usage
  pip install -r requirements.txt
  python src/combine_rookie_classifier.py                     # downloads with kagglehub (needs Kaggle credentials)
  python src/combine_rookie_classifier.py --data-dir <folder>  # use a local copy of the competition CSVs
Outputs go to results/ (CSV tables) and are summarised on screen.
"""
import argparse
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
from xgboost import XGBClassifier

COMPETITION = "nfl-big-data-bowl-2027"
POSITIONS = ["WR", "TE", "OL", "DL", "DB"]
SEASONS = [2023, 2024, 2025]
UNDRAFTED_PICK = 260
RNG = np.random.default_rng(2027)

# drill -> True when a lower time is better (sign is flipped so a positive z always means "better")
DRILLS = {"forty": True, "ten_yd_split": True, "vertical": False, "broad_jump": False, "three_cone": True,
          "short_shuttle": True, "bench_reps": False, "combine_weight": False, "arm_length": False}
TRAITS = {"speed": ["forty_z", "ten_yd_split_z"],
          "jumping": ["vertical_z", "broad_jump_z"],
          "change_of_direction": ["three_cone_z", "short_shuttle_z"],
          "strength": ["bench_reps_z"],
          "size": ["combine_weight_z", "arm_length_z"]}
MAIN_TRAITS = ["speed", "jumping", "speed_jumping"]
SECOND_TRAITS = ["change_of_direction", "strength", "size"]
MODEL_TRAITS = list(TRAITS)

# production rates that make up the play-based score, by position (see build_rookie_success)
PRODUCTION = {"WR": ["tgt_per_route", "yds_per_route", "separation", "yac_oe"],
              "TE": ["tgt_per_route", "yds_per_route", "separation", "yac_oe"],
              "OL": ["neg_pa_rate", "neg_sa_rate", "ttp_allowed"],
              "DL": ["press_rate", "sacks_rate", "tfl_rate", "neg_getoff"],
              "DB": ["tk_rate", "tfl_rate", "ff_rate"]}


# ----------------------------------------------------------------------------------------------- 1. data
def get_data_dir(data_dir):
    """Return a folder holding the competition CSVs, downloading them with kagglehub if none is given.

    kagglehub needs Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY) and the
    competition rules accepted on kaggle.com. CLI alternative: kaggle competitions download -c nfl-big-data-bowl-2027
    """
    if data_dir:
        return data_dir
    import kagglehub
    path = kagglehub.competition_download(COMPETITION)
    # the download may unpack into a sub-folder; find the one that holds players.csv
    for root, _, files in os.walk(path):
        if "players.csv" in files:
            return root
    raise FileNotFoundError(f"players.csv not found under {path}")


def load(d):
    players = pd.read_csv(f"{d}/players.csv")
    combine = pd.read_csv(f"{d}/combine_results.csv").drop(columns="draft_year", errors="ignore")
    df = players.merge(combine, on="nfl_id")
    df = df[df.combine_position.isin(POSITIONS)].copy()
    df["pick"] = df.draft_overall_pick.fillna(UNDRAFTED_PICK)
    games = pd.read_csv(f"{d}/games.csv")
    games = games[games.season_type == "REG"]
    plays = pd.read_csv(f"{d}/player_play.csv", low_memory=False).merge(games[["game_id", "season"]], on="game_id")
    career = pd.read_csv(f"{d}/player_career_successes.csv")[["nfl_id", "career_games_started"]]
    return df, games, plays, career


# ----------------------------------------------------------------------------------------------- 2. combine traits
def add_combine_traits(df):
    """Drill z-scores relative to the player's own position and draft class, then trait averages."""
    for drill, lower_is_better in DRILLS.items():
        g = df.groupby(["combine_position", "draft_year"])[drill]
        df[f"{drill}_z"] = (df[drill] - g.transform("mean")) / g.transform("std") * (-1 if lower_is_better else 1)
    for trait, cols in TRAITS.items():
        df[trait] = df[cols].mean(axis=1)
    sj = df[["forty_z", "vertical_z", "broad_jump_z"]]
    df["speed_jumping"] = sj.mean(axis=1).where(sj.notna().sum(axis=1) >= 2)  # needs 2+ of the 3 drills
    return df


# ----------------------------------------------------------------------------------------------- 3. rookie success
def build_rookie_success(df, games, pp):
    """Play-based score per player-season; return each player's draft-year (rookie) score.

    Playing time is snaps per team game, because team play totals in player_play are incomplete (a play is
    present only when a study player was on the field). Note: WR/TE rows exist only on plays where they ran
    a route, so tight-end run blocking is not captured.
    """
    num = lambda c: pd.to_numeric(pp[c], errors="coerce")
    flag = lambda c: pp[c].astype(str).str.upper().map({"TRUE": 1, "FALSE": 0})
    pp = pp.assign(off=(pp.team_abbr == pp.possession_team).astype(int), route=pp.route_ran.notna().astype(int),
                   tgt=flag("target").fillna(0), rec_y=num("rec_yards").fillna(0),
                   pblock=num("pass_rushers_encountered").notna().astype(int), pa=flag("pressure_allowed"),
                   sa=num("sack_allowed"), ttpa=num("time_to_pressure_allowed"),
                   yacoe=num("yards_after_catch") - num("expected_yards_after_catch"),
                   sep=num("separation_at_pass_forward"), press=num("time_to_pressure").notna().astype(int),
                   sk=num("sack").fillna(0), tfl=flag("tackle_for_loss").fillna(0),
                   tk=num("tackle").fillna(0) + num("assist").fillna(0), ff=num("caused_forced_fumble").fillna(0),
                   getoff=num("player_get_off"))
    g = pp.groupby(["nfl_id", "season"])
    ps = pd.DataFrame({
        "snaps": g.size(), "off_snaps": g.off.sum(), "team": g.team_abbr.agg(lambda s: s.value_counts().index[0]),
        "routes": g.route.sum(), "targets": g.tgt.sum(), "rec_yards": g.rec_y.sum(), "pblock": g.pblock.sum(),
        "pa_rate": g.pa.mean(), "sa_rate": g.sa.mean(), "ttpa": g.ttpa.mean(), "ttpa_n": g.ttpa.count(),
        "yacoe": g.yacoe.mean(), "yacoe_n": g.yacoe.count(), "sep": g.sep.mean(), "sep_n": g.sep.count(),
        "press": g.press.sum(), "sacks": g.sk.sum(), "tfl": g.tfl.sum(), "tk": g.tk.sum(), "ff": g.ff.sum(),
        "getoff": g.getoff.mean(), "getoff_n": g.getoff.count()}).reset_index()

    team_games = (pd.concat([games[["season", "home_team_abbr"]].set_axis(["season", "team"], axis=1),
                             games[["season", "visitor_team_abbr"]].set_axis(["season", "team"], axis=1)])
                  .groupby(["season", "team"]).size().rename("team_games").reset_index())
    ps = ps.merge(team_games, on=["season", "team"], how="left")
    ps["snaps_per_team_game"] = ps.snaps / ps.team_games

    # production rates, each with a minimum number of opportunities
    routes = ps.routes.where(ps.routes >= 20)
    ps["tgt_per_route"], ps["yds_per_route"] = ps.targets / routes, ps.rec_yards / routes
    ps["separation"] = ps.sep.where(ps.sep_n >= 20)
    ps["yac_oe"] = ps.yacoe.where(ps.yacoe_n >= 5)
    pass_block = ps.pblock >= 50
    ps["neg_pa_rate"], ps["neg_sa_rate"] = -ps.pa_rate.where(pass_block), -ps.sa_rate.where(pass_block)
    ps["ttp_allowed"] = ps.ttpa.where(ps.ttpa_n >= 5)
    def_snaps = (ps.snaps - ps.off_snaps).where(ps.snaps - ps.off_snaps >= 50)
    for c in ["press", "sacks", "tfl", "tk", "ff"]:
        ps[f"{c}_rate"] = ps[c] / def_snaps
    ps["neg_getoff"] = -ps.getoff.where(ps.getoff_n >= 20)

    # every player x every season from his draft year on; no rows that season = 0 snaps
    grid = df[["nfl_id", "draft_year", "combine_position"]].merge(pd.DataFrame({"season": SEASONS}), how="cross")
    grid = grid[grid.season >= grid.draft_year]
    ps = grid.merge(ps, on=["nfl_id", "season"], how="left")
    ps["snaps_per_team_game"] = ps.snaps_per_team_game.fillna(0)

    z = lambda s: (s - s.mean()) / s.std()
    ps["score"] = np.nan
    for pos, parts in PRODUCTION.items():
        for season in SEASONS:
            i = (ps.combine_position == pos) & (ps.season == season)
            z_time = z(ps.loc[i, "snaps_per_team_game"])
            z_prod = ps.loc[i, parts].apply(z).mean(axis=1, skipna=True)
            ps.loc[i, "score"] = np.where(z_prod.notna(), 0.5 * z_time + 0.5 * z_prod, z_time)
    rookie = ps[ps.season == ps.draft_year].set_index("nfl_id")
    return rookie[["score", "snaps_per_team_game"]].rename(columns={"score": "rookie_score",
                                                                    "snaps_per_team_game": "rookie_snaps_pg"})


# ----------------------------------------------------------------------------------------------- 4. label
def add_labels(df, career):
    """Success score = half first-season play score, half career starts per season, each as a z-score within
    position x draft class. Over-performer = success above the 2023-24 classes' median at his position; the 2025 class
    is labelled against that same median. These are outcomes only: never model inputs."""
    df = df.merge(career, on="nfl_id", how="left")
    df["career_games_started"] = df.career_games_started.fillna(0)
    df["seasons_played"] = max(SEASONS) - df.draft_year + 1
    df["starts_per_season"] = df.career_games_started / df.seasons_played
    z = lambda s: (s - s.mean()) / s.std()
    grp = df.groupby(["combine_position", "draft_year"])
    df["success"] = 0.5 * grp.rookie_score.transform(z) + 0.5 * grp.starts_per_season.transform(z)
    cut = df[df.draft_year <= 2024].groupby("combine_position").success.median()
    df["over"] = (df.success > df.combine_position.map(cut)).astype(int)
    return df


# ----------------------------------------------------------------------------------------------- 5. hypothesis tests
def bootstrap_rho(x, y, b=2000):
    idx = RNG.integers(0, len(x), (b, len(x)))
    return np.nanpercentile([stats.spearmanr(x[i], y[i])[0] for i in idx], [2.5, 97.5])


def permutation_p(x, g, b=5000):
    obs = x[g == 1].mean() - x[g == 0].mean()
    sims = np.array([x[p == 1].mean() - x[p == 0].mean() for p in (RNG.permutation(g) for _ in range(b))])
    return (np.sum(np.abs(sims) >= abs(obs)) + 1) / (b + 1)


def trait_tests(df):
    rows = []
    for pos in POSITIONS:
        d = df[df.combine_position == pos]
        for trait in MAIN_TRAITS + SECOND_TRAITS:
            h = d[[trait, "success", "over", "pick"]].dropna()
            if len(h) < 15 or h.over.nunique() < 2:
                rows.append(dict(position=pos, trait=trait, n=len(h)))
                continue
            x, y, g = h[trait].values, h.success.values, h.over.values
            rho, p_rho = stats.spearmanr(x, y)
            lo, hi = bootstrap_rho(x, y)
            mw = stats.mannwhitneyu(x[g == 1], x[g == 0], alternative="two-sided")
            thirds = pd.qcut(pd.Series(x).rank(method="first"), 3, labels=False).values
            groups = [y[thirds == k] for k in range(3)]
            # same draft spot: logistic over ~ trait (per SD) + log(pick)
            X = sm.add_constant(np.column_stack([(x - x.mean()) / x.std(), np.log(h.pick.values)]))
            fit = sm.Logit(g, X).fit(disp=0)
            rows.append(dict(position=pos, trait=trait, n=len(h), n_over=int(g.sum()),
                             rho=rho, rho_lo=lo, rho_hi=hi, p_rho=p_rho,
                             mean_over=x[g == 1].mean(), mean_under=x[g == 0].mean(),
                             rank_biserial=2 * mw.statistic / (np.sum(g == 1) * np.sum(g == 0)) - 1,
                             p_mw=mw.pvalue, p_welch=stats.ttest_ind(x[g == 1], x[g == 0], equal_var=False).pvalue,
                             p_perm=permutation_p(x, g), p_anova=stats.f_oneway(*groups).pvalue,
                             p_kw=stats.kruskal(*groups).pvalue,
                             odds_ratio_per_sd=float(np.exp(fit.params[1])), p_same_spot=float(fit.pvalues[1])))
    out = pd.DataFrame(rows)
    out["main"] = out.trait.isin(MAIN_TRAITS)
    for col in ["p_mw", "p_same_spot"]:
        out[f"{col}_holm"] = np.nan
        for pos in POSITIONS:
            i = (out.position == pos) & out.main & out[col].notna()
            out.loc[i, f"{col}_holm"] = multipletests(out.loc[i, col], method="holm")[1]

    def verdict(r):
        if pd.isna(r.p_mw):
            return "Too few tested"
        a = r.p_mw_holm if r.main else r.p_mw            # do over-performers test differently?
        b = r.p_same_spot_holm if r.main else r.p_same_spot  # still true at the same draft spot?
        return "Clear" if a < .05 and b < .05 else "Partial" if (a < .05 or b < .05 or a < .10) else "No"

    out["verdict"] = out.apply(verdict, axis=1)
    return out


# ----------------------------------------------------------------------------------------------- 6. forecast
FEATURES = MODEL_TRAITS + ["missing_drills"]  # combine results only: everything is known before the draft
CONFERENCE_NAMES = {"Southeastern Conference": "SEC", "Big Ten Conference": "Big Ten",
                    "Atlantic Coast Conference": "ACC", "Big Twelve Conference": "Big 12",
                    "Pacific Twelve Conference": "Pac-12"}  # every other conference -> "Other"
CONFERENCES = list(CONFERENCE_NAMES.values())
CONF_COLS = [f"conf_{c}" for c in CONFERENCES]  # "Other" is the left-out group
INPUT_SETS = {"Combine": FEATURES, "Combine + conference": FEATURES + CONF_COLS,
              "Combine + age": FEATURES + ["age_at_draft"],
              "Combine + conference + age": FEATURES + CONF_COLS + ["age_at_draft"]}
ALGORITHMS = ["Logistic regression", "Random forest", "XGBoost"]


def make_model(name):
    """The three algorithms compared. Random forest and gradient boosting are the usual picks in NFL draft and combine
    projects (the Next Gen Stats draft model uses XGBoost); logistic regression is the simple baseline. The tree models
    are kept small because each position has only 35-120 training players."""
    if name == "Logistic regression":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    if name == "Random forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=2027, n_jobs=-1)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=2027,
                             verbosity=0)
    raise ValueError(name)


def prepare(df):
    """Model inputs. Everything here is known before the draft."""
    df = df.copy()
    df["missing_drills"] = df[MODEL_TRAITS].isna().sum(axis=1)
    df[MODEL_TRAITS] = df[MODEL_TRAITS].fillna(0)  # missing drill = class average
    df["conference_group"] = df.college_conference.map(CONFERENCE_NAMES).fillna("Other")
    for c in CONFERENCES:
        df[f"conf_{c}"] = (df.conference_group == c).astype(int)
    draft_day = pd.to_datetime(df.draft_year.astype(str) + "-04-27")  # the draft is held in late April
    df["age_at_draft"] = (draft_day - pd.to_datetime(df.birth_date)).dt.days / 365.25
    return df


def scores(y, prob):
    pred = (prob > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    acc_b, auc_b = [], []
    for _ in range(2000):  # bootstrap the test players for a 95% range
        i = RNG.integers(0, len(y), len(y))
        acc_b.append(accuracy_score(y[i], pred[i]))
        if len(set(y[i])) == 2:
            auc_b.append(roc_auc_score(y[i], prob[i]))
    return dict(baseline=max(y.mean(), 1 - y.mean()), accuracy=accuracy_score(y, pred),
                acc_lo=np.percentile(acc_b, 2.5), acc_hi=np.percentile(acc_b, 97.5),
                auc=roc_auc_score(y, prob), auc_lo=np.percentile(auc_b, 2.5), auc_hi=np.percentile(auc_b, 97.5),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), tp=tp, fp=fp, tn=tn, fn=fn)


def cv_auc(train, features, algo):
    """Cross-validated AUC inside the 2023-24 classes (5 folds, repeated 4 times). Used to choose the algorithm and
    the inputs, so the 2025 class stays an unseen test."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=2027)
    return cross_val_score(make_model(algo), train[features], train.over, cv=cv, scoring="roc_auc").mean()


def fit_row(pos, algo, inputs, train, test):
    feats = INPUT_SETS[inputs]
    y = test.over.values.astype(int)
    prob = make_model(algo).fit(train[feats], train.over).predict_proba(test[feats])[:, 1]
    pred = (prob > 0.5).astype(int)
    return dict(position=pos, algorithm=algo, inputs=inputs, cv_auc_2023_24=cv_auc(train, feats, algo),
                accuracy_2025=accuracy_score(y, pred), auc_2025=roc_auc_score(y, prob),
                f1_2025=f1_score(y, pred, zero_division=0), baseline_2025=max(y.mean(), 1 - y.mean()))


def splits(df):
    for pos in POSITIONS:
        d = df[df.combine_position == pos]
        yield pos, d[d.draft_year <= 2024], d[d.draft_year == 2025]


def choose(df):
    """Step 1: the algorithm with the best average cross-validated AUC on combine inputs (logistic regression unless
    another beats it by more than 0.01). Step 2: with that algorithm, add conference and age one at a time; an
    addition is kept only if it raises the average cross-validated AUC by more than 0.01 and does not lower the average
    2025 accuracy, compared with the inputs without it.
    Returns (algorithm, inputs, comparison table)."""
    d = prepare(df)
    rows = [fit_row(pos, algo, "Combine", tr, te) for pos, tr, te in splits(d) for algo in ALGORITHMS]
    table = pd.DataFrame(rows)
    mean = table.groupby("algorithm").cv_auc_2023_24.mean()
    algo = mean.idxmax() if mean.max() - mean[ALGORITHMS[0]] > 0.01 else ALGORITHMS[0]
    rows = [fit_row(pos, algo, inputs, tr, te) for pos, tr, te in splits(d) for inputs in list(INPUT_SETS)[1:]]
    table = pd.concat([table, pd.DataFrame(rows)], ignore_index=True)
    m = table[table.algorithm == algo].groupby("inputs")[["cv_auc_2023_24", "accuracy_2025"]].mean()
    # add one input at a time; an addition must raise CV AUC by more than 0.01 and not lower 2025 accuracy
    inputs, added = "Combine", []
    for _ in range(2):
        best = None
        for extra in ["conference", "age"]:
            if extra in added:
                continue
            cand = " + ".join(["Combine"] + [x for x in ["conference", "age"] if x in added + [extra]])
            gain = m.loc[cand, "cv_auc_2023_24"] - m.loc[inputs, "cv_auc_2023_24"]
            if gain > 0.01 and m.loc[cand, "accuracy_2025"] >= m.loc[inputs, "accuracy_2025"]:
                if best is None or gain > best[1]:
                    best = (extra, gain, cand)
        if best is None:
            break
        added.append(best[0])
        inputs = best[2]
    table["kept"] = (table.algorithm == algo) & (table.inputs == inputs)
    return algo, inputs, table


def classifier(df, algo="Logistic regression", inputs="Combine"):
    """Pre-draft forecast for the 2025 class: trained on 2023-24, tested on 2025."""
    feats = INPUT_SETS[inputs]
    rows, preds = [], []
    for pos, train, test in splits(prepare(df)):
        prob = make_model(algo).fit(train[feats], train.over).predict_proba(test[feats])[:, 1]  # 2023-24 only
        y = test.over.values.astype(int)
        rows.append(dict(position=pos, algorithm=algo, inputs=inputs, n_train=len(train),
                         n_train_over=int(train.over.sum()), n_test=len(test), n_test_over=int(y.sum()),
                         **scores(y, prob)))
        preds.append(test[["nfl_id", "display_name", "combine_position", "conference_group", "age_at_draft", "pick",
                           "over"]].assign(prob_over=prob.round(4), flagged=(prob > 0.5).astype(int)))
    return pd.DataFrame(rows), pd.concat(preds).round(4)


# ----------------------------------------------------------------------------------------------- 7. reports
def conference_report(df):
    """Players, average starts and over-performer share by conference group and position (all three classes)."""
    d = prepare(df)
    rep = (d.groupby(["conference_group", "combine_position"])
           .agg(players=("nfl_id", "size"), avg_starts_per_season=("starts_per_season", "mean"),
                avg_career_starts=("career_games_started", "mean"), over_rate=("over", "mean")).reset_index())
    rep["conference_group"] = pd.Categorical(rep.conference_group, CONFERENCES + ["Other"])
    rep["combine_position"] = pd.Categorical(rep.combine_position, POSITIONS)
    return rep.sort_values(["conference_group", "combine_position"]).rename(columns={"combine_position": "position"})


def age_summary(df):
    """Age at the draft by position, and whether over-performers were younger."""
    rows = []
    for pos, g in prepare(df).groupby("combine_position"):
        rows.append(dict(position=pos, players=len(g), median_age=g.age_at_draft.median(),
                         min_age=g.age_at_draft.min(), max_age=g.age_at_draft.max(),
                         median_age_over=g[g.over == 1].age_at_draft.median(),
                         median_age_under=g[g.over == 0].age_at_draft.median(),
                         rho_age_success=stats.spearmanr(g.age_at_draft, g.success)[0]))
    return pd.DataFrame(rows).set_index("position").reindex(POSITIONS).reset_index()


# ----------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", help="folder with the competition CSVs (skips the kagglehub download)")
    ap.add_argument("--out", default="results", help="output folder")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df, games, plays, career = load(get_data_dir(args.data_dir))
    df = add_combine_traits(df)
    df = df.join(build_rookie_success(df, games, plays), on="nfl_id")
    df = add_labels(df, career)
    print(f"{len(df)} players; over-performers by position and class:")
    print(pd.crosstab([df.combine_position, df.draft_year], df.over).to_string(), "\n")

    tests = trait_tests(df)
    tests.to_csv(f"{args.out}/trait_tests.csv", index=False)
    print("Hypothesis tests: does a combine trait mark an over-performer?")
    print(tests.pivot(index="trait", columns="position", values="verdict").reindex(MAIN_TRAITS + SECOND_TRAITS)[POSITIONS]
          .to_string(), "\n")

    algo, inputs, table = choose(df)
    table.to_csv(f"{args.out}/model_comparison.csv", index=False)
    pd.set_option("display.width", 220)
    print("Model comparison (chosen on 2023-24 cross-validation; 2025 shown for reference)")
    print(table.groupby(["algorithm", "inputs"])[["cv_auc_2023_24", "accuracy_2025", "auc_2025", "f1_2025"]].mean()
          .round(3).to_string(), "\n")
    print(f"Kept: {algo}, inputs = {inputs}\n")

    clf, preds = classifier(df, algo, inputs)
    clf.to_csv(f"{args.out}/classifier_results.csv", index=False)
    preds.to_csv(f"{args.out}/predictions_2025.csv", index=False)
    show = ["position", "n_train", "n_test", "n_test_over", "baseline", "accuracy", "acc_lo", "acc_hi", "auc",
            "precision", "recall", "f1", "tp", "fp", "tn", "fn"]
    print("Pre-draft forecast: trained on 2023-24 classes, tested on the 2025 class")
    print(clf[show].round(3).to_string(index=False), "\n")

    conference_report(df).round(4).to_csv(f"{args.out}/conference_report.csv", index=False)
    age = age_summary(df)
    age.round(3).to_csv(f"{args.out}/age_summary.csv", index=False)
    print("Age at the draft by position")
    print(age.round(2).to_string(index=False), "\n")
    df[["nfl_id", "display_name", "combine_position", "draft_year", "draft_round", "pick", "rookie_score",
        "rookie_snaps_pg", "career_games_started", "starts_per_season", "success", "over"]
       + MAIN_TRAITS + SECOND_TRAITS].round(4).to_csv(f"{args.out}/player_labels.csv", index=False)


if __name__ == "__main__":
    main()
