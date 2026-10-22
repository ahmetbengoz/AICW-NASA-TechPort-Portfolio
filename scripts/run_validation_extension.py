#!/usr/bin/env python3
"""Nested evaluation of the importance-to-ranking conversion.

No outcomes of an outer test project enter weight learning, rarity estimation,
normalisation or model fitting. Results are retrospective proxy-outcome tests,
not stakeholder utility or causal treatment-effect estimates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import ndcg_score
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold

from run_analysis import (
    FEATURES, SEED, DIRECTIONS, prepare_data, metric_row, waspas_scores,
    marcos_scores, merec_weights, ratio_normalize,
)
from figure_io import save_png_atomic

ROOT = Path(__file__).resolve().parents[1]
RF_KWARGS = dict(n_estimators=200, min_samples_leaf=2, max_features=1.0, n_jobs=1)


def rarity_transform(train: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Learn category counts from training projects only; unseen code rarity=1."""
    counts = train.primaryTxCode.value_counts()
    result = candidates[FEATURES].copy()
    result["taxonomy_rarity"] = candidates.primaryTxCode.map(
        lambda x: 1.0 / counts.get(x, 1)
    ).to_numpy()
    return result


def training_normalize(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    columns = []
    for feature in FEATURES:
        x = train[feature].to_numpy(float)
        z = test[feature].to_numpy(float)
        if DIRECTIONS[feature] == "benefit":
            columns.append(z / max(x.max(), 1e-12))
        else:
            positive = x[x > 0]
            scale = positive.min() if len(positive) else 1.0
            columns.append(scale / np.maximum(z, 1e-12))
    # A candidate beyond the training range may have a score >1. Do not use
    # its observed maximum to change the scale, or clip it to hide the excess.
    return np.maximum(np.column_stack(columns), 1e-12)


def objective_profiles(train: pd.DataFrame) -> dict[str, np.ndarray]:
    n, m = train.shape
    lo = train.min().to_numpy(float)
    span = train.max().to_numpy(float) - lo
    z = np.divide(train.to_numpy(float) - lo, span,
                  out=np.zeros((n, m)), where=span > 0)
    for j, f in enumerate(FEATURES):
        if DIRECTIONS[f] == "cost":
            z[:, j] = 1 - z[:, j]
    p = np.divide(z, z.sum(axis=0), out=np.zeros_like(z), where=z.sum(axis=0) > 0)
    entropy = -np.sum(np.where(p > 0, p * np.log(np.maximum(p, 1e-300)), 0), axis=0) / np.log(n)
    information = np.where(span > 0, np.maximum(1 - entropy, 0), 0)
    sd = z.std(axis=0, ddof=1)
    correlation = np.nan_to_num(pd.DataFrame(z).corr().to_numpy(), nan=0.0)
    np.fill_diagonal(correlation, 1.0)
    contrast = sd * (1 - correlation).sum(axis=1)
    def unit(x):
        return x / x.sum() if x.sum() > 0 else np.ones(m) / m
    # MEREC retains the disclosed 1e-12 zero floor of the descriptive analysis.
    return {"Equal": np.ones(m) / m, "Entropy": unit(information),
            "CRITIC": unit(contrast), "SD": unit(sd), "MEREC": merec_weights(train)}


def split_plan(data: pd.DataFrame):
    for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=5,
                                             random_state=SEED).split(data)):
        yield "random", i // 5 + 1, i % 5 + 1, tr, te
    groups = data.programTitle.fillna("Unknown").to_numpy()
    for i, (tr, te) in enumerate(GroupKFold(n_splits=5, shuffle=True,
                                           random_state=SEED).split(data, groups=groups)):
        yield "programme", 1, i + 1, tr, te
    ordered = data.sort_values(["startDate", "projectId"])
    cutoff = ordered.iloc[int(0.8 * len(data))].startDate
    tr = np.flatnonzero((data.startDate < cutoff).to_numpy())
    te = np.flatnonzero((data.startDate >= cutoff).to_numpy())
    assert data.iloc[tr].startDate.max() < data.iloc[te].startDate.min()
    yield "chronological", 1, 1, tr, te


def inner_plan(train: pd.DataFrame, mode: str, seed: int):
    if mode == "programme":
        groups = train.programTitle.fillna("Unknown").to_numpy()
        return list(GroupKFold(n_splits=3, shuffle=True, random_state=seed)
                    .split(train, groups=groups))
    # Chronological weights are cross-fitted only within the historical
    # training pool; no outer candidate or its outcome can enter these folds.
    return list(KFold(n_splits=3, shuffle=True, random_state=seed).split(train))


def learned_profile(train: pd.DataFrame, outcome: str, mode: str, seed: int):
    importances = []
    for j, (a, b) in enumerate(inner_plan(train, mode, seed)):
        assert not set(a) & set(b)
        fit_data, val_data = train.iloc[a], train.iloc[b]
        x_fit = rarity_transform(fit_data, fit_data)
        x_val = rarity_transform(fit_data, val_data)
        estimator = RandomForestRegressor(**RF_KWARGS, random_state=seed + j)
        estimator.fit(x_fit, fit_data[outcome])
        importance = permutation_importance(
            estimator, x_val, val_data[outcome], scoring="neg_mean_absolute_error",
            n_repeats=10, random_state=seed + 100 + j, n_jobs=1,
        ).importances_mean
        importances.append(importance)
    # Average signed loss increases before clipping; do not average already
    # normalised vectors, which would give weak/noisy folds equal influence.
    raw = np.mean(importances, axis=0)
    positive = np.maximum(raw, 0)
    fallback = bool(positive.sum() <= 0)
    weight = positive / positive.sum() if not fallback else np.ones(len(FEATURES)) / len(FEATURES)
    return raw, weight, fallback


def selection_value(y: np.ndarray, score: np.ndarray, fraction: float) -> dict:
    k = max(1, math.ceil(fraction * len(y)))
    threshold = np.sort(score)[-k]
    selected = score > threshold
    tied = np.isclose(score, threshold, rtol=0, atol=0)
    allocation = selected.astype(float)
    allocation[tied] = (k - selected.sum()) / tied.sum()
    value = float(allocation @ y / k)
    candidate_mean = float(y.mean())
    oracle = float(np.sort(y)[-k:].mean())
    return dict(k=k, selected_mean=value, candidate_mean=candidate_mean,
                lift=value - candidate_mean, oracle_mean=oracle,
                oracle_gap=oracle - value,
                ndcg=float(ndcg_score(y[None, :], score[None, :], k=k,
                                     ignore_ties=False)))


def run(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw, primary, secondary = prepare_data(ROOT / "data/raw/techport_projects_batch1.csv")
    selections, predictions, weights, memberships = [], [], [], []
    for outcome_name, data, outcome in [
        ("progress", primary, "realized_trl_progress"),
        ("attainment", secondary, "trl_goal_attainment"),
    ]:
        data = data.reset_index(drop=True)
        for split_no, (mode, repeat, fold, tr, te) in enumerate(split_plan(data)):
            train, test = data.iloc[tr].reset_index(drop=True), data.iloc[te].reset_index(drop=True)
            assert not set(train.projectId) & set(test.projectId)
            if mode == "programme":
                assert not set(train.programTitle) & set(test.programTitle)
            meta = dict(outcome=outcome_name, split=mode, repeat=repeat, fold=fold,
                        train_n=len(train), test_n=len(test))
            for role, d in [("train", train), ("test", test)]:
                memberships.extend(dict(**meta, role=role, project_id=int(i)) for i in d.projectId)
            xtr, xte = rarity_transform(train, train), rarity_transform(train, test)
            ytr, yte = train[outcome].to_numpy(float), test[outcome].to_numpy(float)
            seed = SEED + 20000 + split_no
            raw_importance, proxy, fallback = learned_profile(train, outcome, mode, seed)
            weights.extend(dict(**meta, feature=f, raw_importance=float(a),
                                weight=float(b), equal_fallback=fallback)
                           for f, a, b in zip(FEATURES, raw_importance, proxy))
            profiles = objective_profiles(xtr)
            normal = training_normalize(xtr, xte)
            scores = {name: waspas_scores(normal, w) for name, w in profiles.items()}
            scores["RF_WASPAS"] = waspas_scores(normal, proxy)
            scores["RF_MARCOS"] = marcos_scores(normal, proxy)
            scores["PlannedGain"] = xte.planned_trl_gain.to_numpy(float)
            scores["InitialTRL"] = xte.initial_trl.to_numpy(float)
            full = RandomForestRegressor(**RF_KWARGS, random_state=seed).fit(xtr, ytr)
            full_prediction = full.predict(xte)
            scores["DirectRF"] = full_prediction
            # Reduced predictive baselines isolate mechanical headroom from
            # residual multivariate information, on the identical test cases.
            no_gain = [f for f in FEATURES if f != "planned_trl_gain"]
            reduced = RandomForestRegressor(**RF_KWARGS, random_state=seed).fit(xtr[no_gain], ytr)
            gain_only = LinearRegression().fit(xtr[["planned_trl_gain"]], ytr)
            headroom = LinearRegression().fit(xtr[["initial_trl", "planned_trl_gain"]], ytr)
            ablations = {"Mean": np.full(len(test), ytr.mean()),
                         "GainLinear": gain_only.predict(xte[["planned_trl_gain"]]),
                         "HeadroomLinear": headroom.predict(xte[["initial_trl", "planned_trl_gain"]]),
                         "RF_without_gain": reduced.predict(xte[no_gain]),
                         "RF_all": full_prediction}
            for name, pred in ablations.items():
                for pid, obs, estimate in zip(test.projectId, yte, pred):
                    predictions.append(dict(**meta, model=name, project_id=int(pid),
                                            observed=float(obs), prediction=float(estimate)))
            for name, score in scores.items():
                for fraction in [0.1, 0.2, 0.3]:
                    selections.append(dict(**meta, policy=name, fraction=fraction,
                                           **selection_value(yte, score, fraction)))
            if fold == 1:
                print(json.dumps(dict(**meta, status="completed")), flush=True)

    pd.DataFrame(memberships).to_csv(output_dir / "split_membership.csv", index=False)
    pd.DataFrame(weights).to_csv(output_dir / "nested_proxy_weights.csv", index=False)
    pred = pd.DataFrame(predictions)
    pred.to_csv(output_dir / "outer_predictions.csv", index=False)
    selection = pd.DataFrame(selections)
    selection.to_csv(output_dir / "shortlist_fold_results.csv", index=False)
    summary = selection.groupby(["outcome", "split", "policy", "fraction"], sort=False).agg(
        selected_mean=("selected_mean", "mean"), lift=("lift", "mean"),
        lift_min=("lift", "min"), lift_max=("lift", "max"),
        oracle_gap=("oracle_gap", "mean"), ndcg=("ndcg", "mean"),
        evaluated_folds=("lift", "size"),
    ).reset_index()
    summary.to_csv(output_dir / "shortlist_summary.csv", index=False)
    metric_rows = []
    for key, frame in pred.groupby(["outcome", "split", "repeat", "model"]):
        metric_rows.append(dict(zip(["outcome", "split", "repeat", "model"], key),
                                **metric_row(frame.observed.to_numpy(), frame.prediction.to_numpy())))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(output_dir / "ablation_repeat_metrics.csv", index=False)
    metrics.groupby(["outcome", "split", "model"])[["r2", "mae", "rmse"]].agg(
        ["mean", "std"]).to_csv(output_dir / "ablation_summary.csv")

    # Label permutation is a falsification check for predictive signal, not
    # a p-value for the contribution of any individual correlated feature.
    permutation_rows = []
    for outcome_name, data, outcome in [("progress", primary, "realized_trl_progress"),
                                         ("attainment", secondary, "trl_goal_attainment")]:
        data = data.reset_index(drop=True)
        splits = list(KFold(n_splits=5, shuffle=True, random_state=SEED).split(data))
        y = data[outcome].to_numpy(float)
        rng = np.random.default_rng(SEED + 91000)
        for draw in range(101):
            target = y if draw == 0 else rng.permutation(y)
            oof = np.zeros(len(y))
            for j, (tr, te) in enumerate(splits):
                train, test = data.iloc[tr], data.iloc[te]
                model = RandomForestRegressor(**RF_KWARGS, random_state=SEED + j)
                model.fit(rarity_transform(train, train), target[tr])
                oof[te] = model.predict(rarity_transform(train, test))
            permutation_rows.append(dict(outcome=outcome_name, draw=draw,
                                         **metric_row(target, oof)))
            if draw in [0, 25, 50, 75, 100]:
                print(json.dumps(dict(test="label_permutation", outcome=outcome_name,
                                      draw=draw)), flush=True)
    perm = pd.DataFrame(permutation_rows)
    perm.to_csv(output_dir / "label_permutation.csv", index=False)
    null_summary = []
    for name, frame in perm.groupby("outcome"):
        observed = float(frame.loc[frame.draw.eq(0), "r2"].iloc[0])
        null = frame.loc[frame.draw.gt(0), "r2"].to_numpy()
        null_summary.append(dict(outcome=name, observed_r2=observed,
                                 null_mean_r2=float(null.mean()),
                                 null_max_r2=float(null.max()),
                                 permutation_p=(1 + int((null >= observed).sum())) / (1 + len(null))))
    pd.DataFrame(null_summary).to_csv(output_dir / "label_permutation_summary.csv", index=False)

    figure_dir = ROOT / "figures"
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "aicw-v3"})
    policies = ["Equal", "Entropy", "CRITIC", "MEREC", "PlannedGain", "RF_WASPAS", "RF_MARCOS", "DirectRF"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6), sharey=True)
    for ax, (outcome, title) in zip(axes, [("progress", "Raw TRL progress"), ("attainment", "Target attainment")]):
        for j, (mode, marker, color) in enumerate([("random", "o", "#0072B2"),
                                                 ("programme", "s", "#D55E00"),
                                                 ("chronological", "^", "#009E73")]):
            panel = summary.loc[(summary.outcome == outcome) & (summary.split == mode) & (summary.fraction == .2)].set_index("policy")
            ax.scatter(panel.loc[policies, "lift"], np.arange(len(policies)) + .18 * (j - 1),
                       marker=marker, color=color, label=mode.capitalize(), s=36)
        ax.axvline(0, color="grey", lw=.8)
        ax.set_title(title)
        ax.set_yticks(np.arange(len(policies)), [p.replace("RF_", "RF ").replace("DirectRF", "Direct RF").replace("PlannedGain", "Planned gain") for p in policies])
        ax.set_xlabel("Observed outcome lift in the top 20%")
        ax.grid(axis="x", alpha=.2)
        ax.invert_yaxis()
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "Figure7_nested_shortlist.svg", bbox_inches="tight", metadata={"Date": None})
    save_png_atomic(fig, figure_dir / "Figure7_nested_shortlist.png", bbox_inches="tight")
    plt.close(fig)
    manifest = dict(seed=SEED, random_outer_folds=25, programme_outer_folds=5,
                    chronological_cutoff="2021-10-01", inner_folds=3,
                    inner_permutations=10, label_permutations=100,
                    selection_fraction_primary=.2, tie_rule="fractional allocation at boundary",
                    rarity="training-only reciprocal category count; unseen=1",
                    versions={"numpy": np.__version__, "pandas": pd.__version__},
                    outputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in sorted(output_dir.glob("*.csv"))})
    (output_dir / "extension_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"status": "ok", "outer_evaluations": len(selection), "null_tests": null_summary}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "validation_extension")
    run(parser.parse_args().output_dir)
