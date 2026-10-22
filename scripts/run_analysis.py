#!/usr/bin/env python3
"""Reproduce the leakage-aware NASA TechPort analysis.

The script uses the frozen repository snapshot and writes all analytical
tables and publication figures to ``outputs/`` and ``figures/``.  It does not
query the live TechPort API, so repeated runs are deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import zipfile
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
from PIL import Image
from scipy.optimize import linprog
from scipy.stats import kendalltau, spearmanr, t, wilcoxon
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from figure_io import save_png_atomic


SEED = 20260811
N_SPLITS = 5
N_REPEATS = 5
N_PERMUTATIONS = 10
N_WEIGHT_DRAWS = 5000

FEATURES = [
    "initial_trl",
    "planned_duration_months",
    "planned_trl_gain",
    "taxonomy_level",
    "taxonomy_rarity",
    "other_organization_count",
]

FEATURE_LABELS = {
    "initial_trl": "Initial TRL",
    "planned_duration_months": "Planned duration",
    "planned_trl_gain": "Planned TRL gain",
    "taxonomy_level": "Taxonomy level",
    "taxonomy_rarity": "Taxonomy rarity",
    "other_organization_count": "Other organizations",
}

DIRECTIONS = {
    "initial_trl": "benefit",
    "planned_duration_months": "cost",
    "planned_trl_gain": "benefit",
    "taxonomy_level": "benefit",
    "taxonomy_rarity": "benefit",
    "other_organization_count": "benefit",
}

LEARNER_ORDER = ["ElasticNet", "RandomForest", "ExtraTrees", "HistGradientBoosting"]

PUBLICATION_FIGURE_STEMS = [
    "Figure1_workflow",
    "Figure2_predictive_performance",
    "Figure3_proxy_weights",
    "Figure4_learner_agreement",
    "Figure5_leakage_diagnostic",
    "Figure6_weight_perturbation",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=root / "data" / "raw" / "techport_projects_batch1.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "outputs")
    parser.add_argument("--figure-dir", type=Path, default=root / "figures")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    raw["startDate"] = pd.to_datetime(raw["startDate"], errors="coerce")
    raw["endDate"] = pd.to_datetime(raw["endDate"], errors="coerce")

    taxonomy_counts = raw["primaryTxCode"].dropna().value_counts()
    raw["taxonomy_rarity"] = raw["primaryTxCode"].map(
        lambda value: 1.0 / taxonomy_counts[value] if pd.notna(value) else np.nan
    )
    raw["planned_duration_months"] = (
        (raw["endDate"] - raw["startDate"]).dt.days / 30.4375
    )
    raw["realized_trl_progress"] = raw["trlCurrent"] - raw["trlBegin"]
    raw["planned_trl_gain"] = raw["trlEnd"] - raw["trlBegin"]
    raw["remaining_trl_gap"] = raw["trlEnd"] - raw["trlCurrent"]

    raw = raw.rename(
        columns={
            "trlBegin": "initial_trl",
            "trlCurrent": "current_trl",
            "primaryTxLevel": "taxonomy_level",
            "otherOrgCount": "other_organization_count",
        }
    )

    required = [
        "projectId",
        "status",
        "initial_trl",
        "current_trl",
        "planned_duration_months",
        "planned_trl_gain",
        "remaining_trl_gap",
        "taxonomy_level",
        "taxonomy_rarity",
        "other_organization_count",
        "realized_trl_progress",
    ]
    complete = raw.dropna(subset=required).copy()
    primary = complete.loc[complete["status"].str.casefold().eq("completed")].copy()
    primary = primary.loc[
        (primary["planned_duration_months"] > 0)
        & (primary["planned_trl_gain"] >= 0)
        & (primary["realized_trl_progress"] >= 0)
    ].copy()
    primary = primary.sort_values("projectId").reset_index(drop=True)

    secondary = primary.loc[primary["planned_trl_gain"] > 0].copy()
    secondary["trl_goal_attainment"] = (
        secondary["realized_trl_progress"] / secondary["planned_trl_gain"]
    ).clip(0, 1)

    return raw, primary, secondary


def make_models(seed: int) -> dict[str, object]:
    elastic_pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                ElasticNet(
                    max_iter=50000,
                    random_state=seed,
                ),
            ),
        ]
    )
    # Fit the scaler inside each inner split, not before tuning.
    elastic = GridSearchCV(
        elastic_pipeline,
        {"model__alpha": np.logspace(-4, 1, 60),
         "model__l1_ratio": [0.1, 0.5, 0.9, 1.0]},
        scoring="neg_mean_squared_error", cv=5, n_jobs=1,
    )
    return {
        "DummyMean": DummyRegressor(strategy="mean"),
        "ElasticNet": elastic,
        "RandomForest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=seed,
            n_jobs=1,
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=150,
            max_leaf_nodes=15,
            min_samples_leaf=10,
            l2_regularization=0.1,
            random_state=seed,
        ),
    }


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
    }


def repeated_cv_experiment(
    data: pd.DataFrame,
    features: list[str],
    outcome: str,
    outcome_specification: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X = data[features].reset_index(drop=True)
    y = data[outcome].to_numpy(dtype=float)
    splitter = RepeatedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    splits = list(splitter.split(X))

    predictions = {
        (learner, repeat): np.full(len(data), np.nan)
        for learner in make_models(SEED)
        for repeat in range(N_REPEATS)
    }
    fold_metrics: list[dict[str, object]] = []
    importance_rows: list[dict[str, object]] = []

    for split_index, (train_index, test_index) in enumerate(splits):
        repeat = split_index // N_SPLITS
        fold = split_index % N_SPLITS
        models = make_models(SEED + split_index)
        for learner, estimator in models.items():
            fitted = clone(estimator).fit(X.iloc[train_index], y[train_index])
            prediction = fitted.predict(X.iloc[test_index])
            predictions[(learner, repeat)][test_index] = prediction
            row = {
                "outcome_specification": outcome_specification,
                "learner": learner,
                "repeat": repeat + 1,
                "fold": fold + 1,
            }
            row.update(metric_row(y[test_index], prediction))
            fold_metrics.append(row)

            if learner == "DummyMean":
                continue
            perm = permutation_importance(
                fitted,
                X.iloc[test_index],
                y[test_index],
                scoring="neg_mean_absolute_error",
                n_repeats=N_PERMUTATIONS,
                random_state=SEED + 1000 + split_index,
                n_jobs=1,
            )
            raw_importance = perm.importances_mean.astype(float)
            clipped = np.clip(raw_importance, 0.0, None)
            if clipped.sum() <= 0:
                normalized = np.repeat(1.0 / len(features), len(features))
                fallback_equal = True
            else:
                normalized = clipped / clipped.sum()
                fallback_equal = False
            for feature, raw_value, weight in zip(
                features, raw_importance, normalized, strict=True
            ):
                importance_rows.append(
                    {
                        "outcome_specification": outcome_specification,
                        "learner": learner,
                        "repeat": repeat + 1,
                        "fold": fold + 1,
                        "feature": feature,
                        "permutation_importance_mae": raw_value,
                        "normalized_proxy_weight": weight,
                        "negative_before_clipping": raw_value < 0,
                        "equal_weight_fallback": fallback_equal,
                    }
                )

    repeat_rows: list[dict[str, object]] = []
    for (learner, repeat), prediction in predictions.items():
        if np.isnan(prediction).any():
            raise RuntimeError(f"Incomplete OOF predictions for {learner}, repeat {repeat}")
        row = {
            "outcome_specification": outcome_specification,
            "learner": learner,
            "repeat": repeat + 1,
        }
        row.update(metric_row(y, prediction))
        repeat_rows.append(row)

    repeat_metrics = pd.DataFrame(repeat_rows)
    summary_rows: list[dict[str, object]] = []
    for learner, subset in repeat_metrics.groupby("learner", sort=False):
        for metric in ["r2", "mae", "rmse"]:
            values = subset[metric].to_numpy()
            sem = values.std(ddof=1) / math.sqrt(len(values))
            half_width = t.ppf(0.975, len(values) - 1) * sem
            summary_rows.append(
                {
                    "outcome_specification": outcome_specification,
                    "learner": learner,
                    "metric": metric,
                    "mean": values.mean(),
                    "sd": values.std(ddof=1),
                    "ci95_low": values.mean() - half_width,
                    "ci95_high": values.mean() + half_width,
                    "n_repeats": len(values),
                }
            )

    return (
        pd.DataFrame(fold_metrics),
        repeat_metrics,
        pd.DataFrame(summary_rows),
        pd.DataFrame(importance_rows),
    )


def summarize_importance(importances: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = importances.groupby(
        ["outcome_specification", "learner", "feature"], sort=False
    )
    for (specification, learner, feature), subset in grouped:
        values = subset["normalized_proxy_weight"].to_numpy()
        rows.append(
            {
                "outcome_specification": specification,
                "learner": learner,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "mean_weight": values.mean(),
                "sd_weight": values.std(ddof=1),
                "ci95_low": np.quantile(values, 0.025),
                "ci95_high": np.quantile(values, 0.975),
                "negative_importance_fraction": subset[
                    "negative_before_clipping"
                ].mean(),
                "n_outer_folds": len(values),
            }
        )
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    count = len(p)
    for position, index in enumerate(order):
        candidate = (count - position) * p[index]
        running = max(running, candidate)
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def paired_performance_tests(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    primary = repeat_metrics.loc[
        repeat_metrics["outcome_specification"].eq("primary_progress")
        & ~repeat_metrics["learner"].eq("DummyMean")
    ]
    mean_mae = primary.groupby("learner")["mae"].mean()
    reference = mean_mae.idxmin()
    rows: list[dict[str, object]] = []
    for metric in ["mae", "rmse", "r2"]:
        pivot = primary.pivot(index="repeat", columns="learner", values=metric)
        for comparator in LEARNER_ORDER:
            if comparator == reference:
                continue
            difference = pivot[reference] - pivot[comparator]
            if np.allclose(difference, 0):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = wilcoxon(difference, zero_method="wilcox")
            rows.append(
                {
                    "metric": metric,
                    "reference_learner": reference,
                    "comparator": comparator,
                    "mean_paired_difference_reference_minus_comparator": difference.mean(),
                    "wilcoxon_statistic": statistic,
                    "p_value": p_value,
                }
            )
    adjusted = holm_adjust([row["p_value"] for row in rows])
    for row, p_adjusted in zip(rows, adjusted, strict=True):
        row["holm_adjusted_p"] = p_adjusted
    return pd.DataFrame(rows)


def weight_matrix(summary: pd.DataFrame, specification: str) -> pd.DataFrame:
    subset = summary.loc[summary["outcome_specification"].eq(specification)]
    matrix = subset.pivot(index="feature", columns="learner", values="mean_weight")
    matrix = matrix.reindex(index=FEATURES, columns=LEARNER_ORDER)
    matrix = matrix.div(matrix.sum(axis=0), axis=1)
    return matrix


def agreement_rows(
    vectors: dict[str, np.ndarray], context: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for first, second in combinations(vectors, 2):
        a = np.asarray(vectors[first], dtype=float)
        b = np.asarray(vectors[second], dtype=float)
        rows.append(
            {
                "context": context,
                "profile_1": first,
                "profile_2": second,
                "spearman_rho": spearmanr(a, b).statistic,
                "kendall_tau": kendalltau(a, b).statistic,
                "l1_distance": np.abs(a - b).sum(),
            }
        )
    return pd.DataFrame(rows)


def fucom_weights(order: list[str], ratios: list[float]) -> np.ndarray:
    """Solve the standard FUCOM consistency program for synthetic judgments."""
    m = len(order)
    # Variables are ordered weights plus the non-negative deviation chi.
    c = np.zeros(m + 1)
    c[-1] = 1.0
    a_ub: list[np.ndarray] = []
    b_ub: list[float] = []

    def add_absolute_constraint(coefficients: np.ndarray) -> None:
        upper = np.zeros(m + 1)
        upper[:m] = coefficients
        upper[-1] = -1.0
        a_ub.append(upper)
        b_ub.append(0.0)
        lower = np.zeros(m + 1)
        lower[:m] = -coefficients
        lower[-1] = -1.0
        a_ub.append(lower)
        b_ub.append(0.0)

    for index, ratio in enumerate(ratios):
        coeff = np.zeros(m)
        coeff[index] = 1.0
        coeff[index + 1] = -ratio
        add_absolute_constraint(coeff)
    for index in range(m - 2):
        coeff = np.zeros(m)
        coeff[index] = 1.0
        coeff[index + 2] = -(ratios[index] * ratios[index + 1])
        add_absolute_constraint(coeff)

    result = linprog(
        c,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=np.asarray([[*np.ones(m), 0.0]]),
        b_eq=np.asarray([1.0]),
        bounds=[(0, None)] * (m + 1),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"FUCOM optimization failed: {result.message}")
    ordered = result.x[:m]
    mapping = dict(zip(order, ordered, strict=True))
    return np.asarray([mapping[feature] for feature in FEATURES])


def ratio_normalize(matrix: pd.DataFrame) -> np.ndarray:
    normalized = np.zeros(matrix.shape, dtype=float)
    for column_index, feature in enumerate(matrix.columns):
        values = matrix[feature].to_numpy(dtype=float)
        if DIRECTIONS[feature] == "benefit":
            maximum = values.max()
            normalized[:, column_index] = values / maximum if maximum > 0 else 1.0
        else:
            positive = values[values > 0]
            minimum = positive.min() if len(positive) else 1.0
            normalized[:, column_index] = minimum / np.maximum(values, 1e-12)
    return np.clip(normalized, 1e-12, 1.0)


def sd_weights(matrix: pd.DataFrame) -> np.ndarray:
    normalized = ratio_normalize(matrix)
    dispersion = normalized.std(axis=0, ddof=1)
    return dispersion / dispersion.sum()


def merec_weights(matrix: pd.DataFrame) -> np.ndarray:
    # MEREC normalization makes more desirable values smaller before the
    # logarithmic performance calculation.
    normalized = np.zeros(matrix.shape, dtype=float)
    for column_index, feature in enumerate(matrix.columns):
        values = np.maximum(matrix[feature].to_numpy(dtype=float), 1e-12)
        if DIRECTIONS[feature] == "benefit":
            normalized[:, column_index] = values.min() / values
        else:
            normalized[:, column_index] = values / values.max()
    normalized = np.clip(normalized, 1e-12, 1.0)
    m = normalized.shape[1]
    overall = np.log1p(np.abs(np.log(normalized)).sum(axis=1) / m)
    effects = []
    for column_index in range(m):
        reduced = np.delete(normalized, column_index, axis=1)
        removed = np.log1p(np.abs(np.log(reduced)).sum(axis=1) / m)
        effects.append(np.abs(removed - overall).sum())
    effects = np.asarray(effects)
    return effects / effects.sum()


def waspas_scores(normalized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    additive = normalized @ weights
    multiplicative = np.exp(np.log(normalized) @ weights)
    return 0.5 * additive + 0.5 * multiplicative


def marcos_scores(normalized: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = normalized * weights
    sums = weighted.sum(axis=1)
    ideal_sum = weights.sum()
    anti_ideal = normalized.min(axis=0)
    anti_ideal_sum = float((anti_ideal * weights).sum())
    k_minus = sums / max(anti_ideal_sum, 1e-12)
    k_plus = sums / max(ideal_sum, 1e-12)
    denominator = k_plus + k_minus
    f_minus = np.divide(k_plus, denominator, out=np.zeros_like(k_plus), where=denominator > 0)
    f_plus = np.divide(k_minus, denominator, out=np.zeros_like(k_minus), where=denominator > 0)
    utility_denominator = (
        1.0
        + np.divide(1.0 - f_plus, f_plus, out=np.zeros_like(f_plus), where=f_plus > 0)
        + np.divide(1.0 - f_minus, f_minus, out=np.zeros_like(f_minus), where=f_minus > 0)
    )
    return (k_plus + k_minus) / utility_denominator


def rank_scores(scores: np.ndarray) -> np.ndarray:
    return pd.Series(scores).rank(ascending=False, method="min").to_numpy(dtype=int)


def conventional_profiles(matrix: pd.DataFrame) -> dict[str, np.ndarray]:
    order = [
        "initial_trl",
        "planned_trl_gain",
        "planned_duration_months",
        "taxonomy_level",
        "taxonomy_rarity",
        "other_organization_count",
    ]
    ratios = [1.10, 1.15, 1.20, 1.10, 1.10]
    return {
        "Equal": np.repeat(1 / len(FEATURES), len(FEATURES)),
        "Synthetic_FUCOM": fucom_weights(order, ratios),
        "MEREC": merec_weights(matrix),
        "StandardDeviation": sd_weights(matrix),
    }


def create_rankings(
    data: pd.DataFrame,
    profiles: dict[str, np.ndarray],
    specification: str,
) -> pd.DataFrame:
    matrix = data[FEATURES]
    normalized = ratio_normalize(matrix)
    rows: list[pd.DataFrame] = []
    for profile, weights in profiles.items():
        for method, scorer in {"WASPAS": waspas_scores, "MARCOS": marcos_scores}.items():
            scores = scorer(normalized, weights)
            rows.append(
                pd.DataFrame(
                    {
                        "project_id": data["projectId"].to_numpy(),
                        "outcome_specification": specification,
                        "weight_profile": profile,
                        "ranking_method": method,
                        "score": scores,
                        "rank": rank_scores(scores),
                    }
                )
            )
    return pd.concat(rows, ignore_index=True)


def ranking_agreement(rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (specification, method), group in rankings.groupby(
        ["outcome_specification", "ranking_method"]
    ):
        pivot = group.pivot(index="project_id", columns="weight_profile", values="rank")
        profiles = list(pivot.columns)
        for first, second in combinations(profiles, 2):
            top_first = set(pivot[first].nsmallest(10).index)
            top_second = set(pivot[second].nsmallest(10).index)
            rows.append(
                {
                    "outcome_specification": specification,
                    "ranking_method": method,
                    "profile_1": first,
                    "profile_2": second,
                    "spearman_rho": spearmanr(pivot[first], pivot[second]).statistic,
                    "kendall_tau": kendalltau(pivot[first], pivot[second]).statistic,
                    "top10_overlap": len(top_first & top_second),
                }
            )
    return pd.DataFrame(rows)


def cross_method_agreement(rankings: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (specification, profile), group in rankings.groupby(
        ["outcome_specification", "weight_profile"]
    ):
        pivot = group.pivot(index="project_id", columns="ranking_method", values="rank")
        top_waspas = set(pivot["WASPAS"].nsmallest(10).index)
        top_marcos = set(pivot["MARCOS"].nsmallest(10).index)
        rows.append(
            {
                "outcome_specification": specification,
                "weight_profile": profile,
                "spearman_rho": spearmanr(pivot["WASPAS"], pivot["MARCOS"]).statistic,
                "kendall_tau": kendalltau(pivot["WASPAS"], pivot["MARCOS"]).statistic,
                "top10_overlap": len(top_waspas & top_marcos),
            }
        )
    return pd.DataFrame(rows)


def outcome_sensitivity(
    secondary: pd.DataFrame,
    primary_profiles: dict[str, np.ndarray],
    secondary_profiles: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    normalized = ratio_normalize(secondary[FEATURES])
    for learner in LEARNER_ORDER:
        for method, scorer in {"WASPAS": waspas_scores, "MARCOS": marcos_scores}.items():
            primary_rank = rank_scores(scorer(normalized, primary_profiles[learner]))
            secondary_rank = rank_scores(scorer(normalized, secondary_profiles[learner]))
            primary_top = set(
                secondary.iloc[np.argsort(primary_rank)[:10]]["projectId"]
            )
            secondary_top = set(
                secondary.iloc[np.argsort(secondary_rank)[:10]]["projectId"]
            )
            rows.append(
                {
                    "learner": learner,
                    "ranking_method": method,
                    "spearman_rho": spearmanr(primary_rank, secondary_rank).statistic,
                    "kendall_tau": kendalltau(primary_rank, secondary_rank).statistic,
                    "top10_overlap": len(primary_top & secondary_top),
                }
            )
    return pd.DataFrame(rows)


def leakage_diagnostic(data: pd.DataFrame) -> pd.DataFrame:
    specifications = {
        "legacy_full_current_state": [
            "current_trl",
            "planned_duration_months",
            "remaining_trl_gap",
            "taxonomy_level",
            "taxonomy_rarity",
            "other_organization_count",
        ],
        "current_trl_removed_gap_retained": [
            "planned_duration_months",
            "remaining_trl_gap",
            "taxonomy_level",
            "taxonomy_rarity",
            "other_organization_count",
        ],
        "leakage_aware_ex_ante": FEATURES,
    }
    splitter = RepeatedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED
    )
    splits = list(splitter.split(data))
    y = data["realized_trl_progress"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for specification, features in specifications.items():
        X = data[features].reset_index(drop=True)
        for repeat in range(N_REPEATS):
            prediction = np.full(len(data), np.nan)
            repeat_splits = splits[repeat * N_SPLITS : (repeat + 1) * N_SPLITS]
            for fold, (train_index, test_index) in enumerate(repeat_splits):
                model = RandomForestRegressor(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features=1.0,
                    random_state=SEED + repeat * N_SPLITS + fold,
                    n_jobs=1,
                )
                model.fit(X.iloc[train_index], y[train_index])
                prediction[test_index] = model.predict(X.iloc[test_index])
            row = {"specification": specification, "repeat": repeat + 1}
            row.update(metric_row(y, prediction))
            rows.append(row)
    return pd.DataFrame(rows)


def generalisation_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Evaluate program-separated and chronological transport diagnostics."""
    X = data[FEATURES].reset_index(drop=True)
    y = data["realized_trl_progress"].to_numpy(dtype=float)
    groups = data["programTitle"].fillna("Unknown programme").astype(str).to_numpy()
    rows: list[dict[str, object]] = []

    for repeat in range(N_REPEATS):
        splitter = GroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED + repeat)
        split_list = list(splitter.split(X, y, groups))
        for learner, estimator in make_models(SEED + 5000 + repeat).items():
            if learner == "DummyMean":
                continue
            prediction = np.full(len(data), np.nan)
            for fold, (train_index, test_index) in enumerate(split_list):
                fitted = clone(estimator).fit(X.iloc[train_index], y[train_index])
                prediction[test_index] = fitted.predict(X.iloc[test_index])
            row = {
                "diagnostic": "programme_grouped_5_fold",
                "learner": learner,
                "repeat": repeat + 1,
                "test_n": len(data),
            }
            row.update(metric_row(y, prediction))
            rows.append(row)

    ordered = data.sort_values(["startDate", "projectId"]).index.to_numpy()
    split_point = int(math.floor(0.8 * len(ordered)))
    train_labels = ordered[:split_point]
    test_labels = ordered[split_point:]
    train_index = data.index.get_indexer(train_labels)
    test_index = data.index.get_indexer(test_labels)
    for learner, estimator in make_models(SEED + 9000).items():
        if learner == "DummyMean":
            continue
        fitted = clone(estimator).fit(X.iloc[train_index], y[train_index])
        prediction = fitted.predict(X.iloc[test_index])
        row = {
            "diagnostic": "chronological_earliest_80_latest_20",
            "learner": learner,
            "repeat": 1,
            "test_n": len(test_index),
        }
        row.update(metric_row(y[test_index], prediction))
        rows.append(row)
    return pd.DataFrame(rows)


def monte_carlo_robustness(
    data: pd.DataFrame, base_weights: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(SEED)
    normalized = ratio_normalize(data[FEATURES])
    summary_rows: list[dict[str, object]] = []
    acceptability = {
        method: pd.DataFrame(
            {
                "project_id": data["projectId"].to_numpy(),
                "top1_count": np.zeros(len(data), dtype=int),
                "top10_count": np.zeros(len(data), dtype=int),
            }
        )
        for method in ["WASPAS", "MARCOS"]
    }
    scorers = {"WASPAS": waspas_scores, "MARCOS": marcos_scores}
    baseline = {
        method: rank_scores(scorer(normalized, base_weights))
        for method, scorer in scorers.items()
    }
    baseline_top = {
        method: set(np.argsort(ranks)[:10]) for method, ranks in baseline.items()
    }

    for draw in range(N_WEIGHT_DRAWS):
        weights = base_weights * rng.uniform(0.8, 1.2, len(base_weights))
        weights = weights / weights.sum()
        for method, scorer in scorers.items():
            rank = rank_scores(scorer(normalized, weights))
            top_indices = np.argsort(rank)[:10]
            acceptability[method].loc[top_indices[0], "top1_count"] += 1
            acceptability[method].loc[top_indices, "top10_count"] += 1
            summary_rows.append(
                {
                    "draw": draw + 1,
                    "ranking_method": method,
                    "spearman_rho_with_baseline": spearmanr(
                        baseline[method], rank
                    ).statistic,
                    "kendall_tau_with_baseline": kendalltau(
                        baseline[method], rank
                    ).statistic,
                    "top10_overlap_with_baseline": len(
                        baseline_top[method] & set(top_indices)
                    ),
                }
            )
    acceptability_rows: list[pd.DataFrame] = []
    for method, frame in acceptability.items():
        frame["ranking_method"] = method
        frame["top1_acceptability"] = frame["top1_count"] / N_WEIGHT_DRAWS
        frame["top10_acceptability"] = frame["top10_count"] / N_WEIGHT_DRAWS
        acceptability_rows.append(frame)
    return pd.DataFrame(summary_rows), pd.concat(acceptability_rows, ignore_index=True)


def feature_associations(data: pd.DataFrame, outcome: str, specification: str) -> pd.DataFrame:
    rows = []
    for feature in FEATURES:
        result = spearmanr(data[feature], data[outcome])
        rows.append(
            {
                "outcome_specification": specification,
                "feature": feature,
                "feature_label": FEATURE_LABELS[feature],
                "spearman_rho": result.statistic,
                "p_value_unadjusted": result.pvalue,
            }
        )
    return pd.DataFrame(rows)


def save_figures(
    figure_dir: Path,
    performance_summary: pd.DataFrame,
    importance_summary: pd.DataFrame,
    weight_agreement: pd.DataFrame,
    ranking_agreement_table: pd.DataFrame,
    leakage: pd.DataFrame,
    monte_carlo: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "svg.hashsalt": "aicw-v3",
        }
    )
    colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

    # Figure 1: compact workflow.
    fig, ax = plt.subplots(figsize=(11.5, 3.0))
    ax.axis("off")
    steps = [
        "Frozen TechPort\nsnapshot",
        "Leakage audit &\noutcome design",
        "Repeated 5-fold\ncross-validation",
        "Held-out permutation\nimportance",
        "WASPAS & MARCOS\npropagation",
        "Outcome, learner &\nweight robustness",
    ]
    x_positions = np.linspace(0.08, 0.92, len(steps))
    for index, (x_position, label) in enumerate(zip(x_positions, steps, strict=True)):
        ax.text(
            x_position,
            0.5,
            label,
            ha="center",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.5",
                "facecolor": "#F4F7FA",
                "edgecolor": "#365F91",
                "linewidth": 1.2,
            },
        )
        if index < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.065, 0.5),
                xytext=(x_position + 0.065, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#365F91", "lw": 1.4},
            )
    fig.tight_layout()
    save_png_atomic(fig, figure_dir / "Figure1_workflow.png", bbox_inches="tight")
    fig.savefig(figure_dir / "Figure1_workflow.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

    primary_performance = performance_summary.loc[
        performance_summary["outcome_specification"].eq("primary_progress")
        & performance_summary["learner"].isin(LEARNER_ORDER)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for axis, metric, title in zip(
        axes, ["r2", "mae"], ["Out-of-sample R²", "Out-of-sample MAE"], strict=True
    ):
        subset = primary_performance.loc[primary_performance["metric"].eq(metric)].set_index(
            "learner"
        ).reindex(LEARNER_ORDER)
        values = subset["mean"].to_numpy()
        errors = np.vstack(
            [values - subset["ci95_low"].to_numpy(), subset["ci95_high"].to_numpy() - values]
        )
        axis.bar(range(len(values)), values, color=colors, edgecolor="black", linewidth=0.5)
        axis.errorbar(range(len(values)), values, yerr=errors, fmt="none", ecolor="black", capsize=3)
        axis.axhline(0, color="#666666", lw=0.8)
        axis.set_xticks(range(len(values)), ["Elastic\nNet", "Random\nForest", "Extra\nTrees", "Hist. Gradient\nBoosting"])
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_png_atomic(fig, figure_dir / "Figure2_predictive_performance.png", bbox_inches="tight")
    fig.savefig(figure_dir / "Figure2_predictive_performance.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

    primary_importance = importance_summary.loc[
        importance_summary["outcome_specification"].eq("primary_progress")
    ]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    x = np.arange(len(FEATURES))
    width = 0.19
    for learner_index, (learner, color) in enumerate(zip(LEARNER_ORDER, colors, strict=True)):
        subset = primary_importance.loc[primary_importance["learner"].eq(learner)].set_index(
            "feature"
        ).reindex(FEATURES)
        offset = (learner_index - 1.5) * width
        values = subset["mean_weight"].to_numpy()
        errors = np.vstack(
            [values - subset["ci95_low"].to_numpy(), subset["ci95_high"].to_numpy() - values]
        )
        ax.bar(x + offset, values, width, label=learner, color=color, edgecolor="black", linewidth=0.4)
        ax.errorbar(x + offset, values, yerr=errors, fmt="none", ecolor="black", capsize=2, lw=0.7)
    ax.set_xticks(x, [FEATURE_LABELS[feature] for feature in FEATURES], rotation=22, ha="right")
    ax.set_ylabel("Normalized held-out permutation importance")
    ax.set_title("Learner-specific proxy-importance profiles (primary outcome)")
    ax.legend(ncol=2, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_png_atomic(fig, figure_dir / "Figure3_proxy_weights.png", bbox_inches="tight")
    fig.savefig(figure_dir / "Figure3_proxy_weights.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

    weight_pivot = pd.DataFrame(np.eye(len(LEARNER_ORDER)), index=LEARNER_ORDER, columns=LEARNER_ORDER)
    for _, row in weight_agreement.loc[weight_agreement["context"].eq("primary_weights")].iterrows():
        weight_pivot.loc[row["profile_1"], row["profile_2"]] = row["spearman_rho"]
        weight_pivot.loc[row["profile_2"], row["profile_1"]] = row["spearman_rho"]
    rank_subset = ranking_agreement_table.loc[
        ranking_agreement_table["outcome_specification"].eq("primary_progress")
        & ranking_agreement_table["ranking_method"].eq("WASPAS")
        & ranking_agreement_table["profile_1"].isin(LEARNER_ORDER)
        & ranking_agreement_table["profile_2"].isin(LEARNER_ORDER)
    ]
    rank_pivot = pd.DataFrame(np.eye(len(LEARNER_ORDER)), index=LEARNER_ORDER, columns=LEARNER_ORDER)
    for _, row in rank_subset.iterrows():
        rank_pivot.loc[row["profile_1"], row["profile_2"]] = row["spearman_rho"]
        rank_pivot.loc[row["profile_2"], row["profile_1"]] = row["spearman_rho"]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    for axis, matrix, title in zip(
        axes,
        [weight_pivot, rank_pivot],
        ["Weight-profile agreement", "WASPAS ranking agreement"],
        strict=True,
    ):
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdYlBu")
        axis.set_xticks(range(len(LEARNER_ORDER)), ["EN", "RF", "ET", "HGB"])
        axis.set_yticks(range(len(LEARNER_ORDER)), ["EN", "RF", "ET", "HGB"])
        axis.set_title(title)
        for row in range(len(LEARNER_ORDER)):
            for column in range(len(LEARNER_ORDER)):
                axis.text(column, row, f"{matrix.iloc[row, column]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Spearman correlation")
    save_png_atomic(fig, figure_dir / "Figure4_learner_agreement.png", bbox_inches="tight")
    fig.savefig(figure_dir / "Figure4_learner_agreement.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

    leakage_summary = leakage.groupby("specification")[["r2", "mae"]].agg(["mean", "std"])
    labels = ["Legacy full", "Current TRL\nremoved only", "Leakage-aware\nex ante"]
    order = [
        "legacy_full_current_state",
        "current_trl_removed_gap_retained",
        "leakage_aware_ex_ante",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.1))
    for axis, metric, title in zip(axes, ["r2", "mae"], ["Repeated-CV R²", "Repeated-CV MAE"], strict=True):
        means = leakage_summary.loc[order, (metric, "mean")].to_numpy()
        standard_deviations = leakage_summary.loc[order, (metric, "std")].to_numpy()
        axis.bar(range(3), means, yerr=standard_deviations, color=["#D55E00", "#F0E442", "#009E73"], capsize=3, edgecolor="black", linewidth=0.5)
        axis.set_xticks(range(3), labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_png_atomic(fig, figure_dir / "Figure5_leakage_diagnostic.png", bbox_inches="tight")
    fig.savefig(figure_dir / "Figure5_leakage_diagnostic.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.1))
    for axis, method, color in zip(axes, ["WASPAS", "MARCOS"], ["#0072B2", "#009E73"], strict=True):
        subset = monte_carlo.loc[monte_carlo["ranking_method"].eq(method)]
        axis.hist(subset["spearman_rho_with_baseline"], bins=25, color=color, edgecolor="white")
        axis.set_title(method)
        axis.set_xlabel("Spearman correlation with baseline")
        axis.set_ylabel("Perturbation draws")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Bounded ±20% weight-perturbation robustness", y=1.02)
    fig.tight_layout()
    save_png_atomic(fig, figure_dir / "Figure6_weight_perturbation.png", bbox_inches="tight")
    fig.savefig(figure_dir / "Figure6_weight_perturbation.svg", bbox_inches="tight", metadata={"Date": None})
    plt.close(fig)

    # Atomic exports already produced opaque RGB PNGs. Validate without
    # reopening the output path for writing or accepting partial images.
    for stem in PUBLICATION_FIGURE_STEMS:
        path = figure_dir / f"{stem}.png"
        # Never accept a truncated PNG and silently pad the missing rows.
        with Image.open(path) as check:
            check.verify()
        with Image.open(path) as source:
            source.load()
            if source.mode != "RGB":
                raise RuntimeError(f"Non-opaque publication export: {path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    raw, primary, secondary = prepare_data(args.raw_data)
    primary.to_csv(args.output_dir / "analysis_sample_primary.csv", index=False)
    secondary.to_csv(args.output_dir / "analysis_sample_secondary.csv", index=False)

    primary_results = repeated_cv_experiment(
        primary, FEATURES, "realized_trl_progress", "primary_progress"
    )
    secondary_results = repeated_cv_experiment(
        secondary, FEATURES, "trl_goal_attainment", "secondary_goal_attainment"
    )
    fold_metrics = pd.concat([primary_results[0], secondary_results[0]], ignore_index=True)
    repeat_metrics = pd.concat([primary_results[1], secondary_results[1]], ignore_index=True)
    performance_summary = pd.concat([primary_results[2], secondary_results[2]], ignore_index=True)
    importances = pd.concat([primary_results[3], secondary_results[3]], ignore_index=True)
    importance_summary = summarize_importance(importances)

    primary_matrix = weight_matrix(importance_summary, "primary_progress")
    secondary_matrix = weight_matrix(importance_summary, "secondary_goal_attainment")
    primary_profiles = {column: primary_matrix[column].to_numpy() for column in primary_matrix}
    secondary_profiles = {column: secondary_matrix[column].to_numpy() for column in secondary_matrix}
    primary_profiles["CrossLearnerMedian"] = np.median(
        np.column_stack([primary_profiles[name] for name in LEARNER_ORDER]), axis=1
    )
    primary_profiles["CrossLearnerMedian"] /= primary_profiles["CrossLearnerMedian"].sum()
    secondary_profiles["CrossLearnerMedian"] = np.median(
        np.column_stack([secondary_profiles[name] for name in LEARNER_ORDER]), axis=1
    )
    secondary_profiles["CrossLearnerMedian"] /= secondary_profiles["CrossLearnerMedian"].sum()

    all_primary_profiles = {
        **primary_profiles,
        **conventional_profiles(primary[FEATURES]),
    }
    all_secondary_profiles = {
        **secondary_profiles,
        **conventional_profiles(secondary[FEATURES]),
    }

    rankings_primary = create_rankings(primary, all_primary_profiles, "primary_progress")
    rankings_secondary = create_rankings(
        secondary, all_secondary_profiles, "secondary_goal_attainment"
    )
    rankings = pd.concat([rankings_primary, rankings_secondary], ignore_index=True)
    ranking_agreement_table = ranking_agreement(rankings)
    cross_method = cross_method_agreement(rankings)

    weight_agreement = pd.concat(
        [
            agreement_rows(
                {name: primary_profiles[name] for name in LEARNER_ORDER},
                "primary_weights",
            ),
            agreement_rows(
                {name: secondary_profiles[name] for name in LEARNER_ORDER},
                "secondary_weights",
            ),
        ],
        ignore_index=True,
    )
    outcome_weight_rows = []
    for learner in LEARNER_ORDER:
        a = primary_profiles[learner]
        b = secondary_profiles[learner]
        outcome_weight_rows.append(
            {
                "learner": learner,
                "spearman_rho": spearmanr(a, b).statistic,
                "kendall_tau": kendalltau(a, b).statistic,
                "l1_distance": np.abs(a - b).sum(),
            }
        )
    outcome_weight_agreement = pd.DataFrame(outcome_weight_rows)

    outcome_ranking_sensitivity = outcome_sensitivity(
        secondary, primary_profiles, secondary_profiles
    )
    leakage = leakage_diagnostic(primary)
    generalisation = generalisation_diagnostics(primary)
    monte_carlo, rank_acceptability = monte_carlo_robustness(
        primary, primary_profiles["CrossLearnerMedian"]
    )
    associations = pd.concat(
        [
            feature_associations(primary, "realized_trl_progress", "primary_progress"),
            feature_associations(
                secondary, "trl_goal_attainment", "secondary_goal_attainment"
            ),
        ],
        ignore_index=True,
    )

    profile_rows = []
    for specification, profiles in [
        ("primary_progress", all_primary_profiles),
        ("secondary_goal_attainment", all_secondary_profiles),
    ]:
        for profile, vector in profiles.items():
            for feature, value in zip(FEATURES, vector, strict=True):
                profile_rows.append(
                    {
                        "outcome_specification": specification,
                        "weight_profile": profile,
                        "feature": feature,
                        "feature_label": FEATURE_LABELS[feature],
                        "weight": value,
                    }
                )
    profiles_long = pd.DataFrame(profile_rows)

    dataset_summary = pd.DataFrame(
        [
            ["Frozen records", len(raw)],
            ["Complete current-state records", raw.dropna(subset=["initial_trl", "current_trl", "planned_trl_gain"]).shape[0]],
            ["Primary completed-project sample", len(primary)],
            ["Secondary positive-target-gain sample", len(secondary)],
            ["Primary progress mean", primary["realized_trl_progress"].mean()],
            ["Primary progress standard deviation", primary["realized_trl_progress"].std(ddof=1)],
            ["Secondary attainment mean", secondary["trl_goal_attainment"].mean()],
        ],
        columns=["metric", "value"],
    )

    decision_matrix_primary = primary[
        ["projectId", *FEATURES, "realized_trl_progress"]
    ].rename(columns={"projectId": "project_id"})
    decision_matrix_secondary = secondary[
        ["projectId", *FEATURES, "trl_goal_attainment"]
    ].rename(columns={"projectId": "project_id"})

    outputs = {
        "dataset_summary.csv": dataset_summary,
        "decision_matrix_primary.csv": decision_matrix_primary,
        "decision_matrix_secondary.csv": decision_matrix_secondary,
        "cv_fold_metrics.csv": fold_metrics,
        "cv_repeat_metrics.csv": repeat_metrics,
        "predictive_performance_summary.csv": performance_summary,
        "paired_performance_tests.csv": paired_performance_tests(repeat_metrics),
        "permutation_importance_folds.csv": importances,
        "proxy_weight_summary.csv": importance_summary,
        "weight_profiles.csv": profiles_long,
        "learner_weight_agreement.csv": weight_agreement,
        "outcome_weight_agreement.csv": outcome_weight_agreement,
        "feature_outcome_associations.csv": associations,
        "rankings.csv": rankings,
        "ranking_agreement.csv": ranking_agreement_table,
        "cross_method_agreement.csv": cross_method,
        "outcome_ranking_sensitivity.csv": outcome_ranking_sensitivity,
        "leakage_diagnostic.csv": leakage,
        "generalisation_diagnostics.csv": generalisation,
        "monte_carlo_weight_perturbation.csv": monte_carlo,
        "rank_acceptability.csv": rank_acceptability,
    }
    for filename, frame in outputs.items():
        frame.to_csv(args.output_dir / filename, index=False)

    archive_path = args.output_dir / "full_results.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(args.output_dir.glob("*.csv")):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 8, 11, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes())

    save_figures(
        args.figure_dir,
        performance_summary,
        importance_summary,
        weight_agreement,
        ranking_agreement_table,
        leakage,
        monte_carlo,
    )
    from render_revision_workflow import render as render_revision_workflow
    render_revision_workflow(args.figure_dir)

    manifest = {
        "analysis_seed": SEED,
        "repeated_cv": {"folds": N_SPLITS, "repeats": N_REPEATS},
        "permutation_repeats_per_outer_fold": N_PERMUTATIONS,
        "weight_perturbation_draws": N_WEIGHT_DRAWS,
        "raw_data_sha256": sha256(args.raw_data),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "outputs": {
            path.name: sha256(path)
            for path in sorted(args.output_dir.glob("*.csv"))
        },
        "compressed_full_results": {
            archive_path.name: sha256(archive_path)
        },
        "figures": {
            path.name: sha256(path)
            for stem in PUBLICATION_FIGURE_STEMS
            for path in (args.figure_dir / f"{stem}.png", args.figure_dir / f"{stem}.svg")
        },
    }
    (args.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": "ok", "primary_n": len(primary), "secondary_n": len(secondary)}))


if __name__ == "__main__":
    main()
