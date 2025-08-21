#!/usr/bin/env python3
"""
Train a modern multiclass classifier on EEG band powers from eeg-data.csv

Key features:
- Filters to good-quality rows: signal_quality < threshold (default 128) and label != 'unlabeled'
- Uses subject id (column 'id') for group-aware splits to avoid subject leakage
- Parses eeg_power JSON list (8 bands), with optional engineered features
- Supports label scheme: 'full' or 'simplified' (collapse versioned/Instruction labels)
- Models: Logistic Regression (multinomial) or Random Forest
- Evaluation: accuracy, macro F1, classification report, confusion matrix
- Saves artifacts: pipeline (preprocess + model), label encoder, metrics, confusion matrix

Bands order assumption (NeuroSky/ThinkGear typical):
0: delta, 1: theta, 2: low-alpha, 3: high-alpha, 4: low-beta, 5: high-beta, 6: low-gamma, 7: mid-gamma
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier  # optional
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class Config:
    csv_path: str
    output_dir: str
    min_signal_quality: int = 128
    label_scheme: str = "simplified"  # or 'full' or 'relax_math'
    model_name: str = "logreg"  # or 'rf'
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    use_engineered: bool = True
    do_cv_tuning: bool = False
    n_splits: int = 5
    n_iters: int = 30


SIMPLIFY_MAP = [
    (re.compile(r"^video-ver[12]$"), "video"),
    (re.compile(r"^thinkOfItems-ver[12]$"), "thinkOfItems"),
    (re.compile(r"^colorInstruction[12]$"), "instruction"),
    (re.compile(r"^.*Instruction.*$"), "instruction"),
]


def simplify_label(label: str) -> str:
    for pat, repl in SIMPLIFY_MAP:
        if pat.match(label):
            return repl
    return label


def apply_label_scheme(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """Apply label mapping/filtering according to scheme.
    - 'full': no change
    - 'simplified': collapse versioned/Instruction labels
    - 'relax_math': keep only 'relax' and math tasks (math1..math12), map all to {'relax','math'}
    """
    df = df.copy()
    if scheme == "full":
        return df
    if scheme == "simplified":
        df["label"] = df["label"].astype(str).apply(simplify_label)
        return df
    if scheme == "relax_math":
        labels = df["label"].astype(str)
        is_relax = labels == "relax"
        is_math_num = labels.str.match(r"^math\d+$", na=False)
        # exclude mathInstruction or any other instruction labels
        keep = is_relax | is_math_num
        df = df.loc[keep].copy()
        # map mathN -> math
        df.loc[df["label"].astype(str).str.match(r"^math\d+$", na=False), "label"] = "math"
        return df
    raise ValueError(f"Unknown label scheme: {scheme}")


def load_and_filter(csv_path: str, min_signal_quality: int) -> pd.DataFrame:
    usecols = ["id", "eeg_power", "signal_quality", "label"]
    df = pd.read_csv(csv_path, usecols=usecols)
    # filter good quality and labeled
    df = df[(df["signal_quality"] < min_signal_quality) & df["label"].notna() & (df["label"] != "unlabeled")]
    return df.reset_index(drop=True)


def parse_eeg_power(eeg_str: str) -> List[float]:
    try:
        arr = json.loads(eeg_str)
        if not isinstance(arr, list) or len(arr) != 8:
            return [np.nan] * 8
        return [float(x) if x is not None else np.nan for x in arr]
    except Exception:
        return [np.nan] * 8


def engineer_features(bands: np.ndarray, use_engineered: bool = True) -> Tuple[np.ndarray, List[str]]:
    """
    Input bands shape: (n_samples, 8)
    Returns X, feature_names
    """
    n = bands.shape[0]
    # raw powers
    delta, theta, lalpha, halpha, lbeta, hbeta, lgamma, mgamma = [bands[:, i] for i in range(8)]
    alpha = lalpha + halpha
    beta = lbeta + hbeta
    gamma = lgamma + mgamma
    total = np.clip(delta + theta + alpha + beta + gamma, 1e-9, None)

    features = []
    names = []

    # log1p raw bands
    raw = np.log1p(bands)
    features.append(raw)
    names.extend([f"log_{b}" for b in ["delta", "theta", "lalpha", "halpha", "lbeta", "hbeta", "lgamma", "mgamma"]])

    # percentages
    pct = np.c_[delta / total, theta / total, alpha / total, beta / total, gamma / total]
    features.append(pct)
    names.extend(["pct_delta", "pct_theta", "pct_alpha", "pct_beta", "pct_gamma"])

    if use_engineered:
        # ratios (add small eps to avoid div0)
        eps = 1e-9
        ratios = np.c_[
            (alpha + eps) / (beta + eps),
            (theta + eps) / (alpha + eps),
            (beta + eps) / (gamma + eps),
            (theta + eps) / (beta + eps),
        ]
        features.append(ratios)
        names.extend(["alpha_beta", "theta_alpha", "beta_gamma", "theta_beta"])

    X = np.concatenate(features, axis=1)
    return X, names


def build_model(model_name: str, n_classes: int, seed: int):
    if model_name == "logreg":
        clf = LogisticRegression(
            multi_class="multinomial",
            class_weight="balanced",
            solver="saga",
            max_iter=5000,
            random_state=seed,
            n_jobs=1,
        )
    elif model_name == "rf":
        clf = RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "xgb":
        if not _HAS_XGB:
            raise ValueError("XGBoost is not installed. Please install xgboost to use --model xgb.")
        clf = XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob" if n_classes > 2 else "binary:logistic",
            eval_metric="mlogloss" if n_classes > 2 else "logloss",
            random_state=seed,
            n_jobs=-1,
        )
    elif model_name == "mlp":
        # Compact MLP baseline with early stopping
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])
    return pipe


def tune_model(pipe: Pipeline, model_name: str, X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int, n_iters: int):
    cv = GroupKFold(n_splits=n_splits)
    rng = np.random.RandomState(seed)
    if model_name == "rf":
        param_distributions = {
            "clf__n_estimators": [200, 300, 400, 600, 800],
            "clf__max_depth": [None, 6, 8, 10, 12, 16],
            "clf__max_features": ["sqrt", "log2", None, 0.5, 0.8],
            "clf__min_samples_leaf": [1, 2, 3, 5],
        }
    elif model_name == "xgb":
        param_distributions = {
            "clf__n_estimators": [200, 300, 400, 600],
            "clf__max_depth": [3, 4, 5, 6, 8],
            "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
            "clf__subsample": [0.6, 0.8, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0],
            "clf__reg_lambda": [0.5, 1.0, 2.0],
        }
    else:
        raise ValueError("Tuning is only implemented for rf and xgb")

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_distributions,
        n_iter=n_iters,
        scoring="f1_macro",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X, y, clf__sample_weight=None, groups=groups)
    return search.best_estimator_, search.best_params_, search.best_score_


def stratified_group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int, test_size: float, val_size: float):
    """
    Group-aware split by subject id in two stages using GroupShuffleSplit.
    Maintains approx. label stratification via random seed but avoids leakage by grouping strictly by subject.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss.split(X, y, groups))

    # Split train into train/val
    groups_train = groups[train_val_idx]
    y_train = y[train_val_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx_rel, val_idx_rel = next(gss2.split(X[train_val_idx], y_train, groups_train))

    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return train_idx, val_idx, test_idx


def plot_confmat(cm: np.ndarray, classes: List[str], out_path: str):
    plt.figure(figsize=(10, 8))
    cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_feature_importance(model: Pipeline, feature_names: List[str], out_path: str):
    clf = model.named_steps.get("clf")
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif _HAS_XGB and isinstance(clf, XGBClassifier):
        try:
            importances = clf.feature_importances_
        except Exception:
            importances = None
    if importances is None:
        return
    idx = np.argsort(importances)[::-1]
    topk = min(25, len(importances))
    plt.figure(figsize=(8, 10))
    sns.barplot(x=importances[idx][:topk], y=[feature_names[i] for i in idx[:topk]], orient="h")
    plt.title("Top feature importances")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main(cfg: Config):
    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Loading and filtering data from {cfg.csv_path}...")
    df = load_and_filter(cfg.csv_path, cfg.min_signal_quality)

    # Apply label scheme
    df = apply_label_scheme(df, cfg.label_scheme)

    # Parse eeg_power
    print("Parsing eeg_power column...")
    bands_list = df["eeg_power"].astype(str).apply(parse_eeg_power).tolist()
    bands = np.array(bands_list, dtype=float)

    # Drop rows with NaNs in bands
    mask = ~np.isnan(bands).any(axis=1)
    dropped = int((~mask).sum())
    if dropped:
        print(f"Dropping {dropped} rows with invalid eeg_power values")
    df = df.loc[mask].reset_index(drop=True)
    bands = bands[mask]

    # Features
    X, feature_names = engineer_features(bands, use_engineered=cfg.use_engineered)

    # Labels
    le = LabelEncoder()
    y = le.fit_transform(df["label"].astype(str).values)
    classes = list(le.classes_)

    # Groups: subject id
    groups = df["id"].astype(str).values

    # Splits
    print("Creating group-aware train/val/test splits by subject id...")
    train_idx, val_idx, test_idx = stratified_group_split(X, y, groups, cfg.seed, cfg.test_size, cfg.val_size)

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Model
    print(f"Building model: {cfg.model_name}")
    model = build_model(cfg.model_name, n_classes=len(classes), seed=cfg.seed)

    # Optional CV tuning using GroupKFold on train set
    if cfg.do_cv_tuning and cfg.model_name in ("rf", "xgb"):
        print("Running GroupKFold CV tuning (RandomizedSearchCV)...")
        model, best_params, best_score = tune_model(
            model, cfg.model_name, X_train, y_train, groups[train_idx], cfg.seed, cfg.n_splits, cfg.n_iters
        )
        print(f"Best CV macro F1: {best_score:.4f}\nBest params: {best_params}")

    print("Training...")
    model.fit(X_train, y_train)

    def eval_split(name, Xs, ys):
        yp = model.predict(Xs)
        acc = accuracy_score(ys, yp)
        f1m = f1_score(ys, yp, average="macro")
        print(f"{name} - acc: {acc:.4f}, macro_f1: {f1m:.4f}")
        return yp, acc, f1m

    y_pred_tr, acc_tr, f1_tr = eval_split("Train", X_train, y_train)
    y_pred_va, acc_va, f1_va = eval_split("Val", X_val, y_val)
    y_pred_te, acc_te, f1_te = eval_split("Test", X_test, y_test)

    # Reports
    print("Generating reports and saving artifacts...")
    report = classification_report(y_test, y_pred_te, target_names=classes, digits=4)
    cm = confusion_matrix(y_test, y_pred_te)

    # Save metrics
    metrics = {
        "train": {"accuracy": float(acc_tr), "macro_f1": float(f1_tr)},
        "val": {"accuracy": float(acc_va), "macro_f1": float(f1_va)},
        "test": {"accuracy": float(acc_te), "macro_f1": float(f1_te)},
        "classes": classes,
        "label_scheme": cfg.label_scheme,
        "model": cfg.model_name,
        "feature_names": feature_names,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w") as f:
        import json as _json
        _json.dump(metrics, f, indent=2)

    with open(os.path.join(cfg.output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix plot
    plot_confmat(cm, classes, os.path.join(cfg.output_dir, "confusion_matrix.png"))

    # Per-subject metrics on test set
    test_subjects = df.iloc[test_idx]["id"].astype(str).values
    import pandas as _pd
    per_subj = _pd.DataFrame({
        "subject_id": test_subjects,
        "y_true": y_test,
        "y_pred": y_pred_te,
    })
    subj_acc = per_subj.groupby("subject_id").apply(lambda g: float((g["y_true"]==g["y_pred"]).mean())).rename("accuracy").reset_index()
    subj_acc.to_csv(os.path.join(cfg.output_dir, "per_subject_accuracy.csv"), index=False)

    # Feature importance plot
    try:
        plot_feature_importance(model, feature_names, os.path.join(cfg.output_dir, "feature_importance.png"))
    except Exception as e:
        print(f"Feature importance plotting skipped: {e}")

    # Save pipeline (scaler + model) and label encoder
    joblib.dump(model, os.path.join(cfg.output_dir, "model.joblib"))
    joblib.dump(le, os.path.join(cfg.output_dir, "label_encoder.joblib"))

    print("Done. Artifacts saved to:")
    print(f"- {os.path.join(cfg.output_dir, 'model.joblib')}")
    print(f"- {os.path.join(cfg.output_dir, 'label_encoder.joblib')}")
    print(f"- {os.path.join(cfg.output_dir, 'metrics.json')}")
    print(f"- {os.path.join(cfg.output_dir, 'classification_report.txt')}")
    print(f"- {os.path.join(cfg.output_dir, 'confusion_matrix.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EEG model from eeg-data.csv")
    parser.add_argument("--csv", dest="csv_path", type=str, default="eeg-data.csv", help="Path to eeg-data.csv")
    parser.add_argument("--output-dir", type=str, default="eeg_model_artifacts", help="Directory to save artifacts")
    parser.add_argument("--min-signal-quality", type=int, default=128, help="Max signal_quality to include (strictly less than this value)")
    parser.add_argument("--label-scheme", type=str, choices=["full", "simplified", "relax_math"], default="simplified")
    parser.add_argument("--model", dest="model_name", type=str, choices=["logreg", "rf", "xgb", "mlp"], default="rf")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--no-engineered", action="store_true", help="Disable engineered features and use only log bands + percentages")
    parser.add_argument("--cv", dest="do_cv_tuning", action="store_true", help="Enable GroupKFold CV tuning (RandomizedSearchCV) for rf/xgb")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of GroupKFold splits for CV")
    parser.add_argument("--n-iters", type=int, default=30, help="Number of parameter settings sampled in RandomizedSearchCV")

    args = parser.parse_args()

    cfg = Config(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        min_signal_quality=args.min_signal_quality,
        label_scheme=args.label_scheme,
        model_name=args.model_name,
        seed=args.seed,
        test_size=args.test_size,
        val_size=args.val_size,
        use_engineered=not args.no_engineered,
        do_cv_tuning=args.do_cv_tuning,
        n_splits=args.n_splits,
        n_iters=args.n_iters,
    )

    main(cfg)
