from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"


FEATURES = [
    "ADJOE",
    "ADJDE",
    "EFG_O",
    "EFG_D",
    "TOR",
    "TORD",
    "ORB",
    "DRB",
    "FTR",
    "FTRD",
    "2P_O",
    "2P_D",
    "3P_O",
    "3P_D",
    "ADJ_T",
]
TARGET = "W"
META_COLS = ["TEAM", "CONF", "YEAR"]


def load_data() -> pd.DataFrame:
    main_path = DATA_DIR / "cleaned_mensD1bball.csv"
    df_main = pd.read_csv(main_path)

    # 2020 file may not include YEAR; add it if present
    extra_path = DATA_DIR / "cleaned_mensD1bball2020.csv"
    if extra_path.exists():
        df_2020 = pd.read_csv(extra_path)
        if "YEAR" not in df_2020.columns:
            df_2020["YEAR"] = 2020
        df = pd.concat([df_main, df_2020], ignore_index=True, sort=False)
    else:
        df = df_main

    return df


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    needed_cols = FEATURES + [TARGET, "TEAM", "YEAR"]
    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df = df.dropna(subset=FEATURES + [TARGET, "YEAR", "TEAM"])

    # Ensure YEAR is int
    df["YEAR"] = df["YEAR"].astype(int)

    return df


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    # VIF = 1 / (1 - R^2) for each feature regressed on the others
    vifs = []
    for col in X.columns:
        X_other = X.drop(columns=[col])
        y = X[col]
        if X_other.shape[1] == 0:
            vifs.append(np.nan)
            continue
        lr = LinearRegression()
        lr.fit(X_other, y)
        r2 = lr.score(X_other, y)
        if r2 >= 0.9999:
            vif = np.inf
        else:
            vif = 1.0 / (1.0 - r2)
        vifs.append(vif)

    return pd.DataFrame({"feature": X.columns, "vif": vifs}).sort_values("vif", ascending=False)


def build_models() -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {}

    models["LinearRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])

    ridge_alphas = np.logspace(-3, 3, 25)
    models["RidgeCV"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RidgeCV(alphas=ridge_alphas)),
    ])

    lasso_alphas = np.logspace(-3, 1, 25)
    models["LassoCV"] = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LassoCV(alphas=lasso_alphas, max_iter=10000)),
    ])

    return models


def evaluate_model(model, X_train, y_train, X_test, y_test) -> Tuple[Dict[str, float], np.ndarray]:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"R2": r2, "MAE": mae, "RMSE": rmse}, preds


def evaluate_baseline_mean(y_train, y_test) -> Tuple[Dict[str, float], np.ndarray]:
    mean_value = float(np.mean(y_train))
    preds = np.full_like(y_test, fill_value=mean_value, dtype=float)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"R2": r2, "MAE": mae, "RMSE": rmse}, preds


def evaluate_baseline_adj(X_train, y_train, X_test, y_test) -> Tuple[Dict[str, float], np.ndarray]:
    # Baseline using only ADJOE - ADJDE
    train_feature = (X_train["ADJOE"] - X_train["ADJDE"]).values.reshape(-1, 1)
    test_feature = (X_test["ADJOE"] - X_test["ADJDE"]).values.reshape(-1, 1)
    lr = LinearRegression()
    lr.fit(train_feature, y_train)
    preds = lr.predict(test_feature)
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return {"R2": r2, "MAE": mae, "RMSE": rmse}, preds


def run_evaluations(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]], Dict[str, np.ndarray], pd.DataFrame]:
    years = sorted(df["YEAR"].unique())
    latest_year = max(years)

    results_rows: List[Dict[str, object]] = []
    holdout_preds: Dict[str, np.ndarray] = {}
    holdout_metrics: Dict[str, Dict[str, float]] = {}

    # Season holdout: train on all years < latest, test on latest year
    train_df = df[df["YEAR"] < latest_year]
    test_df = df[df["YEAR"] == latest_year]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    # Baselines
    metrics, preds = evaluate_baseline_mean(y_train, y_test)
    results_rows.append({"model_name": "BaselineMean", "evaluation_method": f"SeasonHoldout_{latest_year}", **metrics})
    holdout_preds["BaselineMean"] = preds
    holdout_metrics["BaselineMean"] = metrics

    if "ADJOE" in FEATURES and "ADJDE" in FEATURES:
        metrics, preds = evaluate_baseline_adj(X_train, y_train, X_test, y_test)
        results_rows.append({"model_name": "Baseline_ADJOE_minus_ADJDE", "evaluation_method": f"SeasonHoldout_{latest_year}", **metrics})
        holdout_preds["Baseline_ADJOE_minus_ADJDE"] = preds
        holdout_metrics["Baseline_ADJOE_minus_ADJDE"] = metrics

    # Main models
    models = build_models()
    for name, model in models.items():
        metrics, preds = evaluate_model(model, X_train, y_train, X_test, y_test)
        results_rows.append({"model_name": name, "evaluation_method": f"SeasonHoldout_{latest_year}", **metrics})
        holdout_preds[name] = preds
        holdout_metrics[name] = metrics

    # Rolling evaluation for last 5 seasons if possible
    rolling_years = [y for y in years if y <= latest_year and y >= latest_year - 4]
    if len(rolling_years) < 3:
        rolling_years = years[-3:]

    for year in rolling_years:
        train_df = df[df["YEAR"] < year]
        test_df = df[df["YEAR"] == year]
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[FEATURES]
        y_train = train_df[TARGET]
        X_test = test_df[FEATURES]
        y_test = test_df[TARGET]

        # Baselines
        metrics, _ = evaluate_baseline_mean(y_train, y_test)
        results_rows.append({"model_name": "BaselineMean", "evaluation_method": f"Rolling_{year}", **metrics})

        if "ADJOE" in FEATURES and "ADJDE" in FEATURES:
            metrics, _ = evaluate_baseline_adj(X_train, y_train, X_test, y_test)
            results_rows.append({"model_name": "Baseline_ADJOE_minus_ADJDE", "evaluation_method": f"Rolling_{year}", **metrics})

        # Main models
        for name, model in models.items():
            metrics, _ = evaluate_model(model, X_train, y_train, X_test, y_test)
            results_rows.append({"model_name": name, "evaluation_method": f"Rolling_{year}", **metrics})

    metrics_df = pd.DataFrame(results_rows)

    return metrics_df, holdout_metrics, holdout_preds, df[df["YEAR"] == latest_year]


def choose_model(holdout_metrics: Dict[str, Dict[str, float]]) -> str:
    # Choose model with lowest MAE among the main models; prefer LinearRegression if within 0.1 wins
    candidate_names = ["LinearRegression", "RidgeCV", "LassoCV"]
    candidate_metrics = {k: v for k, v in holdout_metrics.items() if k in candidate_names}
    best = min(candidate_metrics.items(), key=lambda kv: kv[1]["MAE"])
    best_name, best_metrics = best

    # Simplicity tie-breaker
    if best_name != "LinearRegression":
        lin_mae = candidate_metrics["LinearRegression"]["MAE"]
        if abs(best_metrics["MAE"] - lin_mae) <= 0.10:
            return "LinearRegression"

    return best_name


def plot_predicted_vs_actual(y_true, y_pred, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="black")
    plt.xlabel("Actual Wins")
    plt.ylabel("Predicted Wins")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_residuals_vs_predicted(y_pred, residuals, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", color="black")
    plt.xlabel("Predicted Wins")
    plt.ylabel("Residual (Pred - Actual)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_residuals_hist(residuals, path: Path, title: str) -> None:
    plt.figure(figsize=(7, 6))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual (Pred - Actual)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_coefficients(model: Pipeline, feature_names: List[str], path: Path, title: str) -> None:
    coef = model.named_steps["model"].coef_
    coef_df = pd.DataFrame({"feature": feature_names, "coef": coef})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(coef_df["feature"], coef_df["coef"])
    plt.xlabel("Standardized Coefficient")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_learning_curve(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    path: Path,
    title: str,
) -> pd.DataFrame:
    n_splits = min(5, groups.nunique())
    if n_splits < 2:
        raise ValueError("Need at least 2 groups for learning curve.")

    cv = GroupKFold(n_splits=n_splits)
    train_sizes = np.linspace(0.1, 1.0, 6)

    sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        groups=groups,
        cv=cv,
        scoring="neg_mean_absolute_error",
        train_sizes=train_sizes,
    )

    train_mae = -train_scores.mean(axis=1)
    val_mae = -val_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure(figsize=(7, 6))
    plt.plot(sizes, train_mae, marker="o", label="Train MAE")
    plt.plot(sizes, val_mae, marker="o", label="CV MAE")
    plt.fill_between(sizes, train_mae - train_std, train_mae + train_std, alpha=0.15)
    plt.fill_between(sizes, val_mae - val_std, val_mae + val_std, alpha=0.15)
    plt.xlabel("Training Set Size")
    plt.ylabel("MAE (Wins)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    return pd.DataFrame({
        "train_size": sizes,
        "train_mae": train_mae,
        "train_mae_std": train_std,
        "cv_mae": val_mae,
        "cv_mae_std": val_std,
    })


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_data()
    df = prepare_dataset(df_raw)

    # Correlation matrix and VIF for notebook usage
    corr_matrix = df[FEATURES + [TARGET]].corr()
    corr_matrix.to_csv(RESULTS_DIR / "feature_correlations.csv", index=True)

    vif_df = compute_vif(df[FEATURES])
    vif_df.to_csv(RESULTS_DIR / "vif.csv", index=False)

    metrics_df, holdout_metrics, holdout_preds, holdout_df = run_evaluations(df)
    metrics_path = RESULTS_DIR / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    chosen_model_name = choose_model(holdout_metrics)

    # Refit chosen model on holdout split to get predictions and plots
    latest_year = holdout_df["YEAR"].max()
    train_df = df[df["YEAR"] < latest_year]
    test_df = df[df["YEAR"] == latest_year]

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    models = build_models()
    chosen_model = models[chosen_model_name]
    chosen_model.fit(X_train, y_train)
    preds = chosen_model.predict(X_test)

    # Save plots
    plot_predicted_vs_actual(
        y_test,
        preds,
        FIG_DIR / "predicted_vs_actual_holdout.png",
        title=f"Predicted vs Actual Wins (Holdout {latest_year})",
    )

    residuals = preds - y_test.values
    plot_residuals_vs_predicted(
        preds,
        residuals,
        FIG_DIR / "residuals_vs_predicted_holdout.png",
        title=f"Residuals vs Predicted (Holdout {latest_year})",
    )

    plot_residuals_hist(
        residuals,
        FIG_DIR / "residuals_hist_holdout.png",
        title=f"Residuals Distribution (Holdout {latest_year})",
    )

    plot_coefficients(
        chosen_model,
        FEATURES,
        FIG_DIR / "standardized_coefficients.png",
        title=f"Standardized Coefficients ({chosen_model_name})",
    )

    # Learning curve for underfitting/overfitting check
    lc_df = plot_learning_curve(
        chosen_model,
        df[FEATURES],
        df[TARGET],
        df["YEAR"],
        FIG_DIR / "learning_curve_mae.png",
        title=f"Learning Curve (MAE) - {chosen_model_name}",
    )
    lc_df.to_csv(RESULTS_DIR / "learning_curve_mae.csv", index=False)

    # Top errors
    errors_df = pd.DataFrame({
        "team": test_df["TEAM"].values,
        "season": test_df["YEAR"].values,
        "actual_wins": y_test.values,
        "predicted_wins": preds,
    })
    errors_df["error"] = errors_df["predicted_wins"] - errors_df["actual_wins"]

    over = errors_df.sort_values("error", ascending=False).head(10)
    under = errors_df.sort_values("error", ascending=True).head(10)
    top_errors = pd.concat([over, under], ignore_index=True)
    top_errors.to_csv(RESULTS_DIR / "top_errors.csv", index=False)

    # Save summary of chosen model for quick reference
    summary = {
        "chosen_model": chosen_model_name,
        "holdout_year": int(latest_year),
        "holdout_R2": float(r2_score(y_test, preds)),
        "holdout_MAE": float(mean_absolute_error(y_test, preds)),
        "holdout_RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "holdout_summary.csv", index=False)


if __name__ == "__main__":
    main()
