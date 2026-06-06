import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import hopsworks

HOPSWORKS_KEY     = os.getenv("HOPSWORKS_API_KEY", "YOUR_HOPSWORKS_KEY_HERE")
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "aqi_predictorrr")

FEATURE_COLS = [
    "pm25", "pm10", "o3", "no2", "so2", "co",
    "temperature", "humidity", "wind_speed", "pressure",
    "hour", "day_of_week", "month", "is_weekend", "aqi_category"
]
TARGET_COL = "aqi"


def load_features():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    print("Loading features from Feature Store...")
    fg = fs.get_feature_group(name="aqi_features", version=1)
    df = fg.read()
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    return df, project


def prepare_data(df):
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    df["humidity"] = df["humidity"].clip(lower=0)
    print(f"   Dataset after cleaning: {len(df)} rows")
    print(f"   AQI range: {df[TARGET_COL].min():.1f} - {df[TARGET_COL].max():.1f}")
    print(f"   AQI mean:  {df[TARGET_COL].mean():.1f}")
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"   Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    return X_train, X_test, y_train, y_test


def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse  = np.sqrt(mean_squared_error(y_test, preds))
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"   {name:<30} RMSE={rmse:.2f}  MAE={mae:.2f}  R2={r2:.4f}")
    return {"name": name, "model": model, "rmse": rmse, "mae": mae, "r2": r2, "preds": preds}


def train_all_models(X_train, X_test, y_train, y_test):
    print("\nTraining models...\n")
    models = {
        "Ridge Regression": Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "Random Forest": RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_split=5, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        result = evaluate_model(name, model, X_test, y_test)
        results.append(result)
    best = min(results, key=lambda x: x["rmse"])
    print(f"\nBest model: {best['name']} (RMSE={best['rmse']:.2f}, R2={best['r2']:.4f})")
    return best, results


def plot_feature_importance(best):
    model = best["model"]
    estimator = model.named_steps["model"] if hasattr(model, "named_steps") else model
    if not hasattr(estimator, "feature_importances_"):
        return
    importances = estimator.feature_importances_
    feat_df = pd.DataFrame({"feature": FEATURE_COLS, "importance": importances}).sort_values("importance", ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    plt.xlabel("Importance")
    plt.title(f"Feature Importance - {best['name']}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)
    plt.close()
    print("   Saved: feature_importance.png")


def plot_predictions(best, y_test):
    preds = best["preds"]
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.4, color="teal", edgecolors="k", linewidths=0.3)
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    plt.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Perfect fit")
    plt.xlabel("Actual AQI")
    plt.ylabel("Predicted AQI")
    plt.title(f"Actual vs Predicted - {best['name']}")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=120)
    plt.close()
    print("   Saved: actual_vs_predicted.png")


def save_model_to_registry(best, project, metrics):
    print("\nSaving model to Hopsworks Model Registry...")
    joblib.dump(best["model"], "aqi_model.pkl")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    mr = project.get_model_registry()
    aqi_model = mr.sklearn.create_model(
        name="aqi_predictor",
        metrics=metrics,
        description=f"Best AQI prediction model: {best['name']}",
    )
    aqi_model.save("aqi_model.pkl")
    print("Model saved to registry as 'aqi_predictor'")
    print(f"   Metrics: {metrics}")


def run_training():
    print(f"\n{'='*55}")
    print(f"  AQI Training Pipeline")
    print(f"{'='*55}\n")

    df, project = load_features()

    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)

    best, all_results = train_all_models(X_train, X_test, y_train, y_test)

    print("\nGenerating plots...")
    plot_feature_importance(best)
    plot_predictions(best, y_test)

    metrics = {
        "rmse": round(float(best["rmse"]), 4),
        "mae":  round(float(best["mae"]),  4),
        "r2":   round(float(best["r2"]),   4),
    }
    save_model_to_registry(best, project, metrics)

    print(f"\nTraining pipeline complete!\n")
    return best


if __name__ == "__main__":
    best_model = run_training()
