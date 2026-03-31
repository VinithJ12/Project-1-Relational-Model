import duckdb
import logging
import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.ensemble         import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection  import train_test_split
from sklearn.metrics          import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score,
    mean_squared_error, r2_score
)
from sklearn.preprocessing    import LabelEncoder
 
# CONFIGURATION 
DB_PATH     = "fema.db"
LOG_PATH    = "pipeline.log"
FIGURES_DIR = "figures"
DATA_DIR    = "data"
 
RANDOM_STATE = 42   # for reproducibility
 
# LOGGING SETUP
 
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
 
 
# HELPERS
 
def save_fig(filename: str) -> None:
    """Save current matplotlib figure to figures/."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path}")
    print(f"  Saved: {path}")
 
 
# STEP 1 — LOAD FEATURE MATRIX
 
def load_features() -> pd.DataFrame:
    """
    Loads ml_features.csv produced by analysis.py.
    Encodes the state column as integers so sklearn can use it.
    Drops any remaining rows with nulls (should be very few).
    """
    path = os.path.join(DATA_DIR, "ml_features.csv")
    try:
        df = pd.read_csv(path)
        logger.info(f"Loaded ml_features.csv: {len(df):,} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        logger.error(f"ml_features.csv not found — run analysis.py first")
        raise
 
    # Encode state as integer category
    le = LabelEncoder()
    df["state_encoded"] = le.fit_transform(df["state"].astype(str))
    df = df.drop(columns=["state"])
 
    # Drop any rows still containing nulls
    before = len(df)
    df = df.dropna()
    after  = len(df)
    if before - after > 0:
        logger.warning(f"Dropped {before - after} rows with remaining nulls")
 
    logger.info(f"Feature matrix ready: {len(df):,} rows")
    return df
 
 
# STEP 2 — MODEL 1: RANDOM FOREST CLASSIFIER
# Predicts: habitabilityRepairsRequired (0 or 1)
 
def train_classifier(df: pd.DataFrame) -> tuple:
    """
    Trains a Random Forest classifier to predict whether a household
    requires habitability repairs.
 
    Key decisions:
    - class_weight='balanced': compensates for 61/39 class imbalance
      by upweighting the minority class during training
    - n_estimators=100: 100 trees, standard starting point
    - max_depth=10: prevents overfitting on 500k row dataset
    - test_size=0.2: 80/20 train/test split, standard practice
    - random_state=42: ensures reproducible results
 
    Features used: all columns except target and rpfvl
    (rpfvl is a target for Model 2, not a predictor here)
    """
    logger.info("Training Random Forest classifier...")
 
    # Define features and target
    drop_cols = ["target", "rpfvl"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["target"].astype(int)
 
    feature_names = X.columns.tolist()
 
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
 
    # Train model
    clf = RandomForestClassifier(
        n_estimators  = 100,
        max_depth     = 10,
        class_weight  = "balanced",
        random_state  = RANDOM_STATE,
        n_jobs        = -1          # use all CPU cores
    )
    clf.fit(X_train, y_train)
    logger.info("Random Forest training complete")
 
    # Evaluate
    y_pred = clf.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred)
 
    print(f"\n── Model 1: Random Forest Classifier ──")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(y_test, y_pred,
          target_names=["No repairs", "Repairs required"]))
 
    logger.info(f"Classifier — Accuracy: {acc:.4f} | F1: {f1:.4f}")
 
    return clf, X_test, y_test, y_pred, feature_names
 
 
def plot_confusion_matrix(y_test, y_pred) -> None:
    """
    Plots a normalized confusion matrix showing what the model
    got right and wrong.
 
    Rationale: raw counts are hard to interpret at 500k scale.
    Normalization shows the *rate* of correct/incorrect predictions
    which is more meaningful for comparing across classes.
    """
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    labels = ["No repairs", "Repairs required"]
 
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8}
    )
    ax.set_xlabel("Predicted label", fontsize=10)
    ax.set_ylabel("True label", fontsize=10)
    ax.set_title(
        "Confusion matrix — Random Forest classifier\n(normalized by true class)",
        fontsize=11, fontweight="bold", pad=12
    )
    plt.tight_layout()
    save_fig("confusion_matrix.png")
 
 
def plot_feature_importance(clf, feature_names: list) -> None:
    """
    Horizontal bar chart of top 15 feature importances from
    the Random Forest model.
 
    Rationale: feature importance tells us which household and
    damage characteristics most strongly predict whether a home
    needs habitability repairs. This is the core analytical result
    of the project and the recommended press release chart.
 
    Feature importance in Random Forest = mean decrease in impurity
    across all trees, normalized to sum to 1.
    """
    importances = pd.Series(
        clf.feature_importances_, index=feature_names
    ).sort_values(ascending=True).tail(15)
 
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#D85A30" if i >= importances.max() * 0.5
              else "#378ADD" for i in importances]
    importances.plot(kind="barh", ax=ax, color=colors, edgecolor="none")
 
    ax.set_xlabel("Feature importance (mean decrease in impurity)", fontsize=10)
    ax.set_title(
        "Top 15 predictors of habitability repair requirements\nRandom Forest — FEMA Individual Assistance dataset",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
 
    # Add value labels
    for i, (val, name) in enumerate(zip(importances.values, importances.index)):
        ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=8)
 
    plt.tight_layout()
    save_fig("feature_importance.png")
 
 
# STEP 3 — MODEL 2: GRADIENT BOOSTING REGRESSOR
# Predicts: rpfvl (real property field visit loss)
 
def train_regressor(df: pd.DataFrame) -> None:
    """
    Trains a Gradient Boosting regressor to predict real property
    field visit loss (rpfvl) — the dollar value of property damage.
 
    Key decisions:
    - Filter to rpfvl > 0: only households with actual damage
      (zero rpfvl means no recorded property loss — not useful
      for predicting damage severity)
    - log1p transform on target: rpfvl is heavily right-skewed
      (most values small, a few very large). Log transform
      stabilizes variance and improves model fit.
    - n_estimators=100, max_depth=4: conservative settings
      to avoid overfitting on financial data
    - Subsample of 100k rows for speed — regressor is slower
      than classifier on large datasets
    """
    logger.info("Training Gradient Boosting regressor...")
 
    # Filter to rows with actual property damage
    df_reg = df[df["rpfvl"] > 0].copy()
    logger.info(f"Regressor training set: {len(df_reg):,} rows with rpfvl > 0")
 
    # Log-transform the target
    df_reg["rpfvl_log"] = np.log1p(df_reg["rpfvl"])
 
    # Subsample for speed
    if len(df_reg) > 100000:
        df_reg = df_reg.sample(100000, random_state=RANDOM_STATE)
        logger.info("Subsampled to 100,000 rows for regressor training")
 
    # Features and target
    drop_cols = ["target", "rpfvl", "rpfvl_log"]
    X = df_reg.drop(columns=[c for c in drop_cols if c in df_reg.columns])
    y = df_reg["rpfvl_log"]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
 
    # Train model
    reg = GradientBoostingRegressor(
        n_estimators = 100,
        max_depth    = 4,
        learning_rate= 0.1,
        subsample    = 0.8,
        random_state = RANDOM_STATE
    )
    reg.fit(X_train, y_train)
    logger.info("Gradient Boosting training complete")
 
    # Evaluate on log scale
    y_pred_log = reg.predict(X_test)
    rmse_log   = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2         = r2_score(y_test, y_pred_log)
 
    # Convert back to dollar scale for interpretability
    y_pred_dollars = np.expm1(y_pred_log)
    y_test_dollars = np.expm1(y_test)
    rmse_dollars   = np.sqrt(mean_squared_error(y_test_dollars, y_pred_dollars))
 
    print(f"\n── Model 2: Gradient Boosting Regressor ──")
    print(f"  R² Score       : {r2:.4f}")
    print(f"  RMSE (log scale): {rmse_log:.4f}")
    print(f"  RMSE (dollars) : ${rmse_dollars:,.0f}")
    print(f"  Median actual  : ${y_test_dollars.median():,.0f}")
    print(f"  Median predicted: ${pd.Series(y_pred_dollars).median():,.0f}")
 
    logger.info(f"Regressor — R²: {r2:.4f} | RMSE: ${rmse_dollars:,.0f}")
 
    # Plot predicted vs actual
    plot_predicted_vs_actual(y_test_dollars, y_pred_dollars)
 
 
def plot_predicted_vs_actual(y_test, y_pred) -> None:
    """
    Scatter plot of predicted vs actual property loss values.
    Capped at $50k for readability (removes extreme outliers).
 
    Rationale: a predicted vs actual plot is the standard way to
    evaluate regression models visually. Points close to the
    diagonal = good predictions. The diagonal line is the
    "perfect model" reference.
    """
    y_test = pd.Series(y_test).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)
 
    # Cap at 50k for visualization clarity
    mask   = (y_test < 50000) & (y_pred < 50000)
    y_t    = y_test[mask]
    y_p    = y_pred[mask]
 
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_t, y_p, alpha=0.15, s=4, color="#378ADD", rasterized=True)
 
    # Perfect prediction line
    max_val = max(y_t.max(), y_p.max())
    ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="Perfect prediction")
 
    ax.set_xlabel("Actual property loss (USD)", fontsize=10)
    ax.set_ylabel("Predicted property loss (USD)", fontsize=10)
    ax.set_title(
        "Predicted vs actual property loss\nGradient Boosting Regressor (capped at $50k)",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k"))
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
 
    plt.tight_layout()
    save_fig("predicted_vs_actual.png")
 
 
# MAIN
 
def main():
    logger.info("model.py started...")
    os.makedirs(FIGURES_DIR, exist_ok=True)
 
    # Load features
    print("\n── Loading feature matrix ──")
    df = load_features()
 
    # Model 1 — classifier
    clf, X_test, y_test, y_pred, feature_names = train_classifier(df)
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importance(clf, feature_names)
 
    # Model 2 — regressor
    train_regressor(df)
 
    logger.info("model.py finished")
    print("\n✓ Models trained successfully")
    print("✓ Figures saved to figures/")
    print("✓ Log updated in pipeline.log")
    print("\nAll Python scripts complete!")
    print("Next step: build pipeline.ipynb")
 
 
if __name__ == "__main__":
    main()
 
