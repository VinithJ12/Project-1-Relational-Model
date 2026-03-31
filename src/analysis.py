import duckdb
import logging
import os
import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend, safe for scripts
import matplotlib.pyplot as plt
import seaborn as sns
 
# CONFIGURATION
DB_PATH     = "fema.db"
LOG_PATH    = "pipeline.log"
FIGURES_DIR = "figures"
DATA_DIR    = "data"
 
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
def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    """Connect to fema.db."""
    try:
        con = duckdb.connect(db_path)
        logger.info(f"Connected to {db_path}")
        return con
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise
 
 
def ensure_dirs() -> None:
    """Create output folders if they don't exist."""
    for d in [FIGURES_DIR, DATA_DIR]:
        os.makedirs(d, exist_ok=True)
    logger.info("Output directories ready")
 
 
def save_fig(filename: str) -> None:
    """Save the current matplotlib figure to figures/ folder."""
    path = os.path.join(FIGURES_DIR, filename)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {path}")
    print(f"  Saved: {path}")
 
 
# QUERY 1 — Habitability repair rate by state
# Research question: which states had the worst
# damage requiring habitability repairs?
 
def query_repair_rate_by_state(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Joins damage_assessment + location to compute the percentage
    of inspected households requiring habitability repairs per state.
    Only includes states with at least 1,000 inspected households
    to avoid noise from tiny samples.
    """
    logger.info("Running query: repair rate by state")
    df = con.execute("""
        SELECT
            l.damagedStateAbbreviation                          AS state,
            COUNT(*)                                            AS total,
            SUM(d.habitabilityRepairsRequired)                  AS repairs_needed,
            ROUND(
                SUM(d.habitabilityRepairsRequired) * 100.0
                / COUNT(*), 1
            )                                                   AS repair_rate_pct
        FROM damage_assessment d
        JOIN location l ON d.id = l.id
        GROUP BY l.damagedStateAbbreviation
        HAVING COUNT(*) >= 1000
        ORDER BY repair_rate_pct DESC
    """).df()
    print("\n── Habitability repair rate by state ──")
    print(df.to_string(index=False))
    return df
 
 
def plot_repair_rate_by_state(df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of repair rates by state.
    Publication quality: clean background, labeled bars, clear title.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#D85A30" if r > 40 else "#378ADD" for r in df["repair_rate_pct"]]
    bars = ax.barh(df["state"], df["repair_rate_pct"], color=colors, edgecolor="none")
 
    # Label each bar with its value
    for bar, val in zip(bars, df["repair_rate_pct"]):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", ha="left", fontsize=9
        )
 
    ax.set_xlabel("% of inspected households requiring habitability repairs", fontsize=10)
    ax.set_title(
        "Habitability repair rates by state\nFEMA Individual Assistance — inspected households only",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.set_xlim(0, df["repair_rate_pct"].max() + 10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
 
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#D85A30", label="> 40% repair rate"),
        Patch(facecolor="#378ADD", label="≤ 40% repair rate"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
 
    plt.tight_layout()
    save_fig("repair_rate_by_state.png")
 
# QUERY 2 — Damage type breakdown
# Research question: which damage types most
# commonly co-occur with habitability repairs?
 
def query_damage_type_breakdown(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    For households that required habitability repairs vs those that
    didn't, compute the rate of each damage type.
    This tells us which damage types are most predictive.
    """
    logger.info("Running query: damage type breakdown")
    df = con.execute("""
        SELECT
            habitabilityRepairsRequired                         AS repairs_required,
            COUNT(*)                                            AS n,
            ROUND(AVG(floodDamage)    * 100, 1)                AS pct_flood,
            ROUND(AVG(foundationDamage) * 100, 1)              AS pct_foundation,
            ROUND(AVG(roofDamage)     * 100, 1)                AS pct_roof,
            ROUND(AVG(destroyed)      * 100, 1)                AS pct_destroyed,
            ROUND(AVG(CAST(waterLevel AS DOUBLE)), 2)          AS avg_water_level,
            ROUND(AVG(rpfvl), 2)                               AS avg_rpfvl
        FROM damage_assessment
        GROUP BY habitabilityRepairsRequired
        ORDER BY habitabilityRepairsRequired
    """).df()
    print("\n── Damage types: repairs required vs not ──")
    print(df.to_string(index=False))
    return df
 
 
def plot_damage_type_breakdown(df: pd.DataFrame) -> None:
    """
    Grouped bar chart comparing damage type rates for
    households with and without habitability repairs.
    """
    damage_types = ["pct_flood", "pct_foundation", "pct_roof", "pct_destroyed"]
    labels       = ["Flood", "Foundation", "Roof", "Destroyed"]
 
    no_repair  = df[df["repairs_required"] == 0][damage_types].values[0]
    yes_repair = df[df["repairs_required"] == 1][damage_types].values[0]
 
    x     = range(len(labels))
    width = 0.35
 
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width/2 for i in x], no_repair,  width, label="No repairs needed", color="#B5D4F4", edgecolor="none")
    ax.bar([i + width/2 for i in x], yes_repair, width, label="Repairs required",  color="#D85A30", edgecolor="none")
 
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("% of households with this damage type", fontsize=10)
    ax.set_title(
        "Damage type rates: repairs required vs not required\nFEMA Individual Assistance dataset",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
 
    # Add value labels on bars
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
 
    plt.tight_layout()
    save_fig("damage_type_breakdown.png")
 
 
# QUERY 3 — Income distribution by repair status
# Research question: are lower-income households
# more likely to need habitability repairs?
 
def query_income_by_repair_status(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Joins registrants + damage_assessment to compare income
    distributions between households that needed repairs vs not.
    Caps income at 200k to remove extreme outliers for visualization.
    """
    logger.info("Running query: income by repair status")
    df = con.execute("""
        SELECT
            r.grossIncome,
            d.habitabilityRepairsRequired
        FROM registrants r
        JOIN damage_assessment d ON r.id = d.id
        WHERE r.grossIncome > 0
          AND r.grossIncome < 200000
          AND r.grossIncome_imputed = 0      -- only observed income, not imputed
    """).df()
    logger.info(f"Income query returned {len(df):,} rows")
    return df
 
 
def plot_income_vs_repair(df: pd.DataFrame) -> None:
    """
    Overlapping histogram showing income distributions for
    households with and without repairs required.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
 
    no_repair  = df[df["habitabilityRepairsRequired"] == 0]["grossIncome"]
    yes_repair = df[df["habitabilityRepairsRequired"] == 1]["grossIncome"]
 
    ax.hist(no_repair,  bins=60, alpha=0.55, color="#378ADD", label="No repairs needed", density=True)
    ax.hist(yes_repair, bins=60, alpha=0.55, color="#D85A30", label="Repairs required",  density=True)
 
    # Median lines
    ax.axvline(no_repair.median(),  color="#185FA5", linestyle="--", linewidth=1.2,
               label=f"Median (no repair): ${no_repair.median():,.0f}")
    ax.axvline(yes_repair.median(), color="#993C1D", linestyle="--", linewidth=1.2,
               label=f"Median (repair req): ${yes_repair.median():,.0f}")
 
    ax.set_xlabel("Gross annual household income (USD)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title(
        "Income distribution by habitability repair status\nObserved income only (imputed values excluded)",
        fontsize=11, fontweight="bold", pad=12
    )
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=9)
 
    # Format x axis as currency
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1000:.0f}k")
    )
 
    plt.tight_layout()
    save_fig("income_vs_repair.png")
 
# QUERY 4 — Rental vs owner repair rates
# Research question: do renters and owners
# experience different repair rates?
 
def query_repair_by_tenure(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """
    Joins registrants + damage_assessment to compare habitability
    repair rates between owners and renters.
    """
    logger.info("Running query: repair rate by tenure")
    df = con.execute("""
        SELECT
            r.ownRent,
            COUNT(*)                                            AS total,
            SUM(d.habitabilityRepairsRequired)                  AS repairs_needed,
            ROUND(
                SUM(d.habitabilityRepairsRequired) * 100.0
                / COUNT(*), 1
            )                                                   AS repair_rate_pct,
            ROUND(AVG(r.grossIncome), 0)                       AS avg_income
        FROM registrants r
        JOIN damage_assessment d ON r.id = d.id
        WHERE r.ownRent IN ('Owner', 'Renter')
        GROUP BY r.ownRent
        ORDER BY repair_rate_pct DESC
    """).df()
    print("\n── Repair rate by tenure (owner vs renter) ──")
    print(df.to_string(index=False))
    return df
 
 
# STEP 5 — BUILD ML FEATURE MATRIX
# This is the cleaned, joined, encoded dataset
# that model.py will load directly.
 
def build_ml_features(con: duckdb.DuckDBPyConnection) -> None:
    """
    Joins all 4 tables and builds the feature matrix for modeling.
 
    Feature engineering decisions:
    - ownRent encoded as binary: 1=Owner, 0=Renter, NULL dropped
    - residenceType encoded as dummy variable (house/duplex as baseline)
    - All binary damage flags kept as-is (already 0/1)
    - Dollar amounts kept as continuous features
    - grossIncome_imputed flag kept so model can learn imputation signal
    - Sample capped at 500,000 rows for training speed
      (still massive, statistically representative)
 
    Target variable: habitabilityRepairsRequired (0 or 1)
    """
    logger.info("Building ML feature matrix...")
 
    con.execute("""
        CREATE OR REPLACE TABLE ml_features AS
        SELECT
            -- Target
            d.habitabilityRepairsRequired                       AS target,
 
            -- Demographic features
            r.householdComposition,
            r.grossIncome,
            r.grossIncome_imputed,
            r.specialNeeds,
            CASE WHEN r.ownRent = 'Owner'  THEN 1
                 WHEN r.ownRent = 'Renter' THEN 0
                 ELSE NULL END                                  AS is_owner,
            CASE WHEN r.residenceType = 'House/Duplex'   THEN 1 ELSE 0 END AS res_house,
            CASE WHEN r.residenceType = 'Apartment'      THEN 1 ELSE 0 END AS res_apartment,
            CASE WHEN r.residenceType = 'Mobile Home'    THEN 1 ELSE 0 END AS res_mobile,
            CASE WHEN r.residenceType = 'Townhouse'      THEN 1 ELSE 0 END AS res_townhouse,
            r.homeOwnersInsurance,
            r.floodInsurance,
            r.primaryResidence,
 
            -- Damage features
            d.waterLevel,
            d.floodDamage,
            d.foundationDamage,
            d.foundationDamageAmount,
            d.roofDamage,
            d.roofDamageAmount,
            d.rpfvl,
            d.ppfvl,
            d.destroyed,
 
            -- Location (state as categorical)
            l.damagedStateAbbreviation                          AS state
 
        FROM damage_assessment d
        JOIN registrants r        ON d.id = r.id
        JOIN location l           ON d.id = l.id
 
        WHERE d.habitabilityRepairsRequired IS NOT NULL
          AND r.ownRent IN ('Owner', 'Renter')     -- drop the tiny 'Unknown' group
          AND r.grossIncome IS NOT NULL
 
        USING SAMPLE 500000                        -- cap for training speed
    """)
 
    n = con.execute("SELECT COUNT(*) FROM ml_features").fetchone()[0]
    target_dist = con.execute("""
        SELECT target, COUNT(*) as n,
               ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as pct
        FROM ml_features GROUP BY target ORDER BY target
    """).df()
 
    print(f"\n── ML feature matrix: {n:,} rows ──")
    print(target_dist.to_string(index=False))
 
    # Export to CSV for model.py
    out_path = os.path.join(DATA_DIR, "ml_features.csv")
    con.execute(f"COPY ml_features TO '{out_path}' (HEADER, DELIMITER ',')")
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    logger.info(f"ml_features exported: {out_path} ({size_mb:.1f} MB)")
    print(f"  Exported ml_features.csv ({size_mb:.1f} MB)")
 
 
# MAIN
 
def main():
    logger.info("analysis.py started...")
    ensure_dirs()
    con = connect(DB_PATH)
 
    # Query 1 — repair rate by state
    print("\n── Query 1: Repair rate by state ──")
    df_state = query_repair_rate_by_state(con)
    plot_repair_rate_by_state(df_state)
 
    # Query 2 — damage type breakdown
    print("\n── Query 2: Damage type breakdown ──")
    df_damage = query_damage_type_breakdown(con)
    plot_damage_type_breakdown(df_damage)
 
    # Query 3 — income vs repair status
    print("\n── Query 3: Income distribution by repair status ──")
    df_income = query_income_by_repair_status(con)
    plot_income_vs_repair(df_income)
 
    # Query 4 — repair rate by tenure
    print("\n── Query 4: Repair rate by tenure ──")
    query_repair_by_tenure(con)
 
    # Build ML feature matrix
    print("\n── Building ML feature matrix ──")
    build_ml_features(con)
 
    con.close()
    logger.info("analysis.py finished")
    print("\n✓ 3 figures saved to figures/")
    print("✓ ml_features.csv saved to data/")
    print("✓ Log updated in pipeline.log")
    print("\nNext step: run model.py")
 
 
if __name__ == "__main__":
    main()
 
