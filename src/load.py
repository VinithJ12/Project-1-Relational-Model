import duckdb
import logging
import os
import sys

CSV_PATH = "IndividualAssistanceHousingRegistrantsLargeDisasters.csv"

DB_PATH = "fema.db"

LOG_PATH = "pipeline.log"

#Logging Setup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),   # writes to pipeline.log
        logging.StreamHandler(sys.stdout) # also prints to terminal
    ]
)
 
logger = logging.getLogger(__name__)

#HELPER FUNCTIONS

def check_csv_exists(path: str) -> None:
    """
    Verifies the CSV file exists before attempting to load.
    Raises FileNotFoundError with a helpful message if not found.
    """
    if not os.path.exists(path):
        logger.error(f"CSV file not found at: {path}")
        raise FileNotFoundError(
            f"\n\nCould not find the CSV at: {path}"
            f"\nPlease update CSV_PATH at the top of load.py to point to your file."
        )
    logger.info(f"CSV found: {path}")
 
 
def connect_to_db(db_path: str) -> duckdb.DuckDBPyConnection:
    """
    Creates or connects to a DuckDB database file.
    Returns the connection object.
    """
    try:
        con = duckdb.connect(db_path)
        logger.info(f"Connected to DuckDB database: {db_path}")
        return con
    except Exception as e:
        logger.error(f"Failed to connect to DuckDB: {e}")
        raise
 
 
def drop_existing_table(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """
    Drops a table if it already exists so we can reload cleanly.
    Useful during development when you want to re-run the script.
    """
    try:
        con.execute(f"DROP TABLE IF EXISTS {table_name}")
        logger.info(f"Dropped existing table '{table_name}' (if it existed)")
    except Exception as e:
        logger.error(f"Failed to drop table '{table_name}': {e}")
        raise
 
 
def load_csv_to_duckdb(con: duckdb.DuckDBPyConnection, csv_path: str, table_name: str) -> None:
    """
    Reads the CSV into DuckDB using read_csv_auto.
    
    read_csv_auto is great for this dataset because:
      - It auto-detects column types
      - It handles the large file size efficiently (streams it, doesn't load all into RAM)
      - ignore_errors=True skips malformed rows without crashing
      - null_padding=True fills short rows with NULLs instead of erroring
    """
    try:
        logger.info(f"Loading CSV into table '{table_name}' — this may take a moment for a 979MB file...")
        con.execute(f"""
            CREATE TABLE {table_name} AS
            SELECT * FROM read_csv_auto(
                '{csv_path}',
                ignore_errors = TRUE,
                null_padding  = TRUE,
                header        = TRUE
            )
        """)
        logger.info(f"Successfully created table '{table_name}'")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        raise
 
 
def print_data_summary(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """
    Prints a quick summary of the loaded data so you can immediately
    sanity-check what you're working with:
      - Row and column counts
      - All column names and their inferred data types
      - Null counts per column (helps spot missing data issues early)
      - A few sample rows
    """
    logger.info("Generating data summary...")
 
    # ── Row and column count ──
    row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    col_info  = con.execute(f"DESCRIBE {table_name}").fetchdf()
    col_count = len(col_info)
 
    print(f"  TABLE: {table_name}")
    print(f"  Rows : {row_count:,}")
    print(f"  Cols : {col_count}")
 
    # ── Column names and types ──
    print("\n── Column names & inferred types ──")
    print(col_info[["column_name", "column_type"]].to_string(index=False))
 
    # ── Null counts per column ──
    print("\n── Null counts per column ──")
    null_checks = " + ".join(
        [f"COUNT(*) FILTER (WHERE \"{col}\" IS NULL)" for col in col_info["column_name"]]
    )
    # Build a query that counts nulls for every column
    null_query_parts = [
        f"SUM(CASE WHEN \"{col}\" IS NULL THEN 1 ELSE 0 END) AS \"{col}\""
        for col in col_info["column_name"]
    ]
    null_query = f"SELECT {', '.join(null_query_parts)} FROM {table_name}"
    null_counts = con.execute(null_query).fetchdf().T  # transpose so cols become rows
    null_counts.columns = ["null_count"]
    null_counts["pct_null"] = (null_counts["null_count"] / row_count * 100).round(1)
    null_counts = null_counts[null_counts["null_count"] > 0].sort_values("pct_null", ascending=False)
 
    if null_counts.empty:
        print("  No nulls found — nice clean data!")
    else:
        print(null_counts.to_string())
 
    # ── Sample rows ──
    print("\n── First 3 rows (sample) ──")
    sample = con.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchdf()
    # Print transposed so we can read all columns easily
    print(sample.T.to_string())
 
    logger.info("Data summary complete")
 
 
def show_key_column_distributions(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    """
    Prints value distributions for a handful of key columns
    that are most important for our project:
      - ownRent        (who is renting vs owning)
      - residenceType  (what kind of housing)
      - damagedState   (which states)
      - rentalAssistanceEligible (our main ML target)
      - habitabilityRepairsRequired (our other ML target)
    """
    print("── Key column distributions ──\n")
 
    key_columns = [
        ("ownRent",                    "Owner vs Renter split"),
        ("residenceType",              "Residence types"),
        ("damagedStateAbbreviation",   "Top 10 states"),
        ("rentalAssistanceEligible",   "Rental assistance eligible (ML target 1)"),
        ("habitabilityRepairsRequired","Habitability repairs required (ML target 2)"),
    ]
 
    for col, label in key_columns:
        try:
            print(f"  {label} [{col}]")
            result = con.execute(f"""
                SELECT 
                    "{col}"          AS value,
                    COUNT(*)         AS count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) AS pct
                FROM {table_name}
                WHERE "{col}" IS NOT NULL
                GROUP BY "{col}"
                ORDER BY count DESC
                LIMIT 10
            """).fetchdf()
            print(result.to_string(index=False))
            print()
        except Exception as e:
            logger.warning(f"Could not compute distribution for '{col}': {e}")
 
 # MAIN
 
def main():
    """
    Main entry point. Runs all steps in order:
      1. Verify CSV exists
      2. Connect to DuckDB
      3. Drop old table if re-running
      4. Load CSV into raw_fema table
      5. Print summary so we can see what we loaded
      6. Show key column distributions
    """
    logger.info("load.py started...")
 
    # Step 1 — check file exists
    check_csv_exists(CSV_PATH)
 
    # Step 2 — connect to DB
    con = connect_to_db(DB_PATH)
 
    # Step 3 — drop if re-running (safe to re-run this script)
    drop_existing_table(con, "raw_fema")
 
    # Step 4 — load the CSV
    load_csv_to_duckdb(con, CSV_PATH, "raw_fema")
 
    # Step 5 — print summary
    print_data_summary(con, "raw_fema")
 
    # Step 6 — key distributions
    show_key_column_distributions(con, "raw_fema")
 
    # Close connection cleanly
    con.close()
    logger.info("load.py finished — database saved to: " + DB_PATH )
    print(f"✓ Database saved to: {DB_PATH}")
    print(f"✓ Log written to:    {LOG_PATH}")
    print(f"\nNext step: run clean.py to normalize into 4 tables")
 
 
if __name__ == "__main__":
    main()
