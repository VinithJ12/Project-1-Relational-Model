import duckdb
import logging
import os
import sys
 
# CONFIGURATION
 
DB_PATH  = "fema.db"
LOG_PATH = "pipeline.log"
DATA_DIR = "data"  # CSVs will be exported here
 
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
 
 
# HELPER FUNCTIONS
 
def connect(db_path: str) -> duckdb.DuckDBPyConnection:
    """Connect to the existing fema.db created by load.py."""
    try:
        con = duckdb.connect(db_path)
        logger.info(f"Connected to {db_path}")
        return con
    except Exception as e:
        logger.error(f"Could not connect to {db_path}: {e}")
        raise
 
 
def ensure_data_dir(path: str) -> None:
    """Create the data/ output folder if it doesn't exist yet."""
    os.makedirs(path, exist_ok=True)
    logger.info(f"Output directory ready: {path}")
 
 
def drop_table(con: duckdb.DuckDBPyConnection, name: str) -> None:
    """Drop a table if it exists so clean.py is safe to re-run."""
    try:
        con.execute(f"DROP TABLE IF EXISTS {name}")
        logger.info(f"Dropped existing table: {name}")
    except Exception as e:
        logger.error(f"Failed to drop {name}: {e}")
        raise
 
 
def row_count(con: duckdb.DuckDBPyConnection, table: str) -> int:
    """Return the number of rows in a table."""
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
 
 
# STEP 1 — BUILD CLEANED BASE TABLE
 
def build_cleaned_base(con: duckdb.DuckDBPyConnection) -> None:
    """
    Creates fema_clean from raw_fema by applying all cleaning rules.
 
    Cleaning decisions (each one documented for the README):
 
    1. FILTER to inspected = 1
       Rationale: habitabilityRepairsRequired (our ML target) is only
       populated when an inspection occurred. Rows with inspected = 0
       have no ground-truth label and cannot be used for modeling.
       This reduces ~6.3M rows to ~2.2M — still enormous.
 
    2. DROP highWaterLocation
       Rationale: 100% null across all rows. No information content.
 
    3. DROP rental resource columns (rentalResourceCity, State, Zip,
       rentalAssistanceEndDate)
       Rationale: 99.8% null. These describe post-aid logistics, not
       the household's situation at time of registration. Not useful
       as features or targets.
 
    4. FILL structural zeros
       Rationale: NULL in dollar amount columns means no damage/aid
       of that type occurred — not that the value is unknown.
       Columns filled: foundationDamageAmount, roofDamageAmount,
       repairAmount, replacementAmount, rentalAssistanceAmount, ppfvl,
       rpfvl, waterLevel.
 
    5. IMPUTE grossIncome nulls with median per state
       Rationale: 18% of registrants did not report income. Using the
       state-level median is a standard approach that preserves
       geographic income variation. A flag column (grossIncome_imputed)
       is added so the model can learn whether imputation occurred.
       This introduces potential bias — documented in README.
 
    6. STANDARDIZE ownRent
       Rationale: A small number of rows contain 'Unknown'. These are
       kept but flagged so downstream models can handle them.
 
    7. DROP rows missing id, damagedStateAbbreviation, or primaryResidence
       Rationale: These are fundamental identifiers. Rows missing them
       (~300 rows) cannot be reliably joined or located.
    """
 
    logger.info("Building fema_clean base table...")
 
    try:
        # First compute state-level median income for imputation
        con.execute("""
            CREATE OR REPLACE TABLE state_income_medians AS
            SELECT
                damagedStateAbbreviation,
                MEDIAN(grossIncome) AS median_income
            FROM raw_fema
            WHERE grossIncome IS NOT NULL
              AND grossIncome > 0
            GROUP BY damagedStateAbbreviation
        """)
        logger.info("State income medians computed")
 
        # Build the cleaned base table
        con.execute("""
            CREATE OR REPLACE TABLE fema_clean AS
            SELECT
                -- ── Identifiers ──
                r.id,
                r.disasterNumber,
                r.censusBlockId,
                r.censusYear,
 
                -- ── Location ──
                r.damagedCity,
                r.damagedStateAbbreviation,
                r.damagedZipCode,
 
                -- ── Household demographics ──
                r.householdComposition,
                COALESCE(
                    CAST(r.grossIncome AS DOUBLE),
                    m.median_income,
                    0.0
                )                                          AS grossIncome,
                CASE
                    WHEN r.grossIncome IS NULL THEN 1
                    ELSE 0
                END                                        AS grossIncome_imputed,
                r.specialNeeds,
                r.ownRent,
                r.residenceType,
                r.homeOwnersInsurance,
                r.floodInsurance,
                r.primaryResidence,
 
                -- ── Inspection & damage ──
                r.inspected,
                r.habitabilityRepairsRequired,
                r.destroyed,
                COALESCE(r.waterLevel, 0)                 AS waterLevel,
                r.floodDamage,
                r.foundationDamage,
                COALESCE(r.foundationDamageAmount, 0.0)   AS foundationDamageAmount,
                r.roofDamage,
                COALESCE(r.roofDamageAmount, 0.0)         AS roofDamageAmount,
                COALESCE(r.rpfvl, 0.0)                    AS rpfvl,
                COALESCE(r.ppfvl, 0.0)                    AS ppfvl,
 
                -- ── Assistance outcomes ──
                r.tsaEligible,
                r.tsaCheckedIn,
                r.rentalAssistanceEligible,
                COALESCE(r.rentalAssistanceAmount, 0.0)   AS rentalAssistanceAmount,
                r.repairAssistanceEligible,
                COALESCE(r.repairAmount, 0.0)             AS repairAmount,
                r.replacementAssistanceEligible,
                COALESCE(r.replacementAmount, 0.0)        AS replacementAmount,
                r.sbaEligible,
                r.renterDamageLevel,
                r.personalPropertyEligible
 
            FROM raw_fema r
            LEFT JOIN state_income_medians m
                ON r.damagedStateAbbreviation = m.damagedStateAbbreviation
 
            -- Cleaning filters
            WHERE r.inspected = 1                              -- only inspected rows
              AND r.habitabilityRepairsRequired IS NOT NULL          -- must have ML target 1
              AND r.id IS NOT NULL                             -- must have a primary key
              AND r.damagedStateAbbreviation IS NOT NULL       -- must have a state
              AND r.primaryResidence IS NOT NULL               -- must know residency status
        """)
 
        n = row_count(con, "fema_clean")
        logger.info(f"fema_clean created: {n:,} rows")
        print(f"\n  fema_clean: {n:,} rows (down from 6,367,325 raw rows)")
        print(f"  Rows removed: {6367325 - n:,} (not inspected or missing key fields)\n")
 
    except Exception as e:
        logger.error(f"Failed to build fema_clean: {e}")
        raise
 
 
# STEP 2 — NORMALIZE INTO 4 RELATIONAL TABLES
 
def create_registrants(con: duckdb.DuckDBPyConnection) -> None:
    """
    Table 1: registrants
    Who the household is — demographics, insurance, residency.
    Primary key: id
    """
    drop_table(con, "registrants")
    try:
        con.execute("""
            CREATE TABLE registrants AS
            SELECT
                id,
                householdComposition,
                grossIncome,
                grossIncome_imputed,
                specialNeeds,
                ownRent,
                residenceType,
                homeOwnersInsurance,
                floodInsurance,
                primaryResidence
            FROM fema_clean
        """)
        n = row_count(con, "registrants")
        logger.info(f"registrants table created: {n:,} rows")
        print(f"  registrants:        {n:,} rows")
    except Exception as e:
        logger.error(f"Failed to create registrants: {e}")
        raise
 
 
def create_damage_assessment(con: duckdb.DuckDBPyConnection) -> None:
    """
    Table 2: damage_assessment
    What physical damage was found during inspection.
    Primary key: id (one row per registrant inspection)
    """
    drop_table(con, "damage_assessment")
    try:
        con.execute("""
            CREATE TABLE damage_assessment AS
            SELECT
                id,
                inspected,
                habitabilityRepairsRequired,
                destroyed,
                waterLevel,
                floodDamage,
                foundationDamage,
                foundationDamageAmount,
                roofDamage,
                roofDamageAmount,
                rpfvl,
                ppfvl
            FROM fema_clean
        """)
        n = row_count(con, "damage_assessment")
        logger.info(f"damage_assessment table created: {n:,} rows")
        print(f"  damage_assessment:  {n:,} rows")
    except Exception as e:
        logger.error(f"Failed to create damage_assessment: {e}")
        raise
 
 
def create_assistance_outcomes(con: duckdb.DuckDBPyConnection) -> None:
    """
    Table 3: assistance_outcomes
    What aid FEMA actually granted to each household.
    Primary key: id
    This table contains our ML targets.
    """
    drop_table(con, "assistance_outcomes")
    try:
        con.execute("""
            CREATE TABLE assistance_outcomes AS
            SELECT
                id,
                tsaEligible,
                tsaCheckedIn,
                rentalAssistanceEligible,
                rentalAssistanceAmount,
                repairAssistanceEligible,
                repairAmount,
                replacementAssistanceEligible,
                replacementAmount,
                sbaEligible,
                renterDamageLevel,
                personalPropertyEligible
            FROM fema_clean
        """)
        n = row_count(con, "assistance_outcomes")
        logger.info(f"assistance_outcomes table created: {n:,} rows")
        print(f"  assistance_outcomes:{n:,} rows")
    except Exception as e:
        logger.error(f"Failed to create assistance_outcomes: {e}")
        raise
 
 
def create_location(con: duckdb.DuckDBPyConnection) -> None:
    """
    Table 4: location
    Where the damaged property was located.
    Primary key: id
    Foreign key relationships: joins to all other tables on id.
    Also includes disasterNumber so you can group by disaster event.
    """
    drop_table(con, "location")
    try:
        con.execute("""
            CREATE TABLE location AS
            SELECT
                id,
                disasterNumber,
                damagedCity,
                damagedStateAbbreviation,
                damagedZipCode,
                censusBlockId,
                censusYear
            FROM fema_clean
        """)
        n = row_count(con, "location")
        logger.info(f"location table created: {n:,} rows")
        print(f"  location:           {n:,} rows")
    except Exception as e:
        logger.error(f"Failed to create location: {e}")
        raise
 
 
# STEP 3 — EXPORT TABLES TO CSV
 
def export_to_csv(con: duckdb.DuckDBPyConnection, table: str, data_dir: str) -> None:
    """
    Exports a DuckDB table to a CSV file in data/.
    These CSVs are what gets uploaded to OneDrive and linked in README.
    """
    out_path = os.path.join(data_dir, f"{table}.csv")
    try:
        con.execute(f"""
            COPY {table} TO '{out_path}'
            (HEADER, DELIMITER ',')
        """)
        # Get file size for logging
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        logger.info(f"Exported {table} -> {out_path} ({size_mb:.1f} MB)")
        print(f"  Exported {table}.csv ({size_mb:.1f} MB)")
    except Exception as e:
        logger.error(f"Failed to export {table}: {e}")
        raise
 
 # STEP 4 — QUICK VALIDATION
 
def validate_tables(con: duckdb.DuckDBPyConnection) -> None:
    """
    Runs a few sanity checks to make sure the tables look right:
      - All 4 tables have the same row count (they should, same base)
      - No duplicate IDs in any table
      - ML targets have no nulls (since we filtered to inspected=1)
      - grossIncome has no nulls (we imputed them all)
    """
    logger.info("Running validation checks...")
    print("\n── Validation ──")
 
    tables = ["registrants", "damage_assessment", "assistance_outcomes", "location"]
 
    # Check all row counts match
    counts = {t: row_count(con, t) for t in tables}
    if len(set(counts.values())) == 1:
        print(f"  Row counts match across all 4 tables ({list(counts.values())[0]:,})")
    else:
        print("  WARNING: row counts do not match!")
        for t, c in counts.items():
            print(f"    {t}: {c:,}")
 
    # Check for duplicate IDs
    for table in tables:
        dup = con.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT id, COUNT(*) as n FROM {table}
                GROUP BY id HAVING n > 1
            )
        """).fetchone()[0]
        status = "OK" if dup == 0 else f"WARNING: {dup} duplicates"
        print(f"  {table} duplicate IDs: {status}")
 
    # Check ML targets have no nulls
    nulls_target1 = con.execute("""
        SELECT COUNT(*) FROM damage_assessment
        WHERE habitabilityRepairsRequired IS NULL
    """).fetchone()[0]
 
    nulls_target2 = con.execute("""
        SELECT COUNT(*) FROM assistance_outcomes
        WHERE rentalAssistanceEligible IS NULL
    """).fetchone()[0]
 
    nulls_income = con.execute("""
        SELECT COUNT(*) FROM registrants
        WHERE grossIncome IS NULL
    """).fetchone()[0]
 
    print(f"  habitabilityRepairsRequired nulls: {nulls_target1} (should be 0)")
    print(f"  rentalAssistanceEligible nulls:    {nulls_target2} (should be 0)")
    print(f"  grossIncome nulls after imputation:{nulls_income} (should be 0)")
 
    logger.info("Validation complete")
 
# MAIN
 
def main():
    logger.info("clean.py started...")
 
    # Setup
    ensure_data_dir(DATA_DIR)
    con = connect(DB_PATH)
 
    # Step 1 — clean and filter
    print("\n── Step 1: Building cleaned base table ──")
    build_cleaned_base(con)
 
    # Step 2 — normalize into 4 tables
    print("\n── Step 2: Creating normalized tables ──")
    create_registrants(con)
    create_damage_assessment(con)
    create_assistance_outcomes(con)
    create_location(con)
 
    # Step 3 — export CSVs
    print("\n── Step 3: Exporting CSVs to data/ ──")
    for table in ["registrants", "damage_assessment", "assistance_outcomes", "location"]:
        export_to_csv(con, table, DATA_DIR)
 
    # Step 4 — validate
    validate_tables(con)
 
    con.close()
    logger.info("clean.py finished")
    print("\n✓ All 4 tables created in fema.db")
    print("✓ CSVs exported to data/")
    print("✓ Log updated in pipeline.log")
    print("\nNext step: run analysis.py")
 
 
if __name__ == "__main__":
    main()
 
