import logging
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


"""
Database connection and structure functions
"""


def fetch_targets(
    engine, period: int, reform_id: Optional[int] = 0
) -> pd.DataFrame:
    """
    Fetch targets for a specific period, and reform scenario.

    Args:
        engine: SQLAlchemy engine
        period: Time period (typically year)
        reform_id: Reform scenario ID (0 for baseline)

    Returns:
        DataFrame with target data joined with stratum information
    """
    query = """
    SELECT 
        t.target_id,
        t.stratum_id,
        t.variable,
        t.period,
        t.reform_id,
        t.value,
        t.active,
        t.tolerance,
        s.stratum_group_id,
        s.parent_stratum_id,
        s.definition_hash
    FROM targets t
    JOIN strata s ON t.stratum_id = s.stratum_id
    WHERE t.period = :period
      AND t.reform_id = :reform_id
      AND t.active = true
    ORDER BY s.parent_stratum_id NULLS FIRST, s.stratum_group_id, s.stratum_id
    """

    return pd.read_sql(
        query,
        engine,
        params={
            "period": period,
            "reform_id": reform_id,
        },
    )


def get_uprating_factors(
    system,
    population_path: str = "calibration.gov.census.populations.total",
    inflation_path: str = "gov.bls.cpi.cpi_u",
    current_year: int = 2023,
    start_year: Optional[int] = 2020,
    end_year: Optional[int] = 2034,
):
    """
    Get population growth factors and inflation factors as a DataFrame indexed to current_year = 1.000.

    Args:
        system: The policy engine country system instance to retrieve uprating factors from.
        population_path (str): The parameter path for population data.
        inflation_path (str): The parameter path for inflation data.
        current_year (int): The current year for which to retrieve factors.
        start_year (Optional[int]): The start year for the range of years to retrieve factors.
        end_year (Optional[int]): The end year for the range of years to retrieve factors.

    Returns:
        pd.DataFrame: A DataFrame containing the population and inflation factors.
    """
    # Get parameters
    population = system.parameters.get_child(population_path)
    cpi_u = system.parameters.get_child(inflation_path)

    # Get base year values
    base_population = population(current_year)
    base_cpi = cpi_u(current_year)

    # Create DataFrame
    years = list(range(start_year, end_year + 1))
    population_factors = [
        round(population(year) / base_population, 3) for year in years
    ]
    inflation_factors = [round(cpi_u(year) / base_cpi, 3) for year in years]

    df = pd.DataFrame(
        {
            "Year": years,
            "Population_factor": population_factors,
            "Inflation_factor": inflation_factors,
        }
    )

    return df


"""
Uprating calculation functions
"""


def calculate_uprating_factor(
    uprating_factors_df: pd.DataFrame,
    from_year: int,
    to_year: int,
    factor_type: str = "inflation",
) -> float:
    """
    Calculate uprating factor from one year to another.

    Args:
        uprating_factors_df: DataFrame with uprating factors
        from_year: Source year
        to_year: Target year
        factor_type: Type of factor to use ('inflation' or 'population')

    Returns:
        Uprating factor to apply
    """
    factor_column = f"{factor_type.title()}_factor"

    from_factor = uprating_factors_df[
        uprating_factors_df["Year"] == from_year
    ][factor_column]
    to_factor = uprating_factors_df[uprating_factors_df["Year"] == to_year][
        factor_column
    ]

    if from_factor.empty or to_factor.empty:
        logger.warning(
            f"Missing {factor_type} factor for year {from_year} or {to_year}"
        )
        return 1.0

    return float(to_factor.iloc[0] / from_factor.iloc[0])


def uprate_targets_for_period(
    targets_df: pd.DataFrame,
    uprating_factors_df: pd.DataFrame,
    from_period: int,
    to_period: int,
    factor_type: str = "inflation",
) -> pd.DataFrame:
    """
    Uprate all targets from one period to another using specified factor type.

    Args:
        targets_df: DataFrame with target data
        uprating_factors_df: DataFrame with uprating factors
        from_period: Source period (year)
        to_period: Target period (year)
        factor_type: Type of uprating factor ('inflation' or 'population')

    Returns:
        DataFrame with uprated targets
    """
    uprated_df = targets_df.copy()

    uprating_factor = calculate_uprating_factor(
        uprating_factors_df, from_period, to_period, factor_type
    )

    uprated_df["uprated_value"] = targets_df["value"] * uprating_factor
    uprated_df["uprating_factor"] = uprating_factor
    uprated_df["original_period"] = from_period
    uprated_df["uprated_period"] = to_period
    uprated_df["factor_type"] = factor_type

    uprated_df["period"] = to_period

    logger.info(
        f"Uprated {len(targets_df)} targets from {from_period} to {to_period} using {factor_type} factor ({uprating_factor:.4f})"
    )

    return uprated_df


"""
Functions for preparing and updating database
"""


def prepare_insert_data(uprated_df: pd.DataFrame) -> List[Dict]:
    """Prepare data for database insertion of uprated targets."""
    inserts = []
    for _, row in uprated_df.iterrows():
        inserts.append(
            {
                "stratum_id": row["stratum_id"],
                "variable": row["variable"],
                "period": row["uprated_period"],
                "reform_id": row["reform_id"],
                "value": row["uprated_value"],
                "active": True,
                "tolerance": row["tolerance"],
            }
        )
    return inserts


def insert_uprated_targets_in_db(engine, inserts: List[Dict]) -> int:
    """
    Insert uprated target values as new records in the database.

    Returns:
        Number of records inserted
    """
    if not inserts:
        return 0

    with engine.begin() as conn:
        for insert in inserts:
            # Check if target already exists for this combination
            check_query = text(
                """
                SELECT target_id FROM targets 
                WHERE stratum_id = :stratum_id 
                AND variable = :variable 
                AND period = :period 
                AND reform_id = :reform_id
            """
            )

            result = conn.execute(check_query, insert)
            existing = result.fetchone()

            if existing:
                # Update existing target
                update_query = text(
                    """
                    UPDATE targets
                    SET value = :value, tolerance = :tolerance, active = :active
                    WHERE target_id = :target_id
                """
                )
                conn.execute(
                    update_query, {**insert, "target_id": existing[0]}
                )
            else:
                # Insert new target
                insert_query = text(
                    """
                    INSERT INTO targets (stratum_id, variable, period, reform_id, value, active, tolerance)
                    VALUES (:stratum_id, :variable, :period, :reform_id, :value, :active, :tolerance)
                """
                )
                conn.execute(insert_query, insert)

    return len(inserts)


"""
Main uprating function to be called externally
"""


def uprate_calibration_targets(
    system,
    db_uri: str,
    from_period: int,
    to_period: int,
    variable: Optional[str] = None,
    reform_id: Optional[int] = 0,
    population_path: str = "calibration.gov.census.populations.total",
    inflation_path: str = "gov.bls.cpi.cpi_u",
    update_database: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Main function to uprate calibration targets from one period to another.

    Automatically selects uprating factor based on variable name:
    - Variables containing "_count" use population uprating factor
    - All other variables use inflation uprating factor

    Args:
        system: Tax benefit system object from which to retrieve uprating parameters
        db_uri: Database connection string
        from_period: Source period (year) to uprate from
        to_period: Target period (year) to uprate to
        variable: Target variable to uprate (None = all variables)
        reform_id: Reform scenario ID (0 for baseline)
        population_path: Parameter path for population data
        inflation_path: Parameter path for inflation data
        update_database: If True, insert uprated targets into database

    Returns:
        DataFrame with original and uprated values
    """
    # Connect to database
    engine = create_engine(db_uri)

    # Get uprating factors
    uprating_factors_df = get_uprating_factors(
        system=system,
        population_path=population_path,
        inflation_path=inflation_path,
        current_year=from_period,
        start_year=min(from_period, to_period),
        end_year=max(from_period, to_period),
    )

    # Fetch targets for the source period
    targets_df = fetch_targets(engine, from_period, reform_id)

    if targets_df.empty:
        logger.warning(f"No targets found for period {from_period}")
        return pd.DataFrame()

    # Filter by variable if specified
    if variable is not None:
        targets_df = targets_df[targets_df["variable"] == variable]
        if targets_df.empty:
            logger.warning(
                f"No targets found for variable '{variable}' in period {from_period}"
            )
            return pd.DataFrame()

    logger.info(
        f"Found {len(targets_df)} targets to uprate from {from_period} to {to_period}"
    )

    # Group targets by variable and apply appropriate uprating factor
    all_uprated_dfs = []

    for var_name in targets_df["variable"].unique():
        var_targets = targets_df[targets_df["variable"] == var_name]

        if "_count" in var_name:
            factor_type = "population"
            logger.info(
                f"Using population factor for variable '{var_name}' (contains '_count')"
            )
        else:
            factor_type = "inflation"
            logger.info(f"Using inflation factor for variable '{var_name}'")

        uprated_var_df = uprate_targets_for_period(
            var_targets,
            uprating_factors_df,
            from_period,
            to_period,
            factor_type,
        )

        all_uprated_dfs.append(uprated_var_df)

    # Combine all uprated results
    if all_uprated_dfs:
        uprated_df = pd.concat(all_uprated_dfs, ignore_index=True)
    else:
        logger.warning("No targets to uprate")
        return pd.DataFrame()

    results_df = uprated_df[
        [
            "target_id",
            "stratum_id",
            "stratum_group_id",
            "parent_stratum_id",
            "variable",
            "original_period",
            "uprated_period",
            "reform_id",
            "value",
            "uprated_value",
            "uprating_factor",
            "factor_type",
            "tolerance",
        ]
    ].copy()

    # Update database if requested
    if update_database:
        inserts = prepare_insert_data(uprated_df)
        inserted_count = insert_uprated_targets_in_db(engine, inserts)
        logger.info(
            f"Inserted/updated {inserted_count} uprated targets in database"
        )
    else:
        logger.info(
            "Update database was set to False - no database updates performed"
        )

    logger.info(f"Total targets uprated: {len(results_df)}")

    return results_df


if __name__ == "__main__":
    from policyengine_us.system import system

    from policyengine_data.calibration.target_rescaling import (
        download_database,
    )

    # Connection to database in huggingface hub
    db_uri = download_database()

    # Example: uprate 2022 targets to 2023
    results = uprate_calibration_targets(
        system=system, db_uri=db_uri, from_period=2022, to_period=2023
    )

    print("\nUprating Results:")
    print(results)

    # Show factor type breakdown
    if not results.empty:
        print(f"\nFactor Type Summary:")
        factor_summary = results.groupby("factor_type")["variable"].unique()
        for factor_type, variables in factor_summary.items():
            print(f"{factor_type.title()} factor used for: {list(variables)}")
