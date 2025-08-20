import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


"""
Database connection and structure functions
"""


def download_database(
    filename: Optional[str] = "policy_data.db",
    repo_id: Optional[str] = "policyengine/policyengine-us-data",
) -> create_engine:
    """
    Download the SQLite database from Hugging Face Hub and return the connection string.

    Args:
        filename: optional name of the database file to download
        repo_id: optional Hugging Face repository ID where the database is stored

    Returns:
        Connection string for the SQLite database
    """
    import os

    from huggingface_hub import hf_hub_download

    # Download the file to the current working directory
    try:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir="download/",
            local_dir_use_symlinks=False,  # Recommended to avoid symlinks
            force_download=True,  # Always download, ignore cache
        )
        path = os.path.abspath(downloaded_path)
        logger.info(f"File downloaded successfully to: {path}")
        return f"sqlite:///{path}"

    except Exception as e:
        raise ValueError(f"An error occurred: {e}")


def fetch_targets(
    engine, variable: str, period: int, reform_id: Optional[int] = 0
) -> pd.DataFrame:
    """
    Fetch targets for a specific variable, period, and reform scenario.

    Args:
        engine: SQLAlchemy engine
        variable: Target variable name (e.g., 'income_tax')
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
    WHERE t.variable = :variable
      AND t.period = :period
      AND t.reform_id = :reform_id
      AND t.active = true
    ORDER BY s.parent_stratum_id NULLS FIRST, s.stratum_group_id, s.stratum_id
    """

    return pd.read_sql(
        query,
        engine,
        params={
            "variable": variable,
            "period": period,
            "reform_id": reform_id,
        },
    )


def fetch_all_targets(engine: create_engine) -> pd.DataFrame:
    """
    Fetch all active targets from the database.

    Returns:
        DataFrame with all target data joined with stratum information
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
    WHERE t.active = true
    ORDER BY t.variable, t.period, t.reform_id, 
             s.parent_stratum_id NULLS FIRST, s.stratum_group_id, s.stratum_id
    """

    return pd.read_sql(query, engine)


def get_unique_combinations(
    targets_df: pd.DataFrame,
) -> List[Tuple[str, int, int]]:
    """
    Get all unique combinations of (variable, period, reform_id) from targets.

    Returns:
        List of tuples containing unique combinations
    """
    combinations = targets_df[
        ["variable", "period", "reform_id"]
    ].drop_duplicates()
    return [tuple(row) for row in combinations.values]


"""
Hierarchy analysis functions
"""


def get_root_strata(targets_df: pd.DataFrame) -> pd.DataFrame:
    """Get all root strata (those without parents)."""
    return targets_df[
        (targets_df["parent_stratum_id"].isna())
        | (targets_df["parent_stratum_id"] == 0)
    ]


def get_children_strata(
    targets_df: pd.DataFrame, parent_id: int
) -> pd.DataFrame:
    """Get all children of a specific parent stratum."""
    return targets_df[targets_df["parent_stratum_id"] == parent_id]


"""
Calculation functions
"""


def calculate_group_total(
    group_df: pd.DataFrame, value_column: Optional[str] = "value"
) -> float:
    """Calculate the sum of values for a group of strata."""
    return group_df[value_column].sum()


def calculate_scaling_factor(
    parent_total: float, children_total: float
) -> float:
    """
    Calculate the scaling factor needed to match parent total.

    Returns 1.0 if children_total is 0 to avoid division by zero.
    """
    if children_total == 0:
        logger.warning("Children total is 0, returning scaling factor of 1.0")
        return 1.0
    return parent_total / children_total


"""
Rescaling functions
"""


def rescale_children_to_parent(
    targets_df: pd.DataFrame,
    parent_id: int,
) -> pd.DataFrame:
    """
    Rescale all children of a parent to match the parent's total.

    Returns updated dataframe with scaled values.
    """
    # Get parent scaled value
    parent_rows = targets_df[targets_df["stratum_id"] == parent_id]
    if parent_rows.empty:
        logger.warning(f"Parent stratum {parent_id} not found")
        return targets_df

    parent_value = parent_rows.iloc[0]["scaled_value"]

    # Get all children
    children = get_children_strata(targets_df, parent_id)
    if children.empty:
        return targets_df

    # Process each stratum group separately
    for group_id in children["stratum_group_id"].unique():
        group_children = children[children["stratum_group_id"] == group_id]

        # Calculate scaling factor based on current scaled values
        children_total = calculate_group_total(group_children, "scaled_value")
        scaling_factor = calculate_scaling_factor(parent_value, children_total)

        # Update scaled values and scaling factors directly in the main dataframe
        for _, child in group_children.iterrows():
            mask = targets_df["target_id"] == child["target_id"]
            targets_df.loc[mask, "scaled_value"] = (
                child["scaled_value"] * scaling_factor
            )
            targets_df.loc[mask, "scaling_factor"] = scaling_factor

    return targets_df


def get_hierarchy_levels(targets_df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Organize strata by hierarchy level.

    Returns:
        Dict mapping level number to list of stratum_ids at that level
    """
    levels = {}
    processed = set()

    # Level 0: root strata
    root_strata = get_root_strata(targets_df)
    levels[0] = root_strata["stratum_id"].tolist()
    processed.update(levels[0])

    # Build subsequent levels
    level = 1
    while len(processed) < len(targets_df):
        current_level = []

        # Find all strata whose parents are in the previous level
        for parent_id in levels[level - 1]:
            children = get_children_strata(targets_df, parent_id)
            current_level.extend(children["stratum_id"].tolist())

        if not current_level:
            break

        levels[level] = current_level
        processed.update(current_level)
        level += 1

    return levels


def rescale_targets_hierarchically(targets_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform hierarchical rescaling of targets.

    Processes the hierarchy top-down, ensuring each level's children
    sum to their parent's total.
    """
    # Initialize scaled values
    targets_df["scaled_value"] = targets_df["value"].copy()
    targets_df["scaling_factor"] = 1.0

    # Get hierarchy levels
    levels = get_hierarchy_levels(targets_df)

    # Process each level (starting from level 1, as level 0 has no parents)
    for level in sorted(levels.keys())[1:]:
        for stratum_id in levels[level - 1]:  # Parents are in previous level
            targets_df = rescale_children_to_parent(targets_df, stratum_id)

    return targets_df


"""
Functions for preparing and updating database
"""


def prepare_update_data(targets_df: pd.DataFrame) -> List[Dict]:
    """Prepare data for database update."""
    updates = []
    for _, row in targets_df.iterrows():
        updates.append(
            {
                "target_id": row["target_id"],
                "reescaled_value": row["scaled_value"],
                "original_value": row["value"],
                "scaling_factor": row["scaling_factor"],
            }
        )
    return updates


def update_targets_in_db(engine, updates: List[Dict]) -> int:
    """
    Update target values in the database.

    Returns:
        Number of records updated
    """
    if not updates:
        return 0

    with engine.begin() as conn:
        for update in updates:
            query = text(
                """
                UPDATE targets
                SET value = :reescaled_value
                WHERE target_id = :target_id
            """
            )
            conn.execute(query, update)

    return len(updates)


"""
Main rescaling function to be called externally
"""


def rescale_calibration_targets(
    db_uri: str,
    variable: Optional[str] = None,
    period: Optional[int] = None,
    reform_id: Optional[int] = None,
    update_database: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Main function to rescale calibration targets hierarchically.

    Args:
        db_uri: Database connection string
        variable: Target variable to rescale (None = all variables)
        period: Time period (None = all periods)
        reform_id: Reform scenario ID (None = all reforms, default 0 for baseline)
        update_database: If True, update the database with rescaled values

    Returns:
        DataFrame with original and rescaled values
    """
    # Connect to database
    engine = create_engine(db_uri)

    # Determine what to rescale
    if variable is None or period is None or reform_id is None:
        # Need to fetch all targets to determine combinations
        all_targets = fetch_all_targets(engine)

        if all_targets.empty:
            logger.warning("No active targets found in database")
            return pd.DataFrame()

        # Get unique combinations to process
        combinations = get_unique_combinations(all_targets)

        # Filter combinations based on provided parameters
        filtered_combinations = []
        for var, per, ref in combinations:
            if (
                (variable is None or var == variable)
                and (period is None or per == period)
                and (reform_id is None or ref == reform_id)
            ):
                filtered_combinations.append((var, per, ref))

        logger.info(
            f"Found {len(filtered_combinations)} unique combinations to rescale"
        )

        # Process each combination
        all_results = []
        for var, per, ref in filtered_combinations:
            logger.info(
                f"\nProcessing: variable='{var}', period={per}, reform_id={ref}"
            )

            # Get targets for this combination
            targets_df = all_targets[
                (all_targets["variable"] == var)
                & (all_targets["period"] == per)
                & (all_targets["reform_id"] == ref)
            ].copy()

            # Perform rescaling
            rescaled_df = rescale_targets_hierarchically(targets_df)

            # Add to results
            all_results.append(rescaled_df)

        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
        else:
            logger.warning("No targets to rescale")
            return pd.DataFrame()

    else:
        # Single combination specified
        targets_df = fetch_targets(engine, variable, period, reform_id)

        if targets_df.empty:
            logger.warning(
                f"No targets found for {variable} in period {period}"
            )
            return pd.DataFrame()

        # Perform rescaling
        logger.info(f"Rescaling {len(targets_df)} targets for {variable}")
        combined_results = rescale_targets_hierarchically(targets_df)

    # Prepare results
    results_df = combined_results[
        [
            "target_id",
            "stratum_id",
            "stratum_group_id",
            "parent_stratum_id",
            "variable",
            "period",
            "reform_id",
            "value",
            "scaled_value",
            "scaling_factor",
            "tolerance",
        ]
    ].copy()

    # Update database if requested
    if update_database:
        updates = prepare_update_data(combined_results)
        updated_count = update_targets_in_db(engine, updates)
        logger.info(f"\nTotal: Updated {updated_count} records in database")
    else:
        logger.info(
            "\nUpdate database was set to False - no database updates performed"
        )

    logger.info(f"Total targets processed: {len(results_df)}")
    changed = results_df[results_df["scaling_factor"] != 1.0]
    logger.info(f"Targets that were rescaled: {len(changed)}")

    return results_df


if __name__ == "__main__":
    # Connection to database in huggingface hub
    db_uri = download_database()

    results = rescale_calibration_targets(db_uri=db_uri)

    print("\nRescaling Results:")
    print(results)
