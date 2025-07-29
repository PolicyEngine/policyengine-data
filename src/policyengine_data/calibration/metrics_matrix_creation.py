import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


def fetch_targets_from_database(
    engine, time_period: int, reform_id: Optional[int] = 0
) -> pd.DataFrame:
    """
    Fetch all targets for a specific time period and reform from the database.

    Args:
        engine: SQLAlchemy engine
        time_period: The year to fetch targets for
        reform_id: The reform scenario ID (0 for baseline)

    Returns:
        DataFrame with target data including target_id, variable, value, etc.
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
        t.notes,
        s.stratum_group_id,
        s.parent_stratum_id
    FROM targets t
    JOIN strata s ON t.stratum_id = s.stratum_id
    WHERE t.period = :period
      AND t.reform_id = :reform_id
    ORDER BY t.target_id
    """

    return pd.read_sql(
        query, engine, params={"period": time_period, "reform_id": reform_id}
    )


def fetch_stratum_constraints(engine, stratum_id: int) -> pd.DataFrame:
    """
    Fetch all constraints for a specific stratum from the database.

    Args:
        engine: SQLAlchemy engine
        stratum_id: The stratum ID

    Returns:
        DataFrame with constraint data
    """
    query = """
    SELECT 
        stratum_id,
        constraint_variable,
        value,
        operation,
        notes
    FROM stratum_constraints
    WHERE stratum_id = :stratum_id
    ORDER BY constraint_variable
    """

    return pd.read_sql(query, engine, params={"stratum_id": stratum_id})


def calculate_variable_at_household_level(
    sim: Microsimulation, variable: str, system
) -> np.ndarray:
    """
    Calculate a variable and ensure it's at the household level.

    Args:
        sim: Microsimulation instance
        variable: Variable name to calculate
        system: System object containing variable definitions

    Returns:
        Array of values at household level
    """
    # Calculate the variable
    values = sim.calculate(variable).values

    # Get the entity level of the variable
    values_entity = system.variables[variable].entity.key

    # Map to household level if needed
    if values_entity != "household":
        values = sim.map_result(values, values_entity, "household")

    return values


def parse_constraint_value(value: str, operation: str):
    """
    Parse constraint value based on its type and operation.

    Args:
        value: String value from constraint
        operation: Operation type

    Returns:
        Parsed value (could be list, float, int, or string)
    """
    # Handle special operations that might use lists
    if operation == "in" and "," in value:
        # Parse as list
        return [v.strip() for v in value.split(",")]

    # Try to convert to numeric
    try:
        num_value = float(value)
        if num_value.is_integer():
            return int(num_value)
        return num_value
    except ValueError:
        return value


def apply_single_constraint(
    values: np.ndarray, operation: str, constraint_value
) -> np.ndarray:
    """
    Apply a single constraint operation to create a boolean mask.

    Args:
        values: Array of values to apply constraint to
        operation: Operation type
        constraint_value: Parsed constraint value

    Returns:
        Boolean array indicating which values meet the constraint
    """
    operations = {
        "equals": lambda v, cv: v == cv,
        "greater_than": lambda v, cv: v > cv,
        "greater_than_or_equal": lambda v, cv: v >= cv,
        "less_than": lambda v, cv: v < cv,
        "less_than_or_equal": lambda v, cv: v <= cv,
        "not_equals": lambda v, cv: v != cv,
    }

    # "in" operation - check if constraint value is contained in string values
    if operation == "in":
        if isinstance(constraint_value, list):
            # Check if any of the constraint values are contained in the string representation
            mask = np.zeros(len(values), dtype=bool)
            for cv in constraint_value:
                mask |= np.array([str(cv) in str(v) for v in values])
            return mask
        else:
            # Single value - check if it's contained in each value's string representation
            return np.array([str(constraint_value) in str(v) for v in values])

    if operation not in operations:
        raise ValueError(f"Unknown operation: {operation}")

    return operations[operation](values, constraint_value)


def create_constraint_mask(
    sim: Microsimulation, constraints_df: pd.DataFrame, system
) -> np.ndarray:
    """
    Create a boolean mask for households that meet all stratum constraints.

    Args:
        sim: Microsimulation instance
        constraints_df: DataFrame with constraint data
        system: System object containing variable definitions

    Returns:
        Boolean array at household level
    """
    # Get number of households
    n_households = len(sim.calculate("household_id").values)

    # If no constraints, all households are included
    if constraints_df.empty:
        return np.ones(n_households, dtype=bool)

    # Start with all True
    combined_mask = np.ones(n_households, dtype=bool)

    # Apply each constraint
    for _, constraint in constraints_df.iterrows():
        # Calculate the constraint variable at household level
        variable_values = calculate_variable_at_household_level(
            sim, constraint["constraint_variable"], system
        )

        # Parse the constraint value
        parsed_value = parse_constraint_value(
            constraint["value"], constraint["operation"]
        )

        # Apply the constraint
        constraint_mask = apply_single_constraint(
            variable_values, constraint["operation"], parsed_value
        )

        # Combine with AND logic
        combined_mask &= constraint_mask

    return combined_mask


def calculate_target_variable(
    sim: Microsimulation, variable: str, stratum_mask: np.ndarray, system
) -> np.ndarray:
    """
    Calculate target variable values and apply stratum mask.

    Args:
        sim: Microsimulation instance
        variable: Target variable name
        stratum_mask: Boolean mask for the stratum
        system: System object containing variable definitions

    Returns:
        Array of masked values at household level
    """
    # Calculate the variable at household level
    values = calculate_variable_at_household_level(sim, variable, system)

    # Apply stratum mask (zero out values outside the stratum)
    return values * stratum_mask


def parse_constraint_for_name(constraint: pd.Series) -> str:
    """
    Parse a single constraint into a human-readable format for naming.

    Args:
        constraint: pandas Series with constraint data

    Returns:
        Human-readable constraint description
    """
    var = constraint["constraint_variable"]
    op = constraint["operation"]
    val = constraint["value"]

    # Map operations to symbols for readability
    op_symbols = {
        "equals": "=",
        "greater_than": ">",
        "greater_than_or_equal": ">=",
        "less_than": "<",
        "less_than_or_equal": "<=",
        "not_equals": "!=",
        "in": "in",
    }

    # Get the symbol or use the operation name if not found
    symbol = op_symbols.get(op, op)

    # Format the constraint
    if op == "in":
        # Replace commas with underscores for "in" operations
        return f"{var}_in_{val.replace(',', '_')}"
    else:
        # Use the symbol format for all other operations
        return f"{var}{symbol}{val}"


def build_target_name(variable: str, constraints_df: pd.DataFrame) -> str:
    """
    Build a descriptive name for a target with variable and constraints.

    Args:
        variable: Target variable name
        constraints_df: DataFrame with constraint data

    Returns:
        Descriptive string name
    """
    parts = [variable]

    if not constraints_df.empty:
        # Sort constraints to ensure consistent naming
        # First by whether it's ucgid, then alphabetically
        constraints_sorted = constraints_df.copy()
        constraints_sorted["is_ucgid"] = constraints_sorted[
            "constraint_variable"
        ].str.contains("ucgid")
        constraints_sorted = constraints_sorted.sort_values(
            ["is_ucgid", "constraint_variable"], ascending=[False, True]
        )

        # Add each constraint
        for _, constraint in constraints_sorted.iterrows():
            parts.append(parse_constraint_for_name(constraint))

    return "_".join(parts)


def process_single_target(
    sim: Microsimulation,
    target: pd.Series,
    constraints_df: pd.DataFrame,
    system,
) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Process a single target to calculate its metric values and info.

    Args:
        sim: Microsimulation instance
        target: pandas Series with target data
        constraints_df: DataFrame with constraint data
        system: System object containing variable definitions

    Returns:
        Tuple of (metric_values, target_info_dict)
    """
    # Create stratum mask
    stratum_mask = create_constraint_mask(sim, constraints_df, system)

    # Calculate target variable with mask applied
    metric_values = calculate_target_variable(
        sim, target["variable"], stratum_mask, system
    )

    # Build target info dictionary
    target_info = {
        "name": build_target_name(target["variable"], constraints_df),
        "active": bool(target["active"]),
        "tolerance": (
            target["tolerance"] if pd.notna(target["tolerance"]) else None
        ),
    }

    return metric_values, target_info


def initialize_simulation(
    time_period: int,
    sim: Optional[Microsimulation] = None,
    dataset: Optional[type] = None,
) -> Microsimulation:
    """
    Initialize or validate the microsimulation instance.

    Args:
        time_period: Time period for the simulation
        sim: Optional existing Microsimulation instance
        dataset: Optional dataset type for creating new simulation

    Returns:
        Microsimulation instance
    """
    if sim is None:
        if dataset is None:
            raise ValueError("Either 'sim' or 'dataset' must be provided")
        sim = Microsimulation(dataset=dataset)

    sim.default_calculation_period = time_period
    return sim


def create_metrics_matrix(
    db_uri: str,
    time_period: int,
    sim: Optional[Microsimulation] = None,
    dataset: Optional[type] = None,
    reform_id: Optional[int] = 0,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, Dict[str, any]]]:
    """
    Create the metrics matrix from the targets database.

    This function processes all targets in the database to create a matrix where:
    - Rows represent households
    - Columns represent targets
    - Values represent the metric calculation for each household-target combination

    Args:
        db_uri: Database connection string
        time_period: Time period for the simulation
        sim: Optional existing Microsimulation instance
        dataset: Optional dataset type for creating new simulation
        reform_id: Reform scenario ID (0 for baseline)

    Returns:
        Tuple of:
        - metrics_matrix: DataFrame with target_id as columns, households as rows
        - target_values: Array of target values in same order as columns
        - target_info: Dictionary mapping target_id to info dict with keys:
            - name: Descriptive name
            - active: Boolean active status
            - tolerance: Tolerance percentage (or None)
    """
    # Setup database connection
    engine = create_engine(db_uri)

    # Initialize simulation
    sim = initialize_simulation(time_period, sim, dataset)

    # Get the system object for variable entity mapping
    system = sim.tax_benefit_system

    # Get household IDs for matrix index
    household_ids = sim.calculate("household_id").values
    n_households = len(household_ids)

    # Fetch all targets from database
    targets_df = fetch_targets_from_database(engine, time_period, reform_id)
    logger.info(
        f"Processing {len(targets_df)} targets for period {time_period}"
    )

    # Initialize outputs
    target_values = []
    target_info = {}
    metrics_list = []
    target_ids = []

    # Process each target
    for idx, target in targets_df.iterrows():
        target_id = target["target_id"]

        try:
            # Fetch constraints for this target's stratum
            constraints_df = fetch_stratum_constraints(
                engine, target["stratum_id"]
            )

            # Process the target
            metric_values, info_dict = process_single_target(
                sim, target, constraints_df, system
            )

            # Store results
            metrics_list.append(metric_values)
            target_ids.append(target_id)
            target_values.append(target["value"])
            target_info[target_id] = info_dict

            logger.debug(
                f"Processed target {target_id}: {info_dict['name']} "
                f"(active={info_dict['active']}, tolerance={info_dict['tolerance']})"
            )

        except Exception as e:
            logger.error(f"Error processing target {target_id}: {str(e)}")
            # Add zero column for failed targets
            metrics_list.append(np.zeros(n_households))
            target_ids.append(target_id)
            target_values.append(target["value"])
            target_info[target_id] = {
                "name": f"ERROR_{target['variable']}",
                "active": False,
                "tolerance": None,
            }

    # Create the metrics matrix DataFrame
    metrics_matrix = pd.DataFrame(
        data=np.column_stack(metrics_list),
        index=household_ids,
        columns=target_ids,
    )

    # Convert target values to numpy array
    target_values = np.array(target_values)

    logger.info(f"Created metrics matrix with shape {metrics_matrix.shape}")
    logger.info(
        f"Active targets: {sum(info['active'] for info in target_info.values())}"
    )

    return metrics_matrix, target_values, target_info


def validate_metrics_matrix(
    metrics_matrix: pd.DataFrame,
    target_values: np.ndarray,
    weights: Optional[np.ndarray] = None,
    target_info: Optional[Dict[int, Dict[str, any]]] = None,
) -> pd.DataFrame:
    """
    Validate the metrics matrix by checking estimates vs targets.

    Args:
        metrics_matrix: The metrics matrix
        target_values: Array of target values
        weights: Optional weights array (defaults to uniform weights)
        target_info: Optional target info dictionary

    Returns:
        DataFrame with validation results
    """
    if weights is None:
        weights = np.ones(len(metrics_matrix)) / len(metrics_matrix)

    estimates = weights @ metrics_matrix.values

    validation_data = {
        "target_id": metrics_matrix.columns,
        "target_value": target_values,
        "estimate": estimates,
        "absolute_error": np.abs(estimates - target_values),
        "relative_error": np.abs(
            (estimates - target_values) / (target_values + 1e-10)
        ),
    }

    # Add target info if provided
    if target_info is not None:
        validation_data["name"] = [
            target_info.get(tid, {}).get("name", "Unknown")
            for tid in metrics_matrix.columns
        ]
        validation_data["active"] = [
            target_info.get(tid, {}).get("active", False)
            for tid in metrics_matrix.columns
        ]
        validation_data["tolerance"] = [
            target_info.get(tid, {}).get("tolerance", None)
            for tid in metrics_matrix.columns
        ]

    validation_df = pd.DataFrame(validation_data)

    return validation_df


if __name__ == "__main__":
    # Loading local database
    db_uri = "sqlite:///src/policyengine_data/calibration/policy_data.db"

    # Create metrics matrix
    metrics_matrix, target_values, target_info = create_metrics_matrix(
        db_uri=db_uri,
        time_period=2023,
        dataset="hf://policyengine/policyengine-us-data/cps_2023.h5",
        reform_id=0,
    )

    # Validate the matrix
    validation_results = validate_metrics_matrix(
        metrics_matrix, target_values, target_info=target_info
    )

    print("\nValidation Results Summary:")
    print(f"Total targets: {len(validation_results)}")
    print(f"Active targets: {validation_results['active'].sum()}")
