from typing import Any, Optional

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import (
    UCGID,
)

from ..dataset_legacy import Dataset
from ..single_year_dataset import SingleYearDataset

"""
Functions using the legacy Dataset class to operate datasets given their dependency on Microsimulation objects.
"""


def load_dataset_for_geography_legacy(
    year: Optional[int] = 2023,
    dataset: Optional[str] = None,
    dataset_subsample_size: Optional[int] = None,
    geography_variable: Optional[str] = "ucgid",
    geography_identifier: Optional[Any] = UCGID("0100000US"),
) -> "Microsimulation":
    """
    Load the necessary dataset from the legacy Dataset class, making it specific to a geography area. (e.g., CPS for the state of California).

    Args:
        year (Optional[int]): The year for which to calibrate the dataset.
        dataset (Optional[None]): The dataset to load. If None, defaults to the CPS dataset for the specified year.
        dataset_subsample_size (Optional[int]): The size of the base dataset subsample to use for calibration. If None, the full dataset will be used for stacking.
        geography_variable (Optional[str]): The variable representing the geography in the dataset.
        geography_identifier (Optional[str]): The identifier for the geography to calibrate.

    Returns:
        Microsimulation: The Microsimulation object with the specified geography.
    """
    if dataset is None:
        dataset = f"hf://policyengine/policyengine-us-data/cps_{year}.h5"

    sim = Microsimulation(dataset=dataset)
    sim.default_input_period = year
    sim.build_from_dataset()

    if dataset_subsample_size is not None:
        df = sim.to_input_dataframe()

        # Find the household ID column (it should be named with the year)
        household_id_column = None
        for col in df.columns:
            if col.startswith("household_id__"):
                household_id_column = col
                break

        if household_id_column is None:
            raise KeyError(
                "Could not find household_id column in simulation dataframe"
            )

        # Get unique household IDs
        unique_household_ids = df[household_id_column].unique()

        # Subsample households if we have more than requested
        if len(unique_household_ids) > dataset_subsample_size:
            np.random.seed(42)  # For reproducible results
            subsampled_household_ids = np.random.choice(
                unique_household_ids,
                size=dataset_subsample_size,
                replace=False,
            )

            # Filter dataframe to only include subsampled households
            subset_df = df[
                df[household_id_column].isin(subsampled_household_ids)
            ].copy()

            # Create new simulation from subsampled data
            sim = Microsimulation()
            sim.dataset = Dataset.from_dataframe(subset_df, year)
            sim.default_input_period = year
            sim.build_from_dataset()

    hhs = len(sim.calculate("household_id").values)
    geo_values = [geography_identifier] * hhs
    sim.set_input(geography_variable, year, geo_values)

    ucgid_values = sim.calculate(geography_variable).values
    assert all(val == geography_identifier.name for val in ucgid_values)

    return sim


def minimize_calibrated_dataset_legacy(
    sim: Microsimulation, year: int, optimized_weights: pd.Series
) -> "SingleYearDataset":
    """
    Use sparse weights to minimize the calibrated dataset storing in the legacy Dataset class.

    Args:
        sim (Microsimulation): The Microsimulation object with the dataset to minimize.
        year (int): Year the dataset is representing.
        optimized_weights (pd.Series): The calibrated, regularized weights used to minimize the dataset.

    Returns:
        SingleYearDataset: The regularized dataset
    """
    # Copy all existing variable data to the target year
    for variable_name in sim.tax_benefit_system.variables:
        holder = sim.get_holder(variable_name)
        known_periods = holder.get_known_periods()
        if known_periods and variable_name != "household_weight":
            # Copy from the first available period to target year
            source_period = known_periods[0]
            try:
                values = sim.calculate(variable_name, source_period).values
                sim.set_input(variable_name, year, values)
            except Exception:
                # Skip variables that can't be copied
                continue

    # Set the calibrated household weights for the target year
    sim.set_input("household_weight", year, optimized_weights)

    df = sim.to_input_dataframe()

    # Use the target year for column names
    household_weight_column = f"household_weight__{year}"
    df_household_id_column = f"household_id__{year}"

    # Fallback: if target year columns don't exist, detect the actual year from column names
    if (
        household_weight_column not in df.columns
        or df_household_id_column not in df.columns
    ):
        for col in df.columns:
            if col.startswith("household_weight__"):
                detected_year = col.split("__")[1].split("-")[0]
                household_weight_column = f"household_weight__{detected_year}"
                df_household_id_column = f"household_id__{detected_year}"
                break
        else:
            raise KeyError(
                "Could not find household_weight or household_id columns"
            )

    # Group by household ID and get the first entry for each group
    h_df = df.groupby(df_household_id_column).first()
    h_ids = pd.Series(h_df.index)
    h_weights = pd.Series(h_df[household_weight_column].values)

    # Filter to housholds with non-zero weights
    h_ids = h_ids[h_weights > 0]
    h_weights = h_weights[h_weights > 0]

    subset_df = df[df[df_household_id_column].isin(h_ids)].copy()

    # Update the dataset and rebuild the simulation
    sim = Microsimulation()
    sim.dataset = Dataset.from_dataframe(subset_df, year)
    sim.default_input_period = year
    sim.build_from_dataset()

    single_year_dataset = SingleYearDataset.from_simulation(sim, year)

    return single_year_dataset


"""
Functions using the new SingleYearDataset class once the Microsimulation object is adapted to it.
"""


def load_dataset_for_geography(
    year: Optional[int] = 2023,
    dataset: Optional[str] = None,
    geography_variable: Optional[str] = "ucgid",
    geography_identifier: Optional[Any] = UCGID("0100000US"),
) -> "SingleYearDataset":
    """
    Load the necessary dataset from the legacy Dataset class into the new SingleYearDataset, or directly from it, making it specific to a geography area. (e.g., CPS for the state of California).

    Args:
        year (Optional[int]): The year for which to calibrate the dataset.
        dataset (Optional[None]): The dataset to load. If None, defaults to the CPS dataset for the specified year.
        geography_variable (Optional[str]): The variable representing the geography in the dataset.
        geography_identifier (Optional[str]): The identifier for the geography to calibrate.

    Returns:
        SingleYearDataset: The calibrated dataset after applying regularization.
    """
    from policyengine_us import Microsimulation

    if dataset is None:
        dataset = f"hf://policyengine/policyengine-us-data/cps_{year}.h5"

    sim = Microsimulation(dataset=dataset)

    # To load from the Microsimulation object for compatibility with legacy Dataset class
    single_year_dataset = SingleYearDataset.from_simulation(
        sim, time_period=year
    )
    # To load from the SingleYearDataset class directly
    # single_year_dataset = SingleYearDataset(file_path=dataset)
    single_year_dataset.time_period = year

    household_vars = single_year_dataset.entities["household"]
    household_vars[geography_variable] = geography_identifier
    single_year_dataset.entities["household"] = household_vars

    return single_year_dataset


def minimize_calibrated_dataset(
    dataset: SingleYearDataset,
) -> "SingleYearDataset":
    """
    Use sparse weights to minimize the calibrated dataset.

    To come after policyengine_core adaptation.
    """
    pass
