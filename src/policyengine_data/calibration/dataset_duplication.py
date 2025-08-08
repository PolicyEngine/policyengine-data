from typing import Any, Optional

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
    geography_variable: Optional[str] = "ucgid",
    geography_identifier: Optional[Any] = UCGID("0100000US"),
) -> "Microsimulation":
    """
    Load the necessary dataset from the legacy Dataset class, making it specific to a geography area. (e.g., CPS for the state of California).

    Args:
        year (Optional[int]): The year for which to calibrate the dataset.
        dataset (Optional[None]): The dataset to load. If None, defaults to the CPS dataset for the specified year.
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
    hhs = len(sim.calculate("household_id").values)
    geo_values = [geography_identifier] * hhs
    sim.set_input(geography_variable, year, geo_values)

    return sim


def minimize_calibrated_dataset_legacy(
    sim: Microsimulation, year: int, optimized_sparse_weights: pd.Series
) -> "SingleYearDataset":
    """
    Use sparse weights to minimize the calibrated dataset storing in the legacy Dataset class.

    Args:
        sim (Microsimulation): The Microsimulation object with the dataset to minimize.
        year (int): Year the dataset is representing.
        optimized_sparse_weights (pd.Series): The calibrated, regularized weights used to minimize the dataset.

    Returns:
        SingleYearDataset: The regularized dataset
    """
    sim.set_input("household_weight", year, optimized_sparse_weights)

    df = sim.to_input_dataframe()  # Not at household level

    household_weight_column = f"household_weight__{year}"
    df_household_id_column = f"household_id__{year}"

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
