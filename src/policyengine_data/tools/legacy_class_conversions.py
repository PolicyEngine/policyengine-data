"""
Utilities to convert back from SingleYearDataset to the legacy Dataset class.
"""

from pathlib import Path
from typing import Union

import h5py
import numpy as np

from ..single_year_dataset import SingleYearDataset


def SingleYearDataset_to_Dataset(
    dataset: SingleYearDataset,
    output_path: Union[str, Path],
    time_period: int = None,
) -> None:
    """
    Convert a SingleYearDataset to legacy Dataset format and save as h5 file.

    This function loads entity tables from a SingleYearDataset, separates them into
    variable arrays, and saves them in the legacy ARRAYS format used
    by the legacy Dataset class.

    Args:
        dataset: SingleYearDataset instance with entity tables
        output_path: Path where to save the h5 file
        time_period: Time period for the data (defaults to dataset.time_period)
    """
    if time_period is None:
        time_period = dataset.time_period

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert entity tables to variable arrays dictionary with proper type handling
    variable_arrays = {}

    for entity_name, entity_df in dataset.entities.items():
        # Extract each column as a separate variable array
        for column_name in entity_df.columns:
            values = entity_df[column_name].values

            # Handle special data type conversions following CPS pattern
            if values.dtype == object:
                # Try to determine if this should be string or numeric
                try:
                    # Check if it's actually string data that should be encoded
                    if hasattr(values, "decode_to_str"):
                        values = values.decode_to_str().astype("S")
                    elif column_name == "county_fips":
                        values = values.astype("int32")
                    else:
                        # For other object types, try to preserve as string
                        values = np.array(values, dtype="S")
                except:
                    # Fallback: convert to string
                    values = np.array(
                        [str(v).encode() for v in values], dtype="S"
                    )

            variable_arrays[column_name] = values

    # Save in ARRAYS format (direct variable datasets)
    with h5py.File(output_path, "w") as f:
        for variable_name, values in variable_arrays.items():
            try:
                # Store each variable directly as a dataset (no time period grouping)
                f.create_dataset(variable_name, data=values)
            except Exception as e:
                print(f"  Warning: Could not save {variable_name}: {e}")
                continue
