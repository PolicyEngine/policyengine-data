"""
Utilities to convert back from SingleYearDataset to the legacy Dataset class.
"""

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import pandas as pd

from ..single_year_dataset import SingleYearDataset


def SingleYearDataset_to_Dataset(
    dataset: SingleYearDataset,
    output_path: Union[str, Path],
    time_period: int = None,
) -> None:
    """
    Convert a SingleYearDataset to legacy Dataset format and save as h5 file.

    This function loads entity tables from a SingleYearDataset, separates them into
    variable arrays, and saves them in the flat ARRAYS format expected by PolicyEngine.

    Args:
        dataset: SingleYearDataset instance with entity tables
        output_path: Path where to save the h5 file
        time_period: Time period for the data (defaults to dataset.time_period)
    """
    if time_period is None:
        time_period = dataset.time_period

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save in flat ARRAYS format (all variables as datasets at root level)
    with h5py.File(output_path, "w") as f:
        for entity_name, entity_df in dataset.entities.items():
            # Process each column as a variable
            for column_name in entity_df.columns:
                column = entity_df[column_name]
                values = column.to_numpy()

                # Handle special data type conversions
                if (
                    pd.api.types.is_string_dtype(column.dtype)
                    or values.dtype == object
                ):
                    try:
                        # Try to convert to appropriate type
                        if column_name in [
                            "state_name",
                            "state_code",
                            "state_code_str",
                        ]:
                            # String columns - encode as fixed-length strings
                            strings = [
                                "" if pd.isna(v) else str(v) for v in values
                            ]
                            max_len = max(
                                (len(value) for value in strings), default=1
                            )
                            values = np.array(
                                strings,
                                dtype=f"S{max_len}",
                            )
                        elif column_name == "county_fips":
                            values = values.astype("int32")
                        else:
                            # Try numeric conversion first
                            try:
                                values = pd.to_numeric(values, errors="raise")
                                # Keep integers as integers for certain variables
                                if column_name.endswith(
                                    "_id"
                                ) or column_name in ["age", "count", "year"]:
                                    values = values.astype("int64")
                                else:
                                    values = values.astype("float64")
                            except:
                                # Fall back to string
                                values = np.array(
                                    [str(v).encode() for v in values],
                                    dtype="S",
                                )
                    except Exception as e:
                        # Final fallback
                        values = np.array(
                            [str(v).encode() for v in values], dtype="S"
                        )

                # Convert bool to int
                elif pd.api.types.is_bool_dtype(column.dtype):
                    values = values.astype("int64")

                # Preserve integer types for ID variables
                elif pd.api.types.is_integer_dtype(column.dtype):
                    if column_name.endswith("_id"):
                        values = values.astype("int64")
                    else:
                        values = values.astype("float64")

                # Use float64 for other numeric types (matching CPS format)
                elif pd.api.types.is_float_dtype(column.dtype):
                    values = values.astype("float64")

                try:
                    # Store variable directly at root level (flat structure)
                    f.create_dataset(column_name, data=values)
                except Exception as e:
                    print(f"  Warning: Could not save {column_name}: {e}")
                    continue
