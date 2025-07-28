"""
Class for handling multi-year datasets in PolicyEngine.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd

from policyengine_data.single_year_dataset import SingleYearDataset


class MultiYearDataset:
    datasets: Dict[int, SingleYearDataset]

    def __init__(
        self,
        file_path: Optional[str] = None,
        datasets: Optional[List[SingleYearDataset]] = None,
    ):
        if datasets is not None:
            self.datasets = {}
            for dataset in datasets:
                if not isinstance(dataset, SingleYearDataset):
                    raise TypeError(
                        "All items in datasets must be of type SingleYearDataset."
                    )
                year = dataset.time_period
                self.datasets[year] = dataset

        if file_path is not None:
            self.validate_file_path(file_path)
            with pd.HDFStore(file_path) as f:
                self.datasets = {}

                # First, discover all years and entities in the file
                years_entities = {}  # {year: {entity_name: df}}

                for key in f.keys():
                    parts = key.strip("/").split("/")
                    if len(parts) == 2 and parts[0] != "time_period":
                        entity_name, year_str = parts
                        year = int(year_str)
                        if year not in years_entities:
                            years_entities[year] = {}
                        years_entities[year][entity_name] = f[key]

                # Create SingleYearDataset for each year
                for year, entities in years_entities.items():
                    self.datasets[year] = SingleYearDataset(
                        entities=entities,
                        time_period=year,
                    )

        self.data_format = "time_period_arrays"  # remove once -core does not expect different data formats
        self.time_period = (
            list(sorted(self.datasets.keys()))[0] if self.datasets else None
        )

    def get_year(self, time_period: int) -> "SingleYearDataset":
        if time_period in self.datasets:
            return self.datasets[time_period]
        else:
            raise ValueError(f"No dataset found for year {time_period}.")

    def __getitem__(self, time_period: int) -> "SingleYearDataset":
        return self.get_year(time_period)

    def save(self, file_path: str) -> None:
        Path(file_path).unlink(
            missing_ok=True
        )  # Remove existing file if it exists
        with pd.HDFStore(file_path) as f:
            for year, dataset in self.datasets.items():
                for entity_name, entity_df in dataset.entities.items():
                    f.put(
                        f"{entity_name}/{year}",
                        entity_df,
                        format="table",
                        data_columns=True,
                    )
                f.put(
                    f"time_period/{year}",
                    pd.Series([year]),
                    format="table",
                    data_columns=True,
                )

    def copy(self) -> "MultiYearDataset":
        new_datasets = {
            year: dataset.copy() for year, dataset in self.datasets.items()
        }
        return MultiYearDataset(datasets=list(new_datasets.values()))

    @staticmethod
    def validate_file_path(file_path: str) -> None:
        if not file_path.endswith(".h5"):
            raise ValueError(
                "File path must end with '.h5' for MultiYearDataset."
            )
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check if the file contains datasets for multiple years
        with h5py.File(file_path, "r") as f:
            required_entities = ["person", "household"]
            for entity in required_entities:
                if entity not in f:
                    raise ValueError(
                        f"No data for '{entity}' found in file: {file_path}"
                    )
                entity_group = f[entity]
                if not any(key.isdigit() for key in entity_group.keys()):
                    raise ValueError(
                        f"No year data for '{entity}' found in file: {file_path}"
                    )

    def load(self) -> Dict[str, Dict[int, np.ndarray]]:
        data = {}
        for year, dataset in self.datasets.items():
            for entity_name, entity_df in dataset.entities.items():
                for col in entity_df.columns:
                    if col not in data:
                        data[col] = {}
                    data[col][year] = entity_df[col].values
        return data

    def remove(self) -> None:
        """Removes the dataset from disk."""
        if self.exists():
            self.file_path.unlink()

    def store_file(self, file_path: str):
        """Moves a file to the dataset's file path.

        Args:
            file_path (str): The file path to move.
        """

        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist.")
        shutil.move(file_path, self.file_path)

    @property
    def variables(self) -> Dict[int, Dict[str, List[str]]]:
        """
        Returns a dictionary mapping years to entity variables dictionaries.
        """
        variables_by_year = {}

        for year, dataset in self.datasets.items():
            variables_by_year[year] = dataset.variables

        return variables_by_year

    @property
    def exists(self) -> bool:
        """Checks whether the dataset exists.

        Returns:
            bool: Whether the dataset exists.
        """
        return self.file_path.exists()
