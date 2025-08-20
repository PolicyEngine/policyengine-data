"""
Class for handling single-year datasets in PolicyEngine.
"""

import shutil
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import pandas as pd
from policyengine_core.simulations import Microsimulation


class SingleYearDataset:
    entities: Dict[str, pd.DataFrame]
    time_period: int  # manually convert to str when -core expects str

    def __init__(
        self,
        file_path: Optional[str] = None,
        entities: Optional[Dict[str, pd.DataFrame]] = None,
        time_period: Optional[int] = 2025,
    ) -> None:
        self.entities: Dict[str, pd.DataFrame] = {}

        if file_path is not None:
            self.validate_file_path(file_path)
            with pd.HDFStore(file_path) as f:
                self.time_period = int(f["time_period"].iloc[0])
                # Load all entities from the file (except time_period)
                for key in f.keys():
                    if key != "/time_period":
                        entity_name = key.strip("/")
                        self.entities[entity_name] = f[entity_name]
        else:
            if entities is None:
                raise ValueError(
                    "Must provide either a file path or a dictionary of entities' dataframes."
                )
            self.entities = entities.copy()
            self.time_period = time_period

        self.data_format = "arrays"  # remove once -core does not expect different data formats
        self.tables = tuple(self.entities.values())
        self.table_names = tuple(self.entities.keys())

    @staticmethod
    def validate_file_path(file_path: str) -> None:
        if not file_path.endswith(".h5"):
            raise ValueError("File path must end with '.h5' for Dataset.")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with h5py.File(file_path, "r") as f:
            required_datasets = [
                "time_period",
                "person",
                "household",
            ]  # all datasets will have at least person and household entities
            for dataset in required_datasets:
                if dataset not in f:
                    raise ValueError(
                        f"Dataset '{dataset}' not found in the file: {file_path}"
                    )

    def save(self, file_path: str) -> None:
        with pd.HDFStore(file_path) as f:
            for entity, df in self.entities.items():
                f.put(entity, df, format="table", data_columns=True)
            f.put("time_period", pd.Series([self.time_period]), format="table")

    def load(self) -> Dict[str, pd.Series]:
        data = {}
        for entity_name, entity_df in self.entities.items():
            for col in entity_df.columns:
                data[col] = entity_df[col].values

        return data

    def copy(self) -> "SingleYearDataset":
        return SingleYearDataset(
            entities={name: df.copy() for name, df in self.entities.items()},
            time_period=self.time_period,
        )

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

    def validate(self) -> None:
        # Check for NaNs in the tables
        for df in self.tables:
            for col in df.columns:
                if df[col].isna().any():
                    raise ValueError(f"Column '{col}' contains NaN values.")

    @staticmethod
    def from_simulation(
        simulation: "Microsimulation",
        time_period: int = 2025,
        entity_names_to_include: Optional[List[str]] = None,
        include_all_variables: bool = False,
        additional_variables: Optional[Dict[str, List[str]]] = None,
    ) -> "SingleYearDataset":
        """
        Create a SingleYearDataset from a Microsimulation.

        Args:
            simulation: The Microsimulation to extract data from
            time_period: The time period for the data
            entity_names_to_include: Specific entities to include (None = all)
            include_all_variables: If True, include ALL variables (both input and calculated)
            additional_variables: Dict mapping entity names to lists of additional variables to include beyond input variables.
            E.g., {"person": ["employment_income"], "tax_unit": ["eitc", "adjusted_gross_income"]}

        Returns:
            SingleYearDataset with the specified variables
        """
        entity_dfs = {}

        # Determine which entities to process
        if include_all_variables:
            # When including all variables, get entities from all variables
            entity_names = list(
                set(
                    simulation.tax_benefit_system.variables[var].entity.key
                    for var in simulation.tax_benefit_system.variables
                )
            )
        elif entity_names_to_include is not None:
            entity_names = entity_names_to_include
        else:
            # Default: get entities from input variables
            entity_names = list(
                set(
                    simulation.tax_benefit_system.variables[var].entity.key
                    for var in simulation.input_variables
                )
            )

        # Process each entity
        for entity in entity_names:
            variables_to_include = []

            if include_all_variables:
                # Get ALL variables for this entity (already filtered by entity)
                variables_to_include = [
                    var_name
                    for var_name in simulation.tax_benefit_system.variables
                    if simulation.tax_benefit_system.variables[
                        var_name
                    ].entity.key
                    == entity
                ]
            else:
                # Start with input variables for this entity
                variables_to_include = [
                    variable
                    for variable in simulation.input_variables
                    if simulation.tax_benefit_system.variables[
                        variable
                    ].entity.key
                    == entity
                ]

                # Add any additional specified variables for this entity
                if additional_variables and entity in additional_variables:
                    for var in additional_variables[entity]:
                        # Verify the variable exists, belongs to this entity, and isn't already included
                        if (
                            var in simulation.tax_benefit_system.variables
                            and simulation.tax_benefit_system.variables[
                                var
                            ].entity.key
                            == entity
                            and var not in variables_to_include
                        ):
                            variables_to_include.append(var)

            # Calculate all variables for this entity (all should belong to the entity now)
            if variables_to_include:
                entity_dfs[entity] = simulation.calculate_dataframe(
                    variables_to_include, period=time_period
                )

        return SingleYearDataset(
            entities=entity_dfs,
            time_period=time_period,
        )

    @property
    def variables(self) -> Dict[str, List[str]]:
        """
        Returns a dictionary mapping entity names to lists of variables (column names).
        """
        variables_by_entity = {}

        for entity_name, entity_df in self.entities.items():
            variables_by_entity[entity_name] = entity_df.columns.tolist()

        return variables_by_entity

    @property
    def exists(self) -> bool:
        """Checks whether the dataset exists.

        Returns:
            bool: Whether the dataset exists.
        """
        return self.file_path.exists()
