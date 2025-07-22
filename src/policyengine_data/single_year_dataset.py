from pathlib import Path
from typing import Dict, Optional

import h5py
import pandas as pd
from policyengine_core.microsimulation import Microsimulation


class SingleYearDataset:
    def __init__(
        self,
        file_path: Optional[str] = None,
        entities: Optional[Dict[str, pd.DataFrame]] = None,
        fiscal_year: Optional[int] = 2025,
    ) -> None:
        self.entities: Dict[str, pd.DataFrame] = {}

        if file_path is not None:
            self.validate_file_path(file_path)
            with pd.HDFStore(file_path) as f:
                self.time_period = str(f["time_period"].iloc[0])
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
            self.time_period = str(fiscal_year)

        self.data_format = "arrays"
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
            fiscal_year=self.time_period,
        )

    def validate(self) -> None:
        # Check for NaNs in the tables
        for df in self.tables:
            for col in df.columns:
                if df[col].isna().any():
                    raise ValueError(f"Column '{col}' contains NaN values.")

    @staticmethod
    def from_simulation(
        simulation: "Microsimulation",
        fiscal_year: int = 2025,
        entity_names_to_include: Optional[list] = None,
    ) -> "SingleYearDataset":
        entity_dfs = {}

        # If no entity names specified, use all available entities
        if entity_names_to_include is None:
            entity_names = list(
                set(
                    simulation.tax_benefit_system.variables[var].entity.key
                    for var in simulation.input_variables
                )
            )
        else:
            entity_names = entity_names_to_include

        for entity in entity_names:
            input_variables = [
                variable
                for variable in simulation.input_variables
                if simulation.tax_benefit_system.variables[variable].entity.key
                == entity
            ]
            entity_dfs[entity] = simulation.calculate_dataframe(
                input_variables, period=fiscal_year
            )

        return SingleYearDataset(
            entities=entity_dfs,
            fiscal_year=fiscal_year,
        )
