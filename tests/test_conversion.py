"""
Test SingleYearDataset to legacy Dataset conversion functions.
"""

import sys

sys.path.insert(0, "src")

from policyengine_data.single_year_dataset import SingleYearDataset
from policyengine_data.tools.legacy_class_conversions import (
    SingleYearDataset_to_Dataset,
)
import numpy as np
import h5py
from pathlib import Path


def test_conversion():
    """Test the conversion functions"""
    from policyengine_us import Microsimulation

    start_year = 2023
    dataset = "hf://policyengine/policyengine-us-data/cps_2023.h5"

    # Load original CPS data
    sim = Microsimulation(dataset=dataset)
    single_year_dataset = SingleYearDataset.from_simulation(
        sim, time_period=start_year
    )
    single_year_dataset.time_period = start_year

    # Assert we have expected entities
    assert (
        len(single_year_dataset.entities) == 6
    ), f"Expected 6 entities, got {len(single_year_dataset.entities)}"
    expected_entities = {
        "person",
        "household",
        "tax_unit",
        "spm_unit",
        "family",
        "marital_unit",
    }
    actual_entities = set(single_year_dataset.entities.keys())
    assert (
        actual_entities == expected_entities
    ), f"Entity mismatch: {actual_entities} vs {expected_entities}"

    # Test conversion to legacy format
    output_path = Path("test_legacy_dataset.h5")
    SingleYearDataset_to_Dataset(
        single_year_dataset, output_path, time_period=2024
    )

    # Assert output file was created
    assert output_path.exists(), f"Output file {output_path} was not created"

    # Verify h5 file structure
    with h5py.File(output_path, "r") as f:
        variables = list(f.keys())
        assert (
            len(variables) > 100
        ), f"Expected >100 variables, got {len(variables)}"

        # Check that important variables exist
        important_vars = [
            "person_id",
            "household_id",
            "age",
            "employment_income_last_year",
        ]
        for var in important_vars:
            assert (
                var in variables
            ), f"Important variable {var} missing from saved file"

    # Loading back to SingleYearDataset
    sim_loaded = Microsimulation(dataset=str(output_path))
    loaded_single_year_dataset = SingleYearDataset.from_simulation(
        sim_loaded, time_period=2024
    )

    # Assert loaded dataset has same entities
    assert len(loaded_single_year_dataset.entities) == len(
        single_year_dataset.entities
    ), f"Loaded dataset has {len(loaded_single_year_dataset.entities)} entities, expected {len(single_year_dataset.entities)}"

    # Compare original and loaded data
    for entity_name in single_year_dataset.entities.keys():
        assert (
            entity_name in loaded_single_year_dataset.entities
        ), f"Entity {entity_name} missing in loaded dataset"

        original = single_year_dataset.entities[entity_name]
        loaded = loaded_single_year_dataset.entities[entity_name]

        assert len(original) == len(
            loaded
        ), f"{entity_name}: Record count mismatch - original {len(original)}, loaded {len(loaded)}"
        assert len(original.columns) == len(
            loaded.columns
        ), f"{entity_name}: Column count mismatch - original {len(original.columns)}, loaded {len(loaded.columns)}"

        common_cols = set(original.columns) & set(loaded.columns)
        assert len(common_cols) == len(
            original.columns
        ), f"{entity_name}: Not all columns preserved. Missing: {set(original.columns) - common_cols}"

    # Clean up
    output_path.unlink(missing_ok=True)
