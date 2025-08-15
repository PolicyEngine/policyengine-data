"""
Test the logic for assigning a dataset to a geographic level and minimizing it.
"""

from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import (
    UCGID,
)
from policyengine_data import SingleYearDataset


def test_dataset_assignment_to_geography() -> None:
    """Test that a dataset can be assigned to a geographic level without errors."""
    from policyengine_data.calibration import load_dataset_for_geography_legacy

    sim = load_dataset_for_geography_legacy()

    assert hasattr(sim, "dataset")
    assert hasattr(sim, "default_input_period")
    assert sim.default_input_period == 2023

    # Verify household data exists
    household_ids = sim.calculate("household_id").values
    assert len(household_ids) > 0

    # Verify geography is set correctly
    ucgid_values = sim.calculate("ucgid").values
    expected_ucgid = UCGID("0100000US")
    # The system returns enum names as strings, so compare with the name
    assert all(val == expected_ucgid.name for val in ucgid_values)

    # Test with California state identifier
    california_ucgid = UCGID("0400000US06")
    sim = load_dataset_for_geography_legacy(
        geography_identifier=california_ucgid
    )

    # Verify geography is set correctly
    ucgid_values = sim.calculate("ucgid").values
    # The system returns enum names as strings, so compare with the name
    assert all(val == california_ucgid.name for val in ucgid_values)


def test_dataset_minimization() -> None:
    """Test that a dataset can be minimized using sparse weights."""
    from policyengine_data.calibration import (
        minimize_calibrated_dataset_legacy,
    )
    from policyengine_us import Microsimulation
    import pandas as pd

    # Load the dataset
    sim = Microsimulation(
        dataset="hf://policyengine/policyengine-us-data/cps_2023.h5"
    )
    sim.default_input_period = 2023
    sim.build_from_dataset()

    before_minimizing = SingleYearDataset.from_simulation(
        sim, time_period=2023
    )
    before_minimizing.time_period = 2023

    # Create dummy sparse weights
    household_ids = sim.calculate("household_id").values
    optimized_sparse_weights = pd.Series(
        [1.0] * (len(household_ids) // 2)
        + [0.0] * (len(household_ids) - (len(household_ids) // 2))
    )

    # Get age values before minimization for comparison
    age_before = sim.calculate("age", 2023).values

    # Minimize the dataset
    after_minimizing = minimize_calibrated_dataset_legacy(
        sim, year=2023, optimized_weights=optimized_sparse_weights
    )

    assert len(before_minimizing.entities["household"]) > len(
        after_minimizing.entities["household"]
    )
    assert (
        abs(
            len(before_minimizing.entities["household"])
            - 2 * len(after_minimizing.entities["household"])
        )
        < 2
    )

    # Check that age values did not change for the records that were kept
    age_after = after_minimizing.entities["person"]["age"].values
    kept_person_ids = after_minimizing.entities["person"]["person_id"].values

    # Find the indices of these person IDs in the original dataset
    original_person_ids = before_minimizing.entities["person"][
        "person_id"
    ].values
    kept_indices = [
        i
        for i, pid in enumerate(original_person_ids)
        if pid in kept_person_ids
    ]

    # Compare age values for kept records
    age_before_kept = age_before[kept_indices]
    assert pd.Series(age_before_kept).equals(
        pd.Series(age_after)
    ), "Age values should not change for records that were kept"


def test_dataset_subsampling() -> None:
    """Test that dataset subsampling works correctly."""
    from policyengine_data.calibration import load_dataset_for_geography_legacy

    # Load full dataset first
    sim_full = load_dataset_for_geography_legacy()
    full_households = len(sim_full.calculate("household_id").unique())

    # Test subsampling with a smaller size
    subsample_size = min(
        100, full_households // 2
    )  # Ensure we're actually reducing the size
    sim_subsampled = load_dataset_for_geography_legacy(
        dataset_subsample_size=subsample_size
    )

    subsampled_households = len(
        sim_subsampled.calculate("household_id").unique()
    )

    # Verify the subsampled dataset has the expected number of households
    assert (
        subsampled_households == subsample_size
    ), f"Expected {subsample_size} households, got {subsampled_households}"

    # Verify geography is still set correctly after subsampling
    expected_ucgid = UCGID("0100000US")
    ucgid_values = sim_subsampled.calculate("ucgid").values
    assert all(val == expected_ucgid.name for val in ucgid_values)

    # Test with a subsample size larger than available households (should return original)
    sim_large_subsample = load_dataset_for_geography_legacy(
        dataset_subsample_size=full_households + 1000
    )
    large_subsample_households = len(
        sim_large_subsample.calculate("household_id").unique()
    )
    assert large_subsample_households == full_households
