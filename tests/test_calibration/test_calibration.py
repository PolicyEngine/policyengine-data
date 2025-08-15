"""
Test the calibration logic for different geographic levels that integrates all other calibration pipeline components.
"""

import pytest

areas_in_national_level = {
    "United States": "0100000US",
}

areas_in_state_level = {
    "Alabama": "0400000US01",
    "Alaska": "0400000US02",
    "Arizona": "0400000US04",
    "Arkansas": "0400000US05",
    "California": "0400000US06",
    "Colorado": "0400000US08",
    "Connecticut": "0400000US09",
    "Delaware": "0400000US10",
    "District of Columbia": "0400000US11",
    "Florida": "0400000US12",
    "Georgia": "0400000US13",
    "Hawaii": "0400000US15",
    "Idaho": "0400000US16",
    "Illinois": "0400000US17",
    "Indiana": "0400000US18",
    "Iowa": "0400000US19",
    "Kansas": "0400000US20",
    "Kentucky": "0400000US21",
    "Louisiana": "0400000US22",
    "Maine": "0400000US23",
    "Maryland": "0400000US24",
    "Massachusetts": "0400000US25",
    "Michigan": "0400000US26",
    "Minnesota": "0400000US27",
    "Mississippi": "0400000US28",
    "Missouri": "0400000US29",
    "Montana": "0400000US30",
    "Nebraska": "0400000US31",
    "Nevada": "0400000US32",
    "New Hampshire": "0400000US33",
    "New Jersey": "0400000US34",
    "New Mexico": "0400000US35",
    "New York": "0400000US36",
    "North Carolina": "0400000US37",
    "North Dakota": "0400000US38",
    "Ohio": "0400000US39",
    "Oklahoma": "0400000US40",
    "Oregon": "0400000US41",
    "Pennsylvania": "0400000US42",
    "Rhode Island": "0400000US44",
    "South Carolina": "0400000US45",
    "South Dakota": "0400000US46",
    "Tennessee": "0400000US47",
    "Texas": "0400000US48",
    "Utah": "0400000US49",
    "Vermont": "0400000US50",
    "Virginia": "0400000US51",
    "Washington": "0400000US53",
    "West Virginia": "0400000US54",
    "Wisconsin": "0400000US55",
    "Wyoming": "0400000US56",
}


def test_calibration_per_geographic_level_iteration():
    """
    Test and example of the calibration routine involving calibrating one geographic level at a time from lowest to highest in the hierarchy and generating sparsity in all but the last levels.

    Conversion between dataset class types is necessary until full migration to the new SingleYearDataset class in the policyengine_core repository.
    """
    from policyengine_data.tools.legacy_class_conversions import (
        SingleYearDataset_to_Dataset,
    )
    from policyengine_data.calibration.calibrate import (
        calibrate_single_geography_level,
    )

    # Calibrate the state level dataset with sparsity
    state_level_calibrated_dataset = calibrate_single_geography_level(
        areas_in_state_level,
        "hf://policyengine/policyengine-us-data/cps_2023.h5",
        dataset_subsample_size=2000,  # approximately 10% of the base dataset to decrease computation costs
        use_dataset_weights=False,
        regularize_with_l0=True,
    )

    state_level_weights = state_level_calibrated_dataset.entities["household"][
        "household_weight"
    ].values

    SingleYearDataset_to_Dataset(
        state_level_calibrated_dataset, output_path="Dataset_state_level.h5"
    )

    # Calibrate the national level dataset using the previously calibrated state dataset, without sparsity, and without initial noise (trying to minimize deviation from state-calibrated weights)
    national_level_calibrated_dataset = calibrate_single_geography_level(
        areas_in_national_level,
        dataset="Dataset_state_level.h5",
        stack_datasets=False,
        noise_level=0.0,
        use_dataset_weights=True,  # use the previously calibrated weights
        regularize_with_l0=False,
    )

    national_level_weights = national_level_calibrated_dataset.entities[
        "household"
    ]["household_weight"].values

    SingleYearDataset_to_Dataset(
        national_level_calibrated_dataset,
        output_path="Dataset_national_level.h5",
    )

    assert len(state_level_calibrated_dataset.entities["household"]) == len(
        national_level_calibrated_dataset.entities["household"]
    ), "Household record counts do not match after national calibration."

    assert (
        state_level_weights - national_level_weights
    ).sum() > 0, "Household weights do not differ between state and national levels, suggesting national calibration was unsucessful."


def test_calibration_combining_all_levels_at_once():
    """
    Test and example of the calibration routine involving stacking datasets at a single (most often lowest) geographic level for increased data richness and then calibrating said stacked dataset for all geographic levels at once.

    Conversion between dataset class types is necessary until full migration to the new SingleYearDataset class in the policyengine_core repository.
    """
    from policyengine_data.tools.legacy_class_conversions import (
        SingleYearDataset_to_Dataset,
    )
    from policyengine_data.calibration.calibrate import (
        calibrate_all_levels,
    )

    # Calibrate the full dataset at once (only passing the identifyers of the areas for which the base dataset will be stacked)
    fully_calibrated_dataset = calibrate_all_levels(
        areas_in_state_level,
        "hf://policyengine/policyengine-us-data/cps_2023.h5",
        db_uri="sqlite:///policy_data.db",  # remove once online database is updated
        geo_hierarchy=["0100000US", "0400000US"],
        dataset_subsample_size=2000,
        regularize_with_l0=True,
        raise_error=False,  # this will avoid raising an error if some targets have no records contributing to them (given sampling)
    )

    weights = fully_calibrated_dataset.entities["household"][
        "household_weight"
    ].values

    SingleYearDataset_to_Dataset(
        fully_calibrated_dataset, output_path="Dataset_fully_calibrated.h5"
    )

    assert len(weights) < 2000 * len(
        areas_in_state_level
    ), "Weight vector length should be less than the sampled 2000 per area after regularization."
