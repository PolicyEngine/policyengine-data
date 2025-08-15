"""
This file will contain the logic for calibrating policy engine data from start to end. It will include different calibration routine options, from calibration at one geographic level to full calibration across all levels.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from policyengine_data import SingleYearDataset, normalise_table_keys
from policyengine_data.calibration.dataset_duplication import (
    load_dataset_for_geography_legacy,
    minimize_calibrated_dataset_legacy,
)
from policyengine_data.calibration.metrics_matrix_creation import (
    create_metrics_matrix,
    validate_metrics_matrix,
)
from policyengine_data.calibration.target_rescaling import (
    download_database,
    rescale_calibration_targets,
)
from policyengine_data.calibration.utils import (
    create_geographic_normalization_factor,
)
from policyengine_data.tools.legacy_class_conversions import (
    SingleYearDataset_to_Dataset,
)

logger = logging.getLogger(__name__)

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


def calibrate_single_geography_level(
    calibration_areas: Dict[str, str],
    dataset: str,
    stack_datasets: Optional[bool] = True,
    dataset_subsample_size: Optional[int] = None,
    geo_db_filter_variable: Optional[str] = "ucgid_str",
    geo_sim_filter_variable: Optional[str] = "ucgid",
    year: Optional[int] = 2023,
    db_uri: Optional[str] = None,
    noise_level: Optional[float] = 10.0,
    use_dataset_weights: Optional[bool] = True,
    regularize_with_l0: Optional[bool] = False,
    raise_error: Optional[bool] = True,
) -> "SingleYearDataset":
    """
    This function will calibrate the dataset for a specific geography level, defaulting to stacking the base dataset per area within it.
    It will handle conversion between dataset classes to enable:
        1. Rescaling calibration targets.
        2. Loading the base dataset and reassigning it to the specified geography.
        3. Selecting the appropriate targets that match each area at the geography level.
        4. Creating a metrics matrix that enables computing estimates for those targets.
        5. Calibrating the dataset's household weights with regularization.
        6. Filtering the resulting dataset to only include households with non-zero weights.
        7. Stacking all areas at that level into a single dataset.

    Args:
        calibration_areas (Dict[str, str]): A dictionary mapping area names to their corresponding geography level.
        dataset (str): The name of the dataset to be calibrated.
        stack_datasets (Optional[bool]): Whether to assign the dataset to each area in the geography level and combine them. Default: True.
        dataset_subsample_size (Optional[int]): The size of the base dataset subsample to use for calibration. If None, the full dataset will be used for stacking when enabled.
        year (Optional[int]): The year for which the calibration is being performed. Default: 2023.
        geo_db_filter_variable (str): The variable used to filter the database by geography. Default in the US: "ucgid_str".
        geo_sim_filter_variable (str): The variable used to filter the simulation by geography. Default in the US: "ucgid".
        db_uri (Optional[str]): The URI of the database to use for rescaling targets. If None, it will download the database from the default URI.
        noise_level (Optional[float]): The level of noise to apply during calibration. Default: 10.0.
        use_dataset_weights (Optional[bool]): Whether to use original dataset weights as the starting weights for calibration. Default: True.
        regularize_with_l0 (Optional[bool]): Whether to use L0 regularization during calibration. Default: False.
        raise_error (Optional[bool]): Whether to raise an error if matrix creation fails. Default: True.

    Returns:
        geography_level_calibrated_dataset (SingleYearDataset): The calibrated dataset for the specified geography level.
    """
    if db_uri is None:
        db_uri = download_database()

    # Rescale targets for consistency across geography areas
    rescaling_results = rescale_calibration_targets(
        db_uri=db_uri, update_database=True
    )

    geography_level_calibrated_dataset = None
    for area, geo_identifier in calibration_areas.items():
        logger.info(f"Calibrating dataset for {area}...")

        if stack_datasets:
            # Load dataset configured for the specific geography first
            from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import (
                UCGID,
            )

            sim_data_to_calibrate = load_dataset_for_geography_legacy(
                year=year,
                dataset=dataset,
                dataset_subsample_size=dataset_subsample_size,
                geography_variable=geo_sim_filter_variable,
                geography_identifier=UCGID(
                    geo_identifier
                ),  # will need a non-hardcoded solution to assign geography_identifier in the future
            )
        else:
            from policyengine_us import Microsimulation

            sim_data_to_calibrate = Microsimulation(dataset=dataset)
            sim_data_to_calibrate.default_input_period = year
            sim_data_to_calibrate.build_from_dataset()

        # Create metrics matrix for the area based on strata constraints using configured simulation
        metrics_matrix, targets, target_info = create_metrics_matrix(
            db_uri=db_uri,
            time_period=year,
            sim=sim_data_to_calibrate,
            stratum_filter_variable=geo_db_filter_variable,
            stratum_filter_value=geo_identifier,
            stratum_filter_operation="in",
        )
        metrics_evaluation = validate_metrics_matrix(
            metrics_matrix,
            targets,
            target_info=target_info,
            raise_error=raise_error,
        )

        target_names = []
        excluded_targets = []
        for target_id, info in target_info.items():
            target_names.append(info["name"])
            if not info["active"]:
                excluded_targets.append(target_id)
        target_names = np.array(target_names)

        if use_dataset_weights:
            weights = sim_data_to_calibrate.calculate(
                "household_weight"
            ).values
        else:
            weights = np.ones(len(metrics_matrix))

        # Calibrate with L0 regularization
        from microcalibrate import Calibration

        calibrator = Calibration(
            weights=weights,
            targets=targets,
            target_names=target_names,
            estimate_matrix=metrics_matrix,
            epochs=600,
            learning_rate=0.2,
            noise_level=noise_level,
            excluded_targets=(
                excluded_targets if len(excluded_targets) > 0 else None
            ),
            sparse_learning_rate=0.1,
            regularize_with_l0=regularize_with_l0,
            csv_path=f"{area}_calibration.csv",
        )
        performance_log = calibrator.calibrate()
        optimized_sparse_weights = calibrator.sparse_weights
        optimized_weights = calibrator.weights

        # Minimize the calibrated dataset storing only records with non-zero weights
        single_year_calibrated_dataset = minimize_calibrated_dataset_legacy(
            sim=sim_data_to_calibrate,
            year=year,
            optimized_weights=(
                optimized_sparse_weights
                if regularize_with_l0
                else optimized_weights
            ),
        )

        # Detect ids that require resetting after minimization
        primary_id_variables = {}
        for entity in single_year_calibrated_dataset.entities:
            primary_id_variables[entity] = f"{entity}_id"

        foreign_id_variables = {}
        for entity in single_year_calibrated_dataset.entities:
            entity_foreign_keys = {}
            for target_entity in single_year_calibrated_dataset.entities:
                if entity != target_entity:
                    foreign_key_name = f"{entity}_{target_entity}_id"
                    if (
                        foreign_key_name
                        in sim_data_to_calibrate.tax_benefit_system.variables
                    ) and (
                        foreign_key_name
                        in single_year_calibrated_dataset.entities[
                            entity
                        ].columns
                    ):
                        entity_foreign_keys[foreign_key_name] = target_entity

            if entity_foreign_keys:
                foreign_id_variables[entity] = entity_foreign_keys

        # Combine area datasets
        if geography_level_calibrated_dataset is None:
            geography_level_calibrated_dataset = single_year_calibrated_dataset
            single_year_calibrated_dataset.entities = normalise_table_keys(
                single_year_calibrated_dataset.entities,
                primary_keys=primary_id_variables,
                foreign_keys=foreign_id_variables,
                start_index=None,
            )
        else:
            previous_max_ids = {}
            for entity in single_year_calibrated_dataset.entities:
                previous_max_ids[entity] = (
                    geography_level_calibrated_dataset.entities[entity][
                        f"{entity}_id"
                    ].max()
                    + 1
                )

            single_year_calibrated_dataset.entities = normalise_table_keys(
                single_year_calibrated_dataset.entities,
                primary_keys=primary_id_variables,
                foreign_keys=foreign_id_variables,
                start_index=previous_max_ids,
            )

            geography_level_calibrated_dataset.entities = {
                entity: pd.concat(
                    [
                        geography_level_calibrated_dataset.entities[entity],
                        single_year_calibrated_dataset.entities[entity],
                    ],
                    ignore_index=True,
                )
                for entity in geography_level_calibrated_dataset.entities.keys()
            }

    return geography_level_calibrated_dataset


# TODO: create normalization factor to pass into Calibrator balancing targets at different levels
def calibrate_all_levels(
    database_stacking_areas: Dict[str, str],
    dataset: str,
    dataset_subsample_size: Optional[int] = None,
    geo_sim_filter_variable: Optional[str] = "ucgid",
    geo_hierarchy: Optional[List[str]] = None,
    year: Optional[int] = 2023,
    db_uri: Optional[str] = None,
    noise_level: Optional[float] = 10.0,
    regularize_with_l0: Optional[bool] = False,
    raise_error: Optional[bool] = True,
) -> "SingleYearDataset":
    """
    This function will calibrate the dataset for all geography levels in the database, defaulting to stacking the base dataset per area within the specified level (it is recommended to use the lowest in the hierarchy for stacking). (Eg. when calibrating for district, state and national levels in the US, this function will stack the CPS dataset for each district and calibrate the stacked dataset for the three levels' targets.)
    It will handle conversion between dataset classes to enable:
        1. Rescaling calibration targets.
        2. Loading the base dataset and reassigning it to the geographic areas within the lowest level.
        3. Stacking all areas at that level into a single dataset.
        4. Selecting all targets that match the specified geography levels.
        5. Creating a metrics matrix that enables computing estimates for those targets.
        6. Calibrating the dataset's household weights with regularization.
        7. Filtering the resulting dataset to only include households with non-zero weights.

    Args:
        database_stacking_areas (Dict[str, str]): A dictionary mapping area names to their identifiers for base dataset stacking.
        dataset (str): Path to the base dataset to stack.
        dataset_subsample_size (Optional[int]): The size of the subsample to use for calibration.
        geo_sim_filter_variable (Optional[str]): The variable to use for geographic similarity filtering. Default in the US: "ucgid".
        geo_hierarchy (Optional[List[str]]): The geographic hierarchy to use for calibration.
        year (Optional[int]): The year to use for calibration. Default: 2023.
        db_uri (Optional[str]): The database URI to use for calibration. If None, it will download the database from the default URI.
        noise_level (Optional[float]): The noise level to use for calibration. Default: 10.0.
        regularize_with_l0 (Optional[bool]): Whether to use L0 regularization for calibration. Default: False.
        raise_error (Optional[bool]): Whether to raise an error if matrix creation fails. Default: True.

    Returns:
        fully_calibrated_dataset (SingleYearDataset): The calibrated dataset for all geography levels.
    """
    if db_uri is None:
        db_uri = download_database()

    # Rescale targets for consistency across geography areas
    rescaling_results = rescale_calibration_targets(
        db_uri=db_uri, update_database=True
    )

    stacked_dataset = None
    for area, geo_identifier in database_stacking_areas.items():
        logger.info(f"Stacking dataset for {area}...")

        # Load dataset configured for the specific geographic area
        from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import (
            UCGID,
        )

        sim_data_to_stack = load_dataset_for_geography_legacy(
            year=year,
            dataset=dataset,
            dataset_subsample_size=dataset_subsample_size,
            geography_variable=geo_sim_filter_variable,
            geography_identifier=UCGID(
                geo_identifier
            ),  # will need a non-hardcoded solution to assign geography_identifier in the future
        )

        single_year_dataset = SingleYearDataset.from_simulation(
            simulation=sim_data_to_stack,
            time_period=year,
        )

        # Detect ids that require resetting
        primary_id_variables = {}
        for entity in single_year_dataset.entities:
            primary_id_variables[entity] = f"{entity}_id"

        foreign_id_variables = {}
        for entity in single_year_dataset.entities:
            entity_foreign_keys = {}
            for target_entity in single_year_dataset.entities:
                if entity != target_entity:
                    foreign_key_name = f"{entity}_{target_entity}_id"
                    if (
                        foreign_key_name
                        in sim_data_to_stack.tax_benefit_system.variables
                    ) and (
                        foreign_key_name
                        in single_year_dataset.entities[entity].columns
                    ):
                        entity_foreign_keys[foreign_key_name] = target_entity

            if entity_foreign_keys:
                foreign_id_variables[entity] = entity_foreign_keys

        # Combine datasets
        if stacked_dataset is None:
            stacked_dataset = single_year_dataset
            single_year_dataset.entities = normalise_table_keys(
                single_year_dataset.entities,
                primary_keys=primary_id_variables,
                foreign_keys=foreign_id_variables,
                start_index=None,
            )
        else:
            previous_max_ids = {}
            for entity in single_year_dataset.entities:
                previous_max_ids[entity] = (
                    stacked_dataset.entities[entity][f"{entity}_id"].max() + 1
                )

            single_year_dataset.entities = normalise_table_keys(
                single_year_dataset.entities,
                primary_keys=primary_id_variables,
                foreign_keys=foreign_id_variables,
                start_index=previous_max_ids,
            )

            stacked_dataset.entities = {
                entity: pd.concat(
                    [
                        stacked_dataset.entities[entity],
                        single_year_dataset.entities[entity],
                    ],
                    ignore_index=True,
                )
                for entity in stacked_dataset.entities.keys()
            }

    SingleYearDataset_to_Dataset(
        stacked_dataset,
        output_path="Dataset_stacked.h5",
    )

    logger.info(
        "Stacked dataset created successfully, starting to process it for calibration..."
    )

    metrics_matrix, targets, target_info = create_metrics_matrix(
        db_uri=db_uri,
        time_period=year,
        dataset="Dataset_stacked.h5",
        reform_id=0,
    )
    metrics_evaluation = validate_metrics_matrix(
        metrics_matrix,
        targets,
        target_info=target_info,
        raise_error=raise_error,
    )

    normalization_factor = create_geographic_normalization_factor(
        geo_hierarchy=geo_hierarchy, target_info=target_info
    )

    target_names = []
    excluded_targets = []
    for target_id, info in target_info.items():
        target_names.append(info["name"])
        if not info["active"]:
            excluded_targets.append(target_id)
    target_names = np.array(target_names)

    weights = np.ones(len(metrics_matrix))

    # Calibrate with L0 regularization
    from microcalibrate import Calibration

    calibrator = Calibration(
        weights=weights,
        targets=targets,
        target_names=target_names,
        estimate_matrix=metrics_matrix,
        epochs=600,
        learning_rate=0.2,
        noise_level=noise_level,
        excluded_targets=(
            excluded_targets if len(excluded_targets) > 0 else None
        ),
        normalization_factor=normalization_factor,
        sparse_learning_rate=0.1,
        regularize_with_l0=regularize_with_l0,
        csv_path=f"full_calibration.csv",
    )
    performance_log = calibrator.calibrate()
    optimized_sparse_weights = calibrator.sparse_weights
    optimized_weights = calibrator.weights

    from policyengine_us import Microsimulation

    sim_stacked_dataset = Microsimulation(dataset="Dataset_stacked.h5")
    sim_stacked_dataset.default_input_period = year
    sim_stacked_dataset.build_from_dataset()

    # Minimize the calibrated dataset storing only records with non-zero weights
    fully_calibrated_dataset = minimize_calibrated_dataset_legacy(
        sim=sim_stacked_dataset,
        year=year,
        optimized_weights=(
            optimized_sparse_weights
            if regularize_with_l0
            else optimized_weights
        ),
    )

    return fully_calibrated_dataset


if __name__ == "__main__":
    state_level_calibrated_dataset = calibrate_single_geography_level(
        areas_in_state_level,
        "hf://policyengine/policyengine-us-data/cps_2023.h5",
        use_dataset_weights=False,
        regularize_with_l0=True,
    )

    state_level_weights = state_level_calibrated_dataset.entities["household"][
        "household_weight"
    ].values

    SingleYearDataset_to_Dataset(
        state_level_calibrated_dataset,
        output_path="Dataset_state_level_age_medicaid_snap_eitc_agi_targets.h5",
    )

    print("Completed calibration for state level dataset.")

    print(
        "Number of household records at the state level:",
        len(state_level_calibrated_dataset.entities["household"]),
    )

    national_level_calibrated_dataset = calibrate_single_geography_level(
        areas_in_national_level,
        dataset="Dataset_state_level_age_medicaid_snap_eitc_agi_targets.h5",
        stack_datasets=False,
        noise_level=0.0,
        use_dataset_weights=True,
        regularize_with_l0=False,
    )

    national_level_weights = national_level_calibrated_dataset.entities[
        "household"
    ]["household_weight"].values

    SingleYearDataset_to_Dataset(
        national_level_calibrated_dataset,
        output_path="Dataset_national_level_age_medicaid_snap_eitc_agi_targets.h5",
    )

    print("Completed calibration for national level dataset.")

    print(
        "Number of household records at the national level:",
        len(national_level_calibrated_dataset.entities["household"]),
    )

    print("Weights comparison:")
    print(
        f"Weights from state level calibration - mean: {state_level_weights.mean()}; range: [{state_level_weights.min()}, {state_level_weights.max()}]"
    )
    print(
        f"Weights from national level calibration - mean: {national_level_weights.mean()}; range: [{national_level_weights.min()}, {national_level_weights.max()}]"
    )
