"""
This file will contain the logic for calibrating policy engine data from start to end. It will include functions for target rescaling, matrix creation, household duplication and assignment to new geographic areas, and final calibration.
"""

import logging
from typing import Dict, Optional

import numpy as np

from .. import normalise_table_keys
from .dataset_duplication import (
    load_dataset_for_geography_legacy,
    minimize_calibrated_dataset_legacy,
)
from .metrics_matrix_creation import (
    create_metrics_matrix,
    validate_metrics_matrix,
)
from .target_rescaling import download_database, rescale_calibration_targets

logger = logging.getLogger(__name__)


areas_in_geography_level = {"California": "0400000US06"}


def calibrate_geography_level(
    calibration_areas: Dict[str, str],
    dataset: str,
    geo_db_filter_variable: str = "ucgid_str",
    geo_sim_filter_variable: str = "ucgid",
    year: Optional[int] = 2023,
    db_uri: Optional[str] = None,
):
    """
    This function will calibrate the dataset for a specific geography level.
    It will handle conversion between dataset classes to enable:
        1. Rescaling calibration targets.
        2. Selecting the appropriate targets that match each area at the geography level.
        3. Creating a metrics matrix that enables computing estimates for those targets.
        4. Loading the dataset and reassigning it to the specified geography.
        5. Calibrating the dataset's household weights with regularization.
        6. Filtering the resulting dataset to only include households with non-zero weights.
        7. Stacking all areas at that level into a single dataset.

    Args:
        calibration_areas (Dict[str, str]): A dictionary mapping area names to their corresponding geography level.
        dataset (str): The name of the dataset to be calibrated.
        year (Optional[int]): The year for which the calibration is being performed. Default: 2023.
        geo_db_filter_variable (str): The variable used to filter the database by geography. Default in the US: "ucgid_str".
        geo_sim_filter_variable (str): The variable used to filter the simulation by geography. Default in the US: "ucgid".
        db_uri (Optional[str]): The URI of the database to use for rescaling targets. If None, it will download the database from the default URI.
    """
    if db_uri is None:
        db_uri = download_database()

    rescaling_results = rescale_calibration_targets(
        db_uri=db_uri, update_database=True
    )

    geography_level_calibrated_dataset = None
    for area, geo_identifier in calibration_areas.items():
        logger.info(f"Calibrating dataset for {area}...")

        metrics_matrix, targets, target_info = create_metrics_matrix(
            db_uri=db_uri,
            time_period=year,
            dataset=dataset,
            stratum_filter_variable=geo_db_filter_variable,
            stratum_filter_value=geo_identifier,
            stratum_filter_operation="in",
        )
        metrics_evaluation = validate_metrics_matrix(
            metrics_matrix,
            targets,
            target_info=target_info,
        )

        target_names = np.array()
        excluded_targets = []
        for target_id, info in target_info.items():
            target_names = np.append(target_names, info["name"])
            if not info["active"]:
                excluded_targets.append(target_id)

        from policyengine_us.variables.household.demographic.geographic.county.county_enum import (
            County,
        )

        sim_data_to_calibrate = load_dataset_for_geography_legacy(
            year=year,
            dataset=dataset,
            geography_variable=geo_sim_filter_variable,
            geography_identifier=County.cast(geo_identifier),
        )
        weights = sim_data_to_calibrate.calculate("household_weight").values

        from microcalibrate import Calibration

        calibrator = Calibration(
            weights=weights,
            targets=targets,
            target_names=target_names,
            estimate_matrix=metrics_matrix,
            epochs=200,
            excluded_targets=(
                excluded_targets if len(excluded_targets) > 0 else None
            ),
            regularize_with_l0=True,
        )
        performance_log = calibrator.calibrate()
        optimized_sparse_weights = calibrator.sparse_weights

        single_year_calibrated_dataset = minimize_calibrated_dataset_legacy(
            sim=sim_data_to_calibrate,
            year=year,
            optimized_sparse_weights=optimized_sparse_weights,
        )

        if geography_level_calibrated_dataset is None:
            geography_level_calibrated_dataset = single_year_calibrated_dataset
            single_year_calibrated_dataset.entities = normalise_table_keys(
                single_year_calibrated_dataset.entities,
                primary_keys={
                    "person": "person_id",
                    "household": "household_id",
                },
                foreign_keys={"person": {"household_id": "household"}},
                start_index=0,  # each id may need a different start index
            )
