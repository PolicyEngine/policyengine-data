"""
Test the logic for creating an estimate matrix from a database.
"""


def test_matrix_creation() -> None:
    from policyengine_data.calibration.metrics_matrix_creation import (
        create_metrics_matrix,
        validate_metrics_matrix,
    )

    # Loading local database
    local_file = "sqlite:///src/policyengine_data/calibration/policy_data.db"
    # Placeholder for actual database URI once it is published from when published to Google Cloud
    db_uri = None

    # Create metrics matrix
    metrics_matrix, target_values, target_info = create_metrics_matrix(
        db_uri=local_file,
        time_period=2023,
        dataset="hf://policyengine/policyengine-us-data/cps_2023.h5",
        reform_id=0,
    )

    # Validate the matrix
    validation_results = validate_metrics_matrix(
        metrics_matrix, target_values, target_info=target_info
    )

    assert metrics_matrix.columns.tolist() == [
        i for i in range(1, 937)
    ], "Metrics matrix columns do not match expected target ids"
    # once ucgid_str is fixed in -us and the database
    # assert metrics_matrix.iloc[:, 0].sum() != 0, "The first column of the metrics matrix should not be full of zeros"
