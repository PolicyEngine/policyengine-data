"""
Test the logic for creating an estimate matrix from a database.
"""


def test_matrix_creation() -> None:
    from policyengine_data.calibration import (
        create_metrics_matrix,
        validate_metrics_matrix,
        download_database,
    )

    # Download database from Hugging Face Hub
    db_uri = download_database()

    # Create metrics matrix
    metrics_matrix, target_values, target_info = create_metrics_matrix(
        db_uri=db_uri,
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
    # this will work after the database online is updated
    # assert all(
    #     validation_results[validation_results["target_id"] < 19]["estimate"]
    #     != 0
    # ), "Metrics matrix should have all estimates non-zero for federal age targets"
