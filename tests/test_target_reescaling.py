"""
Test the logic for rescaling calibration targets from a database.
"""


def test_rescale_calibration_targets_from_db() -> None:
    from policyengine_data.calibration import rescale_calibration_targets

    # Local SQLite database Ben created
    db_uri = "sqlite:////Users/movil1/Desktop/PYTHONJOBS/PolicyEngine/policy_data.db"

    results = rescale_calibration_targets(db_uri=db_uri)

    assert results.columns.tolist() == [
        "target_id",
        "stratum_id",
        "stratum_group_id",
        "parent_stratum_id",
        "variable",
        "period",
        "reform_id",
        "value",
        "scaled_value",
        "scaling_factor",
    ], "Missing columns for rescaled targets and scaling factors"
    assert results.value.equals(
        results.scaled_value
    ), "Values and scaled values should match for age targets as state and national values matched"
