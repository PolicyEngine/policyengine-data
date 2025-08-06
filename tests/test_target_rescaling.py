"""
Test the logic for rescaling calibration targets from a database.
"""


def setup_test_database():
    """
    Creates an in-memory SQLite database for testing.
    Populates it with a geographic hierarchy where children do not sum to the parent.
    - Parent: USA (stratum_id=1), value=1000
    - Children: NC (stratum_id=2, value=400), CA (stratum_id=3, value=500)
    - Children Sum = 900, which requires scaling.
    """
    import pandas as pd
    from sqlalchemy import create_engine

    # It's going to create a database from scratch
    db_uri = "sqlite:///test_policy_data.db"
    engine = create_engine(db_uri)

    # Define schema
    strata_schema = """
    CREATE TABLE strata (
        stratum_id INTEGER PRIMARY KEY,
        stratum_group_id INTEGER,
        parent_stratum_id INTEGER,
        notes TEXT,
        definition_hash TEXT
    )
    """
    targets_schema = """
    CREATE TABLE targets (
        target_id INTEGER PRIMARY KEY,
        stratum_id INTEGER,
        variable TEXT,
        period INTEGER,
        reform_id INTEGER,
        value REAL,
        active BOOLEAN,
        tolerance REAL,
        FOREIGN KEY(stratum_id) REFERENCES strata(stratum_id)
    )
    """

    with engine.connect() as conn:
        # Drop tables if they exist to ensure clean state
        conn.exec_driver_sql("DROP TABLE IF EXISTS targets")
        conn.exec_driver_sql("DROP TABLE IF EXISTS strata")
        conn.exec_driver_sql(strata_schema)
        conn.exec_driver_sql(targets_schema)

    # Create test data
    strata_data = pd.DataFrame(
        [
            {
                "stratum_id": 1,
                "stratum_group_id": 10,
                "parent_stratum_id": None,
                "notes": "USA Total",
            },
            {
                "stratum_id": 2,
                "stratum_group_id": 11,
                "parent_stratum_id": 1,
                "notes": "North Carolina",
            },
            {
                "stratum_id": 3,
                "stratum_group_id": 11,
                "parent_stratum_id": 1,
                "notes": "California",
            },
            # Add another stratum for a different hierarchy to ensure isolation
            {
                "stratum_id": 4,
                "stratum_group_id": 20,
                "parent_stratum_id": None,
                "notes": "Wealth Total",
            },
        ]
    )

    targets_data = pd.DataFrame(
        [
            # --- Primary test case: 'income' for 2025 ---
            {
                "target_id": 101,
                "stratum_id": 1,
                "variable": "income",
                "period": 2025,
                "reform_id": 0,
                "value": 1000.0,
                "active": True,
                "tolerance": 0.01,
            },
            {
                "target_id": 102,
                "stratum_id": 2,
                "variable": "income",
                "period": 2025,
                "reform_id": 0,
                "value": 400.0,
                "active": True,
                "tolerance": 0.05,
            },
            {
                "target_id": 103,
                "stratum_id": 3,
                "variable": "income",
                "period": 2025,
                "reform_id": 0,
                "value": 500.0,
                "active": True,
                "tolerance": 0.05,
            },
            # --- Data that should be ignored by the test ---
            # Inactive target
            {
                "target_id": 104,
                "stratum_id": 2,
                "variable": "income",
                "period": 2025,
                "reform_id": 0,
                "value": 9999.0,
                "active": False,
                "tolerance": 0.05,
            },
            # Target for a different variable
            {
                "target_id": 105,
                "stratum_id": 4,
                "variable": "wealth",
                "period": 2025,
                "reform_id": 0,
                "value": 5000.0,
                "active": True,
                "tolerance": 0.01,
            },
        ]
    )

    # Load data into the database
    strata_data.to_sql("strata", engine, if_exists="append", index=False)
    targets_data.to_sql("targets", engine, if_exists="append", index=False)

    return db_uri


def test_rescale_with_geographic_scaling():
    """
    Tests that child strata (states) are correctly scaled to match the
    parent stratum (nation) total.
    """
    import pytest
    from policyengine_data.calibration.target_rescaling import (
        rescale_calibration_targets,
    )

    db_uri = setup_test_database()
    variable_to_test = "income"
    period_to_test = 2025
    reform_id_to_test = 0

    # Execute the function
    results_df = rescale_calibration_targets(
        db_uri=db_uri,
        variable=variable_to_test,
        period=period_to_test,
        reform_id=reform_id_to_test,
        update_database=False,  # We don't need to update the DB for this test
    )

    # --- Verification ---
    assert not results_df.empty, "Result DataFrame should not be empty"
    assert (
        len(results_df) == 3
    ), "Should only process the 3 active 'income' targets"

    # Define expected values
    parent_value = 1000.0
    children_sum = 400.0 + 500.0
    expected_scaling_factor = (
        parent_value / children_sum
    )  # 1000 / 900 = 1.111...

    # Extract results for each stratum for clarity
    parent_usa = results_df[results_df["stratum_id"] == 1].iloc[0]
    child_nc = results_df[results_df["stratum_id"] == 2].iloc[0]
    child_ca = results_df[results_df["stratum_id"] == 3].iloc[0]

    # 1. Check Parent (USA) - should be unchanged
    assert parent_usa["value"] == 1000.0
    assert parent_usa["scaled_value"] == 1000.0
    assert parent_usa["scaling_factor"] == 1.0

    # 2. Check Children (NC and CA) - should be scaled
    # Use pytest.approx for floating point comparisons
    assert child_nc["scaling_factor"] == pytest.approx(expected_scaling_factor)
    assert child_nc["scaled_value"] == pytest.approx(
        400.0 * expected_scaling_factor
    )

    assert child_ca["scaling_factor"] == pytest.approx(expected_scaling_factor)
    assert child_ca["scaled_value"] == pytest.approx(
        500.0 * expected_scaling_factor
    )

    # 3. Check consistency: sum of scaled children should equal parent value
    sum_of_scaled_children = (
        child_nc["scaled_value"] + child_ca["scaled_value"]
    )
    assert sum_of_scaled_children == pytest.approx(parent_usa["scaled_value"])

    print("\nTest passed: Geographic scaling successful.")
    print(f"Original Children Sum: {children_sum}")
    print(f"Parent Total: {parent_value}")
    print(f"Calculated Scaling Factor: {expected_scaling_factor:.4f}")
    print(
        f"Sum of Scaled Children: {sum_of_scaled_children:.4f} (matches parent total)"
    )
    print("\nFinal Results DataFrame:")
    print(results_df)

    # Cleanup
    import os

    if os.path.exists("test_policy_data.db"):
        os.remove("test_policy_data.db")
