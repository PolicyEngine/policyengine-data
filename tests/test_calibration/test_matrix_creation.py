"""
Test the logic for creating an estimate matrix from a database.
"""

import numpy as np
import pandas as pd
import pytest


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


def test_parse_constraint_value():
    """Test parsing constraint values from strings."""
    from policyengine_data.calibration.metrics_matrix_creation import (
        parse_constraint_value,
    )

    # Test boolean values
    assert parse_constraint_value("true", "equals") == True
    assert parse_constraint_value("false", "equals") == False
    assert parse_constraint_value("True", "equals") == True
    assert parse_constraint_value("FALSE", "equals") == False

    # Test integer values
    assert parse_constraint_value("42", "equals") == 42
    assert parse_constraint_value("0", "equals") == 0
    assert parse_constraint_value("-10", "greater_than") == -10

    # Test float values
    assert parse_constraint_value("3.14", "less_than") == 3.14
    assert parse_constraint_value("0.0", "equals") == 0

    # Test list values for "in" operation
    result = parse_constraint_value("apple,banana,cherry", "in")
    assert result == ["apple", "banana", "cherry"]

    # Test string values
    assert parse_constraint_value("hello", "equals") == "hello"
    assert parse_constraint_value("test_string", "not_equals") == "test_string"


def test_apply_single_constraint():
    """Test applying single constraints to create boolean masks."""
    from policyengine_data.calibration.metrics_matrix_creation import (
        apply_single_constraint,
    )

    # Test data
    values = np.array([1, 2, 3, 4, 5])

    # Test equals operation
    mask = apply_single_constraint(values, "equals", 3)
    expected = np.array([False, False, True, False, False])
    np.testing.assert_array_equal(mask, expected)

    # Test greater_than operation
    mask = apply_single_constraint(values, "greater_than", 3)
    expected = np.array([False, False, False, True, True])
    np.testing.assert_array_equal(mask, expected)

    # Test less_than_or_equal operation
    mask = apply_single_constraint(values, "less_than_or_equal", 3)
    expected = np.array([True, True, True, False, False])
    np.testing.assert_array_equal(mask, expected)

    # Test not_equals operation
    mask = apply_single_constraint(values, "not_equals", 3)
    expected = np.array([True, True, False, True, True])
    np.testing.assert_array_equal(mask, expected)

    # Test "in" operation with string values
    str_values = np.array(["apple", "banana", "cherry", "date"])
    mask = apply_single_constraint(str_values, "in", "an")
    expected = np.array([False, True, False, False])  # "an" is in "banana"
    np.testing.assert_array_equal(mask, expected)

    # Test "in" operation with list
    mask = apply_single_constraint(str_values, "in", ["app", "che"])
    expected = np.array(
        [True, False, True, False]
    )  # "app" in "apple", "che" in "cherry"
    np.testing.assert_array_equal(mask, expected)

    # Test invalid operation
    with pytest.raises(ValueError, match="Unknown operation"):
        apply_single_constraint(values, "invalid_op", 3)


def test_parse_constraint_for_name():
    """Test parsing constraints into human-readable names."""
    from policyengine_data.calibration.metrics_matrix_creation import (
        parse_constraint_for_name,
    )

    # Test different operations
    constraint_data = [
        (
            {
                "constraint_variable": "age",
                "operation": "equals",
                "value": "30",
            },
            "age=30",
        ),
        (
            {
                "constraint_variable": "income",
                "operation": "greater_than",
                "value": "50000",
            },
            "income>50000",
        ),
        (
            {
                "constraint_variable": "score",
                "operation": "less_than_or_equal",
                "value": "100",
            },
            "score<=100",
        ),
        (
            {
                "constraint_variable": "status",
                "operation": "not_equals",
                "value": "active",
            },
            "status!=active",
        ),
        (
            {
                "constraint_variable": "category",
                "operation": "in",
                "value": "A,B,C",
            },
            "category_in_A_B_C",
        ),
    ]

    for constraint_dict, expected in constraint_data:
        constraint = pd.Series(constraint_dict)
        result = parse_constraint_for_name(constraint)
        assert result == expected, f"Expected {expected}, got {result}"


def test_build_target_name():
    """Test building descriptive target names."""
    from policyengine_data.calibration.metrics_matrix_creation import (
        build_target_name,
    )

    # Test with no constraints
    assert build_target_name("population", pd.DataFrame()) == "population"

    # Test with single constraint
    constraints = pd.DataFrame(
        [{"constraint_variable": "age", "operation": "equals", "value": "30"}]
    )
    result = build_target_name("income", constraints)
    assert result == "income_age=30"

    # Test with multiple constraints (should be sorted)
    constraints = pd.DataFrame(
        [
            {
                "constraint_variable": "state",
                "operation": "equals",
                "value": "CA",
            },
            {
                "constraint_variable": "age",
                "operation": "greater_than",
                "value": "18",
            },
        ]
    )
    result = build_target_name("population", constraints)
    assert result == "population_age>18_state=CA"

    # Test with ucgid constraint (should come first)
    constraints = pd.DataFrame(
        [
            {
                "constraint_variable": "age",
                "operation": "equals",
                "value": "25",
            },
            {
                "constraint_variable": "ucgid",
                "operation": "equals",
                "value": "123456",
            },
        ]
    )
    result = build_target_name("count", constraints)
    assert result == "count_ucgid=123456_age=25"


def test_validate_metrics_matrix():
    """Test validating metrics matrix with synthetic data."""
    from policyengine_data.calibration.metrics_matrix_creation import (
        validate_metrics_matrix,
    )

    # Create synthetic metrics matrix
    np.random.seed(42)  # For reproducible results
    n_households = 100
    n_targets = 5

    # Create metrics matrix with known values
    metrics_data = np.random.rand(n_households, n_targets) * 1000
    household_ids = range(1000, 1000 + n_households)
    target_ids = range(1, n_targets + 1)

    metrics_matrix = pd.DataFrame(
        data=metrics_data, index=household_ids, columns=target_ids
    )

    # Create target values
    target_values = np.array([5000, 3000, 8000, 1500, 6000])

    # Create target info
    target_info = {
        1: {"name": "target_1", "active": True, "tolerance": 0.1},
        2: {"name": "target_2", "active": True, "tolerance": 0.05},
        3: {"name": "target_3", "active": False, "tolerance": None},
        4: {"name": "target_4", "active": True, "tolerance": 0.2},
        5: {"name": "target_5", "active": True, "tolerance": 0.15},
    }

    # Test with uniform weights
    validation_results = validate_metrics_matrix(
        metrics_matrix, target_values, target_info=target_info
    )

    # Check structure
    assert len(validation_results) == n_targets
    assert set(validation_results.columns) == {
        "target_id",
        "target_value",
        "estimate",
        "absolute_error",
        "relative_error",
        "name",
        "active",
        "tolerance",
    }

    # Check values
    assert (
        validation_results["target_value"].tolist() == target_values.tolist()
    )
    assert validation_results["name"].tolist() == [
        "target_1",
        "target_2",
        "target_3",
        "target_4",
        "target_5",
    ]
    assert validation_results["active"].tolist() == [
        True,
        True,
        False,
        True,
        True,
    ]

    # Test with custom weights
    custom_weights = np.random.rand(n_households)
    custom_weights /= custom_weights.sum()  # Normalize

    validation_results_weighted = validate_metrics_matrix(
        metrics_matrix,
        target_values,
        weights=custom_weights,
        target_info=target_info,
    )

    # Should have same structure but different estimates
    assert len(validation_results_weighted) == n_targets
    # Estimates should be different with different weights
    assert not np.allclose(
        validation_results["estimate"].values,
        validation_results_weighted["estimate"].values,
    )

    # Test error raising with all-zero matrix
    zero_matrix = pd.DataFrame(
        data=np.zeros((n_households, n_targets)),
        index=household_ids,
        columns=target_ids,
    )

    with pytest.raises(ValueError, match="Record.*has all zero estimates"):
        validate_metrics_matrix(zero_matrix, target_values, raise_error=True)

    # Test with household that meets no constraints (all zeros in row)
    mixed_matrix = metrics_matrix.copy()
    mixed_matrix.iloc[0, :] = 0  # First household meets no constraints

    with pytest.raises(ValueError, match="Record.*has all zero estimates"):
        validate_metrics_matrix(mixed_matrix, target_values, raise_error=True)


def test_validate_metrics_matrix_zero_estimates():
    """Test validate_metrics_matrix with zero estimate columns."""
    from policyengine_data.calibration.metrics_matrix_creation import (
        validate_metrics_matrix,
    )

    # Create a matrix where one column will have zero estimates
    # but no individual records are all zero
    metrics_matrix = pd.DataFrame(
        data=[[1, 0], [2, 0], [3, 0]],  # Second column is all zeros
        index=[1, 2, 3],
        columns=[101, 102],
    )
    target_values = np.array([6, 1])  # Second target will have zero estimate

    with pytest.raises(ValueError, match="estimate.*contain zero values"):
        validate_metrics_matrix(
            metrics_matrix, target_values, raise_error=True
        )


def test_validate_metrics_matrix_without_target_info():
    """Test validate_metrics_matrix without target_info parameter."""
    from policyengine_data.calibration.metrics_matrix_creation import (
        validate_metrics_matrix,
    )

    # Simple test case
    metrics_matrix = pd.DataFrame(
        data=[[10, 20], [30, 40], [50, 60]],
        index=[1, 2, 3],
        columns=[101, 102],
    )
    target_values = np.array([90, 120])  # Sum of columns: [90, 120]

    validation_results = validate_metrics_matrix(metrics_matrix, target_values)

    # Check basic columns exist
    expected_cols = {
        "target_id",
        "target_value",
        "estimate",
        "absolute_error",
        "relative_error",
    }
    assert set(validation_results.columns) == expected_cols

    # Check estimates (uniform weights: 1/3 each)
    expected_estimates = np.array([30, 40])  # (10+30+50)/3, (20+40+60)/3
    np.testing.assert_array_almost_equal(
        validation_results["estimate"].values, expected_estimates
    )

    # Check errors
    expected_abs_errors = np.abs(expected_estimates - target_values)
    np.testing.assert_array_almost_equal(
        validation_results["absolute_error"].values, expected_abs_errors
    )
