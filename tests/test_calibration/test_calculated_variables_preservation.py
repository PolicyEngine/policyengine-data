"""
Test that calculated variables are properly identified and preserved during dataset minimization.
"""

import numpy as np
import pandas as pd
from policyengine_us import Microsimulation
from policyengine_data.calibration.dataset_duplication import (
    identify_calculated_variables,
    minimize_calibrated_dataset_legacy,
)


class TestCalculatedVariablesPreservation:
    """Test suite for verifying calculated variables are preserved during dataset operations."""

    def test_identify_calculated_variables_cps(self):
        """Test that identify_calculated_variables correctly identifies calculated vars in CPS."""
        # Identify calculated variables in CPS dataset
        calculated_vars = identify_calculated_variables(
            "hf://policyengine/policyengine-us-data/cps_2023.h5",
            Microsimulation,
        )

        # CPS should have these calculated variables
        assert "person" in calculated_vars
        assert "employment_income" in calculated_vars["person"]
        assert "self_employment_income" in calculated_vars["person"]
        assert "weekly_hours_worked" in calculated_vars["person"]

        # Should have exactly 3 person-level calculated variables
        assert len(calculated_vars.get("person", [])) == 3

    def test_minimize_preserves_calculated_variables(self):
        """Test that minimize_calibrated_dataset_legacy preserves calculated variables and their values."""
        # Load CPS dataset
        sim = Microsimulation(
            dataset="hf://policyengine/policyengine-us-data/cps_2023.h5"
        )
        sim.default_period = 2023

        # Get original values for calculated variables
        orig_employment_income = sim.calculate(
            "employment_income", 2023
        ).values
        orig_self_employment_income = sim.calculate(
            "self_employment_income", 2023
        ).values
        orig_weekly_hours = sim.calculate("weekly_hours_worked", 2023).values

        # Store original statistics
        orig_emp_sum = orig_employment_income.sum()
        orig_emp_nonzero_count = (orig_employment_income > 0).sum()
        orig_self_emp_sum = orig_self_employment_income.sum()
        orig_self_emp_nonzero_count = (orig_self_employment_income > 0).sum()
        orig_hours_sum = orig_weekly_hours.sum()
        orig_hours_nonzero_count = (orig_weekly_hours > 0).sum()

        # Verify we have non-zero values to start with
        assert (
            orig_emp_sum > 0
        ), "Original employment income should have non-zero values"
        assert (
            orig_emp_nonzero_count > 0
        ), "Should have people with employment income"
        assert (
            orig_self_emp_sum > 0
        ), "Original self-employment income should have non-zero values"
        assert (
            orig_self_emp_nonzero_count > 0
        ), "Should have people with self-employment income"

        # Create a subset with some households (use original weights for subset)
        household_ids = sim.calculate("household_id", 2023).values
        unique_hh_ids = np.unique(household_ids)[
            :100
        ]  # Take first 100 households
        orig_weights = sim.calculate("household_weight", 2023).values

        # Create subset weights
        subset_weights = np.zeros_like(orig_weights)
        for hh_id in unique_hh_ids:
            mask = household_ids == hh_id
            subset_weights[mask] = orig_weights[mask]

        # Identify calculated variables
        calculated_vars = identify_calculated_variables(
            "hf://policyengine/policyengine-us-data/cps_2023.h5",
            Microsimulation,
        )

        # Minimize the dataset preserving calculated variables
        minimized_dataset = minimize_calibrated_dataset_legacy(
            Microsimulation,
            sim,
            2023,
            pd.Series(subset_weights),
            include_all_variables=False,
            important_variables=calculated_vars,
        )

        # Verify the minimized dataset has the right structure
        assert "person" in minimized_dataset.entities
        person_df = minimized_dataset.entities["person"]

        # Check that calculated variables are present
        assert (
            "employment_income" in person_df.columns
        ), "employment_income should be preserved"
        assert (
            "self_employment_income" in person_df.columns
        ), "self_employment_income should be preserved"
        assert (
            "weekly_hours_worked" in person_df.columns
        ), "weekly_hours_worked should be preserved"

        # Check that values are not all zero
        min_emp_income = person_df["employment_income"].values
        min_self_emp_income = person_df["self_employment_income"].values
        min_weekly_hours = person_df["weekly_hours_worked"].values

        assert (
            min_emp_income.sum() > 0
        ), "Minimized employment income should not be all zeros"
        assert (
            min_emp_income > 0
        ).sum() > 0, "Should have some non-zero employment income"

        assert (
            min_self_emp_income.sum() > 0
        ), "Minimized self-employment income should not be all zeros"
        assert (
            min_self_emp_income > 0
        ).sum() > 0, "Should have some non-zero self-employment income"

        assert (
            min_weekly_hours.sum() > 0
        ), "Minimized weekly hours should not be all zeros"
        assert (
            min_weekly_hours > 0
        ).sum() > 0, "Should have some non-zero weekly hours"

        # Verify input variables are also present
        assert (
            "person_id" in person_df.columns
        ), "ID variables should be preserved"
        assert (
            "age" in person_df.columns
        ), "Input variables like age should be preserved"

    def test_minimize_with_all_variables(self):
        """Test that minimize_calibrated_dataset_legacy works with include_all_variables=True."""
        # Load CPS dataset
        sim = Microsimulation(
            dataset="hf://policyengine/policyengine-us-data/cps_2023.h5"
        )
        sim.default_period = 2023

        # Create a small subset for speed
        household_ids = sim.calculate("household_id", 2023).values
        unique_hh_ids = np.unique(household_ids)[
            :20
        ]  # Just 20 households for all variables test
        orig_weights = sim.calculate("household_weight", 2023).values

        subset_weights = np.zeros_like(orig_weights)
        for hh_id in unique_hh_ids:
            mask = household_ids == hh_id
            subset_weights[mask] = orig_weights[mask]

        # Minimize with ALL variables
        minimized_dataset = minimize_calibrated_dataset_legacy(
            Microsimulation,
            sim,
            2023,
            pd.Series(subset_weights),
            include_all_variables=True,  # Include everything
            important_variables=None,
        )

        # Should have many more variables
        person_df = minimized_dataset.entities["person"]
        assert (
            len(person_df.columns) > 100
        ), "Should have many variables when include_all=True"

        # Key calculated variables should still be there and non-zero
        assert "employment_income" in person_df.columns
        assert (
            person_df["employment_income"].sum() > 0 or len(person_df) == 0
        )  # Allow for empty subset

    def test_calculated_variables_consistency(self):
        """Test that calculated variable sums are non-zero after minimization."""
        # Load CPS dataset
        sim = Microsimulation(
            dataset="hf://policyengine/policyengine-us-data/cps_2023.h5"
        )
        sim.default_period = 2023

        # Get original calculated values
        orig_employment_income = sim.calculate(
            "employment_income", 2023
        ).values
        orig_self_employment_income = sim.calculate(
            "self_employment_income", 2023
        ).values
        orig_weekly_hours = sim.calculate("weekly_hours_worked", 2023).values

        # Verify we have data diversity to start with
        assert (
            orig_employment_income.sum() > 0
        ), "Original employment income should have non-zero sum"
        assert (
            orig_self_employment_income.sum() > 0
        ), "Original self-employment income should have non-zero sum"
        assert (
            orig_weekly_hours.sum() > 0
        ), "Original weekly hours should have non-zero sum"

        # Create a subset with some households
        household_ids = sim.calculate("household_id", 2023).values
        unique_hh_ids = np.unique(household_ids)[
            :100
        ]  # Take first 100 households
        orig_weights = sim.calculate("household_weight", 2023).values

        # Create subset weights
        subset_weights = np.zeros_like(orig_weights)
        for hh_id in unique_hh_ids:
            mask = household_ids == hh_id
            subset_weights[mask] = orig_weights[mask]

        # Identify calculated variables
        calculated_vars = identify_calculated_variables(
            "hf://policyengine/policyengine-us-data/cps_2023.h5",
            Microsimulation,
        )

        # Minimize the dataset
        minimized_dataset = minimize_calibrated_dataset_legacy(
            Microsimulation,
            sim,
            2023,
            pd.Series(subset_weights),
            include_all_variables=False,
            important_variables=calculated_vars,
        )

        # Check that calculated variables have non-zero sums (data diversity)
        person_df = minimized_dataset.entities["person"]

        assert (
            "employment_income" in person_df.columns
        ), "employment_income should be preserved"
        assert (
            "self_employment_income" in person_df.columns
        ), "self_employment_income should be preserved"
        assert (
            "weekly_hours_worked" in person_df.columns
        ), "weekly_hours_worked should be preserved"

        # Check for data diversity - sums should not be zero
        min_emp_income_sum = person_df["employment_income"].sum()
        min_self_emp_sum = person_df["self_employment_income"].sum()
        min_hours_sum = person_df["weekly_hours_worked"].sum()

        assert (
            min_emp_income_sum > 0
        ), "Minimized employment income sum should not be zero - ensuring data diversity"
        assert (
            min_self_emp_sum > 0
        ), "Minimized self-employment income sum should not be zero - ensuring data diversity"
        assert (
            min_hours_sum > 0
        ), "Minimized weekly hours sum should not be zero - ensuring data diversity"
