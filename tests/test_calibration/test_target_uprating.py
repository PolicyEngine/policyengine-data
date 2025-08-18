"""
Test the logic for uprating calibration targets from one period to another.
"""

import os
import tempfile
import pandas as pd
import pytest
from unittest.mock import Mock, patch
from sqlalchemy import create_engine

from policyengine_data.calibration.target_uprating import (
    uprate_calibration_targets,
    get_uprating_factors,
    calculate_uprating_factor,
    uprate_targets_for_period,
    fetch_targets,
    prepare_insert_data,
    insert_uprated_targets_in_db,
)


def setup_test_database() -> str:
    """
    Creates an in-memory SQLite database for testing.
    Populates it with test targets for multiple years and variables.
    """
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(db_fd)
    db_uri = f"sqlite:///{db_path}"
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
        conn.exec_driver_sql("DROP TABLE IF EXISTS targets")
        conn.exec_driver_sql("DROP TABLE IF EXISTS strata")
        conn.exec_driver_sql(strata_schema)
        conn.exec_driver_sql(targets_schema)

    # Create test strata data
    strata_data = pd.DataFrame(
        [
            {
                "stratum_id": 1,
                "stratum_group_id": 10,
                "parent_stratum_id": None,
                "notes": "Total Population",
            },
            {
                "stratum_id": 2,
                "stratum_group_id": 11,
                "parent_stratum_id": 1,
                "notes": "Income < 50k",
            },
            {
                "stratum_id": 3,
                "stratum_group_id": 11,
                "parent_stratum_id": 1,
                "notes": "Income >= 50k",
            },
            {
                "stratum_id": 4,
                "stratum_group_id": 20,
                "parent_stratum_id": None,
                "notes": "All States",
            },
        ]
    )

    # Create test targets data for multiple years
    targets_data = pd.DataFrame(
        [
            # 2021 data (baseline year)
            {
                "target_id": 101,
                "stratum_id": 1,
                "variable": "income_tax",
                "period": 2021,
                "reform_id": 0,
                "value": 1000000.0,
                "active": True,
                "tolerance": 0.01,
            },
            {
                "target_id": 102,
                "stratum_id": 2,
                "variable": "income_tax",
                "period": 2021,
                "reform_id": 0,
                "value": 300000.0,
                "active": True,
                "tolerance": 0.05,
            },
            {
                "target_id": 103,
                "stratum_id": 3,
                "variable": "income_tax",
                "period": 2021,
                "reform_id": 0,
                "value": 700000.0,
                "active": True,
                "tolerance": 0.05,
            },
            # 2021 SNAP data
            {
                "target_id": 104,
                "stratum_id": 1,
                "variable": "snap",
                "period": 2021,
                "reform_id": 0,
                "value": 80000.0,
                "active": True,
                "tolerance": 0.02,
            },
            {
                "target_id": 105,
                "stratum_id": 2,
                "variable": "snap",
                "period": 2021,
                "reform_id": 0,
                "value": 60000.0,
                "active": True,
                "tolerance": 0.05,
            },
            {
                "target_id": 106,
                "stratum_id": 3,
                "variable": "snap",
                "period": 2021,
                "reform_id": 0,
                "value": 20000.0,
                "active": True,
                "tolerance": 0.05,
            },
            # 2021 Count data (should use population factor)
            {
                "target_id": 110,
                "stratum_id": 1,
                "variable": "household_count",
                "period": 2021,
                "reform_id": 0,
                "value": 130000.0,
                "active": True,
                "tolerance": 0.02,
            },
            {
                "target_id": 111,
                "stratum_id": 2,
                "variable": "person_count",
                "period": 2021,
                "reform_id": 0,
                "value": 320000.0,
                "active": True,
                "tolerance": 0.03,
            },
            # 2022 data (for testing different periods)
            {
                "target_id": 107,
                "stratum_id": 1,
                "variable": "income_tax",
                "period": 2022,
                "reform_id": 0,
                "value": 1050000.0,
                "active": True,
                "tolerance": 0.01,
            },
            # Inactive target (should be ignored)
            {
                "target_id": 108,
                "stratum_id": 4,
                "variable": "income_tax",
                "period": 2021,
                "reform_id": 0,
                "value": 999999.0,
                "active": False,
                "tolerance": 0.01,
            },
            # Different reform_id (should be isolated)
            {
                "target_id": 109,
                "stratum_id": 1,
                "variable": "income_tax",
                "period": 2021,
                "reform_id": 1,
                "value": 1100000.0,
                "active": True,
                "tolerance": 0.01,
            },
        ]
    )

    # Load data into the database
    strata_data.to_sql("strata", engine, if_exists="append", index=False)
    targets_data.to_sql("targets", engine, if_exists="append", index=False)

    return db_uri, db_path


class TestUpratingFactors:
    """Test the uprating factors calculation."""

    def test_get_uprating_factors_structure(self):
        """Test that get_uprating_factors returns the expected structure."""
        mock_system = Mock()
        # Mock the system parameters
        mock_pop = Mock()
        mock_cpi = Mock()
        mock_system.parameters.get_child.side_effect = lambda path: (
            mock_pop if "population" in path else mock_cpi
        )
        mock_pop.side_effect = (
            lambda year: 100000 + (year - 2023) * 1000
        )  # Growing population
        mock_cpi.side_effect = (
            lambda year: 250 + (year - 2023) * 10
        )  # Growing inflation

        factors_df = get_uprating_factors(
            mock_system, current_year=2023, start_year=2021, end_year=2025
        )

        assert isinstance(factors_df, pd.DataFrame)
        assert list(factors_df.columns) == [
            "Year",
            "Population_factor",
            "Inflation_factor",
        ]
        assert len(factors_df) == 5  # 2021-2025
        assert (
            factors_df[factors_df["Year"] == 2023]["Population_factor"].iloc[0]
            == 1.0
        )  # Base year
        assert (
            factors_df[factors_df["Year"] == 2023]["Inflation_factor"].iloc[0]
            == 1.0
        )  # Base year

    def test_calculate_uprating_factor(self):
        """Test uprating factor calculation between years."""
        # Create test uprating factors DataFrame
        uprating_df = pd.DataFrame(
            {
                "Year": [2021, 2022, 2023, 2024],
                "Population_factor": [0.95, 0.97, 1.00, 1.03],
                "Inflation_factor": [0.90, 0.95, 1.00, 1.05],
            }
        )

        # Test inflation uprating from 2021 to 2024
        inflation_factor = calculate_uprating_factor(
            uprating_df, 2021, 2024, "inflation"
        )
        expected = 1.05 / 0.90  # ≈ 1.1667
        assert inflation_factor == pytest.approx(expected, rel=1e-4)

        # Test population uprating from 2022 to 2023
        pop_factor = calculate_uprating_factor(
            uprating_df, 2022, 2023, "population"
        )
        expected = 1.00 / 0.97  # ≈ 1.0309
        assert pop_factor == pytest.approx(expected, rel=1e-4)

        # Test missing year (should return 1.0)
        missing_factor = calculate_uprating_factor(
            uprating_df, 2025, 2026, "inflation"
        )
        assert missing_factor == 1.0


class TestTargetUprating:
    """Test the target uprating functionality."""

    def setup_method(self):
        """Set up test database for each test."""
        self.db_uri, self.db_path = setup_test_database()

    def teardown_method(self):
        """Clean up test database after each test."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_fetch_targets(self):
        """Test fetching targets from database."""
        engine = create_engine(self.db_uri)

        # Fetch 2021 baseline targets
        targets_df = fetch_targets(engine, period=2021, reform_id=0)

        assert not targets_df.empty
        assert (
            len(targets_df) == 8
        )  # 3 income_tax + 3 snap + 2 count targets for 2021, reform_id=0
        assert all(targets_df["period"] == 2021)
        assert all(targets_df["reform_id"] == 0)
        assert all(targets_df["active"] == True)

        # Test variable filtering
        income_targets = targets_df[targets_df["variable"] == "income_tax"]
        assert len(income_targets) == 3

        # Test no results for non-existent period
        empty_df = fetch_targets(engine, period=2030, reform_id=0)
        assert empty_df.empty

    def test_uprate_targets_for_period(self):
        """Test uprating targets from one period to another."""
        # Create mock uprating factors
        uprating_df = pd.DataFrame(
            {
                "Year": [2021, 2022, 2023, 2024],
                "Population_factor": [0.95, 0.97, 1.00, 1.03],
                "Inflation_factor": [0.90, 0.95, 1.00, 1.05],
            }
        )

        # Create sample targets
        targets_df = pd.DataFrame(
            [
                {
                    "target_id": 1,
                    "stratum_id": 1,
                    "variable": "income_tax",
                    "period": 2021,
                    "reform_id": 0,
                    "value": 1000.0,
                    "tolerance": 0.01,
                },
                {
                    "target_id": 2,
                    "stratum_id": 2,
                    "variable": "income_tax",
                    "period": 2021,
                    "reform_id": 0,
                    "value": 500.0,
                    "tolerance": 0.05,
                },
            ]
        )

        # Uprate from 2021 to 2024 using inflation
        uprated_df = uprate_targets_for_period(
            targets_df,
            uprating_df,
            from_period=2021,
            to_period=2024,
            factor_type="inflation",
        )

        expected_factor = 1.05 / 0.90  # ≈ 1.1667

        assert len(uprated_df) == 2
        for _, row in uprated_df.iterrows():
            assert row["uprating_factor"] == pytest.approx(
                expected_factor, rel=1e-3
            )
        assert all(uprated_df["original_period"] == 2021)
        assert all(uprated_df["uprated_period"] == 2024)
        assert all(uprated_df["period"] == 2024)
        assert all(uprated_df["factor_type"] == "inflation")

        # Check uprated values
        assert uprated_df.iloc[0]["uprated_value"] == pytest.approx(
            1000.0 * expected_factor, rel=1e-3
        )
        assert uprated_df.iloc[1]["uprated_value"] == pytest.approx(
            500.0 * expected_factor, rel=1e-3
        )

    @patch(
        "policyengine_data.calibration.target_uprating.get_uprating_factors"
    )
    def test_uprate_calibration_targets_basic(self, mock_get_factors):
        """Test the main uprating function with basic functionality."""
        # Mock uprating factors
        mock_get_factors.return_value = pd.DataFrame(
            {
                "Year": [2021, 2024],
                "Population_factor": [0.95, 1.03],
                "Inflation_factor": [0.90, 1.05],
            }
        )
        mock_system = Mock()

        # Run uprating for income_tax (should use inflation factor)
        results_df = uprate_calibration_targets(
            system=mock_system,
            db_uri=self.db_uri,
            from_period=2021,
            to_period=2024,
            variable="income_tax",
            reform_id=0,
            update_database=False,
        )

        assert not results_df.empty
        assert len(results_df) == 3  # 3 income_tax targets for 2021
        assert all(results_df["original_period"] == 2021)
        assert all(results_df["uprated_period"] == 2024)
        assert all(results_df["factor_type"] == "inflation")

        # Check that uprating factor was calculated correctly
        expected_factor = 1.05 / 0.90
        for _, row in results_df.iterrows():
            assert row["uprating_factor"] == pytest.approx(
                expected_factor, rel=1e-3
            )

    @patch(
        "policyengine_data.calibration.target_uprating.get_uprating_factors"
    )
    def test_uprate_calibration_targets_automatic_factor_selection(
        self, mock_get_factors
    ):
        """Test automatic factor selection based on variable name."""
        mock_get_factors.return_value = pd.DataFrame(
            {
                "Year": [2021, 2024],
                "Population_factor": [0.95, 1.03],
                "Inflation_factor": [0.90, 1.05],
            }
        )
        mock_system = Mock()

        # Test count variables (should use population factor)
        results_df = uprate_calibration_targets(
            system=mock_system,
            db_uri=self.db_uri,
            from_period=2021,
            to_period=2024,
            variable="household_count",
            reform_id=0,
            update_database=False,
        )

        assert not results_df.empty
        assert len(results_df) == 1  # 1 household_count target for 2021
        assert all(results_df["factor_type"] == "population")

        # Check that population factor was used
        expected_pop_factor = 1.03 / 0.95
        for _, row in results_df.iterrows():
            assert row["uprating_factor"] == pytest.approx(
                expected_pop_factor, rel=1e-3
            )

        # Test non-count variables (should use inflation factor)
        results_df_snap = uprate_calibration_targets(
            system=mock_system,
            db_uri=self.db_uri,
            from_period=2021,
            to_period=2024,
            variable="snap",
            reform_id=0,
            update_database=False,
        )

        assert not results_df_snap.empty
        assert len(results_df_snap) == 3  # 3 SNAP targets for 2021
        assert all(results_df_snap["factor_type"] == "inflation")

        # Check that inflation factor was used
        expected_inflation_factor = 1.05 / 0.90
        for _, row in results_df_snap.iterrows():
            assert row["uprating_factor"] == pytest.approx(
                expected_inflation_factor, rel=1e-3
            )

    @patch(
        "policyengine_data.calibration.target_uprating.get_uprating_factors"
    )
    def test_uprate_calibration_targets_mixed_factor_types(
        self, mock_get_factors
    ):
        """Test uprating all variables with mixed factor types."""
        mock_get_factors.return_value = pd.DataFrame(
            {
                "Year": [2021, 2022],
                "Population_factor": [0.95, 0.97],
                "Inflation_factor": [0.90, 0.95],
            }
        )
        mock_system = Mock()

        results_df = uprate_calibration_targets(
            system=mock_system,
            db_uri=self.db_uri,
            from_period=2021,
            to_period=2022,
            variable=None,  # All variables
            reform_id=0,
            update_database=False,
        )

        assert not results_df.empty
        assert len(results_df) == 8  # 3 income_tax + 3 snap + 2 count targets

        # Check all variables are present
        variables = results_df["variable"].unique()
        assert set(variables) == {
            "income_tax",
            "snap",
            "household_count",
            "person_count",
        }

        # Check factor types are correctly assigned
        count_variables = results_df[
            results_df["variable"].str.contains("_count")
        ]
        non_count_variables = results_df[
            ~results_df["variable"].str.contains("_count")
        ]

        assert all(count_variables["factor_type"] == "population")
        assert all(non_count_variables["factor_type"] == "inflation")

        # Verify different uprating factors were applied
        pop_factor = 0.97 / 0.95
        inflation_factor = 0.95 / 0.90

        # Check each count variable has the population factor
        for _, row in count_variables.iterrows():
            assert row["uprating_factor"] == pytest.approx(
                pop_factor, rel=1e-3
            )

        # Check each non-count variable has the inflation factor
        for _, row in non_count_variables.iterrows():
            assert row["uprating_factor"] == pytest.approx(
                inflation_factor, rel=1e-3
            )

    def test_uprate_calibration_targets_no_data(self):
        """Test uprating when no targets exist for the specified period."""
        with patch(
            "policyengine_data.calibration.target_uprating.get_uprating_factors"
        ) as mock_get_factors:
            mock_get_factors.return_value = pd.DataFrame(
                {
                    "Year": [2030, 2031],
                    "Population_factor": [1.0, 1.1],
                    "Inflation_factor": [1.0, 1.05],
                }
            )

            mock_system = Mock()
            results_df = uprate_calibration_targets(
                system=mock_system,
                db_uri=self.db_uri,
                from_period=2030,  # Non-existent period
                to_period=2031,
                update_database=False,
            )

            assert results_df.empty

    def test_prepare_insert_data(self):
        """Test preparation of data for database insertion."""
        uprated_df = pd.DataFrame(
            [
                {
                    "stratum_id": 1,
                    "variable": "income_tax",
                    "uprated_period": 2024,
                    "reform_id": 0,
                    "uprated_value": 1200.0,
                    "tolerance": 0.01,
                },
                {
                    "stratum_id": 2,
                    "variable": "income_tax",
                    "uprated_period": 2024,
                    "reform_id": 0,
                    "uprated_value": 600.0,
                    "tolerance": 0.05,
                },
            ]
        )

        inserts = prepare_insert_data(uprated_df)

        assert len(inserts) == 2
        assert inserts[0]["stratum_id"] == 1
        assert inserts[0]["variable"] == "income_tax"
        assert inserts[0]["period"] == 2024
        assert inserts[0]["value"] == 1200.0
        assert inserts[0]["active"] == True

    def test_insert_uprated_targets_in_db(self):
        """Test inserting uprated targets into database."""
        engine = create_engine(self.db_uri)

        # Prepare test data
        inserts = [
            {
                "stratum_id": 1,
                "variable": "new_variable",
                "period": 2025,
                "reform_id": 0,
                "value": 5000.0,
                "active": True,
                "tolerance": 0.01,
            }
        ]

        # Insert data
        inserted_count = insert_uprated_targets_in_db(engine, inserts)
        assert inserted_count == 1

        # Verify insertion
        result_df = pd.read_sql(
            "SELECT * FROM targets WHERE variable = 'new_variable' AND period = 2025",
            engine,
        )
        assert len(result_df) == 1
        assert result_df.iloc[0]["value"] == 5000.0

    def test_insert_uprated_targets_update_existing(self):
        """Test updating existing targets during insertion."""
        engine = create_engine(self.db_uri)

        # Update existing target (stratum_id=1, income_tax, 2021, reform_id=0)
        inserts = [
            {
                "stratum_id": 1,
                "variable": "income_tax",
                "period": 2021,
                "reform_id": 0,
                "value": 9999999.0,
                "active": True,
                "tolerance": 0.001,
            }
        ]

        # This should update the existing record
        updated_count = insert_uprated_targets_in_db(engine, inserts)
        assert updated_count == 1

        # Verify update
        result_df = pd.read_sql(
            "SELECT * FROM targets WHERE stratum_id = 1 AND variable = 'income_tax' AND period = 2021 AND reform_id = 0",
            engine,
        )
        assert len(result_df) == 1
        assert result_df.iloc[0]["value"] == 9999999.0
        assert result_df.iloc[0]["tolerance"] == 0.001


class TestIntegration:
    """Integration tests for the complete uprating workflow."""

    def setup_method(self):
        """Set up test database for integration tests."""
        self.db_uri, self.db_path = setup_test_database()

    def teardown_method(self):
        """Clean up test database after integration tests."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    @patch(
        "policyengine_data.calibration.target_uprating.get_uprating_factors"
    )
    def test_complete_uprating_workflow_with_database_update(
        self, mock_get_factors
    ):
        """Test complete workflow including database update."""
        mock_get_factors.return_value = pd.DataFrame(
            {
                "Year": [2021, 2025],
                "Population_factor": [0.9, 1.1],
                "Inflation_factor": [0.85, 1.15],
            }
        )
        mock_system = Mock()

        # Run complete uprating with database update
        results_df = uprate_calibration_targets(
            system=mock_system,
            db_uri=self.db_uri,
            from_period=2021,
            to_period=2025,
            variable="income_tax",
            update_database=True,  # This will insert new records
        )

        # Verify results
        assert not results_df.empty
        assert len(results_df) == 3

        # Verify database was updated
        engine = create_engine(self.db_uri)
        new_targets = pd.read_sql(
            "SELECT * FROM targets WHERE period = 2025 AND variable = 'income_tax'",
            engine,
        )
        assert len(new_targets) == 3

        # Check that values were correctly uprated
        expected_factor = 1.15 / 0.85
        original_total = 1000000.0  # Original total value for stratum_id=1
        expected_uprated = original_total * expected_factor

        uprated_total = new_targets[new_targets["stratum_id"] == 1][
            "value"
        ].iloc[0]
        assert uprated_total == pytest.approx(expected_uprated, rel=1e-4)

    @patch(
        "policyengine_data.calibration.target_uprating.get_uprating_factors"
    )
    def test_different_reform_scenarios_isolated(self, mock_get_factors):
        """Test that different reform scenarios are processed independently."""
        mock_get_factors.return_value = pd.DataFrame(
            {
                "Year": [2021, 2024],
                "Population_factor": [1.0, 1.05],
                "Inflation_factor": [1.0, 1.10],
            }
        )
        mock_system = Mock()

        # Uprate baseline scenario (reform_id=0)
        baseline_results = uprate_calibration_targets(
            system=mock_system,
            db_uri=self.db_uri,
            from_period=2021,
            to_period=2024,
            reform_id=0,
            update_database=False,
        )

        # Uprate reform scenario (reform_id=1)
        reform_results = uprate_calibration_targets(
            system=mock_system,
            db_uri=self.db_uri,
            from_period=2021,
            to_period=2024,
            reform_id=1,
            update_database=False,
        )

        # Baseline should have more targets (8 vs 1)
        assert len(baseline_results) > len(reform_results)
        assert len(reform_results) == 1  # Only one target with reform_id=1

        # But uprating factors should be the same
        if not reform_results.empty:
            baseline_factor = baseline_results["uprating_factor"].iloc[0]
            reform_factor = reform_results["uprating_factor"].iloc[0]
            assert baseline_factor == pytest.approx(reform_factor)


def test_error_handling():
    """Test error handling in various scenarios."""
    # Test with invalid database URI
    with pytest.raises(Exception):
        mock_system = Mock()
        uprate_calibration_targets(
            system=mock_system,
            db_uri="invalid://database/path",
            from_period=2021,
            to_period=2024,
        )
