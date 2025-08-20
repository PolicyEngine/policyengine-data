"""
Tests for key normalisation functionality.
"""

import pandas as pd
import pytest

from policyengine_data.normalise_keys import (
    _auto_detect_foreign_keys,
    normalise_single_table_keys,
    normalise_table_keys,
)


class TestNormaliseTableKeys:
    """Test cases for normalise_table_keys function."""

    def test_simple_single_table(self):
        """Test normalisation of a single table with no foreign keys."""
        persons = pd.DataFrame(
            {"person_id": [101, 105, 103], "name": ["Alice", "Bob", "Carol"]}
        )

        tables = {"persons": persons}
        primary_keys = {"persons": "person_id"}

        result = normalise_table_keys(tables, primary_keys)

        assert len(result) == 1
        assert "persons" in result

        normalised_persons = result["persons"]
        assert list(normalised_persons["person_id"]) == [0, 1, 2]
        assert list(normalised_persons["name"]) == ["Alice", "Bob", "Carol"]

    def test_custom_start_index(self):
        """Test normalisation with custom start index."""
        persons = pd.DataFrame(
            {"person_id": [101, 105, 103], "name": ["Alice", "Bob", "Carol"]}
        )

        households = pd.DataFrame(
            {
                "household_id": [201, 205, 207],
                "person_id": [105, 101, 105],
                "income": [25000, 15000, 42000],
            }
        )

        tables = {"persons": persons, "households": households}
        primary_keys = {"persons": "person_id", "households": "household_id"}
        foreign_keys = {"households": {"person_id": "persons"}}

        result = normalise_table_keys(
            tables,
            primary_keys,
            foreign_keys,
            start_index={"persons": 10, "households": 20},
        )

        assert len(result) == 2
        assert "persons" in result
        assert "households" in result

        normalised_persons = result["persons"]
        assert list(normalised_persons["person_id"]) == [10, 11, 12]
        assert list(normalised_persons["name"]) == ["Alice", "Bob", "Carol"]
        normalised_households = result["households"]
        assert list(normalised_households["household_id"]) == [20, 21, 22]

    def test_two_tables_with_foreign_keys(self):
        """Test normalisation with explicit foreign key relationships."""
        persons = pd.DataFrame(
            {"person_id": [101, 105, 103], "name": ["Alice", "Bob", "Carol"]}
        )

        households = pd.DataFrame(
            {
                "household_id": [201, 205, 207],
                "person_id": [105, 101, 105],
                "income": [25000, 15000, 42000],
            }
        )

        tables = {"persons": persons, "households": households}
        primary_keys = {"persons": "person_id", "households": "household_id"}
        foreign_keys = {"households": {"person_id": "persons"}}

        result = normalise_table_keys(tables, primary_keys, foreign_keys)

        # Check persons table
        normalised_persons = result["persons"]
        assert set(normalised_persons["person_id"]) == {0, 1, 2}

        # Check households table
        normalised_households = result["households"]
        assert set(normalised_households["household_id"]) == {0, 1, 2}

        # Check foreign key relationships are preserved
        # Original: person 105 had households 201, 207
        # After normalisation: find which index 105 became
        person_105_new_id = normalised_persons[
            normalised_persons["name"] == "Bob"
        ]["person_id"].iloc[0]
        bob_households = normalised_households[
            normalised_households["person_id"] == person_105_new_id
        ]
        assert len(bob_households) == 2
        assert set(bob_households["income"]) == {25000, 42000}

    def test_auto_detect_foreign_keys(self):
        """Test automatic detection of foreign key relationships."""
        persons = pd.DataFrame(
            {"person_id": [101, 105, 103], "name": ["Alice", "Bob", "Carol"]}
        )

        households = pd.DataFrame(
            {
                "household_id": [201, 205, 207],
                "person_id": [105, 101, 105],
                "income": [25000, 15000, 42000],
            }
        )

        tables = {"persons": persons, "households": households}
        primary_keys = {"persons": "person_id", "households": "household_id"}

        # Test without explicit foreign keys - should auto-detect
        result = normalise_table_keys(tables, primary_keys)

        # Verify relationships are still preserved
        normalised_persons = result["persons"]
        normalised_households = result["households"]

        # Bob should still have his two households
        person_105_new_id = normalised_persons[
            normalised_persons["name"] == "Bob"
        ]["person_id"].iloc[0]
        bob_households = normalised_households[
            normalised_households["person_id"] == person_105_new_id
        ]
        assert len(bob_households) == 2

    def test_multiple_foreign_keys(self):
        """Test table with multiple foreign key relationships."""
        persons = pd.DataFrame(
            {"person_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}
        )

        benefit_units = pd.DataFrame(
            {
                "benefit_unit_id": [10, 20, 30],
                "benefit_type": ["Disability", "Unemployment", "Family"],
            }
        )

        households = pd.DataFrame(
            {
                "household_id": [100, 200, 300],
                "person_id": [2, 1, 2],
                "benefit_unit_id": [20, 10, 30],
                "income": [25000, 15000, 42000],
            }
        )

        tables = {
            "persons": persons,
            "benefit_units": benefit_units,
            "households": households,
        }
        primary_keys = {
            "persons": "person_id",
            "benefit_units": "benefit_unit_id",
            "households": "household_id",
        }

        result = normalise_table_keys(tables, primary_keys)

        # Verify all tables have zero-based keys
        for table_name, df in result.items():
            pk_col = primary_keys[table_name]
            assert set(df[pk_col]) == {0, 1, 2}

        # Verify relationships preserved
        normalised_households = result["households"]
        normalised_persons = result["persons"]

        # Bob (original person_id=2) should have 2 households
        bob_new_id = normalised_persons[normalised_persons["name"] == "Bob"][
            "person_id"
        ].iloc[0]
        bob_households = normalised_households[
            normalised_households["person_id"] == bob_new_id
        ]
        assert len(bob_households) == 2

    def test_empty_tables(self):
        """Test with empty input."""
        result = normalise_table_keys({}, {})
        assert result == {}

    def test_missing_primary_key_column(self):
        """Test error handling for missing primary key column."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        tables = {"persons": df}
        primary_keys = {"persons": "missing_id"}

        with pytest.raises(
            ValueError, match="Primary key column 'missing_id' not found"
        ):
            normalise_table_keys(tables, primary_keys)

    def test_missing_foreign_key_column(self):
        """Test error handling for missing foreign key column."""
        persons = pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        households = pd.DataFrame(
            {"household_id": [100, 200], "income": [25000, 15000]}
        )

        tables = {"persons": persons, "households": households}
        primary_keys = {"persons": "person_id", "households": "household_id"}
        foreign_keys = {"households": {"missing_person_id": "persons"}}

        with pytest.raises(
            ValueError,
            match="Foreign key column 'missing_person_id' not found",
        ):
            normalise_table_keys(tables, primary_keys, foreign_keys)

    def test_missing_referenced_table(self):
        """Test error handling for missing referenced table."""
        households = pd.DataFrame(
            {
                "household_id": [100, 200],
                "person_id": [1, 2],
                "income": [25000, 15000],
            }
        )

        tables = {"households": households}
        primary_keys = {"households": "household_id"}
        foreign_keys = {"households": {"person_id": "missing_persons"}}

        with pytest.raises(
            ValueError, match="Referenced table 'missing_persons' not found"
        ):
            normalise_table_keys(tables, primary_keys, foreign_keys)


class TestNormaliseSingleTableKeys:
    """Test cases for normalise_single_table_keys function."""

    def test_basic_normalisation(self):
        """Test basic single table key normalisation."""
        df = pd.DataFrame({"id": [101, 105, 103], "value": ["A", "B", "C"]})

        result = normalise_single_table_keys(df, "id")

        assert list(result["id"]) == [0, 1, 2]
        assert list(result["value"]) == ["A", "B", "C"]

    def test_custom_start_index(self):
        """Test normalisation with custom start index."""
        df = pd.DataFrame({"id": [101, 105, 103], "value": ["A", "B", "C"]})

        result = normalise_single_table_keys(df, "id", start_index=10)

        assert list(result["id"]) == [10, 11, 12]
        assert list(result["value"]) == ["A", "B", "C"]

    def test_duplicate_keys_preserved(self):
        """Test that duplicate keys are handled correctly."""
        df = pd.DataFrame(
            {"id": [101, 105, 101, 103], "value": ["A", "B", "A2", "C"]}
        )

        result = normalise_single_table_keys(df, "id")

        # Should have 3 unique normalised values (0, 1, 2) for 3 unique original values
        unique_normalised = result["id"].unique()
        assert len(unique_normalised) == 3
        assert set(unique_normalised) == {0, 1, 2}

        # Duplicate original keys should map to same normalised key
        original_101_rows = df[df["id"] == 101]
        normalised_101_rows = result[
            result.index.isin(original_101_rows.index)
        ]
        assert len(normalised_101_rows["id"].unique()) == 1

    def test_missing_key_column(self):
        """Test error handling for missing key column."""
        df = pd.DataFrame({"value": ["A", "B", "C"]})

        with pytest.raises(
            ValueError, match="Key column 'missing_id' not found"
        ):
            normalise_single_table_keys(df, "missing_id")


class TestAutoDetectForeignKeys:
    """Test cases for _auto_detect_foreign_keys function."""

    def test_simple_detection(self):
        """Test basic foreign key detection."""
        persons = pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        households = pd.DataFrame(
            {"household_id": [100, 200], "person_id": [1, 2]}
        )

        tables = {"persons": persons, "households": households}
        primary_keys = {"persons": "person_id", "households": "household_id"}

        result = _auto_detect_foreign_keys(tables, primary_keys)

        expected = {"households": {"person_id": "persons"}}
        assert result == expected

    def test_no_foreign_keys(self):
        """Test when no foreign keys are detected."""
        persons = pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        benefit_units = pd.DataFrame(
            {"benefit_unit_id": [100, 200], "name": ["Disability", "Family"]}
        )

        tables = {"persons": persons, "benefit_units": benefit_units}
        primary_keys = {
            "persons": "person_id",
            "benefit_units": "benefit_unit_id",
        }

        result = _auto_detect_foreign_keys(tables, primary_keys)

        assert result == {}

    def test_multiple_foreign_keys_detection(self):
        """Test detection of multiple foreign keys in one table."""
        persons = pd.DataFrame({"person_id": [1, 2], "name": ["Alice", "Bob"]})
        benefit_units = pd.DataFrame(
            {"benefit_unit_id": [10, 20], "name": ["Disability", "Family"]}
        )
        households = pd.DataFrame(
            {
                "household_id": [100, 200],
                "person_id": [1, 2],
                "benefit_unit_id": [10, 20],
            }
        )

        tables = {
            "persons": persons,
            "benefit_units": benefit_units,
            "households": households,
        }
        primary_keys = {
            "persons": "person_id",
            "benefit_units": "benefit_unit_id",
            "households": "household_id",
        }

        result = _auto_detect_foreign_keys(tables, primary_keys)

        expected = {
            "households": {
                "person_id": "persons",
                "benefit_unit_id": "benefit_units",
            }
        }
        assert result == expected
