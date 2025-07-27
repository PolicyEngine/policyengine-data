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
        users = pd.DataFrame(
            {"user_id": [101, 105, 103], "name": ["Alice", "Bob", "Carol"]}
        )

        tables = {"users": users}
        primary_keys = {"users": "user_id"}

        result = normalise_table_keys(tables, primary_keys)

        assert len(result) == 1
        assert "users" in result

        normalised_users = result["users"]
        assert list(normalised_users["user_id"]) == [0, 1, 2]
        assert list(normalised_users["name"]) == ["Alice", "Bob", "Carol"]

    def test_two_tables_with_foreign_keys(self):
        """Test normalisation with explicit foreign key relationships."""
        users = pd.DataFrame(
            {"user_id": [101, 105, 103], "name": ["Alice", "Bob", "Carol"]}
        )

        orders = pd.DataFrame(
            {
                "order_id": [201, 205, 207],
                "user_id": [105, 101, 105],
                "amount": [25.99, 15.50, 42.00],
            }
        )

        tables = {"users": users, "orders": orders}
        primary_keys = {"users": "user_id", "orders": "order_id"}
        foreign_keys = {"orders": {"user_id": "users"}}

        result = normalise_table_keys(tables, primary_keys, foreign_keys)

        # Check users table
        normalised_users = result["users"]
        assert set(normalised_users["user_id"]) == {0, 1, 2}

        # Check orders table
        normalised_orders = result["orders"]
        assert set(normalised_orders["order_id"]) == {0, 1, 2}

        # Check foreign key relationships are preserved
        # Original: user 105 had orders 201, 207
        # After normalisation: find which index 105 became
        user_105_new_id = normalised_users[normalised_users["name"] == "Bob"][
            "user_id"
        ].iloc[0]
        bob_orders = normalised_orders[
            normalised_orders["user_id"] == user_105_new_id
        ]
        assert len(bob_orders) == 2
        assert set(bob_orders["amount"]) == {25.99, 42.00}

    def test_auto_detect_foreign_keys(self):
        """Test automatic detection of foreign key relationships."""
        users = pd.DataFrame(
            {"user_id": [101, 105, 103], "name": ["Alice", "Bob", "Carol"]}
        )

        orders = pd.DataFrame(
            {
                "order_id": [201, 205, 207],
                "user_id": [105, 101, 105],
                "amount": [25.99, 15.50, 42.00],
            }
        )

        tables = {"users": users, "orders": orders}
        primary_keys = {"users": "user_id", "orders": "order_id"}

        # Test without explicit foreign keys - should auto-detect
        result = normalise_table_keys(tables, primary_keys)

        # Verify relationships are still preserved
        normalised_users = result["users"]
        normalised_orders = result["orders"]

        # Bob should still have his two orders
        user_105_new_id = normalised_users[normalised_users["name"] == "Bob"][
            "user_id"
        ].iloc[0]
        bob_orders = normalised_orders[
            normalised_orders["user_id"] == user_105_new_id
        ]
        assert len(bob_orders) == 2

    def test_multiple_foreign_keys(self):
        """Test table with multiple foreign key relationships."""
        users = pd.DataFrame(
            {"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}
        )

        categories = pd.DataFrame(
            {
                "category_id": [10, 20, 30],
                "category_name": ["Electronics", "Books", "Clothing"],
            }
        )

        orders = pd.DataFrame(
            {
                "order_id": [100, 200, 300],
                "user_id": [2, 1, 2],
                "category_id": [20, 10, 30],
                "amount": [25.99, 15.50, 42.00],
            }
        )

        tables = {"users": users, "categories": categories, "orders": orders}
        primary_keys = {
            "users": "user_id",
            "categories": "category_id",
            "orders": "order_id",
        }

        result = normalise_table_keys(tables, primary_keys)

        # Verify all tables have zero-based keys
        for table_name, df in result.items():
            pk_col = primary_keys[table_name]
            assert set(df[pk_col]) == {0, 1, 2}

        # Verify relationships preserved
        normalised_orders = result["orders"]
        normalised_users = result["users"]

        # Bob (original user_id=2) should have 2 orders
        bob_new_id = normalised_users[normalised_users["name"] == "Bob"][
            "user_id"
        ].iloc[0]
        bob_orders = normalised_orders[
            normalised_orders["user_id"] == bob_new_id
        ]
        assert len(bob_orders) == 2

    def test_empty_tables(self):
        """Test with empty input."""
        result = normalise_table_keys({}, {})
        assert result == {}

    def test_missing_primary_key_column(self):
        """Test error handling for missing primary key column."""
        df = pd.DataFrame({"name": ["Alice", "Bob"]})
        tables = {"users": df}
        primary_keys = {"users": "missing_id"}

        with pytest.raises(
            ValueError, match="Primary key column 'missing_id' not found"
        ):
            normalise_table_keys(tables, primary_keys)

    def test_missing_foreign_key_column(self):
        """Test error handling for missing foreign key column."""
        users = pd.DataFrame({"user_id": [1, 2], "name": ["Alice", "Bob"]})
        orders = pd.DataFrame(
            {"order_id": [100, 200], "amount": [25.99, 15.50]}
        )

        tables = {"users": users, "orders": orders}
        primary_keys = {"users": "user_id", "orders": "order_id"}
        foreign_keys = {"orders": {"missing_user_id": "users"}}

        with pytest.raises(
            ValueError, match="Foreign key column 'missing_user_id' not found"
        ):
            normalise_table_keys(tables, primary_keys, foreign_keys)

    def test_missing_referenced_table(self):
        """Test error handling for missing referenced table."""
        orders = pd.DataFrame(
            {
                "order_id": [100, 200],
                "user_id": [1, 2],
                "amount": [25.99, 15.50],
            }
        )

        tables = {"orders": orders}
        primary_keys = {"orders": "order_id"}
        foreign_keys = {"orders": {"user_id": "missing_users"}}

        with pytest.raises(
            ValueError, match="Referenced table 'missing_users' not found"
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
        users = pd.DataFrame({"user_id": [1, 2], "name": ["Alice", "Bob"]})
        orders = pd.DataFrame({"order_id": [100, 200], "user_id": [1, 2]})

        tables = {"users": users, "orders": orders}
        primary_keys = {"users": "user_id", "orders": "order_id"}

        result = _auto_detect_foreign_keys(tables, primary_keys)

        expected = {"orders": {"user_id": "users"}}
        assert result == expected

    def test_no_foreign_keys(self):
        """Test when no foreign keys are detected."""
        users = pd.DataFrame({"user_id": [1, 2], "name": ["Alice", "Bob"]})
        products = pd.DataFrame(
            {"product_id": [100, 200], "name": ["Widget", "Gadget"]}
        )

        tables = {"users": users, "products": products}
        primary_keys = {"users": "user_id", "products": "product_id"}

        result = _auto_detect_foreign_keys(tables, primary_keys)

        assert result == {}

    def test_multiple_foreign_keys_detection(self):
        """Test detection of multiple foreign keys in one table."""
        users = pd.DataFrame({"user_id": [1, 2], "name": ["Alice", "Bob"]})
        categories = pd.DataFrame(
            {"category_id": [10, 20], "name": ["Electronics", "Books"]}
        )
        orders = pd.DataFrame(
            {
                "order_id": [100, 200],
                "user_id": [1, 2],
                "category_id": [10, 20],
            }
        )

        tables = {"users": users, "categories": categories, "orders": orders}
        primary_keys = {
            "users": "user_id",
            "categories": "category_id",
            "orders": "order_id",
        }

        result = _auto_detect_foreign_keys(tables, primary_keys)

        expected = {
            "orders": {"user_id": "users", "category_id": "categories"}
        }
        assert result == expected
