"""
Key normalisation utilities for tables with primary and foreign keys.

This module provides functionality to normalise primary and foreign keys
in related tables to zero-based sequential indices while preserving
relationships between tables.
"""

from typing import Any, Dict, List, Optional, Union

import pandas as pd


def normalise_table_keys(
    tables: Dict[str, pd.DataFrame],
    primary_keys: Dict[str, str],
    foreign_keys: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Normalise primary and foreign keys across multiple tables to zero-based indices.

    This function takes a collection of related tables and converts their primary
    and foreign keys to zero-based sequential integers while preserving all
    relationships between tables.

    Args:
        tables: Dictionary mapping table names to DataFrames
        primary_keys: Dictionary mapping table names to their primary key column names
        foreign_keys: Optional dictionary mapping table names to their foreign key
                     relationships. Format: {table_name: {fk_column: referenced_table}}
                     If None, foreign keys will be auto-detected based on column names
                     matching primary key names from other tables.

    Returns:
        Dictionary of normalised tables with zero-based integer keys

    Example:
        >>> users = pd.DataFrame({
        ...     'user_id': [101, 105, 103],
        ...     'name': ['Alice', 'Bob', 'Carol']
        ... })
        >>> orders = pd.DataFrame({
        ...     'order_id': [201, 205, 207],
        ...     'user_id': [105, 101, 105],
        ...     'amount': [25.99, 15.50, 42.00]
        ... })
        >>> tables = {'users': users, 'orders': orders}
        >>> primary_keys = {'users': 'user_id', 'orders': 'order_id'}
        >>> foreign_keys = {'orders': {'user_id': 'users'}}
        >>> normalised = normalise_table_keys(tables, primary_keys, foreign_keys)
        >>> # Result: user_ids become 0,1,2 and order_ids become 0,1,2
        >>> # with foreign key relationships preserved
    """
    if not tables:
        return {}

    if foreign_keys is None:
        foreign_keys = _auto_detect_foreign_keys(tables, primary_keys)

    # Create mapping dictionaries for each primary key
    key_mappings = {}
    normalised_tables = {}

    # First pass: create mappings for primary keys
    for table_name, df in tables.items():
        if table_name not in primary_keys:
            raise ValueError(
                f"No primary key specified for table '{table_name}'"
            )

        pk_column = primary_keys[table_name]
        if pk_column not in df.columns:
            raise ValueError(
                f"Primary key column '{pk_column}' not found in table '{table_name}'"
            )

        # Get unique values and create zero-based mapping
        unique_keys = df[pk_column].unique()
        key_mappings[table_name] = {
            old_key: new_key for new_key, old_key in enumerate(unique_keys)
        }

    # Second pass: apply mappings to all tables
    for table_name, df in tables.items():
        normalised_df = df.copy()
        pk_column = primary_keys[table_name]

        # Map primary key
        normalised_df[pk_column] = normalised_df[pk_column].map(
            key_mappings[table_name]
        )

        # Map foreign keys
        if table_name in foreign_keys:
            for fk_column, referenced_table in foreign_keys[
                table_name
            ].items():
                if fk_column not in df.columns:
                    raise ValueError(
                        f"Foreign key column '{fk_column}' not found in table '{table_name}'"
                    )
                if referenced_table not in key_mappings:
                    raise ValueError(
                        f"Referenced table '{referenced_table}' not found"
                    )

                normalised_df[fk_column] = normalised_df[fk_column].map(
                    key_mappings[referenced_table]
                )

        normalised_tables[table_name] = normalised_df

    return normalised_tables


def _auto_detect_foreign_keys(
    tables: Dict[str, pd.DataFrame], primary_keys: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    """
    Auto-detect foreign key relationships based on column name matching.

    Args:
        tables: Dictionary of table names to DataFrames
        primary_keys: Dictionary of primary key column names per table

    Returns:
        Dictionary of detected foreign key relationships
    """
    foreign_keys = {}
    pk_columns = set(primary_keys.values())

    for table_name, df in tables.items():
        table_fks = {}
        pk_column = primary_keys[table_name]

        # Look for columns that match primary keys from other tables
        for column in df.columns:
            if column != pk_column and column in pk_columns:
                # Find which table this primary key belongs to
                for ref_table, ref_pk in primary_keys.items():
                    if ref_pk == column and ref_table != table_name:
                        table_fks[column] = ref_table
                        break

        if table_fks:
            foreign_keys[table_name] = table_fks

    return foreign_keys


def normalise_single_table_keys(
    df: pd.DataFrame, key_column: str, start_index: int = 0
) -> pd.DataFrame:
    """
    Normalise keys in a single table to sequential indices.

    Args:
        df: DataFrame to normalise
        key_column: Name of the key column to normalise
        start_index: Starting index for normalisation (default: 0)

    Returns:
        DataFrame with normalised keys

    Example:
        >>> df = pd.DataFrame({
        ...     'id': [101, 105, 103],
        ...     'value': ['A', 'B', 'C']
        ... })
        >>> normalised = normalise_single_table_keys(df, 'id')
        >>> # Result: ids become 0, 1, 2
    """
    if key_column not in df.columns:
        raise ValueError(f"Key column '{key_column}' not found in DataFrame")

    normalised_df = df.copy()
    unique_keys = df[key_column].unique()
    key_mapping = {
        old_key: new_key + start_index
        for new_key, old_key in enumerate(unique_keys)
    }

    normalised_df[key_column] = normalised_df[key_column].map(key_mapping)
    return normalised_df
