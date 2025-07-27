# Key normalisation

The `normalise_keys` module provides utilities for normalising primary and foreign keys in related database tables to zero-based sequential indices while preserving relationships.

## Functions

### `normalise_table_keys(tables, primary_keys, foreign_keys=None)`

Normalises primary and foreign keys across multiple related tables.

**Parameters:**
- `tables` (Dict[str, pd.DataFrame]): Dictionary mapping table names to DataFrames
- `primary_keys` (Dict[str, str]): Dictionary mapping table names to their primary key column names  
- `foreign_keys` (Optional[Dict[str, Dict[str, str]]]): Dictionary mapping table names to their foreign key relationships. Format: `{table_name: {fk_column: referenced_table}}`. If None, foreign keys are auto-detected.

**Returns:**
- Dict[str, pd.DataFrame]: Dictionary of normalised tables with zero-based integer keys

**Example:**
```python
import pandas as pd
from policyengine_data import normalise_table_keys

users = pd.DataFrame({
    'user_id': [101, 105, 103],
    'name': ['Alice', 'Bob', 'Carol']
})

orders = pd.DataFrame({
    'order_id': [201, 205, 207], 
    'user_id': [105, 101, 105],
    'amount': [25.99, 15.50, 42.00]
})

tables = {'users': users, 'orders': orders}
primary_keys = {'users': 'user_id', 'orders': 'order_id'}

# Auto-detect foreign keys
normalised = normalise_table_keys(tables, primary_keys)

# Or specify foreign keys explicitly
foreign_keys = {'orders': {'user_id': 'users'}}
normalised = normalise_table_keys(tables, primary_keys, foreign_keys)
```

After normalisation:
- User IDs become 0, 1, 2 (instead of 101, 105, 103)
- Order IDs become 0, 1, 2 (instead of 201, 205, 207)  
- Foreign key relationships are preserved (Bob's orders still reference Bob's new ID)

### `normalise_single_table_keys(df, key_column, start_index=0)`

Normalises keys in a single table to sequential indices.

**Parameters:**
- `df` (pd.DataFrame): DataFrame to normalise
- `key_column` (str): Name of the key column to normalise
- `start_index` (int): Starting index for normalisation (default: 0)

**Returns:**
- pd.DataFrame: DataFrame with normalised keys

**Example:**
```python
import pandas as pd
from policyengine_data import normalise_single_table_keys

df = pd.DataFrame({
    'id': [101, 105, 103],
    'value': ['A', 'B', 'C'] 
})

normalised = normalise_single_table_keys(df, 'id')
# Result: IDs become 0, 1, 2
```

## Key features

- **Relationship preservation**: All foreign key relationships between tables are maintained after normalisation
- **Auto-detection**: Foreign keys can be automatically detected based on column name matching
- **Zero-based indexing**: Keys are normalised to start from 0 and increment sequentially
- **Flexible input**: Works with any pandas DataFrames and column names
- **Error handling**: Clear error messages for missing columns or invalid references
- **Duplicate handling**: Properly handles duplicate keys within tables

## Use cases

This functionality is particularly useful for:

- Preparing data for machine learning models that expect sequential indices
- Converting legacy database exports with non-sequential primary keys  
- Standardising key formats across multiple related datasets
- Reducing memory usage by converting large integer keys to compact sequential indices
- Creating consistent test datasets with predictable key patterns

## Implementation notes

The normalisation process works in two phases:

1. **Mapping creation**: Unique values in each primary key column are mapped to zero-based sequential integers
2. **Application**: These mappings are applied to both primary keys and corresponding foreign keys across all tables

Foreign key auto-detection works by identifying columns that share names with primary key columns from other tables. For more complex relationships, explicit foreign key specification is recommended.