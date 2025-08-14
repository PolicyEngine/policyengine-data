# PolicyEngine survey weight calibration guide

This notebook demonstrates how to use the two main calibration routines available in PolicyEngine Data:

1. **Geographic level iteration**: Calibrating one geographic level at a time from lowest to highest in hierarchy
2. **All levels at once**: Stacking datasets at the lowest level and calibrating for all geographic levels simultaneously

Both methods adjust household weights to match official statistics (targets) while maintaining data representativeness with a gradient descent algorithm implemented in PolicyEngine's [`microcalibrate`](https://policyengine.github.io/microcalibrate/) package.

```python
# Import required libraries
import logging
import numpy as np
import pandas as pd

from policyengine_data.calibration.calibrate import (
    calibrate_single_geography_level,
    calibrate_all_levels,
    areas_in_state_level,
    areas_in_national_level
)
from policyengine_data.tools.legacy_class_conversions import (
    SingleYearDataset_to_Dataset,
)

# Set up logging to see calibration progress
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
```

## Method 1: geographic level iteration

This approach calibrates one geographic level at a time, moving from the lowest (e.g., state) to highest (e.g., national) in the hierarchy. It uses sparsity regularization with an L0 penalty at lower levels to reduce computational costs, then refines weights at higher levels.

### Key features:
- **Sequential calibration**: From the lowest level to the highest (Eg. in the US, first congressional districts, then states, the national)
- **Sparsity regularization**: L0 regularization reduces the number of non-zero weights, then the dataset is minimized to store only the records whose weights are non-zero
- **Weight preservation**: Each calibration starts from the previous level's calibrated weights
- **Computational efficiency**: Datasets are regularized at each step except for final calibration

```python
# Step 1: Calibrate state level with sparsity
print("=== Step 1: State level calibration ===")

# Use a small subset of states for demonstration
demo_state_areas = {
    "California": "0400000US06",
    "Texas": "0400000US48",
    "New York": "0400000US36"
}

state_level_calibrated_dataset = calibrate_single_geography_level(
    calibration_areas=demo_state_areas,
    dataset="hf://policyengine/policyengine-us-data/cps_2023.h5",
    dataset_subsample_size=10000,  # Small sample for faster execution
    use_dataset_weights=False,  # Start with equal weights
    regularize_with_l0=True,  # Enable sparsity
    noise_level=10.0
)

# Examine the results
state_weights = state_level_calibrated_dataset.entities["household"]["household_weight"].values
print(f"State calibration completed:")
print(f"  - Number of households: {len(state_weights)}")
print(f"  - Non-zero weights: {np.count_nonzero(state_weights)}")
print(f"  - Sparsity ratio: {1 - np.count_nonzero(state_weights)/len(state_weights):.2%}")
print(f"  - Weight range: [{state_weights.min():.2f}, {state_weights.max():.2f}]")

# Save state-calibrated dataset
SingleYearDataset_to_Dataset(
    state_level_calibrated_dataset, 
    output_path="Demo_Dataset_state_level.h5"
)

# Step 2: Calibrate national level using state-calibrated weights
print("\n=== Step 2: National level calibration ===")

national_level_calibrated_dataset = calibrate_single_geography_level(
    calibration_areas=areas_in_national_level,
    dataset="Demo_Dataset_state_level.h5",  # Use state-calibrated dataset
    stack_datasets=False,  # Don't stack since we're using pre-stacked data
    db_uri="sqlite:///policy_data.db",
    noise_level=0.0,  # Minimal noise to preserve state calibration
    use_dataset_weights=True,  # Start from state-calibrated weights
    regularize_with_l0=False  # No sparsity at national level
)

# Compare results
national_weights = national_level_calibrated_dataset.entities["household"]["household_weight"].values
print(f"National calibration completed:")
print(f"  - Number of households: {len(national_weights)}")
print(f"  - Weight range: [{national_weights.min():.2f}, {national_weights.max():.2f}]")

# Save final dataset
SingleYearDataset_to_Dataset(
    national_level_calibrated_dataset,
    output_path="Demo_Dataset_national_level.h5"
)

# Verify that calibration changed weights
weight_difference = abs(state_weights - national_weights).sum()
print(f"\nTotal weight change from state to national: {weight_difference:.2f}")
print(f"Average absolute change per household: {weight_difference/len(state_weights):.4f}")
```

## Method 2: all levels at once

This approach stacks the base dataset for multiple geographic areas at the lowest level and then calibrates said dataset for all levels simultaneously. It provides richer data but requires more computational resources.

### Key features:
- **Simultaneous calibration**: All geographic levels calibrated together
- **Data stacking**: Base dataset replicated for each geographic area at the specified level (most often the lowest level in the geographic hierarchy)
- **Data richness**: More observations as the dataset is not regularized until the final calibration

```python 
print("=== Method 2: All levels at once ===")

# Use the same subset of states for fair comparison
fully_calibrated_dataset = calibrate_all_levels(
    database_stacking_areas=demo_state_areas,
    dataset="hf://policyengine/policyengine-us-data/cps_2023.h5",
    db_uri="sqlite:///policy_data.db",
    dataset_subsample_size=1000,  # Sample size per area
    regularize_with_l0=True,  # Enable sparsity
    noise_level=10.0,
    raise_error=False  # Don't fail if some targets have no contributing records
)

# Examine results
full_weights = fully_calibrated_dataset.entities["household"]["household_weight"].values
print(f"Full calibration completed:")
print(f"  - Number of households: {len(full_weights)}")
print(f"  - Expected max (before sparsity): {1000 * len(demo_state_areas)}")
print(f"  - Non-zero weights: {np.count_nonzero(full_weights)}")
print(f"  - Sparsity ratio: {1 - np.count_nonzero(full_weights)/len(full_weights):.2%}")
print(f"  - Weight range: [{full_weights.min():.2f}, {full_weights.max():.2f}]")

# Save fully calibrated dataset
SingleYearDataset_to_Dataset(
    fully_calibrated_dataset, 
    output_path="Demo_Dataset_fully_calibrated.h5"
)
```

## When to use each method

### Geographic level iteration (`calibrate_single_geography_level`)

**Use when:**
- You have limited computational resources
- You want fine-grained control over each geographic level
- You need to debug calibration issues at specific levels
- You have hierarchical targets that should be calibrated sequentially

**Key parameters:**
- `regularize_with_l0=True`: Enable sparsity at lower levels
- `noise_level=0.0`: Minimize changes when refining upper levels
- `use_dataset_weights=True`: Preserve previous calibration results
- `stack_datasets=False`: Use pre-processed datasets in subsequent steps

### All levels at once (`calibrate_all_levels`)

**Use when:**
- You have sufficient computational resources
- You want to optimize across all geographic levels simultaneously
- You need maximum data richness for statistical accuracy
- Your targets are independent across geographic levels or not present in each of the levels

**Key parameters:**
- `dataset_subsample_size`: Balance between accuracy and computation time
- `regularize_with_l0=True`: Control sparsity in the final result

## Best practices

1. **Start small**: Use subsamples and limited geographic areas for testing
2. **Monitor sparsity**: High sparsity reduces computation but may lose representativeness, explore the `microcalibrate` repo to understand the hyperparameters that affect it and adjust them
3. **Validate results**: Check that calibrated weights produce expected target values (`microcalibrate` includes a dashboard that allows close evaluation)
4. **Save intermediate results**: Keep state-level datasets for debugging
5. **Use appropriate noise levels**: Higher noise helps avoid local minima, but too much distorts results, specially when building on previous calibrations
