"""
Test the logic for creating a normalization factor per target to balance targets represented at different concepts or geographic levels.
"""

import torch
from policyengine_data.calibration.utils import (
    create_geographic_normalization_factor,
)


def test_multiple_geo_levels_normalization() -> None:
    """
    Test normalization factors with multiple geographic levels.
    """
    geo_hierarchy = ["0100000US", "0400000US", "0500000US"]
    target_info = {
        1: {"name": "0100000US_population", "active": True},
        2: {"name": "0400000US01_population", "active": True},
        3: {"name": "0400000US02_population", "active": True},
        4: {"name": "0500000US0101_population", "active": True},
        5: {"name": "0500000US0102_population", "active": False},
    }

    normalization_factor = create_geographic_normalization_factor(
        geo_hierarchy, target_info
    )
    active_targets = sum(
        [1 for info in target_info.values() if info["active"]]
    )

    # Should return factors for active targets
    assert (
        len(normalization_factor) == active_targets
    ), "Normalization factor length does not match number of active targets."

    # Active factors should have mean = 1.0 (due to mean normalization)
    mean_factor = normalization_factor.mean().item()
    assert (
        abs(mean_factor - 1.0) < 1e-6
    ), "Normalization factor does not have a mean of 1."


def test_single_geo_level_returns_ones() -> None:
    """
    Test that single geographic level returns tensor of ones for active targets.
    """
    geo_hierarchy = ["0100000US", "0400000US", "0500000US"]
    target_info = {
        1: {"name": "0400000US01_population", "active": True},
        2: {"name": "0400000US02_population", "active": True},
        3: {"name": "0400000US03_population", "active": False},
    }

    normalization_factor = create_geographic_normalization_factor(
        geo_hierarchy, target_info
    )

    # Active targets should have factor = 1.0
    assert normalization_factor[0] == 1.0
    assert normalization_factor[1] == 1.0


def test_all_inactive_targets() -> None:
    """
    Test behavior when all targets are inactive.
    """
    geo_hierarchy = ["0100000US", "0400000US"]
    target_info = {
        1: {"name": "0100000US_population", "active": False},
        2: {"name": "0400000US01_population", "active": False},
    }

    normalization_factor = create_geographic_normalization_factor(
        geo_hierarchy, target_info
    )

    # All factors should be zero
    assert (
        len(normalization_factor) == 0
    ), "Normalization factor length should be 0 as there are no active targets."


def test_no_matching_geo_codes() -> None:
    """
    Test behavior when target names don't match geographic codes.
    """
    geo_hierarchy = ["0100000US", "0400000US"]
    target_info = {
        1: {"name": "some_other_target", "active": True},
        2: {"name": "another_target", "active": True},
    }

    normalization_factor = create_geographic_normalization_factor(
        geo_hierarchy, target_info
    )

    # All factors should be zero since no geo codes match
    assert torch.all(
        normalization_factor == 0
    ), "Normalization factors should be zero since no geo codes match."
