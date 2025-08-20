"""
Additional utilities for the calibration process.
"""

from typing import Dict, List

import numpy as np
import torch


def create_geographic_normalization_factor(
    geo_hierarchy: List[str],
    target_info: Dict[int, Dict[str, any]],
) -> torch.Tensor:
    """
    Create a normalization factor for the calibration process to balance targets that belong to different geographic areas or concepts.

    Args:
        geo_hierarchy (List[str]): Geographic hierarchy levels' codes (e.g., ["0100000US", "0400000US", "0500000US"]). Make sure to pass the part of the code general to all areas within a given level.
        target_info (Dict[int, Dict[str, any]]): A dictionary containing information about each target, including its name which denotes geographic area and its active status.

    Returns:
        normalization_factor (torch.Tensor): Normalization factor for each active target.
    """
    is_active = []
    geo_codes = []
    geo_level_sum = {}

    for code in geo_hierarchy:
        geo_level_sum[code] = 0

    # First pass: collect active status and geo codes for all targets
    for target_id, info in target_info.items():
        is_active.append(info["active"])
        target_name = info["name"]
        matched_geo = None

        for code in geo_hierarchy:
            if code in target_name:
                matched_geo = code
                if info["active"]:
                    geo_level_sum[code] += 1
                break

        geo_codes.append(matched_geo)

    is_active = torch.tensor(is_active, dtype=torch.float32)
    normalization_factor = torch.zeros_like(is_active)

    # Assign normalization factors based on geo level for each target
    for i, (is_target_active, geo_code) in enumerate(
        zip(is_active, geo_codes)
    ):
        if (
            is_target_active
            and geo_code is not None
            and geo_level_sum[geo_code] > 0
        ):
            normalization_factor[i] = 1.0 / geo_level_sum[geo_code]

    # Check if only one geographic level is represented among active targets
    active_geo_levels = set()
    for i, is_target_active in enumerate(is_active):
        if is_target_active and geo_codes[i] is not None:
            active_geo_levels.add(geo_codes[i])

    # If no matching geo codes for active targets, return zeros for active targets
    if len(active_geo_levels) == 0:
        active_factors = torch.zeros(sum(is_active.bool()))
        return active_factors

    # If only one geographic level is present, return tensor of ones for active targets
    if len(active_geo_levels) <= 1:
        normalization_factor = torch.where(
            is_active.bool(), torch.tensor(1.0), torch.tensor(0.0)
        )
    else:
        # Apply mean normalization for multiple geographic levels
        active_factors = normalization_factor[is_active.bool()]
        if len(active_factors) > 0 and active_factors.sum() > 0:
            inv_mean_norm = 1.0 / active_factors.mean()
            normalization_factor = normalization_factor * inv_mean_norm

    return normalization_factor[is_active.bool()]
