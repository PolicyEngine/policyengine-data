"""Compatibility wrapper for PolicyEngine US UCGID identifiers."""

try:
    from policyengine_us.variables.household.demographic.geographic.ucgid.ucgid_enum import (
        UCGID,
    )
except ModuleNotFoundError:

    class UCGID(str):
        @property
        def name(self):
            return str(self)
