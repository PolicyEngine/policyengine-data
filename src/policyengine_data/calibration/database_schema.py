"""Compatibility helpers for calibration database schema drift."""


def ensure_calibration_schema(engine) -> None:
    """Add nullable columns expected by current calibration queries if absent."""
    with engine.begin() as conn:
        table_names = {
            row[0]
            for row in conn.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

        if "strata" in table_names:
            strata_columns = {
                row[1]
                for row in conn.exec_driver_sql("PRAGMA table_info(strata)")
            }
            if "stratum_group_id" not in strata_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE strata ADD COLUMN stratum_group_id INTEGER"
                )
                conn.exec_driver_sql(
                    "UPDATE strata SET stratum_group_id = stratum_id "
                    "WHERE stratum_group_id IS NULL"
                )
            if "parent_stratum_id" not in strata_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE strata ADD COLUMN parent_stratum_id INTEGER"
                )
            if "definition_hash" not in strata_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE strata ADD COLUMN definition_hash TEXT"
                )

        if "targets" in table_names:
            target_columns = {
                row[1]
                for row in conn.exec_driver_sql("PRAGMA table_info(targets)")
            }
            if "active" not in target_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE targets ADD COLUMN active BOOLEAN"
                )
                conn.exec_driver_sql(
                    "UPDATE targets SET active = 1 WHERE active IS NULL"
                )
            if "tolerance" not in target_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE targets ADD COLUMN tolerance REAL"
                )
            if "notes" not in target_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE targets ADD COLUMN notes TEXT"
                )
            if "reform_id" not in target_columns:
                conn.exec_driver_sql(
                    "ALTER TABLE targets ADD COLUMN reform_id INTEGER"
                )
                conn.exec_driver_sql(
                    "UPDATE targets SET reform_id = 0 WHERE reform_id IS NULL"
                )
