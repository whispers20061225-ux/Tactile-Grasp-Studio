"""
Data processor shim for perception module.
"""

try:
    from tactile_perception.data_processor import *  # type: ignore
except Exception:
    class DataProcessor:
        """Minimal placeholder for data processing."""

        def process(self, data):
            return data


__all__ = [name for name in globals() if not name.startswith("_")]
