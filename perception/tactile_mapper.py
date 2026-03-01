"""
Tactile mapper shim for perception module.
"""

try:
    from tactile_perception.tactile_mapper import *  # type: ignore
except Exception:
    class TactileMapper:
        """Minimal placeholder for tactile mapping."""

        def map(self, data):
            return data


__all__ = [name for name in globals() if not name.startswith("_")]
