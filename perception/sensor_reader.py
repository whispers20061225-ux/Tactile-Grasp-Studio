"""
Sensor reader shim.

Delegates to tactile_perception.sensor_reader when available to maintain
backward compatibility with the legacy package layout.
"""

try:
    from tactile_perception.sensor_reader import *  # type: ignore
except Exception:
    class SensorReader:
        """Minimal placeholder implementation."""

        def __init__(self, *args, **kwargs):
            self.port = kwargs.get("port")

        def open(self):
            return False

        def read(self):
            return None

        def close(self):
            return None


__all__ = [name for name in globals() if not name.startswith("_")]
