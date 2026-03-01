"""
Perception package stub.

Provides thin wrappers over existing tactile_perception modules and stubs for
vision components to satisfy imports in the rest of the project.
"""

from importlib import import_module


def _proxy(module_name):
    try:
        return import_module(f"tactile_perception.{module_name}")
    except Exception:
        return None


# Re-export tactile perception modules if available
sensor_reader = _proxy("sensor_reader")
data_processor = _proxy("data_processor")
tactile_mapper = _proxy("tactile_mapper")

__all__ = [
    "sensor_reader",
    "data_processor",
    "tactile_mapper",
]
