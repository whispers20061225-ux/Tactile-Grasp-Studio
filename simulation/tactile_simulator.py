"""
Tactile simulator stub.
"""


class TactileSimulator:
    def __init__(self, sensor_config=None):
        self.sensor_config = sensor_config

    def simulate_contact(self, contact):
        return {"contact": contact, "forces": []}
