"""
Gripper simulator stub.
"""


class GripperSimulator:
    def __init__(self, config=None):
        self.config = config

    def simulate(self, command):
        return {"command": command, "result": "not_implemented"}
