"""
Trajectory generator stub.
"""


class TrajectoryGenerator:
    def __init__(self, planner=None):
        self.planner = planner

    def generate(self, path):
        return {"path": path, "trajectory": []}
