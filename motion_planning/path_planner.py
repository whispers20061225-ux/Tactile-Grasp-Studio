"""
Path planner stub leveraging perception and arm control.
"""


class PathPlanner:
    def __init__(self, planner=None):
        self.planner = planner

    def plan(self, start, goal):
        return {"start": start, "goal": goal, "path": []}
