"""
Grasp planner stub.
"""


class GraspPlanner:
    def __init__(self, predictor=None):
        self.predictor = predictor

    def plan_grasp(self, perception_data=None):
        return {"grasp": None, "score": 0.0}
