"""
Collision checker stub.
"""


class CollisionChecker:
    def __init__(self, scene=None):
        self.scene = scene

    def check(self, path):
        return {"path": path, "collisions": []}
