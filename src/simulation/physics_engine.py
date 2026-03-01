"""
Physics engine stub.
"""


class PhysicsEngine:
    def __init__(self, backend="pybullet"):
        self.backend = backend

    def load(self):
        return True

    def step(self):
        return None
