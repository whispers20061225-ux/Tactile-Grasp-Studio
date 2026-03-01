"""
Scene builder stub.
"""


class SceneBuilder:
    def __init__(self):
        self.objects = []

    def add_object(self, obj):
        self.objects.append(obj)

    def build(self):
        return self.objects
