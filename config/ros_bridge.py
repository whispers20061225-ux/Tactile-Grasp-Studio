"""
ROS bridge stub.
"""


class ROSBridge:
    def __init__(self, namespace="/learm_arm"):
        self.namespace = namespace

    def publish(self, topic, msg):
        return {"topic": topic, "msg": msg}

    def subscribe(self, topic, callback):
        return {"topic": topic, "callback": callback}
