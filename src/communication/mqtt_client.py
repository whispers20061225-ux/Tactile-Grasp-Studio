"""
MQTT client stub.
"""


class MQTTClient:
    def __init__(self, host="localhost", port=1883):
        self.host = host
        self.port = port

    def connect(self):
        return True

    def publish(self, topic, payload):
        return {"topic": topic, "payload": payload}

    def subscribe(self, topic):
        return {"topic": topic}
