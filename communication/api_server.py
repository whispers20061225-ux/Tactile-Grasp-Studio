"""
REST API server stub.
"""


class APIServer:
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port

    def start(self):
        return True

    def stop(self):
        return True
