"""
Experiment recorder stub.
"""


class ExperimentRecorder:
    def __init__(self, log_path="logs/experiment.log"):
        self.log_path = log_path

    def record(self, entry):
        return True
