"""
Performance monitor stub.
"""

import time


class PerformanceMonitor:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        if self.start_time is None:
            return 0
        elapsed = time.time() - self.start_time
        self.start_time = None
        return elapsed
