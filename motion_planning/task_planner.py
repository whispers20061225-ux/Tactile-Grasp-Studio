"""
Task planner stub.
"""


class TaskPlanner:
    def __init__(self):
        self.tasks = []

    def add_task(self, task):
        self.tasks.append(task)

    def next_task(self):
        return self.tasks[0] if self.tasks else None
