from src.handlers import TaskHandler
from .base import FaultHandler
from config import Config

"""
This is the scenario where the agents are tested under nominal conditions for durations longer than their training episodes.
"""

class Nominal(FaultHandler):
    def __init__(self, wrapped_handler: TaskHandler, config: Config):
        self._handler = wrapped_handler
        self.ep_length = config.faults['nominal'].ep_length

    # Called when an attribute/method isn't found on FaultHandler itself.
    def __getattr__(self, name):
        return getattr(self._handler, name)