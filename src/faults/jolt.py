from src.handlers import TaskHandler
from .base import FaultHandler
import numpy as np
from config import Config

"""
This fault class defines the behaviours in the jolt fault scenario:
- First fault at 5s to 6s, maximum positive deflection
- Second fault at 10s to 11s, maximum negative deflection
- Works as normal outside these intervals
"""

class JoltFault(FaultHandler):
    def __init__(self, wrapped_handler: TaskHandler, config: Config):
        self._handler = wrapped_handler
        self.timestep = 0.0
        self.max = config.globals.max_action
        self.min = -1 * config.globals.max_action

        self.ep_length = config.faults['jolt'].ep_length
        self.start_maxd = config.faults['jolt'].start_maxd
        self.end_maxd = config.faults['jolt'].end_maxd
        self.start_mind = config.faults['jolt'].start_mind
        self.end_mind = config.faults['jolt'].end_mind
        
        self.mapping = {
            "Pitch": self._apply_pitch,
            "Roll": self._apply_roll,
            "PitchRoll": self._apply_pitchroll
        }
    
    # Called when an attribute/method isn't found on FaultHandler itself.
    def __getattr__(self, name):
        return getattr(self._handler, name)
    
    # Introduce faults at specified intervals
    def mean_action(self, state):
        action_vector, action = self._handler.mean_action(state)

        if self.start_maxd < self.timestep < self.end_maxd:
            action_vector = self.mapping[self._handler.type](action_vector, "max")

        elif self.start_mind < self.timestep < self.end_mind:
            action_vector = self.mapping[self._handler.type](action_vector, "min")

        self.timestep += 0.01
        
        return action_vector, action
    
    def actuator_list(self, action_vector):
        if self._handler.type.lower() == 'pitch':
            return np.rad2deg(action_vector[0])
        
        elif self._handler.type.lower() == 'roll':
            return np.rad2deg(action_vector[1])
        
        elif self._handler.type.lower() == 'pitchroll':
            return [np.rad2deg(action_vector[0]), np.rad2deg(action_vector[1])]

    # Jolt the elevator to max or min deflection
    def _apply_pitch(self, v, direction) -> list[float]:
        if direction == "max":
            v[0] = self.max
        else:
            v[0] = self.min
        
        return v

    # Jolt the aileron to max or min deflection
    def _apply_roll(self, v, direction) -> list[float]:
        if direction == "max":
            v[1] = self.max
        else:
            v[1] = self.min

        return v

    # Jolt both elevator and aileron to max or min deflection
    def _apply_pitchroll(self, v, direction) -> list[float]:
        if direction == "max":
            v[0], v[1] = self.max, self.max
        else:
            v[0], v[1] = self.min, self.min

        return v