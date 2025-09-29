from src.handlers import TaskHandler
from .base import FaultHandler
import numpy as np

"""
This fault class defines the behaviours in the jolt fault scenario:
- First fault at 5s to 6s, maximum positive deflection
- Second fault at 10s to 11s, maximum negative deflection
- Works as normal outside these intervals
"""

class JoltFault(FaultHandler):
    def __init__(self, wrapped_handler: TaskHandler):
        self._handler = wrapped_handler
        self.timestep = 0.0
        self.max = 0.26
        self.min = -0.26
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

        if 5.0 < self.timestep < 6.0:
            action_vector = self.mapping[self._handler.type](action_vector, "max")

        elif 10.0 < self.timestep < 11.0:
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