from src.handlers import TaskHandler
from .base import FaultHandler
import numpy as np
from config import Config

"""
This fault class defines the behaviours in the quarter efficiency fault scenario:
- Evaluation reference is random
- The actuator works at quarter efficiency
"""

class QuarterEfficiencyFault(FaultHandler):
    def __init__(self, wrapped_handler: TaskHandler, config: Config):
        self._handler = wrapped_handler
        self.eff = 0.25
        self.ep_length = config.faults['quarter'].ep_length
        self.mapping = {
            "Pitch": self._apply_pitch,
            "Roll": self._apply_roll,
            "PitchRoll": self._apply_pitchroll,
        }
    
    # Called when an attribute/method isn't found on FaultHandler itself.
    def __getattr__(self, name):
        return getattr(self._handler, name)

    # Use a random reference instead of the eval reference
    def eval_reference(self, timestep):
        return self._handler.random_reference(timestep)
    
    # Scale the action by the efficiency factor
    def mean_action(self, state):
        action_vector, action = self._handler.mean_action(state)

        self.mapping[self._handler.type](action_vector)

        return action_vector, action
    
    def actuator_list(self, action_vector):
        if self._handler.type.lower() == 'pitch':
            return np.rad2deg(action_vector[0])
        
        elif self._handler.type.lower() == 'roll':
            return np.rad2deg(action_vector[1])
        
        elif self._handler.type.lower() == 'pitchroll':
            return [np.rad2deg(action_vector[0]), np.rad2deg(action_vector[1])]
    
    def _apply_pitch(self, v):
        v[0] *= self.eff

    def _apply_roll(self, v):
        v[1] *= self.eff

    def _apply_pitchroll(self, v):
        v[0] *= self.eff
        v[1] *= self.eff
