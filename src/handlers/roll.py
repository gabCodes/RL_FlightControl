from .base import TaskHandler
import torch
import numpy as np
from src.signals import generate_ref, roll_eval_ref
from config import Config

class RollHandler(TaskHandler):
    def __init__(self, agent, ep_length, config: Config):
        self.type = "Roll"
        self.agent = agent
        self.ep_length = ep_length

        self.offset = config.tasks['roll'].offset
        self.trim_inputs = config.globals.trim_inputs
        self.reward_weight = config.tasks['roll'].reward_weight

        self.ref_function = generate_ref(self.ep_length, offset = self.offset)
        
    # The roll is the 7th output, roll rate is the 1st output
    def give_initial_state(self, output):
        state = [np.rad2deg(self.offset - output[6]), np.rad2deg(output[0])]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        return state_tensor

    def random_reference(self, timestep):
        return self.ref_function(timestep)
    
    def eval_reference(self, timestep):
        return roll_eval_ref(timestep)

    # Sample stochastic action from the policy
    def sample_action(self, state):
        _, action, _ = self.agent.actor.sample(state)
        action = action.detach().cpu().numpy()[0]

        self.trim_inputs[1] = action.item()
        
        return self.trim_inputs, action
    
    # Sample mean action from the policy
    def mean_action(self, state):
        action, _, _ = self.agent.actor.sample(state)
        action = action.detach().cpu().numpy()[0]

        self.trim_inputs[1] = action.item()
        
        return self.trim_inputs, action
    

    def state_list(self, output):
        l = [np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
            np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
            np.rad2deg(output[8]),output[9]]
        
        return l
    
    def action_list(self,action):
        return [0,np.rad2deg(action.item())]
    
    def ref_list(self, reference):
        return np.rad2deg(reference)
    
    def add_buffer(self, state, action, next_state, reward, done):
        self.agent.replay_buffer.add(state, action, next_state, reward, done)
    
    def compute_state_and_reward(self, output, reference):
        next_state = torch.FloatTensor([np.rad2deg(reference - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = self.reward_weight*np.abs(next_state[0][0])
        
        return next_state, reward