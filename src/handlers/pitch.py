from .base import TaskHandler
import torch
import numpy as np
from src.signals import generate_ref, pitch_eval_ref
from config import Config

class PitchHandler(TaskHandler):
    def __init__(self, agent, ep_length, config: Config):
        self.type = "Pitch"
        self.agent = agent
        self.ep_length = ep_length

        self.offset = config.tasks['pitch'].offset
        self.trim_inputs = config.globals.trim_inputs
        self.reward_weight = config.tasks['pitch'].reward_weight

        self.ref_function = generate_ref(self.ep_length, offset = self.offset)


    # The pitch is the 8th output, rate is the 2nd output, 0.032 is the trim pitch
    def give_initial_state(self, output):
        state = [np.rad2deg(self.offset - output[7]), np.rad2deg(output[1])]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        return state_tensor

    def random_reference(self, timestep):
        return self.ref_function(timestep)
    
    def eval_reference(self, timestep):
        return pitch_eval_ref(timestep)

    # Sample stochastic action from the policy, the entry is the roll trim input
    def sample_action(self, state):
        _, action, _ = self.agent.actor.sample(state)
        action = action.detach().cpu().numpy()[0]
        self.trim_inputs[0] = action.item()
        
        return self.trim_inputs, action
    
    # Sample mean action from the policy
    def mean_action(self, state):
        action, _, _ = self.agent.actor.sample(state)
        action = action.detach().cpu().numpy()[0]
        self.trim_inputs[0] = action.item()
        
        return self.trim_inputs, action
    
    def state_list(self, output):
        l = [np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
            np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
            np.rad2deg(output[8]),output[9]]
        
        return l
    
    def action_list(self,action):
        return [np.rad2deg(action.item()),0]
    
    def ref_list(self, reference):
        return np.rad2deg(reference)
        
    def add_buffer(self, state, action, next_state, reward, done):
        self.agent.replay_buffer.add(state, action, next_state, reward, done)
    
    def compute_state_and_reward(self, output, reference):
        next_state = torch.FloatTensor([np.rad2deg(reference - output[7]), np.rad2deg(output[1])]).unsqueeze(0)
        reward = self.reward_weight * np.abs(next_state[0][0])
        
        return next_state, reward
