from .base import TaskHandler
import torch
import numpy as np
from src.signals import generate_ref, pitch_eval_ref, roll_eval_ref
from config import Config

class PitchRollHandler(TaskHandler):
    def __init__(self, agent, ep_length, config: Config):
        self.type = "PitchRoll"
        self.agent = agent
        self.ep_length = ep_length
        
        self.offset = config.tasks['pitchroll'].offset
        self.trim_inputs = config.globals.trim_inputs
        self.reward_weight = config.tasks['pitchroll'].reward_weight

        self.ref_pitch = generate_ref(self.ep_length, offset = self.offset[0])
        self.ref_roll = generate_ref(self.ep_length, offset = self.offset[1])
        
    # Pitch and pitch rate: 8th and 2nd outputs
    # Roll and roll rate: 7th and 1st outputs
    def give_initial_state(self, output):
        state = [np.rad2deg(self.offset[0] - output[7]), np.rad2deg(output[1]), np.rad2deg(self.offset[1] - output[6]), np.rad2deg(output[0])]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        return state_tensor

    def random_reference(self, timestep):
        return (self.ref_pitch(timestep), self.ref_roll(timestep))
    
    def eval_reference(self, timestep):
        return (pitch_eval_ref(timestep), roll_eval_ref(timestep))

    # Sample stochastic action from the policy
    def sample_action(self, state):
        _, action, _ = self.agent.actor.sample(state)
        action = action.detach().cpu().numpy()[0]
        pitch_action = action[0].item()
        roll_action = action[1].item()

        self.trim_inputs[0], self.trim_inputs[1] = pitch_action, roll_action

        return self.trim_inputs, action
    
    # Sample mean action from the policy
    def mean_action(self, state):
        action, _, _ = self.agent.actor.sample(state)
        action = action.detach().cpu().numpy()[0]
        pitch_action = action[0].item()
        roll_action = action[1].item()

        self.trim_inputs[0], self.trim_inputs[1] = pitch_action, roll_action

        return self.trim_inputs, action
    
    def state_list(self, output):
        l = [np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
            np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
            np.rad2deg(output[8]),output[9]]
        
        return l
    
    def action_list(self,action):
        return [np.rad2deg(action[0].item()),np.rad2deg(action[1].item())]
    
    def ref_list(self, reference):
        pitch_ref, roll_ref = reference
        
        return [np.rad2deg(pitch_ref), np.rad2deg(roll_ref)]
    
    def add_buffer(self, state, action, next_state, reward, done):
        self.agent.replay_buffer.add(state, action, next_state, reward, done)

    # Weights for reward function were found by trial and error, better choice may exist
    def compute_state_and_reward(self, output, reference):
        pitch_ref, roll_ref = reference
        next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[7]), np.rad2deg(output[1]), 
                                            np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = self.reward_weight[0]*np.abs(next_state[0][0]) - self.reward_weight[1]*np.abs(next_state[0][2])
        
        return next_state, reward