from .base import TaskHandler
import torch
import numpy as np
from src.signals import generate_ref, pitch_eval_ref, roll_eval_ref

class PitchRollHandler(TaskHandler):
    def __init__(self, agent, ep_length):
        self.type = "PitchRoll"
        self.agent = agent
        self.ep_length = ep_length
        self.ref_pitch = generate_ref(self.ep_length, offset = 0.032)
        self.ref_roll = generate_ref(self.ep_length, offset = 0)
        
    # Pitch and pitch rate: 8th and 2nd outputs
    # Roll and roll rate: 7th and 1st outputs
    # 0.032 is the trim pitch, 0.0 is the trim roll
    def give_initial_state(self, output):
        state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1]), np.rad2deg(output[6]), np.rad2deg(output[0])]
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

        return [pitch_action, roll_action, 0, 0, 0, 0, 0, 0, 1449.775, 1449.775], action
    
    # Sample mean action from the policy
    def mean_action(self, state):
        action, _, _ = self.agent.actor.sample(state)
        action = action.detach().cpu().numpy()[0]
        pitch_action = action[0].item()
        roll_action = action[1].item()

        return [pitch_action, roll_action, 0, 0, 0, 0, 0, 0, 1449.775, 1449.775], action
    
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
        reward = -0.6*np.abs(next_state[0][0]) - 0.4*np.abs(next_state[0][2])
        
        return next_state, reward