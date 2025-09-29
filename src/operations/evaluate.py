import torch
from citation import initialize, step, terminate
from config import Config
from .util import _choose_handler
from src.agents import SACAgent, REDQSACAgent
from src.plotter import plot_states



# Evaluate agents based on task
def evaluate(agent: SACAgent | REDQSACAgent, task: str, fault: str, config: Config, plot: bool = False, allstates: bool = False) -> float:

    ep_length, trim_inputs = config.phases['eval'].ep_length, config.globals.trim_inputs

    handler = _choose_handler(agent, task, ep_length, fault)
    
    state_list, action_list, actuator_list, ref_list, time_list = [], [], [], [], []

    ep_reward = 0
    timestep = 0
    done = False
    terminated = False

    initialize()

    for _ in range(2000):
        output = step(trim_inputs)

    state_tensor = handler.give_initial_state(output)


    while not (done or terminated):

        reference = handler.eval_reference(timestep)

        action_vector, action = handler.mean_action(state_tensor)

        output = step(action_vector)

        next_state, reward = handler.compute_state_and_reward(output, reference)

        if torch.isnan(next_state).any():
            terminated = True
            break
            
        state_list.append(handler.state_list(output))
        action_list.append(handler.action_list(action))

        if fault:
            actuator_list.append(handler.actuator_list(action_vector))

        ref_list.append(handler.ref_list(reference))
        time_list.append(timestep)
        
        state_tensor = next_state
        ep_reward += reward
        timestep += 0.01

        done = timestep > ep_length

    terminate()
    
    if plot:
        if allstates:
            if fault:
                plot_states(task, time_list, ref_list, state_list, action_list, zoom=False, extra=actuator_list)
            else:
                plot_states(task, time_list, ref_list, state_list, action_list, zoom=False)

        else:
            if fault:
                plot_states(task, time_list, ref_list, state_list, action_list, zoom=True, extra=actuator_list)
            else:
                plot_states(task, time_list, ref_list, state_list, action_list, zoom=True)

    return ep_reward