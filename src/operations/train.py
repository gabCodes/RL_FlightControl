import os
import torch
from citation import initialize, step, terminate
from config import Config
from src.agents import SACAgent, REDQSACAgent
from .util import _agentChooser, _choose_handler

# Train agents based on task
def train(agent: SACAgent | REDQSACAgent, task: str, config: Config, save = True) -> None:
    short_eps, ep_num, ep_length, short_res, resolution, trim_inputs, save_dir = _trainLoader(config)

    handler = _choose_handler(agent, task, config, ep_length)

    task = task.upper()
    
    for episode in range(ep_num):
        ep_name = f'EP{episode+1}'
        ep_dir = os.path.join(save_dir, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        done = False
        terminated = False
        count = 1
        timestep = 0

        initialize()

        # Let it run at trim conditions for 20s to deal with initial jitteriness
        for _ in range(2000):
            output = step(trim_inputs)

        state_tensor = handler.give_initial_state(output)


        while not (done or terminated):
            reference = handler.random_reference(timestep)

            action_vector, action = handler.sample_action(state_tensor)

            output = step(action_vector)

            next_state, reward = handler.compute_state_and_reward(output, reference)

            if torch.isnan(next_state).any():
                terminated = True
                break
            
            handler.add_buffer(state_tensor, action, next_state, reward, done)

            if agent.replay_buffer.size > agent.batch_size:
                _, _, _, _, _ = agent.update()

            if save:

                if episode + 1 <= short_eps and count % short_res == 0:
                    save_path = os.path.join(ep_dir, f"{task}{short_res}{agentType}STEP_{count}.pt")
                    agent.save_weights(save_path)
                    print(f"Saved agent at step {count} to {save_path}")

                if count % resolution == 0:
                    save_path = os.path.join(ep_dir, f"{task}{resolution}{agentType}STEP_{count}.pt")
                    agent.save_weights(save_path)
                    print(f"Saved agent at step {count} to {save_path}")

            state_tensor = next_state
            timestep += 0.01
            count += 1
            done = timestep >= ep_length

        terminate()

    return agent

# Loads necessary parameters for training
def _trainLoader(config: Config) -> tuple[int, int, int, int, int, list[float], str]:
    short_eps = config.phases['train'].ep_num[0]
    ep_num = config.phases['train'].ep_num[1]
    ep_length = config.phases['train'].ep_length
    short_res = config.phases['train'].resolution[0]
    long_res = config.phases['train'].resolution[1]
    save_dir = config.phases['train'].save_dir
    trim_inputs = config.globals.trim_inputs

    return short_eps, ep_num, ep_length, short_res, long_res, trim_inputs, save_dir


