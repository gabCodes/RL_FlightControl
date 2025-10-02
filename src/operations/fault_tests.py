import os
import numpy as np
from .util import _agentChooser
from .evaluate import evaluate
from .run import _has_weights, _choose_paths
from config import Config

# Loads the agents and evaluates the per run. Returns the error per axis, episode reward and whether it was terminated.
def fault_test(agentType: str, task: str, fault: str, config: Config, plot = False) -> list[float] | list[float, float]:
    
    error_list, reward_list, fail_list = [], [], []

    nr_runs, ep_num, step, resolution, run_dir = _faultLoader(config)
    agent = _agentChooser(agentType, task, config)

    task = task.upper()

    for run_nr in range(nr_runs):
        run_name = f'RUN{run_nr + 1}'
        run_dir = os.path.join(run_dir, run_name)

        if not _has_weights(agentType, task, run_dir, ep_num, resolution):
            print(f"Skipping run {run_name}: Missing last step weights")
            continue

        ep_name = f'EP{ep_num}'
        ep_dir = os.path.join(run_dir, ep_name)

        actor_path, critic_path = _choose_paths(agentType, agent, task, ep_dir, resolution, step)

        agent.load_weights(actor_path, critic_path)

        error, ep_reward, terminated = evaluate(agent, task, fault, config, plot = plot)
        error_list.append(error)
        reward_list.append(ep_reward)
        fail_list.append(terminated)

    error_array = np.array(error_list)
    err = np.mean(error_array, axis=0)
    reward_array = np.array(reward_list)
    reward = np.mean(reward_array,axis=0)
    nr_fails = sum(fail_list)

    print("==================================")
    print(f"Agent: {agentType}, Fault: {fault}, Task: {task}")
    print("Mean error across columns:", err)
    print("Mean reward:", reward)
    print("Number of failures:", nr_fails)

    return error_array

# Loads the desired parameters from the config file
def _faultLoader(config: Config) -> tuple[int, int, int, int, str]:
    nr_runs = config.phases['run'].nr_runs
    ep_num = config.phases['run'].ep_num[1]
    step = config.phases['train'].ep_length * 100 #Since simulation runs at 0.01 timesteps
    resolution = config.phases['train'].resolution[1]
    run_dir = config.phases['train'].save_dir

    return nr_runs, ep_num, step, resolution, run_dir
