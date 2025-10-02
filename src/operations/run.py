import os
from config import Config
from src.agents import SACAgent, REDQSACAgent
from .util import _agentChooser
from .train import train
from .evaluate import evaluate

# Train agents, save their weights and evaluate
def runs(agentType: str, task: str, config: Config, nr_runs = 30, training: bool = False, prefix = "") -> None:
    short_eps, long_eps, short_res, long_res, ep_length, w_folder, p_folder = _runLoader(config)

    for run_nr in range(nr_runs):
        agent = _agentChooser(agentType, task, config)
        
        if training == True:
            run_name = f'RUN{run_nr+1}'
            run_dir = os.path.join(w_folder, run_name)
            os.makedirs(run_dir, exist_ok=True)
            train(agentType, task, config, run_dir)
    
    # Short scope
    filename = f'{prefix}{agentType}{nr_runs}{task.upper()}{short_res}.txt'
    filename = os.path.join(p_folder, filename)

    _scope_evaluator(agent, agentType, task, nr_runs, short_eps, ep_length, short_res, long_res, filename, w_folder, config)

    # Long scope
    filename = f'{prefix}{agentType}{nr_runs}{task.upper()}{long_res}.txt'
    filename = os.path.join(p_folder, filename)

    _scope_evaluator(agent, agentType, task, nr_runs, long_eps, ep_length, long_res, long_res, filename, w_folder, config)


# Checks if runs have the necessary amount of steps (not terminated)
def _has_weights(agentType: str, task: str, run_dir: str, ep_num: int, resolution: str) -> bool:
    step = resolution * 10

    for episode in range(ep_num):
        ep_name = f'EP{episode + 1}'
        ep_dir = os.path.join(run_dir, ep_name)

        actor_path = os.path.join(
            ep_dir,
            f"{task}{resolution}{agentType}STEP_{step}.pt_actor.pth"
        )

        if not os.path.isfile(actor_path):
            print(f"Missing [{task}{resolution}{agentType}STEP_{step}.pt_actor.pth] actor file at step 2500 in {ep_name}")
            return False
        
    return True

# Evaluates the agent in both the short scope and the long scope
def _scope_evaluator(agent: SACAgent | REDQSACAgent, agentType: str, task: str, nr_runs: int, ep_num: int,
                      ep_length: int, res: int, long_res:int, filename: str, folder: str, config: Config):
    with open(filename, 'a') as f:

        for run_nr in range(nr_runs):
            runreward_list = []
            run_name = f'RUN{run_nr+1}'
            run_dir = os.path.join(folder, run_name)

            if not _has_weights(agentType, task, run_dir, res, long_res):
                print(f"Skipping run {run_name}: Missing weights.")
                continue

            for episode in range(ep_num):
                ep_name = f'EP{episode+1}'
                ep_dir = os.path.join(run_dir, ep_name)

                for step in range(res, int(ep_length*100) + 1, res):
                    actor_path, critic_path = _choose_paths(agentType, agent, task, ep_dir, res, step)

                    agent.load_weights(actor_path, critic_path)
                    step_reward = evaluate(agent, task, None, config, plot = True)
                    runreward_list.append(step_reward.tolist())
                    print(f"Run: {run_nr + 1}, Episode: {episode + 1}, Step: {step}, Reward: {step_reward}")

            f.write(f"{runreward_list}\n")


def _choose_paths(agentType: str, agent: SACAgent | REDQSACAgent, task: str, ep_dir: str, res: int, step: int) -> tuple[str, str]:
    actor_path = os.path.join(ep_dir, f"{task}{res}{agentType}STEP_{step}.pt_actor.pth")

    if "RED" in agentType:
        critic_path = []

        for i in range(agent.nr_critics):
            c_path = os.path.join(ep_dir, f"{task}{res}{agentType}STEP_{step}.pt_critic{i+1}.pth")
            critic_path.append(c_path)

    else:
        critic_path = os.path.join(ep_dir, f"{task}{res}{agentType}STEP_{step}.pt_critic.pth")

    return actor_path, critic_path

# Loads the evaluation phase specific parameters for the evaluation function to use
def _runLoader(config: Config) -> tuple[int, int, int, int, int, str, str]:

    short_eps = config.phases['run'].ep_num[0]
    long_eps = config.phases['run'].ep_num[1]
    short_res = config.phases['run'].resolution[0]
    long_res = config.phases['run'].resolution[1]
    ep_length = config.phases['run'].ep_length
    p_folder = config.phases['run'].save_dir
    w_folder = config.phases['train'].save_dir

    return short_eps, long_eps, short_res, long_res, ep_length, w_folder, p_folder

