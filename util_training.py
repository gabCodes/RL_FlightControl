import numpy as np
import torch
import os
from handlers import PitchHandler, RollHandler, PitchRollHandler
from faults import QuarterEfficiencyFault, JoltFault
from agents import SACAgent, REDQSACAgent
from plotter import plot_states
from citation import initialize, step, terminate

# Check if short training cycle has all 2500 weights
def sac_has_2500_weights(task: str, run_dir: str, ep_num: int, resolution: str) -> bool:
    step = 2500

    for episode in range(ep_num):
        ep_name = f'EP{episode + 1}'
        ep_dir = os.path.join(run_dir, ep_name)

        actor_path = os.path.join(
            ep_dir,
            f"{task}{resolution}SACSTEP_{step}.pt_actor.pth"
        )

        if not os.path.isfile(actor_path):
            print(f"Missing [{task}{resolution}SACSTEP_{step}.pt_actor.pth] actor file at step 2500 in {ep_name}")
            return False
        
    return True

# Check if short training cycle has all 2500 weights for REDQ
def redq_has_2500_weights(task: str, run_dir: str, ep_num: int, resolution: str, NR_CRITICS: int, UTD: int) -> bool:
    step = 2500

    for episode in range(ep_num):
        ep_name = f'EP{episode + 1}'
        ep_dir = os.path.join(run_dir, ep_name)

        actor_path = os.path.join(
            ep_dir,
            f"{task}{resolution}RED{NR_CRITICS}Q{UTD}STEP_{step}.pt_actor.pth"
        )
        if not os.path.isfile(actor_path):
            print(f"Missing: [{task}{resolution}RED{NR_CRITICS}Q{UTD}STEP_{step}.pt_actor.pth]")
            return False

    return True

# Train agents based on task
def train(agent: SACAgent | REDQSACAgent, task: str, ep_num: int, resolution: int, run_dir: str, ep_length: int) -> None:

    algo = agent.type

    handler = _choose_handler(agent, task, ep_length)

    for episode in range(ep_num):
        ep_name = f'EP{episode+1}'
        ep_dir = os.path.join(run_dir, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        done = False
        terminated = False
        count = 1
        timestep = 0

        initialize()

        for _ in range(2000):
            output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

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
            
            if episode + 1 <= 5 and count % 50 == 0:
                save_path = os.path.join(ep_dir, f"{task}50{algo}STEP_{count}.pt")
                agent.save_weights(save_path)
                print(f"Saved agent at step {count} to {save_path}")

            if count % resolution == 0:
                save_path = os.path.join(ep_dir, f"{task}{resolution}{algo}STEP_{count}.pt")
                agent.save_weights(save_path)
                print(f"Saved agent at step {count} to {save_path}")

            state_tensor = next_state
            timestep += 0.01
            count += 1
            done = timestep >= ep_length

        terminate()

# Evaluate agents based on task
def evaluate(agent: SACAgent | REDQSACAgent,  task: str, fault: str = None, 
             ep_length: int = 20, plot: bool = False, allstates: bool = False) -> float:

    handler = _choose_handler(agent, task, ep_length, fault)
    
    state_list, action_list, actuator_list, ref_list, time_list = [], [], [], [], []

    ep_reward = 0
    timestep = 0
    done = False
    terminated = False

    initialize()

    for _ in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

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

# Train 30 SAC agents and save their weights
def sac_30_runs(task: str, resolution: int, ep_num: int, training: bool = False) -> None:
    GAMMA	= 0.989
    TAU	= 0.07645
    LR	= 0.00063
    BATCH_SIZE	= 64
    BUFFER_SIZE = 50000
    max_action = 0.26
    nr_runs = 30
    ep_length = 25
    caps_s = 5
    caps_t = 15

    if "PITCHROLL" in task:
        caps_s = 30
        caps_t = 60

    for run_nr in range(nr_runs):
        if "PITCHROLL" in task:
            agent = SACAgent(
                    state_dim=4,
                    action_dim=2,
                    max_action=max_action,
                    gamma=GAMMA,
                    tau=TAU,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    buffer_size = BUFFER_SIZE,
                    lam_s = caps_s,
                    lam_t = caps_t
                    )
            
        else:
            agent = SACAgent(
            state_dim=2,
            action_dim=1,
            max_action=max_action,
            gamma=GAMMA,
            tau=TAU,
            lr=LR,
            batch_size=BATCH_SIZE,
            buffer_size = BUFFER_SIZE,
            lam_s = 5,
            lam_t = 15
            )
        
        if training == True:
            run_name = f'RUN{run_nr+1}'
            run_dir = os.path.join("checkpoints", run_name)
            os.makedirs(run_dir, exist_ok=True)
            train(agent, task, ep_num, resolution, run_dir, ep_length=ep_length)
    
    with open(f'SAC30{task}{resolution}.txt', 'a') as f:

        for run_nr in range(nr_runs):
            runreward_list = []
            run_name = f'RUN{run_nr+1}'
            run_dir = os.path.join("checkpoints", run_name)

            if not sac_has_2500_weights(task, run_dir, ep_num, resolution):
                print(f"Skipping run {run_name}: Missing step-2500 weights.")
                continue

            for episode in range(ep_num):
                ep_name = f'EP{episode+1}'
                ep_dir = os.path.join(run_dir, ep_name)

                for step in range(resolution, int(ep_length*100) + 1, resolution):
                    actor_path = os.path.join(ep_dir, f"{task}{resolution}SACSTEP_{step}.pt_actor.pth")
                    critic_path = os.path.join(ep_dir, f"{task}{resolution}SACSTEP_{step}.pt_critic.pth")
                    agent.load_weights(actor_path, critic_path)
                    step_reward = evaluate(agent, task)
                    runreward_list.append(step_reward.tolist())
                    print(f"Run: {run_nr + 1}, Episode: {episode + 1}, Step: {step}, Reward: {step_reward}")

            f.write(f"{runreward_list}\n")

# Train 30 REDQ agents and save their weights
def redq_30_runs(task: str, resolution: int, ep_num: int, training: bool = False, q_nr: int = 3) -> None:

    GAMMA	= 0.989
    TAU	= 0.07645
    LR	= 0.00063
    BATCH_SIZE	= 64
    BUFFER_SIZE = 50000
    NR_CRITICS = q_nr
    UTD = q_nr
    DROPOUT = 0 
    max_action = 0.26
    nr_runs = 30
    ep_length = 25
    caps_s = 5
    caps_t = 15


    if "PITCHROLL" in task:
        caps_s = 30
        caps_t = 60

    for run_nr in range(nr_runs):
        if "PITCHROLL" in task:
            agent = REDQSACAgent(
                    state_dim=4,
                    action_dim=2,
                    max_action=max_action,
                    gamma=GAMMA,
                    tau=TAU,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    buffer_size=BUFFER_SIZE,
                    nr_critics=NR_CRITICS,
                    utd=UTD,
                    dropout=DROPOUT,
                    lam_s=caps_s,
                    lam_t=caps_t
                    )
            
        else:
            agent = REDQSACAgent(
                    state_dim=2,
                    action_dim=1,
                    max_action=max_action,
                    gamma=GAMMA,
                    tau=TAU,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    buffer_size=BUFFER_SIZE,
                    nr_critics=NR_CRITICS,
                    utd=UTD,
                    dropout=DROPOUT,
                    lam_s=caps_s,
                    lam_t=caps_t
                    )
        
        if training == True:
            run_name = f'RUN{run_nr+1}'
            run_dir = os.path.join("checkpoints", run_name)
            os.makedirs(run_dir, exist_ok=True)
            train(agent, task, ep_num, resolution, run_dir, ep_length=ep_length)
    
    with open(f'RED{q_nr}Q30{task}{resolution}.txt', 'a') as f:

        for run_nr in range(nr_runs):
            runreward_list = []
            run_name = f'RUN{run_nr+1}'
            run_dir = os.path.join("checkpoints", run_name)

            if not redq_has_2500_weights(task,run_dir, ep_num, resolution, NR_CRITICS, UTD):
                print(f"Skipping run {run_name}: Missing step_2500 weights.")
                continue

            for episode in range(ep_num):
                ep_name = f'EP{episode+1}'
                ep_dir = os.path.join(run_dir, ep_name)

                for step in range(resolution, int(ep_length*100) + 1, resolution):
                    actor_path = os.path.join(ep_dir, f"{task}{resolution}RED{NR_CRITICS}Q{UTD}STEP_{step}.pt_actor.pth")
                    critics_path = []

                    for i in range(NR_CRITICS):
                        critic_path = os.path.join(ep_dir, f"{task}{resolution}RED{NR_CRITICS}Q{UTD}STEP_{step}.pt_critic{i+1}.pth")
                        critics_path.append(critic_path)

                    agent.load_weights(actor_path, critics_path)
                    step_reward = evaluate(agent, 'PITCHROLL', fault = "jolt", plot = True, allstates=True)
                    runreward_list.append(step_reward)
                    print(f"Run: {run_nr + 1}, Episode: {episode + 1}, Step: {step}, Reward: {step_reward}")

            f.write(f"{runreward_list}\n")

def _choose_handler(agent: SACAgent | REDQSACAgent, task: str, ep_length: int, fault = None):
    handler = None

    mapping = {
        "pitch": PitchHandler,
        "roll": RollHandler,
        "pitchroll": PitchRollHandler
    }

    handler = mapping[task.lower()](agent, ep_length)

    if fault:
        mapping = {
            "eff": QuarterEfficiencyFault,
            "jolt": JoltFault
        }
        handler = mapping[fault.lower()](handler)

    return handler

if __name__ == "__main__":
    # The task names must have either PITCH, ROLL or PITCHROLL in them
    # When training=True, the agents will be trained. Set to False to only evaluate already trained agents
    
    #redq_30_runs("v4ROLL", 250, 10, training=True, q_nr=5)
    redq_30_runs("PITCHROLL", 50, 5, training=False, q_nr=5)

    # redq_30_runs("v2ROLL", 250, 10, training=True, q_nr=5)
    # redq_30_runs("v3ROLL", 250, 10, training=True, q_nr=3)

    # redq_30_runs("v2PITCHROLL", 250, 10, training=True, q_nr=5)
    # redq_30_runs("v3PITCHROLL", 250, 10, training=True, q_nr=3)