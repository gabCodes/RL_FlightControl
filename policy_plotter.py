import numpy as np
import matplotlib.pyplot as plt
from citation import initialize, step, terminate
from matplotlib.colors import Normalize
import numpy as np
import torch
from sac_torch import SACAgent
from redq_sac_torch import REDQSACAgent

def train_pitch_withstates(agent: SACAgent | REDQSACAgent, ep_length: int) -> tuple[list, list]:
    done = False
    terminated = False
    count = 0
    timestep = 0
    ref_function = generate_ref(ep_length)
    error_list = []
    pitch_list = []

    initialize()

    for _ in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

    state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    while not (done or terminated):
        pitch_ref = ref_function(timestep)
        _, action, _ = agent.actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]
        output = step([action.item(),0,0,0,0,0,0,0,1449.775,1449.775])
        next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[7]), np.rad2deg(output[1])]).unsqueeze(0)
        reward = -1*np.abs(next_state[0][0])
        if count % 1 == 0:
            error_list.append(next_state[0,0].item())
            pitch_list.append(next_state[0,1].item())


        if torch.isnan(next_state).any():
            terminated = True
            break
        
        agent.replay_buffer.add(state_tensor, action, next_state, reward, done)

        if agent.replay_buffer.size > agent.batch_size:
            _, _, _, _, _ = agent.update()
                
        state_tensor = next_state
        count += 1
        timestep += 0.01
        done = timestep >= ep_length

    terminate()

    return error_list, pitch_list

def train_roll_withstates(agent: SACAgent | REDQSACAgent, ep_length: int) -> tuple[list, list]:
    done = False
    terminated = False
    count = 0
    timestep = 0
    ref_function = generate_ref(ep_length)
    error_list = []
    roll_list = []

    initialize()

    for _ in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

    state = [np.rad2deg(output[6]), np.rad2deg(output[0])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    while not (done or terminated):
        roll_ref = ref_function(timestep)
        _, action, _ = agent.actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]
        output = step([-0.025,action.item(),0,0,0,0,0,0,1449.775,1449.775])
        next_state = torch.FloatTensor([np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = -1*np.abs(next_state[0][0])

        if count % 1 == 0:
            error_list.append(next_state[0,0].item())
            roll_list.append(next_state[0,1].item())

        if torch.isnan(next_state).any():
            terminated = True
            break
        
        agent.replay_buffer.add(state_tensor, action, next_state, reward, done)

        if agent.replay_buffer.size > agent.batch_size:
            _, _, _, _, _ = agent.update()

        state_tensor = next_state
        count += 1
        timestep += 0.01
        done = timestep >= ep_length

    terminate()

    return error_list, roll_list

def train_pitchroll_withstates(agent: SACAgent | REDQSACAgent, ep_length: int) -> None:
            done = False
            terminated = False
            count = 1
            timestep = 0
            ref_function = generate_ref(25)
            ref_function2 = generate_ref(25, offset = 0)

            initialize()

            for t in range(2000):
                output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

            state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1]), np.rad2deg(output[6]), np.rad2deg(output[0])]
            state_tensor = torch.FloatTensor(state).unsqueeze(0)


            while not (done or terminated):
                # Define references
                pitch_ref = ref_function(timestep)
                roll_ref = ref_function2(timestep)

                # Choose action
                _, action, _ = agent.actor.sample(state_tensor)
                action = action.detach().cpu().numpy()[0]
                pitch_action = action[0].item()
                roll_action = action[1].item()

                # Input to model
                output = step([pitch_action,roll_action,0,0,0,0,0,0,1449.775,1449.775])
                next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[7]), np.rad2deg(output[1]), 
                                                np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
                reward = -0.6*np.abs(next_state[0][0]) - 0.4*np.abs(next_state[0][2])

                if torch.isnan(next_state).any():
                    terminated = True
                    break
                
                agent.replay_buffer.add(state_tensor, action, next_state, reward, done)

                if agent.replay_buffer.size > agent.batch_size:
                    _, _, _, _, _ = agent.update()
                        
                state_tensor = next_state
                timestep += 0.01
                count += 1
                done = timestep >= ep_length

            terminate()

def plot_policy(agent: SACAgent | REDQSACAgent, task: str, error_list: list = None, rate_list: list = None) -> None:

    TASK_HANDLERS = {
    "pitch": _compute_pitch_actions,
    "roll":  _compute_roll_actions,
    "pitchroll": _compute_pitchroll_actions
    }

    TASK_LABELS = {
    "pitch": [r'$\theta$ error (deg)', 'q (deg/s)', 'Pitch mean action',
              r'$\theta$ error (deg)', 'q (deg/s)', 'Pitch stochastic action'],
    "roll":  [r'$\phi$ error (deg)', 'p (deg/s)', 'Roll mean action',
              r'$\phi$ error (deg)', 'p (deg/s)', 'Roll stochastic action'],
    "pitchroll": [r'$\theta$ error (deg)', 'q (deg/s)', 'Pitch mean action',
                  r'$\phi$ error (deg)', 'p (deg/s)', 'Roll mean action']
    } 

    handler = TASK_HANDLERS.get(task)
    labels = TASK_LABELS.get(task)

    if not handler or not labels:
        raise ValueError(f"Unknown task: {task}")

    error_vals = np.linspace(-30, 30, 400)
    rate_vals = np.linspace(-30, 30, 400)
    error, rate = np.meshgrid(error_vals, rate_vals)
    actions = np.zeros_like(error)
    actions2 = np.zeros_like(error)

    # Compute the action for each point in the state space
    for i in range(error.shape[0]):
        for j in range(error.shape[1]):
            actions[i, j], actions2[i, j] = handler(agent, error[i, j], rate[i, j])

    # Create figure with three subplots
    fig, axs = plt.subplots(1, 2, figsize=(30, 6), constrained_layout=True)

    # First subplot: plot for deterministic actions
    c1 = axs[0].contourf(error, rate, actions, 20, cmap='coolwarm', norm=Normalize(vmin=np.min(actions), vmax=np.max(actions)))
    fig.colorbar(c1, ax=axs[0], label='Policy action (deg)')
    axs[0].set_xlabel(labels[0])
    axs[0].set_ylabel(labels[1])
    axs[0].set_title(labels[2])

    # Second subplot: plot for stochastic actions or for other state (depending on task)
    c2 = axs[1].contourf(error, rate, actions2, 20, cmap='coolwarm', norm=Normalize(vmin=np.min(actions2), vmax=np.max(actions2)))
    fig.colorbar(c2, ax=axs[1], label='Policy action (deg)')
    axs[1].set_xlabel(labels[3])
    axs[1].set_ylabel(labels[4])
    axs[1].set_title(labels[5])

    if task != 'pitchroll' and error_list and rate_list:
        # Third plot: scatter plot of trajectories 
        axs[2].scatter(error_list, rate_list, s=10, alpha=0.7, color='green')
        axs[2].set_xlim(-30, 30)
        axs[2].set_ylim(-30, 30)
        axs[2].set_xlabel(labels[0])
        axs[2].set_ylabel(labels[1])
        axs[2].set_title('Training trajectory spread')

    # Show the plots
    plt.show()

def _compute_pitch_actions(agent, error, rate):
    state = torch.FloatTensor([error, rate, 0, 0]).unsqueeze(0)
    action, action2, _ = agent.actor.sample(state)

    return np.rad2deg(action[0][0].item()), np.rad2deg(action2[0][0].item())

def _compute_roll_actions(agent, error, rate):
    state = torch.FloatTensor([0, 0, error, rate]).unsqueeze(0)
    action, action2, _ = agent.actor.sample(state)

    return np.rad2deg(action[0][1].item()), np.rad2deg(action2[0][1].item())

def _compute_pitchroll_actions(agent, error, rate):
        state = torch.FloatTensor([error, rate, 0, 0]).unsqueeze(0)
        state2 = torch.FloatTensor([0, 0, error, rate]).unsqueeze(0)
        action, _, _  = agent.actor.sample(state)
        action2, _, _ = agent.actor.sample(state2)

        return np.rad2deg(action[0][0].item()), np.rad2deg(action2[0][1].item())

if __name__ == "__main__":

    NR_CRITICS = 3
    UTD = 5
    DROPOUT = 0
    GAMMA	= 0.989
    TAU	= 0.07645
    LR	= 0.00063
    BATCH_SIZE	= 64
    BUFFER_SIZE = 50000
    max_action = 0.26

    # Pitch or roll
    # lam_s, lam_t = 5, 15

    # Pitch and roll
    lam_s, lam_t = 30, 60

    agent = SACAgent(
                state_dim=4,
                action_dim=2,
                max_action=max_action,
                gamma=GAMMA,
                tau=TAU,
                lr=LR,
                batch_size=BATCH_SIZE,
                buffer_size = BUFFER_SIZE,
                lam_s = lam_s,
                lam_t = lam_t
                )

    # Choose task: 'pitch', 'roll' or 'pitchroll'
    task = 'pitchroll'
    for episode in range(20):
        train_pitchroll_withstates(agent, ep_length=25)
        plot_policy(agent, task)

