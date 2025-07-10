import numpy as np
import matplotlib.pyplot as plt
from citation import initialize, step, terminate
from matplotlib.colors import Normalize
from util_training import *
import numpy as np
import torch
from sac_torch import SACAgent
from redq_sac_torch import REDQSACAgent

def train_pitch_withstates(agent, ep_length, plot=False):
    done = False
    terminated = False
    count = 0
    timestep = 0
    ref_function = generate_ref(ep_length)
    error_list = []
    pitch_list = []
    initialize()
    for t in range(2000):
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

def train_roll_withstates(agent, ep_length, plot=False):
    done = False
    terminated = False
    count = 0
    timestep = 0
    ref_function = generate_ref(ep_length)
    error_list = []
    roll_list = []
    initialize()
    for t in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])
    state = [np.rad2deg(output[6]), np.rad2deg(output[0])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)


    while not (done or terminated):
        pitch_ref = ref_function(timestep)
        # pitch_ref = piecewise_ref(timestep)
        _, action, _ = agent.actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]
        output = step([-0.025,action.item(),0,0,0,0,0,0,1449.775,1449.775])
        next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
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

NR_CRITICS = 3
UTD = 5
DROPOUT = 0
GAMMA	= 0.989
TAU	= 0.07645
LR	= 0.00063
BATCH_SIZE	= 64
BUFFER_SIZE = 50000
max_action = 0.26
reward_list = []

agent = SACAgent(
            state_dim=4,
            action_dim=2,
            max_action=max_action,
            gamma=GAMMA,
            tau=TAU,
            lr=LR,
            batch_size=BATCH_SIZE,
            buffer_size = BUFFER_SIZE,
            lam_s = 30,
            lam_t = 60
            )

for episode in range(20):
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
            done = timestep >= 25

        terminate()

# agent = REDQSACAgent(
#         state_dim=2,
#         action_dim=1,
#         max_action=max_action,
#         gamma=GAMMA,
#         tau=TAU,
#         lr=LR,
#         batch_size=BATCH_SIZE,
#         buffer_size=BUFFER_SIZE,
#         nr_critics=NR_CRITICS,
#         utd=UTD,
#         dropout=DROPOUT,
#         lam_s=5,
#         lam_t=15
#         )
    
# num_episodes = 20
# ep_length = 25
# error_list = []
# rate_list = []
# for episode in range(num_episodes):
#     a,b = train_roll_withstates(agent, ep_length=ep_length)
#     error_list.extend(a)
#     rate_list.extend(b)
#Error is reference minus theta
error_vals = np.linspace(-30, 30, 400)
rate_vals = np.linspace(-30, 30, 400)
error, rate = np.meshgrid(error_vals, rate_vals)
actions = np.zeros_like(error)
actions2 = np.zeros_like(error)

# Compute the action for each point in the state space
for i in range(error.shape[0]):
    for j in range(error.shape[1]):
        state = torch.FloatTensor([error[i, j], rate[i, j], 0, 0]).unsqueeze(0)
        state2 = torch.FloatTensor([0, 0, error[i, j], rate[i, j]]).unsqueeze(0)
        action, _, _  = agent.actor.sample(state)
        action2, _, _ = agent.actor.sample(state2)
        pitch_action = action[0][0].item()
        roll_action = action2[0][1].item()
        actions[i, j] = np.rad2deg(pitch_action)
        actions2[i, j] = np.rad2deg(roll_action)

# Create figure with three subplots
fig, axs = plt.subplots(1, 2, figsize=(30, 6), constrained_layout=True)

# First subplot: plot for deterministic actions
c1 = axs[0].contourf(error, rate, actions, 20, cmap='coolwarm', norm=Normalize(vmin=np.min(actions), vmax=np.max(actions)))
fig.colorbar(c1, ax=axs[0], label='Policy action (deg)')
axs[0].set_xlabel(r'$\theta$ error (deg)')
axs[0].set_ylabel('q (deg/s)')
axs[0].set_title('Pitch mean action')

# Second subplot: plot for stochastic actions
c2 = axs[1].contourf(error, rate, actions2, 20, cmap='coolwarm', norm=Normalize(vmin=np.min(actions2), vmax=np.max(actions2)))
fig.colorbar(c2, ax=axs[1], label='Policy action (deg)')
axs[1].set_xlabel(r'$\phi$ error (deg)')
axs[1].set_ylabel('p (deg/s)')
axs[1].set_title('Roll mean action')

# # Third plot: scatter plot of trajectories 
# axs[2].scatter(error_list, rate_list, s=10, alpha=0.7, color='green')
# axs[2].set_xlim(-30, 30)
# axs[2].set_ylim(-30, 30)
# axs[2].set_xlabel(r'$\phi$ error (deg)')
# axs[2].set_ylabel('p (deg/s)')
# axs[2].set_title('Training trajectory spread')

# Show the plots
plt.show()

