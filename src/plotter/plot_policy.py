import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Plot the policy over the relevant states for the pitch, roll and pitchroll tasks
def plot_policy(agent, task: str, error_list: list = None, rate_list: list = None) -> None:

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