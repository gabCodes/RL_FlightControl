import numpy as np
import torch
import os
from sac_torch import SACAgent
from redq_sac_torch import REDQSACAgent
from plot_generation import state_plotter
from scipy.signal import butter, filtfilt
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

# Applying low pass filter to ensure aircraft is able to follow
def low_pass_filter(data: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff, btype='low', analog=False)

    return filtfilt(b, a, data)

# Randomly generating the references signals
def generate_ref(duration: float, max_amp: float = 0.26, num_terms: int = 4, dt: float = 0.01, offset: float = 0.032) -> callable:
    fs = 1 / dt  # Sampling frequency
    t = np.arange(0, duration, dt)

    # divide line into segments so they add up to max amp
    points = np.sort(np.random.uniform(0, max_amp, num_terms - 1))
    points = np.concatenate(([0], points, [max_amp]))
    amps = np.diff(points)

    # randomly assign signs
    sign_mask = np.random.choice([-1, 1], size=len(amps))
    amps = amps * sign_mask
    freqs = np.random.uniform(0.05, 0.2, num_terms)

    # raw reference signal
    raw_signal = np.dot(amps, np.sin(np.outer(freqs, t))) + offset

    f_max = 0.3
    filtered_signal = low_pass_filter(raw_signal, f_max)

    # precompute time and values for fast lookup
    time_values = t
    signal_values = filtered_signal

    def ref_function(t_query):
        # function lookup table
        idx = np.searchsorted(time_values, t_query, side='left')

        if idx >= len(signal_values):
            return signal_values[-1]  # return last value if index bigger than signal length
        
        return signal_values[idx]

    return ref_function

# Train SAC agent for pitch
def train_pitch(agent: SACAgent | REDQSACAgent, task: str, ep_num: int, resolution: int, run_dir: str, ep_length: int, SAC: bool = True) -> None:
    if SAC == True:
        algo = "SAC"
    
    else:
        UTD = agent.fetchutd()
        NR_CRITICS = agent.fetch_nrcritics()
        algo = f"RED{NR_CRITICS}Q{UTD}"

    for episode in range(ep_num):
        ep_name = f'EP{episode+1}'
        ep_dir = os.path.join(run_dir, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        done = False
        terminated = False
        count = 1
        timestep = 0
        ref_function = generate_ref(ep_length)
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

            if torch.isnan(next_state).any():
                terminated = True
                break
            
            agent.replay_buffer.add(state_tensor, action, next_state, reward, done)

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

# Train SAC agent for roll
def train_roll(agent: SACAgent | REDQSACAgent, task: str, ep_num: int, resolution: int, run_dir: str, ep_length: int, SAC: bool = True) -> None:
    if SAC == True:
        algo = "SAC"
    
    else:
        UTD = agent.fetchutd()
        NR_CRITICS = agent.fetch_nrcritics()
        algo = f"RED{NR_CRITICS}Q{UTD}"

    for episode in range(ep_num):
        ep_name = f'EP{episode+1}'
        ep_dir = os.path.join(run_dir, ep_name)
        os.makedirs(ep_dir, exist_ok=True)

        done = False
        terminated = False
        count = 1
        timestep = 0
        ref_function = generate_ref(ep_length, offset=0)

        initialize()

        for t in range(2000):
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

            if torch.isnan(next_state).any():
                terminated = True
                break
            
            agent.replay_buffer.add(state_tensor, action, next_state, reward, done)

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

# Train SAC agent for pitch and roll
def train_pitchroll(agent: SACAgent | REDQSACAgent, task: str, ep_num: int, resolution: int, run_dir: str, ep_length: float, SAC: bool = True) -> None:
    if SAC == True:
        algo = "SAC"
    
    else:
        UTD = agent.fetchutd()
        NR_CRITICS = agent.fetch_nrcritics()
        algo = f"RED{NR_CRITICS}Q{UTD}"
    
    for episode in range(ep_num):
        ep_name = f'EP{episode+1}'
        ep_dir = os.path.join(run_dir, ep_name)
        os.makedirs(ep_dir, exist_ok=True)
        done = False
        terminated = False
        count = 1
        timestep = 0
        ref_function = generate_ref(ep_length)
        ref_function2 = generate_ref(ep_length, offset = 0)

        initialize()

        for t in range(2000):
            output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

        state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1]), np.rad2deg(output[6]), np.rad2deg(output[0])]
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        while not (done or terminated):
            #Define references
            pitch_ref = ref_function(timestep)
            roll_ref = ref_function2(timestep)

            #Choose action
            _, action, _ = agent.actor.sample(state_tensor)
            action = action.detach().cpu().numpy()[0]
            pitch_action = action[0].item()
            roll_action = action[1].item()

            #Input to model
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

# Evaluate SAC agent for pitch
def evaluate_pitch(agent: SACAgent | REDQSACAgent, plot: bool = False, allstates: bool = False,
                    ep_length: int = 20, eff: float =1.0, fault: bool = False) -> tuple[float, bool, float]:
    
    def get_piecewise_ref():
        def piecewise_ref(t):
            return 0.12 * np.sin(0.4 * t) + 0.032
        return piecewise_ref
    
    action_list = []
    actuator_list = []
    pitch_list = []
    ref_list = []
    time_list = []
    state_list = []
    ep_reward = 0
    timestep = 0
    done = False
    terminated = False
    if eff < 1.0:
        ref = generate_ref(ep_length)

    else:
        ref = get_piecewise_ref()

    initialize()

    for t in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

    state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)


    while not (done or terminated):
        pitch_ref = ref(timestep)
        action, _, _ = agent.actor.sample(state_tensor) #Deterministic action
        action = action.detach().cpu().numpy()[0]
        output = step([action.item()*eff,0,0,0,0,0,0,0,1449.775,1449.775])
        next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[7]), np.rad2deg(output[1])]).unsqueeze(0)
        reward = -1*np.abs(next_state[0][0])

        if torch.isnan(next_state).any():
            terminated = True
            break
            
        pitch_list.append(np.rad2deg(output[7]))
        state_list.append([np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
                          np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
                          np.rad2deg(output[8]),output[9]])
        action_list.append([np.rad2deg(action.item()),0])

        if fault == True:
            actuator_list.append(np.rad2deg(action.item()*eff))
        ref_list.append(np.rad2deg(pitch_ref))
        time_list.append(timestep)
        
        state_tensor = next_state
        ep_reward += reward
        timestep += 0.01

        done = timestep > ep_length

    terminate()

    state_names = ['p (deg/s)','q (deg/s)','r (deg/s)','$V_{TAS}$ (m/s)','$\\alpha$ (deg)','$\\beta$ (deg)','$\\phi$ (deg)',
        '$\\theta$ (deg)', '$\\psi$ (deg)', '$h_e$ (m)']
    
    if plot == True:
        if allstates == True:
            if fault == False:
                state_plotter("pitch", time_list, ref_list, state_list, action_list, state_names, zoom=False)
            else:
                state_plotter("pitch", time_list, ref_list, state_list, action_list, state_names, zoom=False, extra=actuator_list)
        else:
            if fault == False:
                state_plotter("pitch", time_list, ref_list, state_list, action_list, state_names, zoom=True)
            else:
                state_plotter("pitch", time_list, ref_list, state_list, action_list, state_names, zoom=False, extra=actuator_list)

    roughness_factor = np.sum(np.abs(action_list))

    return ep_reward, terminated, roughness_factor

# Evaluate SAC agent for roll
def evaluate_roll(agent: SACAgent | REDQSACAgent, plot: bool = False, allstates: bool = False,
                   ep_length: float = 20, eff: float = 1.0, fault: bool = False) -> tuple[float, bool, float]:
    
    def get_piecewise_ref():
        def piecewise_ref(t):
            return 0.12 * np.sin(0.4 * t) + 0.032
        return piecewise_ref
    
    action_list = []
    actuator_list = []
    ref_list = []
    time_list = []
    state_list = []
    ep_reward = 0
    timestep = 0
    done = False
    terminated = False

    if eff < 1.0:
        ref = generate_ref(ep_length, offset=0)

    else:
        ref = get_piecewise_ref()

    initialize()

    for t in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

    state = [np.rad2deg(output[6]), np.rad2deg(output[0])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    while not (done or terminated):
        roll_ref = ref(timestep)
        action, _, _ = agent.actor.sample(state_tensor) #Deterministic action
        action = action.detach().cpu().numpy()[0]
        output = step([-0.025,action.item()*eff,0,0,0,0,0,0,1449.775,1449.775])
        next_state = torch.FloatTensor([np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = -1*np.abs(next_state[0][0])

        if torch.isnan(next_state).any():
            terminated = True
            break

        state_list.append([np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
                          np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
                          np.rad2deg(output[8]),output[9]])
        action_list.append([0,np.rad2deg(action.item())])

        if fault == True:
            actuator_list.append(np.rad2deg(action.item()*eff))
        ref_list.append(np.rad2deg(roll_ref))
        time_list.append(timestep)
        
        state_tensor = next_state
        ep_reward += reward
        timestep += 0.01

        done = timestep > ep_length

    terminate()

    state_names = ['p (deg/s)','q (deg/s)','r (deg/s)','$V_{TAS}$ (m/s)','$\\alpha$ (deg)','$\\beta$ (deg)','$\\phi$ (deg)',
        '$\\theta$ (deg)', '$\\psi$ (deg)', '$h_e$ (m)']
    
    if plot == True:
        if allstates == True:
            if fault == False:
                state_plotter("roll", time_list, ref_list, state_list, action_list, state_names, zoom=False)
            else:
                state_plotter("roll", time_list, ref_list, state_list, action_list, state_names, zoom=False, extra=actuator_list)
        else:
            if fault == False:
                state_plotter("roll", time_list, ref_list, state_list, action_list, state_names, zoom=True)
            else:
                state_plotter("roll", time_list, ref_list, state_list, action_list, state_names, zoom=True)

    roughness_factor = np.sum(np.abs(action_list))

    return ep_reward, terminated, roughness_factor

# Evaluate SAC agent for pitch and roll
def evaluate_pitchroll(agent: SACAgent | REDQSACAgent, plot: bool = False, allstates: bool = False,
                        ep_length: float = 20) -> tuple[float, bool, float]:
    def piecewise_ref(t):
        return 0.12*np.sin(0.4*t) + 0.032
    
    def roll_piecewise_ref(t):
        return 0.12*np.sin(0.4*t)
    
    action_list = []
    ref_list = []
    time_list = []
    state_list = []
    ep_reward = 0
    timestep = 0
    done = False
    terminated = False

    initialize()

    for t in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])

    state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1]), np.rad2deg(output[6]), np.rad2deg(output[0])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    while not (done or terminated):
        pitch_ref = piecewise_ref(timestep)
        roll_ref = roll_piecewise_ref(timestep)

        #Choose action
        action, _, _ = agent.actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]
        pitch_action = action[0].item()
        roll_action = action[1].item()

        #Input to model
        output = step([pitch_action,roll_action,0,0,0,0,0,0,1449.775,1449.775])
        next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[7]), np.rad2deg(output[1]), 
                                        np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = -0.6*np.abs(next_state[0][0]) - 0.4*np.abs(next_state[0][2])

        if torch.isnan(next_state).any():
            terminated = True
            break
            
        state_list.append([np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
                          np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
                          np.rad2deg(output[8]),output[9]])
        action_list.append([np.rad2deg(pitch_action),np.rad2deg(roll_action)])
        ref_list.append([np.rad2deg(pitch_ref), np.rad2deg(roll_ref)])
        time_list.append(timestep)
        
        state_tensor = next_state
        ep_reward += reward
        timestep += 0.01

        done = timestep > ep_length

    terminate()

    state_names = ['p (deg/s)','q (deg/s)','r (deg/s)','$V_{TAS}$ (m/s)','$\\alpha$ (deg)','$\\beta$ (deg)','$\\phi$ (deg)',
        '$\\theta$ (deg)', '$\\psi$ (deg)', '$h_e$ (m)']
    
    if plot == True:
        if allstates == True:
            state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=False)

        else:
            state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=True)

    roughness_factor = np.sum(np.abs(action_list))

    return ep_reward, terminated, roughness_factor

# Decide which task to train based on input
def train(agent: SACAgent | REDQSACAgent, task: str, ep_num: int, resolution: int, run_dir: str, ep_length: float, SAC: bool = True) -> None:
    if "PITCHROLL" in task:
        train_pitchroll(agent, task, ep_num, resolution, run_dir, ep_length, SAC=SAC)
        return

    if "PITCH" in task:
        train_pitch(agent, task, ep_num, resolution, run_dir, ep_length, SAC=SAC)
        return
    
    if "ROLL" in task:
        train_roll(agent, task, ep_num, resolution, run_dir, ep_length, SAC=SAC)
        return

# Decide which task to evaluate based on input
def evaluate_agent(agent: SACAgent | REDQSACAgent, task: str, plot: bool = False, allstates: bool =False) -> float:
    if "PITCHROLL" in task:
        ep_reward, _, _ = evaluate_pitchroll(agent, plot=plot, allstates=allstates)
        return ep_reward

    if "PITCH" in task:
        ep_reward, _, smooth_factor = evaluate_pitch(agent, plot=plot, allstates=allstates)
        return ep_reward
    
    if "ROLL" in task:
        ep_reward, _, _ = evaluate_roll(agent, plot=plot, allstates=allstates)
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
            train(agent, task, ep_num, resolution, run_dir, ep_length=ep_length, SAC=True)
    
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
                    step_reward = evaluate_agent(agent, task)
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
            train(agent, task, ep_num, resolution, run_dir, ep_length=ep_length, SAC=False)
    
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
                    step_reward = evaluate_agent(agent, task)
                    runreward_list.append(step_reward.tolist())
                    print(f"Run: {run_nr + 1}, Episode: {episode + 1}, Step: {step}, Reward: {step_reward}")

            f.write(f"{runreward_list}\n")

if __name__ == "__main__":
    # The task names must have either PITCH, ROLL or PITCHROLL in them
    # When training=True, the agents will be trained. Set to False to only evaluate already trained agents
    
    redq_30_runs("v3PITCH", 250, 10, training=True, q_nr=5)
    redq_30_runs("v3PITCH", 50, 5, training=False, q_nr=5)

    # redq_30_runs("v2ROLL", 250, 10, training=True, q_nr=5)
    # redq_30_runs("v3ROLL", 250, 10, training=True, q_nr=3)

    # redq_30_runs("v2PITCHROLL", 250, 10, training=True, q_nr=5)
    # redq_30_runs("v3PITCHROLL", 250, 10, training=True, q_nr=3)