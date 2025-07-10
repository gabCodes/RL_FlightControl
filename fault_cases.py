from citation import initialize, step, terminate
import numpy as np
import torch
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from redq_sac_torch import REDQSACAgent
from sac_torch import SACAgent
from util_training import *
from plot_generation import state_plotter

# Evaluation under nominal conditions
def evaluate_nominal_pitchroll(agent, plot=False, allstates = False, ep_length = 20, eff=1.0):
    
    def get_p_ref():
        def piecewise_ref(t):
            return 0.12 * np.sin(0.4 * t) + 0.032
        return piecewise_ref
    
    def get_r_ref():
        def piecewise_ref(t):
            return 0.12 * np.sin(0.4 * t) + 0.032
        return piecewise_ref
    
    actuator_list = []
    action_list = []
    pitch_list = []
    ref_list = []
    time_list = []
    state_list = []
    error_list = []
    ep_reward = 0
    timestep = 0
    done = False
    terminated = False
    fault = False
    initialize()
    if eff < 1.0:
        p_ref = generate_ref(ep_length)
        r_ref = generate_ref(ep_length, offset=0)
        fault = True
    else:
        p_ref = get_p_ref()
        r_ref = get_r_ref()

    for t in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])
    state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1]), np.rad2deg(output[6]), np.rad2deg(output[0])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)


    while not (done or terminated):
        pitch_ref = p_ref(timestep)
        roll_ref = r_ref(timestep)

        #Choose action
        action, _, _ = agent.actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]
        pitch_action = action[0].item()
        roll_action = action[1].item()

        #Input to model
        output = step([pitch_action*eff,roll_action*eff,0,0,0,0,0,0,1449.775,1449.775])
        next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[7]), np.rad2deg(output[1]), 
                                        np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = -0.6*np.abs(next_state[0][0]) - 0.4*np.abs(next_state[0][2])

        if torch.isnan(next_state).any():
            terminated = True
            break
        
        error_list.append([np.abs(np.rad2deg(pitch_ref - output[7])), np.abs(np.rad2deg(roll_ref - output[6]))])
        pitch_list.append(np.rad2deg(output[7]))
        state_list.append([np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
                          np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
                          np.rad2deg(output[8]),output[9]])
        action_list.append([np.rad2deg(pitch_action),np.rad2deg(roll_action)])
        if fault == True:
            actuator_list.append([np.rad2deg(pitch_action*eff),np.rad2deg(roll_action*eff)])
        ref_list.append([np.rad2deg(pitch_ref), np.rad2deg(roll_ref)])
        time_list.append(timestep)
        
        state_tensor = next_state
        ep_reward += reward
        timestep += 0.01

        done = timestep > ep_length

    terminate()

    ep_error = np.array(error_list)
    ep_error = np.sum(ep_error, axis=0)/(ep_length*100)

    state_names = ['p (deg/s)','q (deg/s)','r (deg/s)','$V_{TAS}$ (m/s)','$\\alpha$ (deg)','$\\beta$ (deg)','$\\phi$ (deg)',
        '$\\theta$ (deg)', '$\\psi$ (deg)', '$h_e$ (m)']
    if plot == True:
        if allstates == True:
            if fault == True:
                state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=False, extra=actuator_list)
            else:
                state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=False)
        else:
            if fault == True:
                state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=True, extra=actuator_list)
            else:
                state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=True)
    return ep_error, ep_reward, terminated

# Evaluation under jolt actuator condition for biaxial task
def evaluate_jolt_pitchroll(agent, plot=False, allstates = False, ep_length = 20, eff=1.0):
    actuator_list = []
    action_list = []
    pitch_list = []
    ref_list = []
    time_list = []
    state_list = []
    error_list = []
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
        pitch_ref = 0
        roll_ref = 0

        #Choose action
        action, _, _ = agent.actor.sample(state_tensor)
        action = action.detach().cpu().numpy()[0]
        pitch_action = action[0].item()
        roll_action = action[1].item()

        #Input to model
        if 5.0 < timestep < 6.0:
            output = step([0.26,0.26,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append([np.rad2deg(0.26),np.rad2deg(0.26)])
        elif 10.0 < timestep < 11.0:
            output = step([-0.26,-0.26,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append([np.rad2deg(-0.26),np.rad2deg(-0.26)])
        else:
            output = step([pitch_action*eff,roll_action*eff,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append([np.rad2deg(pitch_action*eff) , np.rad2deg(roll_action*eff)])

        next_state = torch.FloatTensor([np.rad2deg(pitch_ref - output[7]), np.rad2deg(output[1]), 
                                        np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = -0.6*np.abs(next_state[0][0]) - 0.4*np.abs(next_state[0][2])

        if torch.isnan(next_state).any():
            terminated = True
            break
        
        error_list.append([np.abs(np.rad2deg(pitch_ref - output[7])), np.abs(np.rad2deg(roll_ref - output[6]))])
        pitch_list.append(np.rad2deg(output[7]))
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

    ep_error = np.array(error_list)
    ep_error = np.sum(ep_error, axis=0)/(ep_length*100)

    state_names = ['p (deg/s)','q (deg/s)','r (deg/s)','$V_{TAS}$ (m/s)','$\\alpha$ (deg)','$\\beta$ (deg)','$\\phi$ (deg)',
        '$\\theta$ (deg)', '$\\psi$ (deg)', '$h_e$ (m)']
    if plot == True:
        if allstates == True:
            state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=False, extra=actuator_list)
        else:
            state_plotter("pitchroll", time_list, ref_list, state_list, action_list, state_names, zoom=True, extra=actuator_list)
    return ep_error, ep_reward, terminated

# Evaluation under jolt actuator condition for pitch task
def evaluate_jolt_pitch(agent, plot=False, allstates=False, ep_length=20, eff=1.0):
        
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
    initialize()
    for t in range(2000):
        output = step([-0.025,0,0,0,0,0,0,0,1449.775,1449.775])
    state = [np.rad2deg(0.032 - output[7]), np.rad2deg(output[1])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)


    while not (done or terminated):
        pitch_ref = 0
        action, _, _ = agent.actor.sample(state_tensor) #Deterministic action
        action = action.detach().cpu().numpy()[0]
        output = step([action.item()*eff,0,0,0,0,0,0,0,1449.775,1449.775])
        if 5.0 < timestep < 6:
            output = step([0.26,0,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append(np.rad2deg(0.26))
        elif 10.0 < timestep < 11.0:
            output = step([-0.26,0,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append(np.rad2deg(-0.26))
        else:
            output = step([action.item()*eff,0,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append(np.rad2deg(action.item()*eff))
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
            state_plotter("pitch", time_list, ref_list, state_list, action_list, state_names, zoom=False, extra=actuator_list)
        else:
            state_plotter("pitch", time_list, ref_list, state_list, action_list, state_names, zoom=True, extra=actuator_list)
    roughness_factor = np.sum(np.abs(action_list))
    return ep_reward, terminated, roughness_factor

# Evaluation under jolt actuator condition for roll task
def evaluate_jolt_roll(agent, plot=False, allstates=False, ep_length=20, eff=1.0):
    actuator_list = []    
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
    state = [np.rad2deg(output[6]), np.rad2deg(output[0])]
    state_tensor = torch.FloatTensor(state).unsqueeze(0)


    while not (done or terminated):
        roll_ref = 0
        action, _, _ = agent.actor.sample(state_tensor) #Deterministic action
        action = action.detach().cpu().numpy()[0]
        if 5.0 < timestep < 6:
            output = step([-0.025,0.26,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append(np.rad2deg(0.26))
        elif 10.0 < timestep < 11.0:
            output = step([-0.025,-0.26,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append(np.rad2deg(-0.26))
        else:
            output = step([-0.025,action.item()*eff,0,0,0,0,0,0,1449.775,1449.775])
            actuator_list.append(np.rad2deg(action.item()*eff))

        next_state = torch.FloatTensor([np.rad2deg(roll_ref - output[6]), np.rad2deg(output[0])]).unsqueeze(0)
        reward = -1*np.abs(next_state[0][0])

        if torch.isnan(next_state).any():
            terminated = True
            break

        state_list.append([np.rad2deg(output[0]),np.rad2deg(output[1]),np.rad2deg(output[2]),output[3],
                          np.rad2deg(output[4]),np.rad2deg(output[5]),np.rad2deg(output[6]),np.rad2deg(output[7]),
                          np.rad2deg(output[8]),output[9]])
        action_list.append([0,np.rad2deg(action.item())])
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
            state_plotter("roll", time_list, ref_list, state_list, action_list, state_names, zoom=False, extra=actuator_list)
        else:
            state_plotter("roll", time_list, ref_list, state_list, action_list, state_names, zoom=True, extra=actuator_list)
    roughness_factor = np.sum(np.abs(action_list))
    return ep_reward, terminated, roughness_factor

# Fault handling logic, points to other methods depending on fault, task and agent architecture
def fault_test(fault, task, resolution, ep_num, plot=False, REDQ=False, Q_nr=3):
    if fault == "nom":
        ep_length = 120
        eff = 1.0
    
    if fault == "eff":
        ep_length = 60
        eff = 0.25
    
    if fault == "jolt":
        ep_length = 20
        eff = 1.0

    GAMMA = 0.989
    TAU = 0.07645
    LR = 0.00063
    BATCH_SIZE = 64
    BUFFER_SIZE = 250000
    error_list = []
    reward_list = []
    fail_list = []
    max_action = 0.26
    nr_runs = 30
    if REDQ == True:
        NR_CRITICS = Q_nr
        UTD = Q_nr
        DROPOUT = 0

        for run_nr in range(nr_runs):
            run_name = f'RUN{run_nr + 1}'
            run_dir = os.path.join("checkpoints", run_name)

            if not redq_has_2500_weights(task, run_dir, ep_num, resolution, NR_CRITICS, UTD):
                print(f"Skipping run {run_name}: Missing step-2500 weights.")
                continue

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
                    lam_s=5,
                    lam_t=15
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
                buffer_size = BUFFER_SIZE,
                nr_critics=NR_CRITICS,
                utd=UTD,
                lam_s = 5,
                lam_t = 15
                )

            step = 2500
            ep_name = f'EP{ep_num}'
            ep_dir = os.path.join(run_dir, ep_name)

            actor_path = os.path.join(
                ep_dir, f"{task}{resolution}RED{NR_CRITICS}Q{UTD}STEP_{step}.pt_actor.pth"
            )
            critics_path = [
                os.path.join(
                    ep_dir, f"{task}{resolution}RED{NR_CRITICS}Q{UTD}STEP_{step}.pt_critic{i + 1}.pth"
                ) for i in range(NR_CRITICS)
            ]

            agent.load_weights(actor_path, critics_path)
            if "PITCHROLL" in task:
                if fault == "jolt":
                    step_error, ep_reward, terminated = evaluate_jolt_pitchroll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                else:
                    step_error, ep_reward, terminated = evaluate_nominal_pitchroll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                fail_list.append(terminated)
                error_list.append(step_error)
                reward_list.append(ep_reward)
            
            elif "PITCH" in task:
                if fault == "jolt":
                    ep_reward, terminated, _ = evaluate_jolt_pitch(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                else:
                    ep_reward, terminated, _ = evaluate_pitch(agent, plot=plot, allstates=True, ep_length = ep_length, eff=eff, fault=eff<1)
                step_error = ep_reward/(ep_length*100)
                fail_list.append(terminated)
                error_list.append(step_error)
                reward_list.append(ep_reward)

            elif "ROLL" in task:
                if fault == "jolt":
                    ep_reward, terminated, _ = evaluate_jolt_roll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                else:
                    ep_reward, terminated, _ = evaluate_roll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff, fault=eff<1)
                step_error = ep_reward/(ep_length*100)
                fail_list.append(terminated)
                error_list.append(step_error)
                reward_list.append(ep_reward)

        error_array = np.array(error_list)
        error = np.mean(error_array, axis=0)
        reward_array = np.array(reward_list)
        reward = np.mean(reward_array,axis=0)
        print("==================================")
        print(f"Fault: {fault}, Task: {task}")
        print(f"REDQ? {REDQ}, NRQ{Q_nr}")
        print("Mean error across columns:", error)
        print("Mean reward:", reward)
        nr_fails = sum(fail_list)
        print("Number of failures:", nr_fails)
        return error_array

    else:
        for run_nr in range(nr_runs):
            run_name = f'RUN{run_nr + 1}'
            run_dir = os.path.join("checkpoints", run_name)
            if not sac_has_2500_weights(task, run_dir, ep_num, resolution):
                print(f"Skipping run {run_name}: Missing step-2500 weights.")
                continue
            
            if "PITCHROLL" in task:
                agent = SACAgent(
                    state_dim=4,
                    action_dim=2,
                    max_action=max_action,
                    gamma=GAMMA,
                    tau=TAU,
                    lr=LR,
                    batch_size=BATCH_SIZE,
                    buffer_size=BUFFER_SIZE,
                    lam_s=5,
                    lam_t=15
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

            step = 2500
            ep_name = f'EP{ep_num}'
            ep_dir = os.path.join(run_dir, ep_name)

            actor_path = os.path.join(
                ep_dir, f"{task}{resolution}SACSTEP_{step}.pt_actor.pth"
            )
            critic_path = os.path.join(
                    ep_dir, f"{task}{resolution}SACSTEP_{step}.pt_critic.pth"
                )
            

            agent.load_weights(actor_path, critic_path)
            if "PITCHROLL" in task:
                if fault == "jolt":
                    step_error, ep_reward, terminated = evaluate_jolt_pitchroll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                else:
                    step_error, ep_reward, terminated = evaluate_nominal_pitchroll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                fail_list.append(terminated)
                error_list.append(step_error)
                reward_list.append(ep_reward)
            
            elif "PITCH" in task:
                if fault == "jolt":
                    ep_reward, terminated, _ = evaluate_jolt_pitch(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                else:
                    ep_reward, terminated, _ = evaluate_pitch(agent, plot=plot, allstates=True, ep_length = ep_length, eff=eff, fault=eff<1)
                step_error = ep_reward/(ep_length*100)
                fail_list.append(terminated)
                error_list.append(step_error)
                reward_list.append(ep_reward)

            elif "ROLL" in task:
                if fault == "jolt":
                    ep_reward, terminated, _ = evaluate_jolt_roll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff)
                else:
                    ep_reward, terminated, _ = evaluate_roll(agent, plot=plot, allstates=True, ep_length=ep_length, eff=eff, fault=eff<1)
                step_error = ep_reward/(ep_length*100)
                fail_list.append(terminated)
                error_list.append(step_error)
                reward_list.append(ep_reward)

        error_array = np.array(error_list)
        error = np.mean(error_array, axis=0)
        reward_array = np.array(reward_list)
        reward = np.mean(reward_array,axis=0)
        print("==================================")
        print(f"Fault: {fault}, Task: {task}")
        print(f"REDQ? {REDQ}, NRQ{Q_nr}")
        print("Mean error across columns:", error)
        print("Mean reward:", reward)
        nr_fails = sum(fail_list)
        print("Number of failures:", nr_fails)
        print("==================================")
        return error_array
    
# PITCH - NOM
err_nom_pitch1 = np.abs(fault_test("nom", "v2PITCH", 250, 10, plot=False))
err_nom_pitch2 = np.abs(fault_test("nom", "v2PITCH", 250, 10, REDQ=True, plot=False))
err_nom_pitch3 = np.abs(fault_test("nom", "v2PITCH", 250, 10, REDQ=True, Q_nr=5, plot=False))

# ROLL - NOM
err_nom_roll1 = np.abs(fault_test("nom", "v2ROLL", 250, 10, plot=False))
err_nom_roll2 = np.abs(fault_test("nom", "v2ROLL", 250, 10, REDQ=True, plot=False))
err_nom_roll3 = np.abs(fault_test("nom", "v2ROLL", 250, 10, REDQ=True, Q_nr=5, plot=False))

# PITCH - EFF
err_eff_pitch1 = np.abs(fault_test("eff", "v2PITCH", 250, 10, plot=True))
err_eff_pitch2 = np.abs(fault_test("eff", "v2PITCH", 250, 10, REDQ=True, plot=False))
err_eff_pitch3 = np.abs(fault_test("eff", "v2PITCH", 250, 10, REDQ=True, Q_nr=5, plot=False))

# ROLL - EFF
err_eff_roll1 = np.abs(fault_test("eff", "v2ROLL", 250, 10, plot=False))
err_eff_roll2 = np.abs(fault_test("eff", "v2ROLL", 250, 10, REDQ=True, plot=False))
err_eff_roll3 = np.abs(fault_test("eff", "v2ROLL", 250, 10, REDQ=True, Q_nr=5, plot=False))

# PITCH - JOLT
err_jolt_pitch1 = np.abs(fault_test("jolt", "v2PITCH", 250, 10, plot=True))
err_jolt_pitch2 = np.abs(fault_test("jolt", "v2PITCH", 250, 10, REDQ=True, plot=False))
err_jolt_pitch3 = np.abs(fault_test("jolt", "v2PITCH", 250, 10, REDQ=True, Q_nr=5, plot=False))

# ROLL - JOLT
err_jolt_roll1 = np.abs(fault_test("jolt", "v2ROLL", 250, 10, plot=False))
err_jolt_roll2 = np.abs(fault_test("jolt", "v2ROLL", 250, 10, REDQ=True, plot=False))
err_jolt_roll3 = np.abs(fault_test("jolt", "v2ROLL", 250, 10, REDQ=True, Q_nr=5, plot=False))

#========================================================================= End of first plot above

labels = ['SAC', 'RED3Q', 'RED5Q']
groups = {
    'Pitch - Nom': [err_nom_pitch1, err_nom_pitch2, err_nom_pitch3],
    'Pitch - Eff': [err_eff_pitch1, err_eff_pitch2, err_eff_pitch3],
    'Pitch - Jolt': [err_jolt_pitch1, err_jolt_pitch2, err_jolt_pitch3],
    'Roll - Nom': [err_nom_roll1, err_nom_roll2, err_nom_roll3],
    'Roll - Eff': [err_eff_roll1, err_eff_roll2, err_eff_roll3],
    'Roll - Jolt': [err_jolt_roll1, err_jolt_roll2, err_jolt_roll3],
}

for title, data_list in groups.items():
    dfs = []
    for agent_label, data in zip(labels, data_list):
        # Make sure data is a numpy array for convenience and take abs if needed
        arr = np.abs(np.array(data)).flatten()  
        dfs.append(pd.DataFrame({
            'Error': arr,
            'Agent': agent_label
        }))

    df = pd.concat(dfs, ignore_index=True)

    plt.figure(figsize=(7, 5))
    sns.boxplot(x='Agent', y='Error', data=df, palette='Set2', fill=False)
    plt.yscale('log')
    plt.xlabel("Agent")
    plt.ylabel("Mean error (deg)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


agents = ['SAC', 'RED3Q', 'RED5Q']

def prepare_df_pitchroll(data_list, condition):
    """
    data_list: list of np arrays, each shape (N_samples, 2), ordered as [SAC, RED3Q, RED5Q]
    condition: string for condition label
    """
    rows = []
    for i, agent in enumerate(agents):
        arr = data_list[i]  # shape (N_samples, 2)
        pitch_vals = arr[:, 0]
        roll_vals = arr[:, 1]
        for v in pitch_vals:
            rows.append({'Agent': agent, 'Angle': '$\\theta_e$', 'Value': v, 'Condition': condition})
        for v in roll_vals:
            rows.append({'Agent': agent, 'Angle': '$\\phi_e$', 'Value': v, 'Condition': condition})
    return pd.DataFrame(rows)

# Compose the lists of arrays per condition, in SAC, RED3Q, RED5Q order

#Pitchroll nom
err_nom_pitchroll1 = np.abs(fault_test("nom", "v2PITCHROLL", 250, 10, plot=False))
err_nom_pitchroll2 = np.abs(fault_test("nom", "v2PITCHROLL", 250, 10, REDQ=True, plot=True))
err_nom_pitchroll3 = np.abs(fault_test("nom", "v2PITCHROLL", 250, 10, REDQ=True, Q_nr=5, plot=False))

#Pitchroll eff
err_eff_pitchroll1 = np.abs(fault_test("eff", "v2PITCHROLL", 250, 10, plot=False))
err_eff_pitchroll2 = np.abs(fault_test("eff", "v2PITCHROLL", 250, 10, REDQ=True, plot=False))
err_eff_pitchroll3 = np.abs(fault_test("eff", "v2PITCHROLL", 250, 10, REDQ=True, Q_nr=5, plot=False))

# Pitchroll jolt
err_jolt_pitchroll1 = np.abs(fault_test("jolt", "v2PITCHROLL", 250, 10, plot=False))
err_jolt_pitchroll2 = np.abs(fault_test("jolt", "v2PITCHROLL", 250, 10, REDQ=True, plot=True))
err_jolt_pitchroll3 = np.abs(fault_test("jolt", "v2PITCHROLL", 250, 10, REDQ=True, Q_nr=5, plot=False))

#Pitchroll
nom_data = [err_nom_pitchroll1, err_nom_pitchroll2, err_nom_pitchroll3]
eff_data = [err_eff_pitchroll1, err_eff_pitchroll2, err_eff_pitchroll3]
jolt_data = [err_jolt_pitchroll1, err_jolt_pitchroll2, err_jolt_pitchroll3]

# Prepare DataFrames
df_nom = prepare_df_pitchroll(nom_data, 'Nominal')
df_eff = prepare_df_pitchroll(eff_data, 'Efficient')
df_jolt = prepare_df_pitchroll(jolt_data, 'Jolt')

def plot_condition(df, condition):
    plt.figure(figsize=(8,6))
    sns.boxplot(data=df, x='Agent', y='Value', hue='Angle', palette='Set2', fill=False)
    plt.yscale('log')  # Optional: use if error values vary widely
    plt.ylabel("Mean absolute error (deg)")
    plt.xlabel("Agent")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Error')
    plt.tight_layout()
    plt.show()

# Plot all three conditions separately
plot_condition(df_nom, 'Nominal')
plot_condition(df_eff, 'Efficient')
plot_condition(df_jolt, 'Jolt')