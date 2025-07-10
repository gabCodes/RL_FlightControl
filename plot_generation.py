import numpy as np
import matplotlib.pyplot as plt
import optuna
import scipy.stats as stats
import plotly.express as px
import os
#from optuna.visualization import plot_parallel_coordinate

def state_plotter(task, time_list, ref_list, state_list, action_list, state_names, zoom = True, extra = []):
    fontsize_labels = 14
    fontsize_ticks = 12
    #fontsize_legend = 12
    if "pitchroll" in task:
        state_array = np.array(state_list)
        action_array = np.array(action_list)
        extra = np.array(extra)
        ref_array = np.array(ref_list)
        num_states = state_array.shape[1]

        if zoom == True:
            fig, axs = plt.subplots(2, 2, figsize=(10, 8)) 
            axs = axs.flatten()

            axs[0].plot(time_list, state_array[:, 7], label=state_names[7], linewidth=1.8)
            axs[0].set_xlabel('Time (s)')
            axs[0].set_ylabel(state_names[7])
            axs[0].plot(time_list, ref_array[:,0], label='$\\theta_r$ (deg)', linestyle='--', color='red')
            axs[0].legend(loc='right')
            axs[0].grid(True)

            axs[1].plot(time_list, action_array[:,0], label='Agent action', color='green')
            if len(extra) !=0:
                axs[1].plot(time_list, extra[:,0], label='Actuator action', linestyle='--', color='red')
                axs[1].legend(loc='right')
            axs[1].set_xlabel('Time (s)')
            axs[1].set_ylabel('$\\delta_e$ (deg)')
            axs[1].grid(True)

            axs[2].plot(time_list, state_array[:, 6], label=state_names[6], linewidth=1.8)
            axs[2].set_xlabel('Time (s)')
            axs[2].set_ylabel(state_names[6])
            axs[2].plot(time_list, ref_array[:,1], label='$\\phi_r$ (deg)', linestyle='--', color='red')
            axs[2].legend(loc='right')
            axs[2].grid(True)

            axs[3].plot(time_list, action_array[:,1], label='Agent action', color='green')
            if len(extra) !=0:
                axs[3].plot(time_list, extra[:,1], label='Actuator action', linestyle='--', color='red')
                axs[3].legend(loc='right')
            axs[3].set_xlabel('Time (s)')
            axs[3].set_ylabel('$\\delta_a$ (deg)')
            axs[3].grid(True)
            plt.tight_layout()
            plt.show()
            return

        else:
            fig, axs = plt.subplots(6, 2, figsize=(15, 10)) 
            axs = axs.flatten()

            # Plot 12 states
            for i in range(num_states):
                axs[i].plot(time_list, state_array[:, i], label=state_names[i], linewidth=1.8)
                axs[i].set_xlabel('Time (s)')
                axs[i].set_ylabel(state_names[i])
                axs[i].grid(True)

            # we plot the reference on the theta state graph
            axs[6].plot(time_list, ref_array[:,1], label='$\\phi_r$ (deg)', linestyle='--', color='red')
            axs[6].legend(loc='right')
            axs[7].plot(time_list, ref_array[:,0], label='$\\theta_r$ (deg)', linestyle='--', color='red')
            axs[7].legend(loc='right')

            axs[10].plot(time_list, action_array[:,0], label='Agent action', color='green')
            if len(extra) != 0:
                axs[10].plot(time_list, extra[:,0], label='Actuator action', linestyle='--', color='red')
                axs[10].legend(loc='right')
            axs[10].set_xlabel('Time (s)')
            axs[10].set_ylabel('$\\delta_e$ (deg)')
            axs[10].grid(True)

            axs[11].plot(time_list, action_array[:,1], label='Agent action', color='green')
            if len(extra) != 0:
                axs[11].plot(time_list, extra[:,1], label='Actuator action', linestyle='--', color='red')
                axs[11].legend(loc='right')
            axs[11].set_xlabel('Time (s)')
            axs[11].set_ylabel('$\\delta_a$ (deg)')
            # axs[11].tick_params(axis='both', which='major', labelsize=fontsize_ticks)
            plt.tight_layout()
            plt.savefig("high_quality_plot.pdf") 
            plt.show()
            return

    if "pitch" in task:
        state_array = np.array(state_list)
        action_array = np.array(action_list)
        ref_array = np.array(ref_list)
        if zoom == True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.plot(time_list, state_array[:, 7], label=state_names[7], linewidth=1.8)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel(state_names[7])
            ax1.plot(time_list, ref_list, label='$\\theta_r$ (deg)', linestyle='--', color='red')
            ax1.legend(loc='right')
            ax1.grid(True)

            ax2.plot(time_list, action_array[:,0], label='Agent action', color='green')
            if len(extra) != 0:
                ax2.plot(time_list, extra, label='Actuator action', linestyle='--', color='red')
                ax2.legend(loc='right')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('$\\delta_e$ (deg)')
            ax2.grid(True)
            plt.tight_layout()
            plt.show()
            return

        else:
            state_array = np.array(state_list)
            action_array = np.array(action_list)
            num_states = state_array.shape[1]

            fig, axs = plt.subplots(6, 2, figsize=(15, 10)) 
            axs = axs.flatten()

            # Plot 11 states
            for i in range(num_states):
                axs[i].plot(time_list, state_array[:, i], label=state_names[i], linewidth=1.8)
                axs[i].set_xlabel('Time (s)')
                axs[i].set_ylabel(state_names[i])
                axs[i].grid(True)

            # we plot the reference on the theta state graph
            axs[7].plot(time_list, ref_list, label='Ref signal', linestyle='--', color='red')
            axs[7].legend(loc='right')

            axs[10].plot(time_list, action_array[:,0], label='Agent action', color='green')
            axs[10].set_xlabel('Time (s)')
            axs[10].set_ylabel('$\\delta_e$ (deg)')
            if len(extra) != 0:
                axs[10].plot(time_list, extra, label='Actuator action', linestyle='--', color='red')
                axs[10].legend(loc='right')
            axs[10].grid(True)

            axs[11].plot(time_list, action_array[:,1], label='Action', color='green')
            axs[11].set_xlabel('Time (s)')
            axs[11].set_ylabel('$\\delta_a$ (deg)')
            axs[11].grid(True)
            plt.tight_layout()
            plt.savefig("high_quality_plot.pdf") 
            plt.show()
            return

    if "roll" in task:
        state_array = np.array(state_list)
        action_array = np.array(action_list)
        ref_array = np.array(ref_list)
        if zoom == True:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            ax1.plot(time_list, state_array[:, 6], label=state_names[6], linewidth=1.8)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel(state_names[6])
            ax1.plot(time_list, ref_list, label='$\\phi_r$ (deg)', linestyle='--', color='red')
            ax1.legend(loc='right')
            ax1.grid(True)

            ax2.plot(time_list, action_array[:,1], label='Agent action', color='green')
            if len(extra) != 0:
                ax2.plot(time_list, extra, label='Actuator action', linestyle='--', color='red')
                ax2.legend(loc='right')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('$\\delta_a$ (deg)')
            ax2.grid(True)
            plt.tight_layout()
            plt.show()
            return
        
        else:
            state_array = np.array(state_list)
            action_array = np.array(action_list)
            num_states = state_array.shape[1]

            fig, axs = plt.subplots(6, 2, figsize=(15, 10)) 
            axs = axs.flatten()

            # Plot 11 states
            for i in range(num_states):
                axs[i].plot(time_list, state_array[:, i], label=state_names[i], linewidth=1.8)
                axs[i].set_xlabel('Time (s)')
                axs[i].set_ylabel(state_names[i])
                axs[i].grid(True)

            # we plot the reference on the theta state graph
            axs[6].plot(time_list, ref_list, label='Ref signal', linestyle='--', color='red')
            axs[6].legend(loc='right')

            axs[10].plot(time_list, action_array[:,0], label='Action', color='green')
            axs[10].set_xlabel('Time (s)')
            axs[10].set_ylabel('$\\delta_e$ (deg)')
            axs[10].grid(True)

            axs[11].plot(time_list, action_array[:,1], label='Agent action', color='green')
            if len(extra) != 0:
                axs[11].plot(time_list, extra, label='Actuator action', linestyle='--', color='red')
                axs[11].legend(loc='right')
            axs[11].set_xlabel('Time (s)')
            axs[11].set_ylabel('$\\delta_a$ (deg)')
            axs[11].grid(True)
            plt.tight_layout()
            plt.savefig("high_quality_plot.pdf") 
            plt.show()
            return


def create_parallel(storage_url, study_name):

    #storage_url = "sqlite:///CAPS_STUDY.db"
    #study_name = "SAC_30_RUNS"
    #study_name = "REDQ_HYPERPARAMS"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    for trial in study.trials:
        print(f"Trial {trial.number}: Value={trial.value}")
    # Convert the study trials to a dataFrame
    df = study.trials_dataframe(attrs=("number", "params", "value", "state"))

    print(df.columns)


    # Filter out only the completed trials
    df = df[df['state'] == 'COMPLETE']
    df = df[df["number"] != 95]

    # Drop unnecessary columns
    if "REDQ" in study_name:
        df = df[['number', 'params_BATCH_SIZE', 'params_BUFFER_SIZE', 'params_DROPOUT', 'params_GAMMA', 'params_LR', 'params_NR_CRITICS', 'params_TAU', 'params_UTD', 'value']]
        dimensions=['params_BATCH_SIZE', 'params_BUFFER_SIZE', 'params_DROPOUT', 'params_GAMMA', 'params_LR', 'params_NR_CRITICS', 'params_TAU', 'params_UTD']
    
    else:
        df = df[['number', 'params_BATCH_SIZE', 'params_GAMMA', 'params_LR', 'params_TAU', 'value']]
        dimensions=['params_BATCH_SIZE', 'params_GAMMA', 'params_LR', 'params_TAU'] 

    fig = px.parallel_coordinates(
    df,
    dimensions=dimensions,
    color="value", 
    color_continuous_scale=px.colors.diverging.RdBu
    )

    fig.update_layout(
        title="",
        coloraxis_colorbar=dict(title="Objective Value")
    )
    fig.show()

def load_and_process(filename):
    with open(filename, 'r') as f:
        clean_data = [
            [float(x) for x in line.strip().replace('[', '').replace(']', '').split(',') if x.strip()]
            for line in f
        ]
    data = np.array(clean_data)
    if data.shape[1] == 200:
        data = data[:, :data.shape[1] // 2]
    return data

# Compute Mean & CI
def compute_mean_ci(data, confidence=0.95):
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0, ddof=1)
    n = data.shape[0]
    sem = stds / np.sqrt(n)
    h = stats.t.ppf((1 + confidence) / 2, df=n-1) * sem
    return means, means - h, means + h

#Uncomment the relevant datasets to plot the learning curve that we're interested in.
def plot_learningcurves():
    path1 = os.path.join('performances', 'SAC30v2PITCH50.txt')
    path2 = os.path.join('performances', 'RED3Q30v2PITCH50.txt')
    path3 = os.path.join('performances', 'RED5Q30v2PITCH50.txt')

    # path1 = os.path.join('performances', 'SAC30v2PITCH250.txt')
    # path2 = os.path.join('performances', 'RED3Q30v2PITCH250.txt')
    # path3 = os.path.join('performances', 'RED5Q30v2PITCH250.txt')

    # path1 = os.path.join('performances', 'SAC30v2PITCHROLL250.txt')
    # path2 = os.path.join('performances', 'RED3Q30v2PITCHROLL250.txt')
    # path3 = os.path.join('performances', 'RED5Q30v2PITCHROLL250.txt')

    # path1 = os.path.join('performances', 'SAC30v2PITCHROLL50.txt')
    # path2 = os.path.join('performances', 'RED3Q30v2PITCHROLL50.txt')
    # path3 = os.path.join('performances', 'RED5Q30v2PITCHROLL50.txt')

    # path1 = os.path.join('performances', 'SAC30v2ROLL250.txt')
    # path2 = os.path.join('performances', 'RED3Q30v2ROLL250.txt')
    # path3 = os.path.join('performances', 'RED5Q30v2ROLL250.txt')

    # path1 = os.path.join('performances', 'SAC30v2ROLL50.txt')
    # path2 = os.path.join('performances', 'RED3Q30v2ROLL50.txt')
    # path3 = os.path.join('performances', 'RED5Q30v2ROLL50.txt')

    data1 = load_and_process(path1)
    data2 = load_and_process(path2)
    data3 = load_and_process(path3)

    means1, lower1, upper1 = compute_mean_ci(data1)
    means2, lower2, upper2 = compute_mean_ci(data2)
    means3, lower3, upper3 = compute_mean_ci(data3)

    print(f"Final step mean - SAC:     {means1[-1]:.3f}")
    print(f"Final step mean - RED3Q:   {means2[-1]:.3f}")
    print(f"Final step mean - RED5Q:   {means3[-1]:.3f}")

    print(f"Final mean error - SAC:     {means1[-1]/2000:.3f}")
    print(f"Final mean error - RED3Q:   {means2[-1]/2000:.3f}")
    print(f"Final mean error - RED5Q:   {means3[-1]/2000:.3f}")

    #t-test: between SAC and RED3Q
    t_stats_checkpoint = []
    p_values_checkpoint = []

    for i in range(data1.shape[1]):
        t_stat, p_value = stats.ttest_ind(data1[:, i], data2[:, i], equal_var=False)
        t_stats_checkpoint.append(t_stat)
        p_values_checkpoint.append(p_value)

    t_stats_checkpoint = np.array(t_stats_checkpoint)
    p_values_checkpoint = np.array(p_values_checkpoint)

    #fix x-axis for different resolutions
    if data1.shape[1] == 100:
        training_steps = np.arange(data1.shape[1]) * 250
    else:
        training_steps = np.arange(data1.shape[1]) * 50
    # Create two vertically stacked plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Learning curves
    ax1.plot(training_steps, means1, label='SAC agent', color='blue')
    ax1.fill_between(training_steps, lower1, upper1, color='blue', alpha=0.3)

    ax1.plot(training_steps, means2, label='RED3Q agent', color='red')
    ax1.fill_between(training_steps, lower2, upper2, color='red', alpha=0.3)

    ax1.plot(training_steps, means3, label='RED5Q agent', color='green')
    ax1.fill_between(training_steps, lower3, upper3, color='green', alpha=0.3)

    ax1.set_ylabel('Average return')
    ax1.legend()
    ax1.grid(True)

    # p-values
    ax2.scatter(training_steps, p_values_checkpoint, color='red', alpha=0.6, label='p-values')
    ax2.axhline(0.05, color='green', linestyle='--', label='Significance threshold (0.05)')
    ax2.set_xlabel('Training steps')
    ax2.set_ylabel('p-value')
    ax2.legend(loc='upper left')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

# plot_learningcurves()