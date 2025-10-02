import numpy as np
import matplotlib.pyplot as plt

# Function for plotting all of the states per task type, can choose whether to zoom in on the controlled states or plot all states
def plot_states(task: str, time_list: list[int], ref_list: list[int], state_list: list[int], action_list: list[int],
                   zoom: bool = True, extra: list[int] = []) -> None:
    
    state_names = ['p (deg/s)','q (deg/s)','r (deg/s)','$V_{TAS}$ (m/s)','$\\alpha$ (deg)','$\\beta$ (deg)','$\\phi$ (deg)',
        '$\\theta$ (deg)', '$\\psi$ (deg)', '$h_e$ (m)']

    if "pitchroll" in task.lower():
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

            # Plot actuator action if given
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

            # Plot Actuator action if given
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

            # We plot the reference on the theta state graph
            axs[6].plot(time_list, ref_array[:,1], label='$\\phi_r$ (deg)', linestyle='--', color='red')
            axs[6].legend(loc='right')
            axs[7].plot(time_list, ref_array[:,0], label='$\\theta_r$ (deg)', linestyle='--', color='red')
            axs[7].legend(loc='right')

            axs[10].plot(time_list, action_array[:,0], label='Agent action', color='green')

            # Plot Actuator action if given
            if len(extra) != 0:
                axs[10].plot(time_list, extra[:,0], label='Actuator action', linestyle='--', color='red')
                axs[10].legend(loc='right')

            axs[10].set_xlabel('Time (s)')
            axs[10].set_ylabel('$\\delta_e$ (deg)')
            axs[10].grid(True)

            axs[11].plot(time_list, action_array[:,1], label='Agent action', color='green')

            # Plot Actuator action if given
            if len(extra) != 0:
                axs[11].plot(time_list, extra[:,1], label='Actuator action', linestyle='--', color='red')
                axs[11].legend(loc='right')

            axs[11].set_xlabel('Time (s)')
            axs[11].set_ylabel('$\\delta_a$ (deg)')

            plt.tight_layout()
            plt.savefig("high_quality_plot.pdf") 
            plt.show()

            return

    if "pitch" in task.lower():
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

            # Plot actuator action if given
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

            # Plot Actuator action if given
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

    if "roll" in task.lower():
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

            # Plot Actuator action if given
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

            # Plot Actuator action if given
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