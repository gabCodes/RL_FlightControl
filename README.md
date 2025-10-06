# Deep Reinforcement Learning for Aircraft Control

For my Master's thesis, I designed sample-efficient deep reinforcement learning agents for attitude (pitch and roll) tracking on a flight-validated high-fidelity Cessna 550 model. The full thesis document can be found in TU Delft's [repository](https://repository.tudelft.nl/record/uuid:272ae912-ddb2-41ef-a273-ba513c7687fc) for more intricate details.

## Problem statement

Modern aircraft rely heavily on flight control systems to manage flight tasks. Traditionally, these systems use linearised aircraft models to compute control gains for specific operating points. Since these gains are only valid near their respective points, the entire flight envelope must be covered by discretising it into multiple operating points and assigning gains accordingly. This approach is known as gain scheduling.

Reinforcement Learning (RL) offers an alternative to gain scheduling by enabling agents to learn control policies directly through interaction with the environment. Compared to conventional methods, RL provides the following advantages:

- Model-free learning → agents can operate without explicit system models
- Nonlinear adaptability → no need for predefined scheduling across flight regimes
- Generalisation → learned policies can perform well beyond the conditions they were trained on

My thesis, in particular, looked at how Randomised Ensemble Double Q-Learning ([REDQ](https://arxiv.org/abs/2101.05982)) can be used with a Soft Actor-Critic ([SAC](https://arxiv.org/abs/1801.01290)) base to train agents offline that are able to perform tracking in a sample-efficient, fault-tolerant and model-free way. See below for a state plot showing a trained agent tracking both pitch and roll sinusoidal references (yaw is not actively controlled).

<div align="center">
	<img src="images\RED3Q_SMOOTH.png" alt="Bi-axial tracking" width="600">
</div>

## Results

All agents were able to converge to policies able to track their given references. Some specifics regarding agent training:

- Divided into two agent types: SAC and REDQ. SAC served as the benchmark performance
- There were two REDQ agents: RED3Q and RED5Q. The number signifies the number of critics
- Trained to track pitch, roll and a bi-axial task combining both
- 30 agents of each type were trained per task to get representative results

For the brevity of this write-up, the results shown here will use a non-exhaustive selection of plots from the three types of tasks.

### Tracking performance

The tracking error for trained agents is mostly under 1°, see below for one such realisation. The left plot shows a time series with a reference pitch and the aircraft pitch controlled by agents, the right plot shows the elevator command deflections given by the agents. Agents track references accurately and their actions are smooth.

<div align="center">
	<img src="images\PITCH_TRACKING.png" alt="Tracking example" width="600">
</div>

### Learning curves

The learning curve for the pitch tracking task below shows that REDQ agents learn about twice as fast. It's a combined plot that contains:

1. The upper plot - shows the average return evaluated at intervals of 50 steps
2. The lower plot - shows the statistical significance of the improvement in performance between RED3Q and SAC agents using a Welch t-test (assumes normality but allows for different variances) with N = 30 per agent

<div align="center">
	<img src="images\PITCH_COMBINEDPLOT.png" alt="Agent learning curves" width="600">
</div>

### Fault scenario analysis

After training the agents using randomly (yet possible to control) generated signals, the performance of the agents was evaluated under faults they had never experienced during their training. Three scenarios were considered:

1. Nominal scenario - no faults are introduced but the evaluation length is extended beyond the training horizon. Tests to see if they're able to operate under longer time periods
2. Quarter actuator efficiency - agent actions are attenuated to 25% of their output. Tests to see if agents are robust to diminished control
3. Jolt introduction - two separate jolts to the actuator are introduced causing maximum deflection in both directions, forcing the aircraft to deviate far from the reference. Tests if agents are able to recover from short sudden deviations

###### Nominal

The nominal scenario performance was the same as that seen in [tracking performance](#tracking-performance), the extended episode duration did not affect it at all.

###### Quarter actuator efficiency

The quarter actuator efficiency fault also did not seem to affect the agent's performance at all, suggesting that the actions chosen are non-minimal. This is seen below by the accurate roll tracking even though the actuator only sees 25% of the agent's actions. The agents could probably get away with smaller actuator commands when tracking.

<div align="center">
	<img src="images\RED5QROLLeff.png" alt="Roll quarter actuator efficiency for RED5Q" width="600">
</div>

###### Jolt

The majority of agents were mostly able to recover from jolt faults and resume tracking after the introduced large deviations. Below is one such case, where a pitch tracking agent is jolted in opposite maximal deflections at 5s and 10s respectively. This is most easily seen in the bottom left plot, where the agent is forced to move opposite it's desired intent at 5 and 10 seconds. Agents trying to oppose the disturbance intuitively validate their learned behaviour.

<div align="center">
	<img src="images\SACpitchJOLT.png" alt="Pitch jolt fault scenario for SAC" width="600">
</div>

### Key take-aways

Main observations from the research:

1. Agents are successfully able to learn control tasks and track references in under ~4000 steps without an a priori model of the aircraft dynamics
2. No scheduling was needed, agents were able to act across the tested flight regime 
3. Initial sample efficiency doubled with REDQ agents
4. The majority of agents were able to generalise to tracking even with faults introduced, though some (particularly in the jolt fault) showed very poor performance

## Code details

This section gives an overview of how to run the code for results as well as gives insight into what the different modules do.

### Workflow

The minimal workflow necessary to obtain similar results (not exactly replicate due to the many randomisations) is given in the `main.py` file. All that's needed is to run it.

1. HPO (Optional) → conducts a hyperparameter optimisation
2. TRAINING → trains with the best hyperparameters, saves weights in `.checkpoints\` folder
3. EVALUATION → evaluates the agents, saves performance to `.performance\` folder and plots to `.figures\`

> **Note:** Training and evaluation is both time-consuming (~1 day for 30 REDQ agents, ~1 hour per agent) and storage-intensive (10s of GBs of neural network snapshots).

### Config

The majority of the pipeline parameters are specified in the `config.yml` file for a centralised way to quickly load or edit them as seen fit. It contains the agent hyperparameters and variables specific to agents, handlers, faults and operations.

### Citation model

The Cessna 550 model comes from a validated MATLAB Simulink model. The file `_citation.cp310-win_amd64.pyd` is generated via SWIG in order to allow for Python interfacing with a C compiled version of the Cessna 550 Simulink model. In order for the citation model to work without any issues, make sure to:

- Run using Python 3.10
- Use a Windows OS as the model was compiled for Windows
- Ensure all environment requirements in `env_requirements.txt` are met

### Modules

Module-specific information is given below.

#### Agents

This module contains all the code defining the actor, critics, replay buffer and the agent class that contains all of them. There are two types of agents in the code:

- SACAgent → SAC agent implementation
- REDQSACAgent → REDQ augmented SAC agents

Since the REDQ agent is an extension of SAC, only the snippet for REDQ is given below. The hyperparameters highlighted in red are the tweaks REDQ introduces over the standard SAC. These are:

- $U$ - the update-to-data ratio $U$, determines how many updates per update step
- $N$ - the number of critics

To get the SAC algorithm simply substitute $U = 1$ and $N = 2$. Note that compared to the REDQ paper this snippet removes the critic update subset hyperparameter $M$, adds automatic entropy tuning and adds Conditioning for Action Policy Smoothness ([CAPS](https://arxiv.org/abs/2012.06644)) to smoothen actuator outputs (very important for use in physical systems to reduce actuator energy consumption and wear).

<div align="center">
	<img src="images\REDQ_Alg.png" alt="REDQ Algorithm" width="600">
</div>

The network architecture for the actors and critics is shown below for the bi-axial task. Note that since in REDQ the number of critics $N$ is a hyperparameter, there will be $N$ critics. The small box at the bottom right indicates that when training the agent samples from a distribution in order to produce stochastic actions. 

<div align="center">
	<img src="images\Networks.png" alt="Actor and critic network architectures" width="600">
</div>

#### Citation

This is a SWIG auto-generated wrapper function that allows for the compiled model to initialise, step and terminate.

#### Signals

The signal module contains how the deterministic and randomly generated signals are constructed. 

- Deterministic → aggressive sinusoids with an angular frequency of 23°/s, designed to evaluate how well they're able to track aggressive yet feasible manoeuvres  
- Randomly generated → consist of sums of sinusoids with constraints on their maximum constructive amplitude and are passed through a low-pass filter to ensure tracking feasibility

#### Operations

The operations module contains the code for:

- Training
- Evaluation
- Hyperparameter tuning
- Result gathering runs
- Fault tests

###### Training & evaluation `train.py, evaluate.py`

The training and evaluation modules are based on the control logic seen below (running example will be for the bi-axial tracking task but others are analogous). In essence, the agent is fed the reference errors and the rates in order to output the actuator commands to control the aircraft.

<div align="center">
	<img src="images\pitchroll_loop.png" alt="Bi-axial block diagram" width="600">
</div>

The state space is defined by:

$$s = [\theta_e \quad q \quad \phi_e \quad p]^{T}$$

The action space is defined below, where $\delta_e$ is the elevator deflection relative to trim and $\delta_a$ are the aileron deflections.

$$a = [\delta_e \quad \delta_a]^T$$

The reward function is defined as follows, with the weights found through manual experimentation.

$$r = -0.6\times|\theta_e| -0.4\times|\phi_e|$$

During training, agent networks are saved to `.checkpoints\` at multiple steps for later evaluation in order to capture the learning performance. Shorter-spaced granular snapshots are used to capture the initial learning performance and fully trained weights are later used for testing fault performance.

###### Hyperparameter optimisation `HPO.py`

Trains and evaluates the agents using Optuna's hyperparameter search, the bounds for the search can be adapted to whatever search space is desired. The objective function is defined as the mean episodic reward over 20 episodes as it learns.

###### Result gathering runs `run.py`

Wrapper function that trains and evaluates agents over many runs in order to log data and network weights necessary for plotting learning curves and fault evaluations. Handles all the logic regarding reading and writing to files.

###### Fault tests `fault_tests.py`

Contains the code for evaluating the agents per fault condition. This is used to generate data to create boxplots per fault that showed the spread in performance with regards to agent, task and fault.

#### Handlers

Handlers were designed to modularise operations so they work for every task and agent type combination. Their behaviour is prescribed by their parent class `TaskHandler`. Handlers perform task and agent type specific operation steps. This can be extended to accommodate, for example, yaw control.

- PitchHandler → handles logic for the pitch control task, deals with $s = [\theta_e \quad q]^{T}$ and $a = [\delta_e]^T$
- RollHandler → handles logic for the roll control task, deals with $s = [\phi_e \quad p]^{T}$ and $a = [\delta_a]^T$
- PitchRollHandler → handles logic for bi-axial control task; see [operations](#operations)

#### Faults

Faults (`FaultHandlers`) are compositions of handlers that perturb certain parts of handler functions to mimic faults. For example, JoltFault overrides the agent's actions outputted to the system during the intervals in which a jolt is introduced. This framework can be extended to contain a larger suite of faults as desired.

- Nominal → behaviour is not changed but evaluation lasts longer
- QuarterEfficiencyFault → agent actions are attenuated to 25% of the value
- JoltFault → maximal 1s long deflections are introduced during evaluation

#### Plotter

This module contains all the plotting functions to plot: states, learning curves, boxplots, policy plots and parallel coordinate plots.