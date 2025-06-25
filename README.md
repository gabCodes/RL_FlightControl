# Code: A study on sample-efficient model-free algorithms for flight control tasks

Code used for my master's thesis, will be updated to be more user friendly (for files like policy_plotter, faul_cases, etc).

> **Note:** The code currently requires **Windows OS** as the Cessna Citation model is compiled for a Windows architecture.


## Requirements

- All dependencies are listed in `requirements.txt`.
- **Network weights** are **not** uploaded due to size limits. If you're interested in the weights, reach out via GitHub and we’ll figure out a way to share them.
- You can still generate your own weights using the code (see below).

## Core files
The files that do most the work are:

- `util_training.py`: Handles the training of agents.
- `fault_cases.py`: Handles the fault cases and generates the box plots.
- `plot_generation.py`: Generates learning curves, parallel co-ordinate plots and time-series plots.
- `policy_plotter.py`: For generating policy plots.

## Minimal workflow to generate weights and generate plots

To train agents and generate weights:

1. Open `util_training.py`.
2. Use `sac_30_runs()` or `redq_30_runs()` (see comments in the file).
3. Training is:
   - **Time-consuming** (~1 day for 30 REDQ agents, ~1 hour per agent).
   - **Storage-intensive**.

Once training is complete, weights will be saved in the `.\checkpoints\` subfolder.

After weights are generated:

- Run the code at the bottom of `fault_cases.py`.
- Set `plot=True` to generate time series plots per condition.
- Leave `plot=False` to evaluate all agents and produce box plots.