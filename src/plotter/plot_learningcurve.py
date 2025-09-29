import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os


# Plots the learning curves for the different agents
def plot_learningcurve(files: list[str]) -> None:
    path1 = os.path.join('performance', files[0])
    path2 = os.path.join('performance', files[1])
    path3 = os.path.join('performance', files[2])

    data1 = _load_and_process(path1)
    data2 = _load_and_process(path2)
    data3 = _load_and_process(path3)

    means1, lower1, upper1 = _compute_mean_ci(data1)
    means2, lower2, upper2 = _compute_mean_ci(data2)
    means3, lower3, upper3 = _compute_mean_ci(data3)

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

    return

# Load and clean agent performance data
def _load_and_process(filename: str) -> np.ndarray:
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
def _compute_mean_ci(data: np.ndarray, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0, ddof=1)
    n = data.shape[0]
    sem = stds / np.sqrt(n)
    h = stats.t.ppf((1 + confidence) / 2, df=n-1) * sem

    return means, means - h, means + h