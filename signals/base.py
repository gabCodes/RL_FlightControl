import numpy as np
from scipy.signal import butter, filtfilt

# Applying low pass filter to ensure aircraft is able to follow
def low_pass_filter(data: np.ndarray, cutoff: float, order: int = 4) -> np.ndarray:
    b, a = butter(order, cutoff, btype='low', analog=False)

    return filtfilt(b, a, data)

# Randomly generating the references signals
def generate_ref(duration: float, max_amp: float = 0.26, num_terms: int = 4, dt: float = 0.01, offset: float = 0.0) -> callable:
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

def pitch_eval_ref(t):
    return 0.12 * np.sin(0.4 * t) + 0.032

def roll_eval_ref(t):
    return 0.12*np.sin(0.4*t)


