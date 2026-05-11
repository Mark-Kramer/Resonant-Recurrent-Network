import numpy as np


def simulated_data(
    firing_rate_hz: float,
    min_frequency_hz: float | None = None,
    max_frequency_hz: float | None = None,
) -> np.ndarray:
    """
    Generate a binary Poisson spike train.

    Parameters
    ----------
    firing_rate_hz : float
        Average firing rate in Hz.
    min_frequency_hz : float | None, optional
        Minimum sinusoidal modulation frequency in Hz.
        If both min/max are None, the firing rate is constant.
        If only min is provided, that fixed frequency is used.
    max_frequency_hz : float | None, optional
        Maximum sinusoidal modulation frequency in Hz.
        If provided with min, one fixed frequency is sampled uniformly in [min, max] and used for the entire returned spike train.

    Returns
    -------
    np.ndarray
        Binary spike train of shape (500,) sampled at 1000 Hz for 0.5 s.
    """
    if firing_rate_hz < 0:
        raise ValueError("firing_rate_hz must be non-negative.")
    if min_frequency_hz is not None and min_frequency_hz < 0:
        raise ValueError("min_frequency_hz must be non-negative when provided.")
    if max_frequency_hz is not None and max_frequency_hz < 0:
        raise ValueError("max_frequency_hz must be non-negative when provided.")
    if min_frequency_hz is None and max_frequency_hz is not None:
        raise ValueError("max_frequency_hz requires min_frequency_hz.")
    if (
        min_frequency_hz is not None
        and max_frequency_hz is not None
        and max_frequency_hz < min_frequency_hz
    ):
        raise ValueError("max_frequency_hz must be >= min_frequency_hz.")

    fs         = 1000.0
    duration_s = 0.1
    n_samples  = int(fs * duration_s)
    dt         = 1.0 / fs

    t = np.arange(n_samples) * dt
    if min_frequency_hz is None:
        lambda_t = np.full(n_samples, firing_rate_hz, dtype=float)
    else:
        if max_frequency_hz is None:
            f_hz = float(min_frequency_hz)
        else:
            f_hz = float(np.random.uniform(min_frequency_hz, max_frequency_hz))
        # Modulate around the mean rate while keeping rate non-negative.
        lambda_t = firing_rate_hz * (1.0 + np.sin(2.0 * np.pi * f_hz * t))

    # For a Poisson process, probability of >=1 event in a bin is 1 - exp(-lambda(t)*dt).
    p_spike = 1.0 - np.exp(-lambda_t * dt)
    spike_train = (np.random.rand(n_samples) < p_spike).astype(np.int8)
    return spike_train
