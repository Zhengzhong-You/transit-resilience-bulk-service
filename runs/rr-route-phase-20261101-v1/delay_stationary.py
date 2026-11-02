"""Joint residual-queue/displacement calculation with capacity marked by phase."""

import time

import numpy as np
from scipy.fft import next_fast_len, rfft, irfft
from scipy.sparse.csgraph import connected_components
from scipy.stats import poisson


def joint_transition_operator(states, transition, headway, arrival_rate, capacity, cutoff,
                              capacity_kernel=None):
    capacities = np.broadcast_to(np.asarray(capacity, dtype=int), len(states))
    intervals = headway + states[None, :] - states[:, None]
    if np.any(intervals[transition > 0] < -1e-9):
        raise ValueError('The phase kernel contains a negative service interval.')
    means = arrival_rate * np.maximum(intervals, 0)
    maximum_capacity = (int(capacities.max()) if capacity_kernel is None
                        else capacity_kernel.shape[0] - 1)
    arrivals = poisson.pmf(np.arange(cutoff + maximum_capacity + 1)[:, None, None], means)
    length = next_fast_len(2 * cutoff + 2 * maximum_capacity + 1)
    if capacity_kernel is None:
        kernel = rfft(arrivals * transition[None, :, :], n=length, axis=0)
    else:
        # The shift M-c turns Y-c into a nonnegative polynomial exponent.
        kernel = (rfft(arrivals, n=length, axis=0)
                  * rfft(capacity_kernel[::-1], n=length, axis=0))
        capacities = np.full(len(states), maximum_capacity)

    def apply(joint):
        transformed = np.einsum('fi,fij->fj', rfft(joint, n=length, axis=0),
                                kernel, optimize=True)
        totals = irfft(transformed, n=length, axis=0)
        result = np.empty_like(joint)
        for j, batch in enumerate(capacities):
            result[0, j] = totals[:batch + 1, j].sum()
            result[1:cutoff, j] = totals[batch + 1:batch + cutoff, j]
        # All omitted Poisson mass goes to the explicitly capped upper state.
        result[cutoff] = joint.sum(axis=0) @ transition - result[:cutoff].sum(axis=0)
        if result.min() < -1e-11:
            raise ArithmeticError('Negative probability exceeds numerical roundoff.')
        result = np.maximum(result, 0)
        return result / result.sum()

    return apply


def stationary_queue(states, transition, headway, arrival_rate, capacity, cutoff,
                     tolerance=1e-10, max_iterations=20000, capacity_kernel=None):
    start = time.perf_counter()
    states = np.asarray(states, dtype=float)
    transition = np.asarray(transition, dtype=float)
    capacities = np.broadcast_to(np.asarray(capacity, dtype=float), len(states))
    if (not np.all(np.isfinite(capacities)) or np.any(capacities < 0)
            or np.any(capacities != np.floor(capacities))):
        raise ValueError('Phase capacities must be nonnegative integers.')
    capacities = capacities.astype(int)
    if (not np.all(np.isfinite(transition)) or np.any(transition < 0)
            or not np.allclose(transition.sum(axis=1), 1)):
        raise ValueError('The phase transition matrix must be stochastic.')
    if connected_components(transition > 0, directed=True, connection='strong')[0] != 1:
        raise ValueError('The fitted phase kernel must be irreducible.')
    matrix = transition.T - np.eye(len(states))
    matrix[-1] = 1
    rhs = np.zeros(len(states))
    rhs[-1] = 1
    phase = np.linalg.solve(matrix, rhs)
    if phase.min() < -1e-12:
        raise ArithmeticError('Invalid stationary phase probabilities.')
    phase = np.maximum(phase, 0)
    phase /= phase.sum()
    if capacity_kernel is None:
        mean_capacity = float(phase @ capacities)
    else:
        capacity_kernel = np.asarray(capacity_kernel, dtype=float)
        if (capacity_kernel.ndim != 3 or capacity_kernel.shape[1:] != transition.shape
                or not np.all(np.isfinite(capacity_kernel)) or np.any(capacity_kernel < 0)
                or not np.allclose(capacity_kernel.sum(axis=0), transition)):
            raise ValueError('Capacity marks must sum to the phase transition kernel.')
        mean_capacity = float(np.einsum('i,cij,c->', phase, capacity_kernel,
                                        np.arange(len(capacity_kernel))))
    if not 0 < arrival_rate * headway < mean_capacity:
        raise ValueError('Strict stability is required before solving a capped queue.')
    joint = np.zeros((cutoff + 1, len(states)))
    joint[0] = phase
    apply = joint_transition_operator(states, transition, headway, arrival_rate, capacities,
                                     cutoff, capacity_kernel)
    for iteration in range(1, max_iterations + 1):
        updated = apply(joint)
        change = np.abs(updated - joint).sum()
        joint = updated
        if change < tolerance:
            break
    else:
        raise RuntimeError('Joint stationary iteration did not converge.')
    balance = float(np.abs(apply(joint) - joint).sum())
    levels = np.arange(cutoff + 1)
    conditional_first = levels @ joint
    intervals = np.maximum(headway + states[None, :] - states[:, None], 0)
    mean_interval = float(np.sum(phase[:, None] * transition * intervals))
    second_interval = float(np.sum(phase[:, None] * transition * intervals ** 2))
    residual_times_interval = float(conditional_first @ np.sum(transition * intervals, axis=1))
    second_by_phase = levels ** 2 @ joint
    interval_second_by_phase = np.sum(transition * intervals ** 2, axis=1)
    third_interval = float(np.sum(phase[:, None] * transition * intervals ** 3))
    mean_time_queue = (residual_times_interval + arrival_rate * second_interval / 2) / headway
    second_time_queue = (
        second_by_phase @ np.sum(transition * intervals, axis=1)
        + arrival_rate * conditional_first @ interval_second_by_phase
        + arrival_rate * second_interval / 2 + arrival_rate ** 2 * third_interval / 3
    ) / headway
    variance_wait = (second_time_queue - mean_time_queue ** 2 - mean_time_queue) / arrival_rate ** 2
    mean_wait = (residual_times_interval + arrival_rate * second_interval / 2) / (
        arrival_rate * headway
    )
    mean_queue = float(conditional_first.sum() + arrival_rate * headway)
    variance_queue = float(second_by_phase.sum() + 2 * arrival_rate * residual_times_interval
                           + arrival_rate * headway + arrival_rate ** 2 * second_interval
                           - mean_queue ** 2)
    return {'joint': joint, 'phase': phase, 'mean_residual': float(conditional_first.sum()),
            'mean_queue': mean_queue, 'variance_queue': variance_queue,
            'variance_residual': float(second_by_phase.sum() - conditional_first.sum() ** 2),
            'mean_wait': mean_wait, 'variance_wait': float(variance_wait),
            'mean_headway': mean_interval,
            'mean_available_capacity': mean_capacity,
            'residual_times_next_headway': residual_times_interval,
            'upper_boundary_probability': float(joint[-1].sum()),
            'balance_residual_l1': balance, 'iterations': iteration,
            'elapsed_seconds': time.perf_counter() - start}
