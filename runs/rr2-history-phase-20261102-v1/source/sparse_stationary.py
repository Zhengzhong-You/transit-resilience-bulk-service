"""Sparse version of delay_stationary.stationary_queue for large phase spaces.

The queue iteration, moment formulas and checks are those of stationary_queue; only the
phase-level quantities are computed on the nonzero transitions instead of dense arrays.
"""
import time

import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from sparse_phase_queue import sparse_phase_transition_operator


def stationary_queue_sparse(states, transition, headway, arrival_rate, capacity, cutoff,
                            tolerance=1e-10, max_iterations=20000, initial=None):
    start = time.perf_counter()
    states = np.asarray(states, dtype=float)
    P = csr_matrix(transition, dtype=float)
    size = len(states)
    capacities = np.broadcast_to(np.asarray(capacity, dtype=float), size)
    if (not np.all(np.isfinite(capacities)) or np.any(capacities < 0)
            or np.any(capacities != np.floor(capacities))):
        raise ValueError('Phase capacities must be nonnegative integers.')
    capacities = capacities.astype(int)
    if (not np.all(np.isfinite(P.data)) or np.any(P.data < 0)
            or not np.allclose(np.asarray(P.sum(axis=1)).ravel(), 1)):
        raise ValueError('The phase transition matrix must be stochastic.')
    if connected_components(P, directed=True, connection='strong')[0] != 1:
        raise ValueError('The fitted phase kernel must be irreducible.')
    A = (P.T - identity(size, format='csr')).tolil()
    A[size - 1, :] = np.ones(size)
    rhs = np.zeros(size)
    rhs[-1] = 1
    phase = spsolve(A.tocsc(), rhs)
    if phase.min() < -1e-12:
        raise ArithmeticError('Invalid stationary phase probabilities.')
    phase = np.maximum(phase, 0)
    phase /= phase.sum()
    mean_capacity = float(phase @ capacities)
    if not 0 < arrival_rate * headway < mean_capacity:
        raise ValueError('Strict stability is required before solving a capped queue.')
    coo = P.tocoo()
    rows, cols, prob = coo.row, coo.col, coo.data
    intervals = np.maximum(headway + states[cols] - states[rows], 0)
    weight = phase[rows] * prob
    mean_interval = float(weight @ intervals)
    second_interval = float(weight @ intervals ** 2)
    third_interval = float(weight @ intervals ** 3)
    row_interval = np.bincount(rows, weights=prob * intervals, minlength=size)
    row_interval_second = np.bincount(rows, weights=prob * intervals ** 2, minlength=size)
    apply = sparse_phase_transition_operator(states, P, headway, arrival_rate, capacities, cutoff)
    if initial is None:
        joint = np.zeros((cutoff + 1, size))
        joint[0] = phase
    else:
        joint = np.asarray(initial, dtype=float)
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
    second_by_phase = levels ** 2 @ joint
    residual_times_interval = float(conditional_first @ row_interval)
    mean_time_queue = (residual_times_interval + arrival_rate * second_interval / 2) / headway
    second_time_queue = (
        second_by_phase @ row_interval
        + arrival_rate * conditional_first @ row_interval_second
        + arrival_rate * second_interval / 2 + arrival_rate ** 2 * third_interval / 3
    ) / headway
    variance_wait = (second_time_queue - mean_time_queue ** 2 - mean_time_queue) / arrival_rate ** 2
    mean_wait = (residual_times_interval + arrival_rate * second_interval / 2) / (arrival_rate * headway)
    mean_queue = float(conditional_first.sum() + arrival_rate * headway)
    variance_queue = float(second_by_phase.sum() + 2 * arrival_rate * residual_times_interval
                           + arrival_rate * headway + arrival_rate ** 2 * second_interval
                           - mean_queue ** 2)
    return {'joint': joint, 'phase': phase, 'mean_residual': float(conditional_first.sum()),
            'mean_queue': mean_queue, 'variance_queue': variance_queue,
            'variance_residual': float(second_by_phase.sum() - conditional_first.sum() ** 2),
            'mean_wait': mean_wait, 'variance_wait': float(variance_wait),
            'mean_headway': mean_interval, 'mean_available_capacity': mean_capacity,
            'residual_times_next_headway': residual_times_interval,
            'upper_boundary_probability': float(joint[-1].sum()),
            'balance_residual_l1': balance, 'iterations': iteration,
            'elapsed_seconds': time.perf_counter() - start}
