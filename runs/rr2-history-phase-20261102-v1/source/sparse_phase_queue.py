"""Sparse Fourier operator for a displacement/capacity phase queue."""

import numpy as np
from scipy.fft import rfft, irfft, next_fast_len
from scipy.sparse import csr_matrix
from scipy.stats import poisson


def sparse_phase_transition_operator(states, transition, headway, arrival_rate,
                                      capacity, cutoff, capacity_kernel=None):
    if capacity_kernel is not None:
        raise ValueError('This operator represents capacity by the destination phase.')
    states = np.asarray(states)
    capacity = np.broadcast_to(capacity, len(states)).astype(int)
    sparse = csr_matrix(transition)
    rows, columns = sparse.nonzero()
    intervals = headway + states[columns] - states[rows]
    if np.any(intervals < -1e-9):
        raise ValueError('A phase transition has a negative service interval.')
    maximum_capacity = int(capacity.max())
    length = next_fast_len(2 * cutoff + 2 * maximum_capacity + 1)
    alias_bound = poisson.sf(length - cutoff - maximum_capacity - 1,
                             arrival_rate * intervals.max())
    if alias_bound > 1e-14:
        raise ValueError('Increase the FFT range to bound wrapped Poisson mass.')
    z = np.exp(-2j * np.pi * np.arange(length // 2 + 1) / length)
    exponent = arrival_rate * (z - 1)
    if 2 * arrival_rate * states.max() > 650:
        raise ValueError('The factorized transform would exceed the floating-point range.')
    left = np.exp(-exponent[:, None] * states[None, :])
    right = (np.exp(exponent[:, None] * (headway + states[None, :]))
             * z[:, None] ** (maximum_capacity - capacity[None, :]))

    def apply(joint):
        transformed = rfft(joint, n=length, axis=0) * left
        transformed = sparse.T.dot(transformed.T).T * right
        totals = irfft(transformed, n=length, axis=0)
        result = np.empty_like(joint)
        result[0] = totals[:maximum_capacity + 1].sum(axis=0)
        result[1:cutoff] = totals[maximum_capacity + 1:maximum_capacity + cutoff]
        result[cutoff] = sparse.T.dot(joint.sum(axis=0)) - result[:cutoff].sum(axis=0)
        if result.min() < -1e-11:
            raise ArithmeticError('Sparse phase probabilities exceed roundoff tolerance.')
        result = np.maximum(result, 0)
        return result / result.sum()

    return apply


def fit_joint_phase(displacements, capacities, step=1.2, tail=.0005):
    upper = int(np.floor(np.quantile(displacements, 1 - tail) / step))
    labels = np.minimum(np.floor(displacements / step + 1e-8).astype(int), upper)
    phases, encoded = np.unique(np.c_[labels, capacities], axis=0, return_inverse=True)
    size = len(phases)
    pairs, counts = np.unique(encoded[:-1] * size + encoded[1:], return_counts=True)
    rows, columns = pairs // size, pairs % size
    totals = np.bincount(rows, weights=counts, minlength=size)
    if np.any(totals == 0):
        raise ValueError('A joint calibration phase has no outgoing transitions.')
    transition = csr_matrix((counts / totals[rows], (rows, columns)), shape=(size, size))
    return phases[:, 0] * step, phases[:, 1], transition, upper * step
