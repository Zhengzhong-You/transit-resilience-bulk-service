"""History-augmented displacement/capacity phase.

The phase of vehicle l at a station records its displacement class, its available capacity and the
class of the expected upstream backlog destined beyond the station once vehicle l has served the
upstream stations. The backlog is generated upstream, so it is independent of the local Poisson
arrivals, and the joint phase calculation applies unchanged: phases that share displacement and
capacity differ only in their transition probabilities. One backlog class recovers the first-order
displacement/capacity phase.
"""
import numpy as np
from scipy.sparse import csr_matrix, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

GRID_DIVISOR = 16
CLASS_LADDER = (16, 8, 4, 2, 1)
STATE_LIMIT = 4000
BLOCKS = (1, 10, 100, 1000, 3000)


def upstream_backlog(residuals, destinations, station):
    """Expected passengers left upstream of a 1-based station whose destination lies beyond it."""
    if station == 1:
        return np.zeros(len(residuals))
    p = np.asarray(destinations)
    weights = np.array([p[j, station:].sum() for j in range(station - 1)])
    return residuals[:, :station - 1] @ weights


def backlog_classes(backlog, classes):
    """Class 0 is an empty backlog; positive values are split at equal-frequency quantiles."""
    out = np.zeros(len(backlog), dtype=int)
    positive = backlog > 0
    if classes <= 1 or not positive.any():
        return out
    if classes == 2:
        out[positive] = 1
        return out
    edges = np.quantile(backlog[positive], np.linspace(0, 1, classes)[1:-1])
    out[positive] = 1 + np.searchsorted(edges, backlog[positive], side='right')
    return out


def phase_labels(displacement, capacity, backlog, step, classes, tail=.0005):
    upper = int(np.floor(np.quantile(displacement, 1 - tail) / step))
    displacement_class = np.minimum(np.floor(displacement / step + 1e-8).astype(int), upper)
    return np.c_[displacement_class, capacity, backlog_classes(backlog, classes)], upper * step


def fit_history_phase(displacement, capacity, backlog, step, classes, tail=.0005):
    """Empirical transition kernel on the largest strongly connected class of observed phases."""
    labels, upper = phase_labels(displacement, capacity, backlog, step, classes, tail)
    phases, encoded = np.unique(labels, axis=0, return_inverse=True)
    encoded = encoded.ravel()
    size = len(phases)
    pairs, counts = np.unique(encoded[:-1] * size + encoded[1:], return_counts=True)
    rows, columns = pairs // size, pairs % size
    graph = csr_matrix((counts, (rows, columns)), shape=(size, size))
    _, component = connected_components(graph, directed=True, connection='strong')
    visits = np.bincount(component, weights=np.bincount(encoded, minlength=size))
    keep = component == np.argmax(visits)
    inside = keep[rows] & keep[columns]
    index = np.cumsum(keep) - 1
    rows, columns, kept = index[rows[inside]], index[columns[inside]], counts[inside]
    totals = np.bincount(rows, weights=kept, minlength=int(keep.sum()))
    transition = csr_matrix((kept / totals[rows], (rows, columns)), shape=(int(keep.sum()),) * 2)
    removed = 1 - kept.sum() / counts.sum()
    chosen = phases[keep]
    return dict(displacement=chosen[:, 0] * step, capacity=chosen[:, 1].astype(int),
                backlog_class=chosen[:, 2].astype(int), transition=transition,
                upper_displacement=upper, removed_transition_fraction=float(removed))


def phase_states(displacement, capacity, backlog, step, classes, tail=.0005):
    labels, _ = phase_labels(displacement, capacity, backlog, step, classes, tail)
    return len(np.unique(labels, axis=0))


def choose_resolution(displacement, capacity, backlog, headway):
    """Grid h/16 and the most backlog classes whose phase has at most STATE_LIMIT states."""
    step = headway / GRID_DIVISOR
    for classes in CLASS_LADDER:
        states = phase_states(displacement, capacity, backlog, step, classes)
        if states <= STATE_LIMIT:
            return step, classes, states
    raise ValueError('Even the first-order phase exceeds the state limit.')


def stationary_phase(transition):
    size = transition.shape[0]
    matrix = (transition.T - identity(size, format='csr')).tolil()
    matrix[size - 1, :] = np.ones(size)
    rhs = np.zeros(size)
    rhs[-1] = 1
    phase = np.maximum(spsolve(matrix.tocsc(), rhs), 0)
    return phase / phase.sum()


def net_input_chain(delta, capacity, transition, phase, rate, headway, blocks=BLOCKS):
    """Per-vehicle variance of sum(A_l - c_l) over each block length under the fitted phase."""
    d, c = np.asarray(delta, float), np.asarray(capacity, float)
    md, mc = phase @ d, phase @ c
    vd, vc = d.copy(), c.copy()
    gcc, gdd, gcd, gdc = ([phase @ (c * c) - mc ** 2], [phase @ (d * d) - md ** 2],
                          [phase @ (c * d) - mc * md], [phase @ (d * c) - md * mc])
    for _ in range(max(blocks)):
        vd, vc = transition @ vd, transition @ vc
        gcc.append(phase @ (c * vc) - mc ** 2)
        gdd.append(phase @ (d * vd) - md ** 2)
        gcd.append(phase @ (c * vd) - mc * md)
        gdc.append(phase @ (d * vc) - md * mc)
    gcc, gdd, gcd, gdc = map(np.array, (gcc, gdd, gcd, gdc))
    out = {}
    for m in blocks:
        lags = np.arange(1, m + 1)
        var_c = m * gcc[0] + 2 * np.sum((m - lags[:-1]) * gcc[1:m])
        var_d = 2 * gdd[0] - 2 * gdd[m]
        cov = np.sum(gcd[m - lags] - gdc[lags])
        out[m] = float((rate * m * headway + rate ** 2 * var_d + var_c - 2 * rate * cov) / m)
    return out


def net_input_calibration(displacement, capacity, rate, headway, blocks=BLOCKS):
    """The same per-vehicle variance computed from the calibration path."""
    z = rate * (headway + np.diff(displacement)) - np.asarray(capacity, float)[1:]
    s = np.cumsum(np.r_[0, z - z.mean()])
    return {m: float(rate * headway + (s[m:] - s[:-m]).var() / m) for m in blocks}
