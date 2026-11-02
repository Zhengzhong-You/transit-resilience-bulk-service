"""Root-free contour moments for iid bulk service (Oblakova et al., 2019, Eq.20).

Variance follows by differentiating the same logarithmic factorization; neither
quadrature refinement nor sampled contour counts is an exact numerical certificate.
"""

import time

import numpy as np
from scipy.special import ndtr, erfcx


def effective_arrival_pgf(capacity, utilization, availability=1., headway=4.8, sigma=0.):
    """A(z)=E[z^(Y+C-S)], S~Bin(C,availability), independent of mixed-Poisson Y."""
    mu = headway
    if sigma == 0:
        hm = np.array([mu, mu ** 2, mu ** 3])
    else:
        a = mu / sigma
        phi = np.exp(-a ** 2 / 2) / np.sqrt(2 * np.pi)
        hm = np.array([mu * ndtr(a) + sigma * phi,
            (mu ** 2 + sigma ** 2) * ndtr(a) + mu * sigma * phi,
            (mu ** 3 + 3 * mu * sigma ** 2) * ndtr(a)
            + (mu ** 2 * sigma + 2 * sigma ** 3) * phi])
    rate = utilization * capacity * availability / hm[0]
    y = hm * rate ** np.arange(1, 4)
    u = np.array([capacity * (1 - availability),
                  capacity * (capacity - 1) * (1 - availability) ** 2,
                  capacity * (capacity - 1) * (capacity - 2) * (1 - availability) ** 3])
    derivatives = np.array([y[0] + u[0], y[1] + 2 * y[0] * u[0] + u[1],
                           y[2] + 3 * y[1] * u[0] + 3 * y[0] * u[1] + u[2]])

    def values(z):
        z = np.asarray(z, dtype=complex)
        t = rate * (z - 1)
        if sigma == 0:
            yp = np.exp(mu * t)
            yd = rate * mu * yp
        else:
            weighted = (.5 * np.exp(-.5 * (mu / sigma) ** 2)
                        * erfcx(-(mu / sigma + sigma * t) / np.sqrt(2)))
            yp = ndtr(-mu / sigma) + weighted
            yd = rate * ((mu + sigma ** 2 * t) * weighted
                         + sigma * np.exp(-.5 * (mu / sigma) ** 2) / np.sqrt(2 * np.pi))
        base = availability + (1 - availability) * z
        up = base ** capacity
        ud = capacity * (1 - availability) * base ** (capacity - 1)
        return yp * up, yd * up + yp * ud

    def characteristic(z):
        return z ** capacity - values(z)[0]

    characteristic.scale = lambda z: max(1., abs(z ** capacity), abs(values(z)[0]))
    return values, derivatives, characteristic


def contour_moments(capacity, pgf, tolerance=1e-8, maximum_points=1048576):
    start = time.perf_counter()
    radius = 1.05
    for _ in range(40):
        value = float(pgf(radius)[0].real)
        if value < radius ** capacity:
            break
        radius = 1 + (radius - 1) / 2
    else:
        raise RuntimeError('No numerically separated zero-free contour was found.')
    previous = None
    points = 1024
    while points <= maximum_points:
        angles = 2 * np.pi * (np.arange(points) + .5) / points
        z = radius * np.exp(1j * angles)
        a, derivative = pgf(z)
        d = z ** capacity - a
        dp = capacity * z ** (capacity - 1) - derivative
        # Remove the known zero at z=1 before applying the residue identity.
        logarithmic = dp / d - 1 / (z - 1)
        first = np.mean(logarithmic * z / (1 - z))
        second = np.mean(logarithmic * z / (1 - z) ** 2)
        current = np.array([first.real, (first - second).real])
        imaginary = max(abs(first.imag), abs((first - second).imag))
        if previous is not None:
            error = float(np.max(np.abs(current - previous) / np.maximum(1., np.abs(current))))
            if error < tolerance and imaginary < tolerance * max(1., abs(current).max()):
                return {'mean_residual': float(current[0]), 'variance_residual': float(current[1]),
                        'points': points, 'radius': radius, 'relative_refinement_change': error,
                        'imaginary_residual': float(imaginary),
                        'elapsed_seconds': time.perf_counter() - start}
        previous = current
        points *= 2
    raise RuntimeError('Contour moments did not converge under point refinement.')


def moments_from_roots(roots, capacity, derivatives):
    roots = np.asarray(roots)
    unit = np.argmin(abs(roots - 1))
    others = np.delete(roots, unit)
    d1 = capacity - derivatives[0]
    d2 = capacity * (capacity - 1) - derivatives[1]
    d3 = capacity * (capacity - 1) * (capacity - 2) - derivatives[2]
    first = np.sum(1 / (1 - others)) - d2 / (2 * d1)
    second_log = -np.sum(1 / (1 - others) ** 2) - d3 / (3 * d1) + (d2 / (2 * d1)) ** 2
    return first, first + second_log
