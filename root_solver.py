import math
import time

import numpy as np
from scipy.optimize import least_squares


def scaled_residual(characteristic, z):
    value = complex(characteristic(z))
    scale_function = getattr(characteristic, 'scale', None)
    scale = float(scale_function(z)) if scale_function is not None else max(1.0, abs(z) ** 2)
    if not np.isfinite(value.real) or not np.isfinite(value.imag) or not np.isfinite(scale):
        return float('inf')
    return abs(value) / max(1.0, scale)


def cluster_roots(characteristic, candidates, duplicate_tolerance):
    valid = [complex(z) for z in candidates if np.isfinite(z.real) and np.isfinite(z.imag)]
    valid.sort(key=lambda z: scaled_residual(characteristic, z))
    clusters = []
    for candidate in valid:
        if not clusters:
            clusters.append(candidate)
            continue
        distances = np.asarray([abs(candidate - center) for center in clusters])
        nearest = int(np.argmin(distances))
        if distances[nearest] > duplicate_tolerance:
            clusters.append(candidate)
    return clusters


def winding_number(characteristic, center, radius, samples=1024):
    angles = np.linspace(0.0, 2.0 * math.pi, samples + 1)
    contour = center + radius * np.exp(1j * angles)
    try:
        values = np.asarray([complex(characteristic(z)) for z in contour])
    except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
        return None
    if not np.all(np.isfinite(values.real)) or not np.all(np.isfinite(values.imag)):
        return None
    scale = max(1.0, float(np.max(np.abs(values))))
    if float(np.min(np.abs(values))) <= 1e-13 * scale:
        return None
    phase_change = np.unwrap(np.angle(values))[-1] - np.unwrap(np.angle(values))[0]
    raw_count = phase_change / (2.0 * math.pi)
    count = int(np.rint(raw_count))
    if count < 0 or abs(raw_count - count) > 5e-3:
        return None
    return count


def certify_multiplicities(characteristic, centers, duplicate_tolerance):
    """Estimate zero counts around numerical clusters, including nearby simple zeros.

    The legacy name is retained for compatibility. A sampled winding count does
    not determine algebraic multiplicity and is not an interval certificate.
    """
    multiplicities = []
    for index, center in enumerate(centers):
        distances = [abs(center - other) for j, other in enumerate(centers) if j != index]
        maximum_radius = min(0.05, 0.35 * min(distances)) if distances else 0.05
        if maximum_radius <= 8.0 * duplicate_tolerance:
            multiplicities.append(None)
            continue

        counts = []
        for fraction in 0.5 ** np.arange(12):
            radius = maximum_radius * fraction
            if radius <= 8.0 * duplicate_tolerance:
                break
            coarse = winding_number(characteristic, center, radius, samples=512)
            fine = winding_number(characteristic, center, radius, samples=1024)
            if coarse is not None and coarse == fine and coarse > 0:
                counts.append(coarse)
                if counts[-2:] == [1, 1]:
                    break
        if len(counts) >= 2 and counts[-2] == counts[-1]:
            multiplicities.append(counts[-1])
        else:
            multiplicities.append(None)
    return multiplicities


def _solve_local(characteristic, initial_value, residual_tolerance, disk_tolerance):
    def residual(parts):
        z = complex(parts[0], parts[1])
        try:
            value = complex(characteristic(z))
            scale_function = getattr(characteristic, 'scale', None)
            scale = float(scale_function(z)) if scale_function is not None else max(1.0, abs(z) ** 2)
            result = np.asarray([value.real / scale, value.imag / scale])
        except (FloatingPointError, OverflowError, ValueError, ZeroDivisionError):
            result = np.asarray([1e6, 1e6])
        if not np.all(np.isfinite(result)):
            return np.asarray([1e6, 1e6])
        return result

    result = least_squares(
        residual,
        np.asarray([initial_value.real, initial_value.imag]),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
        max_nfev=2000,
    )
    root = complex(result.x[0], result.x[1])
    root = _polish_root(characteristic, root, 1, residual_tolerance)
    if not result.success or abs(root) > 1.0 + disk_tolerance:
        return None
    if scaled_residual(characteristic, root) > residual_tolerance:
        return None
    return root


def _ordered_root_list(centers, multiplicities):
    paired = list(zip(centers, multiplicities))
    paired.sort(key=lambda item: np.angle(item[0]) % (2.0 * math.pi))
    unit_index = int(np.argmin([abs(center - 1.0) for center, _ in paired]))
    unit_center, unit_multiplicity = paired.pop(unit_index)
    if abs(unit_center - 1.0) <= 1e-5:
        unit_center = complex(1.0, 0.0)
    paired.insert(0, (unit_center, unit_multiplicity))
    roots = []
    for center, multiplicity in paired:
        roots.extend([center] * multiplicity)
    return np.asarray(roots, dtype=complex)


def _polish_root(characteristic, root, multiplicity, residual_tolerance):
    polished = complex(root)
    for _ in range(30):
        value = complex(characteristic(polished))
        if (
            multiplicity == 1
            and scaled_residual(characteristic, polished)
            <= min(1e-13, residual_tolerance / 100.0)
        ):
            break
        step_size = 1e-6 * max(1.0, abs(polished))
        derivative = (
            complex(characteristic(polished + step_size))
            - complex(characteristic(polished - step_size))
        ) / (2.0 * step_size)
        if abs(derivative) <= np.finfo(float).eps:
            break
        updated = polished - multiplicity * value / derivative
        if not np.isfinite(updated.real) or not np.isfinite(updated.imag):
            break
        step = abs(updated - polished)
        polished = updated
        if step <= 1e-13 * max(1.0, abs(polished)):
            break
    return polished


def certified_interpolation_roots(
    characteristic,
    capacity,
    utilization,
    residual_tolerance=1e-9,
    duplicate_tolerance=1e-4,
    disk_tolerance=1e-8,
    max_iterations=None,
    max_interpolation=None,
):
    """Find separated simple roots and check residuals and sampled winding counts.

    The historical public name is retained. Unresolved multiple-zero clusters
    are rejected instead of duplicating one approximate center as a repeated root.
    """
    if capacity < 1:
        raise ValueError('capacity must be positive')
    max_iterations = max_iterations or max(12, capacity)
    max_interpolation = max_interpolation or max(4, int(math.ceil(capacity / 2.0)))

    start_time = time.time()
    initial_radius = float(np.clip(1.0 - 0.5 * utilization, 0.05, 0.98))
    candidates = [complex(1.0, 0.0)]
    for branch in range(capacity):
        angle = 2.0 * math.pi * (branch + 0.5) / capacity
        initial_value = initial_radius * np.exp(1j * angle)
        root = _solve_local(characteristic, initial_value, residual_tolerance, disk_tolerance)
        if root is not None:
            candidates.extend([root, root.conjugate()])

    centers = cluster_roots(characteristic, candidates, duplicate_tolerance)
    ordered_centers = sorted(centers, key=lambda z: np.angle(z) % (2.0 * math.pi))
    if len(ordered_centers) >= 2:
        radii = [abs(z) for z in ordered_centers]
        angles = np.unwrap([np.angle(z) % (2.0 * math.pi) for z in ordered_centers])
        for index in range(2, len(ordered_centers)):
            radius = float(np.clip(2.0 * radii[index - 1] - radii[index - 2], 0.01, 1.1))
            angle = 2.0 * angles[index - 1] - angles[index - 2]
            root = _solve_local(
                characteristic,
                radius * np.exp(1j * angle),
                residual_tolerance,
                disk_tolerance,
            )
            if root is not None:
                candidates.extend([root, root.conjugate()])
        centers = cluster_roots(characteristic, candidates, duplicate_tolerance)
    powell_time = time.time() - start_time
    powell_centers = list(centers)
    interpolation_start = time.time()
    interpolation_level = 2
    iteration = 0

    while True:
        multiplicities = certify_multiplicities(characteristic, centers, duplicate_tolerance)
        if any(value is not None and value > 1 for value in multiplicities):
            raise RuntimeError(
                'Unresolved root cluster contains more than one zero; '
                'sampled contour counts do not distinguish close simple roots '
                'from algebraic multiplicity. Reduce the clustering tolerance '
                'or use the root-free calculation.'
            )
        certified_count = sum(value for value in multiplicities if value is not None)
        if all(value is not None for value in multiplicities) and certified_count == capacity:
            centers = [
                _polish_root(characteristic, center, multiplicity, residual_tolerance)
                for center, multiplicity in zip(centers, multiplicities)
            ]
            roots = _ordered_root_list(centers, multiplicities)
            diagnostics = {
                'verification': 'tolerance-based residual and sampled winding checks for separated simple roots',
                'powell_distinct_count': len(powell_centers),
                'powell_time': powell_time,
                'powell_roots': powell_centers,
                'certified_count': certified_count,
                'distinct_count': len(centers),
                'multiplicities': multiplicities,
                'iterations': iteration,
                'interpolation_level': interpolation_level,
                'interpolation_time': time.time() - interpolation_start,
                'largest_residual': max(scaled_residual(characteristic, z) for z in roots),
            }
            return roots, diagnostics
        if certified_count > capacity:
            raise RuntimeError(
                f'Numerically checked root count exceeded expectation: {certified_count} > {capacity}.'
            )

        strict_residual_tolerance = max(1e-13, residual_tolerance / 100.0)
        retained_centers = [
            center
            for center, multiplicity in zip(centers, multiplicities)
            if multiplicity is not None
            or scaled_residual(characteristic, center) <= strict_residual_tolerance
        ]
        if len(retained_centers) < len(centers):
            centers = retained_centers
            candidates = list(centers)
            continue

        if iteration >= max_iterations or interpolation_level > max_interpolation:
            largest_residual = max(
                (scaled_residual(characteristic, z) for z in centers), default=float('inf')
            )
            contour_radius = 1.0 + max(1e-4, 10.0 * disk_tolerance)
            contour_count = winding_number(
                characteristic,
                complex(0.0, 0.0),
                contour_radius,
                samples=4096,
            )
            raise RuntimeError(
                'Numerical root verification incomplete: '
                f'checked {certified_count} of {capacity} roots, '
                f'found {len(centers)} locations, largest residual {largest_residual:.3e}, '
                f'unit disk contour count {contour_count}, iteration {iteration}, '
                f'interpolation level {interpolation_level}.'
            )

        ordered = sorted(centers, key=lambda z: np.angle(z) % (2.0 * math.pi))
        seeds = []
        for index, first in enumerate(ordered):
            second = ordered[(index + 1) % len(ordered)]
            first_radius = abs(first)
            second_radius = abs(second)
            first_angle = np.angle(first) % (2.0 * math.pi)
            second_angle = np.angle(second) % (2.0 * math.pi)
            if second_angle <= first_angle:
                second_angle += 2.0 * math.pi
            for step in range(1, interpolation_level):
                weight = step / interpolation_level
                radius = (1.0 - weight) * first_radius + weight * second_radius
                angle = (1.0 - weight) * first_angle + weight * second_angle
                for radius_factor in (0.85, 1.0, 1.15):
                    perturbed_radius = float(np.clip(radius * radius_factor, 0.01, 1.0))
                    seeds.append(perturbed_radius * np.exp(1j * angle))

        previous_count = len(centers)
        for seed in seeds:
            root = _solve_local(characteristic, seed, residual_tolerance, disk_tolerance)
            if root is not None:
                candidates.extend([root, root.conjugate()])
        centers = cluster_roots(characteristic, candidates, duplicate_tolerance)
        if len(centers) == previous_count:
            interpolation_level += 1
        iteration += 1
