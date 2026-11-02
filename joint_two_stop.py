"""Direct joint residual-queue calculation for two boarding stops, capacity one.

The input kernel is specified from Poisson rates and destination probabilities;
there is no fitted service phase. Finite queue caps are numerical approximations.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.fft import next_fast_len, rfft2, irfft2
from scipy.stats import poisson


def arrival_probabilities(rate, cutoff):
    # Any count above K+1 reaches the capped level after serving at most one.
    return np.r_[poisson.pmf(np.arange(cutoff + 2), rate), poisson.sf(cutoff + 1, rate)]


def reflected_columns(mass, seats, cutoff):
    result = np.empty((mass.shape[0], cutoff + 1))
    result[:, 0] = mass[:, :seats + 1].sum(axis=1)
    result[:, 1:cutoff] = mass[:, seats + 1:seats + cutoff]
    result[:, cutoff] = mass[:, seats + cutoff:].sum(axis=1)
    return result


def joint_operator(rate1, rate2, destination_probability, cutoff):
    if not (0 < rate1 < 1 and 0 < rate2 and
            rate1 * destination_probability + rate2 < 1):
        raise ValueError('Both strict route-cut conditions are required.')
    if not 0 <= destination_probability <= 1 or cutoff < 2:
        raise ValueError('Invalid destination probability or queue cutoff.')
    p1, p2 = arrival_probabilities(rate1, cutoff), arrival_probabilities(rate2, cutoff)
    length = next_fast_len(2 * cutoff + 3)
    kernel = rfft2(np.outer(p1, p2), s=(length, length))

    def apply(joint):
        mass = irfft2(rfft2(joint, s=(length, length)) * kernel,
                       s=(length, length))[:2 * cutoff + 3, :2 * cutoff + 3]
        if mass.min() < -1e-12:
            raise ArithmeticError('Negative joint-transition probability exceeds roundoff.')
        mass = np.maximum(mass, 0)
        # u=R1+Y1>=1: the first stop boards one passenger, whose unseen mark
        # is Bernoulli(p) independently of count history under iid arrival marks.
        positive = np.vstack([mass[1:cutoff + 1], mass[cutoff + 1:].sum(axis=0)])
        updated = ((1 - destination_probability) * reflected_columns(positive, 1, cutoff)
                   + destination_probability * reflected_columns(positive, 0, cutoff))
        # u=0 leaves the entire vehicle available at the second stop.
        updated[0] += reflected_columns(mass[:1], 1, cutoff)[0]
        return updated / updated.sum()

    return apply


def stationary_joint(rate1, rate2, p, cutoff, tolerance=1e-12):
    start = time.perf_counter()
    apply = joint_operator(rate1, rate2, p, cutoff)
    joint = np.zeros((cutoff + 1, cutoff + 1))
    joint[0, 0] = 1.
    for iteration in range(1, 50001):
        updated = apply(joint)
        change = np.abs(updated - joint).sum()
        joint = updated
        if change < tolerance:
            break
    else:
        raise RuntimeError('The direct joint iteration did not converge.')
    levels = np.arange(cutoff + 1)
    first, second = joint.sum(axis=1), joint.sum(axis=0)
    means = np.array([levels @ first, levels @ second])
    variances = np.array([levels ** 2 @ first, levels ** 2 @ second]) - means ** 2
    covariance = float(np.einsum('i,ij,j->', levels, joint, levels) - means.prod())
    rates = np.array([rate1, rate2])
    result = dict(cutoff=cutoff, iterations=iteration,
        balance_residual=float(np.abs(apply(joint) - joint).sum()),
        upper_boundary_probability=float(joint[-1].sum() + joint[:, -1].sum() - joint[-1, -1]),
        mean_residual=means.tolist(), variance_residual=variances.tolist(),
        mean_queue=(means + rates).tolist(), variance_queue=(variances + rates).tolist(),
        mean_wait=(.5 + means / rates).tolist(),
        variance_wait=(1 / 12 + (variances - means) / rates ** 2).tolist(),
        residual_correlation=float(covariance / np.sqrt(variances.prod())),
        joint_product_total_variation=float(.5 * np.abs(joint - np.outer(first, second)).sum()),
        elapsed_seconds=time.perf_counter() - start)
    return joint, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for name, a, b in [('light', .35, .25), ('binding', .75, .55)]:
        previous = None
        for cutoff in [32, 64, 128, 256, 512]:
            joint, result = stationary_joint(a, b, .5, cutoff)
            difference = (max(np.max(np.abs(np.asarray(result[m]) - previous[m]) /
                         np.maximum(1., np.abs(result[m]))) for m in
                         ['mean_queue', 'variance_queue', 'mean_wait', 'variance_wait'])
                          if previous is not None else None)
            result.update(case=name, rate1=a, rate2=b, p=.5, refinement_change=difference)
            rows.append(result)
            (args.output / 'results.json').write_text(json.dumps(rows, indent=2))
            print(name, cutoff, result['mean_residual'], result['upper_boundary_probability'], flush=True)
            if difference is not None and difference < 1e-6 and result['upper_boundary_probability'] < 1e-12:
                np.savez_compressed(args.output / (name + '-joint.npz'), joint=joint)
                break
            previous = result
        else:
            raise RuntimeError('Direct-joint refinement gate was not met.')
    (args.output / 'status.json').write_text(json.dumps({'status': 'completed'}, indent=2))


if __name__ == '__main__':
    main()
