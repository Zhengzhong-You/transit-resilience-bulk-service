"""Fixed inputs for the integrated revision's independent validation and policies."""

import json

import numpy as np

from route_validation import cut_loads, destination_law


BASE_RATES = np.array([.75, 2., .75, .5, .5, 1., .75, 1.5, .2, 0.])
BASE_ALIGHTING = np.array([0., 0., .1, .25, .25, .8, .5, .3, .75, 1.])


def heterogeneous_destinations():
    p = np.zeros((10, 10))
    for origin in range(9):
        survival = 1.
        for destination in range(origin + 1, 10):
            p[origin, destination] = survival * BASE_ALIGHTING[destination]
            survival *= 1 - BASE_ALIGHTING[destination]
    return p


def scenario(name, rates, destinations, capacity=34, headway=4.8,
             gamma=.2, mean_duration=1., group='validation'):
    rates, destinations = np.asarray(rates), np.asarray(destinations)
    loads = cut_loads(rates, destinations, headway) / capacity
    return dict(name=name, rates=rates.tolist(), destinations=destinations.tolist(),
                capacity=capacity, headway=headway, gamma=gamma,
                mean_duration=mean_duration, group=group,
                cut_loads=loads.tolist(), maximum_cut_load=float(loads.max()),
                first_failing_station=(int(np.flatnonzero(loads >= 1)[0]) + 1
                                       if np.any(loads >= 1) else None))


def scaled_rates(rates, destinations, maximum_load, capacity=34, headway=4.8):
    return (np.asarray(rates) * maximum_load * capacity
            / cut_loads(rates, destinations, headway).max())


def validation_profiles():
    result = []
    p = destination_law()
    for eta in [.8, .9, .97]:
        rates = scaled_rates(np.r_[np.ones(9), 0.], p, eta, capacity=50)
        for gamma in [0., .02, .2]:
            name = f'uniform-e{int(eta * 100):03d}-g{int(gamma * 100):02d}'
            result.append(scenario(name, rates, p, capacity=50, gamma=gamma))
    p = heterogeneous_destinations()
    reference = BASE_RATES * .75
    for eta in [.8, .9, .97]:
        result.append(scenario(f'heterogeneous-e{int(eta * 100):03d}',
                               scaled_rates(reference, p, eta), p))
    # These reproduce the original seven validation inputs as desired dispatch
    # intervals, without treating a mean-cycle sizing rule as fleet feasibility.
    result.extend([
        scenario('original-reference', reference, p),
        scenario('original-C30', reference, p, capacity=30),
        scenario('original-long-headway', reference, p, headway=60 / 7),
        scenario('original-demand1', BASE_RATES, p),
        scenario('original-no-incidents', reference, p, headway=4., gamma=0.),
        scenario('original-frequent-incidents', reference, p, headway=16 / 3, gamma=1 / 3),
        scenario('original-long-incidents', reference, p, headway=5.6, mean_duration=2.)])
    return result


def policy_cases():
    p = heterogeneous_destinations()
    reference = BASE_RATES * .75
    result = []
    for family, rates in [('original', reference),
                           ('near-capacity', scaled_rates(reference, p, .97))]:
        result.append(scenario(f'{family}-policy-reference', rates, p, group='policy'))
        for capacity in [30, 38]:
            result.append(scenario(f'{family}-C{capacity}', rates, p,
                                   capacity=capacity, group='policy'))
        for duration in [.5, 2.]:
            result.append(scenario(f'{family}-duration{duration:g}', rates, p,
                                   mean_duration=duration, group='policy'))
        for h in [2.4, 60 / 7]:
            result.append(scenario(f'{family}-headway{h:.6g}', rates, p,
                                   headway=h, group='policy'))
        for gamma in [0., .1, 1 / 3]:
            result.append(scenario(f'{family}-gamma{gamma:.6g}', rates, p,
                                   gamma=gamma, group='policy'))
        for factor in [.5, 1.]:
            result.append(scenario(f'{family}-policy-demand{factor:g}', rates * factor / .75,
                                   p, group='policy'))
    return result


def input_key(case):
    return json.dumps({k: case[k] for k in ['rates', 'destinations', 'capacity',
                      'headway', 'gamma', 'mean_duration']}, sort_keys=True)


def all_scenarios():
    result, lookup = [], {}
    for case in validation_profiles() + policy_cases():
        key = input_key(case)
        if key not in lookup:
            case = dict(case, aliases=[case['name']], phase_validation=case['group'] == 'validation')
            lookup[key] = case
            result.append(case)
        else:
            lookup[key]['aliases'].append(case['name'])
    return result
