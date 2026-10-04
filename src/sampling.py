import numpy as np
import sympy as sym
import math
from scipy.stats import qmc
from src.utils import *


def generate_values(terms, *args):
    ret = []
    for reg in args:
        sampled_values = generate_sampled_terms(terms, reg)
        ret.append(sampled_values)
    return tuple(ret)

def normalizeComplex(z, imag_min, imag_max):
    # Estrai modulo
    r = np.abs(z)
    newImaginary = np.random.uniform(imag_min, imag_max)
    newReal = np.sqrt(r ** 2 - newImaginary ** 2)

    if np.real(z) < 0:
        newReal = -newReal

    newComplex = newReal + 1j * newImaginary
    return newComplex

def sample_states(constraints, z, n_samples=100):
    """
    Sampling of quantum states using quasi-Monte Carlo methods (Sobol sequence).

    Parameters:
    - constraints: List of dictionaries, each with keys:
        - 'variables': list of sympy symbols (variables involved in the constraint)
        - 'min': minimum sum of probabilities over these variables
        - 'max': maximum sum of probabilities over these variables
    - z: List of sympy symbols representing the state variables
    - n_samples: Number of samples to generate

    Returns:
    - samples: List of dictionaries, each mapping variables to complex amplitudes
    """
    samples = []
    if len(constraints) == 0:
        constraints = [{'variables': [], 'max': 1.0, 'min': 0.0}]

    for constr in constraints:
        variables = constr['variables']
        min_sum = max(constr['min'], 0.0)
        max_sum = min(constr['max'], 1.0)
        if max_sum < min_sum:
            raise ValueError("Infeasible constraint: max_sum < min_sum")

        # Generate quasi-random numbers for the total probability mass S_constr
        # Here, we use 1-dimensional Sobol sequence
        n_samples= 2 ** (math.ceil(math.log2(n_samples)))
        sampler_S = qmc.Sobol(d=1, scramble=False)
        S_constr_samples = sampler_S.random(n=n_samples).flatten()
        # Scale samples to [min_sum, max_sum]
        S_constr_samples = min_sum + S_constr_samples * (max_sum - min_sum)

        # Prepare the Sobol sampler for variable probabilities
        l = len(variables)
        sampler_vars = qmc.Sobol(d=l - 1 if l > 1 else 1, scramble=False)
        vars_samples = sampler_vars.random(n=n_samples)

        # The variables left over after this constraint's own variables are
        # assigned are the same for every sample, so the (deterministic,
        # unscrambled) Sobol sequence used to split the residual mass among
        # them only needs to be generated once per constraint. Generating it
        # inside the per-sample loop, as before, made sampling quadratic in
        # n_samples while producing identical values.
        l_remain_all = len([v for v in z if v not in set(variables)])
        if l_remain_all > 0:
            sampler_remain = qmc.Sobol(d=l_remain_all - 1 if l_remain_all > 1 else 1, scramble=False)
            remain_samples = sampler_remain.random_base2(m=int(np.ceil(np.log2(n_samples))))

        for idx in range(n_samples):
            S_constr = S_constr_samples[idx]
            sample = {v: 0 for v in z}
            res = 1.0 - S_constr  # Residual probability mass
            assigned_vars = set()
            vars_to_assign = variables.copy()
            l = len(vars_to_assign)
            if l > 0:
                # Get the quasi-random numbers for the variables
                if l == 1:
                    random_points = []
                else:
                    random_points = vars_samples[idx, :].tolist()
                # Sort and scale the random points
                random_points = sorted(random_points)
                random_points = [0.0] + random_points + [1.0]
                # Compute the differences to get the proportions
                p_i_list = [S_constr * (random_points[i + 1] - random_points[i]) for i in range(len(random_points) - 1)]
                # Assign probabilities to variables
                for v, p_i in zip(vars_to_assign, p_i_list):
                    ampl = np.random.uniform(0, 2 * np.pi)
                    zi = np.sqrt(p_i) * np.exp(1j * ampl)
                    if v in constr['imConstr'].keys():
                        zi = normalizeComplex(zi, constr['imConstr'][v][0], constr['imConstr'][v][1])
                    sample[v] = zi
                    sample[sym.conjugate(v)] = np.conj(zi)
                    assigned_vars.add(v)
            else:
                res = 1.0  # All probability mass remains

            # Assign remaining variables
            remaining_vars = [v for v in z if v not in assigned_vars]
            l_remain = len(remaining_vars)
            if l_remain > 0:
                remain_idx = idx % len(remain_samples)
                random_points = remain_samples[remain_idx, :].tolist() if l_remain > 1 else []
                # Sort and scale the random points
                random_points = sorted(random_points)
                random_points = [0.0] + random_points + [1.0]
                # Compute the differences to get the proportions
                p_i_list = [res * (random_points[i + 1] - random_points[i]) for i in range(len(random_points) - 1)]
                # Assign probabilities to remaining variables
                for v, p_i in zip(remaining_vars, p_i_list):
                    ampl = np.random.uniform(0, 2 * np.pi)
                    zi = np.sqrt(p_i) * np.exp(1j * ampl)
                    sample[v] = zi
                    sample[sym.conjugate(v)] = np.conj(zi)
            else:
                if abs(res) > 1e-8:
                    raise ValueError("No variables left to assign, but residual probability is non-zero")

            # Append the sample
            samples.append(sample)

    return samples

def sample(Z, n_samples, *args):
    ret = []
    for region in args:
        samples = sample_states(region, Z, n_samples)
        ret.append(samples)
    return tuple(ret)

def generate_sampled_terms(terms : list[tuple], samples) -> list[list]:
    # Evaluate every term (a tuple of symbols to multiply together) at every
    # sample. Rather than looping sample x term in pure Python (which is
    # what a naive read of "evaluate each term at each sample" suggests),
    # look up each *symbol* once per sample (there are far fewer distinct
    # symbols than terms, since terms are degree<=2 combinations of them),
    # then build each term's column via a couple of vectorized numpy
    # multiplies over all samples at once.
    all_symbols = sorted({s for term in terms for s in term}, key=str)
    sym_index = {s: j for j, s in enumerate(all_symbols)}

    values = np.empty((len(samples), len(all_symbols)), dtype=complex)
    for j, s in enumerate(all_symbols):
        values[:, j] = [sample.get(s, s) for sample in samples]

    sampled = np.empty((len(samples), len(terms)), dtype=complex)
    for i, term in enumerate(terms):
        if len(term) == 0:
            sampled[:, i] = 1.0
        else:
            prod = values[:, sym_index[term[0]]]
            for s in term[1:]:
                prod = prod * values[:, sym_index[s]]
            sampled[:, i] = prod

    return np.stack([sampled.real, sampled.imag], axis=-1)

def generate_fx_samples(samples, Z, op):
    dynamic_samples = []
    for sample in samples:
        new_sample = {}
        new_state = apply_op([sample[z] for z in Z], op)
        for i, z in enumerate(Z):
            new_sample[z] = new_state[i]
            new_sample[z.conjugate()] = np.conj(new_state[i])
        dynamic_samples.append(new_sample)
    return dynamic_samples

def generate_k_fx_samples(samples, Z, ops):
    dynamic_samples = []
    for sample in samples:
        new_sample = {}
        new_state= [sample[z] for z in Z]
        for op in ops: new_state = apply_op(new_state, op)
        for i, z in enumerate(Z):
            new_sample[z] = new_state[i]
            new_sample[z.conjugate()] = np.conj(new_state[i])
        dynamic_samples.append(new_sample)
    return dynamic_samples