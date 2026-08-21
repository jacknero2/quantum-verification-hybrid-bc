import logging
import time
from src.sampling import *
from src.constants import *
from src.typings import *
from src.utils import *
from src.linear_problem import *
from src.check_bc import *

logger = logging.getLogger("synthesizeBC")
sampling_logger = logging.getLogger("sampling")

def find_bc(circuit: Circuit,
            type_bc,
            n_samples: int,
            Z : list[sym.Symbol],
            Z0 : list[dict],
            Zu : list[dict],
            deg: int,
            log_level = logging.INFO,
            opt_meth = HIGHSIPM,
            **kwargs):

    logger.setLevel(log_level)
    sampling_logger.setLevel(log_level)
    start = time.perf_counter()

    if type_bc == FINITE_HORIZON:
        required_keys = ["steps"]
        circuit = circuit[0]
    elif type_bc == INFINITE_HORIZON:
        required_keys = ["k", "epsilon","gamma"]
    else:
        raise ValueError("Unknown type_bc")

    for key in required_keys:
        if key not in kwargs: raise ValueError("Required key '%s' not provided" % key)

    # If True, sample once and reuse the same scenario set across every
    # template-growth iteration instead of redrawing fresh samples each
    # time. Opt-in and FINITE_HORIZON-only: default behavior (fresh
    # samples per iteration) is unchanged unless explicitly requested.
    persistent_samples = bool(kwargs.get("persistent_samples", False))
    if type_bc == FINITE_HORIZON and persistent_samples:
        sampling_logger.info("Sampling states (persisted across template growth)...")
        i_samples, u_samples, d_samples = sample(Z, n_samples, Z0, Zu, [])
        dynamic_samples = generate_fx_samples(d_samples, Z, circuit)
        sampling_logger.info("Samples made.")

    # Generate variables
    var = generate_variables(Z)
    # Generating terms
    all_terms = generate_terms(var, deg)
    # Sort terms
    all_terms = sort_terms(all_terms)

    init_terms = [all_terms[0]]
    num_coefficients = None
    q, verification_time, barrier_certificate = None, None, None
    certified = False

    for term in all_terms[1:]:
        init_terms.append(term)
        l = len(init_terms)

        # If the chosen bc is finite horizon
        if type_bc == FINITE_HORIZON:
            # Sampling
            steps = int(kwargs["steps"])
            if not persistent_samples:
                sampling_logger.info("Sampling states...")
                i_samples, u_samples, d_samples = sample(Z, n_samples, Z0, Zu, [])
                dynamic_samples = generate_fx_samples(d_samples, Z, circuit)
                sampling_logger.info("Samples made.")

            num_coefficients = 2 * l + 3
            c = np.zeros(num_coefficients)
            c[-1] = -1
            bounds = [(-np.inf, np.inf)] * num_coefficients
            y_upper_bound = np.random.uniform(1, 10)
            bounds[-1] = (-y_upper_bound, y_upper_bound)

            i_values, u_values, dynamic_values, d_values = generate_values(init_terms, i_samples, u_samples, d_samples, dynamic_samples)

            
            # Generate all LP problem constraints
            logger.info("Making constraints for the linear optimization problem...")
            Aub, bub = generate_all_constraints_fin(i_values, u_values, dynamic_values, d_values, steps, True)
            logger.info("Constraints made.")

            # LP problem
            logger.info("Solving LP problem...")
            barrier_certificate, a, optimals = (solve_lp_fin(c, Aub, bub, bounds, l, init_terms, var, opt_meth))
            end = time.perf_counter()
            generation_time = end - start
            if a is None:
                logger.info("No valid candidate found.")
                logger.info("Extending template...")
                continue
            logger.info("Candidate barrier certificate found")
            logger.info("Time for the generation of the candidate: " + str(generation_time))
            start = time.perf_counter()
            # Chek barrier
            q = check_barrier_fin(Z, barrier_certificate, circuit, var, Z0, Zu, optimals[0], optimals[2], optimals[1])
            end = time.perf_counter()
            verification_time = end - start


        elif type_bc == INFINITE_HORIZON:
            k = int(kwargs["k"])
            epsilon = float(kwargs["epsilon"])
            gamma = float(kwargs["gamma"])

            num_coefficients = len(circuit) * 2 * l + 1
            c = np.zeros(num_coefficients)
            c[-1] = -1
            bounds = [(-np.inf, np.inf)] * num_coefficients
            y_upper_bound = np.random.uniform(1, 10)
            bounds[-1] = (-y_upper_bound, y_upper_bound)

            sampling_logger.info("Sampling states...")
            i_samples, time_samples, k_samples = sample(Z, n_samples, Z0, [], [])
            k_dynamic_samples = generate_k_fx_samples(k_samples, Z, circuit*k if len(circuit)<k else circuit)
            i_values, time_values, k_values, k_dynamic_values = generate_values(init_terms, i_samples, time_samples, k_samples, k_dynamic_samples)
            i_values, time_values, k_values, k_dynamic_values = [i_values], [time_values], [k_values], [k_dynamic_values]

            # create unsafe and dynamic constraints for each barrier function
            u_values, dynamic_initial_values, dynamic_values = [], [], []
            for unitary in circuit:
                u_samples, d_samples = sample(Z, n_samples, Zu, [])
                u_values.append(generate_sampled_terms(init_terms, u_samples))

                dynamic_samples = generate_fx_samples(d_samples, Z, unitary)
                dynamic_initial_values.append(generate_sampled_terms(init_terms, d_samples))
                dynamic_values.append(generate_sampled_terms(init_terms, dynamic_samples))
            sampling_logger.info("Samples made.")

            logger.info("Making constraints for the linear optimization problem...")
            Aub, bub = generate_all_constraints_inf(i_values,u_values,dynamic_values,dynamic_initial_values,time_values,k_dynamic_values,k_values,k,epsilon,gamma,unitaries=circuit)

            logger.info("Constraints made.")
            barrier_certificate, a, y = solve_lp_inf(c, Aub, bub, bounds, l, len(circuit), init_terms, var)
            end = time.perf_counter()
            generation_time = end - start

            if y > k * (epsilon + gamma):
                logger.info("Candidate barrier certificate found")
                for barrier in barrier_certificate: logger.info(str(barrier))
                logger.info("Time for the generation of the candidate: " + str(generation_time))
                start = time.perf_counter()
                q = check_barrier_inf(Z, barrier_certificate, circuit, Z0, Zu, y, epsilon, gamma, k)
                end = time.perf_counter()
                verification_time = end - start
            else:
                logger.info("No valid candidate found.")
                logger.info("Extending template...")
                continue

        if not q:
            logger.info("Barrier certificate is verified by SMT solver")
            logger.info("Time for the verification: " + str(verification_time))
            logger.info(str(barrier_certificate))
            certified = True
            break
        else:
            logger.info("No valid candidate found.")
            logger.info("Extending template...")

    timings: Timings = {"generation_time": generation_time, "verification_time": verification_time}
    return certified, (barrier_certificate if certified else None), timings
