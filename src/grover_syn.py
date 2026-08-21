from src.check_bc import *
from src.check_bc import _sympy_to_z3_rec
from src.log import *
from decimal import getcontext
from scipy.stats import qmc
from scipy.optimize import linprog
import sympy as sym
import numpy as np
from math import *
import time
import logging

logger = logging.getLogger("systhesizeBC")
sampling_logger = logging.getLogger("sampling")
linprog_logger = logging.getLogger("linearOpt")


def generate_constraints(
        sampled_angles, c_index, right_hand_value, constraint_type, initial_angles=None
):
    num_samples = len(sampled_angles)
    # We assume a single decision variable c for simplicity.
    A = np.zeros((num_samples, 1))
    b = np.full((num_samples,), right_hand_value, dtype=np.float64)

    if constraint_type == 'initial':
        for i, theta in enumerate(sampled_angles):
            A[i, c_index] = theta
        A = np.hstack([A, -1 * np.ones((num_samples, 1))])
        A = np.hstack([A, np.zeros((num_samples, 1))])
        A = np.hstack([A, np.zeros((num_samples, 1))])



    elif constraint_type == 'unsafe':
        for i, theta in enumerate(sampled_angles):
            A[i, c_index] = -theta
            b[i] = -right_hand_value
        A = np.hstack([A, np.zeros((num_samples, 1))])
        A = np.hstack([A, np.zeros((num_samples, 1))])
        A = np.hstack([A, np.ones((num_samples, 1))])


    elif constraint_type == 'dynamic':
        for i, theta in enumerate(sampled_angles):
            A[i, c_index] = theta - initial_angles[i]
        A = np.hstack([A, np.zeros((num_samples, 1))])
        A = np.hstack([A, -1 * np.ones((num_samples, 1))])
        A = np.hstack([A, np.zeros((num_samples, 1))])

    else:
        raise ValueError("Invalid constraint_type.")

    return A, b


def grover_qmc(upper, lower, num_samples):
    # Create a Sobol sequence sampler for 1D points (quasi-Monte Carlo)
    sampler = qmc.Sobol(d=1, scramble=False)
    sample = sampler.random(n=num_samples).flatten()
    sample = np.append(sample, [1])

    # Scale the Sobol samples from [0, 1) to [lower_bound_f, upper_bound_f]
    delta = upper - lower
    angles = lower + sample * delta
    return angles


def constraints_Grover(N, M, err, num_samples, mu):
    # Increase precision for high-precision arithmetic
    getcontext().prec = 500

    # Define M_low and M_high with high precision
    M_low = M - err
    M_high = M + err

    # Compute bounds: theta = asin(sqrt(M/N))
    lower_bound_i = asin(sqrt(M_low / N))
    upper_bound_i = asin(sqrt(M_high / N))

    upper_bound_u = (11 / 6) * np.pi
    lower_bound_u = (9 / 6) * np.pi

    theta = 2 * asin(sqrt(M / N))
    G = grover_qmc(theta - mu, theta + mu, num_samples)

    sampling_logger.info("Sampling initial angles...")
    i_angles = grover_qmc(upper_bound_i, lower_bound_i, num_samples)
    sampling_logger.info("Initial angles sampled.")
    sampling_logger.info("Sampling unsafe angles...")
    u_angles = grover_qmc(upper_bound_u, lower_bound_u, num_samples)
    sampling_logger.info("Unsafe angles sampled.")
    sampling_logger.info("Samplng all possible angles...")
    d_angles = grover_qmc(2 * np.pi, 0, num_samples)
    d_angles_G = [(d_angles[i] + G[i]) for i in range(len(d_angles))]
    sampling_logger.info("Angles sampled.")

    c = np.zeros(4)
    c[-1] = -1

    logger.info("Generating linear oprtimization constraints...")
    A_safe, b_safe = generate_constraints(i_angles, c_index=0, right_hand_value=0, constraint_type='initial')

    # Generate LP constraints for the unsafe set: B(theta) = c*cos(theta) >= theta_b
    A_unsafe, b_unsafe = generate_constraints(u_angles, c_index=0, right_hand_value=0, constraint_type='unsafe')
    A_dynamic, b_dynamic = generate_constraints(d_angles_G, c_index=0, right_hand_value=0, constraint_type='dynamic',
                                                initial_angles=d_angles)
    A = np.vstack([A_safe, A_unsafe, A_dynamic])
    b = np.hstack([b_safe, b_unsafe, b_dynamic])
    logger.info("Constraints made.")
    return A, b, lower_bound_i, upper_bound_i, lower_bound_u, upper_bound_u


def find_BC_grover(N_QUBIT, n_samples, err, mu, num_solutions, log_level=logging.INFO):
    logger.setLevel(log_level)
    sampling_logger.setLevel(log_level)
    linprog_logger.setLevel(log_level)

    logger.info("Setting parameters...")
    N = pow(2, N_QUBIT)
    logger.info("Number of states K: " + str(N))
    M = num_solutions
    logger.info("Number of solution states M: " + str(M))
    start = time.perf_counter()
    T = ceil((pi / 4) * sqrt(N / M))
    logger.info("Bound for grover iterations T: " + str(T))
    theta = 2 * asin(sqrt(M / N))
    c = np.zeros(4)
    c[-1] = -1
    Aub, bub, low_i, up_i, low_u, up_u = constraints_Grover(N, M, err, n_samples, mu)
    Aub = np.vstack([Aub, [0, 1, T, -1]])
    bub = np.hstack([bub, [-0.00001]])
    coeff_bounds = np.inf
    bounds = [(-coeff_bounds, coeff_bounds)] * 4
    y_upper_bound = 100
    bounds[-1] = (0.001, y_upper_bound)
    linprog_logger.info("Solving linear optimization problem...")
    result = linprog(c, A_ub=Aub, b_ub=bub, bounds=bounds, method='highs')
    linprog_logger.info("Linear program terminated.")
    if result.success:
        x_optimal = result.x
        linprog_logger.info("Optimization completed: " + result.message)
        lambda_ = x_optimal[-1]
        gamma = x_optimal[1]
        delta = x_optimal[2]
        coeff = x_optimal[0]
        print(x_optimal)
        linprog_logger.info("Optimal lambda: " + str(lambda_))
        linprog_logger.info("Optimal c:" + str(coeff))
        linprog_logger.info("Optimal gamma:" + str(gamma))
        linprog_logger.info("Optimal delta:" + str(delta))
        barrier_phi = sym.poly(coeff * sym.Symbol("phi"))
        end = time.perf_counter()
        generation_time = end - start
        logger.info("Barrier certificate candidate found.")
        logger.info(str(barrier_phi))
        logger.info("Time to generation: " + str(generation_time))
        logger.info("SMT solver checking...")
        start = time.perf_counter()
        real_barrier = sym.poly(sym.re(sym.expand_complex(barrier_phi.as_expr())))
        Z_RI = [sym.re("phi"), sym.re("theta")]
        var_z3_dict = dict(zip(Z_RI, [Real(str(var)) for var in Z_RI]))
        z3_barrier = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())
        z3_barrier_G = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())
        print(z3_barrier)
        logger.info("Initial constraint...")
        initial_conditions = [z3_barrier > gamma + 0.000001, var_z3_dict[sym.re("phi")] >= low_i,
                              var_z3_dict[sym.re("phi")] <= up_i]
        q = run_check(initial_conditions, 300)
        logger.info("Unsafe constraint...")
        unsafe_conditions = [z3_barrier < lambda_ - 0.000001, var_z3_dict[sym.re("phi")] >= low_u,
                             var_z3_dict[sym.re("phi")] <= up_u]
        q = run_check(unsafe_conditions, 300)
        logger.info("Dynamic constraint...")
        # f_z3_barrier = _sympy_to_z3_rec(var_z3_dict, sym.poly(sym.re(sym.expand_complex(sym.poly(x_optimal[0]*(sym.Symbol("phi") + theta)).as_expr()))).as_expr())
        f_z3_barrier = _sympy_to_z3_rec(var_z3_dict, sym.poly(sym.re(sym.expand_complex(
            sym.poly(x_optimal[0] * (sym.Symbol("phi") + sym.Symbol("theta"))).as_expr()))).as_expr())
        z3_phi = var_z3_dict[sym.re("phi")]
        z3_theta = var_z3_dict[sym.re("theta")]
        dynamic_conditions = [f_z3_barrier - z3_barrier > delta + 0.000001, z3_phi >= 0, z3_phi <= 2 * np.pi,
                              z3_theta <= theta + mu, z3_theta >= theta - mu]
        q = run_check(dynamic_conditions, 300)
        end = time.perf_counter()
        verification_time = end - start
        logger.info("Time to verification: " + str(verification_time))

    else:
        linprog_logger.warning("Optimization failed: " + result.message)
    return
