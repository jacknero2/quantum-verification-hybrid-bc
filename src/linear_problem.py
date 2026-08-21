import logging
import numpy as np
import sympy as sym
from typing import Tuple, List
from src.utils import *
from scipy.optimize import linprog
from src.utils import construct_complex_coefficients

logger = logging.getLogger("linearOpt")

def generate_constraints_fin(
    real_imag: np.ndarray,
    constraint_type: str,
    dynamic_real_imag: np.ndarray = None,
    right_hand_value: float = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate constraints for a linear programming (LP) problem.

    Parameters:
    -----------
    real_imag : np.ndarray
        A NumPy array of shape (num_samples, num_terms, 2), where:
        - num_samples is the number of samples.
        - num_terms is the number of terms in each sample.
        - The last dimension contains the real and imaginary parts (Re, Im).
        Represents the polynomial evaluated at certain points.

    constraint_type : str
        The type of constraint to generate.
        - 'initial': constraints for initial samples
        - 'unsafe': constraints for unsafe samples
        - 'dynamic': constraints involving dynamic_real_imag

    dynamic_real_imag : np.ndarray, optional
        Required if constraint_type is 'dynamic'.
        A NumPy array of the same shape as real_imag.
        Represents the dynamic polynomial evaluated at the same points.

    right_hand_value : float, optional
        The right-hand side value for the inequality constraints (default is 0).

    Returns:
    --------
    Aub : np.ndarray
        The inequality constraint matrix (A_ub) for the LP problem.

    bub : np.ndarray
        The inequality constraint vector (b_ub) for the LP problem.

    Aeq : np.ndarray
        The equality constraint matrix (A_eq) for the LP problem.

    beq : np.ndarray
        The equality constraint vector (b_eq) for the LP problem.
    """
    if constraint_type == 'initial':
        # Extract real and imaginary parts
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]
        num_samples, num_terms = re.shape

        # Prepare coefficients for the inequality constraints (Aub)
        # For each sample, interleave re and -im
        Aub_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aub_coeffs[:, 0::2] = re
        Aub_coeffs[:, 1::2] = -im

        # Append -1 for the epsilon variable
        Aub = np.hstack([Aub_coeffs, -1 * np.ones((num_samples, 1))])
        # Append 0 for the sigma variable
        Aub = np.hstack([Aub, np.zeros((num_samples, 1))])
        # Append zero for the 'y' variable
        Aub = np.hstack([Aub, np.zeros((num_samples, 1))])
        # Right-hand side vectors
        bub = np.full(num_samples, right_hand_value, dtype=np.float64)

    elif constraint_type == 'unsafe':
        # Extract real and imaginary parts
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]
        num_samples, num_terms = re.shape

        # Prepare coefficients for the inequality constraints (Aub)
        # For each sample, interleave -re and im
        Aub_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aub_coeffs[:, 0::2] = -re
        Aub_coeffs[:, 1::2] = im

        # Append zero for the epsilon variable
        Aub = np.hstack([Aub_coeffs, np.zeros((num_samples, 1))])
        # Append 0 for the sigma variable
        Aub = np.hstack([Aub, np.zeros((num_samples, 1))])
        # Append one for the 'y' variable (since the inequality involves y)
        Aub = np.hstack([Aub, np.ones((num_samples, 1))])
        # Right-hand side vectors
        bub = np.full(num_samples,right_hand_value , dtype=np.float64)

    elif constraint_type == 'dynamic':
        if dynamic_real_imag is None:
            raise ValueError("dynamic_real_imag must be provided for 'dynamic' constraint_type.")

        # Calculate the difference between dynamic and original values
        delta = dynamic_real_imag  - real_imag  # Shape: (num_samples, num_terms, 2)
        # Separate the real and imaginary parts of the differences
        delta_re = delta[:, :, 0]  # Real parts
        delta_im = delta[:, :, 1]  # Imaginary parts
        num_samples, num_terms = delta_re.shape

        # Prepare coefficients for the inequality constraints (Aub)
        # For each sample, interleave delta_re and -delta_im
        Aub_coeffs = np.empty((num_samples, 2 * num_terms), dtype=np.float64)
        Aub_coeffs[:, 0::2] = delta_re
        Aub_coeffs[:, 1::2] = -delta_im

        # Append zero for the epsilon variable
        Aub = np.hstack([Aub_coeffs, np.zeros((num_samples, 1))])
        # Append -1 for the sigma variable
        Aub = np.hstack([Aub, -1 * np.ones((num_samples, 1))])
        # Append zero for the 'y' variable
        Aub = np.hstack([Aub, np.zeros((num_samples, 1))])
        # Right-hand side vectors
        bub = np.full(num_samples, right_hand_value, dtype=np.float64)
    else:
        raise ValueError("Invalid constraint_type. Must be 'initial', 'unsafe', or 'dynamic'.")

    return Aub, bub

def generate_all_constraints_fin(
    initial_real_imag,
    unsafe_real_imag,
    dynamic_real_imag,
    dynamic_initial_real_imag,
    num_passi: int,
    eps_sigma_constraint: bool,
    right_hand_value: float = 0
):

    vmatr = []
    hmatr = []
    # Generate initial constraints
    if initial_real_imag is not None:
        Aub_initial, bub_initial = generate_constraints_fin(
            real_imag=initial_real_imag,
            constraint_type='initial',
            right_hand_value=0
        )
        vmatr.append(Aub_initial)
        hmatr.append(bub_initial)

    # Generate unsafe constraints
    if unsafe_real_imag is not None:
        Aub_unsafe, bub_unsafe = generate_constraints_fin(
            real_imag=unsafe_real_imag,
            constraint_type='unsafe',
            right_hand_value=right_hand_value
        )
        vmatr.append(Aub_unsafe)
        hmatr.append(bub_unsafe)

    # Generate dynamic constraints
    if dynamic_real_imag is not None:
        Aub_dynamic, bub_dynamic = generate_constraints_fin(
            real_imag=dynamic_initial_real_imag,
            dynamic_real_imag=dynamic_real_imag,
            constraint_type='dynamic',
            right_hand_value=0
        )
        vmatr.append(Aub_dynamic)
        hmatr.append(bub_dynamic)

    # Generate epsilon + num_passi * sigma < delta constraint
    if eps_sigma_constraint:
        # Extract real and imaginary parts
        re = initial_real_imag[:, :, 0]
        im = initial_real_imag[:, :, 1]
        num_samples, num_terms = re.shape
        Aub_coeffs = np.zeros((1, 2 * num_terms), dtype=np.float64)

        # Append 1 for the epsilon variable
        Aub_eps = np.hstack([Aub_coeffs, np.ones((1, 1))])

        # Append num_passi for the sigma variable
        Aub_eps = np.hstack([Aub_eps, num_passi * np.ones((1, 1))])
        # Append -1 for the delta variable
        Aub_eps = np.hstack([Aub_eps, -1 * np.ones((1, 1))])
        # Append small value for < 0
        bub = [-1]
        vmatr.append(Aub_eps)
        hmatr.append(bub)
        Aub_coeffs = np.zeros((1, 2 * num_terms), dtype=np.float64)
        # Append 1 for the epsilon variable
        Aub_eps = np.hstack([Aub_coeffs, np.ones((1, 1))])
        # Append 0 for the sigma variable
        Aub_eps = np.hstack([Aub_eps, 0 * np.ones((1, 1))])
        # Append -1 for the delta variable
        Aub_eps = np.hstack([Aub_eps, -1 * np.ones((1, 1))])
        # Append small value for < 0
        bub = [-0.000001]
        vmatr.append(Aub_eps)
        hmatr.append(bub)

    # Concatenate all inequality constraints
    Aub = np.vstack(vmatr)
    bub = np.hstack(hmatr)

    return Aub, bub


def solve_lp_fin(c, Aub, bub, bounds, l, terms, vars, opt_meth):
    result = linprog(
        c,
        A_ub=Aub,
        b_ub=bub,
        bounds=bounds,
        method=opt_meth,
    )

    if result.success:
        x_optimal = result.x
        Re_a_optimal = x_optimal[0:2*l:2]
        Im_a_optimal = x_optimal[1:2 * l:2]
        epsilon = x_optimal[-3]
        sigma = x_optimal[-2]
        y_optimal = x_optimal[-1]
        a_optimal = construct_complex_coefficients(Re_a_optimal, Im_a_optimal)
        barrier_certificate = generate_barrier_polynomial(a_optimal, terms, vars)
        return barrier_certificate, a_optimal, (epsilon, sigma, y_optimal)

    else:
        print("Optimization failed:", result.message)
        return None, None, None

'''-----------'''


def generate_constraints_inf(
        real_imag_list: List[np.ndarray],
        constraint_type: str,
        dynamic_real_imag_list: List[np.ndarray] = None,
        right_hand_value: float = 0,
        epsilon: float = 0,
        gamma: float = 0,
        unitaries: List[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate constraints for a linear programming (LP) problem.
    ...
    """

    l = len(unitaries)

    if constraint_type == 'initial':
        # Constraints at time t=0
        real_imag = real_imag_list[0]
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]

        num_samples, num_terms = re.shape
        num_vars_per_time = 2 * num_terms  # Real and imaginary parts
        # Prepare Aub
        Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
        Aub_coeffs[:, 0:num_vars_per_time] = np.hstack([re, -im])  # B_0(z)

        # Right-hand side
        Aub = np.vstack(Aub_coeffs)
        bub = np.full(num_samples, right_hand_value, dtype=np.float64)

    elif constraint_type == 'unsafe':
        # Constraints for all time steps
        Aub_list = []
        bub_list = []
        for t in range(len(unitaries)):
            real_imag = real_imag_list[t]
            re = real_imag[:, :, 0]
            im = real_imag[:, :, 1]
            num_samples, num_terms = re.shape
            num_vars_per_time = 2 * num_terms  # Real and imaginary parts

            Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
            start_idx = t * num_vars_per_time
            Aub_coeffs[:, start_idx:start_idx + num_vars_per_time] = np.hstack([-re, im])  # -B_t(z)

            # Append 1 for 'y' variable
            Aub_coeffs[:, -1] = 1.0

            Aub_list.append(Aub_coeffs)
            bub_list.append(np.zeros(num_samples, dtype=np.float64))

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)
    elif constraint_type == 'dynamic':
        # Constraints involving dynamics: B_t(f_t(z)) - B_t(z) <= epsilon
        Aub_list = []
        bub_list = []
        for t in range(len(unitaries)):
            real_imag_t = real_imag_list[t]
            dynamic_real_imag_t = dynamic_real_imag_list[t]

            delta = dynamic_real_imag_t - real_imag_t
            delta_re = delta[:, :, 0]
            delta_im = delta[:, :, 1]
            num_samples, num_terms = delta_re.shape
            num_vars_per_time = 2 * num_terms  # Real and imaginary parts

            Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
            start_idx = t * num_vars_per_time
            Aub_coeffs[:, start_idx:start_idx + num_vars_per_time] = np.hstack(
                [delta_re, -delta_im])  # B_t(f_t(z)) - B_t(z)

            Aub_list.append(Aub_coeffs)
            bub_list.append(np.full(num_samples, epsilon, dtype=np.float64))

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)

    elif constraint_type == 'time_evolution':
        # Constraints involving time evolution: |B_{t+1}(z) - B_t(z)| <= gamma
        Aub_list = []
        bub_list = []
        real_imag = real_imag_list[0]
        re = real_imag[:, :, 0]
        im = real_imag[:, :, 1]
        num_samples, num_terms = re.shape
        num_vars_per_time = 2 * num_terms

        for sign in [1, -1]:
            Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
            start_idx_t = 0
            start_idx_t1 = num_vars_per_time

            # Coefficients for B_{t+1}(z)
            Aub_coeffs[:, start_idx_t1:start_idx_t1 + num_vars_per_time] = sign * np.hstack([re, -im])
            # Subtract coefficients for B_t(z)
            Aub_coeffs[:, start_idx_t:start_idx_t + num_vars_per_time] -= sign * np.hstack([re, -im])
            # Right-hand side
            bub = np.full(num_samples, gamma, dtype=np.float64)
            Aub_list.append(Aub_coeffs)
            bub_list.append(bub)

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)


    elif constraint_type == 'k_inductive':
        # Constraints for k-induction: B_{t+k}(f^{(k)}(z)) - B_t(z) <= 0
        Aub_list = []
        bub_list = []
        real_imag_tk = real_imag_list[0]
        real_imag_t = dynamic_real_imag_list[0]

        delta = real_imag_tk - real_imag_t
        delta_re = delta[:, :, 0]
        delta_im = delta[:, :, 1]
        num_samples, num_terms = delta_re.shape
        num_vars_per_time = 2 * num_terms  # Real and imaginary parts

        Aub_coeffs = np.zeros((num_samples, l * num_vars_per_time + 1))
        start_idx = 0
        Aub_coeffs[:, start_idx:start_idx + num_vars_per_time] = np.hstack([delta_re, -delta_im])  # B_{t+k}(f^{(k)}(z))

        Aub_list.append(Aub_coeffs)
        bub_list.append(np.full(num_samples, right_hand_value, dtype=np.float64))

        Aub = np.vstack(Aub_list)
        bub = np.hstack(bub_list)

    else:
        raise ValueError("Invalid constraint_type.")

    return Aub, bub


def generate_all_constraints_inf(
        initial_real_imag_list,
        unsafe_real_imag_list,
        dynamic_real_imag_list,
        dynamic_initial_real_imag_list,
        time_real_imag_list,
        k_real_imag_list,
        k_initial_real_imag_list,
        K: int,
        epsilon: float,
        gamma: float,
        right_hand_value: float = 0,
        unitaries: List[np.ndarray] = None
):
    """
    Generate all constraints for the LP problem.
    ...
    """
    # Generate initial constraints
    Aub_initial, bub_initial = generate_constraints_inf(
        real_imag_list=initial_real_imag_list,
        constraint_type='initial',
        unitaries=unitaries,
        right_hand_value=right_hand_value
    )

    # Generate unsafe constraints
    Aub_unsafe, bub_unsafe = generate_constraints_inf(
        real_imag_list=unsafe_real_imag_list,
        constraint_type='unsafe',
        right_hand_value=right_hand_value,
        unitaries=unitaries
    )

    # Generate dynamic constraints
    Aub_dynamic, bub_dynamic = generate_constraints_inf(
        real_imag_list=dynamic_initial_real_imag_list,
        dynamic_real_imag_list=dynamic_real_imag_list,
        constraint_type='dynamic',
        epsilon=epsilon,
        unitaries=unitaries
    )

    # Generate k-inductive constraints
    Aub_k_inductive, bub_k_inductive = generate_constraints_inf(
        real_imag_list=k_real_imag_list,
        dynamic_real_imag_list=k_initial_real_imag_list,
        constraint_type='k_inductive',
        unitaries=unitaries,
        right_hand_value=right_hand_value
    )

    # Concatenate all inequality constraints
    Aub = np.vstack([Aub_initial, Aub_unsafe, Aub_k_inductive])
    bub = np.hstack([bub_initial, bub_unsafe, bub_k_inductive])
    if K > 1:
        Aub = np.vstack([Aub, Aub_dynamic])
        bub = np.hstack([bub, bub_dynamic])

    if len(unitaries) > 1:
        # Generate time evolution constraints
        Aub_time_evolution, bub_time_evolution = generate_constraints_inf(
            real_imag_list=time_real_imag_list,
            constraint_type='time_evolution',
            gamma=gamma,
            unitaries=unitaries
        )
        Aub = np.vstack([Aub, Aub_time_evolution])
        bub = np.hstack([bub, bub_time_evolution])

    return Aub, bub




def solve_lp_inf(c, Aub, bub, bounds, l, num_time_steps, terms, vars, log_level=logging.INFO):
    logger.info("Solving linear optimization problem...")
    result = linprog(
        c,
        A_ub=Aub,
        b_ub=bub,
        bounds=bounds,
        method='highs'
    )
    logger.info("Linear program terminated.")
    logger.setLevel(log_level)
    if result.success:
        logger.info("Optimization completed:" + result.message)
        #x_optimal = round_values(result.x)
        x_optimal = result.x
        y_optimal = x_optimal[-1]
        logger.debug(result.x)
        logger.info("Optimal value: " + str(y_optimal))

        # Extract coefficients for each time step
        a_optimal = []
        for t in range(num_time_steps):
            start_idx = t * 2 * l
            end_idx = start_idx + l
            Re_a_optimal = x_optimal[start_idx:end_idx]
            Im_a_optimal = x_optimal[end_idx:end_idx+l]
            a_t = construct_complex_coefficients(Re_a_optimal, Im_a_optimal)
            a_optimal.append(a_t)
            logger.info("Extracting the real part of coefficients a form result...")
            logger.info(str(Re_a_optimal))
            logger.info("Extracting the immaginary part of coefficients a form result...")
            logger.info(str(Im_a_optimal))

        # Generate barrier polynomials for each time step
        barrier_certificates = []
        for t in range(num_time_steps):
            barrier_certificate = generate_barrier_polynomial(a_optimal[t], terms, vars)
            barrier_certificates.append(barrier_certificate)

        return barrier_certificates, a_optimal, y_optimal

    else:
        logger.warning("Optimization falied: " + result.message)
        #print("Optimization failed:", result.message)
        return None, None, None





















