from src.log import *
import logging
from src.utils import *
from z3.z3 import *
import sympy as sym
from sympy.core import *

logger = logging.getLogger("smtcheck")

def _sympy_to_z3_rec(var_map, e):
    'recursive call for sympy_to_z3()'

    rv = None

    if not isinstance(e, Expr):
        raise RuntimeError("Expected sympy Expr: " + repr(e))

    if isinstance(e, sym.re) or isinstance(e, sym.im):
        rv = var_map.get(e)

        if rv == None:
            raise RuntimeError("No var was corresponds to symbol '" + str(e) + "'")

    elif isinstance(e, Number):
        rv = float(e)
    elif isinstance(e, Mul):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv *= _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, Add):
        rv = _sympy_to_z3_rec(var_map, e.args[0])

        for child in e.args[1:]:
            rv += _sympy_to_z3_rec(var_map, child)
    elif isinstance(e, Pow):
        term = _sympy_to_z3_rec(var_map, e.args[0])
        exponent = _sympy_to_z3_rec(var_map, e.args[1])

        if exponent == 0.5:
            # sqrt
            rv = Sqrt(term)
        else:
            rv = term ** exponent

    if rv == None:
        raise RuntimeError("Type '" + str(type(e)) + "' is not yet implemented for convertion to a z3 expresion. " + \
                           "Subexpression was '" + str(e) + "'.")

    return rv


def make_constraints(conditions, constraints, var_z3_dict):
    # Loop over each constraint in Z0_constraints
    for constr in constraints:
        variables = constr['variables']
        min_sum = constr['min']
        max_sum = constr['max']
        # Sum over the modulus squared of the variables
        sum_modulus_squared = sum(
            var_z3_dict[sym.re(v)] ** 2 + var_z3_dict[sym.im(v)] ** 2 for v in variables
        )
        # Add the inequality constraints
        if min_sum is not None:
            conditions.append(sum_modulus_squared >= min_sum)
        if max_sum is not None:
            conditions.append(sum_modulus_squared <= max_sum)
        for k, v in constr['imConstr'].items():
            conditions.append(var_z3_dict[sym.im(k)] >= v[0])
            conditions.append(var_z3_dict[sym.im(k)] <= v[1])
    return conditions


def check_barrier_fin(Z: list[sym.Symbol],
                  barrier: sym.Poly,
                  unitary,
                  var,
                  Z0,
                  Zu,
                  epsilon,
                  delta,
                  sigma,
                  tolerance=1e-5,
                  timeout=300):
    Z_RI = [sym.re(z) for z in Z] + [sym.im(z) for z in Z]
    FZ = np.dot(unitary, Z)

    mapping = {}
    for i in range(len(Z)):
        mapping[Z[i]] = FZ[i]
        mapping[conjugate(Z[i])] = conjugate(FZ[i])

    real_barrier = sym.poly(sym.re(sym.expand_complex(barrier.as_expr())))
    fp = barrier.as_expr().subs(zip(Z, np.dot(unitary, Z)), simultaneous=True) - barrier
    fp = sym.re((sym.expand_complex(fp.as_expr())))

    var_z3_dict = dict(zip(Z_RI, [Real(str(var)) for var in Z_RI]))
    k = [v for v in var_z3_dict.values()]
    norm = Sum([var ** 2 for var in k]) == 1

    z3_barrier = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())


    z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)


    initial_conditions = [norm, z3_barrier > epsilon + tolerance]

    for constr in Z0:
        variables = constr['variables']
        min_sum = constr['min']
        max_sum = constr['max']
        # Sum over the modulus squared of the variables
        sum_modulus_squared = sum(
            var_z3_dict[sym.re(v)] ** 2 + var_z3_dict[sym.im(v)] ** 2 for v in variables
        )
        # Add the inequality constraints
        if min_sum is not None:
            initial_conditions.append(sum_modulus_squared >= min_sum)
        if max_sum is not None:
            initial_conditions.append(sum_modulus_squared <= max_sum)
        for k, v in constr['imConstr'].items():
            initial_conditions.append(var_z3_dict[sym.im(k)] >= v[0])
            initial_conditions.append(var_z3_dict[sym.im(k)] <= v[1])

    i_sample = run_check(initial_conditions, 0)
    if i_sample:
        return True


    unsafe_conditions = [norm, z3_barrier < delta - tolerance]

    for constr in Zu:
        variables = constr['variables']
        min_sum = constr['min']
        max_sum = constr['max']
        # Sum over the modulus squared of the variables
        sum_modulus_squared = sum(
            var_z3_dict[sym.re(v)] ** 2 + var_z3_dict[sym.im(v)] ** 2 for v in variables
        )
        # Add the inequality constraints
        if min_sum is not None:
            unsafe_conditions.append(sum_modulus_squared >= min_sum)
        if max_sum is not None:
            unsafe_conditions.append(sum_modulus_squared <= max_sum)
        for k, v in constr['imConstr'].items():
            unsafe_conditions.append(var_z3_dict[sym.im(k)] >= v[0])
            unsafe_conditions.append(var_z3_dict[sym.im(k)] <= v[1])

    u_sample = run_check(unsafe_conditions, 1)
    if u_sample:
        return True

    dynamic_constraint = [norm, z3_fx_barrier > sigma + tolerance]


    if (z3_fx_barrier > sigma + tolerance) == False:
        return False

    delta_sample = run_check(dynamic_constraint, 2)

    if delta_sample:
        return True

    return False

'''-----------------------------'''




def check_barrier_inf(Z: list[sym.Symbol],
                  barriers: list[sym.Poly],
                  unitaries,
                  Z0,
                  Zu,
                  d,
                  epsilon,
                  gamma,
                  K,
                  timeout=300,
                  log_level=logging.INFO):
    logger.setLevel(log_level)
    Z_RI = [sym.re(z) for z in Z] + [sym.im(z) for z in Z]
    var_z3_dict = dict(zip(Z_RI, [Real(str(var)) for var in Z_RI]))
    k = [v for v in var_z3_dict.values()]
    norm = Sum([var ** 2 for var in k]) == 1

    real_barrier = sym.poly(sym.re(sym.expand_complex(barriers[0].as_expr())))
    z3_barrier = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())
    initial_conditions = [norm, z3_barrier > 0]
    initial_conditions = make_constraints(initial_conditions, Z0, var_z3_dict)
    logger.info("Checking initial constraint...")
    q = run_check(initial_conditions, timeout)
    if q:
        return q

    logger.info("checking unsafe constraints for barriers...")
    for barrier in barriers:
        real_barrier = sym.poly(sym.re(sym.expand_complex(barrier.as_expr())))
        z3_barrier = _sympy_to_z3_rec(var_z3_dict, real_barrier.as_expr())
        unsafe_conditions = [norm, z3_barrier < d - 0.0001]
        unsafe_conditions = make_constraints(unsafe_conditions, Zu, var_z3_dict)
        q = run_check(unsafe_conditions, timeout)
        if q:
            return q

    if K > 1:
        logger.info("checking for dynamic constraint...")
        for unitary, barrier in zip(unitaries, barriers):
            fp = sym.poly(barrier.as_expr().subs(zip(Z, np.dot(unitary, Z)), simultaneous=True) - barrier)
            fp = sym.re((sym.expand_complex(fp.as_expr())))
            z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)
            dynamic_constraint = [norm, z3_fx_barrier > epsilon]
            q = run_check(dynamic_constraint, timeout)
            if q:
                return q

    if len(barriers) > 1:
        logger.info("checking for time evolution constraint...")
        fp = sym.poly(barriers[0] - barriers[1])
        fp = sym.re((sym.expand_complex(fp.as_expr())))
        fp1 = sym.poly(barriers[1] - barriers[0])
        fp1 = sym.re((sym.expand_complex(fp1.as_expr())))
        z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)
        z3_fx_barrier1 = _sympy_to_z3_rec(var_z3_dict, fp1)
        dynamic_constraint = [norm, z3_fx_barrier > gamma, z3_fx_barrier1 > gamma]
        q = run_check(dynamic_constraint, timeout)
        if q:
            return q

    logger.info("checking for k_induction constraint...")
    while (len(unitaries) < K): unitaries = unitaries + unitaries
    FZ = Z
    for i in range(K): FZ = np.dot(unitaries[i], FZ)
    fp = sym.poly(barriers[0].as_expr().subs(zip(Z, FZ), simultaneous=True) - barriers[0])
    fp = sym.re((sym.expand_complex(fp.as_expr())))
    z3_fx_barrier = _sympy_to_z3_rec(var_z3_dict, fp)
    dynamic_constraint = [norm, z3_fx_barrier > 0]
    q = run_check(dynamic_constraint, timeout)
    if q:
        return q

    return q


def run_check(conditions, timeout):
    s = z3.SolverFor('QF_NRA')
    set_option(precision=50)
    # set_option(timeout=10000)
    [s.add(cond) for cond in conditions]
    if s.check() == sat:
        m = s.model()
        logger.warning("Counterexample found: " + str(m))
        return True
    else:
        logger.info("No counterexamples found.")
        return False