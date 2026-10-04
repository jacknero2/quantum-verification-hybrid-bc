from src.utils import generate_symbols
from src.gates import Igate, Xgate
from src.hybrid import HybridSystem, BranchNode
from src.constants import FINITE_HORIZON
from src.find_hybrid_bc import find_hybrid_bc
from src.log import set_logger
import numpy as np
import datetime


def measure_correct_example(p=0.1):
    """
    Paper's Example 1 (Section 5.1): a 2-qubit state is measured, and the
    outcome selects one of four correction gates designed to return the
    state to |00>. Each branch is certified independently against its own
    post-measurement initial state and its own leakage-based unsafe region.
    """
    Z = generate_symbols(2)

    def branch(idx, gate, zu_vars):
        return BranchNode(
            circuit=[gate],
            type_bc=FINITE_HORIZON,
            Z0=[{'variables': [Z[idx]], 'min': 1.0, 'max': 1.0, 'imConstr': {}}],
            Zu=[{'variables': zu_vars, 'min': p, 'max': 1.0, 'imConstr': {}}],
            bc_kwargs={'steps': 1},
        )

    branches = {
        "00": branch(0, np.kron(Igate, Igate), [Z[1], Z[2], Z[3]]),
        "01": branch(1, np.kron(Igate, Xgate), [Z[2], Z[3]]),
        "10": branch(2, np.kron(Xgate, Igate), [Z[1], Z[3]]),
        "11": branch(3, np.kron(Xgate, Xgate), [Z[1], Z[2]]),
    }
    return HybridSystem(Z=Z, branches=branches, log_file="measure_correct_2qubit")


def bitflip_code_example(p=0.1):
    """
    3-qubit bit-flip code. A logical state alpha|000> + beta|111> may
    suffer a single-qubit bit-flip error (at most one of the three
    physical qubits flips). The syndrome (which qubit, if any, flipped)
    selects a Pauli-X correction on that qubit.

    Unlike measure_correct_example, each branch's initial region is the
    *2-dimensional corrupted code subspace* (both amplitudes free, not a
    single basis state), so certifying a branch verifies recovery of an
    arbitrary logical superposition, not just one fixed input.

    Basis index = 4*q0 + 2*q1 + q2 (q0 highest-order tensor factor, matching
    the np.kron convention used throughout gates.py/examples.py). The
    logical codewords are |000>=index 0 and |111>=index 7. A bit-flip on
    qubit i toggles bit i of the index, pairing each codeword index with
    its corrupted counterpart:
      qubit0 flip: 0<->4, 7<->3   (corrupted subspace = {3,4})
      qubit1 flip: 0<->2, 7<->5   (corrupted subspace = {2,5})
      qubit2 flip: 0<->1, 7<->6   (corrupted subspace = {1,6})

    Each correction gate (X on the flipped qubit, I elsewhere) is a pure
    index permutation with four disjoint 2-cycles. The four indices *not*
    in {codeword pair} u {corrupted pair} always form the other two
    2-cycles as a closed set, so their total probability mass is exactly
    invariant under the correction -- giving barrier condition (iii) for
    free (an algebraic identity, not just a sampled approximation), same
    reasoning as the paper's Example 1.
    """
    Z = generate_symbols(3)
    I3 = lambda: np.kron(Igate, np.kron(Igate, Igate))
    Gq = lambda pos: np.kron(Xgate if pos == 0 else Igate,
                              np.kron(Xgate if pos == 1 else Igate,
                                      Xgate if pos == 2 else Igate))

    def branch(x0_pair, gate, zu_indices):
        return BranchNode(
            circuit=[gate],
            type_bc=FINITE_HORIZON,
            Z0=[{'variables': [Z[i] for i in x0_pair], 'min': 1.0, 'max': 1.0, 'imConstr': {}}],
            Zu=[{'variables': [Z[i] for i in zu_indices], 'min': p, 'max': 1.0, 'imConstr': {}}],
            bc_kwargs={'steps': 1},
        )

    branches = {
        "no_error": branch((0, 7), I3(), (1, 2, 3, 4, 5, 6)),
        "q0_flip":  branch((3, 4), Gq(0), (1, 2, 5, 6)),
        "q1_flip":  branch((2, 5), Gq(1), (1, 3, 4, 6)),
        "q2_flip":  branch((1, 6), Gq(2), (2, 3, 4, 5)),
    }
    return HybridSystem(Z=Z, branches=branches, log_file="bitflip_code_3qubit")


def bitflip_code_example_n(n, p=0.1):
    """
    General n-qubit bit-flip code (n>=2): logical state alpha|0...0> +
    beta|1...1> may suffer a single-qubit bit-flip error on any one of the
    n physical qubits. The syndrome selects a Pauli-X correction on the
    affected qubit.

    This generalizes bitflip_code_example (which hand-picks n=3's index
    pairs explicitly) to arbitrary n using the same invariant-pair
    derivation, computed programmatically: basis index =
    sum_i q_i * 2^(n-1-i) (q_0 highest-order tensor factor), codewords are
    index 0 (|0...0>) and index N-1 (|1...1>) with N=2^n. A bit-flip on
    qubit i toggles bit value 2^(n-1-i) of the index, so its correction
    gate (X on qubit i, I elsewhere) is a permutation whose relevant
    2-cycles pair the codeword pair with the corrupted pair
    {bit_i, N-1-bit_i}; every other index forms the remaining, algebraically
    invariant complement used as Zu, exactly as derived by hand for n=3.
    """
    Z = generate_symbols(n)
    N = 2 ** n

    def gate_for_qubit(i):
        # i == -1 -> identity on every qubit (used for the no_error branch)
        factors = [Xgate if j == i else Igate for j in range(n)]
        gate = factors[0]
        for factor in factors[1:]:
            gate = np.kron(gate, factor)
        return gate

    def branch(x0_indices, gate, zu_indices):
        return BranchNode(
            circuit=[gate],
            type_bc=FINITE_HORIZON,
            Z0=[{'variables': [Z[i] for i in x0_indices], 'min': 1.0, 'max': 1.0, 'imConstr': {}}],
            Zu=[{'variables': [Z[i] for i in zu_indices], 'min': p, 'max': 1.0, 'imConstr': {}}],
            bc_kwargs={'steps': 1},
        )

    branches = {
        "no_error": branch([0, N - 1], gate_for_qubit(-1), list(range(1, N - 1))),
    }
    for i in range(n):
        bit_i = 2 ** (n - 1 - i)
        x0 = [bit_i, N - 1 - bit_i]
        zu = [j for j in range(N) if j not in (0, N - 1, bit_i, N - 1 - bit_i)]
        branches[f"q{i}_flip"] = branch(x0, gate_for_qubit(i), zu)

    return HybridSystem(Z=Z, branches=branches, log_file=f"bitflip_code_{n}qubit")


EXAMPLES = {
    "measure_correct": measure_correct_example,
    "bitflip_code": bitflip_code_example,
}


def run_hybrid_example(example, n_samples, poly_degree=2, p=0.1, cegis=False, l1_weight=0.0, max_workers=None):
    system = EXAMPLES[example](p)
    logger = set_logger(system.log_file + ".log")
    logger.info(str(datetime.datetime.now()))
    logger.info(f"Storing logs in {logger.handlers[-1].baseFilename}")
    logger.info(f"Running {system.log_file} hybrid example")
    return find_hybrid_bc(system, n_samples=n_samples, deg=poly_degree, max_workers=max_workers,
                           cegis=cegis, l1_weight=l1_weight)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a hybrid (branched) barrier certificate example.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--example", "-ex", type=str, default="measure_correct",
                        choices=list(EXAMPLES), help="Hybrid example to run.")
    parser.add_argument("-samples", type=int, default=20000, help="Number of samples.")
    parser.add_argument("--barrier-degree", type=int, default=2, help="Maximum degree of generated barrier.")
    parser.add_argument("-p", type=float, default=0.1, help="Acceptable leakage probability.")
    parser.add_argument("--cegis", action="store_true",
                        help="Feed each Z3 counterexample back as a permanent training sample.")
    parser.add_argument("--l1-weight", type=float, default=0.0,
                        help="L1 sparsity penalty weight on the barrier's coefficients (0 = off).")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Max branches to certify in parallel (default: number of CPU cores).")
    args = parser.parse_args()

    overall, results = run_hybrid_example(args.example, args.samples, args.barrier_degree, args.p,
                                           cegis=args.cegis, l1_weight=args.l1_weight, max_workers=args.max_workers)
    print("Overall certified:", overall)
    for branch_path, (certified, barrier, timings) in results.items():
        print(branch_path, "certified:", certified)
