# Barrier Certificates for Hybrid Quantum-Classical Algorithms

> **Credit:** This repository is a modified version of
> [QuantumVerification/Quantum-Verification-QSW-2025](https://github.com/QuantumVerification/Quantum-Verification-QSW-2025),
> the original implementation by Siwei Hu, Victor Lopata, Sadegh Soudjani and Paolo Zuliani
> accompanying their paper *Verification of Quantum Circuits Through Barrier Certificates Using a
> Scenario Approach* (IEEE QSW 2025, pp. 151–161,
> [doi:10.1109/QSW67625.2025.00027](https://doi.org/10.1109/QSW67625.2025.00027)).
> All credit for the original scenario-based method and code belongs to those authors.
> This fork extends it to hybrid (branched) quantum-classical algorithms.

## Overview
The original repository implements a scenario-based methodology for the formal verification of
quantum circuits using barrier certificates: it proves a circuit never reaches undesired states,
even under uncertainties and over different time horizons, without enumerating the state space.

This fork extends that methodology to **hybrid quantum-classical algorithms**, in which quantum
evolution is interleaved with mid-circuit measurement and classically controlled branching. Such
an algorithm is modeled as a *branched quantum system*: a tree in which each measurement outcome
selects the dynamics that follow. The system is certified by finding a separate barrier
certificate for every reachable branch, each checked against that branch's own initial states,
unsafe states and dynamics.

## Features

### From the original implementation
- **Barrier Certificate Synthesis**: Generates formal safety certificates for quantum circuits.
- **Scenario-based Optimization**: Employs sampling techniques to efficiently verify circuit correctness.
- **Supports Finite and Infinite Horizons**: Enables verification for both bounded and unbounded
  time steps (the infinite-horizon mode uses k-inductive barrier certificates).
- **Handles Uncertainty**: Can verify circuits under uncertain initial states and system dynamics.
- **Integration with SMT Solvers**: Uses Z3 for formal verification of synthesized certificates.
- **Linear Programming Optimization**: Employs **`scipy.optimize`** for solving certificate constraints.

### Added in this fork
- **Hybrid (branched) systems**: `BranchNode` / `HybridSystem` (`src/hybrid.py`) describe a tree of
  measurement-dependent branches, and `find_hybrid_bc` (`src/find_hybrid_bc.py`) certifies the
  whole system by certifying every reachable branch.
- **Parallel branch certification**: branches are independent, so they are certified concurrently
  in a process pool (`--max-workers`, default: number of CPU cores).
- **Counterexample-guided sample augmentation (CEGIS)**: every counterexample Z3 finds is fed back
  into the LP as a permanent sample, so a rejected candidate is never proposed again (`--cegis`).
- **L1 sparsity preference**: an optional penalty that steers the LP toward sparse certificates,
  which are much cheaper for Z3 to verify (`--l1-weight`).
- **Vectorized sampling**: monomial evaluation over sampled states is vectorized with NumPy,
  roughly 160× faster than before with identical output.
- **Bug fixes**: `find_bc` now returns `(certified, barrier, timings)`, and the finite-horizon mode
  no longer crashes after solving the LP.

CEGIS, L1 sparsity and the `persistent_samples` option are opt-in; with their defaults, the
original examples behave exactly as before.

## Installing Dependencies
Use Python 3.11 or 3.12 (the pinned NumPy version does not build on 3.14). To install the
dependencies listed in **`requirements.txt`**, run:
```
pip install -r requirements.txt
```

## Usage

### Original examples
The examples directory contains script files for running specific examples. To execute an
example, use the following command: **`zsh ./examples/<example>.sh`**.

### Hybrid examples
Hybrid examples are run with:
```
python3 -m examples.hybrid_examples --example <name> [options]
```

| Example | Description |
|---|---|
| `measure_correct` | Two qubits are measured and one of four correction gates returns the state to `\|00⟩` (4 branches). |
| `bitflip_code` | The three-qubit bit-flip code: a syndrome selects a Pauli-X correction that recovers an arbitrary logical state `α\|000⟩ + β\|111⟩` (4 branches). |

| Option | Default | Meaning |
|---|---|---|
| `-samples` | 20000 | Number of sampled states per region. |
| `--barrier-degree` | 2 | Maximum degree of the barrier polynomial. |
| `-p` | 0.1 | Acceptable leakage probability (defines the unsafe regions). |
| `--cegis` | off | Feed Z3 counterexamples back as training samples. |
| `--l1-weight` | 0 | L1 sparsity penalty weight (0 disables it). |
| `--max-workers` | CPU cores | Maximum number of branches certified in parallel. |

For example:
```
python3 -m examples.hybrid_examples --example measure_correct -samples 2000
python3 -m examples.hybrid_examples --example bitflip_code -samples 2000 --cegis --l1-weight 0.01
```

The bit-flip code is the harder case: in our runs, its `no_error` branch was only certified with
`--cegis` and `--l1-weight` enabled together (about 13 minutes with 2000 samples; the other three
branches finish in about 75 seconds). `measure_correct` certifies all four branches in under 20
seconds without them.

A general n-qubit version of the bit-flip code is available from Python as
`bitflip_code_example_n(n)` in `examples/hybrid_examples.py`; it reproduces `bitflip_code` exactly
for n = 3 but has not been tested for other values of n.
