# Verification of Quantum Circuits through Barrier Certificates using a Scenario Approach

## Overview
This repository contains the implementation of a scenario-based methodology for the formal verification of quantum circuits using barrier certificates. The approach enables proving the correctness of quantum circuits by ensuring they never reach undesired states, even under uncertainties and over different time horizons.

## Features
- **Barrier Certificate Synthesis**: Generates formal safety certificates for quantum circuits.
- **Scenario-based Optimization**: Employs sampling techniques to efficiently verify circuit correctness.
- **Supports Finite and Infinite Horizons**: Enables verification for both bounded and unbounded time steps.
- **Handles Uncertainty**: Can verify circuits under uncertain initial states and system dynamics.
- **Integration with SMT Solvers**: Uses Z3 for formal verification of synthesized certificates.
- **Linear Programming Optimization**: Employs **`scipy.optimize`** for solving certificate constraints.

## Installing Dependencies
To install dependencies listed in **`requirements.txt`**, run the following command in your terminal or command prompt:
```
pip install -r requirements.txt
```

## Usage
The examples directory contains script files for running specific examples. To execute an example, use the following command: **`zsh ./examples/<example>.sh`**.

