from dataclasses import dataclass, field
from src.typings import Circuit
import sympy as sym


@dataclass
class BranchNode:
    """
    One node of a branched quantum-classical system (Definition 3.1).

    A node holds the local dynamics for one reachable classical register
    value c, together with the branch-local initial/unsafe state
    constraints (X0,c / Xu) that its own barrier certificate B_c must
    satisfy (Definition 4.1). Nested measurements are represented by
    `children`, keyed by the next classical outcome label.
    """
    circuit: Circuit
    type_bc: str
    Z0: list[dict]
    Zu: list[dict]
    bc_kwargs: dict = field(default_factory=dict)
    deg: int | None = None
    children: dict[str, "BranchNode"] = field(default_factory=dict)


@dataclass
class HybridSystem:
    """
    A branched quantum-classical system (Definition 3.1), restricted to
    the reachable classical register values needed to state Definition
    4.1: all branches share one Hilbert space `Z`, and `branches` is the
    set of reachable top-level classical outcomes.
    """
    Z: list[sym.Symbol]
    branches: dict[str, BranchNode]
    log_file: str = "hybrid"
