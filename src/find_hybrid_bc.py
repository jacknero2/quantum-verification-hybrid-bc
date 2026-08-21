import logging
from src.find_bc import find_bc
from src.hybrid import HybridSystem, BranchNode
from src.constants import HIGHSIPM

logger = logging.getLogger("hybridBC")


def find_hybrid_bc(system: HybridSystem,
                    n_samples: int,
                    deg: int,
                    log_level=logging.INFO,
                    opt_meth=HIGHSIPM,
                    **kwargs):
    """
    Certify a branched quantum-classical system (Definition 4.1): find a
    barrier certificate B_c for every reachable branch c, each restricted
    to that branch's own local circuit/Z0/Zu. The system is certified iff
    every reachable branch is.
    """
    logger.setLevel(log_level)
    results = {}

    def walk(path_prefix: list[str], label: str, node: BranchNode):
        path_key = ".".join(path_prefix + [label])
        node_deg = node.deg if node.deg is not None else deg

        logger.info(f"Certifying branch '{path_key}'...")
        certified, barrier, timings = find_bc(
            circuit=node.circuit,
            type_bc=node.type_bc,
            n_samples=n_samples,
            Z=system.Z,
            Z0=node.Z0,
            Zu=node.Zu,
            deg=node_deg,
            log_level=log_level,
            opt_meth=opt_meth,
            **node.bc_kwargs,
            **kwargs,
        )
        results[path_key] = (certified, barrier, timings)
        logger.info(f"Branch '{path_key}' {'CERTIFIED' if certified else 'FAILED certification'}.")

        for child_label, child in node.children.items():
            walk(path_prefix + [label], child_label, child)

    for label, node in system.branches.items():
        walk([], label, node)

    overall = all(v[0] for v in results.values())
    logger.info(f"Hybrid system certified: {overall}")
    return overall, results
