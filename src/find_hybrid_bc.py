import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.find_bc import find_bc
from src.hybrid import HybridSystem, BranchNode
from src.constants import HIGHSIPM

logger = logging.getLogger("hybridBC")


def _flatten_branches(system: HybridSystem):
    """
    Collect every (dotted path key, BranchNode) pair in the branch tree.
    Every node's certification is a fully independent problem -- Definition
    4.1 requires a separate B_c for each reachable branch, and nothing about
    a parent branch's own outcome affects a child's local circuit/Z0/Zu -- so
    the whole tree (including nested children, not just top-level branches)
    can be flattened into one flat list of independent jobs and processed in
    parallel, rather than certified one at a time in tree order.
    """
    flat = []

    def walk(path_prefix, label, node: BranchNode):
        path_key = ".".join(path_prefix + [label])
        flat.append((path_key, node))
        for child_label, child in node.children.items():
            walk(path_prefix + [label], child_label, child)

    for label, node in system.branches.items():
        walk([], label, node)
    return flat


def _certify_branch_worker(job):
    path_key, node, Z, n_samples, deg, log_level, opt_meth, extra_kwargs = job

    # Each branch runs in its own fresh process, which has no logging
    # handlers configured by default -- without this, every INFO-level
    # message find_bc logs (candidates found, CEGIS harvests, etc.) would
    # be silently dropped rather than just missing a nice format. Tag each
    # line with the branch's path so parallel branches' interleaved output
    # stays attributable.
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s - branch={path_key} - %(name)-13s - %(levelname)-7s - %(message)s"))
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)

    node_deg = node.deg if node.deg is not None else deg
    certified, barrier, timings = find_bc(
        circuit=node.circuit,
        type_bc=node.type_bc,
        n_samples=n_samples,
        Z=Z,
        Z0=node.Z0,
        Zu=node.Zu,
        deg=node_deg,
        log_level=log_level,
        opt_meth=opt_meth,
        **node.bc_kwargs,
        **extra_kwargs,
    )
    return path_key, certified, barrier, timings


def find_hybrid_bc(system: HybridSystem,
                    n_samples: int,
                    deg: int,
                    log_level=logging.INFO,
                    opt_meth=HIGHSIPM,
                    max_workers=None,
                    **kwargs):
    """
    Certify a branched quantum-classical system (Definition 4.1): find a
    barrier certificate B_c for every reachable branch c, each restricted
    to that branch's own local circuit/Z0/Zu. The system is certified iff
    every reachable branch is.

    Every branch's certification is independent, so branches are run in
    parallel via a process pool rather than one at a time. `max_workers`
    caps how many run concurrently (default: number of available CPU
    cores). Submitting every branch to the pool up front means it handles
    the "N at a time, next one starts as soon as a worker frees up" queueing
    on its own -- no manual batching needed, and a branch that finishes
    quickly doesn't have to wait for a slower sibling in the same "batch"
    before the next one starts.
    """
    logger.setLevel(log_level)
    flat_branches = _flatten_branches(system)

    if max_workers is None:
        max_workers = max(1, os.cpu_count() or 4)
    max_workers = max(1, min(max_workers, len(flat_branches)))

    logger.info(f"Certifying {len(flat_branches)} branch(es), up to {max_workers} in parallel...")

    jobs = [
        (path_key, node, system.Z, n_samples, deg, log_level, opt_meth, kwargs)
        for path_key, node in flat_branches
    ]

    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_certify_branch_worker, job): job[0] for job in jobs}
        for future in as_completed(futures):
            path_key = futures[future]
            try:
                _, certified, barrier, timings = future.result()
            except Exception as e:
                logger.error(f"Branch '{path_key}' raised an exception: {e!r}")
                certified, barrier, timings = False, None, {}
            results[path_key] = (certified, barrier, timings)
            logger.info(f"Branch '{path_key}' {'CERTIFIED' if certified else 'FAILED certification'}.")

    overall = all(v[0] for v in results.values())
    logger.info(f"Hybrid system certified: {overall}")
    return overall, results
