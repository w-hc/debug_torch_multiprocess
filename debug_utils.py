import os


def setup_debugpy():
    """Call early in each worker's entry point, after dist.init_process_group()."""
    if os.environ.get("DEBUGPY_ENABLE") != "1":
        return

    import debugpy
    import torch.distributed as dist

    rank = dist.get_rank()

    debug_host = os.environ.get("DEBUGPY_HOST", "localhost")
    debug_port = int(os.environ.get("DEBUGPY_PORT", "5678"))

    debugpy.connect((debug_host, debug_port))

    wait_ranks = os.environ.get("DEBUGPY_WAIT_RANKS", "all")
    if wait_ranks == "all":
        wait_ranks = {rank}
    else:
        wait_ranks = set(int(r) for r in wait_ranks.split(","))

    if rank in wait_ranks:
        print(f"[Rank {rank}] Waiting for debugger to attach...")
        debugpy.wait_for_client()
        debugpy.breakpoint()
    else:
        print(f"[Rank {rank}] Connected to debugger (not waiting).")
