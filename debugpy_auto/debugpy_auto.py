# debugpy_auto.py — auto-connect to a VSCode debug adapter on startup.
#
# How it works:
#   This module is imported by debugpy_auto.pth, which lives in site-packages.
#   Python's site module executes `import <name>` lines in .pth files during
#   interpreter startup — before any script runs. This means every Python
#   process in this venv will execute this file on startup.
#
# Activation:
#   Set DEBUG=1 in the environment. Without it, this module is a no-op.
#     DEBUG=1 WORLD_SIZE=2 python train.py --lr 0.001
#
# The _DEBUGPY_CONNECTED guard:
#   torch.multiprocessing.spawn uses the "spawn" start method, which launches
#   each child as a fresh Python interpreter. Each child goes through site
#   initialization and imports this module again. Without the guard, every
#   child would call debugpy.connect() independently — but debugpy's adapter
#   only accepts one root connection (debugpy#1501). The parent's connection
#   is the root; children are auto-discovered via the subProcess: true setting
#   in launch.json. Setting _DEBUGPY_CONNECTED=1 after the parent connects
#   prevents children from trying to connect a second time.
#
# Configuration:
#   DEBUG         — set to any truthy value to enable (e.g. DEBUG=1)
#   DEBUG_PORT    — adapter port (default 5678)
#
# Install:
#   cp debugpy_auto/debugpy_auto.py debugpy_auto/debugpy_auto.pth \
#     .venv/lib/python3.12/site-packages/
#
#   If the venv is recreated, re-run the copy.

import os

if os.environ.get("DEBUG"):
    _port = int(os.environ.get("DEBUG_PORT", 5678))

    if not os.environ.get("_DEBUGPY_CONNECTED"):
        import debugpy

        os.environ["_DEBUGPY_CONNECTED"] = "1"
        print(f"[debugpy_auto] process (pid={os.getpid()}): connecting to localhost:{_port}...")
        debugpy.connect(("localhost", _port))
        print(f"[debugpy_auto] Connected. Waiting for VSCode client to attach...")
        debugpy.wait_for_client()
        print(f"[debugpy_auto] Client attached. Resuming execution.")
    else:
        print(f"[debugpy_auto] process (pid={os.getpid()}): skipping connect(); DEBUG is set but _DEBUGPY_CONNECTED guard is active")
