# debugpy_auto.py — auto-connect to a VSCode debug adapter on startup.
#
# How it works:
#   This module is imported by debugpy_auto.pth, which lives in site-packages.
#   Python's site module executes `import <name>` lines in .pth files during
#   interpreter startup — before any script runs. This means every Python
#   process in this venv will execute this file on startup.
#
# Activation:
#   Set DEBUGPY_AUTO=1 in the environment. Without it, this module is a no-op.
#     DEBUGPY_AUTO=1 WORLD_SIZE=2 python train.py --lr 0.001
#
# The _DEBUGPY_AUTO_IS_CONNECTED guard:
#   torch.multiprocessing.spawn uses the "spawn" start method, which launches
#   each child as a fresh Python interpreter. Each child goes through site
#   initialization and imports this module again. Without the guard, every
#   child would call debugpy.connect() independently — but debugpy's adapter
#   only accepts one root connection (debugpy#1501). The parent's connection
#   is the root; children are auto-discovered via the subProcess: true setting
#   in launch.json. Setting _DEBUGPY_AUTO_IS_CONNECTED=1 after the parent connects
#   prevents children from trying to connect a second time.
#
# Configuration:
#   DEBUGPY_AUTO         — set to any truthy value to enable (e.g. DEBUGPY_AUTO=1)
#   DEBUGPY_AUTO_PORT    — adapter port (default 5678)
#
# Install:
#   With the target venv activated:
#   cp debugpy_auto/* "$(python -c 'import site; print(site.getsitepackages()[0])')"/
#   If the venv is recreated, re-run the copy.

import os

if os.environ.get("DEBUGPY_AUTO"):
    _port = int(os.environ.get("DEBUGPY_AUTO_PORT", 5678))

    if not os.environ.get("_DEBUGPY_AUTO_IS_CONNECTED"):
        import debugpy
        import socket, time

        print(f"[debugpy_auto] process (pid={os.getpid()}): waiting for VSCode debugger on localhost:{_port}")
        # Ctrl+C handling: this code runs inside `import site`, so any exception
        # that escapes (KeyboardInterrupt, and even SystemExit from sys.exit())
        # aborts interpreter startup with "Fatal Python error: init_import_site".
        # The builtin exit() isn't defined yet either. os._exit() is the only clean way out.
        # The try must wrap the whole wait: a KeyboardInterrupt raised inside an
        # `except OSError:` handler (i.e. during time.sleep) is not caught by a
        # sibling `except KeyboardInterrupt:` clause.
        try:
            while True:
                try:
                    sock = socket.create_connection(("localhost", _port), timeout=1)
                    sock.close()
                    break
                except OSError:
                    time.sleep(1)

            debugpy.connect(("localhost", _port))
            os.environ["_DEBUGPY_AUTO_IS_CONNECTED"] = "1"
            print(f"[debugpy_auto] Connected. Waiting for VSCode client to attach...")
            debugpy.wait_for_client()
            print(f"[debugpy_auto] Client attached. Resuming execution.")
        except KeyboardInterrupt:
            print(f"\n[debugpy_auto] Interrupted.", flush=True)
            os._exit(130)
    else:
        print(f"[debugpy_auto] process (pid={os.getpid()}): skipping connect(); DEBUGPY_AUTO is set but _DEBUGPY_AUTO_IS_CONNECTED guard is active")
