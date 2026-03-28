# Debugging Distributed PyTorch (torchrun) with VSCode on a Compute Cluster

## Why Do I Need to Use debugpy Explicitly?

VSCode's built-in Python debugger already uses debugpy under the hood — the Python extension bundles it and injects it automatically every time you press F5. In normal single-process development, you never interact with debugpy directly because VSCode manages the entire lifecycle: it spawns `python your_script.py`, attaches the debugger, and you're off.

Two things break this in a distributed training setup:

1. **torchrun is the entrypoint, not python.** VSCode's "launch" mode runs `python your_script.py`. Distributed training needs `torchrun --nproc_per_node=8 your_script.py`, which spawns a process tree that VSCode doesn't know how to manage. It can't launch torchrun and automatically attach to the 8 child workers. This is a known gap — there's an open feature request on the debugpy repo for console-script support.

2. **Execution happens on a different node.** Even if VSCode could launch torchrun, it would run on the VSCode node (the login/head node), not on the compute node with the GPUs. VSCode can't manage the process lifecycle of something running on a remote machine via a job scheduler.

So debugpy isn't an *alternative* to VSCode's debugger — it **is** VSCode's debugger. The difference is whether VSCode manages the process for you (launch mode, works for single-process local scripts) or you manage it yourself and have debugpy phone home to VSCode (attach mode, required here). The explicit `import debugpy; debugpy.connect(...)` calls in your training code are the mechanism that bridges the gap.

If your training ran on the same machine as VSCode and used plain `python` instead of `torchrun`, you'd never need to touch debugpy directly — F5 would just work.

---

## Architecture Overview

Your setup has two nodes:

- **VSCode node**: where your editor runs (e.g., a login/head node)
- **Compute node**: where `torchrun` spawns N GPU workers

They share an NFS filesystem, which simplifies path mapping considerably.

The core idea: **VSCode acts as a debug server (listener), and each worker process connects back to it.** This is called the "reverse connection" pattern and is the most practical approach for cluster environments because:

1. You don't need to know the compute node's hostname/IP in advance for your launch.json.
2. You don't need N different ports on the compute node — all workers connect to the *same* port on the VSCode node.
3. It works naturally with SSH tunneling.

```
┌─────────────────────┐         SSH tunnel or direct TCP        ┌──────────────────────────┐
│    VSCode Node       │◄────────────────────────────────────────│    Compute Node           │
│                      │         port 5678                       │                           │
│  launch.json:        │                                         │  torchrun --nproc=8       │
│    "listen": 5678    │◄─── debugpy.connect(vscode_host, 5678)──│    worker rank 0          │
│                      │◄─── debugpy.connect(vscode_host, 5678)──│    worker rank 1          │
│  (one debug session  │◄─── ...                                 │    ...                    │
│   per connection)    │◄─── debugpy.connect(vscode_host, 5678)──│    worker rank 7          │
└─────────────────────┘                                          └──────────────────────────┘
```

---

## Step 1: Install debugpy on the Compute Environment

On the shared NFS (so it's visible from compute nodes):

```bash
pip install debugpy
```

Verify: `python -c "import debugpy; print(debugpy.__version__)"`

---

## Step 2: Instrument Your Training Script

Add a debug hook function that each worker calls early in its execution. The key design decisions:

- Use `debugpy.connect()` (worker → VSCode), NOT `debugpy.listen()` (which would require a unique port per worker and is the source of most "address already in use" errors in multiprocess setups).
- Guard with an environment variable so the debug code is a no-op in production.
- Only `wait_for_client` on the rank(s) you actually want to pause on.

```python
# debug_utils.py — importable helper

import os

def setup_debugpy():
    """Call this early in each worker's entry point, after dist.init_process_group()."""
    if os.environ.get("DEBUGPY_ENABLE") != "1":
        return

    import debugpy
    import torch.distributed as dist

    rank = dist.get_rank()

    # The host:port where VSCode is listening.
    # If using SSH tunnel, this is localhost on the compute node.
    debug_host = os.environ.get("DEBUGPY_HOST", "localhost")
    debug_port = int(os.environ.get("DEBUGPY_PORT", "5678"))

    debugpy.connect((debug_host, debug_port))

    # Choose which rank(s) to pause on.
    # Other ranks will connect (enabling you to switch to them in VSCode)
    # but won't block.
    wait_ranks = os.environ.get("DEBUGPY_WAIT_RANKS", "0")
    wait_ranks = set(int(r) for r in wait_ranks.split(","))

    if rank in wait_ranks:
        print(f"[Rank {rank}] Waiting for debugger to attach...")
        debugpy.wait_for_client()
        debugpy.breakpoint()  # will pause here once VSCode attaches
    else:
        print(f"[Rank {rank}] Connected to debugger (not waiting).")
```

In your training script:

```python
import torch.distributed as dist

def main():
    dist.init_process_group("nccl")

    # --- debug hook ---
    from debug_utils import setup_debugpy
    setup_debugpy()
    # ------------------

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    model = MyModel().to(device)
    ddp_model = DDP(model, device_ids=[device])
    # ... training loop ...
```

---

## Step 3: Configure VSCode (launch.json)

On the VSCode node, create `.vscode/launch.json` in your project root (which is on NFS):

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach: Listen for Workers",
            "type": "debugpy",
            "request": "attach",
            "listen": {
                "host": "0.0.0.0",
                "port": 5678
            },
            "pathMappings": [
                {
                    // Since both nodes share NFS, these are likely identical.
                    // Adjust if your NFS mount point differs between nodes.
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "${workspaceFolder}"
                }
            ],
            "justMyCode": true
        }
    ]
}
```

**Key points about `"listen"` mode:**

- VSCode opens port 5678 and waits for incoming `debugpy.connect()` calls from workers.
- Each worker that connects will appear as a separate debug session in the **Call Stack** panel.
- You can switch between them to inspect different ranks.

---

## Step 4: Network Connectivity (SSH Tunnel)

On most clusters, compute nodes can't directly reach the VSCode node's port 5678. Two options:

### Option A: SSH Tunnel (most common, works on restrictive clusters)

From the compute node (or in your SLURM job script), open a reverse tunnel:

```bash
# Run on the compute node. This makes "localhost:5678" on the compute node
# forward to port 5678 on the VSCode node.
ssh -N -R 5678:localhost:5678 your_user@vscode-node-hostname &
```

Then set `DEBUGPY_HOST=localhost` (the default) — workers connect to localhost:5678, which tunnels back to VSCode.

**If you launch from the VSCode node via SSH to the compute node**, do a *local* forward instead:

```bash
# Run from the VSCode node when SSH-ing into the compute node:
ssh -L 5678:localhost:5678 your_user@compute-node -t \
    "cd /path/to/project && DEBUGPY_ENABLE=1 torchrun --nproc_per_node=8 train.py"
```

Wait — this actually doesn't help because VSCode is listening, not connecting. The simpler pattern:

```bash
# If the compute node CAN reach the vscode node directly:
export DEBUGPY_HOST=vscode-node-hostname

# If it can't, use an SSH tunnel from *your VSCode node* to the compute node:
# On VSCode node, in a terminal:
ssh -R 5678:localhost:5678 compute-node
# Then on the compute node, DEBUGPY_HOST=localhost (default)
```

### Option B: Direct TCP (if nodes are on same network)

If your cluster allows direct TCP between nodes (common on internal networks):

```bash
export DEBUGPY_HOST=<vscode-node-ip-or-hostname>
export DEBUGPY_PORT=5678
```

No tunnel needed. Workers connect directly to VSCode's listening port.

---

## Step 5: Launch the Debug Session

### On the VSCode node:

1. Open your project folder in VSCode.
2. Go to **Run and Debug** (Ctrl+Shift+D).
3. Select **"Attach: Listen for Workers"** and press **F5**.
4. VSCode is now listening on port 5678, waiting for connections.

### On the compute node (or via your job scheduler):

```bash
export DEBUGPY_ENABLE=1
export DEBUGPY_HOST=localhost          # if using SSH tunnel
export DEBUGPY_PORT=5678
export DEBUGPY_WAIT_RANKS=0            # pause only rank 0; others run freely

# Increase NCCL timeout so other ranks don't die while you're stepping through rank 0
export NCCL_TIMEOUT=1800000            # 30 minutes, in milliseconds

torchrun --nproc_per_node=8 train.py
```

Or in a SLURM script:

```bash
#!/bin/bash
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64

# Set up reverse SSH tunnel back to the VSCode node
ssh -N -R 5678:localhost:5678 your_user@vscode-node &
TUNNEL_PID=$!
sleep 2  # let the tunnel establish

export DEBUGPY_ENABLE=1
export DEBUGPY_HOST=localhost
export DEBUGPY_PORT=5678
export DEBUGPY_WAIT_RANKS=0

# Large timeout so ranks don't die while you debug
export NCCL_TIMEOUT=1800000

torchrun --nproc_per_node=8 train.py

kill $TUNNEL_PID
```

---

## Step 6: Interacting with Multiple Processes in VSCode

Once workers connect, you'll see them in the **Call Stack** panel on the left side of the debug view. Each connected worker appears as a separate debug session (typically labeled by its thread/connection).

**To inspect a specific rank:**

1. Click on the session in the Call Stack panel.
2. The **Variables**, **Watch**, and **Debug Console** panels all switch to that process's context.
3. You can set breakpoints, step through code, and evaluate expressions — all scoped to that rank.

**Practical workflow for multi-rank debugging:**

- Set `DEBUGPY_WAIT_RANKS=0` to pause only rank 0 initially.
- Inspect rank 0's state, then **Continue** (F5) to let it proceed.
- If you need to pause a specific rank, add a conditional breakpoint in your code:
  ```python
  if dist.get_rank() == 3:
      debugpy.breakpoint()
  ```
- Or set `DEBUGPY_WAIT_RANKS=0,3,7` to pause multiple ranks at startup.

---

## Critical Gotcha: DDP Collective Hangs

This is the single most important thing to understand. DDP wraps your model and requires **all ranks to participate in collective operations** (allreduce during backward, broadcast during construction). If you pause one rank in the debugger while others continue:

- The running ranks will block at the next collective and eventually hit `NCCL_TIMEOUT`.
- Your debug session will die.

**Mitigations:**

1. **Set a very large NCCL timeout**: `NCCL_TIMEOUT=1800000` (30 min) or use `dist.init_process_group(timeout=datetime.timedelta(minutes=30))` in code.

2. **Pause ALL ranks at the same point** if you need to step through code that contains collectives. Set `DEBUGPY_WAIT_RANKS=0,1,2,3,4,5,6,7` and then **Continue all** simultaneously, or use breakpoints that all ranks hit.

3. **Debug with fewer ranks**: Use `--nproc_per_node=2` instead of 8. The bugs you're looking for usually reproduce with 2 ranks and you'll have a much easier time managing the sessions.

4. **Debug rank-local code only**: If you're debugging data loading, loss computation, or anything that doesn't involve collectives, you can safely pause one rank while others wait at the next collective (as long as your timeout is long enough).

5. **Use `gloo` backend for CPU debugging**: If you don't need GPUs for the bug you're investigating, `gloo` is more forgiving and can run on a laptop with `--nproc_per_node=2`.

---

## Alternative: `debugpy.listen()` per-rank (Forward Connection)

If the reverse connection pattern doesn't work for your network, you can have each worker listen on a unique port and connect VSCode to them individually:

```python
# In the worker:
import debugpy
rank = dist.get_rank()
debugpy.listen(("0.0.0.0", 5678 + rank))  # rank 0 → 5678, rank 1 → 5679, etc.
if rank == 0:
    debugpy.wait_for_client()
```

Then create multiple attach configurations in launch.json:

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach Rank 0",
            "type": "debugpy",
            "request": "attach",
            "connect": { "host": "compute-node", "port": 5678 },
            "pathMappings": [{"localRoot": "${workspaceFolder}", "remoteRoot": "${workspaceFolder}"}]
        },
        {
            "name": "Attach Rank 1",
            "type": "debugpy",
            "request": "attach",
            "connect": { "host": "compute-node", "port": 5679 },
            "pathMappings": [{"localRoot": "${workspaceFolder}", "remoteRoot": "${workspaceFolder}"}]
        }
        // ... one per rank
    ],
    "compounds": [
        {
            "name": "Attach All Ranks",
            "configurations": ["Attach Rank 0", "Attach Rank 1"]
        }
    ]
}
```

This requires more setup and SSH port forwarding per port, but gives you explicit control. The compound launch starts all debug sessions at once.

---

## Quick Reference

| Variable | Default | Purpose |
|---|---|---|
| `DEBUGPY_ENABLE` | unset | Set to `1` to activate debug hooks |
| `DEBUGPY_HOST` | `localhost` | Where VSCode is listening (use localhost with SSH tunnel) |
| `DEBUGPY_PORT` | `5678` | Port VSCode is listening on |
| `DEBUGPY_WAIT_RANKS` | `0` | Comma-separated ranks that should block until debugger attaches |
| `NCCL_TIMEOUT` | `1800000` | Milliseconds before NCCL operations timeout (increase for debugging) |

## Checklist Before Debugging

- [ ] `debugpy` installed in the compute node's Python environment
- [ ] VSCode debug session started in "listen" mode **before** launching torchrun
- [ ] SSH tunnel established (if needed) **before** launching torchrun
- [ ] `DEBUGPY_ENABLE=1` set in the compute node's environment
- [ ] `NCCL_TIMEOUT` set to a large value
- [ ] Project folder on NFS is the same path on both nodes (or pathMappings configured)
- [ ] `justMyCode` set to `true` unless you need to step into PyTorch internals