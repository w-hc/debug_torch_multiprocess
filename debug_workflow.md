# Debugging Distributed PyTorch (DDP) with VSCode

## Three Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **Remote Tunnels + `mp.spawn()`** | `code tunnel` on compute node; VSCode connects via Microsoft relay; F5 launches `mp.spawn()` with `subProcess: true` | Full IDE: visual breakpoints, variables, Call Stack per rank. Just press F5. Cmd+Shift+C continues all ranks. | Must replace torchrun with `mp.spawn()` (no elastic launch). |
| **Per-rank `debugpy.listen()`** | Each rank listens on its own debug port; SSH tunnels forward ports; VSCode attaches to each | Works with torchrun. Works multi-node. | One attach config + SSH tunnel per rank. Attach mode — can't auto-discover processes. |
| **`torch.distributed.breakpoint()`** | One line in source; paused rank drops into pdb; others wait at built-in barrier | Zero setup. Built-in rank sync — no NCCL timeout risk. | Terminal pdb only (no IDE). Must hardcode breakpoint calls. One rank at a time. |

**Recommendation:** Use Remote Tunnels + `mp.spawn()` for interactive debugging. Fall back to `torch.distributed.breakpoint()` for quick one-off inspections.

---

## Why Can't I Just Press F5?

Two things break normal VSCode debugging in distributed training:

1. **torchrun is the entrypoint, not python.** `torchrun --nproc_per_node=8 train.py` spawns a process tree that VSCode can't manage. There's an open feature request for console-script support ([debugpy#1311](https://github.com/microsoft/debugpy/issues/1311)).

2. **Execution happens on a different node.** Even if VSCode could launch torchrun, it would run locally, not on the compute node with the GPUs.

**The fix:** replace `torchrun` with `torch.multiprocessing.spawn()` for debugging, and get VSCode onto the compute node via Remote Tunnels. Then F5 works — VSCode launches the script, `mp.spawn()` creates workers, and `subProcess: true` lets the debugger auto-discover them.

---

## Architecture

The compute node runs `code tunnel`, which connects outbound to Microsoft's relay. Your MacBook's VSCode connects to the same relay via the Remote Tunnels extension. This makes VSCode effectively "local" to the compute node — no sshd or inbound ports needed. From there, `mp.spawn()` + `subProcess: true` gives full multi-process debugging with zero debugpy boilerplate.

**Why not have workers connect to a single VSCode debug listener?** debugpy's adapter only accepts one root connection — subsequent independent processes are silently rejected ([debugpy#1501](https://github.com/microsoft/debugpy/issues/1501)). The `subProcess` flag doesn't help in attach mode either — it only tracks children spawned from a debugpy-instrumented parent in launch mode.

---

## One-Time Setup

### 1. Install the `code` CLI on NFS

Download the standalone CLI so it's available on all nodes:

```bash
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o /tmp/vscode_cli.tar.gz
tar -xzf /tmp/vscode_cli.tar.gz -C ~/.local/bin/
```

Authenticate with your GitHub account (one-time, from any node with internet):

```bash
code tunnel user login --provider github
```

### 2. Install debugpy on NFS

```bash
pip install debugpy
```

### 3. launch.json

The project has `.vscode/launch.json` on NFS. Since the filesystem is shared, this file is already in place when you connect to the compute node:

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug: mp.spawn (local)",
            "type": "debugpy",
            "request": "launch",
            "python": "${workspaceFolder}/.venv/bin/python",
            "program": "${workspaceFolder}/train.py",
            "args": [],
            "env": {
                "WORLD_SIZE": "2"
            },
            "subProcess": true,
            "justMyCode": true
        }
    ]
}
```

- `"python"` points to the UV venv so the debugger uses the correct environment without manual activation.
- `"subProcess": true` tells debugpy to auto-attach to child processes created by `mp.spawn()`.
- `WORLD_SIZE` controls the number of ranks. Set this to the number of GPUs you want to debug with.

---

## Per-Job Workflow

### 1. Start a SLURM job with `code tunnel`

```bash
#!/bin/bash
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1

# Start a VSCode tunnel from the compute node
code tunnel --accept-server-license-terms --name slurm-debug &

# Keep the job alive while you debug
sleep infinity
```

The first time, `code tunnel` will print a device code and URL — open it in your browser to authorize with GitHub.

### 2. Connect VSCode to the compute node

On your MacBook:
1. Install the **Remote - Tunnels** extension if not already installed.
2. Cmd+Shift+P → **"Remote-Tunnels: Connect to Tunnel..."**
3. Select `slurm-debug` (the name from step 1).
4. Open the project folder (same NFS path).

### 3. Debug

1. Go to **Run and Debug** (Cmd+Shift+D).
2. Select **"Debug: mp.spawn (local)"**.
3. Press **F5**.
4. VSCode launches `train.py`, which calls `mp.spawn()` to create workers.
5. Each worker appears as a separate session in the **Call Stack** panel.
6. Set breakpoints anywhere — all ranks hit them.
7. Click a session in Call Stack to switch context: **Variables**, **Watch**, and **Debug Console** all follow.
8. Press **Cmd+Shift+C** to continue all ranks at once (provided by the `debug-continue-all` extension in `vscode_extension_debug_continue_all/`). Or press F5 on individual sessions to continue them one at a time.

### 4. Clean up

Cancel the SLURM job (`scancel`). The tunnel dies with the job.

---

## How train.py Supports Both Modes

The script auto-detects whether it was launched by torchrun or directly:

```python
if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    if os.environ.get("LOCAL_RANK") is not None:
        # Launched by torchrun — already inside a worker
        rank = int(os.environ["LOCAL_RANK"])
        train(rank, world_size)
    else:
        # Direct launch — use mp.spawn for debuggability
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"
        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

- **Debug:** `python train.py` → `mp.spawn()` path → VSCode sees the spawned children.
- **Production:** `torchrun --nproc_per_node=8 train.py` → torchrun path → no debugger overhead.

---

## Gotchas

### DataLoader Workers

If your DataLoader uses `num_workers > 0`, those worker processes are also spawned via `multiprocessing`. With `subProcess: true`, the debugger catches them too — cluttering the Call Stack with irrelevant dataloader processes.

**Fix:** Use `num_workers=0` during debug sessions.

### DDP Collective Hangs

DDP requires all ranks to participate in collective operations (allreduce during backward, broadcast during construction). If you pause one rank while others continue, the running ranks block at the next collective and eventually timeout.

**Mitigations:**

1. **Large timeout**: `dist.init_process_group(timeout=datetime.timedelta(minutes=30))` in code, and/or `export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` (PyTorch 2.1+).
2. **Pause all ranks at the same point** — set a breakpoint that every rank hits, then inspect each before continuing.
3. **Fewer ranks**: `WORLD_SIZE=2` instead of 8. Most bugs reproduce with 2 ranks.
4. **Debug rank-local code only**: Data loading, loss computation, or anything without collectives can be debugged freely.
5. **`gloo` backend for CPU debugging**: More forgiving than NCCL, runs without GPUs.

---

## About pathMappings

`pathMappings` translates file paths between local and remote machines. It tells VSCode: "when debugpy reports line 42 of `/remote/path/train.py`, display line 42 of `/local/path/train.py`."

With Remote Tunnels, pathMappings are unnecessary — VSCode is running on the compute node, so all paths are local. If you ever debug on a machine without shared NFS (e.g., a cloud instance with rsynced code), you'd add:

```jsonc
"pathMappings": [{
    "localRoot": "/home/user/project",
    "remoteRoot": "/opt/app"
}]
```

**Caveat:** pathMappings assumes the source files are identical on both sides. If they diverge, the debugger won't error — it will silently show incorrect information: breakpoints land on wrong lines, highlighted source won't match execution, and stepping appears erratic.

---

## Alternatives

### `torch.distributed.breakpoint()` (PyTorch 2.2+)

The simplest option when you don't need IDE debugging. All ranks must call it; only the specified rank drops into pdb, others wait at a built-in barrier:

```python
torch.distributed.breakpoint(rank=0)
```

Full pdb commands: `n` (step over), `s` (step into), `c` (continue), `p expr` (evaluate), `b file:line` (set breakpoint). The barrier means other ranks don't race ahead — no NCCL timeout risk. But it requires hardcoding the breakpoint call in your source.

### Per-rank `debugpy.listen()` + attach

When Remote Tunnels isn't feasible (e.g., no outbound internet on compute nodes). Each rank listens on its own port:

```python
debugpy.listen(("0.0.0.0", 5678 + rank))
debugpy.wait_for_client()
```

Forward each debug port via SSH tunnel, then create one attach config per rank in launch.json with a compound launch to attach all at once. More setup but doesn't require Remote Tunnels. See `debug_utils.py` for the implementation and the per-rank attach configs in `launch.json`.

### Useful environment variables

| Variable | Purpose |
|---|---|
| `CUDA_LAUNCH_BLOCKING=1` | Synchronous CUDA errors with accurate stack traces |
| `TORCH_DISTRIBUTED_DEBUG=DETAIL` | Reports collective mismatches (shape, dtype, sequence) across ranks |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800` | 30-minute NCCL timeout (default 300s) for debugging |
| `PYDEVD_DISABLE_FILE_VALIDATION=1` | Suppresses debugpy "frozen modules" warning |

---

## Quick Reference

| What | How |
|---|---|
| Start tunnel | SLURM job: `code tunnel --accept-server-license-terms --name slurm-debug &` |
| Connect VSCode | Remote-Tunnels → `slurm-debug` |
| Launch debugger | Select "Debug: mp.spawn (local)", press F5 |
| Switch ranks | Click session in Call Stack panel |
| Continue all ranks | Cmd+Shift+C (requires `debug-continue-all` extension) |
| Continue one rank | Click the session, press F5 |
| Avoid DDP hangs | `dist.init_process_group(timeout=timedelta(minutes=30))` |
| Avoid DataLoader clutter | `num_workers=0` during debugging |

## Checklist

- [ ] `code` CLI installed on NFS and authenticated with GitHub
- [ ] `debugpy` installed in the Python environment
- [ ] SLURM job running with `code tunnel`
- [ ] VSCode connected via Remote-Tunnels extension
- [ ] `WORLD_SIZE` set in launch.json `env` block
- [ ] `"python"` in launch.json points to the correct venv
- [ ] DDP timeout set to a large value in `train.py`
- [ ] `num_workers=0` in DataLoader
- [ ] `debug-continue-all` extension installed (see `vscode_extension_debug_continue_all/cmds.sh`)
- [ ] `justMyCode` set to `true` unless you need to step into PyTorch internals
