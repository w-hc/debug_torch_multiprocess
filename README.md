# Debugging Distributed PyTorch with VSCode

## Assumption
Your distributed torch code runs on a remote compute node. Your laptop runs the VSCode client.
What's described here should work whether you use a cloud box or slurm + NFS cluster.

## Four Approaches

| Approach | How it works | Pros | Cons |
|---|---|---|---|
| **VSCode Remote Tunnel + `debugpy.connect()` + `torch.multiprocessing.spawn`** | `code tunnel` on compute node; VSCode connects via Microsoft relay; VSCode listens for debug connections; script calls `debugpy.connect()` then `torch.multiprocessing.spawn`; `subProcess: true` auto-discovers workers | Full IDE debugging. Args stay on the command line — launch.json never changes. Works with any entry script. | Must replace torchrun with `torch.multiprocessing.spawn` (ok for debug). Requires `pip install debugpy`. |
| **VSCode Remote Tunnel + F5 launch + `torch.multiprocessing.spawn`** | Same tunnel setup; F5 launches the script directly with `subProcess: true` | Simplest workflow — just press F5. No debugpy boilerplate in code. | CLI args must go in launch.json `"args"` array — painful when args are complex or change frequently. |
| **Per-rank `debugpy.listen()`** | Each rank listens on its own debug port; VSCode attaches to each | Works with torchrun. Works multi-node. | One attach config + SSH tunnel per rank. Attach mode — can't auto-discover processes. |
| **`torch.distributed.breakpoint()`** | One line in source; paused rank drops into pdb; others wait at built-in barrier | Zero setup. Built-in rank sync — no NCCL timeout risk. | Terminal pdb only (no IDE). Must hardcode breakpoint calls. One rank at a time. |

**Recommendation:** Use approach 1 for interactive debugging — it gives you full IDE features while keeping your launch workflow identical to production (`python train.py [args...]`). Fall back to `torch.distributed.breakpoint()` for quick one-off inspections.

---

## Preliminary: VSCode DAP, debugpy

VSCode has a sophisticated multi-language debug toolchain — it speaks DAP (Debug Adapter Protocol) to language-specific debug adapters. For Python, that adapter is `debugpy`. When you press F5, VSCode launches `debugpy`, which instruments the Python process and communicates breakpoints, step commands, and variable inspection back to the IDE over DAP. The VSCode Python/debugpy extension bundles its own copy of debugpy, so you don't need to `pip install` it for normal debugging. You only need a separate `pip install debugpy` if your code explicitly imports it — which the recommended approach does.

The recommended approach uses debugpy in **attach-listen** mode, which is analogous to `gdbserver` for C/C++. VSCode starts a debug adapter that listens on a port, and the script connects to it via `debugpy.connect()`. The key difference from `gdbserver`: VSCode is an **observer**, not the process owner. It can set breakpoints and inspect state, but the process owns its own lifecycle — you get a "disconnect" button instead of "stop", and you Ctrl+C in the terminal to kill the script.


## Technical challenges

Two things break normal VSCode debugging in distributed training:

1. **torchrun is the entrypoint, not python.** `torchrun --nproc_per_node=8 train.py` spawns a process tree that VSCode can't manage. There's an open feature request for console-script support ([debugpy#1311](https://github.com/microsoft/debugpy/issues/1311)).

2. **Execution happens on a different node.** Even if VSCode could launch torchrun, it would run locally, not on the compute node with the GPUs.

**What does not work:** using `debugpy.connect()` to have each worker process independently connect to the single VSCode debug listener. `debugpy`'s adapter only accepts one root connection — subsequent independent processes are silently rejected ([debugpy#1501](https://github.com/microsoft/debugpy/issues/1501)). The `subProcess` flag doesn't help here either — it only tracks children spawned from a debugpy-instrumented parent, not independent processes connecting separately.

**The fix:** have a single parent process call `debugpy.connect()`, then use `torch.multiprocessing.spawn()` to create workers. Because the children are forked from the instrumented parent (not independent processes), `subProcess: true` lets the debug adapter auto-discover them. VSCode Remote Tunnels gets VSCode onto the compute node so the connection is just localhost.

---

## Architecture

The compute node runs `code tunnel`, which connects outbound to Microsoft's relay. Your laptop's VSCode connects to the same relay via the Remote Tunnels extension. This makes VSCode effectively "local" to the compute node — no sshd or inbound ports needed.

For debugging, VSCode starts a debug adapter that listens on localhost. The training script calls `debugpy.connect()` to hook into it, then `torch.multiprocessing.spawn` creates workers. Because the parent process is debugpy-instrumented, `subProcess: true` causes each worker to auto-connect to the same adapter. All ranks appear in the Call Stack panel.

---

## One-Time Setup

### 1. Install the `code` CLI on the remote machine

Download the standalone CLI:

```bash
curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' -o /tmp/vscode_cli.tar.gz
tar -xzf /tmp/vscode_cli.tar.gz -C ~/.local/bin/
```

Authenticate with your GitHub account (one-time):

```bash
code tunnel user login --provider github
```

### 2. Install debugpy in your Python environment

The recommended approach imports debugpy in your training script, so it must be installed:

```bash
pip install debugpy
```

(The VSCode Python extension bundles its own debugpy for F5-launch debugging, but that copy isn't importable by your code.)

### 3. .vscode/launch.json

```jsonc
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Attach: torch.mp.spawn (listen)",
            "type": "debugpy",
            "request": "attach",
            "listen": { "host": "localhost", "port": 5678 },
            "subProcess": true,
            "justMyCode": true
        }
    ]
}
```

- `"request": "attach"` with `"listen"` — VSCode starts a debug adapter that listens on port 5678, waiting for the script to connect. This is the key: **launch.json never changes**, no matter what args your script takes.
- `"subProcess": true` tells the adapter to accept child processes spawned by the connected parent.
- `"justMyCode": true` skips stepping into PyTorch/library internals (set to `false` if you need to).

### 4. Install the debug-continue-all extension

VSCode has no built-in way to continue all debug sessions at once ([microsoft/vscode#245058](https://github.com/microsoft/vscode/issues/245058) — closed as "Not Planned"). With DDP, this means clicking the continue button on each rank individually — tedious with 8 ranks, and risky because ranks that resume early may hit a collective and timeout while you're still clicking through the rest.

The `debug-continue-all` extension in `vscode_extension_debug_continue_all/` solves this. It sends DAP `continue` requests to all active debug sessions in parallel via `Promise.all`, minimizing the window between the first and last rank resuming.

**Keybinding:** Cmd+Shift+C (mac) / Ctrl+Shift+C (linux), or Cmd+Shift+P → "Debug: Continue All Sessions".

**Install** (symlink — edits to source take effect on reload):

```bash
ln -s "$(pwd)/vscode_extension_debug_continue_all" ~/.vscode-server/extensions/debug-continue-all
```

Then reload VSCode: Cmd+Shift+P → "Developer: Reload Window".

With this extension, you set breakpoints in the VSCode UI interactively, and let all ranks
continue til the breakpoint.

---

## Per-Job Workflow

### 1. Start `code tunnel` on the remote machine

```bash
code tunnel --accept-server-license-terms --name my-gpu-box
```

The first time, `code tunnel` will print a device code and URL — open it in your browser to authorize with GitHub.

### 2. Connect VSCode to the compute node

On your laptop:
1. Install the **Remote - Tunnels** extension if not already installed.
2. Cmd+Shift+P → **"Remote-Tunnels: Connect to Tunnel..."**
3. Select `my-gpu-box` (the name from step 1).
4. Open the project folder.

### 3. Debug

1. Go to **Run and Debug** (Cmd+Shift+D).
2. Select **"Attach: torch.mp.spawn (listen)"** and press **F5**. VSCode is now listening on port 5678.
3. In the **integrated terminal**, run your script with whatever args you need:
   ```bash
   DEBUG=1 WORLD_SIZE=2 python train.py --lr 0.001 --batch-size 32
   ```
4. The script calls `debugpy.connect()`, hooks into the waiting adapter, then `torch.multiprocessing.spawn` creates workers.
5. Each worker appears as a separate session in the **Call Stack** panel.
6. Set breakpoints anywhere — all ranks hit them.
7. Click a session in Call Stack to switch context: **Variables**, **Watch**, and **Debug Console** all follow.
8. Press **Cmd+Shift+C** to continue all ranks at once (provided by the `debug-continue-all` extension). Or press F5 on individual sessions to continue them one at a time.
9. When done, let the script finish naturally, or **Ctrl+C in the terminal** to kill it. The VSCode debug session shows a disconnect button (not stop) because VSCode is observing, not owning the process.

### 4. Clean up

Kill the `code tunnel` process (or cancel the job if using a scheduler). The tunnel dies with the process.

---

## How train.py Supports Both Modes

The script auto-detects whether it was launched by torchrun or directly, and optionally connects to the debugger:

```python
if __name__ == "__main__":
    world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))

    if os.environ.get("LOCAL_RANK") is not None:
        # Launched by torchrun — already inside a worker
        rank = int(os.environ["LOCAL_RANK"])
        train(rank, world_size)
    else:
        # Direct launch — use torch.multiprocessing.spawn for debuggability
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        if os.environ.get("DEBUG"):
            import debugpy
            debugpy.connect(("localhost", int(os.environ.get("DEBUG_PORT", 5678))))
            debugpy.wait_for_client()

        torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)
```

- **Debug:** `DEBUG=1 WORLD_SIZE=2 python train.py [args...]` → connects to VSCode, then `torch.multiprocessing.spawn` → all ranks debuggable.
- **Production:** `torchrun --nproc_per_node=8 train.py [args...]` → torchrun path → no debugger overhead.
- **Non-debug direct:** `python train.py [args...]` → `torch.multiprocessing.spawn` without debugpy → no overhead.

The `DEBUG` env var gates debugpy so there's zero import cost in normal runs. `DEBUG_PORT` defaults to 5678 but can be overridden if that port is taken.

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

### Disconnect vs Stop

Because VSCode is in attach-listen mode, it observes the process rather than owning it. The debug toolbar shows **disconnect** instead of **stop**. To end the script, Ctrl+C in the terminal or let it finish naturally.

---

## About pathMappings

`pathMappings` translates file paths between local and remote machines. It tells VSCode: "when debugpy reports line 42 of `/remote/path/train.py`, display line 42 of `/local/path/train.py`."

With Remote Tunnels, pathMappings are unnecessary — VSCode is running on the remote machine, so all paths are local. If you ever debug where the source lives at different paths on each side (e.g., rsynced code), you'd add:

```jsonc
"pathMappings": [{
    "localRoot": "/home/user/project",
    "remoteRoot": "/opt/app"
}]
```

**Caveat:** pathMappings assumes the source files are identical on both sides. If they diverge, the debugger won't error — it will silently show incorrect information: breakpoints land on wrong lines, highlighted source won't match execution, and stepping appears erratic.

---

## Alternatives

### F5 Launch with `torch.multiprocessing.spawn`

If your script has few, stable args, you can skip the `debugpy.connect()` setup and use a standard launch config:

```jsonc
{
    "name": "Debug: torch.mp.spawn (launch)",
    "type": "debugpy",
    "request": "launch",
    "python": "${workspaceFolder}/.venv/bin/python",
    "program": "${workspaceFolder}/train.py",
    "args": ["--lr", "0.001"],
    "env": { "WORLD_SIZE": "2" },
    "subProcess": true,
    "justMyCode": true
}
```

The downside: every time your args change, you edit launch.json. For repos with multiple entry scripts and complex, frequently changing arguments, this gets messy fast.

### `torch.distributed.breakpoint()` (PyTorch 2.2+)

The simplest option when you don't need IDE debugging. All ranks must call it; only the specified rank drops into pdb, others wait at a built-in barrier:

```python
torch.distributed.breakpoint(rank=0)
```

Full pdb commands: `n` (step over), `s` (step into), `c` (continue), `p expr` (evaluate), `b file:line` (set breakpoint). The barrier means other ranks don't race ahead — no NCCL timeout risk. But it requires hardcoding the breakpoint call in your source.

### Per-rank `debugpy.listen()` + attach

When Remote Tunnels isn't feasible (e.g., no outbound internet on compute nodes). Each rank listens on its own port. Requires `pip install debugpy` in your Python environment:

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
| Start tunnel | `code tunnel --accept-server-license-terms --name my-gpu-box &` |
| Connect VSCode | Remote-Tunnels → `my-gpu-box` |
| Start debug listener | Select "Attach: torch.mp.spawn (listen)", press F5 |
| Launch script | `DEBUG=1 WORLD_SIZE=2 python train.py [args...]` |
| Switch ranks | Click session in Call Stack panel |
| Continue all ranks | Cmd+Shift+C (requires `debug-continue-all` extension) |
| Continue one rank | Click the session, press F5 |
| Kill script | Ctrl+C in terminal (or let it finish) |
| Avoid DDP hangs | `dist.init_process_group(timeout=timedelta(minutes=30))` |
| Avoid DataLoader clutter | `num_workers=0` during debugging |

## Checklist

- [ ] `code` CLI installed on the remote machine and authenticated with GitHub
- [ ] `debugpy` installed in the Python environment (`pip install debugpy`)
- [ ] `code tunnel` running on the remote machine
- [ ] VSCode connected via Remote-Tunnels extension
- [ ] launch.json has the `Attach: torch.mp.spawn (listen)` config
- [ ] Training script has the `debugpy.connect()` block gated by `DEBUG` env var
- [ ] DDP timeout set to a large value in `train.py`
- [ ] `num_workers=0` in DataLoader during debug
- [ ] `debug-continue-all` extension installed (see `vscode_extension_debug_continue_all/cmds.sh`)
- [ ] `justMyCode` set to `true` unless you need to step into PyTorch internals
