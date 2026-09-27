"""Install the debugpy_auto setup into another repo.

Usage:
    python install.py <target_repo> [--python PATH]

    --python  interpreter of the venv to install into
              (default: <target_repo>/.venv/bin/python)

What it does:
  1. Installs debugpy into the venv (via uv if available; uv venvs have no pip).
  2. Copies debugpy_auto.py + debugpy_auto.pth into the venv's site-packages.
  3. Adds the "DEBUGPY attach" config to <target_repo>/.vscode/launch.json:
     creates the file if missing, appends to it if it's plain JSON, and
     otherwise (JSONC with comments, which we can't rewrite without losing
     them) prints the entry for you to paste in.

Safe to re-run: step 2 overwrites the hook with the current version, and
step 3 skips a launch.json that already has the config.
Standard library only, so any python3 can run it.
Assumes the mp-spawn-debug extension is already installed on the VSCode server.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
HOOK_FILES = [SRC / "debugpy_auto" / "debugpy_auto.py", SRC / "debugpy_auto" / "debugpy_auto.pth"]
LAUNCH_TEMPLATE = SRC / ".vscode" / "launch.json"
CONFIG_NAME = "DEBUGPY attach"


def run_py(python, code):
    """Run a snippet in the target interpreter and return its stripped stdout."""
    return subprocess.run([python, "-c", code], capture_output=True, text=True, check=True).stdout.strip()


def load_jsonc(text):
    """Parse our own launch.json template: JSON plus whole-line // comments."""
    return json.loads(re.sub(r"^\s*//.*$", "", text, flags=re.MULTILINE))


def install_debugpy(python):
    try:
        version = run_py(python, "import debugpy; print(debugpy.__version__)")
        print(f"[1/3] debugpy already installed ({version})")
        return
    except subprocess.CalledProcessError:
        pass
    print("[1/3] installing debugpy")
    if shutil.which("uv"):
        cmd = ["uv", "pip", "install", "--python", str(python), "debugpy"]
    else:
        cmd = [str(python), "-m", "pip", "install", "debugpy"]
    subprocess.run(cmd, check=True)


def install_hook(python):
    site = Path(run_py(python, "import sysconfig; print(sysconfig.get_paths()['purelib'])"))
    for f in HOOK_FILES:
        shutil.copy2(f, site / f.name)
    # The .pth is processed at interpreter startup, so the module is already in
    # sys.modules if the hook is active.
    if run_py(python, "import sys; print('debugpy_auto' in sys.modules)") != "True":
        sys.exit(f"error: copied hook to {site} but it isn't loaded at startup")
    print(f"[2/3] site hook installed in {site}")


def install_launch_config(repo):
    dest = repo / ".vscode" / "launch.json"
    template = LAUNCH_TEMPLATE.read_text()
    config = next(c for c in load_jsonc(template)["configurations"] if c["name"] == CONFIG_NAME)

    if not dest.exists():
        dest.parent.mkdir(exist_ok=True)
        dest.write_text(template)
        print(f"[3/3] created {dest}")
        return

    text = dest.read_text()
    if f'"{CONFIG_NAME}"' in text:
        print(f'[3/3] {dest} already has the "{CONFIG_NAME}" config')
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        print(f'[3/3] {dest} has comments or isn\'t plain JSON; add this entry to its "configurations" list:')
        print(json.dumps(config, indent=4))
        return
    data.setdefault("configurations", []).append(config)
    dest.write_text(json.dumps(data, indent=4) + "\n")
    print(f'[3/3] added the "{CONFIG_NAME}" config to {dest}')


def main():
    parser = argparse.ArgumentParser(description="Install the debugpy_auto setup into another repo.")
    parser.add_argument("repo", type=Path, help="target repo")
    parser.add_argument("--python", type=Path, help="venv interpreter (default: <repo>/.venv/bin/python)")
    args = parser.parse_args()

    repo = args.repo.resolve()
    if not repo.is_dir():
        sys.exit(f"error: {repo} is not a directory")
    # Don't resolve symlinks here: .venv/bin/python is a symlink to the base
    # interpreter, and resolving it would install into the base, not the venv.
    python = (args.python or repo / ".venv" / "bin" / "python").absolute()
    if not python.exists():
        sys.exit(f"error: no python at {python}; create the venv first or pass --python")

    install_debugpy(python)
    install_hook(python)
    install_launch_config(repo)
    print(f'\nDone. Press F5 on "{CONFIG_NAME}", then: DEBUGPY_AUTO=1 python <script> [args...]')


if __name__ == "__main__":
    main()
