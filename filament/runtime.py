"""Record the actual training parent process for precise stop requests."""

import fcntl
import json
import os
import signal
import time
from pathlib import Path


def configure_deterministic_inference():
    """Fix CUDA algorithm selection before creating any inference CUDA state."""
    import torch

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def remaining_session_seconds():
    state_path = Path(__file__).resolve().parents[1] / "RUN_STATE.json"
    if not state_path.exists():
        return float("inf")
    state = json.loads(state_path.read_text())
    if state.get("status") in {"paused", "stopped", "complete"}:
        return 0.0
    deadline = state.get("session_deadline_unix")
    return float(deadline) - time.time() if deadline else float("inf")


def update_session_state(fields):
    root = Path(__file__).resolve().parents[1]
    path = root / "RUN_STATE.json"
    with (root / ".run_state.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text()) if path.exists() else {}
        state.update(fields)
        temp = root / f".run_state.{os.getpid()}.tmp"
        temp.write_text(json.dumps(state, indent=2))
        temp.replace(path)


def process_start_ticks(pid):
    return (
        (Path("/proc") / str(pid) / "stat").read_text().rsplit(") ", 1)[1].split()[19]
    )


def register_training_process(run):
    pid = os.getpid()
    path = run / "process.json"
    if path.exists():
        previous = json.loads(path.read_text())
        try:
            if process_start_ticks(previous["pid"]) == previous["start_ticks"]:
                raise RuntimeError(
                    f"This run already has a live training process: {previous['pid']}"
                )
        except FileNotFoundError:
            pass
    temp = run / "process.partial.json"
    temp.write_text(
        json.dumps({"pid": pid, "start_ticks": process_start_ticks(pid)}, indent=2)
    )
    temp.replace(path)


def stop_training_process(run):
    record = json.loads((Path(run) / "process.json").read_text())
    pid = record["pid"]
    try:
        actual = process_start_ticks(pid)
    except FileNotFoundError:
        return {"pid": pid, "status": "already_stopped"}
    if actual != record["start_ticks"]:
        raise ValueError("PID was reused; refusing to signal a different process")
    os.kill(pid, signal.SIGTERM)
    return {"pid": pid, "status": "save_and_stop_requested"}
