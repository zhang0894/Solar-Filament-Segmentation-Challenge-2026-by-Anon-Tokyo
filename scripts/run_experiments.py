"""Serial GPU job runner with explicit process identity and session budget."""

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from filament.data import ROOT
from filament.runtime import (
    process_start_ticks,
    remaining_session_seconds,
    update_session_state,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--wait-run", default="")
    p.add_argument("--reserve-minutes", type=float, default=3.0)
    a = p.parse_args()
    if a.reserve_minutes < 0:
        p.error("reserve-minutes must be nonnegative")
    stop = False
    child = None

    def halt(sig, frame):
        nonlocal stop
        stop = True
        if child is not None and child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, halt)
    signal.signal(signal.SIGINT, halt)
    if a.wait_run:
        record = json.loads((Path(a.wait_run) / "process.json").read_text())
        while not stop and remaining_session_seconds() > 120:
            try:
                if process_start_ticks(record["pid"]) != record["start_ticks"]:
                    break
            except FileNotFoundError:
                break
            time.sleep(1)
    for job in json.loads(Path(a.plan).read_text())["jobs"]:
        remaining = remaining_session_seconds()
        if stop or remaining < job["expected_seconds"] + 60 * a.reserve_minutes:
            print(
                json.dumps(
                    {
                        "event": "queue_stopped",
                        "remaining_seconds": remaining,
                        "next_job": job["name"],
                    }
                ),
                flush=True,
            )
            break
        log = ROOT / "runs" / (job["name"] + ".log")
        if log.exists():
            raise ValueError(f"Refusing to overwrite existing job log: {log}")
        update_session_state({"active_experiment": job["name"], "active_queue": a.plan})
        print(
            json.dumps(
                {
                    "event": "job_start",
                    "name": job["name"],
                    "estimated_seconds": job["expected_seconds"],
                }
            ),
            flush=True,
        )
        with log.open("w") as output:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    job.get("module", "scripts.train"),
                    *job["args"],
                ],
                cwd=ROOT,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
            code = child.wait()
        print(
            json.dumps({"event": "job_end", "name": job["name"], "exit_code": code}),
            flush=True,
        )
        if code:
            raise SystemExit(code)
        child = None


if __name__ == "__main__":
    main()
