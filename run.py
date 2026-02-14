#!/usr/bin/env python3
"""Launch the DC Simulator API server and monitoring dashboard.

Usage:
    python run.py                    # Start API + dashboard
    python run.py --api-only         # Start API server only
    python run.py --dashboard-only   # Start dashboard only
    python run.py --port 8080        # Custom API port
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _port_in_use(host: str, port: int) -> bool:
    """Return True if *host:port* is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return False
        except OSError:
            return True


def main():
    parser = argparse.ArgumentParser(description="DC Simulator Launcher")
    parser.add_argument(
        "--port", type=int, default=8000, help="API server port (default: 8000)"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="API server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--api-only", action="store_true", help="Start API server only"
    )
    parser.add_argument(
        "--dashboard-only", action="store_true", help="Start dashboard only"
    )
    args = parser.parse_args()

    processes: list[subprocess.Popen] = []

    def cleanup(signum=None, frame=None):
        """Shut down all child processes cleanly."""
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    # Subprocess environment — add src/ to PYTHONPATH so both dc_sim
    # and agents packages are importable without pip install.
    env = os.environ.copy()
    src_dir = os.path.join(_PROJECT_ROOT, "src")
    env["PYTHONPATH"] = src_dir + os.pathsep + env.get("PYTHONPATH", "")

    if not args.dashboard_only:
        # Pre-flight check: is the port already taken?
        if _port_in_use(args.host, args.port):
            print(
                f"\n  ERROR: Port {args.port} is already in use.\n"
                f"  Kill the other process or choose a different port:\n"
                f"    python run.py --port {args.port + 1}\n"
            )
            sys.exit(1)

        # Start FastAPI server
        api_cmd = [
            sys.executable, "-m", "uvicorn",
            "dc_sim.main:app",
            "--host", args.host,
            "--port", str(args.port),
        ]
        print(f"\n  Starting API server on http://{args.host}:{args.port}")
        print(f"  Interactive docs: http://{args.host}:{args.port}/docs\n")
        api_proc = subprocess.Popen(api_cmd, env=env, cwd=_PROJECT_ROOT)
        processes.append(api_proc)

        if not args.api_only:
            # Give the API a moment to start
            time.sleep(2)

    if not args.api_only:
        # Start Streamlit dashboard
        dashboard_cmd = [
            sys.executable, "-m", "streamlit", "run",
            os.path.join(_PROJECT_ROOT, "dashboard.py"),
            "--server.headless", "true",
        ]
        print("  Starting dashboard on http://localhost:8501\n")
        dash_proc = subprocess.Popen(dashboard_cmd, env=env, cwd=_PROJECT_ROOT)
        processes.append(dash_proc)

    # Wait for any process to exit
    try:
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    print(f"\n  Process exited with code {proc.returncode}. Shutting down...")
                    cleanup()
            time.sleep(1)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
