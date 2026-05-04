"""
start.py — Happy Paws server launcher.

Usage (from the pet-advisor/ directory):

    python start.py            # default: port 7000, no auto-reload
    python start.py --reload   # auto-reload on code changes (development)
    python start.py --port 7001

This script is equivalent to running:
    uvicorn backend.main:app --port 7000

but works more reliably on Windows (no file-watcher issues with --reload).
"""
import argparse
import socket
import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 12)


def require_python_312():
    if sys.version_info[:2] == REQUIRED_PYTHON:
        return

    current = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    required = ".".join(map(str, REQUIRED_PYTHON))
    print(
        f"Python {required} is required for this project, but you are using Python {current}.\n"
        "Create the environment with:\n"
        "  py -3.12 -m venv .venv\n"
        "  .\\.venv\\Scripts\\Activate.ps1\n"
        "  python -m pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


require_python_312()

# Ensure the pet-advisor directory is on sys.path so "backend" is importable.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import uvicorn
except ImportError:
    print("uvicorn not found. Install it with:  pip install uvicorn[standard]")
    sys.exit(1)


def _check_port(host: str, port: int) -> None:
    """Exit with a clear message if the port is already in use."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
    except OSError:
        print(
            f"\n  Error: port {port} is already in use by another process.\n"
            f"  Find and kill it:\n"
            f"    netstat -ano | findstr :{port}\n"
            f"    taskkill /F /PID <pid>\n"
            f"  Or use a different port:\n"
            f"    python start.py --port 8001",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Happy Paws server")
    parser.add_argument("--host",   default="0.0.0.0",      help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",   default=7000, type=int,  help="Port (default: 7000)")
    parser.add_argument("--reload", action="store_true",     help="Enable auto-reload (dev only)")
    args = parser.parse_args()

    _check_port(args.host, args.port)

    print(f"\n  Happy Paws — starting server")
    print(f"  http://localhost:{args.port}/")
    print(f"  Dr Halim runs separately on http://localhost:7001/\n")

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        reload_dirs=[str(ROOT / "backend")] if args.reload else None,
        log_level="info",
    )


if __name__ == "__main__":
    main()