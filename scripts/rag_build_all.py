from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build indexes for all books")
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    cmd = [sys.executable, str(ROOT / "scripts" / "rag_build.py"), "--all"]
    if args.rebuild:
        cmd.append("--rebuild")

    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
