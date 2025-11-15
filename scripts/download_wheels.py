#!/usr/bin/env python3
"""
Download the exact wheel files used by this project into blender_assistant_mcp/wheels.

This script downloads the specific package versions that match the current
wheels directory layout so that a new user's folder structure matches yours.

It does NOT try to discover dependencies or update the manifest.

By default, it targets Windows x86_64 and Python 3.11 with CPython ABI (cp311),
to reproduce filenames like *-cp311-cp311-win_amd64.whl when applicable.

Usage:
  # From repo root (recommended)
  python scripts/download_wheels.py

  # From anywhere
  python path/to/scripts/download_wheels.py

Advanced:
  # Override target tags if needed (ex: building on another OS but want Windows wheels)
  python scripts/download_wheels.py --platform win_amd64 --python 3.11 --implementation cp --abi cp311

  # Skip wheels that already exist in the destination
  python scripts/download_wheels.py --skip-existing

  # Fail if any wheel is missing (default behavior)
  python scripts/download_wheels.py --strict
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

# Repository-relative paths
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
PKG_DIR = REPO_ROOT / "blender_assistant_mcp"
WHEELS_DIR = PKG_DIR / "wheels"

# EXACT wheels list (package==version) to download.
# These are pinned to match the filenames currently present under wheels/.
# Note: Package names follow normalized PyPI naming (hyphens rather than underscores).
EXACT_SPECS: List[str] = [
    "annotated-types==0.7.0",
    "anyio==4.11.0",
    "attrs==25.3.0",
    "certifi==2025.8.3",
    "click==8.3.0",
    "colorama==0.4.6",
    "diskcache==5.6.3",
    "h11==0.16.0",
    "httpcore==1.0.9",
    "httpx==0.28.1",
    "httpx-sse==0.4.1",
    "idna==3.10",
    "Jinja2==3.1.6",
    "jsonschema==4.25.1",
    "jsonschema-specifications==2025.9.1",
    "MarkupSafe==3.0.3",
    "mcp==1.16.0",
    "numpy==2.3.3",
    "Pillow==11.3.0",
    "pydantic==2.11.9",
    "pydantic-core==2.33.2",
    "pydantic-settings==2.11.0",
    "python-dotenv==1.1.1",
    "python-multipart==0.0.20",
    "pywin32==311",
    "referencing==0.36.2",
    "rpds-py==0.27.1",
    "sniffio==1.3.1",
    "sse-starlette==3.0.2",
    "starlette==0.48.0",
    "typing-extensions==4.15.0",
    "typing-inspection==0.4.2",
    "uvicorn==0.37.0",
]


def run(cmd: List[str], cwd: Path | None = None, check: bool = True) -> int:
    print("$ " + " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {proc.returncode}: {' '.join(cmd)}"
        )
    return proc.returncode


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def pip_download(
    specs: Iterable[str],
    dest: Path,
    platform: str | None,
    python_version: str | None,
    implementation: str | None,
    abi: str | None,
    skip_existing: bool,
) -> None:
    ensure_dir(dest)

    base_cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--only-binary",
        ":all:",
        "--no-deps",
        "--dest",
        str(dest),
    ]
    # Cross-target flags to reproduce exact wheel tags (defaults chosen for Blender 4.2 on Windows)
    if platform:
        base_cmd += ["--platform", platform]
    if python_version:
        base_cmd += ["--python-version", python_version]
    if implementation:
        base_cmd += ["--implementation", implementation]
    if abi:
        base_cmd += ["--abi", abi]

    for spec in specs:
        if skip_existing:
            # Best-effort skip: if any wheel with this version already exists in dest, skip download
            # This is a heuristic: match by package name (case-insensitive) and version token.
            name, vers = spec.split("==", 1)
            found = False
            for whl in dest.glob("*.whl"):
                lower = whl.name.lower()
                if name.lower().replace("-", "_") in lower and f"-{vers}-" in lower:
                    found = True
                    break
            if found:
                print(f"[skip-existing] {spec}")
                continue

        cmd = base_cmd + [spec]
        run(cmd, cwd=REPO_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download exact wheel files into blender_assistant_mcp/wheels."
    )
    p.add_argument(
        "--platform",
        default="win_amd64",
        help="Target platform tag for wheels (default: win_amd64).",
    )
    p.add_argument(
        "--python",
        default="3.11",
        help="Target Python version (default: 3.11).",
    )
    p.add_argument(
        "--implementation",
        default="cp",
        help="Target Python implementation tag (default: cp for CPython).",
    )
    p.add_argument(
        "--abi",
        default="cp311",
        help="Target ABI tag (default: cp311).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading a wheel if a matching version already exists in the destination.",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing wheels before downloading.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    print("Destination wheels directory:", WHEELS_DIR)
    ensure_dir(WHEELS_DIR)

    if args.clean and WHEELS_DIR.exists():
        print("[clean] Removing existing wheels...")
        for f in WHEELS_DIR.glob("*.whl"):
            try:
                f.unlink()
            except Exception as e:
                print(f"[warn] Could not remove {f}: {e}")

    # Verify pip is available
    if shutil.which(sys.executable) is None:
        print("Error: Could not find Python executable.", file=sys.stderr)
        return 2

    print("\nDownloading pinned wheels...")
    print(f"  Platform:        {args.platform}")
    print(f"  Python version:  {args.python}")
    print(f"  Implementation:  {args.implementation}")
    print(f"  ABI:             {args.abi}\n")

    try:
        pip_download(
            specs=EXACT_SPECS,
            dest=WHEELS_DIR,
            platform=args.platform,
            python_version=args.python,
            implementation=args.implementation,
            abi=args.abi,
            skip_existing=args.skip_existing,
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130

    print("\n✓ Done. Wheels present in:", WHEELS_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
