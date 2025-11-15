#!/usr/bin/env python3

"""

Update the bundled Ollama binaries inside this repo.



Usage:

  python scripts/update_ollama_bins.py --src "C:\\Program Files\\Ollama" [--dry-run] [--clean]



What it does:

- Creates a timestamped backup of blender_assistant_mcp/bin

- Optionally cleans stale .exe/.dll from blender_assistant_mcp/bin (with --clean)
- Copies all ollama*.exe and all .dll files from --src into blender_assistant_mcp/bin

- Prints the new ollama --version for verification



Notes:

- This is a developer tool: run it locally to refresh the repo's bundled binaries

- End users do not run this; the add-on ships with these updated files

"""

import argparse
import datetime
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIN_DIR = REPO_ROOT / "blender_assistant_mcp" / "bin"

# Typical Windows install locations to try if --src is omitted
DEFAULT_WIN_PATHS = [
    Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Ollama",
    Path("C:/Program Files/Ollama"),
    Path("C:/Program Files (x86)/Ollama"),
]


def find_default_src() -> Path | None:
    for p in DEFAULT_WIN_PATHS:
        if (p / "ollama.exe").exists():
            return p
    return None


def make_backup(dst_dir: Path) -> Path:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = dst_dir.parent / f"{dst_dir.name}_backup_{ts}"
    if backup_dir.exists():
        raise RuntimeError(f"Backup directory already exists: {backup_dir}")
    print(f"[update] Creating backup: {backup_dir}")
    shutil.copytree(dst_dir, backup_dir)
    return backup_dir


def copy_binaries(
    src_dir: Path, dst_dir: Path, dry_run: bool = False, cuda_choice: str = "auto"
) -> list[Path]:
    """Copy server executable and required DLLs from src_dir to dst_dir.



    - Executable: only the CLI server 'ollama.exe' (skips GUI like 'ollama app.exe')

    - DLLs: from lib/ollama plus CUDA runtime DLLs from lib/ollama/cuda_v12 or cuda_v13

            CUDA selection controlled by cuda_choice: 'auto' | '12' | '13'
    Returns list of files copied.

    """

    wanted_exts = {".exe", ".dll"}

    filenames: list[Path] = []

    # Copy only the server executable (exclude GUI launchers)
    exe = src_dir / "ollama.exe"
    if exe.exists():
        filenames.append(exe)

    # Base DLLs under lib/ollama
    lib_dir = src_dir / "lib" / "ollama"
    if lib_dir.exists():
        for p in lib_dir.glob("*.dll"):
            filenames.append(p)

        # CUDA runtime selection
        cuda_dir_13 = lib_dir / "cuda_v13"
        cuda_dir_12 = lib_dir / "cuda_v12"

        selected_cuda_dir: Path | None = None
        if cuda_choice == "13":
            selected_cuda_dir = cuda_dir_13 if cuda_dir_13.exists() else None
        elif cuda_choice == "12":
            selected_cuda_dir = cuda_dir_12 if cuda_dir_12.exists() else None
        else:
            # auto: prefer v13 when present, otherwise v12
            if cuda_dir_13.exists():
                selected_cuda_dir = cuda_dir_13
            elif cuda_dir_12.exists():
                selected_cuda_dir = cuda_dir_12

        if selected_cuda_dir and selected_cuda_dir.exists():
            print(f"[update] Using CUDA DLLs from {selected_cuda_dir}")
            for p in selected_cuda_dir.glob("*.dll"):
                filenames.append(p)

    if not filenames:
        raise RuntimeError(f"No binaries found in {src_dir}")

    copied: list[Path] = []

    for src in filenames:
        if src.suffix.lower() not in wanted_exts:
            continue

        dst = dst_dir / src.name

        print(f"[update] Copy {src} -> {dst}")

        if not dry_run:
            shutil.copy2(src, dst)

        copied.append(dst)

    return copied


def clean_bin_dir(dst_dir: Path, dry_run: bool = False) -> int:
    """Remove existing .exe and .dll files from the destination bin folder.
    Returns number of files slated for removal.
    """
    patterns = ["*.exe", "*.dll"]
    removed = 0
    for pat in patterns:
        for p in dst_dir.glob(pat):
            print(f"[clean] Remove {p}")
            removed += 1
            if not dry_run:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"[clean] Failed to remove {p}: {e}")
    return removed

    # (removed unreachable code; logic moved into copy_binaries)


def run_version(dst_dir: Path) -> str:
    exe = dst_dir / "ollama.exe"
    if not exe.exists():
        return "(ollama.exe not found)"
    try:
        # Some builds support --version; if not, print what we can
        out = subprocess.check_output(
            [str(exe), "--version"], stderr=subprocess.STDOUT, timeout=10
        )
        return out.decode("utf-8", errors="ignore").strip()
    except subprocess.CalledProcessError as e:
        return f"(version command failed: {e.output.decode('utf-8', errors='ignore').strip()})"
    except Exception as e:
        return f"(version check failed: {e})"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Update bundled Ollama binaries in repo (copy ollama*.exe and all DLLs; optional clean)"
    )

    parser.add_argument(
        "--src",
        type=str,
        default=None,
        help="Source folder containing ollama*.exe and DLLs (e.g., C:/Program Files/Ollama)",
    )

    parser.add_argument(
        "--dry-run", action="store_true", help="Show actions without writing"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing .exe/.dll from destination bin before copying",
    )

    parser.add_argument(
        "--cuda",
        choices=["auto", "12", "13"],
        default="auto",
        help="Select CUDA runtime DLLs to copy from lib/ollama (auto prefers v13 when available)",
    )
    args = parser.parse_args(argv)

    src_dir = Path(args.src) if args.src else find_default_src()
    if not src_dir or not (src_dir / "ollama.exe").exists():
        print(
            "[error] Could not locate source ollama.exe. Provide --src to the Ollama install folder."
        )
        print('        Example: --src "C:/Program Files/Ollama"')
        return 1

    if not BIN_DIR.exists():
        print(f"[error] Destination bin folder not found: {BIN_DIR}")
        return 1

    print(f"[update] Repo root: {REPO_ROOT}")

    print(f"[update] Source:    {src_dir}")

    print(f"[update] Dest bin:  {BIN_DIR}")

    print(f"[update] Dry run:   {args.dry_run}")

    print(f"[update] Clean:     {args.clean}")
    print(f"[update] CUDA:      {args.cuda}")

    try:
        if not args.dry_run:
            make_backup(BIN_DIR)

        if args.clean:
            removed = clean_bin_dir(BIN_DIR, dry_run=args.dry_run)

            print(f"[update] Cleaned {removed} files")

        copied = copy_binaries(
            src_dir, BIN_DIR, dry_run=args.dry_run, cuda_choice=args.cuda
        )

        print(f"[update] Copied {len(copied)} files")

        if not args.dry_run:
            ver = run_version(BIN_DIR)

            print(f"[update] New ollama version: {ver}")

        else:
            print("[update] Dry run: version check skipped")

        print("[update] Done.")

        return 0

    except Exception as e:
        print(f"[error] Update failed: {e}")

        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
