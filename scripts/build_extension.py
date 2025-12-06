#!/usr/bin/env python3
"""
Build script for Blender Assistant MCP extension.

- Reads extension id/version from blender_assistant_mcp/blender_manifest.toml
- Produces zip named: <id>-<version>.zip (e.g., blender_assistant_mcp-2.0.0.zip)
- Packages the blender_assistant_mcp directory as the root of the zip
- Excludes rag_db and common cache/temp directories and files

Usage:
  python build_extension.py
  python build_extension.py --output-dir dist
  python build_extension.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = REPO_ROOT / "blender_assistant_mcp"
MANIFEST_PATH = SOURCE_DIR / "blender_manifest.toml"


def read_manifest_id_version(path: Path) -> Tuple[str, str]:
    """
    Read 'id' and 'version' from blender_manifest.toml.
    Uses Python 3.11+ tomllib when available; falls back to a regex otherwise.
    """
    # Prefer tomllib when available (Python 3.11+)
    try:
        import tomllib  # type: ignore

        with path.open("rb") as f:
            data = tomllib.load(f)
        ext_id = data.get("id")
        version = data.get("version")
        if not isinstance(ext_id, str) or not isinstance(version, str):
            raise ValueError("Missing or invalid 'id'/'version' in manifest.")
        return ext_id, version
    except Exception:
        # Fallback to regex parse (robust to basic TOML formatting)
        text = path.read_text(encoding="utf-8")
        id_match = re.search(r'(?m)^\s*id\s*=\s*"([^"]+)"\s*$', text)
        ver_match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', text)
        if not id_match or not ver_match:
            raise RuntimeError("Could not parse 'id' and 'version' from manifest.")
        return id_match.group(1), ver_match.group(1)


EXCLUDED_DIR_NAMES = {
    # common Python/cache/tooling
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".pytype",
    ".ruff_cache",
    ".eggs",
    ".git",
    ".github",
    ".idea",
    ".vscode",
    "build",
    "dist",
    # miscellaneous caches
    "node_modules",
}


EXCLUDED_FILE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".DS_Store",
}

EXCLUDED_FILE_NAMES = {
    # VCS / OS
    ".DS_Store",
    "Thumbs.db",
}


EXCLUDED_PATH_PATTERNS = (
    # Add glob-like substring patterns to exclude if needed
    # Example: "/tests/" to exclude tests within SOURCE_DIR
)


# Packaging options
INCLUDE_RAG_DB = True
EXCLUDE_BIN = False


def should_exclude_path(root: Path, name: str, is_dir: bool) -> bool:
    """

    Determine whether to exclude a directory entry from packaging.

    """

    if is_dir:
        # Allow including rag_db when configured
        if name == "rag_db" and INCLUDE_RAG_DB:
            return False
        if name == "bin" and EXCLUDE_BIN:
            return True
        if name in EXCLUDED_DIR_NAMES:
            return True

    else:  # file
        if name in EXCLUDED_FILE_NAMES:
            return True

        for suffix in EXCLUDED_FILE_SUFFIXES:
            if name.endswith(suffix):
                return True

    # Additional path-based pattern checks

    full_path_str = str((root / name).as_posix())

    for pattern in EXCLUDED_PATH_PATTERNS:
        if pattern in full_path_str:
            return True

    return False


def iter_files_for_zip(base_dir: Path) -> Iterable[Tuple[Path, Path]]:
    """
    Yield (file_path, arcname) tuples for files to include in the zip.

    arcname is the relative path starting at the repository root so the zip
    contains 'blender_assistant_mcp/...'.
    """
    start_dir = base_dir
    arc_root = base_dir.parent  # ensure top-level folder appears in zip

    for root, dirs, files in os.walk(start_dir):
        root_path = Path(root)

        # Filter directories in-place to avoid descending into them
        dirs[:] = [
            d for d in dirs if not should_exclude_path(root_path, d, is_dir=True)
        ]

        # Emit files
        for fname in files:
            if should_exclude_path(root_path, fname, is_dir=False):
                continue

            file_path = root_path / fname
            arcname = file_path.relative_to(arc_root)
            yield file_path, arcname


def build_zip(output_file: Path, base_dir: Path, dry_run: bool = False) -> int:
    """
    Create the zip archive containing the extension. Returns the number of files added.
    """
    added = 0
    if dry_run:
        print(f"[dry-run] Would create: {output_file}")
        for file_path, arcname in iter_files_for_zip(base_dir):
            print(f"[dry-run]   + {arcname}")
            added += 1
        print(f"[dry-run] Total files: {added}")
        return added

    if output_file.exists():
        print(f"Removing old build: {output_file}")
        output_file.unlink()

    print(f"Creating extension package: {output_file}")
    with zipfile.ZipFile(output_file, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path, arcname in iter_files_for_zip(base_dir):
            zipf.write(file_path, arcname)
            added += 1

    size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✓ Extension built successfully")
    print(f"  File: {output_file}")
    print(f"  Files: {added}")
    print(f"  Size: {size_mb:.2f} MB")
    print(f"  Time: {datetime.now().strftime('%A, %B %d, %Y at %I:%M %p')}")
    print("\nTo install:")
    print("  1. Open Blender 4.2+")
    print("  2. Edit → Preferences → Get Extensions")
    print("  3. Click dropdown (⌄) → Install from Disk")
    print(f"  4. Select: {output_file.resolve()}")
    print("  5. Enable the extension")
    return added


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Package the Blender Assistant MCP extension."
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=REPO_ROOT,
        help="Directory to write the zip to (default: repository root).",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be packaged without creating the zip.",
    )

    parser.add_argument(
        "--include-rag-db",
        action="store_true",
        help="Include rag_db folder in the package (excluded by default).",
    )

    parser.add_argument(
        "--exclude-bin",
        action="store_true",
        help="Exclude 'bin' folder (used for lightweight releases).",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)
    global INCLUDE_RAG_DB
    INCLUDE_RAG_DB = args.include_rag_db
    global EXCLUDE_BIN
    EXCLUDE_BIN = args.exclude_bin

    if not SOURCE_DIR.exists():
        print(f"Error: Source directory not found: {SOURCE_DIR}", file=sys.stderr)
        return 2
    if not MANIFEST_PATH.exists():
        print(f"Error: Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    ext_id, version = read_manifest_id_version(MANIFEST_PATH)
    
    # Suffix for lite builds
    suffix = "-lite" if EXCLUDE_BIN else ""
    output_name = f"{ext_id}-{version}{suffix}.zip"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / output_name

    try:
        build_zip(output_file, SOURCE_DIR, dry_run=args.dry_run)
    except Exception as e:
        print(f"Build failed: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
