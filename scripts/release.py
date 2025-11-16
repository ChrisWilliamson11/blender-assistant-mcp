#!/usr/bin/env python3
"""
Release helper for blender-assistant-mcp.

Features:
- Bump version in:
  - blender_assistant_mcp/blender_manifest.toml (id + version)
  - blender_assistant_mcp/__init__.py (__version__ and bl_info["version"])
- Build the zip package via build_extension.py (zip named <id>-<version>.zip)
- Commit changes, create tag, push, and publish GitHub Release via gh CLI

Usage examples:
  # Bump patch (default), build, tag, push, release
  python scripts/release.py

  # Bump minor
  python scripts/release.py --bump minor

  # Bump major
  python scripts/release.py --bump major

  # Set an explicit version
  python scripts/release.py --new-version 2.1.3

  # Dry-run to preview changes and commands without writing or executing them
  python scripts/release.py --dry-run

  # Skip build (only bump + tag + release)
  python scripts/release.py --no-build

  # Mark as pre-release in GitHub
  python scripts/release.py --prerelease

  # Provide custom release notes (overrides --generate-notes)
  python scripts/release.py --notes "Bug fixes and improvements"

  # Specify a GitHub repo explicitly (owner/name). If omitted, gh infers from git remotes
  python scripts/release.py --repo ChrisWilliamson11/blender-assistant-mcp
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

# Paths
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
PKG_DIR = REPO_ROOT / "blender_assistant_mcp"
MANIFEST_PATH = PKG_DIR / "blender_manifest.toml"
INIT_PATH = PKG_DIR / "__init__.py"

# -------------------------------
# Utility helpers
# -------------------------------


def run(
    cmd: list[str],
    cwd: Optional[Path] = None,
    check: bool = True,
    dry_run: bool = False,
) -> subprocess.CompletedProcess | None:
    """Run a command with nice logging."""
    printable = " ".join(cmd)
    print(f"$ {printable}")
    if dry_run:
        return None
    return subprocess.run(cmd, cwd=str(cwd or REPO_ROOT), check=check)


def which(executable: str) -> bool:
    """Return True if executable is on PATH."""
    from shutil import which as _which

    return _which(executable) is not None


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] Would write {path.relative_to(REPO_ROOT)}")
        return
    path.write_text(text, encoding="utf-8")


def error(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"[info] {msg}")


# -------------------------------
# Version parsing/bumping
# -------------------------------

SEMVER_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[.-].*)?$")


def parse_semver(v: str) -> Tuple[int, int, int, str]:
    """
    Parse a semver-like string.
    Returns (major, minor, patch, suffix) where suffix includes any pre-release/build metadata.
    """
    m = SEMVER_RE.match(v)
    if not m:
        raise ValueError(f"Invalid semantic version: {v}")
    major, minor, patch = map(int, m.groups()[:3])
    suffix = v[len(f"{major}.{minor}.{patch}") :]  # retain any suffix
    return major, minor, patch, suffix


def bump_semver(current: str, bump: str) -> str:
    major, minor, patch, _suffix = parse_semver(current)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Unknown bump type: {bump}")


# -------------------------------
# Manifest helpers
# -------------------------------


def read_manifest_id_version(path: Path) -> Tuple[str, str]:
    """
    Read 'id' and 'version' from blender_manifest.toml.
    Prefer tomllib (Python 3.11+) and fallback to regex if not available.
    """
    try:
        import tomllib  # Python 3.11+

        with path.open("rb") as f:
            data = tomllib.load(f)
        ext_id = data.get("id")
        version = data.get("version")
        if not isinstance(ext_id, str) or not isinstance(version, str):
            raise ValueError("Missing or invalid 'id'/'version' in manifest.")
        return ext_id, version
    except Exception:
        text = read_text(path)
        id_match = re.search(r'(?m)^\s*id\s*=\s*"([^"]+)"\s*$', text)
        ver_match = re.search(r'(?m)^\s*version\s*=\s*"([^"]+)"\s*$', text)
        if not id_match or not ver_match:
            raise RuntimeError("Could not parse 'id' and 'version' from manifest.")
        return id_match.group(1), ver_match.group(1)


def write_manifest_version(path: Path, new_version: str, dry_run: bool) -> None:
    text = read_text(path)
    new_text, n = re.subn(
        r'(?m)^(\s*version\s*=\s*")([^"]+)(".*)$',
        rf"\g<1>{new_version}\3",
        text,
        count=1,
    )
    if n == 0:
        raise RuntimeError("Failed to update version in blender_manifest.toml")
    write_text(path, new_text, dry_run)


# -------------------------------
# __init__.py helpers
# -------------------------------


def set_or_insert_dunder_version(init_text: str, new_version: str) -> str:
    """
    Ensure __version__ = "x.y.z" exists and is updated.
    If not found, insert it after the first block of imports or after the docstring.
    """
    # Update existing __version__
    if re.search(r'(?m)^\s*__version__\s*=\s*["\']([^"\']+)["\']\s*$', init_text):
        init_text = re.sub(
            r'(?m)^(\s*__version__\s*=\s*["\'])([^"\']+)(["\']\s*)$',
            rf"\g<1>{new_version}\3",
            init_text,
            count=1,
        )
        return init_text

    # Insert after first import block if possible
    lines = init_text.splitlines()
    insert_idx = 0
    # Skip shebang/encoding/comments and leading docstring
    i = 0
    in_triple = False
    triple_delim = None
    while i < len(lines):
        ln = lines[i]
        stripped = ln.strip()
        if not in_triple and (stripped.startswith('"""') or stripped.startswith("'''")):
            triple_delim = stripped[:3]
            if stripped.count(triple_delim) >= 2 and len(stripped) > 5:
                # one-line docstring, continue
                i += 1
                continue
            in_triple = True
            i += 1
            continue
        if in_triple:
            if triple_delim and triple_delim in stripped:
                in_triple = False
            i += 1
            continue
        # After docstring, skip blank lines and import block
        if (
            stripped == ""
            or stripped.startswith("import ")
            or stripped.startswith("from ")
        ):
            i += 1
            continue
        # Insert before this line
        insert_idx = i
        break
    else:
        insert_idx = len(lines)

    lines.insert(insert_idx, f'__version__ = "{new_version}"')
    return "\n".join(lines) + ("\n" if init_text.endswith("\n") else "")


def set_or_update_bl_info_version(init_text: str, new_version: str) -> str:
    """
    Ensure bl_info contains a "version": (X, Y, Z) entry with new version numbers.
    If bl_info exists but has no version, insert it after the opening brace.
    """
    major, minor, patch, _ = parse_semver(new_version)

    # Try to locate bl_info = { ... }
    bl_info_start = re.search(r"(?m)^\s*bl_info\s*=\s*\{\s*$", init_text)
    if not bl_info_start:
        # If no bl_info block, we won't invent one; rely on __version__
        return init_text

    # Find the end of the dict by scanning braces
    start_idx = bl_info_start.start()
    # Find the line index of the starting brace
    start_line = init_text[:start_idx].count("\n")
    lines = init_text.splitlines()
    # Scan lines to find closing brace for this dict
    brace_depth = 0
    end_line = None
    for i in range(start_line, len(lines)):
        brace_depth += lines[i].count("{")
        brace_depth -= lines[i].count("}")
        if brace_depth == 0:
            end_line = i
            break
    if end_line is None:
        return init_text  # malformed, do nothing

    # Determine indentation for entries inside bl_info
    opening_line = lines[start_line]
    indent = re.match(r"^(\s*)", opening_line).group(1) + "    "

    # Search for existing "version": (...) within bl_info block
    version_re = re.compile(r'(?m)^\s*["\']version["\']\s*:\s*\([^)]+\)\s*,?\s*$')
    found_line = None
    for i in range(start_line + 1, end_line + 1):
        if version_re.search(lines[i]):
            found_line = i
            break

    new_tuple = f"({major}, {minor}, {patch})"
    if found_line is not None:
        # Replace existing tuple
        lines[found_line] = re.sub(
            r'(["\']version["\']\s*:\s*)\([^)]+\)',
            rf"\1{new_tuple}",
            lines[found_line],
        )
    else:
        # Insert a new version field after the opening brace line
        insert_at = start_line + 1
        lines.insert(insert_at, f'{indent}"version": {new_tuple},')

    return "\n".join(lines) + ("\n" if init_text.endswith("\n") else "")


def update_init_py_version(path: Path, new_version: str, dry_run: bool) -> None:
    text = read_text(path)
    updated = set_or_insert_dunder_version(text, new_version)
    updated = set_or_update_bl_info_version(updated, new_version)
    if updated != text:
        write_text(path, updated, dry_run)
    else:
        info(
            "__init__.py did not require changes (no bl_info found and __version__ already correct?)"
        )


# -------------------------------
# Build helpers
# -------------------------------


def build_zip(new_version: str, ext_id: str, dry_run: bool) -> Path:
    """
    Invoke build_extension.py to produce <ext_id>-<new_version>.zip at repo root.
    Returns expected zip path.
    """
    zip_path = REPO_ROOT / f"{ext_id}-{new_version}.zip"

    if dry_run:
        print(f"[dry-run] Would build: {zip_path.name}")
        return zip_path

    # Ensure repo root is on sys.path so we can import scripts.build_extension

    sys.path.insert(0, str(REPO_ROOT))

    try:
        from scripts import build_extension  # type: ignore

    except Exception as e:
        error(f"Could not import scripts/build_extension.py: {e}")

        raise

    # Call its main() entrypoint if present, otherwise fall back to build_extension.build_extension()
    if hasattr(build_extension, "main"):
        rc = build_extension.main([])
        if rc != 0:
            raise RuntimeError(f"build_extension.py failed with exit code {rc}")
    elif hasattr(build_extension, "build_extension"):
        build_extension.build_extension()
    else:
        raise RuntimeError("build_extension.py has no callable entrypoint")

    if not zip_path.exists():
        raise FileNotFoundError(f"Expected build artifact not found: {zip_path}")

    return zip_path


# -------------------------------
# Git/Release helpers
# -------------------------------


def git_is_repo(dry_run: bool) -> bool:
    if dry_run:
        return True
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], check=True)
        return True
    except Exception:
        return False


def git_status_porcelain() -> str:
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return ""


def git_add_commit(files: Iterable[Path], message: str, dry_run: bool) -> None:
    paths = [str(p.relative_to(REPO_ROOT)) for p in files]
    run(["git", "add", *paths], dry_run=dry_run)
    # Commit may fail if nothing changed; handle gracefully
    try:
        run(["git", "commit", "-m", message], dry_run=dry_run)
    except Exception as e:
        info(f"git commit skipped or failed (possibly no changes): {e}")


def git_tag(tag: str, message: str, dry_run: bool) -> None:
    run(["git", "tag", "-a", tag, "-m", message], dry_run=dry_run)


def git_push_with_tags(dry_run: bool) -> None:
    run(["git", "push"], dry_run=dry_run)
    run(["git", "push", "--tags"], dry_run=dry_run)


def gh_release_create(
    tag: str,
    asset: Path,
    title: str,
    notes: Optional[str],
    prerelease: bool,
    repo: Optional[str],
    dry_run: bool,
) -> None:
    if not which("gh"):
        raise RuntimeError(
            "GitHub CLI 'gh' not found on PATH. Install from https://cli.github.com/ and run 'gh auth login'."
        )

    cmd = ["gh", "release", "create", tag, str(asset)]
    cmd += ["-t", title]
    if notes:
        cmd += ["-n", notes]
    else:
        cmd += ["--generate-notes"]
    if prerelease:
        cmd += ["--prerelease"]
    if repo:
        cmd += ["-R", repo]
    run(cmd, dry_run=dry_run)


def gh_release_upload(
    tag: str,
    asset: Path,
    repo: Optional[str],
    clobber: bool,
    dry_run: bool,
) -> None:
    if not which("gh"):
        raise RuntimeError(
            "GitHub CLI 'gh' not found on PATH. Install from https://cli.github.com/ and run 'gh auth login'."
        )
    cmd = ["gh", "release", "upload", tag, str(asset)]
    if clobber:
        cmd += ["--clobber"]
    if repo:
        cmd += ["-R", repo]
    run(cmd, dry_run=dry_run)


# -------------------------------

# CLI / Main

# -------------------------------


@dataclass
class Args:
    bump: Optional[str]

    new_version: Optional[str]

    no_build: bool

    prerelease: bool

    repo: Optional[str]

    notes: Optional[str]

    update_existing: bool
    dry_run: bool


def parse_args(argv: Optional[Iterable[str]] = None) -> Args:
    parser = argparse.ArgumentParser(
        description="Bump version, build zip, tag, and publish a GitHub release."
    )
    bump_group = parser.add_mutually_exclusive_group()
    bump_group.add_argument(
        "--bump",
        choices=["major", "minor", "patch"],
        default="patch",
        help="Bump strategy (default: patch)",
    )
    bump_group.add_argument(
        "--new-version", help="Set an explicit version instead of bumping"
    )

    parser.add_argument("--no-build", action="store_true", help="Skip building the zip")
    parser.add_argument(
        "--prerelease", action="store_true", help="Mark GitHub release as a pre-release"
    )
    parser.add_argument(
        "--repo",
        help="GitHub repository in 'owner/name' form (if omitted, gh infers from git remotes)",
    )

    parser.add_argument(
        "--notes", help="Custom release notes (overrides --generate-notes)"
    )

    parser.add_argument(
        "--update-existing",
        action="store_true",
        help="Upload asset to an existing GitHub release (skip creating a new release)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files or run external commands",
    )

    ns = parser.parse_args(list(argv) if argv is not None else None)
    return Args(
        bump=ns.bump if ns.new_version is None else None,
        new_version=ns.new_version,
        no_build=ns.no_build,
        prerelease=ns.prerelease,
        repo=ns.repo,
        notes=ns.notes,
        update_existing=ns.update_existing,
        dry_run=ns.dry_run,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)

    # Preconditions
    if not MANIFEST_PATH.exists():
        error(f"Manifest not found: {MANIFEST_PATH}")
        return 2
    if not INIT_PATH.exists():
        error(f"__init__.py not found: {INIT_PATH}")
        return 2
    if not git_is_repo(args.dry_run):
        error(
            "Not a Git repository. Initialize with 'git init' and add a remote before releasing."
        )
        return 2

    ext_id, current_version = read_manifest_id_version(MANIFEST_PATH)
    target_version = args.new_version or bump_semver(
        current_version, args.bump or "patch"
    )
    print(f"Current version: {current_version}")
    print(f"Target version:  {target_version}")

    # Update files
    info("Updating blender_manifest.toml version")
    write_manifest_version(MANIFEST_PATH, target_version, args.dry_run)

    info("Updating __init__.py version (__version__ and bl_info['version'])")
    update_init_py_version(INIT_PATH, target_version, args.dry_run)

    # Stage and commit
    info("Committing version bump")
    git_add_commit(
        [MANIFEST_PATH, INIT_PATH], f"chore(release): v{target_version}", args.dry_run
    )

    # Build zip
    if args.no_build:
        info("Skipping build (--no-build set)")
        zip_path = REPO_ROOT / f"{ext_id}-{target_version}.zip"
    else:
        info("Building extension zip")
        try:
            zip_path = build_zip(target_version, ext_id, args.dry_run)
        except Exception as e:
            error(f"Build failed: {e}")
            return 1

    # Tag and push
    tag = f"v{target_version}"
    info(f"Tagging {tag}")
    try:
        git_tag(tag, tag, args.dry_run)
    except Exception as e:
        error(f"Failed to create tag {tag}: {e}")
        return 1

    info("Pushing commits and tags")
    try:
        git_push_with_tags(args.dry_run)
    except Exception as e:
        error(f"Failed to push changes: {e}")
        return 1

    # Create or update GitHub release

    try:
        if args.update_existing:
            info("Uploading asset to existing GitHub release via gh")

            gh_release_upload(
                tag=tag,
                asset=zip_path,
                repo=args.repo,
                clobber=True,
                dry_run=args.dry_run,
            )
        else:
            info("Creating GitHub release via gh")
            gh_release_create(
                tag=tag,
                asset=zip_path,
                title=target_version,
                notes=args.notes,
                prerelease=args.prerelease,
                repo=args.repo,
                dry_run=args.dry_run,
            )

    except Exception as e:
        error(f"Failed to publish GitHub release: {e}")

        return 1

    print("\n✓ Release complete")
    print(f"  Tag:    {tag}")
    print(f"  Asset:  {zip_path.relative_to(REPO_ROOT)}")
    print(f"  Repo:   {args.repo or '(inferred by gh)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
