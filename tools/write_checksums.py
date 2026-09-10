#!/usr/bin/env python3
"""Write SHA-256 checksums for all versioned release-package files."""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "CHECKSUMS.sha256"
EXCLUDED_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".git",
    ".venv",
    "build",
    "dist",
}


def excluded(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return (
        bool(EXCLUDED_PARTS.intersection(parts))
        or any(part.endswith(".egg-info") for part in parts)
        or path.suffix in {".pyc", ".pyo"}
    )


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def checksum_text() -> str:
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and path != OUTPUT
        and not excluded(path)
    ]
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in sorted(files)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify CHECKSUMS.sha256 without modifying it.",
    )
    args = parser.parse_args()
    current = checksum_text()
    if args.check:
        if not OUTPUT.is_file():
            raise SystemExit(f"Checksum file is missing: {OUTPUT}")
        if OUTPUT.read_text(encoding="utf-8") != current:
            raise SystemExit(
                "CHECKSUM VERIFICATION FAILED: packaged files differ from CHECKSUMS.sha256"
            )
        print(f"Verified checksums for {len(current.splitlines())} packaged files")
        return

    OUTPUT.write_text(current, encoding="utf-8")
    lines = current.splitlines()
    print(f"Wrote {len(lines)} checksums to {OUTPUT}")


if __name__ == "__main__":
    main()
