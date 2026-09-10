#!/usr/bin/env python3
"""Static release gate for the public repository contents."""

from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "examples/data/synthetic_classification.csv",
    "examples/data/synthetic_regression.csv",
    "pyproject.toml",
    "src/nanoheldi_ml/data.py",
    "src/nanoheldi_ml/models.py",
    "src/nanoheldi_ml/evaluation.py",
    "src/nanoheldi_ml/cli.py",
    "tests/test_grouped_evaluation.py",
    "docs/legacy_code_inventory.csv",
}
FORBIDDEN_CODE_PATTERNS = {
    "post-hoc state search": re.compile(
        r"(?i)(seed[_ -]?search|best[_ -]?seed|top[_ -]?seed|for\s+seed\s+in)"
    ),
    "sample-level random split": re.compile(r"\btrain_test_split\s*\("),
    "pre-CV resampling": re.compile(r"\b(RandomOverSampler|SMOTE|ADASYN)\s*\("),
    "interactive debugger": re.compile(r"\b(pdb\.set_trace|breakpoint)\s*\("),
}
PRIVATE_PATTERNS = {
    "absolute user path": re.compile(r"(?i)([A-Z]:\\\\|/Users/|/home/)"),
    "study-style participant identifier": re.compile(r"(?i)(Id9\d{3}|Subject[_ -]?9\d{3})"),
    "possible credential": re.compile(
        r"(?i)(api[_-]?key\s*=\s*['\"][^'\"]+|password\s*=\s*['\"][^'\"]+)"
    ),
}


def main() -> None:
    errors: list[str] = []
    for required in sorted(REQUIRED):
        if not (ROOT / required).is_file():
            errors.append(f"missing required file: {required}")

    excluded_scanners = {ROOT / "tools/audit_legacy_code.py", Path(__file__).resolve()}
    for path in sorted(ROOT.rglob("*.py")):
        if path in excluded_scanners:
            continue
        relative = path.relative_to(ROOT)
        text = path.read_text(encoding="utf-8")
        try:
            ast.parse(text)
        except SyntaxError as exc:
            errors.append(f"syntax error in {relative}: {exc}")
        for label, pattern in FORBIDDEN_CODE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} in {relative}")
        for label, pattern in PRIVATE_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"{label} in {relative}")

    study_files = [
        path
        for path in (ROOT / "data").glob("*")
        if path.is_file() and path.name != "README.md"
    ]
    if study_files:
        errors.append("data/ contains files other than its disclosure README")
    caches = [path.relative_to(ROOT) for path in ROOT.rglob("__pycache__")]
    caches.extend(path.relative_to(ROOT) for path in ROOT.rglob(".pytest_cache"))
    if caches:
        errors.append("cache directories present: " + ", ".join(map(str, caches)))

    if errors:
        raise SystemExit("RELEASE CHECK FAILED\n- " + "\n- ".join(errors))
    print("RELEASE CHECK PASS")
    print("- required repository files present")
    print("- Python source parses successfully")
    print("- no post-hoc state search, sample-level split, or pre-CV resampling")
    print("- no raw study data, participant IDs, credentials, or absolute user paths")


if __name__ == "__main__":
    main()
