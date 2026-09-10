#!/usr/bin/env python3
"""Create a non-invasive inventory of legacy Python, notebook, and JSON files."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PATTERNS = {
    "seed_search": re.compile(
        r"(?i)(seed[_ -]?search|best[_ -]?seed|top[_ -]?seed|for\s+seed\s+in|range\([^\n]*1000)"
    ),
    "random_state": re.compile(r"(?i)(random_state\s*=|np\.random\.seed|random\.seed|manual_seed)"),
    "sample_level_split": re.compile(r"\btrain_test_split\s*\("),
    "ordinary_cv": re.compile(r"\b(StratifiedKFold|KFold|cross_val_score|cross_validate)\s*\("),
    "grouped_cv": re.compile(
        r"\b(StratifiedGroupKFold|GroupKFold|GroupShuffleSplit|LeaveOneGroupOut)\s*\("
    ),
    "global_scaling": re.compile(
        r"(StandardScaler|MinMaxScaler|RobustScaler)\s*\([^\n]*\)\.fit_transform|"
        r"fit_transform\s*\(\s*X"
    ),
    "global_feature_selection": re.compile(
        r"(SelectKBest|RFE|PCA)\s*\([^\n]*\)\.fit_transform|"
        r"selector\.fit_transform\s*\(\s*X"
    ),
    "resampling": re.compile(r"\b(RandomOverSampler|SMOTE|ADASYN|RandomUnderSampler)\s*\("),
    "pipeline": re.compile(r"\b(Pipeline|ImbPipeline|make_pipeline)\s*\("),
    "hyperparameter_search": re.compile(r"\b(GridSearchCV|RandomizedSearchCV|BayesSearchCV)\s*\("),
    "model_persistence": re.compile(
        r"\b(joblib\.dump|pickle\.dump|torch\.save|save_model)\s*\("
    ),
    "absolute_path": re.compile(r"(?i)([A-Z]:\\\\|/Users/|/home/)"),
    "participant_linked_record": re.compile(r"(?i)(Id\d{4}|Subject[_ -]?\d{4})"),
    "exact_date": re.compile(r"\b(?:20\d{2}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]20\d{2})\b"),
    "credential_marker": re.compile(
        r"(?i)(api[_-]?key\s*=|secret[_-]?key\s*=|password\s*=|BEGIN [A-Z ]*PRIVATE KEY)"
    ),
    "debugger": re.compile(r"\b(pdb\.set_trace|breakpoint)\s*\("),
}


def notebook_text(path: Path) -> tuple[str, str]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    outputs = json.dumps(
        [cell.get("outputs", []) for cell in notebook.get("cells", [])], ensure_ascii=False
    )
    return code, outputs


def disposition(relative: Path, flags: list[str], syntax_status: str) -> str:
    if relative.parts and relative.parts[0] == ".codex-review":
        return "excluded_internal_tooling"
    if syntax_status != "PASS":
        return "excluded_syntax_error"
    if "participant_linked_record" in flags or "absolute_path" in flags:
        return "excluded_sensitive_or_machine_specific"
    if "credential_marker" in flags:
        return "excluded_possible_credential"
    if "debugger" in flags:
        return "excluded_debugger_or_incomplete"
    replacement_flags = {
        "seed_search",
        "sample_level_split",
        "global_scaling",
        "global_feature_selection",
        "resampling",
        "ordinary_cv",
    }
    if replacement_flags.intersection(flags):
        return "replaced_by_canonical_grouped_pipeline"
    if relative.suffix.lower() == ".json":
        return "excluded_data_or_run_configuration"
    if {"grouped_cv", "pipeline", "hyperparameter_search", "model_persistence"}.intersection(flags):
        return "reviewed_superseded_modeling"
    return "reviewed_reference_only"


def scan(root: Path, excluded_roots: set[str], redact_paths: bool = False) -> list[dict]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".py", ".ipynb", ".json"}:
            continue
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in excluded_roots:
            continue

        syntax_status = "PASS"
        output_text = ""
        try:
            if path.suffix.lower() == ".ipynb":
                text, output_text = notebook_text(path)
            else:
                text = path.read_text(encoding="utf-8")
            if path.suffix.lower() in {".py", ".ipynb"}:
                ast.parse(text)
            else:
                json.loads(text)
        except (SyntaxError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            text = path.read_text(encoding="utf-8", errors="replace")
            syntax_status = f"FAIL: {type(exc).__name__}"

        combined = text + "\n" + output_text
        flags = sorted(name for name, pattern in PATTERNS.items() if pattern.search(combined))
        display_path = relative.as_posix()
        if redact_paths:
            digest = hashlib.sha256(display_path.encode("utf-8")).hexdigest()[:12]
            display_path = f"legacy_file_{digest}{path.suffix.lower()}"
        rows.append(
            {
                "path": display_path,
                "type": path.suffix.lower().lstrip("."),
                "code_lines": len(text.splitlines()),
                "syntax_or_json": syntax_status,
                "flags": ";".join(flags),
                "disposition": disposition(relative, flags, syntax_status),
            }
        )
    return rows


def write_report(rows: list[dict], csv_path: Path, markdown_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    type_counts = Counter(row["type"] for row in rows)
    disposition_counts = Counter(row["disposition"] for row in rows)
    flag_counts = Counter(
        flag for row in rows for flag in row["flags"].split(";") if flag
    )
    failures = [row for row in rows if row["syntax_or_json"] != "PASS"]
    lines = [
        "# Legacy code inventory",
        "",
        "## Material Passport",
        "",
        "- Material ID: `nanoheldi-legacy-code-audit-2026-09-09`",
        "- Type: reproducibility and privacy code audit",
        "- Verification status: `ANALYZED`",
        "- Source scope: private working-directory code; file contents and private filenames are not reproduced",
        "- Output: aggregate inventory and replacement disposition",
        "",
        "This report was generated by a read-only heuristic scanner. Flags identify code",
        "patterns requiring review; they are not proof that every execution was biased.",
        "No legacy file was modified, deleted, or copied into the release package.",
        "",
        f"- Files inspected: {len(rows)}",
        "- Types: " + ", ".join(f"{key}={value}" for key, value in sorted(type_counts.items())),
        "- Syntax/JSON failures: " + str(len(failures)),
        "",
        "## Pattern counts",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(flag_counts.items()))
    lines.extend(["", "## Disposition counts", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in sorted(disposition_counts.items()))
    if failures:
        lines.extend(["", "## Parse failures", ""])
        lines.extend(
            f"- `{row['path']}`: {row['syntax_or_json']}" for row in failures
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Seed-search scripts, sample-level splits, preprocessing outside folds, and",
            "pre-CV resampling were excluded from the release implementation. Historical",
            "visualization and utility scripts were retained only in the inventory. The",
            "canonical implementation under `src/nanoheldi_ml/` replaces the affected ML",
            "workflows with deterministic participant-grouped validation and fold-local",
            "preprocessing.",
            "",
            "See `legacy_code_inventory.csv` for the file-by-file record.",
        ]
    )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument(
        "--redact-paths",
        action="store_true",
        help="Replace private relative filenames with stable hash-based file identifiers",
    )
    args = parser.parse_args()
    rows = scan(args.root.resolve(), set(args.exclude), redact_paths=args.redact_paths)
    if not rows:
        raise RuntimeError("No supported files were found")
    write_report(rows, args.csv.resolve(), args.markdown.resolve())
    print(f"Inspected {len(rows)} files; wrote {args.csv} and {args.markdown}")


if __name__ == "__main__":
    main()
