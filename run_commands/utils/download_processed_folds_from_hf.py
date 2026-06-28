#!/usr/bin/env python3
"""Download preprocessed EyeBench folds from a Hugging Face dataset repo.

This helper intentionally bypasses the raw EyeBench preprocessing pipeline. It
only materializes data/<dataset>/{processed,folds,folds_metadata}, which is the
minimum data layout needed by the existing EyeBench train/test code.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

DEFAULT_REPO_ID = os.environ.get("HF_DATASET_REPO", "skboy/eyebench-processed-folds")
KNOWN_DATASETS = (
    "CopCo",
    "IITBHGC",
    "MECOL2",
    "MECOL2W1",
    "MECOL2W2",
    "OneStop",
    "PoTeC",
    "SBSAT",
)
REQUIRED_PROCESSED_FILES = ("ia.feather", "fixations.feather", "trial_level.feather")


def _split_dataset_args(values: Iterable[str]) -> list[str]:
    datasets: list[str] = []
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                datasets.append(item)
    return datasets


def normalize_datasets(values: list[str]) -> list[str]:
    requested = _split_dataset_args(values or ["PoTeC"])
    if any(item.lower() in {"all", "*"} for item in requested):
        return list(KNOWN_DATASETS)

    by_lower = {dataset.lower(): dataset for dataset in KNOWN_DATASETS}
    normalized: list[str] = []
    unknown: list[str] = []
    for item in requested:
        dataset = by_lower.get(item.lower())
        if dataset is None:
            unknown.append(item)
            continue
        if dataset not in normalized:
            normalized.append(dataset)

    if unknown:
        valid = ", ".join((*KNOWN_DATASETS, "all"))
        raise SystemExit(f"Unknown dataset(s): {', '.join(unknown)}. Valid values: {valid}")
    return normalized


def build_allow_patterns(datasets: list[str], include_manifests: bool) -> list[str]:
    patterns: list[str] = []
    if include_manifests:
        patterns.extend(["README.md", "manifest_sizes.txt", "manifest_files.txt"])

    for dataset in datasets:
        patterns.extend(
            [
                f"data/{dataset}/processed/*",
                f"data/{dataset}/processed/**/*",
                f"data/{dataset}/folds/fold_*/*",
                f"data/{dataset}/folds/**/*",
                f"data/{dataset}/folds_metadata/*/*",
                f"data/{dataset}/folds_metadata/**/*",
            ]
        )
    return patterns


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file() and item.name != ".DS_Store")


def verify_dataset(local_dir: Path, dataset: str, expected_folds: int) -> bool:
    dataset_root = local_dir / "data" / dataset
    processed_dir = dataset_root / "processed"
    folds_dir = dataset_root / "folds"
    metadata_dir = dataset_root / "folds_metadata"

    ok = True
    for required_dir in (processed_dir, folds_dir, metadata_dir):
        if not required_dir.is_dir():
            print(f"[missing] {required_dir}", file=sys.stderr)
            ok = False

    for filename in REQUIRED_PROCESSED_FILES:
        path = processed_dir / filename
        if not path.is_file():
            print(f"[missing] {path}", file=sys.stderr)
            ok = False

    fold_dirs = sorted(path for path in folds_dir.glob("fold_*") if path.is_dir())
    if len(fold_dirs) < expected_folds:
        print(
            f"[missing] {dataset}: expected at least {expected_folds} fold dirs under {folds_dir}, found {len(fold_dirs)}",
            file=sys.stderr,
        )
        ok = False

    fold_file_count = count_files(folds_dir)
    processed_file_count = count_files(processed_dir)
    metadata_file_count = count_files(metadata_dir)
    if fold_file_count == 0:
        print(f"[missing] {folds_dir} contains no files", file=sys.stderr)
        ok = False
    if processed_file_count == 0:
        print(f"[missing] {processed_dir} contains no files", file=sys.stderr)
        ok = False
    if metadata_file_count == 0:
        print(f"[missing] {metadata_dir} contains no files", file=sys.stderr)
        ok = False

    print(
        f"{dataset}: processed={processed_file_count} files, folds={fold_file_count} files, "
        f"fold_dirs={len(fold_dirs)}, folds_metadata={metadata_file_count} files"
    )
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download EyeBench processed folds from a Hugging Face dataset repo "
            "without running raw data preprocessing."
        )
    )
    parser.add_argument(
        "repo_id",
        nargs="?",
        default=DEFAULT_REPO_ID,
        help="HF dataset repo id. Defaults to HF_DATASET_REPO or skboy/eyebench-processed-folds.",
    )
    parser.add_argument(
        "--local-dir",
        default=".",
        help="EyeBench repository root to download into. Defaults to the current directory.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["PoTeC"],
        help="Dataset names to download. Use comma-separated values or 'all'. Default: PoTeC.",
    )
    parser.add_argument(
        "--expected-folds",
        type=int,
        default=4,
        help="Minimum number of fold_* directories required during verification. Default: 4.",
    )
    parser.add_argument(
        "--no-manifests",
        action="store_true",
        help="Do not download README.md / manifest files from the HF dataset repo.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only use locally cached HF files. Useful for offline verification.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip post-download layout verification.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the allow patterns and exit without downloading.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    datasets = normalize_datasets(args.datasets)
    local_dir = Path(args.local_dir).expanduser().resolve()
    allow_patterns = build_allow_patterns(datasets, include_manifests=not args.no_manifests)

    print(f"HF dataset repo: {args.repo_id}")
    print(f"Local EyeBench root: {local_dir}")
    print(f"Datasets: {', '.join(datasets)}")
    print("Allow patterns:")
    for pattern in allow_patterns:
        print(f"  {pattern}")

    if args.dry_run:
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is not installed. Install it with: "
            "pip install -U 'huggingface_hub[hf_xet]<1.0,>=0.24.0'"
        ) from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
        local_files_only=args.local_files_only,
    )

    if args.skip_verify:
        return 0

    all_ok = True
    for dataset in datasets:
        all_ok = verify_dataset(local_dir, dataset, args.expected_folds) and all_ok

    if not all_ok:
        print("Download finished, but the processed-fold layout is incomplete.", file=sys.stderr)
        return 2

    print("Processed EyeBench data is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
