"""Command-line interface for welllog."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .download import FORCE_ARCHIVE_NAME, download_force_archive
from .io import load_las_directory
from .pipeline import run_pipeline, write_results
from .qc import qc_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="welllog", description="FORCE well-log lithology tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="download and verify the FORCE LAS archive")
    download.add_argument("destination", nargs="?", default=FORCE_ARCHIVE_NAME)
    download.add_argument("--overwrite", action="store_true")

    qc = subparsers.add_parser("qc", help="summarize recursively loaded LAS files")
    qc.add_argument("input")

    train = subparsers.add_parser("train", help="train and evaluate with a grouped well split")
    train.add_argument("input")
    train.add_argument("--output", default="welllog-results")
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--max-iter", type=int, default=150)
    train.add_argument("--penalty-matrix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        path = download_force_archive(args.destination, overwrite=args.overwrite)
        print(path)
        return 0
    if args.command == "qc":
        report = qc_report(load_las_directory(args.input))
        summary = {"rows": report["rows"], "wells": report["wells"]}
        print(json.dumps(summary, sort_keys=True))
        print(report["wells_summary"].to_string())
        return 0
    if args.command == "train":
        result = run_pipeline(
            Path(args.input),
            test_size=args.test_size,
            random_state=args.seed,
            penalty_matrix=args.penalty_matrix,
            max_iter=args.max_iter,
        )
        output = write_results(result, args.output)
        print(json.dumps(result.metrics, sort_keys=True))
        print(output)
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")
