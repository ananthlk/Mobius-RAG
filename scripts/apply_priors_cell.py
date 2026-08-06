#!/usr/bin/env python3
"""Local-file-only apply for a Priors Lab computed cell (Ananth's directive,
2026-08-05, Phase 1). Reads eval/priors_bootstrap.yaml, patches ONE
depth_N/strategy cell in place (preserving every comment), writes it back
locally, and prints the applied fields + a sha256 provenance hash. This is
NOT a live Cloud Run write -- see priors_lab.py's module docstring for why.
After running this, the change still needs `git diff` review + commit +
`./deploy/deploy_cloudrun_dev.sh` to actually reach production, same as
every other priors_bootstrap.yaml edit this session.

Usage:
  python3 scripts/apply_priors_cell.py --depth 3 --strategy a \\
    --recall_lift 0.6489 --accuracy_estimate 0.4236 --authority 0.9543 \\
    --n 65 --k0 9 [--dry-run]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.services.retriever.priors_lab import apply_cell_to_yaml_text, cell_sha256, PriorsApplyError  # noqa: E402

PRIORS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "priors_bootstrap.yaml")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, required=True, choices=range(5))
    p.add_argument("--strategy", required=True, choices=["a", "b", "c", "d", "s"])
    p.add_argument("--recall_lift", type=float)
    p.add_argument("--accuracy_estimate", type=float)
    p.add_argument("--authority", type=float)
    p.add_argument("--n", type=int)
    p.add_argument("--k0", type=int)
    p.add_argument("--include-latency", action="store_true")
    p.add_argument("--latency_p50_ms", type=int)
    p.add_argument("--dry-run", action="store_true", help="show the diff, don't write the file")
    args = p.parse_args()

    cell = {
        "recall_lift": args.recall_lift, "accuracy_estimate": args.accuracy_estimate,
        "authority": args.authority, "n": args.n, "k0": args.k0,
        "latency_p50_ms": args.latency_p50_ms,
    }

    with open(PRIORS_PATH, encoding="utf-8") as f:
        old_text = f.read()

    try:
        new_text, applied = apply_cell_to_yaml_text(
            old_text, args.depth, args.strategy, cell, include_latency=args.include_latency,
        )
    except PriorsApplyError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)

    sha = cell_sha256(args.depth, args.strategy, applied)
    print(f"depth_{args.depth}/{args.strategy}: applied {applied}")
    print(f"sha256: {sha}")

    if new_text == old_text:
        print("no textual change (values already match what's in the file)")
        return

    if args.dry_run:
        import difflib
        diff = difflib.unified_diff(
            old_text.splitlines(keepends=True), new_text.splitlines(keepends=True),
            fromfile="eval/priors_bootstrap.yaml (before)", tofile="eval/priors_bootstrap.yaml (after)",
        )
        sys.stdout.writelines(diff)
        print("\n[dry run -- nothing written]")
        return

    with open(PRIORS_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"written to {PRIORS_PATH}")
    print("NEXT: git diff eval/priors_bootstrap.yaml to review, then commit + deploy to make this live.")


if __name__ == "__main__":
    main()
