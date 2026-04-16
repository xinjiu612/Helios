#!/usr/bin/env python3
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch


DEFAULT_SRC = Path("/world_model_data/SpatialVID/latents_short_straight")
DEFAULT_DST = Path("/world_model_data/SpatialVID/latents_short_turn")


def can_load_pt(path: Path) -> tuple[bool, str]:
    try:
        torch.load(path, map_location="cpu", weights_only=False)
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)


def choose_default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(32, cpu_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check files in latents_short_turn against same-named files in "
            "latents_short_straight and repair bad target files by copying from straight."
        )
    )
    parser.add_argument("--src", type=Path, default=DEFAULT_SRC, help="Source directory.")
    parser.add_argument("--dst", type=Path, default=DEFAULT_DST, help="Target directory to repair.")
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help="Filename glob to compare. Defaults to '*.pt'.",
    )
    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="Skip torch.load validation and only use file existence plus size checks.",
    )
    parser.add_argument(
        "--repair-missing",
        action="store_true",
        help="Also copy files that exist in source but are missing in target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without copying files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only inspect the first N target files. 0 means no limit.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=choose_default_workers(),
        help="Number of worker processes used for torch.load checks.",
    )
    return parser.parse_args()


def inspect_pair(item: tuple[str, str, bool]) -> dict:
    dst_raw, src_raw, skip_load_check = item
    dst_path = Path(dst_raw)
    src_path = Path(src_raw)

    result = {
        "dst": dst_raw,
        "src": src_raw,
        "missing_in_src": False,
        "bad_src": False,
        "bad_src_err": "",
        "reason": None,
    }

    if not src_path.exists():
        result["missing_in_src"] = True
        return result

    src_size = src_path.stat().st_size
    dst_size = dst_path.stat().st_size
    if src_size != dst_size:
        result["reason"] = f"size_mismatch:{dst_size}->{src_size}"
    elif not skip_load_check:
        dst_ok, dst_err = can_load_pt(dst_path)
        if not dst_ok:
            result["reason"] = f"unreadable:{dst_err}"

    if result["reason"] is not None and not skip_load_check:
        src_ok, src_err = can_load_pt(src_path)
        if not src_ok:
            result["bad_src"] = True
            result["bad_src_err"] = src_err
            result["reason"] = None

    return result


def inspect_missing_source(item: tuple[str, bool]) -> dict:
    src_raw, skip_load_check = item
    src_path = Path(src_raw)
    result = {
        "src": src_raw,
        "bad_src": False,
        "bad_src_err": "",
    }
    if not skip_load_check:
        src_ok, src_err = can_load_pt(src_path)
        if not src_ok:
            result["bad_src"] = True
            result["bad_src_err"] = src_err
    return result


def main() -> int:
    args = parse_args()
    src_dir = args.src
    dst_dir = args.dst

    if not src_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {src_dir}")
    if not dst_dir.is_dir():
        raise SystemExit(f"Target directory does not exist: {dst_dir}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")

    dst_files = sorted(dst_dir.glob(args.pattern))
    if args.limit > 0:
        dst_files = dst_files[: args.limit]

    total = len(dst_files)
    copied = 0
    skipped = 0
    missing_in_src = 0
    bad_src = 0
    reasons: dict[str, int] = {}

    print(f"[info] src={src_dir}")
    print(f"[info] dst={dst_dir}")
    print(f"[info] target_files_to_check={total}")
    print(f"[info] workers={args.workers}")

    items = [(str(dst_path), str(src_dir / dst_path.name), args.skip_load_check) for dst_path in dst_files]
    chunksize = max(1, len(items) // max(1, args.workers * 4)) if items else 1

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(inspect_pair, items, chunksize=chunksize), start=1):
            dst_path = Path(result["dst"])
            src_path = Path(result["src"])
            reason = result["reason"]

            if result["missing_in_src"]:
                missing_in_src += 1
                print(f"[warn] missing source counterpart: {src_path}")
                continue

            if result["bad_src"]:
                bad_src += 1
                print(f"[warn] bad source, skip copy: {src_path} :: {result['bad_src_err']}")
                skipped += 1
            elif reason is None:
                skipped += 1
            else:
                reason_key = reason.split(":", 1)[0]
                reasons[reason_key] = reasons.get(reason_key, 0) + 1
                print(f"[repair] {dst_path} <- {src_path} :: {reason}")
                if not args.dry_run:
                    shutil.copy2(src_path, dst_path)
                copied += 1

            if index % 500 == 0 or index == total:
                print(
                    f"[progress] checked={index}/{total} copied={copied} "
                    f"skipped={skipped} missing_in_src={missing_in_src} bad_src={bad_src}"
                )

        if args.repair_missing:
            dst_names = {p.name for p in dst_dir.glob(args.pattern)}
            src_only = sorted(p for p in src_dir.glob(args.pattern) if p.name not in dst_names)
            print(f"[info] source_only_files={len(src_only)}")
            if args.limit > 0:
                src_only = src_only[: args.limit]
            missing_items = [(str(src_path), args.skip_load_check) for src_path in src_only]
            missing_chunksize = max(1, len(missing_items) // max(1, args.workers * 4)) if missing_items else 1
            for result in executor.map(inspect_missing_source, missing_items, chunksize=missing_chunksize):
                src_path = Path(result["src"])
                if result["bad_src"]:
                    bad_src += 1
                    print(f"[warn] bad source, skip missing repair: {src_path} :: {result['bad_src_err']}")
                    continue
                dst_path = dst_dir / src_path.name
                print(f"[repair-missing] {dst_path} <- {src_path}")
                if not args.dry_run:
                    shutil.copy2(src_path, dst_path)
                copied += 1
                reasons["missing"] = reasons.get("missing", 0) + 1

    print("[summary]")
    print(f"checked={total}")
    print(f"copied={copied}")
    print(f"skipped={skipped}")
    print(f"bad_src={bad_src}")
    print(f"missing_in_src={missing_in_src}")
    for key in sorted(reasons):
        print(f"{key}={reasons[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
