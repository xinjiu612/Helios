import argparse
import json
import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from collections import deque

import torch


def iter_pt_files(root_dir, recursive=False):
    if recursive:
        for current_root, _, file_names in os.walk(root_dir):
            for file_name in file_names:
                if file_name.endswith(".pt"):
                    yield os.path.join(current_root, file_name)
        return

    with os.scandir(root_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.endswith(".pt"):
                yield entry.path


def summarize_tensor(tensor, extreme_threshold):
    if not torch.is_tensor(tensor):
        return []

    anomalies = []
    is_float_like = torch.is_floating_point(tensor) or torch.is_complex(tensor)

    if is_float_like:
        finite_mask = torch.isfinite(tensor)
        if not finite_mask.all():
            anomalies.append(
                {
                    "type": "non_finite",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "nan_count": int(torch.isnan(tensor).sum().item()),
                    "inf_count": int(torch.isinf(tensor).sum().item()),
                }
            )

        safe_tensor = torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)
        absmax = float(safe_tensor.abs().max().item())
        if absmax > extreme_threshold:
            anomalies.append(
                {
                    "type": "extreme_absmax",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "absmax": absmax,
                    "min": float(safe_tensor.min().item()),
                    "max": float(safe_tensor.max().item()),
                    "threshold": extreme_threshold,
                }
            )
    else:
        absmax = float(tensor.abs().max().item()) if tensor.numel() > 0 else 0.0
        if absmax > extreme_threshold:
            anomalies.append(
                {
                    "type": "extreme_absmax",
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "absmax": absmax,
                    "threshold": extreme_threshold,
                }
            )

    return anomalies


def check_pt_file(file_path, extreme_threshold):
    try:
        payload = torch.load(file_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {
            "file_path": file_path,
            "status": "load_error",
            "issues": [{"type": "load_error", "message": str(exc)}],
        }

    issues = []
    if not isinstance(payload, dict):
        return {
            "file_path": file_path,
            "status": "bad_payload",
            "issues": [{"type": "bad_payload", "message": f"Expected dict, got {type(payload).__name__}"}],
        }

    for key, value in payload.items():
        tensor_issues = summarize_tensor(value, extreme_threshold)
        for tensor_issue in tensor_issues:
            tensor_issue["key"] = key
            issues.append(tensor_issue)

    return {
        "file_path": file_path,
        "status": "ok" if not issues else "anomaly",
        "issues": issues,
    }


def drain_futures(pending_futures, jsonl_handle, report_limit):
    done, not_done = wait(pending_futures, return_when=FIRST_COMPLETED)
    anomalies = []
    completed_paths = []

    for future in done:
        result = future.result()
        completed_paths.append(result["file_path"])
        if result["status"] != "ok":
            anomalies.append(result)
            if jsonl_handle is not None:
                jsonl_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                jsonl_handle.flush()
            if report_limit <= 0 or len(anomalies) <= report_limit:
                print(json.dumps(result, ensure_ascii=False))

    return list(not_done), anomalies, len(done), completed_paths


def main():
    parser = argparse.ArgumentParser(description="Scan latent .pt files and report anomalous tensors.")
    parser.add_argument(
        "--root",
        type=str,
        default="/world_model_data/SpatialVID/5fps/latents_short_straight",
        help="Root directory containing latent .pt files",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subdirectories")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent file readers")
    parser.add_argument("--max_files", type=int, default=None, help="Optional cap on number of files to scan")
    parser.add_argument(
        "--extreme_threshold",
        type=float,
        default=1e6,
        help="Flag tensors whose absmax exceeds this threshold even if they are finite",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default=None,
        help="Optional path to save anomalies as JSONL",
    )
    parser.add_argument(
        "--report_limit",
        type=int,
        default=100,
        help="Maximum number of anomaly records to print to stdout; <=0 means unlimited",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=1000,
        help="Print progress every N completed files",
    )
    parser.add_argument(
        "--progress_show_files",
        type=int,
        default=3,
        help="How many recently scanned file paths to print at each progress update; <=0 disables path logging",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.root):
        raise FileNotFoundError(f"Root directory does not exist: {args.root}")

    file_paths = []
    for file_path in iter_pt_files(args.root, recursive=args.recursive):
        file_paths.append(file_path)
        if args.max_files is not None and len(file_paths) >= args.max_files:
            break

    print(f"Scanning {len(file_paths)} .pt files under {args.root}")

    total_scanned = 0
    anomaly_results = []
    pending = []
    recent_scanned_paths = deque(maxlen=max(args.progress_show_files, 0) or None)
    next_progress_mark = args.progress_every if args.progress_every > 0 else None
    jsonl_handle = open(args.output_jsonl, "w", encoding="utf-8") if args.output_jsonl else None

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            for file_path in file_paths:
                pending.append(executor.submit(check_pt_file, file_path, args.extreme_threshold))
                if len(pending) >= max(args.workers * 4, 1):
                    pending, new_anomalies, completed_count, completed_paths = drain_futures(
                        pending, jsonl_handle, args.report_limit
                    )
                    total_scanned += completed_count
                    anomaly_results.extend(new_anomalies)
                    if args.progress_show_files > 0:
                        recent_scanned_paths.extend(completed_paths)
                    while next_progress_mark is not None and total_scanned >= next_progress_mark:
                        print(
                            f"Progress: scanned {total_scanned}/{len(file_paths)} files, "
                            f"anomalies={len(anomaly_results)}"
                        )
                        if args.progress_show_files > 0 and recent_scanned_paths:
                            print("  Recent scanned files:")
                            for recent_path in list(recent_scanned_paths)[-args.progress_show_files :]:
                                print(f"    {recent_path}")
                        next_progress_mark += args.progress_every

            while pending:
                pending, new_anomalies, completed_count, completed_paths = drain_futures(
                    pending, jsonl_handle, args.report_limit
                )
                total_scanned += completed_count
                anomaly_results.extend(new_anomalies)
                if args.progress_show_files > 0:
                    recent_scanned_paths.extend(completed_paths)
                while next_progress_mark is not None and total_scanned >= next_progress_mark:
                    print(f"Progress: scanned {total_scanned}/{len(file_paths)} files, anomalies={len(anomaly_results)}")
                    if args.progress_show_files > 0 and recent_scanned_paths:
                        print("  Recent scanned files:")
                        for recent_path in list(recent_scanned_paths)[-args.progress_show_files :]:
                            print(f"    {recent_path}")
                    next_progress_mark += args.progress_every
    finally:
        if jsonl_handle is not None:
            jsonl_handle.close()

    anomaly_count = len(anomaly_results)
    load_error_count = sum(1 for item in anomaly_results if item["status"] == "load_error")
    non_finite_count = sum(
        1 for item in anomaly_results for issue in item["issues"] if issue["type"] == "non_finite"
    )
    extreme_count = sum(
        1 for item in anomaly_results for issue in item["issues"] if issue["type"] == "extreme_absmax"
    )

    print("\nScan complete")
    print(f"  scanned_files: {len(file_paths)}")
    print(f"  anomaly_files: {anomaly_count}")
    print(f"  load_errors: {load_error_count}")
    print(f"  non_finite_issues: {non_finite_count}")
    print(f"  extreme_absmax_issues: {extreme_count}")
    if args.output_jsonl:
        print(f"  anomaly_report: {args.output_jsonl}")


if __name__ == "__main__":
    main()
