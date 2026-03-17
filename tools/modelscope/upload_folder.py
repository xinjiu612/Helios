#!/usr/bin/env python3
"""Upload a local folder into a ModelScope model repository."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO_ID = "xinjiu612/helios"
DEFAULT_FOLDER_NAME = "ablation_stage_1_post_uav_flow"
DEFAULT_FOLDER_PATH = REPO_ROOT / DEFAULT_FOLDER_NAME


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload a local folder to ModelScope. By default this uploads "
            f"{DEFAULT_FOLDER_NAME} into the model repo {DEFAULT_REPO_ID}."
        )
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Target ModelScope model repo, for example owner/repo.",
    )
    parser.add_argument(
        "--folder-path",
        type=Path,
        default=DEFAULT_FOLDER_PATH,
        help="Local folder to upload.",
    )
    parser.add_argument(
        "--path-in-repo",
        default=None,
        help=(
            "Target folder path inside the remote repo. Defaults to the local "
            "folder name so the directory structure is preserved."
        ),
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_API_TOKEN"),
        help=(
            "ModelScope access token. If omitted, the script uses "
            "MODELSCOPE_API_TOKEN or an existing `modelscope login` session."
        ),
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("MODELSCOPE_ENDPOINT"),
        help="Optional ModelScope endpoint. Defaults to the SDK default.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch or tag to upload to. Defaults to the repo default branch.",
    )
    parser.add_argument(
        "--commit-message",
        default=None,
        help="Optional commit message. Defaults to `Upload <path_in_repo>`.",
    )
    parser.add_argument(
        "--commit-description",
        default="Uploaded from the Helios workspace via tools/modelscope/upload_folder.py",
        help="Optional commit description.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Optional concurrent worker count for file uploads.",
    )
    parser.add_argument(
        "--allow-pattern",
        action="append",
        default=[],
        help="Glob pattern to include. Repeat this argument for multiple patterns.",
    )
    parser.add_argument(
        "--ignore-pattern",
        action="append",
        default=[],
        help="Glob pattern to exclude. Repeat this argument for multiple patterns.",
    )
    return parser


def normalize_patterns(patterns: Iterable[str]) -> Optional[List[str]]:
    values = [item for item in patterns if item]
    return values or None


def resolve_folder_path(folder_path: Path) -> Path:
    path = folder_path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Local folder does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Local path is not a directory: {path}")
    return path


def resolve_remote_path(folder_path: Path, path_in_repo: Optional[str]) -> str:
    if path_in_repo is None or not path_in_repo.strip():
        return folder_path.name
    return path_in_repo.strip("/")


def ensure_modelscope_available() -> None:
    try:
        import modelscope  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "ModelScope SDK is not installed. Run `pip install modelscope` first."
        ) from exc


def print_commit_infos(commit_infos) -> None:
    infos = commit_infos if isinstance(commit_infos, list) else [commit_infos]
    print(f"Upload finished with {len(infos)} commit(s).", flush=True)
    for index, info in enumerate(infos, start=1):
        commit_url = getattr(info, "commit_url", "")
        commit_message = getattr(info, "commit_message", "")
        oid = getattr(info, "oid", "")
        print(
            f"[{index}] oid={oid or 'n/a'} message={commit_message or 'n/a'}",
            flush=True,
        )
        if commit_url:
            print(f"    {commit_url}", flush=True)


def main() -> int:
    args = build_parser().parse_args()
    ensure_modelscope_available()

    from modelscope.hub.api import HubApi

    folder_path = resolve_folder_path(args.folder_path)
    path_in_repo = resolve_remote_path(folder_path, args.path_in_repo)
    commit_message = args.commit_message or f"Upload {path_in_repo}"
    token = args.token.strip() if args.token else None

    api_kwargs = {}
    if args.endpoint:
        api_kwargs["endpoint"] = args.endpoint.rstrip("/")
    api = HubApi(**api_kwargs)

    if token:
        print("Logging in to ModelScope with the provided token ...", flush=True)
        api.login(access_token=token)
    else:
        print(
            "No token argument provided. Using cached ModelScope login if available ...",
            flush=True,
        )

    upload_kwargs = {
        "repo_id": args.repo_id,
        "folder_path": str(folder_path),
        "path_in_repo": path_in_repo,
        "commit_message": commit_message,
        "commit_description": args.commit_description,
        "token": token,
        "repo_type": "model",
        "allow_patterns": normalize_patterns(args.allow_pattern),
        "ignore_patterns": normalize_patterns(args.ignore_pattern),
    }
    if args.max_workers is not None:
        upload_kwargs["max_workers"] = args.max_workers
    if args.revision:
        upload_kwargs["revision"] = args.revision

    print(
        f"Uploading `{folder_path}` to `{args.repo_id}` at `{path_in_repo}` ...",
        flush=True,
    )
    commit_infos = api.upload_folder(**upload_kwargs)
    print_commit_infos(commit_infos)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover
        print(f"Upload failed: {exc}", file=sys.stderr, flush=True)
        raise
