# ModelScope Upload Tools

This directory contains scripts for pushing local artifacts to ModelScope.

## Default uploader

`upload_folder.py` is preconfigured for this repository:

- local folder: `ablation_stage_1_post_10000_intergs`
- remote model repo: `xinjiu612/helios`
- remote path in repo: `ablation_stage_1_post_10000_intergs`

## Usage

If you already logged in:

```bash
modelscope login --token <your-modelscope-token>
python tools/modelscope/upload_folder.py
```

Or pass the token directly:

```bash
python tools/modelscope/upload_folder.py --token "ms-96dfb4c5-8614-46f5-9bb0-9640dbe6baff"
```

Common overrides:

```bash
python tools/modelscope/upload_folder.py \
  --folder-path /path/to/local/folder \
  --path-in-repo some/remote/path \
  --repo-id xinjiu612/helios
```
