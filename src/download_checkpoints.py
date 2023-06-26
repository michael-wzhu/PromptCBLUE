from huggingface_hub import snapshot_download

repo_id = "THUDM/visualglm-6b"
downloaded = snapshot_download(
    repo_id,
    cache_dir="./",
)