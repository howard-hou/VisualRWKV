from huggingface_hub import snapshot_download

snapshot_download(repo_id="lmms-lab/M4-Instruct-Data", 
                  repo_type="dataset",
                  local_dir="D:\huggingface_datasets",
                  allow_patterns=["*.zip", "*.json", "*.z01"],
                  ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
                  resume_download=True,
                  local_dir_use_symlinks=False)