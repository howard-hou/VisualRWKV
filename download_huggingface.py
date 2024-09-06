from huggingface_hub import snapshot_download

snapshot_download(repo_id="lmms-lab/M4-Instruct-Data", 
                  repo_type="dataset",
                  local_dir="D:\huggingface_datasets\lmms-lab/M4-Instruct-Data",
                  allow_patterns=["*.zip", "*.json", "*.z01"],
                  ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
                  local_dir_use_symlinks=False)