from huggingface_hub import snapshot_download

snapshot_download(repo_id="ShareGPTVideo/train_video_and_instruction", 
                  repo_type="dataset",
                  local_dir="D:\huggingface_datasets\ShareGPTVideo",
                  allow_patterns=["*.zip", "*.json", "*.z01", "*.tar.gz"],
                  ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
                  local_dir_use_symlinks=False)