from huggingface_hub import snapshot_download, hf_hub_download

snapshot_download(repo_id="lmms-lab/LLaVA-Video-178K", 
                  repo_type="dataset",
                  local_dir="D:\huggingface_datasets\LLaVA-Video-178K",
                  allow_patterns=["30_60_s_youtube_v0_1/*"],
                  #allow_patterns=["*.zip", "*.json", "*.z01", "*.tar.gz"],
                  ignore_patterns=["*.h5", "*.ot", "*.msgpack"],
                  local_dir_use_symlinks=False)