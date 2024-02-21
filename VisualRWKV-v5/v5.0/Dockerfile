FROM nvcr.io/nvidia/pytorch:23.05-py3

# 设置环境变量，确保 Python 输出直接打印到控制台，不会被缓存
ENV PYTHONUNBUFFERED 1

# 使用 pip 安装指定的 Python 包
RUN pip3 install --no-cache-dir pytorch-lightning==1.9.5
RUN pip3 install --no-cache-dir deepspeed
RUN pip3 install --no-cache-dir wandb ninja
RUN pip3 install --no-cache-dir transformers
RUN pip3 install --no-cache-dir datasets