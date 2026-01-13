FROM 172.16.11.78:5000/pytorch/pytorch-houhaowent:23.07-py3

# 设置环境变量，确保 Python 输出直接打印到控制台，不会被缓存
ENV PYTHONUNBUFFERED 1

# 使用 pip 安装指定的 Python 包
RUN pip3 install --upgrade --no-cache-dir --progress-bar off pip -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir --progress-bar off pytorch-lightning==1.9.5 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir --progress-bar off deepspeed==0.12.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir --progress-bar off wandb ninja -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir --progress-bar off transformers -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir --progress-bar off datasets -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install --no-cache-dir --progress-bar off timm -i https://pypi.tuna.tsinghua.edu.cn/simple