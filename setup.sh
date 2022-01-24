#! /bin/bash

# c++编译环境
apt-get update
sudo apt install build-essential

# python 环境
python -m pip install --upgrade pip


pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple
