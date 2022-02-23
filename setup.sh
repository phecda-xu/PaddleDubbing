#! /bin/bash

# c++编译环境
apt-get update
sudo apt install build-essential

# python-dev 出现python.h找不到的错误时安装这个，版本与系统python的版本对应
# sudo apt-get install python3.8-dev


# python 环境
python -m pip install --upgrade pip


pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
pip install paddlespeech -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
