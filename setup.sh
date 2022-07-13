#! /bin/bash

device=$1

pip install pytest-runner -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ $device = 'gpu' ];then
  pip install paddlepaddle-gpu -i https://mirror.baidu.com/pypi/simple
else
  pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
fi

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

cd

if [ ! -d "nltk_data/" ]; then
  wget https://paddlespeech.bj.bcebos.com/Parakeet/tools/nltk_data.tar.gz
  tar -xzvf nltk_data.tar.gz
fi
