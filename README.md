# PaddleDubbing

#### 介绍

基于 streamlit 搭建的可视化界面，模型能力来自 [paddlespeech](https://github.com/PaddlePaddle/PaddleSpeech)；
可以灵活配置、调用模型；可以单句合成也可以批量合成。支持中文和英文以及中英文混合。

- 界面
![](pic/gui.png)

- 视频vlog

[【前期准备工作】](https://www.bilibili.com/video/BV1134y117Jr/)

[【v1.0版本】](https://www.bilibili.com/video/BV1dq4y147Gn/)

[【v2.0版本】](https://www.bilibili.com/video/BV1zq4y1x71Y/)



- 基础环境搭建

```
# c++编译环境
apt-get update
sudo apt install build-essential

# python-dev 出现 'python.h' 找不到的错误时安装这个，版本与系统 python 的版本对应
# sudo apt-get install python3.8-dev

# python 环境 version>=3.6 
# python -m pip install --upgrade pip
```

- 虚拟环境

```commandline
virtualenv -p python venv
source venv/bin/activate

sh setup.sh gpu
```


## 早期版本

```
见 master 分支
```


## v3.0

#### 优化

- 支持最新版 streamlit 和 paddlespeech, 版本详见 requirements.txt；
- 新增 tag 设置，用于指定 模型版本， 新增 Frontend 模型选择；
- 支持 中英文混合 模型, spk_id 建议用174/175；
- 支持本地 fintune 生成模型的加载使用；
- 增加支持模型的列表及下载链接到 `说明` 栏；考虑到网速影响，为了避免长时间等待，修改模型检测逻辑，未下载的模型需要手动下载到指定位置，不再自动下载。模型文件结构参见 `说明` 栏

#### 使用

- 启动

```
streamlit run start.py
```

- fintune 模型使用

```
参照 pretrain_model 下载的模型文件夹形式，将 fintune 得到的模型文件 添加到本地 `models/` 路劲下，重新进入 `配置->应用->语音合成`，选择声学模型即可看到 fintune 模型；
注意模型的文件夹名称 应该参照 fastspeech2_aishell3-zh -> fastspeech2_aishell3-*-*-zh 的形式，- 与 _ 两个符号的使用不能乱用。
```

#### 待做

- v3.0 视频 vlog;
- voice clone / voice conversion


#### 参考资料

- [streamlit api](https://docs.streamlit.io/library/api-reference#id1)
- [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
