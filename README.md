# PaddleDubbing

#### 介绍

基于 streamlit 搭建的可视化界面，模型能力来自 [paddlespeech](https://github.com/PaddlePaddle/PaddleSpeech)；
可以灵活配置、调用模型；可以单句合成也可以批量合成。支持中文和英文，不支持中英混合。

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


## v1.0

```
见 master 分支
```

## v2.0


```
见 master 分支
```

## v3.0

#### 优化

- 改用 streamlit 就行界面搭建，优化显示和调用逻辑；
- 增加历史记录；

#### 使用

- 启动

```
streamlit run start.py
```

#### 待做

- 增加打包下载功能;
- v3.0 视频 vlog;
- 增加语速、声调等设置;
- 增加远程功能;


#### 参考资料

- [streamlit api](https://docs.streamlit.io/library/api-reference#id1)
- [PaddleSpeech](https://github.com/PaddlePaddle/PaddleSpeech)
