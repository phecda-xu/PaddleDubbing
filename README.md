# TTS-GUI

#### 介绍


#### 参考资料

[用Python实现带GUI 的exe](https://blog.csdn.net/miffy2017may/article/details/103391855)

[使用pyinstaller的docker镜像对python程序进行快速打包](https://blog.csdn.net/qq_33181607/article/details/104995891)

[docker-pyinstaller](https://gitee.com/Zyx-A/docker-pyinstaller/)

#### 工具

##### pyinstaller

- 下载

```
docker pull cdrx/pyinstaller-windows
```

- 使用

```
docker run --env PYPI_URL=http://pypi.doubanio.com/ --env PYPI_INDEX_URL=http://pypi.doubanio.com/simple -v "$(pwd):/src/" cdrx/pyinstaller-windows:latest "pyinstaller -F -w example.py"
```
