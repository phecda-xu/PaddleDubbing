# TTS-GUI

#### 介绍


#### 工具

##### pyinstaller

- 下载

```
docker pull cdrx/pyinstaller-windows
```

- 使用

```
docker run --env PYPI_URL=http://pypi.doubanio.com/ --env PYPI_INDEX_URL=http://pypi.doubanio.com/simple -v "$(pwd):/src/" cdrx/pyinstaller-windows:latest "pyinstaller -F -w care.py"
```
