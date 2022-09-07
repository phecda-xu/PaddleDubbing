import os
import struct
from typing import Dict
from paddlespeech.cli import download
from paddlespeech.cli.utils import _get_uncompress_path

def genHeadInfo(sampleRate, bits, sampleNum, channels):
  '''
  # TODO 生成头信息
  :param sampleRate:  采样率，int
  :param bits:        每个采样点的位数，int
  :param sampleNum:   音频字节数，int
  :return:            音频头文件，byte
  '''
  # rHeadInfo = b'\x52\x49\x46\x46'                               # RIFF
  rHeadInfo = b'RIFF'
  fileLength = struct.pack('i', sampleNum + 36)                 # sampleNum = 总采样点数 × 2  = 秒数 × 采样率 × 2
  rHeadInfo += fileLength                                       # 去掉RIFF 部分 剩余的音频文件总长度  32036 -> b'$}\x00\x00'
  # rHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20'              # b'WAVEfmt '
  rHeadInfo += b'WAVEfmt '
  rHeadInfo += struct.pack('i', bits)                           # 每个采样的位数 16 b'\x10\x00\x00\x00'
  rHeadInfo += b'\x01\x00'                                      # 音频数据的编码方式 1
  rHeadInfo += struct.pack('h', channels)                       # 声道数 1 ， short型数据  1->b'\x01\x00',2->b'\x02\x00'
  rHeadInfo += struct.pack('i', sampleRate)                     # 采样率 16000 -> b'\x80>\x00\x00'
  rHeadInfo += struct.pack('i', int(sampleRate * bits / 8))     # 音频传输速率  32000 = 16000*16/8  -> b'\x00}\x00\x00'
  rHeadInfo += b'\x02\x00\x10\x00'                              # [2,16] 作用未知，默认不需改动
  rHeadInfo += b'\x64\x61\x74\x61'                              # data
  rHeadInfo += struct.pack('i', sampleNum)                      # 不包括头文件的音频长度 32000 \x00}\x00\x00
  return rHeadInfo


def decompress(archive: Dict[str, str], path: str) -> os.PathLike:
    """
    Download archieves and decompress to specific path.
    """
    if not os.path.isdir(path):
       os.makedirs(path)

    assert 'url' in archive and 'md5' in archive, \
       'Dictionary keys of "url" and "md5" are required in the archive, but got: {}'.format(list(archive.keys()))

    filepath = os.path.join(path, os.path.basename(archive['url']))
    if os.path.isfile(filepath) and (download._md5check(filepath, archive['md5']) or archive['md5'] == '0'):
       uncompress_path = _get_uncompress_path(filepath)
       if not os.path.isdir(uncompress_path):
          download._decompress(filepath)
    else:
      raise ValueError(f"模型文件错误，请下载模型：{archive['url']}")
    return uncompress_path