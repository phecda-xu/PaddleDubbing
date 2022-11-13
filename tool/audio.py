import os
import math
import struct
import numpy as np
import soundfile as sf

from tqdm import tqdm

def genHeadInfo(sampleRate, bits, sampleNum, channels):
  '''
  # TODO 生成头信息
  :param sampleRate:  采样率，int
  :param bits:        每个采样点的位数，int
  :param sampleNum:   音频字节数，int
  :return:            音频头文件，byte
  '''
  rHeadInfo = b'\x52\x49\x46\x46'                               # RIFF
  fileLength = struct.pack('i', sampleNum + 36)                 # sampleNum = 总采样点数 × 2  = 秒数 × 采样率 × 2
  rHeadInfo += fileLength                                       # 去掉RIFF 部分 剩余的音频文件总长度  32036 -> b'$}\x00\x00'
  rHeadInfo += b'\x57\x41\x56\x45\x66\x6D\x74\x20'              # b'WAVEfmt '
  rHeadInfo += struct.pack('i', bits)                           # 每个采样的位数 16 b'\x10\x00\x00\x00'
  rHeadInfo += b'\x01\x00'                                      # 音频数据的编码方式 1
  rHeadInfo += struct.pack('h', channels)                       # 声道数 1 ， short型数据  1->b'\x01\x00',2->b'\x02\x00'
  rHeadInfo += struct.pack('i', sampleRate)                     # 采样率 16000 -> b'\x80>\x00\x00'
  rHeadInfo += struct.pack('i', int(sampleRate * bits / 8))     # 音频传输速率  32000 = 16000*16/8  -> b'\x00}\x00\x00'
  rHeadInfo += b'\x02\x00\x10\x00'                              # [2,16] 作用未知，默认不需改动
  rHeadInfo += b'\x64\x61\x74\x61'                              # data
  rHeadInfo += struct.pack('i', sampleNum)                      # 不包括头文件的音频长度 32000 \x00}\x00\x00
  return rHeadInfo


class Audio(object):
    def __init__(self, sr):
        self.sr = sr

    def read_pcm(self,filepath):
        f = open(filepath, 'rb')
        str_data = f.read()
        signal = np.fromstring(str_data, dtype=np.short) / (2 ** 15 - 1)
        return signal

    def vad(self, input_buffer, threhold):
        '''
        # TODO 静音检测
        :param bite_buffer:   二进制的音频流数据，byte
        :return:              处理后的二进制音频流，byte
        '''
        if type(input_buffer) == 'bytes':
            array_buffer = np.fromstring(input_buffer, dtype=np.short)
        else:
            array_buffer = input_buffer
        t = 400  # 400ms
        frameLength = self.sr * t // 1000  # 帧长
        frames = [array_buffer[i:i + frameLength] for i
                  in range(0, len(array_buffer), frameLength)]
        entropys = []
        dics = {}
        for fid, frame in tqdm(enumerate(frames), desc="vad process "):
            dic = {}
            for d in frame:
                if d in dic.keys():
                    dic[d] = dic[d] + 1
                else:
                    dic[d] = 1
                if d in dics.keys():
                    dics[d] = dics[d] + 1
                else:
                    dics[d] = 1
            ns = np.array([dic[key] for key in dic.keys()])
            ps = ns / len(frame)
            logps = [math.log(p) for p in ps]
            entropy = -sum(np.array(ps) * np.array(logps))
            entropys.append(entropy)
        enthreshold = np.mean(entropys)
        tags1 = np.array(entropys) > enthreshold * threhold
        tags = np.repeat(tags1, frameLength)[0:len(array_buffer)]
        return tags

    def split_2_index(self, signal, threhold):

        tags = self.vad(signal, threhold=threhold)
        point_list = []
        start_state = 0
        end_state = 0
        n = 0
        for i in tqdm(range(len(signal)), desc="find windows "):
            n += 1
            if tags[i] and start_state == 0 and end_state == 0:
                point_list.append(n)
                start_state = 1
            elif not tags[i] and start_state == 1 and end_state == 0:
                end_state = 1
                start_state = 0
                point_list.append(n)
            elif tags[i] and start_state == 0 and end_state == 1:
                start_state = 1
                end_state = 0
                point_list.append(n)
            else:
                continue
        # point_list.append(len(signal))
        return point_list

    def split_2_files(self, filepath, savepath, threhold, extend_size):
        signal, sr = sf.read(filepath)
        shape_size = signal.shape
        channels = shape_size[1]

        sf.write("test.wav", signal, self.sr)
        tags = self.vad(signal, threhold=threhold)
        point_list = []
        start_state = 0
        end_state = 0
        n = 0
        save_list = []
        for i in tqdm(range(len(signal)), desc="find windows "):
            n += 1
            if tags[i] and start_state == 0 and end_state == 0:
                point_list.append(n)
                start_state = 1
            elif not tags[i] and start_state == 1 and end_state == 0:
                end_state = 1
                start_state = 0
                point_list.append(n)
            elif tags[i] and start_state == 0 and end_state == 1:
                start_state = 1
                end_state = 0
                point_list.append(n)
            else:
                continue

        print(point_list)
        filename = os.path.basename(filepath)[:-4]
        m = 1
        if len(point_list) == 0:
            tmp_buffer = signal
            sf.write(f"{savepath}/{filename}_k000.wav", tmp_buffer, self.sr)
            save_list.append(f"{savepath}/{filename}_k000.wav")
        elif len(point_list) == 1:
            tmp_buffer = signal[point_list[0]:]
            sf.write(f"{savepath}/{filename}_k{m:03d}.wav", tmp_buffer, self.sr)
            save_list.append(f"{savepath}/{filename}_k{m:03d}.wav")
        elif len(point_list) > 0 and len(point_list) % 2 == 0:
            for i in range(0, len(point_list) - 1, 2):
                tmp_buffer = signal[point_list[i]: point_list[i + 1]+extend_size]
                sf.write(f"{savepath}/{filename}_k{m:03d}.wav", tmp_buffer, self.sr)
                save_list.append(f"{savepath}/{filename}_k{m:03d}.wav")
                m += 1
        else:
            point_list.append(len(tags))
            for i in range(0, len(point_list) - 1, 2):
                tmp_buffer = signal[point_list[i]: point_list[i + 1]+extend_size]
                sf.write(f"{savepath}/{filename}_k{m:03d}.wav", tmp_buffer, self.sr)
                save_list.append(f"{savepath}/{filename}_k{m:03d}.wav")
                m += 1

        return save_list
