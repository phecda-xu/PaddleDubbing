import os
import argparse
import numpy as np
import hashlib

parser = argparse.ArgumentParser(description='数据处理配置')
parser.add_argument('--base_path', type=str, default=os.getcwd(), help='数据存储的基础路劲')
args = parser.parse_args()
os.environ['PPSPEECH_HOME'] = args.base_path

import json
import shutil
import string
import paddle
import librosa
import tkinter as tk
import soundfile as sf
import streamlit as st

from tkinter import filedialog
from src.asr import ASRExecutor
from tool.audio import Audio
from tool.audio import genHeadInfo
from pypinyin import lazy_pinyin, Style
from paddlespeech.resource.pretrained_models import asr_dynamic_pretrained_models as pretrained_models

# "st.session_state object:", st.session_state


@st.cache(allow_output_mutation=True)
def load_asr_model(model_type='conformer_wenetspeech',
                   model_tag='1.0',
                   lang='zh',
                   sample_rate=16000,
                   config=None,  # Set `config` and `ckpt_path` to None to use pretrained model.
                   ckpt_path=None,
                   force_yes=False,
                   decode_method='attention_rescoring',
                   device=paddle.get_device()):

    asr_executor = ASRExecutor(model_type=model_type,
                               model_tag=model_tag,
                               lang=lang,
                               sample_rate=sample_rate,
                               config=config,  # Set `config` and `ckpt_path` to None to use pretrained model.
                               ckpt_path=ckpt_path,
                               force_yes=force_yes,
                               decode_method=decode_method,
                               device=device)
    return asr_executor


def load_audio(filepath, dtype=np.int16):
    signal, sr = sf.read(filepath, dtype=dtype)
    return signal, sr


def get_md5(content):
    return hashlib.md5(content.encode(encoding="utf-8")).hexdigest()[:10]


def collect_log_files():
    base_log_save_path = f"{args.base_path}/log/"
    tagging_list = []
    for root, _, files in os.walk(base_log_save_path):
        if root == base_log_save_path:
            continue
        for name in files:
            if name.endswith(".json"):
                log_file = os.path.join(root, name)
                tagging_list.append(log_file)
    return tagging_list


def print_state():
    st.sidebar.write("-"*5)
    with st.sidebar.expander("session_state"):
        st.write(st.session_state)


def config_page():
    st.set_page_config(
        page_title="数据标注工具",
        page_icon=":shark",
        layout="wide",
        initial_sidebar_state="auto",
    )


class Builder(object):
    def __init__(self, args):
        self.args = args
        self.abcd = list(string.ascii_letters)
        self.abcd.extend([str(i) for i in range(9)])

    def start_doc(self):
        st.header("导语")
        st.text("欢迎使用音频处理工具，本工具基于streamlit搭建。")
        st.text("作者: phecda-xu")
        st.markdown("使用中每次修改请按 `enter` 键进行确认!")
        st.subheader("工具实现基本思路")
        st.markdown("这里描述的是为满足具体需求而构思的工具执行思路")
        st.markdown("- 思路一")
        st.markdown(" 1、**多通道音频 -> 单通道** ")
        st.markdown(" 2、**VAD -> 片段** ")
        st.markdown(" 3、**SE  -> 去掉背景音** ")
        st.markdown(" 4、**标注  -> 文本及拼音** ")
        st.markdown(" 5、**生成标注格式数据集** ")
        st.markdown("- 思路二")
        st.markdown(" 1、**合成声音 -> SE 增强** ")


    @staticmethod
    def get_wav_data(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return data

    @staticmethod
    def save_wav_data(bin_file, buffer, file_label='File'):
        with open(bin_file, 'wb') as f:
            data = f.write(buffer)
        return data

    def array_2_raw(self, signal, sr):
        header = genHeadInfo(sr, 16, len(signal) * 2, 1)
        raw_data = signal.tobytes()
        return header + raw_data

    def get_text_input(self):
        uploader_file = st.file_uploader("上传txt文件：")
        if uploader_file is not None:
            res = uploader_file.getvalue().decode("utf-8")
            text = st.text_area("输入音频文件路劲：",
                                value=res, key=None)
        else:
            text = st.text_area("输入音频文件路劲：(在这里输入多行文本,每行文本字数不限,文本框右下角可以拖动。 )",
                                value='', key=None)
        text_list = text.split("\n")
        return text_list

    def gpu_setting(self):
        if self.gpu_option:
            st.markdown("<font size=3.5>(当前使用 gpu:0)</font>", unsafe_allow_html=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
        else:
            st.markdown("<font size=3.5>(当前使用 cpu)</font> ", unsafe_allow_html=True)
            os.environ["CUDA_VISIBLE_DEVICES"] = ''

    def _init_save_(self):
        base_dir = self.args.base_path
        self.log_path = os.path.join(base_dir, "log")
        self.out_path = os.path.join(base_dir, "output", "tagging")
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)

    def model_setting(self):
        zh_model_list = []
        en_model_list = []
        for name_i in list(pretrained_models.keys()):
            if "zh" in name_i and "online" not in name_i:
                zh_model_list.append(name_i.split("-")[0])
            elif "en" in name_i and "online" not in name_i:
                en_model_list.append(name_i.split("-")[0])
            else:
                continue
        lang_list = ["zh", "en"]
        lang_option = st.sidebar.selectbox("语言选择", lang_list, key='lang_option')
        if lang_option == "zh":
            model_option = st.sidebar.selectbox("中文模型", zh_model_list, key='zh_model_option')
            if model_option != "":
                zh_tag_list = list(pretrained_models[model_option + '-zh-16k'].keys())
                tag_option = st.sidebar.selectbox("tag", zh_tag_list, key='zh_tag_option')
        elif lang_option == "en":
            model_option = st.sidebar.selectbox("英文模型", en_model_list, key='en_model_option')
            if model_option != "":
                en_tag_list = list(pretrained_models[model_option+ '-en-16k'].keys())
                tag_option = st.sidebar.selectbox("tag", en_tag_list, key='en_tag_option')
        else:
            raise ValueError("")

        if model_option:
            with st.spinner("模型加载中..."):
                self.asr_executor = load_asr_model(model_type=model_option,
                                                   lang=lang_option,
                                                   model_tag=tag_option)

    def one_flow_process(self, signal, threhold):
        point_list = self.audio_processer.split_2_index(signal, threhold)
        return point_list

    def assist_module(self):
        assist_tab1, assist_tab2 = st.sidebar.tabs(["基础工具", "算法工具"])

        with assist_tab1:
            self.gpu_option = st.checkbox('加速(GPU)')
            self.vad_option = st.checkbox('切分(VAD)')
            self.convert_option = st.checkbox('格式转换(暂不可用)', disabled=True)
            if self.vad_option:
                self.audio_processer = Audio(self.sr)
            if self.gpu_option:
                self.gpu_setting()
            if self.convert_option:
                pass

        with assist_tab2:
            self.asr_option = st.checkbox('语音识别(ASR)')
            self.se_option = st.checkbox('语音增强(SE)(暂不可用)', disabled=True)
            if self.asr_option:
                self.model_setting()
            if self.se_option:
                pass

    def audio_setting(self):
        audio_tab1, audio_tab2 = st.sidebar.tabs(["生成通道数", "生成采样率"])
        with audio_tab1:
            self.channels = st.radio("目前支持生成以下通道数", [1])

        with audio_tab2:
            self.sr = st.radio("目前支持生成以下采样率", [16000, 24000, 44100, 48000])

    def display_source_audio(self):
        signal,sr = load_audio(self.target_file_path, dtype=np.int16)
        # signal, sr = sf.read(filepath, dtype=np.int16)
        shape_size = signal.shape
        if len(shape_size) > 1:
            self.is_single_channel = False
            channels = shape_size[1]
            st.write("-"*5)
            for i in range(channels):
                channel_col1, channel_col2 = st.columns([1,4])
                with channel_col1:
                    st.write("\n")
                    st.write(f"通道 {i}: ")
                with channel_col2:
                    array_data = signal[:, i]
                    array_raw_data = self.array_2_raw(array_data, sr)
                    st.audio(array_raw_data, format='audio/wav')
        else:
            self.is_single_channel = True
            single_col1, single_col2 = st.columns([1, 4])
            with single_col1:
                st.write("\n")
                st.write(f"单通道 : ")
            with single_col2:
                array_raw_data = self.array_2_raw(signal, sr)
                st.audio(array_raw_data, format='audio/wav')
        return signal, sr

    def show_channel_columns(self):
        c_col_1, c_col_2, c_col_3 = st.columns([2, 2, 2])

        with c_col_1:
            st.write("\n")
            def on_change_1():
                if "channel_split" in st.session_state:
                    st.session_state.channel_split = False
                if "channel_single" in st.session_state:
                    st.session_state.channel_single = False
            channel_merge = st.checkbox("合并通道(取平均)",
                                        key="channel_merge",
                                        on_change=on_change_1)
        with c_col_2:
            st.write("\n")
            def on_change_2():
                if "channel_merge" in st.session_state:
                    st.session_state.channel_merge = False
                if "channel_single" in st.session_state:
                    st.session_state.channel_single = False

            channel_split = st.checkbox("分离通道(拼接)",
                                        key="channel_split",
                                        on_change=on_change_2)
        with c_col_3:
            st.write("\n")
            def on_change_3():
                if "channel_merge" in st.session_state:
                    st.session_state.channel_merge = False
                if "channel_split" in st.session_state:
                    st.session_state.channel_split = False

            channel_single = st.checkbox("强制单通道(取第一个通道)",
                                         key="channel_single",
                                         on_change=on_change_3)
        channel_state = "{}{}{}".format(int(channel_merge), int(channel_split), int(channel_single))
        return channel_state

    def audio_process_stage_1(self, signal, sr, channel_state="100"):
        # 多通道处理
        if channel_state == "100":
            signal = np.mean(signal, axis=1).astype(np.int16)
            raw_data = self.array_2_raw(signal, sr)
            st.audio(raw_data, format='audio/wav')
        elif channel_state == "010":
            signal = signal.transpose().flatten()
            raw_data = self.array_2_raw(signal, sr)
            st.audio(raw_data, format='audio/wav')
        elif channel_state == "001":
            channel_list = list(range(signal.shape[1]))
            self.channel_index = st.selectbox("选择通道：", channel_list)
            signal = signal[:, self.channel_index]
            raw_data = self.array_2_raw(signal, sr)
            st.audio(raw_data, format='audio/wav')
        elif channel_state == "000":
            st.info("请选择通道处理方式！")
        else:
            raise ValueError("")
        return signal, sr

    def show_vad_log(self, signal, sr):
        def on_vad_change(key):
            if "start" in key:
                wav_dic[wav_key]["start"] = st.session_state[key]
            elif "end" in key:
                wav_dic[wav_key]["end"] = st.session_state[key]
            elif "text" in key:
                wav_dic[wav_key]["text"] = st.session_state[key]
            else:
                raise ValueError(" key 设置错误！")
            with open(self.log_file_path, "w") as f:
                json.dump(wav_dic, f, ensure_ascii=False, indent=4)

        f = open(self.log_file_path)
        wav_dic = json.load(f)
        wav_list = list(wav_dic.items())
        per_page_list_len = 5
        split_list_len = int(len(wav_list) / per_page_list_len)

        file_md5 = get_md5(self.log_file_path)
        #
        page_num_vad_key = f"{file_md5}_page_num_vad"
        if page_num_vad_key in st.session_state:
            page_num = st.session_state[page_num_vad_key]
        else:
            page_num = 0
        #
        i = 0
        print("page_num:", page_num)
        for wav_key, value in wav_list[page_num*per_page_list_len:(page_num +1)*per_page_list_len]:
            start = value["start"]
            end = value["end"]
            text = value["text"]
            col0, col1, col2, col3, col4 = st.columns([0.7, 1, 1, 1.5, 1.5])
            with col0:
                st.write("-" * 10)
                st.write("\n")
                st.write("\n")
                st.write(wav_key)
            with col1:
                st.write("-" * 10)
                start_key = f"{file_md5}_{wav_key}_start_p{i}"
                if start_key in st.session_state:
                    start = st.session_state[start_key]
                    start_pn = st.number_input("开始",
                                               value=start,
                                               step=1000,
                                               format="%d",
                                               key=start_key,
                                               on_change=on_vad_change(start_key))
                else:
                    start_pn = st.number_input("开始", value=start, step=1000, format="%d", key=start_key)

            with col2:
                st.write("-" * 10)
                end_key = f"{file_md5}_{wav_key}_end_p{i}"
                if end_key in st.session_state:
                    end = st.session_state[end_key]
                    end_pn = st.number_input("结束",
                                             value=end,
                                             step=1000,
                                             format="%d",
                                             key=end_key,
                                             on_change=on_vad_change(end_key))
                else:
                    end_pn = st.number_input("结束", value=end, step=1000, format="%d", key=end_key)

            with col3:
                st.write("-" * 10)
                st.write("\n")
                tmp_buffer = signal[start_pn:end_pn]
                raw_data = self.array_2_raw(tmp_buffer, sr)
                st.audio(raw_data, format='audio/wav')
            with col4:
                st.write("-" * 10)
                text_key = f"{file_md5}_{wav_key}_text_p{i}"
                if text_key in st.session_state:
                    st.text_input("text",
                                  value=st.session_state[text_key],
                                  key=text_key,
                                  on_change=on_vad_change(text_key),
                                  label_visibility="hidden")
                else:
                    st.text_input("text", value=text, key=text_key, label_visibility="hidden")

                i += 1

        st.write("-"*5)
        page_col1, page_col2, page_col3 = st.columns([4,2,4])
        with page_col1:
            pass
        with page_col2:
            if page_num_vad_key in st.session_state:
                st.number_input(f"页数: 总{split_list_len}", value=st.session_state[page_num_vad_key], step=1,
                                min_value=0, max_value=split_list_len, key=page_num_vad_key)
            else:
                st.number_input(f"页数: 总{split_list_len}", value=0, step=1,
                                min_value=0, max_value=split_list_len, key=page_num_vad_key)
        with page_col3:
            pass

    def audio_process_stage_2(self, signal, sr):
        p_col1, p_col2, p_col3 = st.columns([8, 2, 2])
        with p_col1:
            threhold_value = st.number_input("VAD阈值", value=0.9300, step=0.0001, format="%f", key="threhold_value")
        with p_col2:
            start_index = st.number_input("初始编号", value=0, step=1, format="%d", key="start_index")
        with p_col3:
            st.write("\n")
            st.write("\n")
            if st.button("VAD切分", key="vad"):
                # os.remove(self.log_file_path)
                # st.session_state = {}
                with st.spinner("切分中..."):
                    point_list = self.one_flow_process(signal, threhold_value)
                    if point_list:
                        if len(point_list) == 0:
                            tmp_buffer = signal
                            split_wav_dic = {f"{start_index:06d}": {"start": 0, "end": len(tmp_buffer), "text":""}}
                        elif len(point_list) == 1:
                            tmp_buffer = signal
                            split_wav_dic = {f"{start_index:06d}": {"start": point_list[0], "end": len(tmp_buffer), "text": ""}}
                        elif len(point_list) > 0 and len(point_list) % 2 == 0:
                            split_wav_dic = {}
                            i = start_index
                            for n in range(0, len(point_list) - 1, 2):
                                split_wav_dic[f"{i:06d}"] = {"start": point_list[n], "end": point_list[n + 1], "text": ""}
                                i += 1
                        else:
                            point_list.append(len(signal))
                            split_wav_dic = {}
                            i = start_index
                            for n in range(0, len(point_list) - 1, 2):
                                split_wav_dic[f"{i:06d}"] = {"start": point_list[n], "end": point_list[n + 1], "text": ""}
                                i += 1

                        with open(self.log_file_path, "w") as f:
                            json.dump(split_wav_dic, f, ensure_ascii=False, indent=4)

    def audio_process_stage_3(self, signal, sr):
        st.write("\n" * 10)
        st.write("---" * 60)
        with st.expander("最后一步：完成所有标注任务后，点击展开，并点击下方按钮，生成最终标注数据"):
            # datasets_name = st.text_input("数据集名称：", value="tmp_save_data", key="stage_3_datasets_name")
            if self.datasets_name == "":
                st.error("请输入数据集名称！")
            transcripts_path = '{}/{}/transcripts.txt'.format(self.out_path, self.datasets_name)
            st.write(f"数据保存路劲: {self.args.base_path}/output")
            if st.button("生成标注数据", key="generate_audio"):
                if not os.path.exists(os.path.dirname(transcripts_path)):
                    os.makedirs(os.path.dirname(transcripts_path))
                save_txt = open(transcripts_path, "a+")
                f = open(self.log_file_path)
                wav_dic = json.load(f)
                for wav_key, value in wav_dic.items():
                    start = value["start"]
                    end = value["end"]
                    text = value["text"]
                    if text == "":
                        continue
                    tmp_buffer = signal[start:end] / (2**15 - 1)
                    if self.sr != sr:
                        tmp_buffer = librosa.resample(tmp_buffer, sr, self.sr)
                    wav_save_path = os.path.join(os.path.dirname(transcripts_path), f"{wav_key}.wav")
                    sf.write(wav_save_path, tmp_buffer, self.sr)
                    save_txt.write(f"{wav_key}|{text}\n")
                save_txt.close()
                st.balloons()

    def show_nlp_columns(self):
        def on_vad_change(key):
            wav_dic[wav_key]["pinyin"] = st.session_state[key]
            with open(self.log_file_path, "w") as f:
                json.dump(wav_dic, f, ensure_ascii=False, indent=4)
        if os.path.isfile(self.log_file_path):
            f = open(self.log_file_path)
            wav_dic = json.load(f)
            wav_list = list(wav_dic.items())
            per_page_list_len = 5
            split_list_len = int(len(wav_list) / per_page_list_len)
            #
            file_md5 = get_md5(self.log_file_path)
            page_num_nlp_key = f"{file_md5}_page_num_nlp"
            if page_num_nlp_key in st.session_state:
                page_num = st.session_state[page_num_nlp_key]
            else:
                page_num = 0

            i = 0
            for wav_key, value in wav_list[page_num*per_page_list_len:(page_num +1)*per_page_list_len]:
                text = value["text"]
                col0, col1, col2 = st.columns([1, 1.5, 1.5])
                with col0:
                    st.write("-" * 10)
                    st.write("\n")
                    st.write("\n")
                    st.write(wav_key)
                with col1:
                    st.write("-" * 10)
                    text_key = f"{file_md5}_{wav_key}_text_nlp{i}"
                    text = st.text_input("text",
                                         value=text,
                                         key=text_key,
                                         label_visibility="hidden")
                    pinyin = " ".join(lazy_pinyin(text, style=Style.TONE3))
                with col2:
                    st.write("-" * 10)
                    pinyin_key = f"{file_md5}_{wav_key}_pinyin_p{i}"
                    if pinyin_key in st.session_state:
                        st.text_input("pinyin",
                                      value=pinyin,
                                      key=pinyin_key,
                                      on_change=on_vad_change(pinyin_key),
                                      label_visibility="hidden")
                    else:
                        st.text_input("pinyin", value=pinyin, key=pinyin_key, label_visibility="hidden")
                    i += 1
                    wav_dic[wav_key]["pinyin"] = st.session_state[pinyin_key]
                with open(self.log_file_path, "w") as f:
                    json.dump(wav_dic, f, ensure_ascii=False, indent=4)
            st.write("-" * 5)
            page_col1, page_col2, page_col3 = st.columns([4, 2, 4])
            with page_col2:
                if page_num_nlp_key in st.session_state:
                    st.number_input(f"页数: 总{split_list_len}", value=st.session_state[page_num_nlp_key], step=1,
                                    min_value=0, max_value=split_list_len, key=page_num_nlp_key)
                else:
                    st.number_input(f"页数: 总{split_list_len}", value=0, step=1, min_value=0,
                                    max_value=split_list_len, key=page_num_nlp_key)

    def audio_process_stage_4(self):
        st.write("\n" * 10)
        st.write("---" * 60)
        with st.expander("最后一步：完成所有标注任务后，点击展开，并点击下方按钮，生成最终标注数据"):
            if self.datasets_name == "":
                st.error("请输入数据集名称！")
            transcripts_path = '{}/{}/labels.txt'.format(self.out_path, self.datasets_name)
            st.write(f"数据保存路劲: {self.args.base_path}/output")
            if st.button("生成标注数据", key="nlp"):
                if not os.path.exists(os.path.dirname(transcripts_path)):
                    os.makedirs(os.path.dirname(transcripts_path))
                save_txt = open(transcripts_path, "a+")
                f = open(self.log_file_path)
                wav_dic = json.load(f)
                for wav_key, value in wav_dic.items():
                    pinyin = value["pinyin"]
                    if pinyin != "":
                        save_txt.write(f"{wav_key}|{pinyin}\n")
                save_txt.close()
                st.balloons()

    def process_data(self):
        st.write("_" * 5)
        f_col_1, f_col_2 = st.columns([4, 1])
        with f_col_2:
            st.write("\n")
            st.write("\n")
            if st.button("选择数据路劲"):
                folderpath = filedialog.askdirectory(master=root)
                st.session_state["file_path"] = folderpath
        with f_col_1:
            if "file_path" in st.session_state:
                file_dir = st.text_input("file_path", value=st.session_state["file_path"], key="file_path",
                                         label_visibility="hidden")
            else:
                file_dir = st.text_input("file_path", value="", key="file_path", label_visibility="hidden")

        file_list = [""]
        if file_dir != "":
            for i in os.listdir(file_dir):
                if i.endswith("wav"):
                    file_list.append(i)
            target_file_name = st.radio("任务列表", file_list)
            self.target_file_path = os.path.join(file_dir, target_file_name)

            self.wav_dics = {}
            if os.path.isfile(self.target_file_path):
                signal, sr = self.display_source_audio()
                process_tab1, process_tab2, process_tab3 = st.tabs(["通道处理", "VAD 切分", "NLP"])
                with process_tab1:
                    # 展示通道
                    self.channel_index = 0
                    channel_state = "000"
                    if self.is_single_channel:
                        st.info("单通道音频！")
                    else:
                        channel_state = self.show_channel_columns()
                    signal, sr = self.audio_process_stage_1(signal, sr, channel_state)
                self.log_file_path = os.path.join("log",
                                                  target_file_name.replace(".wav",
                                                                           f"_{channel_state}_{self.channel_index}.json"))
                with process_tab2:
                    self.datasets_name = st.sidebar.text_input("数据集名称：",
                                                               value="",
                                                               key="stage_4_datasets_name")
                    if len(signal.shape) > 1:
                        st.warning("多通道数据，请先进行通道处理，否则默认只使用第一个通道数据！")
                        signal = signal[:, 0]
                    raw_data = self.array_2_raw(signal, sr)
                    st.audio(raw_data, format='audio/wav')
                    if self.vad_option:
                        self.audio_process_stage_2(signal, sr)
                        if os.path.isfile(self.log_file_path):
                            self.show_vad_log(signal, sr)
                    else:
                        st.error("请先勾选左侧栏目中的VAD选项，启动VAD功能！")
                    self.audio_process_stage_3(signal, sr)
                with process_tab3:
                    self.show_nlp_columns()
                    self.audio_process_stage_4()

    def __call__(self, *args, **kwargs):
        config_page()
        # top
        st.sidebar.subheader("配置栏")
        app_option = st.sidebar.selectbox(
            '应用',
            ["导语", "数据处理"]
        )
        if app_option == "导语":
            self.start_doc()
        elif app_option == "数据处理":
            with st.expander("系统初始化操作", expanded=False):
                clean_col1, clean_col2, clean_col3 = st.columns([3, 3, 3])
                with clean_col1:
                    st.markdown("*用于清理历史任务记录*")
                with clean_col2:
                    if st.button("初始化系统"):
                        if not os.path.exists(f"{self.args.base_path}/bak"):
                            os.makedirs(f"{self.args.base_path}/bak")
                        shutil.rmtree(f"{self.args.base_path}/save")
                        st.balloons()
                with clean_col3:
                    if st.button("清理系统"):
                        if not os.path.exists(f"{self.args.base_path}/bak"):
                            os.makedirs(f"{self.args.base_path}/bak")
                        shutil.rmtree(f"{self.args.base_path}/save")
                        st.balloons()
            self._init_save_()
            #
            self.audio_setting()
            self.assist_module()
            self.process_data()

            print_state()
        else:
            raise NotImplementedError


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    # Make folder picker dialog appear on top of other windows
    root.wm_attributes('-topmost', 1)

    app = Builder(args)
    app()

