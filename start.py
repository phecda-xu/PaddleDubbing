import os
os.environ['PPSPEECH_HOME'] = os.getcwd()

import json
import tarfile
import paddle
import streamlit as st
from datetime import datetime
from src.transcribe import TTSExecutor


@st.cache(allow_output_mutation=True)
def load_model(am='fastspeech2_aishell3',
               am_config=None,
               am_ckpt=None,
               am_stat=None,
               phones_dict=None,
               tones_dict=None,
               speaker_dict=None,
               voc='pwgan_aishell3',
               voc_config=None,
               voc_ckpt=None,
               voc_stat=None,
               lang='zh',
               device=paddle.get_device()):
    tts_executor = TTSExecutor(logger=None,
                               am=am,
                               am_config=am_config,
                               am_ckpt=am_ckpt,
                               am_stat=am_stat,
                               phones_dict=phones_dict,
                               tones_dict=tones_dict,
                               speaker_dict=speaker_dict,
                               voc=voc,
                               voc_config=voc_config,
                               voc_ckpt=voc_ckpt,
                               voc_stat=voc_stat,
                               lang=lang,
                               device=device)
    return tts_executor


class Builder():
    def __init__(self):
        st.set_page_config(
                page_title="Dubbing配音工具",  # st.get_option(""),
                page_icon=":shark",
                layout="wide",
                initial_sidebar_state="auto",
            )
        self.jsonfile_path = os.path.join(os.getcwd(), "history")
        if not os.path.exists(self.jsonfile_path):
            os.makedirs(self.jsonfile_path)

    def start_doc(self):
        st.header("导语")
        st.text("欢迎使用AI配音工具，本工具给予streamlit搭建。")
        st.text("作者: phecda-xu")
        st.markdown("模型能力由[paddlespeech](https://github.com/PaddlePaddle/PaddleSpeech)提供。")
        st.subheader("操作引导")
        st.markdown("先在`应用`中选择要使用的功能。")
        st.markdown("然后页面右上角选择栏里面有 `说明`，`选我开始`，`历史记录`三个选项。")
        st.markdown("参照`说明`内容配置左侧的内容。")
        st.markdown("配置完成后，选择`选我开始`，然后继续按照`说明`的内容操作即可。")
        st.markdown("所有合成的音频都可以在`历史记录`里面看到。")
        st.markdown("任务执行过程中不要做其他操作，否则会打断当前任务进程。")

    def tts_leader_doc(self):
        st.subheader("语音合成")
        st.markdown("**第一步**: 选择是否使用GPU（GPU默认只使用 device id 为0的设备）")
        st.markdown("**第二步**: 选择语种，目前支持中文（`zh`）,英文（`en`）,不支持双语混合，可用中文音替换。")
        st.markdown("**第三步**: 选择声学模型和声码器，比如： `fastspeech2_aishell3` 和 `pwgan_aishell3` 组合。")
        st.markdown("**注意**:下划线后边的要保持一致比如 `aishell3` 这样合成音频的质量才是好的。")
        st.markdown("选定后页面右侧会出现加载字样和图标，配置参数变动后都会自动重新加载模型。")

        st.markdown("**第四步**: 选择说话人ID，每个ID对应一个说话风格，大部分是女声。")
        st.markdown("**第五步**: 语速和音调暂时不可用。")
        st.markdown("**第六步**: 保存音频的地址，默认为当前代码路劲下的 `output/` 。")


    def gpu_setting(self):
        col1, col2 = st.sidebar.columns([1, 3])
        with col1:
            gpu_option = st.checkbox('GPU')
        with col2:
            if gpu_option:
                st.markdown("<font size=3.5>(use gpu device 0)</font>", unsafe_allow_html=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            else:
                st.markdown("<font size=3.5>(use cpu)</font> ", unsafe_allow_html=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = ''

    def model_setting(self):
        # st.sidebar.header("语种")
        lang_option = st.sidebar.selectbox(
            '选择语言',
            ['zh', 'en']
        )
        # st.sidebar.header("配置模型")
        if lang_option == "zh":
            am_option = st.sidebar.selectbox(
                '声学模型',
                ['', 'speedyspeech_csmsc', 'fastspeech2_csmsc', 'fastspeech2_aishell3']
            )

            voc_option = st.sidebar.selectbox(
                '声码器',
                ['', 'pwgan_csmsc', 'pwgan_aishell3', 'mb_melgan_csmsc',
                 'style_melgan_csmsc', 'hifigan_csmsc']
            )
        elif lang_option == "en":
            am_option = st.sidebar.selectbox(
                '声学模型',
                ['', 'fastspeech2_ljspeech', 'fastspeech2_vctk']
            )

            voc_option = st.sidebar.selectbox(
                '声码器',
                ['', 'pwgan_ljspeech', 'pwgan_vctk']
            )
        else:
            raise ValueError("")

        # 自动加载模型
        if am_option != '' and voc_option != '':
            with st.spinner("模型加载中..."):
                self.tts_executor = load_model(am_option, voc_option, lang_option)

    def other_setting(self):
        st.sidebar.header("参数配置")
        self.spk_option = st.sidebar.number_input(label='说话人ID', min_value=0, step=1, format='%d')
        self.speed_option = st.sidebar.number_input(label='语速', value=1.0, min_value=0.0)
        self.energy_option = st.sidebar.number_input(label='音调', value=1.0, min_value=0.0)

        output_path = os.path.join(os.getcwd(), 'output')
        self.save_option = st.sidebar.text_input("保存路劲", output_path)
        if not os.path.exists(self.save_option):
            os.makedirs(self.save_option)

    def get_text_input(self):
        uploader_file = st.file_uploader("输入文件：")
        if uploader_file is not None:
            res = uploader_file.getvalue().decode("utf-8")
            text = st.text_area("输入文本：",
                                value=res, key=None)
        else:
            text = st.text_area("输入文本：(在这里输入多行文本,每行文本字数不限,文本框右下角可以拖动。 )",
                                value='', key=None)
        text_list = text.split("\n")
        return text_list

    @staticmethod
    def get_wav_data(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return data

    def make_targz_one_by_one(self, output_filename, pathfile):
        tar = tarfile.open(output_filename, "w:gz")
        tar.add(pathfile)
        tar.close()

    def update_history(self):
        for json_file in os.listdir(self.jsonfile_path):
            json_name = os.path.basename(json_file)
            with st.expander(json_name[:-5], expanded=False):
                with open(os.path.join(self.jsonfile_path, json_file), 'r') as f:
                    col1, col2, col3 = st.columns([3, 3, 1])
                    col1.subheader("文本")
                    col2.subheader("音频")
                    col3.subheader("下载")
                    n = 0
                    with st.spinner("加载中..."):
                        for i in f.readlines():
                            json_dict = json.loads(i)
                            col1, col2, col3 = st.columns([3, 3, 1])
                            n += 1
                            with col1:
                                st.write("-" * 60)
                                st.write("句 {} : {} ".format(n, json_dict["text"]))
                                output = json_dict["audio_path"]
                            buffer = self.get_wav_data(output)
                            with col2:
                                st.write("-" * 60)
                                st.audio(buffer, format='audio/wav')
                            with col3:
                                st.write("-" * 60)
                                st.download_button(
                                    label="Download",
                                    data=buffer,
                                    file_name=os.path.basename(output),
                                    mime="application/octet-stream",
                                    key=json_name[:-5] + str(n)
                                )
                    with col3:
                        st.write("-" * 60)
                        # if st.button("打包下载"):
                        #     st.empty()
                        #     tarfile_name = json_name[:-5] + 'tar'
                            # self.make_targz_one_by_one(tarfile_name, )

    def process(self, text_list):
        jsonfile_name = "{}.json".format(datetime.now().strftime('%Y_%m_%d%Z_%H_%M_%S'))

        p_col1, p_col2, p_col3, p_col4 = st.columns([16, 1, 1, 1])
        with p_col1:
            st.write("")
            process_bar = st.progress(0)
            step = int((1 / len(text_list)) * 100)
        with p_col2:
            if text_list[0] == '':
                st.markdown("<font size=5.5> {} 条</font>".format(len(text_list) - 1), unsafe_allow_html=True)
            else:
                st.markdown("<font size=5.5> {} 条</font>".format(len(text_list)), unsafe_allow_html=True)
        with p_col3:
            start_button = st.button("开始")
        with p_col4:
            stop_button = st.button("停止")
        # 开始合成后的过程
        if start_button:
            with st.expander("详细信息", expanded=True):
                col1, col2, col3 = st.columns([3, 3, 1])
                col1.subheader("文本")
                col2.subheader("音频")
                col3.subheader("下载")
                n = 0
                break_state = 0
                for i in text_list:
                    col1, col2, col3 = st.columns([3, 3, 1])
                    n += 1
                    if i == '':
                        st.error("请输入要合成的文本内容！！")
                        break_state = 2
                        break
                    if stop_button:
                        break_state = 2
                        break
                    with col1:
                        st.write("-" * 60)
                        st.write("句 {} : {} ".format(n, i))
                        outfile_name = '{}_spk_{}_{}.wav'.format(n, self.spk_option, i)
                        output = os.path.join(self.save_option, outfile_name)
                        with st.spinner("合成中..."):
                            wav_file = self.tts_executor(
                                text=i,
                                output=output,
                                spk_id=self.spk_option
                            )
                        with open(os.path.join(self.jsonfile_path, jsonfile_name), 'a') as f:
                            f.write(json.dumps({
                                "text": i,
                                "audio_path": output
                            }) + '\n')

                    process_bar.progress(n * step)
                    buffer = self.get_wav_data(wav_file)
                    with col2:
                        st.write("-" * 60)
                        st.audio(buffer, format='audio/wav')
                    with col3:
                        st.write("-" * 60)
                        st.download_button(
                            label="Download",
                            data=buffer,
                            file_name=outfile_name,
                            mime="application/octet-stream",
                            key=n
                        )
                    break_state = 1
                if break_state == 1:
                    process_bar.progress(100)
                    st.balloons()

    def __call__(self, *args, **kwargs):
        # 侧边栏应用分页设置
        st.sidebar.subheader("配置栏")
        app_option = st.sidebar.selectbox(
            '应用',
            ["", "语音合成", "语音克隆"]
        )
        if app_option == "语音合成":
            # 侧边栏GPU相关显示设置
            self.gpu_setting()
            # 侧边栏语种、声学模型、声码器选择设置
            self.model_setting()
            # 侧边栏合成参数、保存路劲配置
            self.other_setting()
            # 显示区显示设置
            col1, col2 = st.columns([17, 2])
            with col1:
                st.empty()
            with col2:
                area_option = st.selectbox(
                    '',
                    ['说明', '选我开始', '历史记录']
                )
            # 主页操作区
            if area_option == '选我开始':
                # 取输入文本
                text_list = self.get_text_input()
                # 合成
                self.process(text_list)
            # 历史记录区
            elif area_option == "历史记录":
                self.update_history()
            else:
                self.tts_leader_doc()

        elif app_option == "语音克隆":
            st.text("模型能力由paddlespeech提供。")
        else:
            self.start_doc()


if __name__ == "__main__":
    app = Builder()
    app()