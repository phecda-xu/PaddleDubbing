import os
os.environ['PPSPEECH_HOME'] = os.getcwd()
import paddle

from tkinter import *
from tkinter import ttk
from ttkbootstrap import Style
from tkinter.messagebox import showinfo, showerror
from tkinter.filedialog import askdirectory, askopenfilename
from src.transcribe import TTSExecutor
from paddlespeech.cli.log import Logger



# tkinter GUI工具居中展示
def center_window(master, width, height):
    screenwidth = master.winfo_screenwidth()
    screenheight = master.winfo_screenheight()
    size = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2,
                            (screenheight - height) / 2)
    master.geometry(size)


def add_scrollbar(master, orient, row=0, column=1, rowspan=1, columnspan=1, sticky=E + NS, padx=10, pady=5):
    sb = Scrollbar(master, orient=orient)
    sb.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, sticky=sticky, padx=padx, pady=pady)
    return sb


class CacheLogger(Logger):
    def __init__(self):
        super(CacheLogger, self).__init__()

    def __call__(self, log_level: int, msg: str):
        loglevel_dict = {20: "info", 30: "warning", 40: "error"}
        self.logger.log(log_level, msg)
        model_Lbox.insert("end", "log_level {}: {}".format(loglevel_dict[log_level], msg))


class VoiceMaker(object):
    def __init__(self):
        self.tts_executor = None
        self.model_setting()
        self.vm_logger = CacheLogger()
        self.n = 0

    def _init_model(self,
                    am='fastspeech2_aishell3',
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
                    device=paddle.get_device()
                    ):
        self.vm_logger(20, "loading model ......")
        self.tts_executor = TTSExecutor(self.vm_logger,
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

        self.button_model.configure(state="disabled")
        self.button_model.configure(text="模型加载完成")

    def process_text(self, text, output_path, spk_id):
        if self.tts_executor is None:
            showerror("错误", "请先选择模型并启动！")
        if text == '':
            showerror("错误", "请先输入文字！")
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        output = os.path.join(output_path, '{}_spk_{}_{}.wav'.format(self.n, spk_id, text))
        wav_file = self.tts_executor(
            text=text,
            output=output,
            spk_id=spk_id
            )
        process_Lbox.insert("end", 'saved : {}'.format(wav_file))
        self.n += 1
        # print('Wave file has been generated: {}'.format(wav_file))

    def process_txt_file(self, master, txt_file, output_path, spk_id):

        if self.tts_executor is None :
            showerror("错误", "请先选择模型并启动！")
        elif txt_file == '':
            showerror("错误", "请输入文件路劲！")
        else:
            self.button_input.configure(text="停止")
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            text_list = self.read_txt(txt_file)
            p_bar = ttk.Progressbar(master, mode="determinate", length=100)
            p_bar.grid(padx=10, pady=20, row=10, column=0, sticky=NSEW)
            p_bar['value'] = 0
            p_bar['maximum'] = len(text_list)
            for i in range(len(text_list)):
                if self.break_in:
                    break
                line = text_list[i]
                output = os.path.join(output_path, "{}_spk_{}_{}.wav".format(self.n, spk_id, line))
                wav_file = self.tts_executor(
                    text=line,
                    output=output,
                    spk_id=spk_id
                )
                p_bar['value'] = i + 1
                self.n += 1
                root.update()
                process_Lbox.insert("end", '保存: {}'.format(wav_file))
                # print('Wave file has been generated: {}'.format(wav_file))
            showinfo("提示", "文件处理完成！")
            self.button_input.configure(text="开始")


    def input_item_adding(self, text, chosen_tuple=None, padx=10, pady=10, fg="red", width=20, font=("黑体", '11')):
        lf = LabelFrame(work_frame, text=text, padx=padx,
                        pady=pady, fg=fg, font=font)
        lf.grid(padx=10, pady=3, sticky=NSEW)
        name = StringVar(lf)
        if chosen_tuple is not None:
            chosen = ttk.Combobox(lf, width=width, textvariable=name)
            chosen.pack()
            chosen['value'] = chosen_tuple
            chosen.grid(row=0, column=0)
            # chosen.current(1)
            return chosen, lf
        else:
            input_entry = Entry(lf, width=width, textvariable=name)
            input_entry.grid(row=0, column=0)
            return input_entry, lf

    def model_bind_event(self, event):
        self.button_model.configure(text="启动模型")
        self.button_model.configure(state="normal")

    def lang_bind_event(self, event):
        if self.lang_entry.get() == "zh":
            am_chosen = ('speedyspeech_csmsc', 'fastspeech2_csmsc', 'fastspeech2_aishell3')
            voc_chosen = ('pwgan_csmsc', 'pwgan_aishell3', 'mb_melgan_csmsc',
                          'style_melgan_csmsc', 'hifigan_csmsc')
            self.am_entry['value'] = am_chosen
            self.voc_entry['value'] = voc_chosen
        elif self.lang_entry.get() == "en":
            am_chosen = ('fastspeech2_ljspeech', 'fastspeech2_vctk')
            voc_chosen = ('pwgan_ljspeech', 'pwgan_vctk')
            self.am_entry['value'] = am_chosen
            self.voc_entry['value'] = voc_chosen
        else:
            showerror("错误", "只支持中文-'ch'和英文-'en'")

    def add_button(self, master, text, width, command, row, column, pady=10, padx=10, sticky=NS):
        submit = Button(master, text=text, width=width, command=command)
        submit.grid(row=row, column=column,padx=padx, pady=pady, sticky=sticky)
        return submit

    def add_directory(self,
                      lf,
                      text_button,
                      width=20,
                      to_dir=False):

        path = StringVar(lf)
        def select_path():
            if to_dir:
                path_ = askdirectory()
            else:
                path_ = askopenfilename()
            path.set(path_)
        input_entry = Entry(lf, width=width, textvariable=path)
        input_entry.grid(row=0, column=0)

        self.add_button(lf,
                        text_button,
                        width=7,
                        row=0,
                        column=1,
                        command=lambda: select_path())
        return input_entry

    @staticmethod
    def read_txt(txt_file):
        text_list = []
        with open(txt_file, "r") as f:
            for line in f.readlines():
                text_list.append(line.strip('\n'))
        return text_list

    def button_input_event(self, event):
        if self.button_input['text'] == "开始":
            self.break_in = 0
        elif self.button_input['text'] == "停止":
            self.break_in = 1


    def outpu_path_set(self, text_frame, text_button, padx=10, pady=10, fg="red", width=20, font=("黑体", '11')):
        lf = LabelFrame(work_frame, text=text_frame, padx=padx,
                        pady=pady, fg=fg, font=font, width=width)
        lf.grid(padx=10, pady=3, sticky=NSEW)
        output_path_entry = self.add_directory(lf, text_button=text_button, width=40, to_dir=True)
        output_path_entry.insert(0, '/media/sf_vbox_share')
        return output_path_entry

    def text_file_path_load(self, text_frame, text_button, padx=10, pady=10, fg="red", width=20, font=("黑体", '11')):
        lf = LabelFrame(work_frame, text=text_frame, padx=padx,
                        pady=pady, fg=fg, font=font, width=width)
        lf.grid(padx=10, pady=3, sticky=NSEW)
        dir_entry = self.add_directory(lf, text_button=text_button, width=40)

        self.break_in = 0
        self.button_input = self.add_button(lf,
                                            text="开始", width=8,
                                            row=10,
                                            column=1,
                                            padx=2,
                                            command=lambda: self.process_txt_file(lf,
                                                                                  txt_file=dir_entry.get(),
                                                                                  output_path=self.output_path_entry.get(),
                                                                                  spk_id=int(self.spk_id_entry.get())),
                                            sticky=E)
        self.button_input.bind("<ButtonPress-1>", self.button_input_event)

    def user_setting(self):
        # spk
        self.spk_id_entry, _ = self.input_item_adding(text="说话人ID(0-20)")
        self.spk_id_entry.insert(0, '1')

        # output
        self.output_path_entry = self.outpu_path_set(text_frame="输出路径", text_button="路径选择...", width=40)

        # text input
        text_entry, text_lf = self.input_item_adding(text="输入文本", width=40)
        self.button_input = self.add_button(text_lf,
                                            text="合成", width=8,
                                            row=0,
                                            column=1,
                                            command=lambda: self.process_text(text=text_entry.get(),
                                                                              output_path=self.output_path_entry.get(),
                                                                              spk_id=int(self.spk_id_entry.get())))

        self.text_file_path_load(text_frame="输入txt文件路劲", text_button="路径选择...", width=40)

    def set_device(self):
        if self.gpu_CheckVar.get():
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.vm_logger(20, "Set gpu, use device 0.")
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ''
            self.vm_logger(20, "Set cpu.")

    def model_setting(self):
        self.gpu_CheckVar = IntVar()
        gpu_check = Checkbutton(work_frame, text="GPU", variable=self.gpu_CheckVar, onvalue=1, offvalue=0, height=2, width=4,
                                command=self.set_device)
        gpu_check.bind("<ButtonPress-1>", self.model_bind_event)
        gpu_check.grid(padx=10, pady=3, sticky=W, row=0, column=0)
        # gpu_check

        am_chosen = ()
        voc_chosen = ()
        lang_chosen = ('zh', 'en')
        self.lang_entry, _ = self.input_item_adding(text="语种", chosen_tuple=lang_chosen, width=40)
        self.lang_entry.bind("<<ComboboxSelected>>", self.lang_bind_event)

        self.am_entry, _ = self.input_item_adding(text="声学模型", chosen_tuple=am_chosen, width=40)
        self.am_entry.bind("<<ComboboxSelected>>", self.model_bind_event)

        self.voc_entry, _ = self.input_item_adding(text="声码器", chosen_tuple=voc_chosen, width=40)
        self.voc_entry.bind("<<ComboboxSelected>>", self.model_bind_event)
        self.button_model = self.add_button(work_frame,
                                            text="启动模型",
                                            width=8,
                                            row=4,
                                            column=0,
                                            command=lambda: self._init_model(am=self.am_entry.get(),
                                                                             voc=self.voc_entry.get(),
                                                                             lang=self.lang_entry.get()))


if __name__ == '__main__':
    # root = Tk()
    style = Style(theme='lumen')
    root = style.master
    center_window(root, 1040, 820)
    root.resizable(width=True, height=True)
    root.title('AI配音工具')

    work_frame = LabelFrame(root, text="工作区", padx=10,
                            pady=10, fg="red", bg='white', font=("黑体", '11'), width=50)
    work_frame.grid(padx=10, pady=3, sticky=NSEW, row=0, column=0)

    log_frame = LabelFrame(root, text="日志区", padx=10,
                            pady=10, fg="red", font=("黑体", '11'), width=50)
    log_frame.grid(padx=10, pady=3, sticky=NSEW, row=0, column=1)

    model_frame = LabelFrame(log_frame, text="模型日志", padx=10,
                             pady=10, fg="red", font=("黑体", '11'), width=50)
    model_frame.grid(padx=10, pady=3, sticky=NSEW, row=0, column=1)

    sb_model_log_x = add_scrollbar(model_frame, row=1, column=0, columnspan=2, sticky=W + EW, orient='horizontal')
    sb_model_log_y = add_scrollbar(model_frame, row=0, column=2, rowspan=5, orient='vertical')
    model_Lbox = Listbox(model_frame, width=50, yscrollcommand=sb_model_log_y.set, selectmode=EXTENDED, height=13)
    model_Lbox.configure(xscrollcommand=sb_model_log_x.set)
    model_Lbox.grid(pady=20, row=0, column=1)
    sb_model_log_x.config(command=model_Lbox.xview)
    sb_model_log_y.config(command=model_Lbox.yview)

    process_frame = LabelFrame(log_frame, text="处理日志", padx=10,
                             pady=10, fg="red", font=("黑体", '11'), width=50)
    process_frame.grid(padx=10, pady=3, sticky=NSEW, row=1, column=1)

    sb_process_log_x = add_scrollbar(process_frame, row=2, column=0, columnspan=2, sticky=W + EW, orient='horizontal')
    sb_process_log_y = add_scrollbar(process_frame, row=1, column=2, rowspan=10, padx=10, pady=5, orient='vertical')
    process_Lbox = Listbox(process_frame, width=50, yscrollcommand=sb_process_log_y.set, selectmode=EXTENDED, height=13)
    process_Lbox.configure(xscrollcommand=sb_process_log_x.set)
    process_Lbox.grid(pady=20, row=1, column=1)
    sb_process_log_x.config(command=process_Lbox.xview())
    sb_process_log_y.config(command=process_Lbox.yview())

    Main = VoiceMaker()
    Main.user_setting()
    root.mainloop()
