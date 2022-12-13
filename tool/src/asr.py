# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
from collections import OrderedDict
from typing import List
from typing import Optional
from typing import Union

import librosa
import numpy as np
import paddle
import soundfile
from yacs.config import CfgNode

from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.cli.download import get_path_from_url
from paddlespeech.cli.log import logger
from paddlespeech.cli.utils import cli_register
from paddlespeech.utils.env import MODEL_HOME
from paddlespeech.cli.utils import stats_wrapper
from paddlespeech.resource.pretrained_models import asr_dynamic_pretrained_models as pretrained_models
# from paddlespeech.resource.model_alias import model_alias
from paddlespeech.s2t.frontend.featurizer.text_featurizer import TextFeaturizer
from paddlespeech.audio.transform.transformation import Transformation
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.s2t.utils.utility import UpdateConfig

__all__ = ['ASRExecutor']

model_alias = {
    "deepspeech2offline": "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "deepspeech2online": "paddlespeech.s2t.models.ds2:DeepSpeech2Model",
    "conformer": "paddlespeech.s2t.models.u2:U2Model",
    "conformer_online": "paddlespeech.s2t.models.u2:U2Model",
    "transformer": "paddlespeech.s2t.models.u2:U2Model",
    "wenetspeech": "paddlespeech.s2t.models.u2:U2Model",
}




@cli_register(
    name='paddlespeech.asr', description='Speech to text infer command.')
class ASRExecutor(object):
    def __init__(self,
                 model_type,
                 model_tag: str = '1.0',
                 lang: str = 'zh',
                 sample_rate: int = 16000,
                 config: os.PathLike = None,
                 ckpt_path: os.PathLike = None,
                 decode_method: str = 'attention_rescoring',
                 force_yes: bool = False,
                 device=paddle.get_device()):
        super().__init__()
        self.model_alias = model_alias
        self.pretrained_models = pretrained_models
        self._inputs = dict()
        self._outputs = dict()
        self.decode_method = decode_method
        self.force_yes = force_yes
        self.model_type = model_type
        self.sample_rate = sample_rate
        paddle.set_device(device)
        self._init_from_path(model_type=model_type,
                             model_tag=model_tag,
                             lang=lang,
                             sample_rate=sample_rate,
                             cfg_path=config,
                             ckpt_path=ckpt_path)

    def _get_pretrained_path(self, model_full_name: str, model_tag: str) -> os.PathLike:
        """
        Download and returns pretrained resources path of current task.
        """
        support_models = list(self.pretrained_models.keys())
        assert model_full_name in self.pretrained_models, 'The model "{}" you want to use has not been supported, please choose other models.\nThe support models includes:\n\t\t{}\n'.format(
            model_full_name, '\n\t\t'.join(support_models))

        res_path = os.path.join(MODEL_HOME, model_full_name)
        print(f"Model save dir: {res_path}")

        decompressed_path = download_and_decompress(self.pretrained_models[model_full_name][model_tag],
                                                    res_path)
        decompressed_path = os.path.abspath(decompressed_path)
        logger.info(
            'Use pretrained model stored in: {}'.format(decompressed_path))

        return decompressed_path

    def _init_from_path(self,
                        model_type: str='wenetspeech',
                        model_tag: str='1.0',
                        lang: str='zh',
                        sample_rate: int=16000,
                        cfg_path: Optional[os.PathLike]=None,
                        decode_method: str='attention_rescoring',
                        ckpt_path: Optional[os.PathLike]=None):
        """
        Init model and other resources from a specific path.
        """
        logger.info("start to init the model")
        if hasattr(self, 'model'):
            logger.info('Model had been initialized.')
            return

        if cfg_path is None or ckpt_path is None:
            sample_rate_str = '16k' if sample_rate == 16000 else '8k'
            model_full_name = model_type + '-' + lang + '-' + sample_rate_str
            res_path = self._get_pretrained_path(model_full_name, model_tag)  # wenetspeech_zh
            self.res_path = res_path
            self.cfg_path = os.path.join(
                res_path, self.pretrained_models[model_full_name][model_tag]['cfg_path'])
            self.ckpt_path = os.path.join(
                res_path,
                self.pretrained_models[model_full_name][model_tag]['ckpt_path'] + ".pdparams")
            logger.info(res_path)

        else:
            self.cfg_path = os.path.abspath(cfg_path)
            self.ckpt_path = os.path.abspath(ckpt_path + ".pdparams")
            self.res_path = os.path.dirname(
                os.path.dirname(os.path.abspath(self.cfg_path)))
        logger.info(self.cfg_path)
        logger.info(self.ckpt_path)

        #Init body.
        self.config = CfgNode(new_allowed=True)
        self.config.merge_from_file(self.cfg_path)

        with UpdateConfig(self.config):
            if "deepspeech2online" in model_type or "deepspeech2offline" in model_type:
                from paddlespeech.s2t.io.collator import SpeechCollator
                self.vocab = self.config.vocab_filepath
                self.config.decode.lang_model_path = os.path.join(
                    MODEL_HOME, 'language_model',
                    self.config.decode.lang_model_path)
                self.collate_fn_test = SpeechCollator.from_config(self.config)
                self.text_feature = TextFeaturizer(
                    unit_type=self.config.unit_type, vocab=self.vocab)
                lm_url = self.pretrained_models[model_full_name][model_tag]['lm_url']
                lm_md5 = self.pretrained_models[model_full_name][model_tag]['lm_md5']
                self.download_lm(
                    lm_url,
                    os.path.dirname(self.config.decode.lang_model_path), lm_md5)

            elif "conformer" in model_type or "transformer" in model_type or "wenetspeech" in model_type:
                self.config.spm_model_prefix = os.path.join(
                    self.res_path, self.config.spm_model_prefix)
                self.text_feature = TextFeaturizer(
                    unit_type=self.config.unit_type,
                    vocab=self.config.vocab_filepath,
                    spm_model_prefix=self.config.spm_model_prefix)
                self.config.decode.decoding_method = decode_method
            else:
                raise Exception("wrong type")
        model_name = model_type[:model_type.rindex(
            '_')]  # model_type: {model_name}_{dataset}
        print("1111111 ", model_name)
        model_class = dynamic_import(model_name, self.model_alias)
        model_conf = self.config
        model = model_class.from_config(model_conf)
        self.model = model
        self.model.eval()

        # load model
        model_dict = paddle.load(self.ckpt_path)
        self.model.set_state_dict(model_dict)

    def preprocess(self, input: Union[str, os.PathLike]):
        """
        Input preprocess and return paddle.Tensor stored in self.input.
        Input content can be a text(tts), a file(asr, cls) or a streaming(not supported yet).
        """

        audio_file = input
        if isinstance(audio_file, (str, os.PathLike)):
            logger.info("Preprocess audio_file:" + audio_file)

        # Get the object for feature extraction
        if "deepspeech2online" in self.model_type or "deepspeech2offline" in self.model_type:
            audio, _ = self.collate_fn_test.process_utterance(
                audio_file=audio_file, transcript=" ")
            audio_len = audio.shape[0]
            audio = paddle.to_tensor(audio, dtype='float32')
            audio_len = paddle.to_tensor(audio_len)
            audio = paddle.unsqueeze(audio, axis=0)
            # vocab_list = collate_fn_test.vocab_list
            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        elif "conformer" in self.model_type or "transformer" in self.model_type or "wenetspeech" in self.model_type:
            logger.info("get the preprocess conf")
            preprocess_conf = self.config.preprocess_config
            preprocess_args = {"train": False}
            preprocessing = Transformation(preprocess_conf)
            logger.info("read the audio file")
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)

            if self.change_format:
                if audio.shape[1] >= 2:
                    audio = audio.mean(axis=1, dtype=np.int16)
                else:
                    audio = audio[:, 0]
                # pcm16 -> pcm 32
                audio = self._pcm16to32(audio)
                audio = librosa.resample(
                    audio,
                    orig_sr=audio_sample_rate,
                    target_sr=self.sample_rate)
                audio_sample_rate = self.sample_rate
                # pcm32 -> pcm 16
                audio = self._pcm32to16(audio)
            else:
                audio = audio[:, 0]

            logger.info(f"audio shape: {audio.shape}")
            # fbank
            audio = preprocessing(audio, **preprocess_args)

            audio_len = paddle.to_tensor(audio.shape[0])
            audio = paddle.to_tensor(audio, dtype='float32').unsqueeze(axis=0)

            self._inputs["audio"] = audio
            self._inputs["audio_len"] = audio_len
            logger.info(f"audio feat shape: {audio.shape}")

        else:
            raise Exception("wrong type")

        logger.info("audio feat process success")

    @paddle.no_grad()
    def infer(self):
        """
        Model inference and result stored in self.output.
        """
        logger.info("start to infer the model to get the output")
        cfg = self.config.decode
        audio = self._inputs["audio"]
        audio_len = self._inputs["audio_len"]
        if "deepspeech2online" in self.model_type or "deepspeech2offline" in self.model_type:
            decode_batch_size = audio.shape[0]
            self.model.decoder.init_decoder(
                decode_batch_size, self.text_feature.vocab_list,
                cfg.decoding_method, cfg.lang_model_path, cfg.alpha, cfg.beta,
                cfg.beam_size, cfg.cutoff_prob, cfg.cutoff_top_n,
                cfg.num_proc_bsearch)

            result_transcripts = self.model.decode(audio, audio_len)
            self.model.decoder.del_decoder()
            self._outputs["result"] = result_transcripts[0]

        elif "conformer" in self.model_type or "transformer" in self.model_type:
            logger.info(
                f"we will use the transformer like model : {self.model_type}")
            try:
                result_transcripts = self.model.decode(
                    audio,
                    audio_len,
                    text_feature=self.text_feature,
                    decoding_method=cfg.decoding_method,
                    beam_size=cfg.beam_size,
                    ctc_weight=cfg.ctc_weight,
                    decoding_chunk_size=cfg.decoding_chunk_size,
                    num_decoding_left_chunks=cfg.num_decoding_left_chunks,
                    simulate_streaming=cfg.simulate_streaming)
                self._outputs["result"] = result_transcripts[0][0]
            except Exception as e:
                logger.exception(e)

        else:
            raise Exception("invalid model name")

    def postprocess(self) -> Union[str, os.PathLike]:
        """
            Output postprocess and return human-readable results such as texts and audio files.
        """
        return self._outputs["result"]

    def download_lm(self, url, lm_dir, md5sum):
        download_path = get_path_from_url(
            url=url,
            root_dir=lm_dir,
            md5sum=md5sum,
            decompress=False, )

    def _pcm16to32(self, audio):
        assert (audio.dtype == np.int16)
        audio = audio.astype("float32")
        bits = np.iinfo(np.int16).bits
        audio = audio / (2**(bits - 1))
        return audio

    def _pcm32to16(self, audio):
        assert (audio.dtype == np.float32)
        bits = np.iinfo(np.int16).bits
        audio = audio * (2**(bits - 1))
        audio = np.round(audio).astype("int16")
        return audio

    def _check(self, audio_file: str, sample_rate: int, force_yes: bool):
        self.sample_rate = sample_rate
        if self.sample_rate != 16000 and self.sample_rate != 8000:
            logger.error(
                "invalid sample rate, please input --sr 8000 or --sr 16000")
            return False

        if isinstance(audio_file, (str, os.PathLike)):
            if not os.path.isfile(audio_file):
                logger.error("Please input the right audio file path")
                return False

        logger.info("checking the audio file format......")
        try:
            audio, audio_sample_rate = soundfile.read(
                audio_file, dtype="int16", always_2d=True)
            audio_duration = audio.shape[0] / audio_sample_rate
            max_duration = 50.0
            if audio_duration >= max_duration:
                logger.error("Please input audio file less then 50 seconds.\n")
                return False
        except Exception as e:
            logger.exception(e)
            logger.error(
                "can not open the audio file, please check the audio file format is 'wav'. \n \
                 you can try to use sox to change the file format.\n \
                 For example: \n \
                 sample rate: 16k \n \
                 sox input_audio.xx --rate 16k --bits 16 --channels 1 output_audio.wav \n \
                 sample rate: 8k \n \
                 sox input_audio.xx --rate 8k --bits 16 --channels 1 output_audio.wav \n \
                 ")
            return False
        logger.info("The sample rate is %d" % audio_sample_rate)
        if audio_sample_rate != self.sample_rate:
            logger.warning("The sample rate of the input file is not {}.\n \
                            The program will resample the wav file to {}.\n \
                            If the result does not meet your expectations，\n \
                            Please input the 16k 16 bit 1 channel wav file. \
                        ".format(self.sample_rate, self.sample_rate))
            if force_yes is False:
                while (True):
                    logger.info(
                        "Whether to change the sample rate and the channel. Y: change the sample. N: exit the prgream."
                    )
                    content = "Y"
                    if content.strip() == "Y" or content.strip(
                    ) == "y" or content.strip() == "yes" or content.strip(
                    ) == "Yes":
                        logger.info(
                            "change the sampele rate, channel to 16k and 1 channel"
                        )
                        break
                    elif content.strip() == "N" or content.strip(
                    ) == "n" or content.strip() == "no" or content.strip(
                    ) == "No":
                        logger.info("Exit the program")
                        return False
                    else:
                        logger.warning("Not regular input, please input again")

            self.change_format = True
        else:
            logger.info("The audio file format is right")
            self.change_format = False

        return True

    @stats_wrapper
    def __call__(self, audio_file: os.PathLike):
        """
        Python API to call an executor.
        """
        audio_file = os.path.abspath(audio_file)
        if not self._check(audio_file, self.sample_rate, self.force_yes):
            sys.exit(-1)
        self.preprocess(audio_file)
        self.infer()
        res = self.postprocess()  # Retrieve result of asr.

        return res
