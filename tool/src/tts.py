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
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import paddle
import soundfile as sf
import yaml
from yacs.config import CfgNode

# from paddlespeech.cli.utils import download_and_decompress
from paddlespeech.utils.env import MODEL_HOME
from paddlespeech.cli.utils import stats_wrapper
from paddlespeech.s2t.utils.dynamic_import import dynamic_import
from paddlespeech.t2s.frontend import English
from paddlespeech.t2s.frontend.zh_frontend import Frontend
from paddlespeech.t2s.frontend.mix_frontend import MixFrontend
from paddlespeech.t2s.modules.normalizer import ZScore
from .utils import decompress
from paddlespeech.resource.pretrained_models import tts_dynamic_pretrained_models, tts_static_pretrained_models, g2pw_onnx_models
# from paddlespeech.t2s.exps.syn_utils import model_alias

__all__ = ['TTSExecutor']


# pretrained_models = {
#     # speedyspeech
#     "speedyspeech_csmsc-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/speedyspeech/speedyspeech_nosil_baker_ckpt_0.5.zip',
#         'md5':
#         '9edce23b1a87f31b814d9477bf52afbc',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_11400.pdz',
#         'speech_stats':
#         'feats_stats.npy',
#         'phones_dict':
#         'phone_id_map.txt',
#         'tones_dict':
#         'tone_id_map.txt',
#     },
#
#     # fastspeech2
#     "fastspeech2_csmsc-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_baker_ckpt_0.4.zip',
#         'md5':
#         '637d28a5e53aa60275612ba4393d5f22',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_76000.pdz',
#         'speech_stats':
#         'speech_stats.npy',
#         'phones_dict':
#         'phone_id_map.txt',
#         'pitch_stats':
#             'pitch_stats.npy',
#         'energy_stats':
#             'energy_stats.npy',
#     },
#     "fastspeech2_ljspeech-en": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_ljspeech_ckpt_0.5.zip',
#         'md5':
#         'ffed800c93deaf16ca9b3af89bfcd747',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_100000.pdz',
#         'speech_stats':
#         'speech_stats.npy',
#         'phones_dict':
#         'phone_id_map.txt',
#         'pitch_stats':
#             'pitch_stats.npy',
#         'energy_stats':
#             'energy_stats.npy',
#     },
#     "fastspeech2_aishell3-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_aishell3_ckpt_0.4.zip',
#         'md5':
#         'f4dd4a5f49a4552b77981f544ab3392e',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_96400.pdz',
#         'speech_stats':
#         'speech_stats.npy',
#         'phones_dict':
#         'phone_id_map.txt',
#         'speaker_dict':
#         'speaker_id_map.txt',
#         'pitch_stats':
#             'pitch_stats.npy',
#         'energy_stats':
#             'energy_stats.npy',
#     },
#     "fastspeech2_vctk-en": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/fastspeech2/fastspeech2_nosil_vctk_ckpt_0.5.zip',
#         'md5':
#         '743e5024ca1e17a88c5c271db9779ba4',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_66200.pdz',
#         'speech_stats':
#         'speech_stats.npy',
#         'phones_dict':
#         'phone_id_map.txt',
#         'speaker_dict':
#         'speaker_id_map.txt',
#         'pitch_stats':
#             'pitch_stats.npy',
#         'energy_stats':
#             'energy_stats.npy',
#     },
#     # pwgan
#     "pwgan_csmsc-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_baker_ckpt_0.4.zip',
#         'md5':
#         '2e481633325b5bdf0a3823c714d2c117',
#         'config':
#         'pwg_default.yaml',
#         'ckpt':
#         'pwg_snapshot_iter_400000.pdz',
#         'speech_stats':
#         'pwg_stats.npy',
#     },
#     "pwgan_ljspeech-en": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_ljspeech_ckpt_0.5.zip',
#         'md5':
#         '53610ba9708fd3008ccaf8e99dacbaf0',
#         'config':
#         'pwg_default.yaml',
#         'ckpt':
#         'pwg_snapshot_iter_400000.pdz',
#         'speech_stats':
#         'pwg_stats.npy',
#     },
#     "pwgan_aishell3-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_aishell3_ckpt_0.5.zip',
#         'md5':
#         'd7598fa41ad362d62f85ffc0f07e3d84',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_1000000.pdz',
#         'speech_stats':
#         'feats_stats.npy',
#     },
#     "pwgan_vctk-en": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/pwgan/pwg_vctk_ckpt_0.1.1.zip',
#         'md5':
#         'b3da1defcde3e578be71eb284cb89f2c',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_1500000.pdz',
#         'speech_stats':
#         'feats_stats.npy',
#     },
#     # mb_melgan
#     "mb_melgan_csmsc-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/mb_melgan/mb_melgan_csmsc_ckpt_0.1.1.zip',
#         'md5':
#         'ee5f0604e20091f0d495b6ec4618b90d',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_1000000.pdz',
#         'speech_stats':
#         'feats_stats.npy',
#     },
#     # style_melgan
#     "style_melgan_csmsc-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/style_melgan/style_melgan_csmsc_ckpt_0.1.1.zip',
#         'md5':
#         '5de2d5348f396de0c966926b8c462755',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_1500000.pdz',
#         'speech_stats':
#         'feats_stats.npy',
#     },
#     # hifigan
#     "hifigan_csmsc-zh": {
#         'url':
#         'https://paddlespeech.bj.bcebos.com/Parakeet/released_models/hifigan/hifigan_csmsc_ckpt_0.1.1.zip',
#         'md5':
#         'dd40a3d88dfcf64513fba2f0f961ada6',
#         'config':
#         'default.yaml',
#         'ckpt':
#         'snapshot_iter_2500000.pdz',
#         'speech_stats':
#         'feats_stats.npy',
#     },
# }

pretrained_models = tts_dynamic_pretrained_models
front_models = g2pw_onnx_models


model_alias = {
    # acoustic model
    "speedyspeech":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeech",
    "speedyspeech_inference":
    "paddlespeech.t2s.models.speedyspeech:SpeedySpeechInference",
    "fastspeech2":
    "paddlespeech.t2s.models.fastspeech2:FastSpeech2",
    "fastspeech2_inference":
    "paddlespeech.t2s.models.fastspeech2:StyleFastSpeech2Inference",
    "tacotron2":
    "paddlespeech.t2s.models.tacotron2:Tacotron2",
    "tacotron2_inference":
    "paddlespeech.t2s.models.tacotron2:Tacotron2Inference",
    # voc
    "pwgan":
    "paddlespeech.t2s.models.parallel_wavegan:PWGGenerator",
    "pwgan_inference":
    "paddlespeech.t2s.models.parallel_wavegan:PWGInference",
    "mb_melgan":
    "paddlespeech.t2s.models.melgan:MelGANGenerator",
    "mb_melgan_inference":
    "paddlespeech.t2s.models.melgan:MelGANInference",
    "style_melgan":
    "paddlespeech.t2s.models.melgan:StyleMelGANGenerator",
    "style_melgan_inference":
    "paddlespeech.t2s.models.melgan:StyleMelGANInference",
    "hifigan":
    "paddlespeech.t2s.models.hifigan:HiFiGANGenerator",
    "hifigan_inference":
    "paddlespeech.t2s.models.hifigan:HiFiGANInference",
    "wavernn":
    "paddlespeech.t2s.models.wavernn:WaveRNN",
    "wavernn_inference":
    "paddlespeech.t2s.models.wavernn:WaveRNNInference",
}

# tuning_dic = {
#     "tuning_npy_path": os.path.join(os.getcwd(), 'models/fastspeech2_csmsc-zh/fastspeech2_nosil_baker_ckpt_0.4'),
# }


class TTSExecutor(object):
    def __init__(self,
                 am: str = 'fastspeech2_csmsc',
                 am_config: Optional[os.PathLike] = None,
                 am_ckpt: Optional[os.PathLike] = None,
                 am_stat: Optional[os.PathLike] = None,
                 am_tag: Optional[str] = None,
                 phones_dict: Optional[os.PathLike] = None,
                 tones_dict: Optional[os.PathLike] = None,
                 speaker_dict: Optional[os.PathLike] = None,
                 voc: str = 'pwgan_csmsc',
                 voc_config: Optional[os.PathLike] = None,
                 voc_ckpt: Optional[os.PathLike] = None,
                 voc_stat: Optional[os.PathLike] = None,
                 voc_tag: Optional[str] = None,
                 lang: str = 'zh',
                 front: str = 'pinyin',
                 device: str = paddle.get_device()
                 ):
        super().__init__()
        self._inputs = dict()
        self._outputs = dict()
        paddle.set_device(device)
        self._init_from_path(
            am=am,
            am_config=am_config,
            am_ckpt=am_ckpt,
            am_stat=am_stat,
            am_tag=am_tag,
            phones_dict=phones_dict,
            tones_dict=tones_dict,
            speaker_dict=speaker_dict,
            voc=voc,
            voc_config=voc_config,
            voc_ckpt=voc_ckpt,
            voc_stat=voc_stat,
            voc_tag=voc_tag,
            front=front,
            lang=lang)
        self.lang = lang
        self.am = am

    def _get_pretrained_path(self, model_name: str, tag: str) -> os.PathLike:
        """
        Download and returns pretrained resources path of current task.
        """
        assert model_name in pretrained_models, 'Can not find pretrained resources of {}.'.format(model_name)

        res_path = os.path.join(MODEL_HOME, model_name)
        if not os.path.isdir(res_path):
            print(
                'Download pretrained model and stored in: {}'.format(res_path))
            raise ValueError(f"请先下载模型文件到 {res_path} 目录下！")
        else:
            decompressed_path = decompress(pretrained_models[model_name][tag],
                                           res_path)
            decompressed_path = os.path.abspath(decompressed_path)
            print(
                'Use pretrained model stored in: {}'.format(decompressed_path))
        return decompressed_path

    def _init_from_path(
            self,
            am: str='fastspeech2_csmsc',
            am_config: Optional[os.PathLike]=None,
            am_ckpt: Optional[os.PathLike]=None,
            am_stat: Optional[os.PathLike]=None,
            am_tag:Optional[str]=None,
            phones_dict: Optional[os.PathLike]=None,
            tones_dict: Optional[os.PathLike]=None,
            speaker_dict: Optional[os.PathLike]=None,
            voc: str='pwgan_csmsc',
            voc_config: Optional[os.PathLike]=None,
            voc_ckpt: Optional[os.PathLike]=None,
            voc_stat: Optional[os.PathLike]=None,
            voc_tag:Optional[str]=None,
            front: Optional[str]=None,
            lang: str='zh', ):
        """
        Init model and other resources from a specific path.
        """
        if hasattr(self, 'am_inference') and hasattr(self, 'voc_inference'):
            print('Models had been initialized.')
            return
        # am
        am_full_name = am + '-' + lang
        if am_ckpt is None or am_config is None or am_stat is None or phones_dict is None:
            am_res_path = self._get_pretrained_path(am_full_name, am_tag)
            self.am_res_path = am_res_path
            self.am_config = os.path.join(am_res_path,
                                          pretrained_models[am_full_name][am_tag]['config'])
            self.am_ckpt = os.path.join(am_res_path,
                                        pretrained_models[am_full_name][am_tag]['ckpt'])
            self.am_stat = os.path.join(
                am_res_path, pretrained_models[am_full_name][am_tag]['speech_stats'])
            # must have phones_dict in acoustic
            self.phones_dict = os.path.join(
                am_res_path, pretrained_models[am_full_name][am_tag]['phones_dict'])
            print("self.phones_dict:", self.phones_dict)
            print("self.phones_dict:{}".format(self.phones_dict))
            print(am_res_path)
            print(self.am_config)
            print(self.am_ckpt)
        else:
            self.am_config = os.path.abspath(am_config)
            self.am_ckpt = os.path.abspath(am_ckpt)
            self.am_stat = os.path.abspath(am_stat)
            self.phones_dict = os.path.abspath(phones_dict)
            self.am_res_path = os.path.dirname(os.path.abspath(self.am_config))
        # print("self.phones_dict:", self.phones_dict)
        print("self.phones_dict:{}".format(self.phones_dict))

        # for speedyspeech
        self.tones_dict = None
        if 'tones_dict' in pretrained_models[am_full_name][am_tag]:
            self.tones_dict = os.path.join(
                am_res_path, pretrained_models[am_full_name][am_tag]['tones_dict'])
            if tones_dict:
                self.tones_dict = tones_dict

        # for multi speaker fastspeech2
        self.speaker_dict = None
        if 'speaker_dict' in pretrained_models[am_full_name][am_tag]:
            self.speaker_dict = os.path.join(
                am_res_path, pretrained_models[am_full_name][am_tag]['speaker_dict'])
            if speaker_dict:
                self.speaker_dict = speaker_dict

        # voc
        if lang == "mix":
            voc_full_name = voc + '-' + 'zh'
        else:
            voc_full_name = voc + '-' + lang
        if voc_ckpt is None or voc_config is None or voc_stat is None:
            voc_res_path = self._get_pretrained_path(voc_full_name, voc_tag)
            self.voc_res_path = voc_res_path
            self.voc_config = os.path.join(voc_res_path,
                                           pretrained_models[voc_full_name][voc_tag]['config'])
            self.voc_ckpt = os.path.join(voc_res_path,
                                         pretrained_models[voc_full_name][voc_tag]['ckpt'])
            self.voc_stat = os.path.join(
                voc_res_path, pretrained_models[voc_full_name][voc_tag]['speech_stats'])
            print(voc_res_path)
            print(self.voc_config)
            print(self.voc_ckpt)
        else:
            self.voc_config = os.path.abspath(voc_config)
            self.voc_ckpt = os.path.abspath(voc_ckpt)
            self.voc_stat = os.path.abspath(voc_stat)
            self.voc_res_path = os.path.dirname(
                os.path.abspath(self.voc_config))

        # Init body.
        with open(self.am_config) as f:
            self.am_config = CfgNode(yaml.safe_load(f))
        with open(self.voc_config) as f:
            self.voc_config = CfgNode(yaml.safe_load(f))

        with open(self.phones_dict, "r") as f:
            phn_id = [line.strip().split() for line in f.readlines()]
        vocab_size = len(phn_id)
        # print("vocab_size:", vocab_size)
        print("vocab_size:{}".format(vocab_size))

        tone_size = None
        if self.tones_dict:
            with open(self.tones_dict, "r") as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            tone_size = len(tone_id)
            # print("tone_size:", tone_size)
            print("tone_size:{}".format(tone_size))

        spk_num = None
        if self.speaker_dict:
            with open(self.speaker_dict, 'rt') as f:
                spk_id = [line.strip().split() for line in f.readlines()]
            spk_num = len(spk_id)
            print("spk_num:{}".format(spk_num))

        # frontend
        if lang == 'zh':
            self.frontend = Frontend(
                g2p_model=front,
                phone_vocab_path=self.phones_dict,
                tone_vocab_path=self.tones_dict)

        elif lang == 'en':
            self.frontend = English(phone_vocab_path=self.phones_dict)
        elif lang == 'mix':
            self.frontend = MixFrontend(
                phone_vocab_path=self.phones_dict, tone_vocab_path=self.tones_dict)

        print("frontend done!")

        # acoustic model
        odim = self.am_config.n_mels
        # model: {model_name}_{dataset}
        am_name = am[:am.rindex('_')]

        am_class = dynamic_import(am_name, model_alias)
        am_inference_class = dynamic_import(am_name + '_inference', model_alias)

        if am_name == 'fastspeech2':
            am = am_class(
                idim=vocab_size,
                odim=odim,
                spk_num=spk_num,
                **self.am_config["model"])
        elif am_name == 'speedyspeech':
            am = am_class(
                vocab_size=vocab_size,
                tone_size=tone_size,
                **self.am_config["model"])
        elif am_name == 'tacotron2':
            am = am_class(idim=vocab_size, odim=odim, **self.am_config["model"])

        am.set_state_dict(paddle.load(self.am_ckpt)["main_params"])
        am.eval()
        am_mu, am_std = np.load(self.am_stat)
        am_mu = paddle.to_tensor(am_mu)
        am_std = paddle.to_tensor(am_std)
        am_normalizer = ZScore(am_mu, am_std)
        # StyleFastSpeech_inference
        if am_name == 'fastspeech2':
            pitch_stats = os.path.join(MODEL_HOME, "other", "pitch_stats.npy")
            energy_stats = os.path.join(MODEL_HOME, "other", 'energy_stats.npy')
            self.am_inference = am_inference_class(am_normalizer, am, pitch_stats, energy_stats)
        else:
            self.am_inference = am_inference_class(am_normalizer, am)
        self.am_inference.eval()
        # print("acoustic model done!")
        print("acoustic model done!")

        # vocoder
        # model: {model_name}_{dataset}
        voc_name = voc[:voc.rindex('_')]
        voc_class = dynamic_import(voc_name, model_alias)
        voc_inference_class = dynamic_import(voc_name + '_inference',
                                             model_alias)
        if voc_name != 'wavernn':
            voc = voc_class(**self.voc_config["generator_params"])
            voc.set_state_dict(paddle.load(self.voc_ckpt)["generator_params"])
            voc.remove_weight_norm()
            voc.eval()
        else:
            voc = voc_class(**self.voc_config["model"])
            voc.set_state_dict(paddle.load(self.voc_ckpt)["main_params"])
            voc.eval()
        voc_mu, voc_std = np.load(self.voc_stat)
        voc_mu = paddle.to_tensor(voc_mu)
        voc_std = paddle.to_tensor(voc_std)
        voc_normalizer = ZScore(voc_mu, voc_std)
        self.voc_inference = voc_inference_class(voc_normalizer, voc)
        self.voc_inference.eval()
        # print("voc done!")
        print("voc done!")

    def preprocess(self, input: Any, *args, **kwargs):
        """
        Input preprocess and return paddle.Tensor stored in self._inputs.
        Input content can be a text(tts), a file(asr, cls), a stream(not supported yet) or anything needed.

        Args:
            input (Any): Input text/file/stream or other content.
        """
        pass

    @paddle.no_grad()
    def infer(self,
              text: str,
              lang: str='zh',
              am: str='fastspeech2_csmsc',
              spk_id: int=0,
              speed_degree: float = 1,
              pitch_degree: float = 1,
              energy_degree: float = 1,
              robot: bool = False):
        """
        Model inference and result stored in self.output.
        """
        am_name = am[:am.rindex('_')]
        am_dataset = am[am.rindex('_') + 1:]
        get_tone_ids = False
        merge_sentences = False
        if am_name == 'speedyspeech':
            get_tone_ids = True
        if lang == 'zh':
            input_ids = self.frontend.get_input_ids(
                text,
                merge_sentences=merge_sentences,
                get_tone_ids=get_tone_ids)
            phone_ids = input_ids["phone_ids"]
            if get_tone_ids:
                tone_ids = input_ids["tone_ids"]
        elif lang == 'en':
            input_ids = self.frontend.get_input_ids(
                text, merge_sentences=merge_sentences)
            phone_ids = input_ids["phone_ids"]
        elif lang == 'mix':
            input_ids = self.frontend.get_input_ids(
                text, merge_sentences=merge_sentences)
            phone_ids = input_ids["phone_ids"]
        else:
            raise ValueError("lang should in {'zh', 'en', 'mix'}!")

        flags = 0
        for i in range(len(phone_ids)):
            part_phone_ids = phone_ids[i]
            # print("am_name: ", am_name)
            # am
            if am_name == 'speedyspeech':
                part_tone_ids = tone_ids[i]
                mel = self.am_inference(part_phone_ids, part_tone_ids)
            elif am_name == 'tacotron2':
                if am_dataset in {"aishell3", "vctk"}:
                    mel = self.am_inference(
                        part_phone_ids, spk_id=paddle.to_tensor(spk_id))
                else:
                    mel = self.am_inference(part_phone_ids)
            # fastspeech2
            else:
                # multi speaker
                durations = None
                durations_scale = 1 / speed_degree
                durations_bias = None
                pitch = None
                pitch_scale = pitch_degree
                pitch_bias = None
                energy = None
                energy_scale = energy_degree
                energy_bias = None
                mel = self.am_inference(part_phone_ids,
                                        spk_id=paddle.to_tensor(spk_id),
                                        durations=durations,
                                        durations_scale=durations_scale,
                                        durations_bias=durations_bias,
                                        pitch=pitch,
                                        pitch_scale=pitch_scale,
                                        pitch_bias=pitch_bias,
                                        energy=energy,
                                        energy_scale=energy_scale,
                                        energy_bias=energy_bias,
                                        robot=robot)
            # voc
            wav = self.voc_inference(mel)
            if flags == 0:
                wav_all = wav
                flags = 1
            else:
                wav_all = paddle.concat([wav_all, wav])
        self._outputs['wav'] = wav_all

    def postprocess(self, output: str='output.wav') -> Union[str, os.PathLike]:
        """
        Output postprocess and return results.
        This method get model output from self._outputs and convert it into human-readable results.

        Returns:
            Union[str, os.PathLike]: Human-readable results such as texts and audio files.
        """
        output = os.path.abspath(os.path.expanduser(output))
        sf.write(
            output, self._outputs['wav'].numpy(), samplerate=self.am_config.fs)
        return output

    @stats_wrapper
    def __call__(self,
                 text: str,
                 spk_id: int = 0,
                 output: str = 'output.wav',
                 speed_degree: float = 1.0,
                 pitch_degree: float = 1.0,
                 energy_degree: float = 1.0,
                 robot: bool = False
                 ):
        """
        Python API to call an executor.
        """
        self.infer(text=text,
                   lang=self.lang,
                   am=self.am,
                   spk_id=spk_id,
                   speed_degree=speed_degree,
                   pitch_degree=pitch_degree,
                   energy_degree=energy_degree,
                   robot=robot)
        res = self.postprocess(output=output)
        return res
