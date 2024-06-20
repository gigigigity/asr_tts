import os
import wave
import argparse
import numpy as np

import torch
import ChatTTS
from  IPython.lib.display import Audio 

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

chat = ChatTTS.Chat()
chat.load_models(compile=True) # 设置为Flase获得更快速度，设置为True获得更佳效果

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch',
    model_revision="v2.0.4")

def list_files_with_extensions(directory):
    wav_files = []
    txt_files = []
    other_files = []

    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            wav_files.append(filename)
        elif filename.endswith('.txt'):
            txt_files.append(filename)
        else:
            other_files.append(filename)
    
    return wav_files, txt_files, other_files


def audio_to_txt(wav_file , input_path , out_path):

    print(input_path + wav_file)
    rec_result = inference_pipeline(input_path+ '/' + wav_file)
    
    with open(out_path + '/txt/' + wav_file[:-4] + '.txt', "w") as f:  # 打开文件
        f.write(rec_result[0]['text'])


def tts(text, oral=3, laugh=3, bk=3):
    '''
    输入文本，输出音频
    '''
    rand_spk = chat.sample_random_speaker()
    # 句子全局设置：讲话人音色和速度
    params_infer_code = {
        'spk_emb': rand_spk,
    }

    # 句子全局设置：口语连接、笑声、停顿程度
    # oral：连接词，AI可能会自己加字，取值范围 0-9，比如：卡壳、嘴瓢、嗯、啊、就是之类的词。不宜调的过高。
    # laugh：笑，取值范围 0-9
    # break：停顿，取值范围 0-9
    params_refine_text = {
        'prompt': '[oral_{}][laugh_{}][break_{}]'.format(oral, laugh, bk)
    }

    wavs = chat.infer(text, params_refine_text=params_refine_text, params_infer_code=params_infer_code)

    return wavs


def vec_to_wav(pcm_vec, wav_file, framerate = 16000):
    """ 
    将numpy数组转为单通道wav文件
    :param pcm_vec: 输入的numpy向量
    :param wav_file: wav文件名
    :param framerate: 采样率
    :return:
    """
    pcm_vec = pcm_vec * 32768
    pcm_vec = pcm_vec.astype(np.int16)
    wave_out = wave.open(wav_file, 'wb')
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(framerate)
    wave_out.writeframes(pcm_vec)

def txt_to_audio(txt_file , input_path , out_path):
    
    with open(input_path + '/' + txt_file, "r") as f: 
        txt_data = f.read()  # 读取文件
        vec = tts(txt_data)
        audio = vec_to_wav(vec[0], out_path+"/wav/" + txt_file[:-4] + '.wav', 24000)


def parse_args():
    parse = argparse.ArgumentParser(description='input and output path')
    parse.add_argument('--input',default = '../input', help='input path')
    parse.add_argument('--output',default = '../output', help='output path')
    args = parse.parse_args()
    return args


def main():
    args = parse_args()
    input_path = args.input
    output_path = args.output

    wav_files , txt_files , _ = list_files_with_extensions(input_path)
    for wav in wav_files:
        audio_to_txt(wav , input_path , output_path)
    for txt in txt_files:
        txt_to_audio(txt , input_path , output_path)

if __name__ == '__main__':
    main()