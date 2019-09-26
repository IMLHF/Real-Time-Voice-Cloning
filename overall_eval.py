import argparse
import os
import re
import numpy as np
import soundfile as sf
from encoder import inference as encoder_infer
from synthesizer.textnorm import get_pinyin
from synthesizer import inference as syn_infer
from encoder import audio as encoder_audio
from synthesizer import audio
from functools import partial
import pypinyin
from synthesizer.hparams import hparams


def run_eval_part1(args):
    speaker_enc_ckpt = args.speaker_encoder_checkpoint
    syn_ckpt = args.syn_checkpoint
    speaker_name = args.speaker_name
    eval_results_dir = os.path.join(args.eval_results_dir,
                                    speaker_name)
    if not os.path.exists(eval_results_dir):
        os.makedirs(eval_results_dir)
    speaker_audio_dirs = {
        "speaker_name": ["speaker_audio_1.wav", "speaker_audio_2.wav"],
        "biaobei_speaker": ["/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000001.wav",
                            "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000002.wav",
                            "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000003.wav",
                            "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000004.wav",
                            "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000005.wav",
                            "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000006.wav",
                            "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000007.wav",
                            ],
        "SLR68_DEV_3756_22": ["/home/zhangwenbo5/lihongfeng/corpus/SLR68/dev/37_5622/37_5622_20170913203118.wav",
                              "/home/zhangwenbo5/lihongfeng/corpus/SLR68/dev/37_5622/37_5622_20170913203322.wav",
                              "/home/zhangwenbo5/lihongfeng/corpus/SLR68/dev/37_5622/37_5622_20170913203824.wav"],
        "SLR38_P00001A": ["/home/zhangwenbo5/lihongfeng/corpus/SLR38/ST-CMDS-20170001_1-OS/20170001P00001A0001.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/SLR38/ST-CMDS-20170001_1-OS/20170001P00001A0002.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/SLR38/ST-CMDS-20170001_1-OS/20170001P00001A0003.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/SLR38/ST-CMDS-20170001_1-OS/20170001P00001A0004.wav",],
        "aishell_C0002": ["/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0001.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0002.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0003.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0004.wav",],
        "aishell_C0896": ["/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0001.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0002.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0003.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0004.wav",],
    }[speaker_name]
    sentences = [
        # '美国主持人听到“中国”就插话',
        # '勉励乡亲们为过上更加幸福美好的生活继续团结奋斗。',
        # '中国基建领域又来了一款“神器”, 哪里不平平哪里',
        # '违反中央八项规定精神和廉洁纪律，违规出入私人会所和打高尔夫球',
        # '陪审团未能就其盗窃和藏匿文物罪名作出裁决',
        # '于美国首都华盛顿国家记者俱乐部召开的新闻发布会上说',
        # '杭州市卫健委某直属单位一名拟提副处级干部刘某公示期间，纪检监察组照例对其个人重大事项进行抽查',
        # '我国森林面积、森林蓄积分别增长一倍左右，人工林面积居全球第一',
        # '打打打打打打打打打打打',
        # '卡尔普陪外孙玩滑梯。',
        # '假语村言，别再拥抱我。',
        # '宝马配挂跛骡鞍，貂蝉怨枕董翁榻。',
        # '中国地震台网速报,'
        # '中国地震台网正式测定,',
        # '06月04日17时46分在台湾台东县海域（北纬22.82度，东经121.75度）发生5.8级地震',
        # '中国地震台网速报，中国地震台网正式测定：06月04日17时46分在台湾台东县海域（北纬22.82度，东经121.75度）发生5.8级地震',
        # '震源深度9千米，震中位于海中，距台湾岛最近约47公里。',
        # '刚刚,台湾发生5.8级地震,与此同时,泉州厦门漳州震感明显,',
        # '此次台湾地震发生后,许多网友为同胞祈福,愿平安,',
        '新世界百货望京店',
        '全聚德烤鸭店王府井店',
        '麻烦帮我把空调温度调整到二十四',
        '请帮我显示中央一套', # aishell IC0896W0001.wav
        '确定下载三帝狂野飙车', # aishell IC0896W0002.wav
        '请帮我开启深圳卫视国际频道', # aishell IC0896W0003.wav
        '您吃饭了吗,我今天吃的太撑了',
        '您吃饭了吗？',
        '你多大了，你到底多大了，我猜你三十了，他多大了，他到底多大了，他猜你三十了',
        '二毛你今天沒课嘛还和李霞聊天',
    ]

    text2pinyin = partial(get_pinyin, std=True, pb=True)
    sentences = [' '.join(text2pinyin(sent)) for sent in sentences]

    print('eval part1> model: %s.' % syn_ckpt)
    syner = syn_infer.Synthesizer(syn_ckpt)
    encoder_infer.load_model(speaker_enc_ckpt)

    ckpt_step = re.compile(r'.*?\.ckpt\-([0-9]+)').match(syn_ckpt)
    ckpt_step = "step-"+str(ckpt_step.group(1)) if ckpt_step else syn_ckpt

    speaker_audio_wav_list = [encoder_audio.preprocess_wav(wav_dir) for wav_dir in speaker_audio_dirs]
    speaker_audio_wav = np.concatenate(speaker_audio_wav_list)
    print(os.path.join(eval_results_dir, '000_refer_speaker_audio.wav'))
    audio.save_wav(speaker_audio_wav, os.path.join(eval_results_dir, '000_refer_speaker_audio.wav'), 
                   hparams.sample_rate)
    speaker_embed = encoder_infer.embed_utterance(speaker_audio_wav)
    for i, text in enumerate(sentences):
        path = os.path.join(eval_results_dir, 
                            "%s-eval-%03d.wav" % (ckpt_step, i))
        print('[{:<10}]: {}'.format('processing', path))
        mel_spec = syner.synthesize_spectrograms([text], [speaker_embed])[0] # batch synthesize
        print('[{:<10}]:'.format('text:'), text)
        # print(np.shape(mel_spec))
        wav = syner.griffin_lim(mel_spec)
        audio.save_wav(wav, path, hparams.sample_rate)
        

def main():
    os.environ['CUDA_VISIBLE_DEVICES']= ''
    parser = argparse.ArgumentParser()
    parser.add_argument('syn_checkpoint',
                        # required=True,
                        help='Path to synthesizer model checkpoint.')
    parser.add_argument('speaker_name',
                        help='Path to target speaker audio.')
    parser.add_argument('--speaker_encoder_checkpoint', default='encoder/saved_models/pretrained.pt',
                        help='Path to speaker encoder nodel checkpoint.')
    parser.add_argument('--eval_results_dir', default='overall_eval_results',
                        help='Overall evaluation results will be saved here.')
    args = parser.parse_args()
    hparams.set_hparam("tacotron_num_gpus", 1) # set tacotron_num_gpus=1 to synthesizer single wav.
    run_eval_part1(args)


if __name__ == '__main__':
    main()


# python3 overall_eval.py ckpt_dir biaobei_speaker
