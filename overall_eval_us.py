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
      "vctk_p225": ["/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p225/p225_001.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p225/p225_002.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p225/p225_003.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p225/p225_004.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p225/p225_005.wav",
                    ],
      "vctk_p226": ["/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p226/p226_001.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p226/p226_002.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p226/p226_003.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p226/p226_004.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p226/p226_005.wav",
                    ],
      "vctk_p227": ["/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p227/p227_001.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p227/p227_002.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p227/p227_003.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p227/p227_004.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p227/p227_005.wav",
                    ],
      "vctk_p228": ["/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p228/p228_001.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p228/p228_002.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p228/p228_003.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p228/p228_004.wav",
                    "/home/zhangwenbo5/lihongfeng/corpus/vctk_dataset/wav16/p228/p228_005.wav",
                    ],
      "biaobei_speaker": ["/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000001.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000002.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000003.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000004.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000005.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000006.wav",
                          "/home/zhangwenbo5/lihongfeng/corpus/BZNSYP/wavs/000007.wav",
                          ],
      "aishell_C0002": ["/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0001.wav",
                        "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0002.wav",
                        "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0003.wav",
                        "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0002/IC0002W0004.wav", ],
      "aishell_C0896": ["/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0001.wav",
                        "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0002.wav",
                        "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0003.wav",
                        "/home/zhangwenbo5/lihongfeng/corpus/aishell2/data/wav/C0896/IC0896W0004.wav", ],
  }[speaker_name]
  sentences = [
    "THAT MATTER OF TROY AND ACHILLES WRATH ONE TWO THREE RATS",
    "ENDED THE QUEST OF THE HOLY GRAAL JERUSALEM A HANDFUL OF ASHES BLOWN BY THE WIND EXTINCT",
    "She can scoop these things into three red bags",
    "and we will go meet her Wednesday at the train station",
    "This was demonstrated in a laboratory experiment with rats."
  ]

  sentences = [sen.upper() for sen in sentences]

  sentences.append("This was demonstrated in a laboratory experiment with rats")

  print('eval part1> model: %s.' % syn_ckpt)
  syner = syn_infer.Synthesizer(syn_ckpt)
  encoder_infer.load_model(speaker_enc_ckpt)

  ckpt_step = re.compile(r'.*?\.ckpt\-([0-9]+)').match(syn_ckpt)
  ckpt_step = "step-"+str(ckpt_step.group(1)) if ckpt_step else syn_ckpt

  speaker_audio_wav_list = [encoder_audio.preprocess_wav(wav_dir) for wav_dir in speaker_audio_dirs]
  speaker_audio_wav = np.concatenate(speaker_audio_wav_list)
  print(os.path.join(eval_results_dir, '%s-000_refer_speaker_audio.wav' % speaker_name))
  audio.save_wav(speaker_audio_wav, os.path.join(eval_results_dir, '%s-000_refer_speaker_audio.wav' % speaker_name),
                 hparams.sample_rate)
  speaker_embed = encoder_infer.embed_utterance(speaker_audio_wav)
  for i, text in enumerate(sentences):
    path = os.path.join(eval_results_dir,
                        "%s-%s-eval-%03d.wav" % (speaker_name, ckpt_step, i))
    print('[{:<10}]: {}'.format('processing', path))
    mel_spec = syner.synthesize_spectrograms([text], [speaker_embed])[
        0]  # batch synthesize
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
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
  main()


# python3 overall_eval.py ckpt_dir biaobei_speaker
