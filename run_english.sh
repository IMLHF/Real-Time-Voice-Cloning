if [ -z "$1" ]; then
  echo "Need a step param:"
  echo "encoder_preprocess_train"
  echo "encoder_preprocess_test"
  echo "encoder_train"
  echo "synthesizer_preprocess_audio"
  echo "synthesizer_preprocess_embeds"
  echo "synthesizer_train"
  echo "vocoder_preprocess"
  echo "vocoder_train"
  exit -1
fi

GPU_DEVICES="0"
step=$1

if [ "$step" = "encoder_preprocess_train" ]; then
    python3.5 encoder_preprocess.py \
            --datasets_root=/home/zhangwenbo5/lihongfeng/english_voice_clone/alldata \
            --datasets=voxceleb1,voxceleb2,librispeech_other \
            --out_dir=encoder_train \
            2>&1 | tee -a log_lhf/encoder_preprocess.log
elif [ "$step" = "encoder_preprocess_test" ]; then
    python3.5 encoder_preprocess.py \
            --datasets_root=/home/zhangwenbo5/lihongfeng/english_voice_clone/alldata \
            --datasets=librispeech_test \
            --out_dir=encoder_test \
            2>&1 | tee -a log_lhf/encoder_preprocess.log
elif [ "$step" = "encoder_train" ]; then # pytorch
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 encoder_train.py \
            --train_data_root=/home/zhangwenbo5/lihongfeng/english_voice_clone/alldata/SV2TTS/encoder_train \
            --test_data_root=/home/zhangwenbo5/lihongfeng/english_voice_clone/alldata/SV2TTS/encoder_test \
            2>&1 | tee -a log_lhf/encoder_train.log
elif [ "$step" = "synthesizer_preprocess_audio" ]; then
    python3.5 synthesizer_preprocess_audio.py /home/zhangwenbo5/lihongfeng/english_voice_clone/alldata LibriSpeech --n_processes=56 \
                                              2>&1 | tee -a log_lhf/synthesizer_preprocess_audio.log
elif [ "$step" = "synthesizer_preprocess_embeds" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 synthesizer_preprocess_embeds.py /home/zhangwenbo5/lihongfeng/english_voice_clone/alldata \
                                                                                 2>&1 | tee -a log_lhf/synthesizer_preprocess_embeds.log
elif [ "$step" = "synthesizer_train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 synthesizer_train.py synthesizer /home/zhangwenbo5/lihongfeng/english_voice_clone/alldata \
                                                                     2>&1 | tee -a log_lhf/synthesizer_train.log
elif [ "$step" = "vocoder_preprocess" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 vocoder_preprocess.py /home/zhangwenbo5/lihongfeng/english_voice_clone/alldata \
                                                                      2>&1 | tee -a log_lhf/vocoder_preprocess.log
elif [ "$step" = "vocoder_train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 vocoder_train.py vocoder /home/zhangwenbo5/lihongfeng/english_voice_clone/alldata \
                                                                 2>&1 | tee -a log_lhf/vocoder_train.log
else
    echo "step param error." && exit -1
fi
