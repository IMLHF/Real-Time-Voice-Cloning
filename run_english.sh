if [ -z "$1" ]; then
  echo "Need a step param:"
  echo "  encoder_preprocess_train"
  echo "  encoder_preprocess_test"
  echo "  encoder_train"
  echo "synthesizer_preprocess_audio"
  echo "synthesizer_preprocess_embeds"
  echo "synthesizer_train"
  exit -1
fi

GPU_DEVICES="3"
step=$1

if [ "$step" = "encoder_preprocess_train" ]; then
    python3.5 encoder_preprocess.py \
            --datasets_root=/home/zhangwenbo5/lihongfeng/english_voice_clone/alldata \
            --datasets=librispeech_other \
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
# elif [ "$step" = "synthesizer_preprocess_audio" ]; then
#     python3.5 synthesizer_preprocess_audio.py /home/zhangwenbo5/corpus aishell2 --n_processes=32 \
#                                               2>&1 | tee -a log_lhf/synthesizer_preprocess_audio.log
# elif [ "$step" = "synthesizer_preprocess_embeds" ]; then
#     CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 synthesizer_preprocess_embeds.py /home/zhangwenbo5/corpus \
#                                                                                  2>&1 | tee -a log_lhf/synthesizer_preprocess_embeds.log
# elif [ "$step" = "synthesizer_train" ]; then
#     CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 synthesizer_train.py synthesizer /home/zhangwenbo5/corpus \
#                                                                      2>&1 | tee -a log_lhf/synthesizer_train.log
else
    echo "step param error." && exit -1
fi
