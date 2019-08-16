if [ -z "$1" ]; then
  echo "Need a step param:"
  echo "  encoder_preprocess"
  echo "  encoder_train"
  exit -1
fi

GPU_DEVICES="2"

step=$1
if [ "$step" = "encoder_preprocess" ]; then
    python3 encoder_preprocess.py --datasets_root=/home/zhangwenbo5/corpus \
                                  --datasets=SLR68 \
                                  2>&1 | tee -a log_lhf/encoder_preprocess.log
elif [ "$step" = "encoder_train" ]; then
    CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3 encoder_train.py --clean_data_root=/home/zhangwenbo5/corpus/SLR68/SV2TTS/encoder \
                                                               2>&1 | tee -a log_lhf/encoder_train.log
fi
