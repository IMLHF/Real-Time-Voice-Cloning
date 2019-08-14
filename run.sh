if [ -z "$1" ]; then
  echo "Need a step param:"
  echo "  encoder_preprocess"
  echo "  encoder_train"
  exit -1
fi

CUDA_VISIBLE_DEVICES=0

step=$1
if [ "$step" = "encoder_preprocess" ]; then
    python3 encoder_preprocess.py --datasets_root=/home/zhangwenbo5/corpus/SLR68 \
                                  --datasets=MAGICDATA_train \
                                  2>&1 | tee -a log_lhf/encoder_preprocess.log
elif [ "$step" = "encoder_train" ]; then
    python3 encoder_train.py --clean_data_root=/home/zhangwenbo5/corpus/SLR68/SV2TTS/encoder \
                                                    2>&1 | tee -a log_lhf/encoder_train.log
fi
