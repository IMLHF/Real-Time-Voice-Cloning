# running English synthesizer, this script cannot run now.

# if [ -z "$1" ]; then
#   echo "Need a step param:"
#   echo "  encoder_preprocess"
#   echo "  encoder_train"
#   echo "synthesizer_preprocess_audio"
#   echo "synthesizer_preprocess_embeds"
#   echo "synthesizer_train"
#   exit -1
# fi

# GPU_DEVICES="2,3"
# step=$1

# if [ "$step" = "encoder_preprocess" ]; then
#     python3.5 encoder_preprocess.py --datasets_root=/home/zhangwenbo5/corpus \
#                                     --datasets=aishell2,SLR68 \
#                                     2>&1 | tee -a log_lhf/encoder_preprocess.log
# elif [ "$step" = "encoder_train" ]; then # pytorch
#     CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 encoder_train.py --clean_data_root=/home/zhangwenbo5/corpus/SV2TTS/encoder \
#                                                                  2>&1 | tee -a log_lhf/encoder_train.log
# elif [ "$step" = "synthesizer_preprocess_audio" ]; then
#     python3.5 synthesizer_preprocess_audio.py /home/zhangwenbo5/corpus aishell2 --n_processes=32 \
#                                               2>&1 | tee -a log_lhf/synthesizer_preprocess_audio.log
# elif [ "$step" = "synthesizer_preprocess_embeds" ]; then
#     CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 synthesizer_preprocess_embeds.py /home/zhangwenbo5/corpus \
#                                                                                  2>&1 | tee -a log_lhf/synthesizer_preprocess_embeds.log
# elif [ "$step" = "synthesizer_train" ]; then
#     CUDA_VISIBLE_DEVICES=$GPU_DEVICES python3.5 synthesizer_train.py synthesizer /home/zhangwenbo5/corpus \
#                                                                      2>&1 | tee -a log_lhf/synthesizer_train.log
# else
#     echo "step param error." && exit -1
# fi
