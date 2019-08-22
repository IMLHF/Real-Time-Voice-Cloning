#!/usr/bin/env bash
# ./toP40.sh lpc-tacotron
if [[ -z "$1" ]]; then
  echo "Default Destination: Real-Time-Voice-Cloning"
  dest="Real-Time-Voice-Cloning"
else
  dest="$1"
fi

rm __pycache__ -rf
rm */__pycache__ -rf
rm */*/__pycache__ -rf

echo "To P40:/DATA/disk1/lihongfeng/$dest"
# exit 0
rsync -avhP -e "ssh -p 22 -o ProxyCommand='ssh -p 8695 zhangwenbo5@120.92.114.84 -W %h:%p'" --exclude-from='.vscode/exclude.lst' ./* zhangwenbo5@ksai-P40-2:/DATA/disk1/lihongfeng/$dest

# rsync -av -e 'ssh -p 15043' --exclude-from='.vscode/exclude.lst' ./* room@speaker.is99kdf.xyz:~/work/speech_synthesis/$1

# if [ "$site" == "15047" ]; then
#   echo "Copy ./nmt/* to $user@$site:~/worklhf/nmt_seq2seq_first/$2"
#   rsync -av -e 'ssh -p '$site --exclude-from='.vscode/exclude.lst' ./nmt/* $user@speaker.is99kdf.xyz:~/worklhf/nmt_seq2seq_first/$2
# else
#   echo "Copy ./nmt/* to $user@$site:~/work/nmt_seq2seq_first/$2"
#   rsync -av -e 'ssh -p '$site --exclude-from='.vscode/exclude.lst' ./nmt/* $user@speaker.is99kdf.xyz:~/work/nmt_seq2seq_first/$2
# fi

# -a ：递归到目录，即复制所有文件和子目录。另外，打开归档模式和所有其他选项（相当于 -rlptgoD）
# -v ：详细输出
# -e ssh ：使用 ssh 作为远程 shell，这样所有的东西都被加密
# --exclude='*.out' ：排除匹配模式的文件，例如 *.out 或 *.c 等。

# scp -r -P 15043 room@speaker.is99kdf.xyz:/home/room/work/paper_se_test/pc001_se/exp/rnn_speech_enhancement/nnet_C001/nnet_iter15* ./
