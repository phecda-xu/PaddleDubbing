#! /bin/bash
source /home/phecda/Code/PaddleSpeech/venv/bin/activate


text_file=$1
save_folder=$2

if [ ! -d "$save_folder/" ];then
  mkdir $save_folder
  echo "build folder: $save_folder"
fi

for line in $(cat $text_file)
do
paddlespeech tts --am fastspeech2_aishell3 --voc pwgan_aishell3 --spk_id 1 --input $line --output $save_folder/$line.wav
done

