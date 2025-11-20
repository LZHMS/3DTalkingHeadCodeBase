#!/bin/bash

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

# step 1: merge the collected txt files into one json file
#python data_explore.py --mode build-json --data-dir json/source_txt --output-dir json
#python data_explore.py --mode convert-av1 --data-dir output
python data_explore.py --mode analyze --data-dir TalkScene

# step 2: download the video clips listed in the merged json file
# python download_clips.py --input_json_path json/builded_Lecture_Speech.json \
#   --output_dir output_clip \
#   --workers 8 --cleanup --full_download

# python download_video.py --input json/builded_Lecture_Speech.json \
#   --output output_HP \
#   --workers 8

# step 3: segment the video clips into short clips based on the timestamps in the json file
# python scene_segmentation.py --data-dir output \
#   --output output_clips \
#   --num-workers 0 --max-threads 48 --threshold 7\
#   --clip-style all --use-fixed-duration False --skip-existing