#!/bin/bash

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

# step 1: merge the collected txt files into one json file
# python data_explore.py --mode build-json --data-dir json/source_txt --output-dir json

# step 2: download the video clips listed in the merged json file
# python raw.py --input_json_path json/online_course_lecture_video_clips.json \
#   --output_dir output \
#   --workers 8 --cleanup

python download_video.py --input json/builded_Lecture_Speech.json \
  --output output \
  --workers 8