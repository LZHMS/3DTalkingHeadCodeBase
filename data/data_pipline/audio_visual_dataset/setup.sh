#!/bin/bash

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

# step 1: merge the collected txt files into one json file
# python data_explore.py --mode build-json --data-dir json/source_txt --output-dir json

# step 2: download the video clips listed in the merged json file
python download_clips.py --input_json_path json/builded_Lecture_Speech.json \
  --output_dir output \
  --workers 8 --cleanup --full_download