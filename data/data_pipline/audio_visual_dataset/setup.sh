#!bin/bash

export http_proxy="http://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"

python download_clips.py --input json/online_course_lecture_video_clips.json \
  --output output \
  --workers 8

python download_clips.py --input json/personal_experience_video_clips.json \
  --output output \
  --workers 8