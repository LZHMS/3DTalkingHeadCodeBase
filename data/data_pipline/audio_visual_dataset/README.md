# Audio-Visual Dataset Processing

This directory contains tools for processing audio-visual datasets, including video scene segmentation and metadata extraction.

## Overview

The main component is a video scene segmentation pipeline that automatically detects scene transitions in videos and segments them into multiple clips. This is particularly useful for creating training datasets for talking head generation and other video processing tasks.

## Features

- **Automatic Scene Detection**: Uses multiple detection algorithms (Adaptive, Histogram, Content, Hash, Threshold)
- **Flexible Export Options**: Export video clips, audio clips, or both
- **Multi-process Processing**: Parallel processing support for handling large video datasets
- **Metadata Preservation**: Automatically saves scene information including timestamps, frames, and video properties
- **Fixed Duration Mode**: Optional mode to normalize clip durations

## Components

### Scene Segmentation (`scene_segmentation.py`)

The main script for detecting and segmenting video scenes.

**Key Functions:**
- `collect_video_data_path()`: Recursively collects all video files from a directory
- `find_scenes_new()`: Detects scene transitions using PySceneDetect library
- `process_video_clips()`: Batch processes videos with scene segmentation
- `find_optimal_thread_count()`: Automatically calculates optimal thread count for parallel processing

## Usage

### Basic Usage

```bash
python scene_segmentation.py \
    --data-dir /path/to/videos \
    --output /path/to/output \
    --detector-type Histogram \
    --detector-threshold 0.085 \
    --clip-style video_only
```

### Parameters

#### Input/Output
- `--data-dir`: Directory containing raw video files (default: `output`)
- `--output`: Output directory for segmented clips (default: `output_clips`)

#### Multi-processing
- `--num-workers`: Number of worker processes (default: 0, auto-calculated)
- `--max-threads`: Maximum available threads (default: 48)
- `--threshold`: Minimum samples per process (default: 7)

#### Export Options
- `--clip-style`: Export mode
  - `none`: Only generate metadata, no clip export
  - `all`: Export both video and audio clips
  - `video_only`: Export video clips only (default)
  - `audio_only`: Export audio clips only
- `--use-fixed-duration`: Use fixed duration for clips (default: False)
- `--clip-duration`: Fixed duration in seconds (default: 5)

#### Scene Detection
- `--detector-type`: Detection algorithm
  - `Adaptive`: Adaptive content-aware detector
  - `Histogram`: Histogram-based detector (default)
  - `Content`: Content-aware detector
  - `Hash`: Hash-based detector
  - `Threshold`: Threshold-based detector
- `--detector-threshold`: Detection sensitivity threshold (default: 0.085)

### Advanced Examples

**Export both video and audio with adaptive detector:**
```bash
python scene_segmentation.py \
    --data-dir ./raw_videos \
    --output ./processed_clips \
    --detector-type Adaptive \
    --detector-threshold 3.0 \
    --clip-style all
```

**Generate only metadata without exporting clips:**
```bash
python scene_segmentation.py \
    --data-dir ./videos \
    --output ./metadata \
    --clip-style none
```

**Use fixed 10-second duration clips:**
```bash
python scene_segmentation.py \
    --data-dir ./videos \
    --output ./clips \
    --use-fixed-duration True \
    --clip-duration 10
```

## Directory Structure

### Input Structure
```
data_dir/
├── category1/
│   ├── video_category_a/
│   │   ├── video1.mp4
│   │   └── video1.m4a (optional)
│   └── video_category_b/
│       └── video2.mp4
└── category2/
    └── video_category_c/
        └── video3.mp4
```

### Output Structure
```
output/
├── category1/
│   ├── video_category_a/
│   │   └── video1/
│   │       ├── scene_1.mp4
│   │       ├── scene_2.mp4
│   │       ├── scene_info.json
│   │       └── video_scene_info.txt
│   └── video_category_b/
│       └── video2/
│           └── ...
└── category2/
    └── ...
```

## Output Files

### `scene_info.json`
JSON Lines format file containing metadata for each detected scene:
```json
{
  "id": "scene_1",
  "video-path": "path/to/scene_1.mp4",
  "audio-path": "path/to/scene_1.m4a",
  "height": 1080,
  "width": 1920,
  "fps": 30.0,
  "start-time": 0.4,
  "start-frame": 12,
  "end-time": 5.6,
  "end-frame": 168,
  "durations": "5.2s",
  "original-video": "path/to/original.mp4",
  "original-audio": "path/to/original.m4a",
  "subtitle_path": null
}
```

### `video_scene_info.txt`
Human-readable text file with scene timestamps:
```
scene 1 infos: start_time 0:0, end_time 0:5
scene 2 infos: start_time 0:5, end_time 0:12
...
```

## Dependencies

```bash
pip install opencv-python numpy tqdm scenedetect
```

**System Requirements:**
- FFmpeg (for video/audio extraction)
- Python 3.7+

## Performance Tips

1. **Multi-processing**: The script automatically calculates optimal process count. Adjust `--max-threads` based on your CPU cores.

2. **Detection Algorithm Selection**:
   - `Histogram`: Fast, good for most videos with clear scene changes
   - `Content`: More accurate but slower, better for subtle transitions
   - `Adaptive`: Best for videos with varying content types

3. **Threshold Tuning**: Lower values = more sensitive (more scenes detected), higher values = less sensitive (fewer scenes)

4. **Memory Management**: For large datasets, process in batches by organizing input directories

## Troubleshooting

**Issue**: No scenes detected
- **Solution**: Lower the `--detector-threshold` value

**Issue**: Too many small scenes
- **Solution**: Increase the `--detector-threshold` value

**Issue**: FFmpeg errors
- **Solution**: Ensure FFmpeg is installed and accessible in system PATH

**Issue**: Out of memory
- **Solution**: Reduce `--max-threads` or process videos in smaller batches

## Attribution

Modified from [TalkVid](https://github.com/FreedomIntelligence/TalkVid/blob/main/data_pipeline/1_video_rough_segmentation/video_clip.py)

## License

Please refer to the main project license.
