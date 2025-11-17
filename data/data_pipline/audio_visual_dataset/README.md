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
- **Dataset Exploration**: Tools for building subsets, analyzing statistics, merging files, and cleaning datasets

## Components

### Scene Segmentation (`scene_segmentation.py`)

The main script for detecting and segmenting video scenes.

**Key Functions:**
- `collect_video_data_path()`: Recursively collects all video files from a directory
- `find_scenes_new()`: Detects scene transitions using PySceneDetect library
- `process_video_clips()`: Batch processes videos with scene segmentation
- `find_optimal_thread_count()`: Automatically calculates optimal thread count for parallel processing

### Data Exploration (`data_explore.py`)

A comprehensive toolkit for exploring and processing audio-visual datasets.

**Key Functions:**
- `build_dataset_subset()`: Build subsets by category and duration constraints
- `analyze_dataset()`: Generate detailed statistics about dataset composition
- `merge_video_audio()`: Merge separate video and audio files
- `clean_non_mp4_files()`: Clean non-MP4 files with backup support

## Usage

### Scene Segmentation

#### Basic Usage

```bash
python scene_segmentation.py \
    --data-dir /path/to/videos \
    --output /path/to/output \
    --detector-type Histogram \
    --detector-threshold 0.085 \
    --clip-style video_only
```

#### Parameters

##### Input/Output
- `--data-dir`: Directory containing raw video files (default: `output`)
- `--output`: Output directory for segmented clips (default: `output_clips`)

##### Multi-processing
- `--num-workers`: Number of worker processes (default: 0, auto-calculated)
- `--max-threads`: Maximum available threads (default: 48)
- `--threshold`: Minimum samples per process (default: 7)

##### Export Options
- `--clip-style`: Export mode
  - `none`: Only generate metadata, no clip export
  - `all`: Export both video and audio clips
  - `video_only`: Export video clips only (default)
  - `audio_only`: Export audio clips only
- `--use-fixed-duration`: Use fixed duration for clips (default: False)
- `--clip-duration`: Fixed duration in seconds (default: 5)

##### Scene Detection
- `--detector-type`: Detection algorithm
  - `Adaptive`: Adaptive content-aware detector
  - `Histogram`: Histogram-based detector (default)
  - `Content`: Content-aware detector
  - `Hash`: Hash-based detector
  - `Threshold`: Threshold-based detector
- `--detector-threshold`: Detection sensitivity threshold (default: 0.085)

#### Advanced Examples

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

### Data Exploration

#### 1. Build Dataset Subset

Create subsets from a larger dataset based on category and duration constraints.

**Basic Usage:**
```bash
python data_explore.py \
    --mode build-subset \
    --input json/filtered_video_clips.json \
    --categories "Personal Experience" "Online Course/Lecture" \
    --max-duration 50400 \
    --output-dir json
```

**Parameters:**
- `--input`: Input JSON file containing video metadata
- `--categories`: List of categories to include (space-separated)
- `--max-duration`: Maximum total duration per category in seconds (default: 50400 = 14 hours)
- `--output-dir`: Output directory for subset JSON files (default: `json`)

**Output:**
- Creates separate JSON files for each category
- Prints detailed statistics (count, duration) per category
- Filters by English language by default

**Example:**
```bash
# Build a 10-hour subset of Personal Experience videos
python data_explore.py \
    --mode build-subset \
    --input filtered_video_clips.json \
    --categories "Personal Experience" \
    --max-duration 36000
```

#### 2. Analyze Dataset Statistics

Generate comprehensive statistics about your dataset including duration, file counts, and averages.

**Basic Usage:**
```bash
python data_explore.py \
    --mode analyze \
    --data-dir ./output \
    --output-csv dataset_statistics.csv
```

**Parameters:**
- `--data-dir`: Root directory of dataset to analyze
- `--output-csv`: Output CSV file path (default: `dataset_statistics.csv`)

**Output:**
- Console table with statistics per category
- CSV file with detailed metrics including:
  - ID count, video count per category
  - Total duration (seconds, minutes, hours)
  - Average duration per video and per ID
  - Average number of videos per ID

**Example Output:**
```
=== Dataset Statistics ===
Category              ID Count  Video Count  Total Duration(h)
Personal Experience        150         450              12.5
Online Course              80          320               8.3
Total                     230          770              20.8
```

#### 3. Merge Video and Audio Files

Merge separate video (.mp4) and audio (.m4a) files into single MP4 files.

**Basic Usage (Preview):**
```bash
python data_explore.py \
    --mode merge \
    --data-dir ./output \
    --dry-run
```

**Execute Merge:**
```bash
python data_explore.py \
    --mode merge \
    --data-dir ./output
```

**Parameters:**
- `--data-dir`: Root directory containing video/audio files
- `--dry-run`: Preview mode without actual merging (default: False)

**Features:**
- Automatically finds matching video/audio pairs
- Skips already merged files (with `_merged` suffix)
- Uses FFmpeg for lossless video copy with AAC audio encoding
- Provides detailed progress and error reporting

**Output:**
- Merged files with `_merged.mp4` suffix
- Summary statistics of merged/skipped files

#### 4. Clean Non-MP4 Files

Remove all non-MP4 files from dataset with optional backup.

**Basic Usage (Preview):**
```bash
python data_explore.py \
    --mode clean \
    --data-dir ./output \
    --dry-run
```

**Execute Cleaning:**
```bash
python data_explore.py \
    --mode clean \
    --data-dir ./output
```

**With Backup:**
```bash
python data_explore.py \
    --mode clean \
    --data-dir ./output \
    --no-backup  # Skip backup creation
```

**Parameters:**
- `--data-dir`: Root directory to clean
- `--dry-run`: Preview mode without actual deletion
- `--no-backup`: Don't create backup before deletion (default: creates backup)

**Features:**
- Creates timestamped backup directory before deletion
- Preserves directory structure in backup
- Shows file type statistics (extensions and counts)
- Excludes `logs` and `json_logs` directories

**Output:**
- Backup folder: `backup_YYYYMMDD_HHMMSS/`
- Summary of deleted file types and counts

#### Common Parameters

All modes support:
- `--data-dir`: Primary data directory path
- `--dry-run`: Preview changes without modifying files

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

### Output Structure (Scene Segmentation)
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

### `dataset_statistics.csv`
Statistical analysis output with dual-table format:
- Main table: Category counts and total durations
- Average table: Per-video and per-ID statistics

## Dependencies

```bash
pip install opencv-python numpy tqdm scenedetect pandas
```

**System Requirements:**
- FFmpeg (for video/audio extraction and merging)
- Python 3.7+

## Performance Tips

### Scene Segmentation

1. **Multi-processing**: The script automatically calculates optimal process count. Adjust `--max-threads` based on your CPU cores.

2. **Detection Algorithm Selection**:
   - `Histogram`: Fast, good for most videos with clear scene changes
   - `Content`: More accurate but slower, better for subtle transitions
   - `Adaptive`: Best for videos with varying content types

3. **Threshold Tuning**: Lower values = more sensitive (more scenes detected), higher values = less sensitive (fewer scenes)

4. **Memory Management**: For large datasets, process in batches by organizing input directories

### Data Exploration

1. **Dry-Run First**: Always use `--dry-run` to preview changes before executing
2. **Backup Important Data**: Use backup option when cleaning files
3. **Batch Processing**: For large datasets, process categories separately
4. **CSV Analysis**: Use generated CSV files with Excel/pandas for further analysis

## Troubleshooting

### Scene Segmentation

**Issue**: No scenes detected
- **Solution**: Lower the `--detector-threshold` value

**Issue**: Too many small scenes
- **Solution**: Increase the `--detector-threshold` value

**Issue**: FFmpeg errors
- **Solution**: Ensure FFmpeg is installed and accessible in system PATH

**Issue**: Out of memory
- **Solution**: Reduce `--max-threads` or process videos in smaller batches

### Data Exploration

**Issue**: Merge fails with codec errors
- **Solution**: Check FFmpeg installation and ensure video/audio codecs are supported

**Issue**: Analysis shows 0 duration
- **Solution**: Verify video files are not corrupted and OpenCV can read them

**Issue**: Backup takes too much space
- **Solution**: Use `--no-backup` option or manually clean old backups

**Issue**: Permission denied when deleting
- **Solution**: Check file permissions and ensure no files are open in other programs

## Workflow Example

Complete workflow for processing a new dataset:

```bash
# Step 1: Segment videos into scenes
python scene_segmentation.py \
    --data-dir ./raw_videos \
    --output ./output \
    --detector-type Histogram \
    --clip-style all

# Step 2: Analyze dataset statistics
python data_explore.py \
    --mode analyze \
    --data-dir ./output

# Step 3: (Optional) Merge video and audio if needed
python data_explore.py \
    --mode merge \
    --data-dir ./output \
    --dry-run  # Preview first

# Step 4: Clean unwanted files
python data_explore.py \
    --mode clean \
    --data-dir ./output \
    --dry-run  # Preview first

# Step 5: Build training subset
python data_explore.py \
    --mode build-subset \
    --input ./output/metadata.json \
    --categories "Personal Experience" \
    --max-duration 36000
```

## Attribution

Scene Segmentation modified from [TalkVid](https://github.com/FreedomIntelligence/TalkVid/blob/main/data_pipeline/1_video_rough_segmentation/video_clip.py)

## License

Please refer to the main project license.
