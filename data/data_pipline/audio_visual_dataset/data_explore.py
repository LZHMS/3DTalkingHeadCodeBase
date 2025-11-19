"""
Audio-Visual Dataset Exploration and Processing Module

This module provides utilities for exploring, analyzing, and processing audio-visual datasets.
It includes functions for building data subsets, statistical analysis, merging video/audio files,
and cleaning non-MP4 files.

Main Features:
    1. Build dataset subsets by category and duration
    2. Analyze dataset statistics (duration, file count, etc.)
    3. Merge separate video and audio files
    4. Clean non-MP4 files with backup option

Usage:
    python data_explore.py --mode build-subset --input filtered_video_clips.json --categories "Personal Experience"
    python data_explore.py --mode analyze --data-dir ./output
    python data_explore.py --mode merge --data-dir ./output
    python data_explore.py --mode clean --data-dir ./output --dry-run
"""

import os
import json
import cv2
import shutil
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple
import pandas as pd

# Change relative import to handle both package and script execution
try:
    from .utils import get_video_id
except ImportError:
    from utils import get_video_id


def build_dataset_subset(input_json: str, categories: List[str], max_duration: float, 
                        output_dir: str = 'json') -> None:
    """
    Build dataset subsets from TalkVid dataset by category and duration.
    
    Filters videos by specified categories and language, collecting up to max_duration
    of content per category. Saves subset metadata to JSON files.
    
    Args:
        input_json (str): Path to input JSON file containing video metadata
        categories (List[str]): List of video categories to include
        max_duration (float): Maximum total duration in seconds per category
        output_dir (str): Output directory for subset JSON files (default: 'json')
    
    Returns:
        None
    
    Examples:
        >>> build_dataset_subset('filtered_video_clips.json', 
        ...                      ['Personal Experience'], 14*3600, 'json')
    """
    # Load JSON data
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for cate in categories:
        # Initialize statistics
        category_stats = defaultdict(lambda: {'count': 0, 'total_duration': 0})
        mini_dataset = []
        
        for item in data:
            # Filter by category and language
            if 'info' in item and 'Video Category' in item['info']:
                category = item['info']['Video Category']
                language = item['info'].get('Language', '')
                
                if category != cate or language != 'English':
                    continue
                
                # Normalize category name
                if category == "Online Course/Lecture":
                    item['info']['Video Category'] = "Online Course"
                    category = "Online Course"
                
                # Check duration limit
                if category_stats[category]['total_duration'] > max_duration:
                    continue
                
                mini_dataset.append(item)
                category_stats[category]['count'] += 1
                
                # Calculate total duration
                if 'durations' in item:
                    if isinstance(item['durations'], list):
                        total_duration = sum(float(d[:-1]) for d in item['durations'])
                    else:
                        total_duration = float(item['durations'][:-1])
                    category_stats[category]['total_duration'] += total_duration
        
        # Save subset to JSON
        save_file_name = os.path.join(output_dir, 
                                     f'{cate.lower().replace("/", "_").replace(" ", "_")}_video_clips.json')
        with open(save_file_name, 'w', encoding='utf-8') as f:
            json.dump(mini_dataset, f, ensure_ascii=False, indent=4)
        
        # Print statistics
        print(f"\n{'='*80}")
        print(f"Category: {cate}")
        print(f"{'='*80}")
        print(f"{'Category':<30} {'Count':<10} {'Duration(s)':<15} {'Duration(m)':<15}")
        print(f"{'-'*80}")
        
        for category, stats in sorted(category_stats.items()):
            count = stats['count']
            duration_seconds = stats['total_duration']
            duration_minutes = duration_seconds / 60
            print(f"{category:<30} {count:<10} {duration_seconds:<15.2f} {duration_minutes:<15.2f}")
        
        data_scale = sum(stats['total_duration'] for stats in category_stats.values())
        print(f"{'='*80}")
        print(f"Total categories: {len(category_stats)}")
        print(f"Total items: {sum(stats['count'] for stats in category_stats.values())}")
        print(f"Total duration: {data_scale:.2f}s / {data_scale/60:.2f}m / {data_scale/3600:.2f}h")
        print(f"Saved to: {save_file_name}\n")


def get_video_duration(video_path: Path) -> float:
    """
    Get video duration in seconds using OpenCV.
    
    Args:
        video_path (Path): Path to video file
    
    Returns:
        float: Duration in seconds, 0 if error occurs
    
    Examples:
        >>> get_video_duration(Path('video.mp4'))
        120.5
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0:
            return frame_count / fps
        return 0
    except Exception as e:
        print(f"Error reading {video_path}: {e}")
        return 0


def analyze_dataset(data_root: str, output_csv: str = 'dataset_statistics.csv') -> Dict:
    """
    Analyze dataset statistics including duration, video count, and ID count.
    
    Traverses dataset directory structure, calculates total duration and counts
    for each category. Generates detailed statistics table and saves to CSV.
    
    Args:
        data_root (str): Root directory of dataset
        output_csv (str): Output CSV file path (default: 'dataset_statistics.csv')
    
    Returns:
        Dict: Statistics dictionary with category-level aggregations
    
    Directory Structure:
        data_root/
        ├── category1/
        │   ├── id_001/
        │   │   ├── video1.mp4
        │   │   └── video2.mp4
        │   └── id_002/
        └── category2/
    
    Examples:
        >>> stats = analyze_dataset('./output')
        >>> print(stats['Personal Experience']['video_count'])
    """
    data_root = Path(data_root)
    
    # Initialize statistics storage
    stats = defaultdict(lambda: {'ids': set(), 'total_duration': 0, 'video_count': 0})
    
    # Supported video formats
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    
    # Traverse dataset
    if not data_root.exists():
        print(f"Data directory does not exist: {data_root}")
        return None
    
    for category_dir in sorted(data_root.iterdir()):
        if not category_dir.is_dir():
            continue
        
        category = category_dir.name
        print(f"Processing category: {category}")
        
        for id_dir in category_dir.iterdir():
            if not id_dir.is_dir():
                continue
            
            id_name = id_dir.name
            stats[category]['ids'].add(id_name)
            
            # Count all videos under this ID
            for video_file in id_dir.iterdir():
                if video_file.suffix.lower() in video_extensions:
                    duration = get_video_duration(video_file)
                    stats[category]['total_duration'] += duration
                    stats[category]['video_count'] += 1
                    print(f"  - {category}/{id_name}/{video_file.name}: {duration:.2f}s")
    
    # Generate statistics tables
    if stats:
        results = []
        for category, info in sorted(stats.items()):
            results.append({
                'Category': category,
                'ID Count': len(info['ids']),
                'Video Count': info['video_count'],
                'Total Duration(s)': round(info['total_duration'], 2),
                'Total Duration(m)': round(info['total_duration'] / 60, 2),
                'Total Duration(h)': round(info['total_duration'] / 3600, 2)
            })
        
        df = pd.DataFrame(results)
        
        # Add total row
        total_row = {
            'Category': 'Total',
            'ID Count': df['ID Count'].sum(),
            'Video Count': df['Video Count'].sum(),
            'Total Duration(s)': round(df['Total Duration(s)'].sum(), 2),
            'Total Duration(m)': round(df['Total Duration(m)'].sum(), 2),
            'Total Duration(h)': round(df['Total Duration(h)'].sum(), 2)
        }
        df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
        
        print("\n=== Dataset Statistics ===")
        print(df.to_string(index=False))
        
        # Create average statistics table
        avg_results = []
        for category, info in sorted(stats.items()):
            avg_duration_per_video = info['total_duration'] / info['video_count'] if info['video_count'] > 0 else 0
            avg_duration_per_id = info['total_duration'] / len(info['ids']) if len(info['ids']) > 0 else 0
            avg_results.append({
                'Category': category,
                'Avg Duration/Video(s)': round(avg_duration_per_video, 2),
                'Avg Duration/ID(m)': round(avg_duration_per_id / 60, 2),
                'Avg Videos/ID': round(info['video_count'] / len(info['ids']), 2) if len(info['ids']) > 0 else 0
            })
        
        df2 = pd.DataFrame(avg_results)
        
        # Create separator columns
        empty_cols = pd.DataFrame({'': [''] * len(df), ' ': [''] * len(df), '  ': [''] * len(df)})
        
        # Combine tables horizontally
        combined_df = pd.concat([df, empty_cols, df2], axis=1)
        
        # Save to CSV
        combined_df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"\nStatistics saved to {output_csv}")
    else:
        print("No data found or analysis failed")
    
    return stats


def merge_video_audio(data_dir: str, dry_run: bool = True) -> Tuple[int, int]:
    """
    Merge separate video and audio files into single MP4 files.
    
    Searches for .mp4 video files and corresponding .m4a audio files,
    then merges them using FFmpeg. Skips files that are already merged.
    
    Args:
        data_dir (str): Root directory containing video/audio files
        dry_run (bool): If True, only preview without actual merging (default: True)
    
    Returns:
        Tuple[int, int]: (merged_count, skipped_count)
    
    File Structure:
        data_dir/
        └── category/
            └── id/
                ├── scene_1.mp4
                ├── scene_1.m4a
                └── scene_1_merged.mp4 (output)
    
    Examples:
        >>> merged, skipped = merge_video_audio('./output', dry_run=False)
        >>> print(f"Merged {merged} files, skipped {skipped}")
    """
    output_dir = Path(data_dir)
    merged_count = 0
    skipped_count = 0
    
    # Traverse all category folders
    for class_folder in output_dir.iterdir():
        if class_folder.is_dir() and class_folder.name not in ['logs', 'json_logs']:
            # Traverse each ID folder
            for id_folder in class_folder.iterdir():
                if id_folder.is_dir():
                    # Find all mp4 video files
                    for video_file in id_folder.glob("*.mp4"):
                        # Skip already merged files
                        if '_merged' in video_file.stem:
                            continue
                        
                        # Build corresponding audio filename
                        audio_file = video_file.with_suffix('.m4a')
                        
                        # Check if audio file exists
                        if not audio_file.exists():
                            print(f"Skip: {video_file.name} - Audio file not found")
                            skipped_count += 1
                            continue
                        
                        # Build merged filename
                        merged_file = video_file.with_name(f"{video_file.stem}_merged.mp4")
                        
                        # Skip if merged file already exists
                        if merged_file.exists():
                            print(f"Exists: {merged_file.name}")
                            continue
                        
                        if dry_run:
                            print(f"[Preview] Will merge: {video_file.name} + {audio_file.name} -> {merged_file.name}")
                            merged_count += 1
                            continue
                        
                        # Use ffmpeg to merge video and audio
                        try:
                            cmd = [
                                'ffmpeg',
                                '-i', str(video_file),
                                '-i', str(audio_file),
                                '-c:v', 'copy',
                                '-c:a', 'aac',
                                '-strict', 'experimental',
                                '-y',  # Overwrite output file
                                str(merged_file)
                            ]
                            
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            
                            if result.returncode == 0:
                                print(f"Successfully merged: {video_file.name} -> {merged_file.name}")
                                merged_count += 1
                            else:
                                print(f"Merge failed: {video_file.name} - {result.stderr}")
                                skipped_count += 1
                        
                        except Exception as e:
                            print(f"Processing error {video_file.name}: {e}")
                            skipped_count += 1
    
    print(f"\n=== Merge {'Preview' if dry_run else 'Complete'} ===")
    print(f"{'Will merge' if dry_run else 'Merged'}: {merged_count} files")
    print(f"Skipped: {skipped_count} files")
    
    if dry_run:
        print("\n⚠️ Preview mode - no files were actually merged")
        print("To execute merge, run with --no-dry-run")
    
    return merged_count, skipped_count


def clean_non_mp4_files(data_dir: str, create_backup: bool = True, 
                       dry_run: bool = True) -> Tuple[int, List[str]]:
    """
    Clean non-MP4 files from dataset with optional backup.
    
    Removes all files that are not .mp4 format. Optionally creates backup
    before deletion and supports dry-run mode for preview.
    
    Args:
        data_dir (str): Root directory to clean
        create_backup (bool): Whether to create backup before deletion (default: True)
        dry_run (bool): If True, only preview without actual deletion (default: True)
    
    Returns:
        Tuple[int, List[str]]: (deleted_count, list of deleted file paths)
    
    Examples:
        >>> count, files = clean_non_mp4_files('./output', create_backup=True, dry_run=False)
        >>> print(f"Deleted {count} files")
    """
    output_dir = Path(data_dir)
    deleted_count = 0
    deleted_files = []
    
    # Create backup directory
    backup_dir = None
    if create_backup and not dry_run:
        backup_dir = Path(f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        backup_dir.mkdir(exist_ok=True)
        print(f"Backup directory: {backup_dir}\n")
    
    # Traverse all category folders
    for class_folder in output_dir.iterdir():
        if class_folder.is_dir() and class_folder.name not in ['logs', 'json_logs']:
            # Traverse each video folder
            for video_folder in class_folder.iterdir():
                if video_folder.is_dir():
                    # Traverse all files in folder
                    for file in video_folder.iterdir():
                        if file.is_file() and file.suffix.lower() != '.mp4':
                            deleted_files.append(str(file))
                            
                            # Backup file
                            if create_backup and not dry_run and backup_dir:
                                backup_path = backup_dir / file.relative_to(output_dir)
                                backup_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(file, backup_path)
                            
                            # Delete file
                            if dry_run:
                                print(f"[Preview] Will delete: {file.relative_to(output_dir)}")
                            else:
                                try:
                                    file.unlink()
                                    deleted_count += 1
                                    print(f"Deleted: {file.relative_to(output_dir)}")
                                except Exception as e:
                                    print(f"Delete failed {file.name}: {e}")
    
    print(f"\n=== {'Preview' if dry_run else 'Cleaning'} Complete ===")
    if dry_run:
        print(f"Found {len(deleted_files)} non-MP4 files")
        print("\n⚠️ Preview mode - no files were actually deleted")
        print("To execute deletion, run with --no-dry-run")
    else:
        print(f"Deleted {deleted_count} non-MP4 files")
        if create_backup and backup_dir:
            print(f"Backup location: {backup_dir}")
    
    # Print file type statistics
    if len(deleted_files) > 0:
        print(f"\nDeleted file type statistics:")
        extensions = {}
        for file_path in deleted_files:
            ext = Path(file_path).suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        for ext, count in sorted(extensions.items()):
            print(f"  {ext if ext else '(no extension)'}: {count} files")
    
    return deleted_count, deleted_files


def txt_to_json(src, json_path: str) -> None:
    """
    Convert text files with video URLs to a structured JSON format.
    Args:
        src (str): Source directory containing text files
        json_path (str): Output JSON file path
    Returns:
        None
    """

    json_info, cates = [], ''
    for file in os.listdir(src):
        if not file.endswith('.txt'):
            continue
        
        category = os.path.splitext(file)[0]
        cates += f"_{category}"
        with open(os.path.join(src, file), 'r', encoding='utf-8') as f_txt:
            lines = f_txt.readlines()
        for line in lines:
            url = line.strip()
            if url == '':
                continue
            item = {
                "id": get_video_id(url),
                "video link": url,
                "video category": category
            }
            json_info.append(item)
    with open(os.path.join(json_path, f"builded{cates}.json"), 'w', encoding='utf-8') as f_json:
        json.dump(json_info, f_json, ensure_ascii=False, indent=4)


def move_files_to_folders(dir_path: str) -> None:
    """
    将 Speech 文件夹中的文件按文件名的第一部分分组，移动到对应的文件夹中。
    例如，文件名为 "abc.def.mp4" 的文件将被移动到 "Speech/abc/" 文件夹中。
    """

    target = Path(dir_path)

    # 检查文件夹是否存在
    if not target.exists():
        print(f"文件夹 {target} 不存在")
    else:
        # 获取所有文件
        files = [f for f in target.iterdir() if f.is_file()]
        
        # 按文件名第一部分分组
        file_groups = {}
        for file in files:
            # 获取文件名（不含扩展名）
            filename = file.name
            # 通过 . 分割并取第一个部分
            prefix = filename.split('.')[0]
            
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append(file)
        
        # 为每个分组创建文件夹并移动文件
        for prefix, group_files in file_groups.items():
            # 创建文件夹
            target_dir = target / prefix
            target_dir.mkdir(exist_ok=True)
            print(f"\n创建文件夹: {target_dir}")
            
            # 移动文件
            for file in group_files:
                target_path = target_dir / file.name
                shutil.move(str(file), str(target_path))
                print(f"  移动: {file.name} -> {prefix}/")
        
        print("\n文件整理完成！")


def convert_av1_to_h264(data_dir: str, dry_run: bool = True) -> Tuple[int, int]:
    """
    Convert AV1 encoded videos to H264 format in batch for better compatibility.
    Only converts files matching pattern: {video_id}.mp4 (excludes {video_id}.f{format}.mp4)
    
    Args:
        data_dir (str): Root directory containing video files (output/{category}/{video_name}/)
        dry_run (bool): If True, only preview without actual conversion (default: True)
    
    Returns:
        Tuple[int, int]: (converted_count, skipped_count)
    
    Directory Structure:
        data_dir/
        ├── category1/
        │   └── video_name/
        │       ├── _jcW-ZgpRbM.mp4          # Will convert
        │       ├── _jcW-ZgpRbM.f398.mp4     # Will skip
        │       └── _jcW-ZgpRbM_h264.mp4     # Output file
        └── category2/
    
    Examples:
        >>> converted, skipped = convert_av1_to_h264('./output', dry_run=False)
    """
    import re
    
    output_dir = Path(data_dir)
    converted_count = 0
    skipped_count = 0
    
    # Pattern: matches files like {video_id}.mp4, but NOT {video_id}.f{number}.mp4
    # Examples: _jcW-ZgpRbM.mp4 (match), _jcW-ZgpRbM.f398.mp4 (no match)
    pattern = re.compile(r'^[^.]+\.mp4$')
    
    # Traverse all category folders
    for category_dir in sorted(output_dir.iterdir()):
        # Skip non-directory and excluded folders
        if not category_dir.is_dir() or category_dir.name in ['logs', 'json_logs']:
            continue
        
        print(f"\nProcessing category: {category_dir.name}")
        
        # Traverse each video folder
        for video_dir in category_dir.iterdir():
            if not video_dir.is_dir():
                continue
            
            # Find all mp4 files matching the pattern
            for video_file in video_dir.glob("*.mp4"):
                # Skip already converted H264 files
                if '_h264' in video_file.stem or '_merged' in video_file.stem:
                    continue
                
                # Check if filename matches pattern (e.g., _jcW-ZgpRbM.mp4)
                if not pattern.match(video_file.name):
                    print(f"  Skip (format file): {video_file.name}")
                    skipped_count += 1
                    continue
                
                # Build output filename
                output_file = video_file.with_name(f"{video_file.stem}_h264.mp4")
                
                # Skip if converted file already exists
                if output_file.exists():
                    print(f"  Exists: {output_file.name}")
                    skipped_count += 1
                    continue
                
                if dry_run:
                    print(f"  [Preview] Will convert: {video_file.name} -> {output_file.name}")
                    converted_count += 1
                    continue
                
                # Convert AV1 to H264
                try:
                    print(f"  Converting: {video_file.name}")
                    cmd = [
                        'ffmpeg',
                        '-i', str(video_file),
                        '-c:v', 'libx264',  # Use H264 encoder
                        '-preset', 'fast',   # Fast encoding preset
                        '-crf', '23',        # Quality setting
                        '-c:a', 'aac',       # Audio codec
                        '-v', 'error',       # Only show errors
                        '-stats',            # Show progress
                        '-y',                # Overwrite output file
                        str(output_file)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0 and output_file.exists():
                        print(f"  Successfully converted: {video_file.name} -> {output_file.name}")
                        converted_count += 1
                    else:
                        print(f"  Conversion failed: {video_file.name} - {result.stderr}")
                        skipped_count += 1
                
                except Exception as e:
                    print(f"  Processing error {video_file.name}: {e}")
                    skipped_count += 1
    
    print(f"\n=== Conversion {'Preview' if dry_run else 'Complete'} ===")
    print(f"{'Will convert' if dry_run else 'Converted'}: {converted_count} files")
    print(f"Skipped: {skipped_count} files")
    
    if dry_run:
        print("\n⚠️ Preview mode - no files were actually converted")
        print("To execute conversion, run with --no-dry-run")
    
    return converted_count, skipped_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio-Visual Dataset Exploration and Processing Tools")
    
    # Mode selection
    parser.add_argument("--mode", type=str, required=True,
                       help="Operation mode: build-subset, analyze, merge, or clean")
    
    # Common arguments
    parser.add_argument("--data-dir", type=str, default='./output',
                       help="Data directory path (default: ./output)")
    parser.add_argument("--dry-run", action='store_true',
                       help="Preview mode without actual modifications")
    
    # build-subset specific arguments
    parser.add_argument("--input", type=str, default='json/filtered_video_clips.json',
                       help="Input JSON file for subset building")
    parser.add_argument("--categories", type=str, nargs='+',
                       default=['Personal Experience', 'Online Course/Lecture'],
                       help="Categories to include in subset")
    parser.add_argument("--max-duration", type=float, default=14*3600,
                       help="Maximum duration per category in seconds (default: 14 hours)")
    parser.add_argument("--output-dir", type=str, default='json',
                       help="Output directory for JSON files (default: json)")
    
    # analyze specific arguments
    parser.add_argument("--output-csv", type=str, default='dataset_statistics.csv',
                       help="Output CSV file for statistics (default: dataset_statistics.csv)")
    
    # clean specific arguments
    parser.add_argument("--no-backup", action='store_true',
                       help="Don't create backup when cleaning files")
    
    args = parser.parse_args()
    
    # Execute corresponding function based on mode
    if args.mode == 'build-subset':
        print(f"Building dataset subset from {args.input}...")
        print(f"Categories: {args.categories}")
        print(f"Max duration: {args.max_duration/3600:.1f} hours per category")
        build_dataset_subset(args.input, args.categories, args.max_duration, args.output_dir)
    
    elif args.mode == 'analyze':
        print(f"Analyzing dataset in {args.data_dir}...")
        analyze_dataset(args.data_dir, args.output_csv)
    
    elif args.mode == 'merge':
        print(f"Merging video and audio files in {args.data_dir}...")
        if args.dry_run:
            print("Running in DRY-RUN mode (preview only)")
        merge_video_audio(args.data_dir, args.dry_run)
    
    elif args.mode == 'clean':
        print(f"Cleaning non-MP4 files from {args.data_dir}...")
        if args.dry_run:
            print("Running in DRY-RUN mode (preview only)")
        clean_non_mp4_files(args.data_dir, not args.no_backup, args.dry_run)
    elif args.mode == 'build-json':
        print(f"Converting text files in {args.data_dir} to JSON format...")
        txt_to_json(args.data_dir, args.output_dir)
    elif args.mode == 'move-files':
        print(f"Moving files in {args.data_dir} to corresponding folders...")
        move_files_to_folders(args.data_dir)
    
    elif args.mode == 'convert-av1':
        print(f"Converting AV1 videos to H264 in {args.data_dir}...")
        if args.dry_run:
            print("Running in DRY-RUN mode (preview only)")
        convert_av1_to_h264(args.data_dir, args.dry_run)
    
    else:
        print(f"Unknown mode: {args.mode}")
        print("Available modes: build-subset, analyze, merge, clean, build-json, move-files, convert-av1")
