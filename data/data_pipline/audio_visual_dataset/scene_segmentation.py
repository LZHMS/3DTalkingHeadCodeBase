"""
Video Scene Segmentation Module

This module automatically detects scene transitions in videos and segments them into multiple clips.
It supports various scene detection algorithms (Adaptive, Histogram, Content, Hash, Threshold),
and allows selective export of video clips, audio clips, or both.

Main Features:
    1. Automatic video scene detection
    2. Scene clip export (video/audio)
    3. Scene metadata storage
    4. Multi-process parallel processing

Modified from https://github.com/FreedomIntelligence/TalkVid/blob/main/data_pipeline/1_video_rough_segmentation/video_clip.py
"""

import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict

import argparse
import subprocess
from multiprocessing import Process

from scenedetect import open_video, SceneManager, AdaptiveDetector, ContentDetector,ThresholdDetector,HistogramDetector
from scenedetect.stats_manager import StatsManager


def find_optimal_thread_count(total_samples, max_threads, threshold):
    """
    Automatically calculate the optimal number of threads based on sample count.
    
    This function balances the number of samples per thread to ensure no thread
    processes too few samples (below threshold), thereby improving multi-process efficiency.
    
    Args:
        total_samples (int): Total number of samples
        max_threads (int): Maximum available threads
        threshold (int): Minimum sample count threshold per thread
    
    Returns:
        int: Optimal number of threads
    
    Examples:
        >>> find_optimal_thread_count(100, 8, 10)
        10
    """
    optimal_thread_count = 1  
    if total_samples <= threshold:
        print(f"The number of samples is too small, and the returned optimal_thread_count is threshold: {threshold}")
        return total_samples
    for thread_count in range(1, max_threads + 1):
        samples_per_thread = total_samples // thread_count
        remaining_samples = total_samples - samples_per_thread * thread_count

        if remaining_samples <= threshold:
            optimal_thread_count = thread_count 

    return optimal_thread_count

def default_dump(obj):
    """
    Convert numpy objects to JSON serializable objects.
    
    Used to handle numpy data types when saving JSON files.
    
    Args:
        obj: Object to convert
    
    Returns:
        Converted serializable object
    """
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def save_json_entry(entry, path):
    """
    Save a single JSON record to file in append mode.
    
    Each record occupies one line for convenient streaming reads.
    
    Args:
        entry (dict): Dictionary data to save
        path (str): Target JSON file path
    """
    with open(path, 'a') as outfile:
        json.dump(entry, outfile, separators=(',', ':'), default=default_dump)
        outfile.write('\n')

def collect_video_data_path(data_dir: str) -> List[Dict]:
    """
    Recursively collect all video file paths and metadata from specified directory.
    
    Directory structure should be: data_dir/category/video_cate/video.mp4
    
    Args:
        data_dir (str): Data root directory path
    
    Returns:
        List[Dict]: Video information list, each element contains:
            - style-cate: Style category
            - video-cate: Video category
            - video-id: Video ID (filename without extension)
            - video-path: Complete video file path
    
    Examples:
        >>> collect_video_data_path('/data/videos')
        [{'style-cate': 'interview', 'video-cate': 'news', ...}, ...]
    """
    video_data = []
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
        
        for root, dirs, files in os.walk(category_path):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    video_id = os.path.splitext(file)[0]
                    video_cate = os.path.basename(root)
                    video_data.append({
                        "style-cate": category,
                        "video-cate": video_cate,
                        "video-id": video_id,
                        "video-path": video_path
                    })
    return video_data

def find_scenes_new(video_path, audio_path, output_subfolder,
                    subtitle_path, args):
    """
    Detect scene transitions in video and export scene clips.
    
    Uses scenedetect library for scene detection with support for multiple algorithms.
    Allows export of video clips, audio clips, or both, and saves scene metadata.
    
    Args:
        video_path (str): Input video file path
        audio_path (str): Input audio file path
        output_subfolder (str): Output directory path
        subtitle_path (str): Subtitle file path (optional)
        args (argparse.Namespace): Command line arguments object containing:
            - detector_type: Detector type
            - detector_threshold: Detection threshold
            - use_fixed_duration: Whether to use fixed duration
            - clip_style: Export mode ('none'/'all'/'video_only'/'audio_only')
    
    Returns:
        List[Dict]: Scene information list, each scene contains:
            - id: Scene ID
            - video-path: Scene video path
            - audio-path: Scene audio path
            - height/width/fps: Video properties
            - start-time/end-time: Time range
            - durations: Duration
            - original-video/audio: Original file paths
    
    Raises:
        IOError: When unable to open video file
    """
    # Get basic video information
    video_cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG) 
    if not video_cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    video_cap.release()

    # Use scenedetect for scene detection
    video = open_video(video_path)
    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)

    # Select detector based on parameters
    if args.detector_type == "Adaptive":
        scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=float(args.detector_threshold))) 

    elif args.detector_type == "Histogram":
        scene_manager.add_detector(HistogramDetector(threshold = float(args.detector_threshold)))
    
    elif args.detector_type == "Content":
        scene_manager.add_detector(ContentDetector(threshold = float(args.detector_threshold)))
    
    elif args.detector_type == "Hash":
        scene_manager.add_detector(HashDetector(threshold = float(args.detector_threshold)))
    
    elif args.detector_type == "Threshold":
        scene_manager.add_detector(ThresholdDetector(threshold = float(args.detector_threshold)))

    # Execute scene detection
    scene_manager.detect_scenes(video=video)
    scene_list = scene_manager.get_scene_list()

    scenes_data, scene_counter = [], 1  # Initialize scene data list and counter
    save_path_scenes_info = os.path.join(output_subfolder, f"video_scece_info.txt")

    # Process each detected scene
    for i, scene in enumerate(scene_list):
        start_time = scene[0].get_seconds() + 0.4 
        end_time = scene[1].get_seconds()
        duration = end_time - start_time

        # If fixed duration mode is enabled, adjust time range based on scene length
        if args.use_fixed_duration:
            # Adjust to fixed duration based on different duration intervals
            if 5.0 <= duration < 10.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 2.5
                end_time = middle + 2.5
            
            elif 10.0 <= duration < 15.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 5.0
                end_time = middle + 5.0
                
            elif 15.0 <= duration < 20.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 7.5
                end_time = middle + 7.5
            elif 20.0 <= duration < 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 10.0
                end_time = middle + 10.0
                
            elif duration >= 25.0:
                middle = (start_time + end_time) / 2
                start_time = middle - 12.5
                end_time = middle + 12.5
        else:
            # Fine-tune start and end times
            start_time = max(0, start_time - 0.1)
            end_time = max(0, end_time - 0.1)   

        # Define output file paths
        output_video_filename = os.path.join(output_subfolder, f"scene_{scene_counter}.mp4")
        output_audio_filename = os.path.join(output_subfolder, f"scene_{scene_counter}.m4a")

        # Record scene information for debugging
        start_minutes, start_seconds = divmod(start_time, 60)
        end_minutes, end_seconds = divmod(end_time, 60)

        with open(save_path_scenes_info, "a") as file: 
            file.write(f"scene {scene_counter} infos: start_time {int(start_minutes)}:{int(start_seconds)}, end_time {int(end_minutes)}:{int(end_seconds)}\n")

        # Determine export content based on clip_style parameter
        if args.clip_style == "none":
            # Skip video and audio export
            continue
        elif args.clip_style == "all":
            # Export both video and audio clips
            subprocess.run([
                "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(end_time - start_time),
                "-c:v", "libx264", "-preset", "medium", "-an", output_video_filename
            ])

            subprocess.run([ 
                "ffmpeg", "-ss", str(start_time), "-i", audio_path, "-t", str(end_time - start_time),
                "-c:a", "aac", output_audio_filename
            ])
        elif args.clip_style == "video_only":
            # Export video clip only
            subprocess.run([
                "ffmpeg", "-ss", str(start_time), "-i", video_path, "-t", str(end_time - start_time),
                "-c:v", "libx264", "-preset", "medium", "-an", output_video_filename
            ])
        elif args.clip_style == "audio_only":
            # Export audio clip only
            subprocess.run([ 
                "ffmpeg", "-ss", str(start_time), "-i", audio_path, "-t", str(end_time - start_time),
                "-c:a", "aac", output_audio_filename
            ])
        else:
            print(f"Unknown clip_style: {args.clip_style}. Skipping clip extraction.")
            continue

        # Save scene metadata
        scenes_data.append({
            "id": f"scene_{scene_counter}",
            "video-path": output_video_filename,
            "audio-path": output_audio_filename,
            "height": height,
            "width": width,
            "fps": fps,
            "start-time": start_time,
            "start-frame": scene[0].get_frames(),

            "end-time": end_time,
            "end-frame": scene[1].get_frames(),
            "durations": f"{round(end_time - start_time, 1)}s",
            "original-video": str(audio_path).replace(".m4a", ".mp4"),
            "original-audio": audio_path,
            "subtitle_path": subtitle_path,
        
        })
        
        scene_counter += 1  # Increment scene counter

    return scenes_data

def process_video_clips(data, args):
    """
    Batch process video list and perform scene segmentation.
    
    Iterates through video data list, performs scene detection and segmentation
    for each video, and saves results to corresponding output directory.
    
    Args:
        data (List[Dict]): Video information list generated by collect_video_data_path
        args (argparse.Namespace): Command line arguments object
    
    Notes:
        - Automatically creates output directory structure: output/style-cate/video-cate/video-id/
        - Scene information for each video is saved in scene_info.json
        - If audio file doesn't exist, video file will be used as audio source
    """
    for item in tqdm(data, desc="processing video clips"):
        try:
            style_cate, video_cate, video_id, video_path = item['style-cate'], item['video-cate'], item['video-id'], item['video-path']
            print(f"\nProcessing video_id: {video_id} in category: {video_cate}")
            print(f"Video path: {video_path}")

            if os.path.exists(video_path) and os.path.isfile(video_path):
                # Create output subdirectory
                output_subfolder = os.path.join(args.output, style_cate, video_cate, video_id)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Check if audio file exists
                audio_path = video_path.replace('.mp4', '.m4a')
                if not os.path.exists(audio_path):
                    audio_path = video_path
                
                subtitle_path = None
                # Perform scene detection and segmentation
                scenes_data = find_scenes_new(
                    video_path, 
                    audio_path,
                    output_subfolder,
                    subtitle_path,
                    args
                )
                
                # Save scene information to JSON file
                json_path = os.path.join(output_subfolder, "scene_info.json")
                for scene_data in scenes_data:
                    save_json_entry(scene_data, json_path)
                
                print(f"Successfully processed {video_id}, generated {len(scenes_data)} scenes")
            else:
                print(f"Warning: Video file does not exist or is not a file: {video_path}")
                print(f"  video_id: {video_id}")
                parent_dir = os.path.dirname(video_path)
                if os.path.exists(parent_dir):
                    print(f"  Parent directory exists, files in it: {os.listdir(parent_dir)[:5]}")
                else:
                    print(f"  Parent directory does not exist: {parent_dir}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run video head filter with multiprocessing.")
    
    # Input/output parameters
    parser.add_argument("--data-dir", type=str, default='output', help="Raw video data directory.")
    parser.add_argument("--output", type=str, default='output_clips', help="Output dir of clips.")
    
    # Multi-process parameters
    parser.add_argument("--num-workers", type=int, default=0, help="Number of worker processes to use.")
    parser.add_argument("--max-threads", type=int, default=48, help="The max threads available.")
    parser.add_argument("--threshold", type=int, default=7, help="The max samples per process.")

    # Export parameters
    parser.add_argument("--clip-style", type=str, default='video_only', 
                       help="Output format style: 'none', 'all', 'video_only', 'audio_only'")
    parser.add_argument("--use-fixed-duration", type=bool, default=False, 
                       help="Whether to use fixed duration for clips.")
    parser.add_argument("--clip-duration", type=int, default=5, help="The fixed duration for clips.")

    # Scene detection parameters
    parser.add_argument("--detector-type", type=str, default="Histogram", 
                       help="Type of detector: 'Adaptive', 'Histogram', 'Content', 'Hash', 'Threshold'")
    parser.add_argument("--detector-threshold", type=float, default=0.085, 
                       help="Threshold for the detector.")

    args = parser.parse_args()
    
    # Collect all video data
    data = collect_video_data_path(args.data_dir)

    part_number, num_samples = 0, len(data)

    # Calculate optimal number of processes
    num_process = find_optimal_thread_count(num_samples, args.max_threads, args.threshold)
    num_per_process = num_samples // num_process
    print(f"num_samples: {num_samples} num_process: {num_process} num_per_process: {num_per_process}")

    # Create and start multiple processes
    processes = []
    for idx in range(num_process):
        start_idx = idx * num_per_process
        end_idx = start_idx + num_per_process if idx < num_process - 1 else num_samples
        p_data = data[start_idx:end_idx]
        p = Process(target=process_video_clips, args=(p_data, args))
        processes.append(p)
        p.start()

    # Wait for all processes to complete
    for p in processes:
        p.join()