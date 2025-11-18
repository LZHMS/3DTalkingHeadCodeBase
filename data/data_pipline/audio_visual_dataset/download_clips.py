"""
Video Clip Downloader Module

This module downloads video/audio clips from online platforms (primarily YouTube) using yt-dlp.
It supports batch downloading, segment-based downloads, concurrent processing, and automatic
cleanup of failed downloads.

Main Features:
    1. JSON-based batch clip downloading
    2. URL availability checking and caching
    3. Multi-process concurrent downloads
    4. Automatic file organization and logging
    5. Failed download tracking and cleanup

Dependencies:
    - yt-dlp: Video downloading
    - rich: Progress bar display
    - subprocess: External command execution

Usage:
    python download_clips.py --input clips.json --output ./output --workers 4

Modified from https://github.com/FreedomIntelligence/TalkVid/blob/main/data_pipeline/0_video_download/download_clips.py
"""

import argparse
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Tuple
import json
import glob 
from pathlib import Path
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
)  # type: ignore


from utils import clip_success_downloaded, get_video_id, load_unavailable_urls, \
check_url_availability, get_yt_dlp_base_cmd, seconds_to_time_string, _match_segment_from_name


class YTDLPDownloader:
    """
    YouTube Downloader using yt-dlp.
    
    This class manages the entire download workflow including task preparation,
    URL validation, concurrent downloading, result handling, and file cleanup.
    
    Attributes:
        args (argparse.Namespace): Command line arguments
        full_download (bool): Whether to download full videos
        logs_dir (str): Directory for download logs
        json_dir (str): Directory for successful download JSON logs
        failed_urls_file (str): File tracking failed URLs
        failed_segments_file (str): File tracking failed segments
        url2cate (Dict[str, str]): Mapping from URL to category
        url2segments (Dict[str, List[Tuple[float, float]]]): Mapping from URL to segments
        unavailable_urls (set): Set of known unavailable URLs
    """
    
    def __init__(self, args):
        """
        Initialize the YTDLPDownloader.
        
        Args:
            args (argparse.Namespace): Command line arguments containing:
                - output_dir: Output directory path
                - full_download: Whether to download full videos
                - cookies_path: Path to cookies file
                - browser: Browser name for cookie extraction
                - extractor_args: Additional yt-dlp extractor arguments
        """
        self.args = args
        self.full_download = args.full_download
        self.output_dir = args.output_dir

        # Initialize logging directories
        self.logs_dir = os.path.join(args.output_dir, "download_logs")
        self.json_dir = os.path.join(self.logs_dir, "success_logs")
        os.makedirs(self.json_dir, exist_ok=True)

        self.failed_urls_file = os.path.join(self.logs_dir, "failed_urls.txt")
        self.failed_segments_file = os.path.join(self.logs_dir, "failed_segments.txt")

    def parsing_json(self, input_json_path: str):
        """
        Parse input JSON file to extract download tasks.
        
        Reads a JSON file containing video clip information and yields download tasks.
        Each task includes URL, start time, end time, and category.
        
        Args:
            input_json_path (str): Path to input JSON file
        
        Yields:
            Tuple[str, float, float, str]: Download task tuple containing:
                - url: Video URL
                - start_val: Start time (seconds) or -2 for full download
                - end_val: End time (seconds) or -1 for full download
                - category: Video category
        
        Raises:
            ValueError: If JSON parsing fails or data format is invalid
            AssertionError: If end-time is not greater than start-time
        
        Examples:
            >>> for url, start, end, cat in downloader.parsing_json("clips.json"):
            ...     print(f"Download {url} from {start}s to {end}s")
        """
        with open(input_json_path, "r", encoding="utf-8") as f:
            try:
                items = json.load(f)
            except json.JSONDecodeError as exc:
                raise ValueError(f"无法解析 JSON 文件: {input_json_path} | {exc}") from exc

        assert isinstance(items, list), ValueError("期望 JSON 顶层为数组（list）")
        for item in items:
            info_dict = item.get("info", {})
            url = info_dict.get("Video Link") or item.get("video link")
            start_val = -2 if self.full_download else float(item.get("start-time"))
            end_val = -1 if self.full_download else float(item.get("end-time"))
            category = info_dict.get("Video Category") or item.get("video category")

            assert end_val > start_val, ValueError("end-time 必须大于 start-time")
            yield (url, start_val, end_val, category)

    def prepare_download_task(self):
        """
        Prepare download tasks by parsing JSON and filtering completed downloads.
        
        This method:
        1. Parses the input JSON file
        2. Checks for already downloaded clips
        3. Groups segments by URL
        4. Loads unavailable URLs from previous runs
        5. Initializes progress bars
        
        Side Effects:
            - Populates self.url2cate and self.url2segments
            - Loads self.unavailable_urls
            - Initializes self.progress with two tasks
            - Prints statistics to console
        
        Notes:
            - Skips segments that have already been successfully downloaded
            - Respects the --limit argument if specified
        """
        segments_to_download, total_segments, skipped_success_segments = 0, 0, 0
        self.url2cate = defaultdict(str)
        self.url2segments: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        # Parse JSON and prepare download tasks
        for (url, start, end, category) in self.parsing_json(args.input_json_path):
            if args.limit is not None and args.limit >= 0 and segments_to_download >= args.limit:
                break
            
            self.url2cate[url], total_segments = category, total_segments + 1
            # Skip already downloaded clips
            if clip_success_downloaded(url, start, end, self.json_dir):
                skipped_success_segments += 1
                continue

            self.url2segments[url].append((start, end))
            segments_to_download += 1
                
        # Print task statistics
        print(f"Total segments found: {total_segments}")
        print(f"Skipped (already downloaded): {skipped_success_segments}")
        print(f"Segments to download: {segments_to_download}")

        if segments_to_download == 0:
            return

        # Load previously failed URLs to avoid re-probing
        self.unavailable_urls = load_unavailable_urls(self.failed_urls_file)
        if self.unavailable_urls:
            print(f"Loaded {len(self.unavailable_urls)} permanently unavailable URLs from logs.")

        # Initialize progress bars for URLs and segments
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        self.progress.__enter__()  # Manually enter context for cross-scope usage
        self.task_urls = self.progress.add_task("URLs", total=len(self.url2segments))
        self.task_segments = self.progress.add_task("Segments", total=segments_to_download)
        

    def run_processing(self):
        """
        Execute download tasks using multi-process pool.
        
        This method:
        1. Creates a process pool executor
        2. Checks URL availability before downloading
        3. Submits download tasks to the executor
        4. Tracks futures for result handling
        
        Side Effects:
            - Populates self.futures_map with submitted tasks
            - Updates progress bars
            - Writes to failed_urls_file and failed_segments_file for unavailable URLs
        
        Notes:
            - Skips URLs that are in the unavailable_urls cache
            - Uses check_url_availability to validate URLs before downloading
            - Number of workers controlled by args.workers
        """
        with ProcessPoolExecutor(max_workers=max(1, self.args.workers)) as executor:
            self.futures_map = {}
            for url, segs in self.url2segments.items():
                # Skip previously marked unavailable URLs
                if url in self.unavailable_urls:
                    reason = "Skipping probe: URL previously marked as permanently unavailable."
                    with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                        for (s, e) in segs:
                            fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
                    # Update progress bars
                    self.progress.update(self.task_urls, advance=1)
                    self.progress.update(self.task_segments, advance=len(segs))
                    continue
                
                # Check URL availability before downloading
                available, reason = check_url_availability(url, self.args.cookies, self.args.browser, self.args.extractor_args)
                if not available:
                    # Log failed URL and segments
                    with open(self.failed_urls_file, "a", encoding="utf-8") as furl:
                        furl.write(f"{url}\t{reason}\n")
                    for (s, e) in segs:
                        with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                            fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
                    self.progress.update(self.task_urls, advance=1)
                    self.progress.update(self.task_segments, advance=len(segs))
                else:
                    # Remove duplicates and sort segments
                    segs = sorted(set(segs))

                # Submit download task
                future = executor.submit(self.run_yt_dlp_download_segments, url, segs)
                self.futures_map[future] = (url, segs)
    
    def handle_results(self):
        """
        Handle download results from multi-process executor.
        
        This method:
        1. Collects results from completed futures
        2. Parses yt-dlp output to locate downloaded files
        3. Saves successful download metadata to JSON logs
        4. Records failed segments to failed_segments_file
        5. Updates progress bars
        6. Closes progress bar display
        
        Side Effects:
            - Creates JSON log files in self.json_dir
            - Appends to failed_segments_file for failures
            - Updates and closes progress bars
            - Prints error messages to console
        
        Notes:
            - Each successful segment gets its own JSON log file
            - JSON logs include source info, download info, and timestamp
            - Segments without output files are marked as failed
        """
        # Process completed download tasks
        for future in as_completed(self.futures_map):
            url0, segs0 = self.futures_map[future]
            self.progress.update(self.task_urls, advance=1)
            self.progress.update(self.task_segments, advance=len(segs0))

            video_id = get_video_id(url0)
            try:
                rc, msg = future.result()
            except Exception as exc:
                rc, msg = 1, f"Task failed with exception: {exc}"
            
            if rc != 0:
                # Log download failure
                print(f"[yt-dlp] Failed: {url0} | {msg}")
                with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                    for (s, e) in segs0:
                        fseg.write(f"FAIL\t{url0}\t{s:.3f}\t{e:.3f}\t{msg}\n")
            else:
                # Parse successful download output
                parsed_results = self.parse_ytdlp_output(msg, segs0, video_id)

                # Save metadata for each successful segment
                for seg_tuple, files_info in parsed_results.items():
                    start, end = seg_tuple
                    log_filename = f"{video_id}_full.json".replace(":", "-") if self.full_download \
                                    else f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
                    log_file = os.path.join(self.json_dir, log_filename)

                    log_data = {"source_info": {
                                    "url": url0,
                                    "start_time": start,
                                    "end_time": end,
                                },
                                "download_info": {
                                    **files_info,
                                    "status": "success",
                                    "error": "",
                                    "download_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                }}
                    with open(log_file, "w", encoding="utf-8") as f:
                        json.dump(log_data, f, ensure_ascii=False, indent=2)

                # Track segments that didn't produce output files
                succeeded_segs = set(parsed_results.keys())
                failed_segs = [s for s in segs0 if s not in succeeded_segs]
                if failed_segs:
                    with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                        for (s, e) in failed_segs:
                            fseg.write(f"FAIL\t{url0}\t{s:.3f}\t{e:.3f}\tNo output file generated\n")


        # Close progress bar display
        self.progress.__exit__(None, None, None)

    def parse_ytdlp_output(self, output: str, segments: List[Tuple[float, float]],
        video_id: str) -> Dict[Tuple[float, float], Dict]:
        """
        Parse yt-dlp output to map downloaded files to segments.
        
        This method parses both stdout and scans the output directory to ensure
        complete file information (video, audio, description, subtitles) is captured.
        Supports both full video downloads and segment-based downloads.
        
        Args:
            output (str): yt-dlp stdout containing file paths
            segments (List[Tuple[float, float]]): List of requested time segments
            video_id (str): Video identifier for directory lookup
        
        Returns:
            Dict[Tuple[float, float], Dict]: Mapping of segments to file information.
                Each value dict contains:
                    - video_clip_file: Path to video file
                    - audio_clip_file: Path to audio file
                    - description_file: Path to description file
                    - subtitle_files: List of subtitle file paths
        
        Notes:
            - For full downloads, uses (-2.0, -1.0) as the segment key
            - Matches files to segments using filename pattern matching
            - Scans output directory as fallback if stdout parsing is incomplete
            - Handles various video/audio formats (.mp4, .m4a, .webm, .mkv)
            - Handles various subtitle formats (.vtt, .srt, .ass)
        
        Examples:
            >>> results = downloader.parse_ytdlp_output(stdout, [(0, 10), (10, 20)], "abc123")
            >>> print(results[(0, 10)]['video_clip_file'])
            '/path/to/abc123_001_0.000_10.000.mp4'
        """
        files_from_stdout = [line.strip() for line in output.splitlines() if line.strip()]

        # Categorize files from stdout
        description_file: str = ""
        subtitle_files: List[str] = []
        clip_files: Dict[Tuple[float, float], List[str]] = defaultdict(list)
        for raw in files_from_stdout:
            possible_path = Path(raw.split(": ")[-1].strip())
            if not possible_path.exists():
                continue

            if possible_path.name.endswith(".description"):
                description_file = str(possible_path)
            elif possible_path.suffix.lower() in {".vtt", ".srt", ".ass"}:
                subtitle_files.append(str(possible_path))
            elif possible_path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
                seg_match = _match_segment_from_name(possible_path.name)
                if self.full_download:
                    clip_files[(-2.0, -1.0)].append(str(possible_path))
                else:
                    # Match file to closest requested segment
                    closest_seg = min(segments, key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1]))
                    clip_files[closest_seg].append(str(possible_path))

        # Scan output directory for additional files (fallback)
        try:
            video_output_dir = os.path.join(self.output_dir, video_id)
            for file_name in os.listdir(video_output_dir):
                file_path = os.path.join(video_output_dir, file_name)
                file_str_path = Path(file_path)
                if file_str_path.name.endswith(".description") and not description_file:
                    description_file = file_path
                elif file_str_path.suffix.lower() in {".vtt", ".srt", ".ass"} and file_path not in subtitle_files:
                    subtitle_files.append(file_path)
                elif file_str_path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
                    seg_match = _match_segment_from_name(file_str_path.name)
                    if self.full_download:
                        if file_path not in clip_files[(-1.0, -1.0)]:
                            clip_files[(-1.0, -1.0)].append(file_path)
                    else:
                        closest_seg = min(segments, key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1]))
                        if file_path not in clip_files[closest_seg]:
                            clip_files[closest_seg].append(file_path)
        except FileNotFoundError:
            pass

        # Assemble final results for each segment
        results: Dict[Tuple[float, float], Dict] = {}
        for seg in segments:
            file_list = clip_files.get(seg, [])
            video_file = next((f for f in file_list if f.endswith((".mp4", ".mkv", ".webm"))), "")
            audio_file = next((f for f in file_list if f.endswith(".m4a")), "")

            results[seg] = {
                "video_clip_file": video_file,
                "audio_clip_file": audio_file,
                "description_file": description_file,
                "subtitle_files": subtitle_files,
            }

        return results


    def run_yt_dlp_download_segments(self, url, segments) -> Tuple[int, str]:
        """
        Execute yt-dlp to download segments from a single URL.
        
        This method constructs and executes yt-dlp commands with multiple format trials.
        It supports both full video downloads and segment-based downloads with precise cuts.
        
        Args:
            url (str): Video URL to download from
            segments (List[Tuple[float, float]]): List of time segments to download.
                Empty list means full video download.
        
        Returns:
            Tuple[int, str]: A tuple containing:
                - return_code: 0 for success, 1 for failure
                - message: stdout on success, error message on failure
        
        Notes:
            - Uses multiple format trials with fallback options
            - First trial: merge to mp4 format
            - Second trial: remux to mp4 format (more lenient)
            - Extracts separate audio track in m4a format
            - Downloads subtitles and video description
            - Creates output directory structure: output_dir/category/video_id/
            - Filenames include segment info: {video_id}_{num}_{start}_{end}.{ext}
            - For full downloads: {video_id}_full.{ext}
        
        Examples:
            >>> rc, msg = downloader.run_yt_dlp_download_segments(url, [(0, 10), (10, 20)])
            >>> if rc == 0:
            ...     print(f"Success: {msg}")
        """
        base_cmd, err_info = get_yt_dlp_base_cmd(self.args.cookies, self.args.browser)
        assert base_cmd, "Unable to locate yt-dlp"

        video_id = get_video_id(url)
        video_output_dir = os.path.join(self.output_dir, self.url2cate[url], video_id)
        os.makedirs(video_output_dir, exist_ok=True)

        # Construct segment download arguments
        section_args: List[str] = []
        if not self.full_download:
            for (start, end) in segments:
                s_str, e_str = seconds_to_time_string(start), seconds_to_time_string(end)
                section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

            output_template = os.path.join(video_output_dir,
                "%(id)s_%(section_number)03d_%(section_start).3f_%(section_end).3f.%(ext)s",
            )
        else:
            output_template = os.path.join(video_output_dir, "%(id)s_full.%(ext)s", )
        
        section_args: List[str] = []
        if not self.full_download:
            for (start, end) in segments:
                s_str, e_str = seconds_to_time_string(start), seconds_to_time_string(end)
                section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

        cmd: List[str] = [
            *base_cmd,
            "-4",
            "--ignore-config",
            "--no-playlist",
            "--retries", "10",
            "--fragment-retries", "10",
            "--concurrent-fragments", "8",
            "-N", "4",
            "--no-warnings",
            "--restrict-filenames",
            "--no-continue", "--no-overwrites",
            # --- 新增功能 ---
            "--print", "after_move:filepath", # 打印最终文件路径
            "--write-subs",
            "--write-auto-subs",
            "--write-description",
            "--extract-audio",
            "--audio-format", "m4a", "--audio-quality", "0",
            "--keep-video",
            "--no-keep-fragments",  # 不保留中间文件
            "--clean-info-json",  # 清理信息文件
            # --- 输出模板 ---
            "-o", output_template,
            # 尽量拿到 H.264+AAC，可无损 remux；退化到 best 也能跑
            # "-s", "vcodec:h264,res,acodec:aac",
            "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "--merge-output-format", "mp4", 
        ]
        if self.args.strict_cuts:
            cmd.append("--force-keyframes-at-cuts")

        if self.args.extractor_args:
            cmd.extend(["--extractor-args", self.args.extractor_args])

        cmd.extend(section_args)
        cmd.append(url)

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            if proc.returncode == 0:
                return 0, proc.stdout.strip()
            err_msg = (proc.stderr.strip() or proc.stdout.strip())
            if "Requested format is not available" in err_msg:
                fallback_cmd = [
                    *base_cmd,
                    "-4", "--ignore-config", "--no-playlist",
                    "--retries", "10", "--fragment-retries", "10",
                    "--concurrent-fragments", "8", "-N", "4",
                    "--no-warnings", "--restrict-filenames",
                    "-c", "--no-overwrites",
                    # --- 新增功能 (回退) ---
                    "--print", "after_move:filepath",
                    "--write-subs", "--write-auto-subs", "--write-description",
                    "--extract-audio", "--audio-format", "m4a", "--keep-video",
                    # --- 输出模板 (回退) ---
                    "-o", output_template,
                    "-f", "bestvideo[ext=mp4][vcodec!=none]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=none]",
                    "--remux-video", "mp4",
                ]
                if self.args.strict_cuts:
                    fallback_cmd.append("--force-keyframes-at-cuts")
                if self.args.extractor_args:
                    fallback_cmd.extend(["--extractor-args", self.args.extractor_args])
                fallback_cmd.extend(section_args)
                fallback_cmd.append(url)

                proc2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding='utf-8')
                if proc2.returncode == 0:
                    return 0, proc2.stdout.strip()
                return proc2.returncode, (proc2.stderr.strip() or proc2.stdout.strip())
            return proc.returncode, err_msg
        except Exception as exc:  # noqa: BLE001
            return 1, f"yt-dlp failed: {exc}"

    # def run_yt_dlp_download_segments(self, url, segments) -> Tuple[int, str]:
    #     """
    #     Execute yt-dlp to download segments from a single URL.
        
    #     This method constructs and executes yt-dlp commands with multiple format trials.
    #     It supports both full video downloads and segment-based downloads with precise cuts.
        
    #     Args:
    #         url (str): Video URL to download from
    #         segments (List[Tuple[float, float]]): List of time segments to download.
    #             Empty list means full video download.
        
    #     Returns:
    #         Tuple[int, str]: A tuple containing:
    #             - return_code: 0 for success, 1 for failure
    #             - message: stdout on success, error message on failure
        
    #     Notes:
    #         - Uses multiple format trials with fallback options
    #         - First trial: merge to mp4 format
    #         - Second trial: remux to mp4 format (more lenient)
    #         - Extracts separate audio track in m4a format
    #         - Downloads subtitles and video description
    #         - Creates output directory structure: output_dir/category/video_id/
    #         - Filenames include segment info: {video_id}_{num}_{start}_{end}.{ext}
    #         - For full downloads: {video_id}_full.{ext}
        
    #     Examples:
    #         >>> rc, msg = downloader.run_yt_dlp_download_segments(url, [(0, 10), (10, 20)])
    #         >>> if rc == 0:
    #         ...     print(f"Success: {msg}")
    #     """
    #     base_cmd, err_info = get_yt_dlp_base_cmd(self.args.cookies, self.args.browser)
    #     assert base_cmd, "Unable to locate yt-dlp"

    #     video_id = get_video_id(url)
    #     video_output_dir = os.path.join(self.output_dir, self.url2cate[url], video_id)
    #     os.makedirs(video_output_dir, exist_ok=True)

    #     # Construct segment download arguments
    #     section_args: List[str] = []
    #     if not self.full_download:
    #         for (start, end) in segments:
    #             s_str, e_str = seconds_to_time_string(start), seconds_to_time_string(end)
    #             section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

    #         output_template = os.path.join(video_output_dir,
    #             "%(id)s_%(section_number)03d_%(section_start).3f_%(section_end).3f.%(ext)s",
    #         )
    #     else:
    #         output_template = os.path.join(video_output_dir, "%(id)s_full.%(ext)s", )
        
    #     # Define format trials with fallback options
    #     format_trials = [
    #         {
    #             # Trial 0: Preferred format (ideal)
    #             "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    #             "extra_flags": [
    #                 "--merge-output-format", "mp4",
    #             ],
    #             "continue_flag": ["--no-continue", "--no-overwrites"],
    #         },
    #         {
    #             # Trial 1: Fallback format (more lenient)
    #             "format": "bestvideo[ext=mp4][vcodec!=none]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=none]",
    #             "extra_flags": [
    #                 "--remux-video", "mp4",
    #             ],
    #             "continue_flag": ["-c", "--no-overwrites"],
    #         },
    #     ]
    #     section_args: List[str] = []
    #     if not self.full_download:
    #         for (start, end) in segments:
    #             s_str, e_str = seconds_to_time_string(start), seconds_to_time_string(end)
    #             section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

    #     # Try each format until success
    #     last_err = ""
    #     for trial in format_trials:
    #         # Construct yt-dlp command
    #         cmd: List[str] = [
    #             *base_cmd,
    #             "-4", "--ignore-config", "--no-playlist",
    #             "--retries", "10", "--fragment-retries", "10",
    #             "--concurrent-fragments", "8", "-N", "4",
    #             "--no-warnings", "--restrict-filenames",
    #             *trial["continue_flag"],
    #             "--print", "after_move:filepath",  # Print final file paths
    #             "--write-subs", "--write-auto-subs", "--write-description",
    #             "--extract-audio", "--audio-format", "m4a",
    #             "--audio-quality", "0", "--keep-video",
    #             "--no-keep-fragments", "--clean-info-json",
    #             "-o", output_template,
    #             "-f", trial["format"],
    #             *trial["extra_flags"],
    #         ]

    #         if self.args.strict_cuts:
    #             cmd.append("--force-keyframes-at-cuts")
    #         if self.args.extractor_args:
    #             cmd.extend(["--extractor-args", self.args.extractor_args])

    #         cmd.extend(section_args)
    #         cmd.append(url)

    #         # Execute yt-dlp command
    #         try:
    #             proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    #         except Exception as exc:  # noqa: BLE001
    #             return 1, f"yt-dlp failed: {exc}"

    #         if proc.returncode == 0:
    #             return 0, proc.stdout.strip()

    #         # Check if we should try next format
    #         last_err = (proc.stderr.strip() or proc.stdout.strip())
    #         if "Requested format is not available" not in last_err:
    #             break

    #     return 1, last_err or "yt-dlp failed with unknown error"

    def cleanup_final_files(self) -> None:
        """
        Clean up unrecorded files based on JSON logs.
        
        This method scans all JSON log files to identify which files were successfully
        downloaded, then removes any files in the output directory that are not
        referenced in the logs. This keeps the output directory clean and organized.
        
        Side Effects:
            - Deletes files not recorded in JSON logs
            - Prints progress messages to console
        
        Notes:
            - Preserves all files mentioned in success logs
            - Skips deletion of log files themselves
            - Only processes files with "success" status in logs
            - Handles video_clip_file, audio_clip_file, description_file, and subtitle_files
        
        Examples:
            >>> downloader.cleanup_final_files()
            Starting final file cleanup...
            Found 150 files recorded in logs. Scanning for unrecorded files...
            Deleted unrecorded file: /path/to/orphan_file.tmp
            Cleanup complete. Deleted 5 unrecorded files.
        """
        print("\nStarting final file cleanup...")

        # Collect all recorded file paths from JSON logs
        recorded_files = set()
        json_files = glob.glob(os.path.join(self.json_dir, "*.json"))
        for log_file in json_files:
            recorded_files.add(os.path.abspath(log_file))
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                info = data.get("download_info", {})
                if info.get("status") != "success":
                    continue

                # Collect all file references
                for key, value in info.items():
                    if key.endswith("_file") and isinstance(value, str) and value:
                        recorded_files.add(os.path.abspath(value))
                    elif key.endswith("_files") and isinstance(value, list):
                        for item in value:
                            if isinstance(item, str) and item:
                                recorded_files.add(os.path.abspath(item))
            except Exception as e:
                print(f"Error processing log file {log_file}: {e}")

        if not recorded_files:
            print("No recorded files found in logs, skipping cleanup.")
            return

        # Scan output directory and delete unrecorded files
        print(f"Found {len(recorded_files)} files recorded in logs. Scanning for unrecorded files...")
        deleted_count = 0
        for root, _, files in os.walk(args.output_dir):
            for file in files:
                # Skip log files
                if "logs" in root and ("failed_urls.txt" in file or "failed_segments.txt" in file):
                    continue

                file_path = os.path.abspath(os.path.join(root, file))
                if file_path not in recorded_files:
                    try:
                        os.remove(file_path)
                        deleted_count += 1
                        print(f"Deleted unrecorded file: {file_path}")
                    except OSError as e:
                        print(f"Failed to delete {file_path}: {e}")
        print(f"Cleanup complete. Deleted {deleted_count} unrecorded files.")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download clips specified in a large JSON file directly with yt-dlp sections on Windows.")
    parser.add_argument("--input_json_path", type=str, default=os.path.join(os.getcwd(), "filtered_video_clips.json"), help="Path to the large JSON file containing 'Video Link', 'start-time', 'end-time' fields.")
    parser.add_argument("--output_dir", type=str, default=os.path.join(os.getcwd(), "clips_output"), help="Directory to store the downloaded clips.")
    parser.add_argument("--mode", choices=["ytdlp"], default="ytdlp", help="Only support 'ytdlp'")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many clip segments (for testing).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for yt-dlp mode.")
    parser.add_argument("--cookies", type=str, default=os.path.join(os.getcwd(), "cookies.txt"), help="Path to cookies.txt to pass to yt-dlp if present.")
    parser.add_argument("--browser", type=str, default=None, help="Use --cookies-from-browser <browser> for YouTube auth (recommended).")
    parser.add_argument("--strict_cuts", type=bool, default=True, help="")
    parser.add_argument("--extractor_args", type=str, default=None, help="Pass through to yt-dlp --extractor-args, e.g. 'youtube:player_client=android'")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup process after downloading to remove unlogged files.")
    parser.add_argument("--full_download", action="store_true", help="Download full videos instead of segments.")

    args = parser.parse_args()
    ytdlp_dl = YTDLPDownloader(args)

    print("Starting yt-dlp clip downloads...\n")
    ytdlp_dl.prepare_download_task()
    ytdlp_dl.run_processing()
    ytdlp_dl.handle_results()
    ytdlp_dl.cleanup_final_files()
    print("yt-dlp clip downloads completed.")


