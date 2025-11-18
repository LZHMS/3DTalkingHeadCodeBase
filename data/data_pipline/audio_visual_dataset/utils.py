"""
Utility Module for Audio-Visual Dataset Processing

This module provides utility functions for downloading and processing YouTube videos,
including URL validation, video availability checking, executable detection, and
time format conversion.

Main Features:
    1. YouTube video ID extraction
    2. Download status verification via JSON logs
    3. Unavailable URL filtering from error logs
    4. yt-dlp executable detection and command building
    5. URL availability checking
    6. Time format conversion utilities
    7. Segment information extraction from filenames

Dependencies:
    - yt-dlp: For YouTube video downloading
    - subprocess: For running external commands
"""

import os
import os
import re
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple
import json
import math

def get_video_id(url: str) -> str:
    """
    Extract video ID from YouTube URL.
    
    Supports multiple YouTube URL formats including standard watch URLs,
    shortened youtu.be URLs, and shorts URLs.
    
    Args:
        url (str): YouTube video URL
    
    Returns:
        str: Extracted video ID, or "unknown_id" if extraction fails
    
    Examples:
        >>> get_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> get_video_id("https://youtu.be/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
        >>> get_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ")
        'dQw4w9WgXcQ'
    """
    if 'watch?v=' in url:
        return url.split('watch?v=')[-1].split('&')[0]
    if 'youtu.be/' in url:
        return url.split('youtu.be/')[-1].split('?')[0]
    if '/shorts/' in url:
        return url.split('/shorts/')[-1].split('?')[0]
    # Fallback for other URL formats or just return a hash
    # For simplicity, we'll just use the last part of the URL
    return url.split('/')[-1] or "unknown_id"


def clip_success_downloaded(url: str, start: float, end: float, output_dir: str) -> bool:
    """
    Check if a video clip has been successfully downloaded by examining JSON logs.
    
    Verifies both the existence of the log file with "success" status and
    the actual video file on disk.
    
    Args:
        url (str): YouTube video URL
        start (float): Clip start time in seconds (use -1 for full video)
        end (float): Clip end time in seconds (use -1 for full video)
        output_dir (str): Directory containing download logs and video files
    
    Returns:
        bool: True if clip was successfully downloaded and file exists, False otherwise
    
    Notes:
        - For full video downloads, use start=-1 and end=-1
        - Log filename format: {video_id}_{start}_{end}.json or {video_id}_full.json
        - Also validates that the actual video file exists and has size > 0
    """
    video_id = get_video_id(url)
    if start < 0 and end < 0:
        log_filename = f"{video_id}_full.json".replace(":", "-")
    else:
        log_filename = f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
    log_file = os.path.join(output_dir, log_filename)

    if not os.path.exists(log_file):
        return False

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        if log_data.get("download_info", {}).get("status") == "success":
            # Additionally check if video file actually exists
            video_file = log_data.get("download_info", {}).get("video_clip_file")
            if video_file and os.path.exists(video_file) and os.path.getsize(video_file) > 0:
                return True
    except (json.JSONDecodeError, KeyError):
        return False
    return False

def load_unavailable_urls(log_file_path: str) -> set:
    """
    Load set of URLs with permanent errors from error log file.
    
    Parses error log to identify videos that are permanently unavailable
    (e.g., deleted, private, account terminated) to avoid repeated download attempts.
    
    Args:
        log_file_path (str): Path to the error log file
    
    Returns:
        set: Set of URLs that are permanently unavailable
    
    Notes:
        - Log file format: URL\\tERROR_MESSAGE per line
        - Recognized permanent error phrases (case-insensitive):
            - "video unavailable"
            - "account associated with this video has been terminated"
            - "private video"
            - "video is private"
            - "user has closed their youtube account"
    
    Examples:
        >>> urls = load_unavailable_urls("error.log")
        >>> "https://youtube.com/watch?v=deleted_video" in urls
        True
    """
    unavailable_urls = set()
    if not os.path.exists(log_file_path):
        return unavailable_urls

    # Common permanent error message phrases (lowercase)
    permanent_error_phrases = [
        "video unavailable",
        "account associated with this video has been terminated",
        "private video",
        "video is private",
        "user has closed their youtube account",
    ]

    try:
        with open(log_file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    url, reason = parts[0], parts[1].lower()
                    if any(phrase in reason for phrase in permanent_error_phrases):
                        unavailable_urls.add(url)
    except Exception as e:
        print(f"Warning: Could not read or parse unavailable URLs log: {e}")

    return unavailable_urls


def find_executable(candidates: List[str]) -> Optional[str]:
    """
    Find the first existing executable from a list of candidate paths.
    
    Checks both system PATH and local file paths to locate executables.
    
    Args:
        candidates (List[str]): List of candidate executable paths or names
    
    Returns:
        Optional[str]: Path to first found executable, or None if none exist
    
    Examples:
        >>> find_executable(["ffmpeg", "/usr/bin/ffmpeg"])
        'ffmpeg'  # or '/usr/bin/ffmpeg' depending on system
    """
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
        # Also consider a relative executable within the current directory
        if os.path.isfile(candidate):
            return candidate
    return None

def get_yt_dlp_base_cmd(cookies_path: Optional[str], browser: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Build base yt-dlp command with cookies and configuration.
    
    Attempts to use yt-dlp as a Python module first, then falls back to
    standalone executable. Configures authentication via cookies file or browser.
    
    Args:
        cookies_path (Optional[str]): Path to cookies.txt file for authentication
        browser (Optional[str]): Browser name to extract cookies from (e.g., 'chrome', 'firefox')
    
    Returns:
        Tuple[Optional[List[str]], Optional[str]]: 
            - List of command arguments if successful, None otherwise
            - Error message if failed, None if successful
    
    Notes:
        - Prefers 'python -m yt_dlp' over standalone executable
        - Automatically adds '-4' (IPv4) and '--ignore-config' flags
        - Browser cookies take precedence over cookies file
        - Searches for yt-dlp.exe in current directory if module not found
    
    Examples:
        >>> cmd, err = get_yt_dlp_base_cmd("cookies.txt", None)
        >>> cmd
        ['python', '-m', 'yt_dlp', '--cookies', 'cookies.txt', '-4', '--ignore-config']
    """
    try:
        import importlib.util  # noqa: F401

        if importlib.util.find_spec("yt_dlp") is not None:
            base_cmd = [sys.executable, "-m", "yt_dlp"]
        else:
            raise ImportError
    except Exception:
        yt_dlp_candidates = [
            os.path.join(os.getcwd(), "yt-dlp_x86.exe"),
            os.path.join(os.getcwd(), "yt-dlp.exe"),
            "yt-dlp",
        ]
        yt_dlp_path = find_executable(yt_dlp_candidates)
        if yt_dlp_path is None:
            return None, "yt-dlp not found. Install with: python -m pip install --user -U yt-dlp"
        base_cmd = [yt_dlp_path]

    # Cookies are added by callers depending on the operation (probe/download)
    if browser:
        base_cmd = [*base_cmd, "--cookies-from-browser", browser]
    elif cookies_path and os.path.exists(cookies_path):
        base_cmd = [*base_cmd, "--cookies", cookies_path]

    # Use IPv4 and ignore user config files for reproducibility
    base_cmd = [*base_cmd, "-4", "--ignore-config"]
    return base_cmd, None

def check_url_availability(
    url: str,
    cookies_path: Optional[str],
    browser: Optional[str],
    extractor_args: Optional[str] = None,
) -> Tuple[bool, str]:
    """
    Check if a YouTube URL is available without downloading.
    
    Uses yt-dlp to probe the URL and verify accessibility. Useful for
    filtering out deleted, private, or region-restricted videos before download.
    
    Args:
        url (str): YouTube video URL to check
        cookies_path (Optional[str]): Path to cookies.txt file
        browser (Optional[str]): Browser name for cookie extraction
        extractor_args (Optional[str]): Additional yt-dlp extractor arguments
    
    Returns:
        Tuple[bool, str]:
            - bool: True if URL is available, False otherwise
            - str: Video ID if available, error message if unavailable
    
    Notes:
        - Uses yt-dlp's '-s' (simulate) flag to avoid downloading
        - Returns video ID on success for verification
        - Captures stderr/stdout for error diagnosis
    
    Examples:
        >>> available, msg = check_url_availability("https://youtube.com/watch?v=dQw4w9WgXcQ", None, None)
        >>> available
        True
        >>> msg
        'dQw4w9WgXcQ'
    """
    base_cmd, err = get_yt_dlp_base_cmd(cookies_path, browser)
    if base_cmd is None:
        return False, err or "yt-dlp not found"

    cmd: List[str] = [
        *base_cmd,
        "-s",
        "--no-warnings",
        "-O",
        "%(id)s",
        url,
    ]
    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode == 0 and proc.stdout.strip():
            return True, proc.stdout.strip()
        # Collect an error message
        msg = (proc.stderr or proc.stdout or "unknown error").strip()
        return False, msg
    except Exception as exc:  # noqa: BLE001
        return False, f"probe failed: {exc}"
    
def seconds_to_time_string(seconds_value: float) -> str:
    """
    Convert seconds to time string in format HH:MM:SS or HH:MM:SS.mmm.
    
    Handles fractional seconds and formats with millisecond precision
    when necessary. Ensures non-negative values.
    
    Args:
        seconds_value (float): Time value in seconds (negative values treated as 0)
    
    Returns:
        str: Formatted time string in HH:MM:SS or HH:MM:SS.mmm format
    
    Examples:
        >>> seconds_to_time_string(3661.5)
        '01:01:01.500'
        >>> seconds_to_time_string(125)
        '00:02:05'
        >>> seconds_to_time_string(-10)
        '00:00:00'
    
    Notes:
        - Milliseconds are omitted if zero
        - Values are floored to millisecond precision
        - Small floating point errors are handled via epsilon (1e-6)
    """
    if seconds_value < 0:
        seconds_value = 0.0
    ms = math.floor(seconds_value * 1000 + 1e-6)  # Floor to milliseconds
    hours, rem = divmod(ms, 3600_000)
    minutes, ms = divmod(rem, 60_000)
    seconds, ms = divmod(ms, 1000)
    if ms == 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _match_segment_from_name(name: str) -> Optional[Tuple[float, float]]:
    """
    Extract segment (start, end) times from filename.
    
    Parses filenames to identify clip time ranges. Supports both segmented
    clips with timestamps and full video downloads.
    
    Args:
        name (str): Filename to parse
    
    Returns:
        Optional[Tuple[float, float]]: 
            - Tuple of (start_time, end_time) in seconds if segment found
            - (-1.0, -1.0) if full video (indicated by '_full.' in filename)
            - None if no match found
    
    Examples:
        >>> _match_segment_from_name("video_id_10.500_20.750.mp4")
        (10.5, 20.75)
        >>> _match_segment_from_name("video_id_full.mp4")
        (-1.0, -1.0)
        >>> _match_segment_from_name("invalid_name.mp4")
        None
    
    Notes:
        - Expected segment filename format: *_{start}_{end}.*
        - Full video filename format: *_full.*
        - Times should be in decimal format (e.g., 10.500)
    """
    # Check if this is a full video download
    if "_full." in name:
        return (-1.0, -1.0)  # Use special marker for full video
    m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)\.", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None