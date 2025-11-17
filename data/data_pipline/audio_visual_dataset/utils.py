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
    """从 YouTube URL 中提取视频 ID。"""
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
    """通过检查 JSON 日志来判断片段是否已成功下载。"""
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
            # 额外检查视频文件是否真实存在
            video_file = log_data.get("download_info", {}).get("video_clip_file")
            if video_file and os.path.exists(video_file) and os.path.getsize(video_file) > 0:
                return True
    except (json.JSONDecodeError, KeyError):
        return False
    return False

def load_unavailable_urls(log_file_path: str) -> set:
    """从日志文件中加载存在永久性错误（如视频不可用）的 URL 集合。"""
    unavailable_urls = set()
    if not os.path.exists(log_file_path):
        return unavailable_urls

    # 常见的永久性错误信息片段（小写）
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
    """Return the first existing candidate executable path or None."""
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
        # Also consider a relative executable within the current directory
        if os.path.isfile(candidate):
            return candidate
    return None

def get_yt_dlp_base_cmd(cookies_path: Optional[str], browser: Optional[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """Build base yt-dlp command, preferring python -m yt_dlp. Returns (cmd, error)."""
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
    """Check if a URL is available by asking yt-dlp to print the id without downloading."""
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
    if seconds_value < 0:
        seconds_value = 0.0
    ms = math.floor(seconds_value * 1000 + 1e-6)  # 向下取整到毫秒
    hours, rem = divmod(ms, 3600_000)
    minutes, ms = divmod(rem, 60_000)
    seconds, ms = divmod(ms, 1000)
    if ms == 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def _match_segment_from_name(name: str) -> Optional[Tuple[float, float]]:
    """从文件名中提取 segment (start,end)。返回 None 如果无法匹配。如果是完整视频，返回特殊标记。"""
    # 检查是否为完整视频下载
    if "_full." in name:
        return (-1.0, -1.0)  # 使用特殊标记表示完整视频
    m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)\.", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None