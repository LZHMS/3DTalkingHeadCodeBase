import argparse
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, Generator, List, Optional, Tuple
import json
import math
import glob 
from pathlib import Path
from unicodedata import category 
from rich.progress import (
    Progress,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TextColumn,
    TaskID,
)  # type: ignore
from rich.table import Table
from rich.live import Live
from rich.console import Console




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


def find_executable(candidates: List[str]) -> Optional[str]:
    """Return the first existing candidate executable path or None."""
    for candidate in candidates:
        if shutil.which(candidate):
            return candidate
        # Also consider a relative executable within the current directory
        if os.path.isfile(candidate):
            return candidate
    return None


def safe_mkdir(path: str) -> None:
    if not path:
        return
    os.makedirs(path, exist_ok=True)


def is_clip_downloaded(
    url: str, start: float, end: float, output_dir: str
) -> bool:
    """通过检查 JSON 日志来判断片段是否已成功下载。"""
    video_id = get_video_id(url)
    # 规范化文件名中的浮点数格式
    log_filename = f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
    log_file = os.path.join(output_dir, "json_logs", log_filename)

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


def is_video_downloaded(video_id: str, output_dir: str) -> bool:
    """通过检查 JSON 日志来判断完整视频是否已成功下载。"""
    log_filename = f"{video_id}.json"
    log_file = os.path.join(output_dir, "json_logs", log_filename)

    if not os.path.exists(log_file):
        return False

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            log_data = json.load(f)
        if log_data.get("download_info", {}).get("status") == "success":
            # 额外检查视频文件是否真实存在
            video_file = log_data.get("download_info", {}).get("video_file")
            if video_file and os.path.exists(video_file) and os.path.getsize(video_file) > 0:
                return True
    except (json.JSONDecodeError, KeyError):
        return False
    return False


def iter_segments_from_big_json(
    input_json_path: str,
) -> Generator[Tuple[str, float, float, str, float], None, None]:
    """
    直接加载并遍历 JSON 文件。

    每一项（dict）应包含键：
        - "Video Link"（或小写变体）
        - "start-time" / "start"
        - "end-time" / "end"

    返回 (url, start, end) 元组。
    """

    with open(input_json_path, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 JSON 文件: {input_json_path} | {exc}") from exc

    if not isinstance(items, list):
        raise ValueError("期望 JSON 顶层为数组（list）")

    for item in items:
        if not isinstance(item, dict):
            continue

        info_dict = item.get("info", {})
        url = info_dict.get("Video Link") or info_dict.get("video_link")
        start_val = item.get("start-time") or item.get("start")
        end_val = item.get("end-time") or item.get("end")
        duration = float(item['durations'][:-1])
        category = info_dict["Video Category"]

        if url is None or start_val is None or end_val is None:
            continue

        try:
            start_f = float(start_val)
            end_f = float(end_val)
        except (TypeError, ValueError):
            continue

        if end_f > start_f:
            yield (url, start_f, end_f, category, duration)


def iter_videos_from_json(
    input_json_path: str,
) -> Generator[Tuple[str, str, str], None, None]:
    """
    从 JSON 文件中解析视频信息。

    每一项（dict）应包含键：
        - "id": 视频ID
        - "video link": 视频链接
        - "video category": 视频分类

    返回 (video_id, url, category) 元组。
    """
    with open(input_json_path, "r", encoding="utf-8") as f:
        try:
            items = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"无法解析 JSON 文件: {input_json_path} | {exc}") from exc

    if not isinstance(items, list):
        raise ValueError("期望 JSON 顶层为数组（list）")

    for item in items:
        if not isinstance(item, dict):
            continue

        video_id = item.get("id")
        url = item.get("video link")
        category = item.get("video category", "Unknown")

        if not video_id or not url:
            continue

        yield (video_id, url, category)


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

def run_yt_dlp_multi_sections(
    url: str,
    segments: List[Tuple[float, float]],
    output_dir: str,
    cookies_path: Optional[str] = None,
    browser: Optional[str] = None,
    extractor_args: Optional[str] = None,
    strict_cuts: bool = False,   # True = 更准的切口（会重编码，慢）；False = 更快（关键帧附近）
) -> Tuple[int, str]:
    """
    对同一 URL 的多个片段，合并为一次 yt-dlp 调用（多个 --download-sections）。
    产物文件名使用 section 变量，避免覆盖。
    """
    safe_mkdir(output_dir)
    base_cmd, err = get_yt_dlp_base_cmd(cookies_path, browser)
    if base_cmd is None:
        return 1, err or "Unable to locate yt-dlp"

    # --- 构造下载命令 ---
    video_id = get_video_id(url)
    video_output_dir = os.path.join(output_dir, video_id)
    safe_mkdir(video_output_dir)

    # 构造多段 sections
    section_args: List[str] = []
    for (s, e) in segments:
        if e <= s:
            continue
        s_str = seconds_to_time_string(s)
        e_str = seconds_to_time_string(e)
        section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

    if not section_args:
        return 1, "No valid segments for this URL"

    # 输出模板：所有文件都放入以 video_id 命名的子目录中
    output_template = os.path.join(
        video_output_dir,
        "%(id)s_%(section_number)03d_%(section_start).3f_%(section_end).3f.%(ext)s",
    )

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
    if strict_cuts:
        cmd.append("--force-keyframes-at-cuts")

    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])

    # 拼上多段
    cmd.extend(section_args)
    cmd.append(url)

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if proc.returncode == 0:
            return 0, proc.stdout.strip()
        # 简单回退：遇到“格式不可用”就退到 best
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
            if strict_cuts:
                fallback_cmd.append("--force-keyframes-at-cuts")
            if extractor_args:
                fallback_cmd.extend(["--extractor-args", extractor_args])
            fallback_cmd.extend(section_args)
            fallback_cmd.append(url)

            proc2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding='utf-8')
            if proc2.returncode == 0:
                return 0, proc2.stdout.strip()
            return proc2.returncode, (proc2.stderr.strip() or proc2.stdout.strip())
        return proc.returncode, err_msg
    except Exception as exc:  # noqa: BLE001
        return 1, f"yt-dlp failed: {exc}"


def run_yt_dlp_full_video(
    url: str,
    video_id: str,
    output_dir: str,
    cookies_path: Optional[str] = None,
    browser: Optional[str] = None,
    extractor_args: Optional[str] = None,
) -> Tuple[int, str]:
    """
    下载完整视频（不分段）。
    """
    safe_mkdir(output_dir)
    base_cmd, err = get_yt_dlp_base_cmd(cookies_path, browser)
    if base_cmd is None:
        return 1, err or "Unable to locate yt-dlp"

    # 输出模板：使用视频ID作为文件名
    output_template = os.path.join(
        output_dir,
        "%(id)s.%(ext)s",
    )

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
        "--print", "after_move:filepath",
        "--write-subs",
        "--write-auto-subs",
        "--write-description",
        "--extract-audio",
        "--audio-format", "m4a", "--audio-quality", "0",
        "--keep-video",
        "--no-keep-fragments",
        "--clean-info-json",
        "-o", output_template,
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
    ]

    if extractor_args:
        cmd.extend(["--extractor-args", extractor_args])

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
                "--print", "after_move:filepath",
                "--write-subs", "--write-auto-subs", "--write-description",
                "--extract-audio", "--audio-format", "m4a", "--keep-video",
                "-o", output_template,
                "-f", "bestvideo[ext=mp4][vcodec!=none]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=none]",
                "--remux-video", "mp4",
            ]
            if extractor_args:
                fallback_cmd.extend(["--extractor-args", extractor_args])
            fallback_cmd.append(url)

            proc2 = subprocess.run(fallback_cmd, capture_output=True, text=True, encoding='utf-8')
            if proc2.returncode == 0:
                return 0, proc2.stdout.strip()
            return proc2.returncode, (proc2.stderr.strip() or proc2.stdout.strip())
        return proc.returncode, err_msg
    except Exception as exc:
        return 1, f"yt-dlp failed: {exc}"


def probe_url_availability(
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


# ---- 解析下载产物 ------------------------------------------------------------


def _match_segment_from_name(name: str) -> Optional[Tuple[float, float]]:
    """从文件名中提取 segment (start,end)。返回 None 如果无法匹配。"""
    m = re.search(r"_(\d+\.\d+)_(\d+\.\d+)\.", name)
    if m:
        return float(m.group(1)), float(m.group(2))
    return None


def parse_ytdlp_output(
    output: str,
    segments: List[Tuple[float, float]],
    video_id: str,
    video_output_dir: str,
) -> Dict[Tuple[float, float], Dict]:
    """
    解析 yt-dlp 的输出，将文件路径与原始分段关联。
    现在不仅解析 stdout，还会回退到扫描输出目录，以确保拿到完整的
    audio / description / subtitle 信息。
    """

    # 起始：先从 stdout 粗略提取
    files_from_stdout = [line.strip() for line in output.splitlines() if line.strip()]

    # 分类容器
    description_file: str = ""
    subtitle_files: List[str] = []
    clip_files: Dict[Tuple[float, float], List[str]] = defaultdict(list)

    for raw in files_from_stdout:
        # 截掉前缀标记，如 "[info] Writing video description to: "
        possible_path = raw.split(": ")[-1].strip()
        path = Path(possible_path)
        if not path.exists():
            # 如果提取出来的不是一个真实路径，则跳过
            continue

        if path.name.endswith(".description"):
            description_file = str(path)
        elif path.suffix.lower() in {".vtt", ".srt", ".ass"}:
            subtitle_files.append(str(path))
        elif path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
            seg_match = _match_segment_from_name(path.name)
            if seg_match:
                closest_seg = min(
                    segments,
                    key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1])
                )
                clip_files[closest_seg].append(str(path))

    # --- 回退方案：扫描输出目录，填补缺失信息 --------------------------------
    try:
        for file_name in os.listdir(video_output_dir):
            file_path = os.path.join(video_output_dir, file_name)
            path = Path(file_path)
            if path.name.endswith(".description") and not description_file:
                description_file = file_path
            elif path.suffix.lower() in {".vtt", ".srt", ".ass"} and file_path not in subtitle_files:
                subtitle_files.append(file_path)
            elif path.suffix.lower() in {".mp4", ".m4a", ".webm", ".mkv"}:
                seg_match = _match_segment_from_name(path.name)
                if seg_match:
                    closest_seg = min(
                        segments,
                        key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1])
                    )
                    if file_path not in clip_files[closest_seg]:
                        clip_files[closest_seg].append(file_path)
    except FileNotFoundError:
        pass

    # 组装最终结果
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


def parse_ytdlp_output_full_video(
    output: str,
    video_id: str,
    video_output_dir: str,
) -> Dict:
    """
    解析 yt-dlp 的输出，提取完整视频的文件路径。
    """
    files_from_stdout = [line.strip() for line in output.splitlines() if line.strip()]

    description_file: str = ""
    subtitle_files: List[str] = []
    video_file: str = ""
    audio_file: str = ""

    for raw in files_from_stdout:
        possible_path = raw.split(": ")[-1].strip()
        path = Path(possible_path)
        if not path.exists():
            continue

        if path.name.endswith(".description"):
            description_file = str(path)
        elif path.suffix.lower() in {".vtt", ".srt", ".ass"}:
            subtitle_files.append(str(path))
        elif path.suffix.lower() == ".m4a":
            audio_file = str(path)
        elif path.suffix.lower() in {".mp4", ".webm", ".mkv"}:
            video_file = str(path)

    # 回退方案：扫描输出目录
    try:
        for file_name in os.listdir(video_output_dir):
            file_path = os.path.join(video_output_dir, file_name)
            path = Path(file_path)
            if path.name.endswith(".description") and not description_file:
                description_file = file_path
            elif path.suffix.lower() in {".vtt", ".srt", ".ass"} and file_path not in subtitle_files:
                subtitle_files.append(file_path)
            elif path.suffix.lower() == ".m4a" and not audio_file:
                audio_file = file_path
            elif path.suffix.lower() in {".mp4", ".webm", ".mkv"} and not video_file:
                video_file = file_path
    except FileNotFoundError:
        pass

    return {
        "video_file": video_file,
        "audio_file": audio_file,
        "description_file": description_file,
        "subtitle_files": subtitle_files,
    }


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


def download_with_ytdlp(
    input_json_path: str,
    output_dir: str,
    cookies_path: Optional[str],
    browser: Optional[str],
    extractor_args: Optional[str],
    limit: Optional[int],
    workers: int,
) -> None:
    """
    下载完整视频（而非片段）。
    """
    json_logs_dir = os.path.join(output_dir, "json_logs")
    safe_mkdir(json_logs_dir)

    # 收集待下载的视频
    videos_to_download = []
    total_videos = 0
    skipped_due_to_log = 0

    for (video_id, url, category) in iter_videos_from_json(input_json_path):
        total_videos += 1
        if limit is not None and limit >= 0 and len(videos_to_download) >= limit:
            break

        # 检查是否已下载
        if is_video_downloaded(video_id, output_dir):
            skipped_due_to_log += 1
            continue

        videos_to_download.append((video_id, url, category))

    print(f"Total videos found: {total_videos}")
    print(f"Skipped (already downloaded): {skipped_due_to_log}")
    print(f"Videos to download: {len(videos_to_download)}")

    if not videos_to_download:
        print("No new videos to download.")
        return

    safe_mkdir(output_dir)
    logs_dir = os.path.join(output_dir, "logs")
    safe_mkdir(logs_dir)
    failed_urls_file = os.path.join(logs_dir, "failed_urls.txt")

    # 加载已知不可用的 URL
    unavailable_urls = load_unavailable_urls(failed_urls_file)
    if unavailable_urls:
        print(f"Loaded {len(unavailable_urls)} permanently unavailable URLs from logs.")

    # 初始化进度条 - 使用更详细的列
    progress = Progress(
        TextColumn("[bold blue]{task.fields[video_id]}", justify="left"),
        BarColumn(bar_width=30),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("{task.fields[status]}", justify="left"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )

    progress.__enter__()
    
    # 创建任务字典来跟踪每个视频的进度
    video_tasks: Dict[str, TaskID] = {}
    
    # 添加总体进度任务
    task_overall = progress.add_task(
        "Overall Progress",
        total=len(videos_to_download),
        video_id="[ALL]",
        status="Initializing..."
    )

    # 提交下载任务
    with ProcessPoolExecutor(max_workers=max(1, workers)) as executor:
        futures_map = {}
        for (video_id, url, category) in videos_to_download:
            # 为每个视频创建进度任务
            task_id = progress.add_task(
                f"Video {video_id}",
                total=100,
                video_id=video_id[:12] + "..." if len(video_id) > 12 else video_id,
                status="Pending..."
            )
            video_tasks[video_id] = task_id
            
            # 跳过已知不可用的 URL
            if url in unavailable_urls:
                reason = "Skipping probe: URL previously marked as permanently unavailable."
                with open(failed_urls_file, "a", encoding="utf-8") as furl:
                    furl.write(f"{url}\t{reason}\n")
                progress.update(
                    task_id,
                    completed=100,
                    status="[red]Skipped (unavailable)"
                )
                progress.update(task_overall, advance=1)
                continue

            # 探测可用性
            progress.update(task_id, completed=10, status="[yellow]Probing...")
            ok, reason = probe_url_availability(url, cookies_path, browser, extractor_args)
            if not ok:
                with open(failed_urls_file, "a", encoding="utf-8") as furl:
                    furl.write(f"{url}\t{reason}\n")
                progress.update(
                    task_id,
                    completed=100,
                    status="[red]Failed probe"
                )
                progress.update(task_overall, advance=1)
                continue

            # 创建分类目录
            category_dir = os.path.join(output_dir, category)
            safe_mkdir(category_dir)
            
            progress.update(task_id, completed=20, status="[cyan]Downloading...")
            future = executor.submit(
                run_yt_dlp_full_video,
                url, video_id, category_dir, cookies_path, browser, extractor_args
            )
            futures_map[future] = (video_id, url, category)

        # 回收结果
        for fut in as_completed(futures_map):
            video_id, url, category = futures_map[fut]
            task_id = video_tasks[video_id]
            
            progress.update(task_id, completed=80, status="[magenta]Processing...")

            try:
                rc, msg = fut.result()
            except Exception as exc:
                rc, msg = 1, f"Task failed with exception: {exc}"

            if rc != 0:
                print(f"[yt-dlp] Failed: {url} | {msg}")
                with open(failed_urls_file, "a", encoding="utf-8") as furl:
                    furl.write(f"{url}\t{msg}\n")
                progress.update(
                    task_id,
                    completed=100,
                    status="[red]Failed"
                )
                progress.update(task_overall, advance=1)
                continue

            # 解析下载结果
            progress.update(task_id, completed=90, status="[yellow]Saving logs...")
            category_dir = os.path.join(output_dir, category)
            parsed_result = parse_ytdlp_output_full_video(msg, video_id, category_dir)

            # 记录日志
            log_filename = f"{video_id}.json"
            log_file = os.path.join(json_logs_dir, log_filename)

            log_data = {
                "source_info": {
                    "video_id": video_id,
                    "url": url,
                    "category": category,
                },
                "download_info": {
                    **parsed_result,
                    "status": "success",
                    "error": "",
                    "download_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                }
            }
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(log_data, f, ensure_ascii=False, indent=2)

            # 检查是否有文件生成
            if not parsed_result.get("video_file"):
                with open(failed_urls_file, "a", encoding="utf-8") as furl:
                    furl.write(f"{url}\tNo video file generated\n")
                progress.update(
                    task_id,
                    completed=100,
                    status="[red]No file"
                )
            else:
                progress.update(
                    task_id,
                    completed=100,
                    status="[green]✓ Complete"
                )
            
            progress.update(task_overall, advance=1)

    progress.__exit__(None, None, None)


def ensure_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg 未找到，安装后再试（yt-dlp 的分段裁切需要 ffmpeg）。")


def cleanup_final_files(output_dir: str) -> None:
    """根据所有 json_logs 清理未被记录的文件，保持输出目录整洁。"""
    print("\nStarting final file cleanup...")
    json_logs_dir = os.path.join(output_dir, "json_logs")
    if not os.path.isdir(json_logs_dir):
        print("json_logs directory not found, skipping cleanup.")
        return

    # 1. 收集所有记录在案的文件路径
    recorded_files = set()
    json_files = glob.glob(os.path.join(json_logs_dir, "*.json"))
    for log_file in json_files:
        recorded_files.add(os.path.abspath(log_file)) # 把日志本身也加入白名单
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            info = data.get("download_info", {})
            if info.get("status") != "success":
                continue

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

    # 2. 遍历输出目录，删除未记录的文件
    print(f"Found {len(recorded_files)} files recorded in logs. Scanning for unrecorded files...")
    deleted_count = 0
    for root, _, files in os.walk(output_dir):
        for file in files:
            # 跳过原始失败日志
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download full videos specified in a JSON file with yt-dlp on Windows."
        )
    )
    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(os.getcwd(), "videos.json"),
        help="Path to the JSON file containing video information.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(os.getcwd(), "clips_output"),
        help="Directory to store the downloaded clips.",
    )
    parser.add_argument(
        "--mode",
        choices=["ytdlp"],
        default="ytdlp",
        help="仅支持 'ytdlp'，已移除 video2dataset 支持。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many clip segments (for testing).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Concurrent workers for yt-dlp mode.",
    )
    parser.add_argument(
        "--cookies",
        type=str,
        default=os.path.join(os.getcwd(), "cookies.txt"),
        help="Path to cookies.txt to pass to yt-dlp if present.",
    )
    parser.add_argument(
        "--browser",
        type=str,
        choices=["edge", "chrome", "firefox", "chromium", "brave", "vivaldi", "opera"],
        default=None,
        help="Use --cookies-from-browser <browser> for YouTube auth (recommended).",
    )
    parser.add_argument(
        "--extractor_args",
        type=str,
        default=None,
        help="Pass through to yt-dlp --extractor-args, e.g. 'youtube:player_client=android'",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Run cleanup process after downloading to remove unlogged files.",
    )
    # Debug
    parser.add_argument('--debug', action='store_true', help='whether to wait for debugger attach')
    

    return parser.parse_args()


def main() -> None:
    ensure_ffmpeg()
    args = parse_args()
    if args.debug:
        import debugpy
        # Use a different port if 6666 is occupied
        port = 6668
        debugpy.listen(port)
        print(f"--- Python script started in debug mode. ---")
        print(f"--- Waiting for debugger to attach on port {port}... ---")
        # Execution will pause here until you attach the debugger
        debugpy.wait_for_client()
        print(f"--- Debugger attached. Resuming execution. ---")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start downloading with yt-dlp")

    download_with_ytdlp(
        input_json_path=args.input,
        output_dir=args.output,
        cookies_path=args.cookies if os.path.exists(args.cookies) else None,
        browser=args.browser,
        extractor_args=args.extractor_args,
        limit=args.limit,
        workers=args.workers,
    )
    print("yt-dlp clip downloads completed.")

    if args.cleanup:
        cleanup_final_files(args.output)


if __name__ == "__main__":
    main()


