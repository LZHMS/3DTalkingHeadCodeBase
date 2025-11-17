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


from .utils import clip_success_downloaded, get_video_id, load_unavailable_urls, \
check_url_availability, get_yt_dlp_base_cmd, seconds_to_time_string, _match_segment_from_name


class YTDLPDownloader:
    def __init__(self, args):
        self.args = args
        self.full_download = args.full_download

        # logging
        self.logs_dir = os.path.join(args.output_dir, "download_logs")
        self.json_dir = os.path.join(args.logs_dir, "success_logs")
        os.makedirs(self.json_dir, exist_ok=True)

        self.failed_urls_file = os.path.join(self.logs_dir, "failed_urls.txt")
        self.failed_segments_file = os.path.join(self.logs_dir, "failed_segments.txt")

    def parsing_json(self, input_json_path: str):
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
        segments_to_download, total_segments, skipped_success_segments = 0, 0, 0
        self.url2cate = defaultdict(str)
        self.url2segments: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

        for (url, start, end, category) in self.parsing_json(args.input_json_path):
            if args.limit is not None and args.limit >= 0 and segments_to_download >= args.limit:
                break
            
            self.url2cate[url], total_segments = category, total_segments + 1
            if clip_success_downloaded(url, start, end, self.json_dir):
                skipped_success_segments += 1
                continue

            self.url2segments[url].append((start, end))
            segments_to_download += 1
                
        print(f"Total segments found: {total_segments}")
        print(f"Skipped (already downloaded): {skipped_success_segments}")
        print(f"Segments to download: {segments_to_download}")

        if segments_to_download == 0:
            return

        self.unavailable_urls = load_unavailable_urls(self.failed_urls_file)
        if self.unavailable_urls:
            print(f"Loaded {len(self.unavailable_urls)} permanently unavailable URLs from logs.")

        # initialize progress bar
        self.progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

        self.progress.__enter__()  # 手动进入，使其可跨多个 with 范围外使用
        self.task_urls = self.progress.add_task("URLs", total=len(self.url2segments))
        self.task_segments = self.progress.add_task("Segments", total=segments_to_download)
        

    def run_processing(self):
        with ProcessPoolExecutor(max_workers=max(1, self.args.workers)) as executor:
            self.futures_map = {}
            for url, segs in self.url2segments.items():
                # 新增：跳过已知的不可用 URL，避免重复探测
                if url in self.unavailable_urls:
                    reason = "Skipping probe: URL previously marked as permanently unavailable."
                    with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                        for (s, e) in segs:
                            fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
                    # 进度条同样推进
                    self.progress.update(self.task_urls, advance=1)
                    self.progress.update(self.task_segments, advance=len(segs))
                    continue
                
                available, reason = check_url_availability(url, self.args.cookies_path, self.args.browser, self.args.extractor_args)
                if not available:
                    with open(self.failed_urls_file, "a", encoding="utf-8") as furl:
                        furl.write(f"{url}\t{reason}\n")
                    for (s, e) in segs:
                        with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                            fseg.write(f"SKIP\t{url}\t{s:.3f}\t{e:.3f}\t{reason}\n")
                    self.progress.update(self.task_urls, advance=1)
                    self.progress.update(self.task_segments, advance=len(segs))
                else:
                    segs = sorted(set(segs))

                future = executor.submit(self.run_yt_dlp_download_segments, url, segs)
                self.futures_map[future] = (url, segs)
    
    def handle_results(self):
        # handle multi-process results
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
                print(f"[yt-dlp] Failed: {url0} | {msg}")
                with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                    for (s, e) in segs0:
                        fseg.write(f"FAIL\t{url0}\t{s:.3f}\t{e:.3f}\t{msg}\n")
            else:
                parsed_results = self.parse_ytdlp_output(msg, segs0, video_id)

                for seg_tuple, files_info in parsed_results.items():
                    start, end = seg_tuple
                    log_filename = f"{video_id}_full.json".replace(":", "-") if self.full_download \
                                    else f"{video_id}_{start:.3f}_{end:.3f}.json".replace(":", "-")
                    log_file = os.path.join(self.json_logs_dir, log_filename)

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

                # 检查哪些分段没有成功产物
                succeeded_segs = set(parsed_results.keys())
                failed_segs = [s for s in segs0 if s not in succeeded_segs]
                if failed_segs:
                    with open(self.failed_segments_file, "a", encoding="utf-8") as fseg:
                        for (s, e) in failed_segs:
                            fseg.write(f"FAIL\t{url0}\t{s:.3f}\t{e:.3f}\tNo output file generated\n")


        # 关闭进度条
        self.progress.__exit__(None, None, None)

    def parse_ytdlp_output(self, output: str, segments: List[Tuple[float, float]],
        video_id: str) -> Dict[Tuple[float, float], Dict]:
        """
        解析 yt-dlp 的输出，将文件路径与原始分段关联。
        现在不仅解析 stdout，还会回退到扫描输出目录，以确保拿到完整的
        audio / description / subtitle 信息。
        支持完整视频下载（segments 为空列表时）。
        """

        files_from_stdout = [line.strip() for line in output.splitlines() if line.strip()]

        # 分类容器
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
                    closest_seg = min(segments, key=lambda s: abs(s[0]-seg_match[0])+abs(s[1]-seg_match[1]))
                    clip_files[closest_seg].append(str(possible_path))

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

    def run_yt_dlp_download_segments(self, url, segments) -> Tuple[int, str]:
        """
        对同一 URL 的多个片段，合并为一次 yt-dlp 调用（多个 --download-sections）。
        产物文件名使用 section 变量，避免覆盖。
        如果 segments 为空列表，则下载完整视频。
        """
        base_cmd, err_info = get_yt_dlp_base_cmd(self.args.cookies_path, self.args.browser)
        assert base_cmd, "Unable to locate yt-dlp"

        video_id = get_video_id(url)
        video_output_dir = os.path.join(self.output_dir, self.url2cate[url], video_id)
        os.makedirs(video_output_dir, exist_ok=True)

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
        
        format_trials = [
            {
                # trial 0: 首选格式（更理想）
                "format": "bestvideo[ext=mp4][vcodec!=none]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=none]",
                "extra_flags": [
                    "--merge-output-format", "mp4",
                ],
                "continue_flag": ["--no-continue", "--no-overwrites"],
            },
            {
                # trial 1: 回退格式（更宽松）
                "format": "bestvideo[ext=mp4][vcodec!=none]+bestaudio[ext=m4a]/best[ext=mp4][vcodec!=none]",
                "extra_flags": [
                    "--remux-video", "mp4",
                ],
                "continue_flag": ["-c", "--no-overwrites"],
            },
        ]
        section_args: List[str] = []
        if not self.full_download:
            for (start, end) in segments:
                s_str, e_str = seconds_to_time_string(start), seconds_to_time_string(end)
                section_args.extend(["--download-sections", f"*{s_str}-{e_str}"])

        last_err = ""
        for trial in format_trials:
            cmd: List[str] = [
                *base_cmd,
                "-4", "--ignore-config", "--no-playlist",
                "--retries", "10", "--fragment-retries", "10",
                "--concurrent-fragments", "8", "-N", "4",
                "--no-warnings", "--restrict-filenames",
                *trial["continue_flag"],
                "--print", "after_move:filepath",  # 打印最终文件路径
                "--write-subs", "--write-auto-subs", "--write-description",
                "--extract-audio", "--audio-format", "m4a",
                "--audio-quality", "0", "--keep-video",
                "--no-keep-fragments", "--clean-info-json",
                "-o", output_template,
                "-f", trial["format"],
                *trial["extra_flags"],
            ]

            if self.args.strict_cuts:
                cmd.append("--force-keyframes-at-cuts")
            if self.args.extractor_args:
                cmd.extend(["--extractor-args", self.args.extractor_args])

            cmd.extend(section_args)
            cmd.append(url)

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                return 1, f"yt-dlp failed: {exc}"

            if proc.returncode == 0:
                return 0, proc.stdout.strip()

            last_err = (proc.stderr.strip() or proc.stdout.strip())
            if "Requested format is not available" not in last_err:
                break

        return 1, last_err or "yt-dlp failed with unknown error"

    def cleanup_final_files(self) -> None:
        """根据所有 json_logs 清理未被记录的文件，保持输出目录整洁。"""
        print("\nStarting final file cleanup...")

        # 1. 收集所有记录在案的文件路径
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
        for root, _, files in os.walk(args.output_dir):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download clips specified in a large JSON file directly with yt-dlp sections on Windows.")
    parser.add_argument("--input", type=str, default=os.path.join(os.getcwd(), "filtered_video_clips.json"), help="Path to the large JSON file containing 'Video Link', 'start-time', 'end-time' fields.")
    parser.add_argument("--output", type=str, default=os.path.join(os.getcwd(), "clips_output"), help="Directory to store the downloaded clips.")
    parser.add_argument("--mode", choices=["ytdlp"], default="ytdlp", help="仅支持 'ytdlp'，已移除 video2dataset 支持。")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many clip segments (for testing).")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent workers for yt-dlp mode.")
    parser.add_argument("--cookies", type=str, default=os.path.join(os.getcwd(), "cookies.txt"), help="Path to cookies.txt to pass to yt-dlp if present.")
    parser.add_argument("--browser", type=str, choices=["edge", "chrome", "firefox", "chromium", "brave", "vivaldi", "opera"], default=None, help="Use --cookies-from-browser <browser> for YouTube auth (recommended).")
    parser.add_argument("--strict_cuts", type=bool, default=True, help="")
    parser.add_argument("--extractor_args", type=str, default=None, help="Pass through to yt-dlp --extractor-args, e.g. 'youtube:player_client=android'")
    parser.add_argument("--cleanup", action="store_true", help="Run cleanup process after downloading to remove unlogged files.")

    args = parser.parse_args()
    ytdlp_dl = YTDLPDownloader(args)

    print("Starting yt-dlp clip downloads...\n")
    ytdlp_dl.prepare_download_task()
    ytdlp_dl.run_processing()
    ytdlp_dl.handle_results()
    ytdlp_dl.cleanup_final_files()
    print("yt-dlp clip downloads completed.")


