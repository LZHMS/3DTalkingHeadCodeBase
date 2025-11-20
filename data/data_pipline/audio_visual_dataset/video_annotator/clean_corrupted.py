#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
清理损坏的视频片段
删除小于10KB的文件并更新scene_info.json和video_scece_info.txt
"""

from pathlib import Path
import json
import shutil

# 配置
OUTPUT_CLIPS_DIR = Path(__file__).parent.parent / 'output_clips'
MIN_FILE_SIZE = 10000  # 小于10KB认为可能损坏
BACKUP_DIR = OUTPUT_CLIPS_DIR.parent / 'corrupted_backup'

def seconds_to_time_str(seconds):
    """将秒数转换为时间字符串"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"

def clean_corrupted_files(backup=True):
    """清理所有损坏的文件"""
    
    if backup:
        BACKUP_DIR.mkdir(exist_ok=True)
        print(f"备份目录: {BACKUP_DIR}")
    
    total_cleaned = 0
    total_updated_dirs = 0
    
    # 按目录处理
    processed_dirs = set()
    
    for video_file in OUTPUT_CLIPS_DIR.rglob("*.mp4"):
        file_size = video_file.stat().st_size
        
        if file_size < MIN_FILE_SIZE:
            video_dir = video_file.parent
            
            if video_dir not in processed_dirs:
                processed_dirs.add(video_dir)
                result = process_video_dir(video_dir, backup)
                if result > 0:
                    total_updated_dirs += 1
                    total_cleaned += result
    
    return total_cleaned, total_updated_dirs

def process_video_dir(video_dir, backup=True):
    """处理单个视频目录"""
    
    rel_path = video_dir.relative_to(OUTPUT_CLIPS_DIR)
    print(f"\n处理目录: {rel_path}")
    
    # 查找损坏的文件
    corrupted_files = []
    for video_file in video_dir.glob("*.mp4"):
        if video_file.stat().st_size < MIN_FILE_SIZE:
            corrupted_files.append(video_file.stem)  # 不带扩展名
    
    if not corrupted_files:
        return 0
    
    print(f"  找到 {len(corrupted_files)} 个损坏的片段: {', '.join(corrupted_files)}")
    
    # 备份
    if backup:
        backup_subdir = BACKUP_DIR / rel_path
        backup_subdir.mkdir(parents=True, exist_ok=True)
        
        for scene_id in corrupted_files:
            for ext in ['.mp4', '.m4a']:
                src_file = video_dir / f"{scene_id}{ext}"
                if src_file.exists():
                    dst_file = backup_subdir / src_file.name
                    shutil.copy2(src_file, dst_file)
        
        # 备份原始scene_info文件
        scene_info_path = video_dir / 'scene_info.json'
        if scene_info_path.exists():
            shutil.copy2(scene_info_path, backup_subdir / 'scene_info.json.bak')
        
        video_scene_info_path = video_dir / 'video_scece_info.txt'
        if video_scene_info_path.exists():
            shutil.copy2(video_scene_info_path, backup_subdir / 'video_scece_info.txt.bak')
        
        print(f"  ✓ 已备份到: {backup_subdir}")
    
    # 删除损坏的文件
    for scene_id in corrupted_files:
        for ext in ['.mp4', '.m4a']:
            file_path = video_dir / f"{scene_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                print(f"  ✓ 删除: {file_path.name}")
    
    # 更新scene_info.json
    scene_info_path = video_dir / 'scene_info.json'
    if scene_info_path.exists():
        scenes = []
        with open(scene_info_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    scene = json.loads(line.strip())
                    if scene['id'] not in corrupted_files:
                        scenes.append(scene)
        
        with open(scene_info_path, 'w', encoding='utf-8') as f:
            for scene in scenes:
                f.write(json.dumps(scene) + '\n')
        
        print(f"  ✓ 更新 scene_info.json (保留 {len(scenes)} 个场景)")
    
    # 更新video_scece_info.txt
    video_scene_info_path = video_dir / 'video_scece_info.txt'
    if video_scene_info_path.exists() and scene_info_path.exists():
        with open(video_scene_info_path, 'w', encoding='utf-8') as f:
            for i, scene in enumerate(scenes, 1):
                start_time = seconds_to_time_str(scene['start-time'])
                end_time = seconds_to_time_str(scene['end-time'])
                f.write(f"scene {i} infos: start_time {start_time}, end_time {end_time}\n")
        
        print(f"  ✓ 更新 video_scece_info.txt")
    
    return len(corrupted_files)

def main():
    print()
    print("=" * 60)
    print("清理损坏的视频文件")
    print("=" * 60)
    print()
    
    choice = input("是否在删除前备份文件？(y/n，默认y): ").strip().lower()
    backup = choice != 'n'
    
    print()
    confirm = input("确定要清理所有损坏的文件吗？(yes/no): ").strip().lower()
    if confirm != 'yes':
        print("操作已取消")
        return
    
    print()
    print("开始清理...")
    print("-" * 60)
    
    total_cleaned, total_dirs = clean_corrupted_files(backup)
    
    print()
    print("=" * 60)
    print("清理完成！")
    print("=" * 60)
    print(f"处理的目录数: {total_dirs}")
    print(f"清理的文件数: {total_cleaned}")
    
    if backup:
        print(f"\n备份位置: {BACKUP_DIR}")
        print("如需恢复，可从备份目录中手动复制")
    
    print()

if __name__ == '__main__':
    main()
