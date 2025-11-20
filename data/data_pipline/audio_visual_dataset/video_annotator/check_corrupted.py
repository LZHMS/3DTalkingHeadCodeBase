#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复损坏的视频片段
检查并重新生成小于10KB的可疑文件
"""

from pathlib import Path
import json

# 配置
OUTPUT_CLIPS_DIR = Path(__file__).parent.parent / 'output_clips'
MIN_FILE_SIZE = 10000  # 小于10KB认为可能损坏

def check_corrupted_files():
    """检查所有可能损坏的视频文件"""
    corrupted = []
    
    print("扫描损坏的视频文件...")
    print(f"扫描目录: {OUTPUT_CLIPS_DIR}")
    print(f"最小文件大小阈值: {MIN_FILE_SIZE} 字节 ({MIN_FILE_SIZE/1024:.1f} KB)")
    print("-" * 60)
    
    for video_file in OUTPUT_CLIPS_DIR.rglob("*.mp4"):
        file_size = video_file.stat().st_size
        if file_size < MIN_FILE_SIZE:
            corrupted.append({
                'path': video_file,
                'size': file_size,
                'relative_path': video_file.relative_to(OUTPUT_CLIPS_DIR)
            })
            print(f"❌ {video_file.relative_to(OUTPUT_CLIPS_DIR)}")
            print(f"   大小: {file_size} 字节 ({file_size/1024:.2f} KB)")
    
    print("-" * 60)
    print(f"找到 {len(corrupted)} 个可能损坏的文件")
    
    return corrupted

def list_scene_info(video_dir):
    """列出视频目录的场景信息"""
    scene_info_path = video_dir / 'scene_info.json'
    
    if not scene_info_path.exists():
        return None
    
    scenes = []
    with open(scene_info_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                scenes.append(json.loads(line.strip()))
    
    return scenes

def main():
    print()
    print("=" * 60)
    print("视频损坏检测工具")
    print("=" * 60)
    print()
    
    corrupted = check_corrupted_files()
    
    if not corrupted:
        print("\n✓ 没有发现损坏的文件！")
        return
    
    print("\n建议操作：")
    print("-" * 60)
    
    # 按目录分组
    by_dir = {}
    for item in corrupted:
        video_dir = item['path'].parent
        if video_dir not in by_dir:
            by_dir[video_dir] = []
        by_dir[video_dir].append(item)
    
    for video_dir, files in by_dir.items():
        rel_dir = video_dir.relative_to(OUTPUT_CLIPS_DIR)
        print(f"\n目录: {rel_dir}")
        print(f"损坏文件数: {len(files)}")
        
        # 显示场景信息
        scenes = list_scene_info(video_dir)
        if scenes:
            print(f"总场景数: {len(scenes)}")
            for f in files:
                scene_id = f['path'].stem  # 不带扩展名的文件名
                scene = next((s for s in scenes if s['id'] == scene_id), None)
                if scene:
                    print(f"  - {scene_id}: {scene.get('durations', 'N/A')}")
        
        print(f"\n  选项1: 删除这些损坏的文件和对应的scene_info记录")
        print(f"  选项2: 保留备份，重新从原始视频截取（如果有original-video信息）")
    
    print("\n" + "=" * 60)
    print("注意事项：")
    print("=" * 60)
    print("1. 这些文件可能是使用 -c copy 模式截取时失败的")
    print("2. 已修改server.py为重新编码模式，之后截取应该正常")
    print("3. 建议：")
    print("   - 删除损坏的scene_2.mp4和scene_2.m4a")
    print("   - 从scene_info.json中移除对应记录")
    print("   - 或者重新从原始视频截取这些片段")
    print()

if __name__ == '__main__':
    main()
