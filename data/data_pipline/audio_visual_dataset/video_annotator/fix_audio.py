#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查并修复损坏的音频文件
从原始音频文件中重新截取对应片段
"""

from pathlib import Path
import subprocess
import json

OUTPUT_CLIPS_DIR = Path(__file__).parent.parent / 'output_clips'
MIN_AUDIO_SIZE = 10000  # 小于10KB认为可能损坏

def check_audio_files():
    """检查所有音频文件"""
    corrupted = []
    
    print("扫描损坏的音频文件...")
    print(f"最小文件大小阈值: {MIN_AUDIO_SIZE} 字节 ({MIN_AUDIO_SIZE/1024:.1f} KB)")
    print("-" * 70)
    
    for video_file in OUTPUT_CLIPS_DIR.rglob("*.mp4"):
        audio_file = video_file.with_suffix('.m4a')
        
        # 检查音频文件是否存在且大小正常
        if audio_file.exists():
            audio_size = audio_file.stat().st_size
            video_size = video_file.stat().st_size
            
            # 如果音频太小，认为可能损坏
            if audio_size < MIN_AUDIO_SIZE:
                # 读取scene_info获取原始音频和时间信息
                scene_info_path = video_file.parent / 'scene_info.json'
                scene_info = None
                
                if scene_info_path.exists():
                    scene_id = video_file.stem
                    with open(scene_info_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                info = json.loads(line.strip())
                                if info['id'] == scene_id:
                                    scene_info = info
                                    break
                
                corrupted.append({
                    'video': video_file,
                    'audio': audio_file,
                    'video_size': video_size,
                    'audio_size': audio_size,
                    'relative_path': video_file.relative_to(OUTPUT_CLIPS_DIR),
                    'scene_info': scene_info
                })
                print(f"[X] {video_file.relative_to(OUTPUT_CLIPS_DIR)}")
                print(f"   视频: {video_size} 字节 ({video_size/1024:.2f} KB)")
                print(f"   音频: {audio_size} 字节 ({audio_size/1024:.2f} KB) <- 太小!")
                if scene_info:
                    print(f"   原始音频: {scene_info.get('original-audio', 'N/A')}")
                    print(f"   时间范围: {scene_info.get('start-time', 0):.2f}s - {scene_info.get('end-time', 0):.2f}s")
        else:
            # 音频文件不存在
            print(f"[!] {video_file.relative_to(OUTPUT_CLIPS_DIR)}")
            print(f"   缺少音频文件!")
            
            # 读取scene_info
            scene_info_path = video_file.parent / 'scene_info.json'
            scene_info = None
            
            if scene_info_path.exists():
                scene_id = video_file.stem
                with open(scene_info_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            info = json.loads(line.strip())
                            if info['id'] == scene_id:
                                scene_info = info
                                break
            
            corrupted.append({
                'video': video_file,
                'audio': None,
                'video_size': video_file.stat().st_size,
                'audio_size': 0,
                'relative_path': video_file.relative_to(OUTPUT_CLIPS_DIR),
                'scene_info': scene_info
            })
    
    print("-" * 70)
    print(f"找到 {len(corrupted)} 个需要修复的音频文件")
    
    return corrupted

def extract_audio_from_original(scene_info, output_audio_file):
    """从原始音频文件中截取片段"""
    try:
        # 获取原始音频路径
        original_audio = scene_info.get('original-audio')
        if not original_audio:
            return False, "scene_info中没有original-audio信息"
        
        # 转换为绝对路径
        original_audio_path = OUTPUT_CLIPS_DIR.parent / original_audio
        
        if not original_audio_path.exists():
            return False, f"原始音频文件不存在: {original_audio_path}"
        
        # 获取时间范围
        start_time = scene_info.get('start-time', 0)
        end_time = scene_info.get('end-time', 0)
        duration = end_time - start_time
        
        if duration <= 0:
            return False, f"无效的时间范围: {start_time} - {end_time}"
        
        # 使用FFmpeg截取音频
        cmd = [
            'ffmpeg', '-i', str(original_audio_path),
            '-ss', str(start_time),
            '-t', str(duration),
            '-c:a', 'aac',  # 音频编码
            '-b:a', '128k',  # 音频比特率
            str(output_audio_file),
            '-y'  # 覆盖已存在的文件
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            new_size = output_audio_file.stat().st_size
            return True, new_size
        else:
            return False, result.stderr
    except Exception as e:
        return False, str(e)

def fix_audio_files(corrupted_list, dry_run=True):
    """修复损坏的音频文件"""
    
    if dry_run:
        print("\n=== 预览模式（不会实际修改文件） ===")
    else:
        print("\n=== 开始修复音频文件 ===")
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for i, item in enumerate(corrupted_list, 1):
        video_file = item['video']
        audio_file = item['audio'] if item['audio'] else video_file.with_suffix('.m4a')
        scene_info = item.get('scene_info')
        
        print(f"\n[{i}/{len(corrupted_list)}] {item['relative_path']}")
        
        if not scene_info:
            print(f"  [SKIP] 缺少scene_info信息")
            skipped_count += 1
            continue
        
        original_audio = scene_info.get('original-audio')
        if not original_audio:
            print(f"  [SKIP] scene_info中没有original-audio")
            skipped_count += 1
            continue
        
        if dry_run:
            print(f"  将从原始音频截取: {original_audio}")
            print(f"    时间: {scene_info.get('start-time', 0):.2f}s - {scene_info.get('end-time', 0):.2f}s")
            print(f"    输出: {audio_file.name}")
        else:
            print(f"  从原始音频截取...")
            success, result = extract_audio_from_original(scene_info, audio_file)
            
            if success:
                print(f"  [OK] 成功! 新音频大小: {result} 字节 ({result/1024:.2f} KB)")
                success_count += 1
            else:
                print(f"  [FAIL] 失败: {result}")
                failed_count += 1
    
    print("\n" + "=" * 70)
    if dry_run:
        print("预览完成。使用 --fix 参数实际修复文件。")
        print(f"可修复: {len(corrupted_list) - skipped_count}")
        print(f"将跳过: {skipped_count}")
    else:
        print("修复完成!")
        print(f"成功: {success_count}")
        print(f"失败: {failed_count}")
        print(f"跳过: {skipped_count}")
    print("=" * 70)

def main():
    import sys
    
    print()
    print("=" * 70)
    print("音频文件检查和修复工具")
    print("=" * 70)
    print()
    
    # 检查损坏的文件
    corrupted = check_audio_files()
    
    if not corrupted:
        print("\n[OK] 没有发现损坏的音频文件!")
        return
    
    print()
    print("发现问题:")
    print(f"  - {len([x for x in corrupted if x['audio'] is None])} 个缺少音频文件")
    print(f"  - {len([x for x in corrupted if x['audio'] and x['audio_size'] < MIN_AUDIO_SIZE])} 个音频文件太小")
    print()
    
    # 检查是否有--fix参数
    if '--fix' in sys.argv:
        confirm = input("确定要从视频中重新提取音频吗？(yes/no): ").strip().lower()
        if confirm != 'yes':
            print("操作已取消")
            return
        
        fix_audio_files(corrupted, dry_run=False)
    else:
        print("使用 --fix 参数来实际修复这些文件")
        print("示例: python fix_audio.py --fix")
        print()
        fix_audio_files(corrupted, dry_run=True)

if __name__ == '__main__':
    main()
