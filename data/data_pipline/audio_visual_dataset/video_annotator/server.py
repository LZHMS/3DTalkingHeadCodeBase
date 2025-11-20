#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地视频片段管理服务器
提供本地视频片段的浏览、删除和截取功能
"""

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import os
import json
import shutil
import subprocess
from pathlib import Path

app = Flask(__name__)
CORS(app)

# 配置路径
OUTPUT_CLIPS_DIR = Path(__file__).parent.parent / 'output_clips'

def parse_time_to_seconds(time_str):
    """将时间字符串转换为秒数"""
    parts = time_str.split(':')
    if len(parts) == 3:
        h, m, s = map(float, parts)
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = map(float, parts)
        return m * 60 + s
    else:
        return float(parts[0])

def seconds_to_time_str(seconds):
    """将秒数转换为时间字符串 HH:MM:SS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """获取所有分类"""
    try:
        categories = [d.name for d in OUTPUT_CLIPS_DIR.iterdir() if d.is_dir()]
        return jsonify({'categories': categories})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/videos/<category>', methods=['GET'])
def get_videos(category):
    """获取指定分类下的所有视频"""
    try:
        category_path = OUTPUT_CLIPS_DIR / category
        if not category_path.exists():
            return jsonify({'error': 'Category not found'}), 404
        
        videos = []
        for video_dir in category_path.iterdir():
            if video_dir.is_dir():
                scene_info_path = video_dir / 'scene_info.json'
                video_scene_info_path = video_dir / 'video_scece_info.txt'
                
                # 读取场景信息
                scenes = []
                if scene_info_path.exists():
                    with open(scene_info_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                scenes.append(json.loads(line.strip()))
                
                videos.append({
                    'id': video_dir.name,
                    'path': str(video_dir.relative_to(OUTPUT_CLIPS_DIR)),
                    'scenes': scenes,
                    'sceneCount': len(scenes)
                })
        
        return jsonify({'videos': videos})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<path:video_path>/scenes', methods=['GET'])
def get_video_scenes(video_path):
    """获取视频的所有场景"""
    try:
        video_dir = OUTPUT_CLIPS_DIR / video_path
        if not video_dir.exists():
            return jsonify({'error': 'Video not found'}), 404
        
        scene_info_path = video_dir / 'scene_info.json'
        scenes = []
        
        if scene_info_path.exists():
            with open(scene_info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        scenes.append(json.loads(line.strip()))
        
        return jsonify({'scenes': scenes})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video/<path:file_path>', methods=['GET'])
def serve_video(file_path):
    """提供视频文件"""
    try:
        video_file = OUTPUT_CLIPS_DIR / file_path
        if not video_file.exists():
            return jsonify({'error': 'Video file not found'}), 404
        
        return send_file(str(video_file))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clip/delete', methods=['POST'])
def delete_clip():
    """删除视频片段"""
    try:
        data = request.json
        video_path = data.get('videoPath')
        scene_id = data.get('sceneId')
        
        if not video_path or not scene_id:
            return jsonify({'error': 'Missing parameters'}), 400
        
        video_dir = OUTPUT_CLIPS_DIR / video_path
        if not video_dir.exists():
            return jsonify({'error': 'Video directory not found'}), 404
        
        # 读取场景信息
        scene_info_path = video_dir / 'scene_info.json'
        video_scene_info_path = video_dir / 'video_scece_info.txt'
        
        scenes = []
        if scene_info_path.exists():
            with open(scene_info_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        scene = json.loads(line.strip())
                        if scene['id'] != scene_id:
                            scenes.append(scene)
                        else:
                            # 删除视频和音频文件
                            video_file = video_dir / f"{scene_id}.mp4"
                            audio_file = video_dir / f"{scene_id}.m4a"
                            if video_file.exists():
                                video_file.unlink()
                            if audio_file.exists():
                                audio_file.unlink()
        
        # 更新scene_info.json
        with open(scene_info_path, 'w', encoding='utf-8') as f:
            for scene in scenes:
                f.write(json.dumps(scene) + '\n')
        
        # 更新video_scece_info.txt
        if video_scene_info_path.exists():
            with open(video_scene_info_path, 'w', encoding='utf-8') as f:
                for i, scene in enumerate(scenes, 1):
                    start_time = seconds_to_time_str(scene['start-time'])
                    end_time = seconds_to_time_str(scene['end-time'])
                    f.write(f"scene {i} infos: start_time {start_time}, end_time {end_time}\n")
        
        return jsonify({'success': True, 'remainingScenes': len(scenes)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clip/extract', methods=['POST'])
def extract_clip():
    """从现有片段中截取新片段"""
    try:
        data = request.json
        video_path = data.get('videoPath')
        scene_id = data.get('sceneId')
        start_time = data.get('startTime')  # 相对于当前片段的时间
        end_time = data.get('endTime')
        
        print(f"\n=== 截取片段请求 ===")
        print(f"视频路径: {video_path}")
        print(f"场景ID: {scene_id}")
        print(f"起始时间: {start_time}秒")
        print(f"结束时间: {end_time}秒")
        
        if not all([video_path, scene_id, start_time is not None, end_time is not None]):
            return jsonify({'error': 'Missing parameters'}), 400
        
        video_dir = OUTPUT_CLIPS_DIR / video_path
        if not video_dir.exists():
            print(f"错误: 视频目录不存在: {video_dir}")
            return jsonify({'error': f'Video directory not found: {video_dir}'}), 404
        
        # 读取场景信息
        scene_info_path = video_dir / 'scene_info.json'
        scenes = []
        target_scene = None
        target_index = -1
        
        if scene_info_path.exists():
            with open(scene_info_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        scene = json.loads(line.strip())
                        scenes.append(scene)
                        if scene['id'] == scene_id:
                            target_scene = scene
                            target_index = i
        
        if not target_scene:
            print(f"错误: 未找到场景 {scene_id}")
            return jsonify({'error': f'Scene {scene_id} not found'}), 404
        
        # 计算新片段的绝对时间
        original_start = target_scene['start-time']
        original_end = target_scene['end-time']
        new_start = original_start + start_time
        new_end = original_start + end_time
        
        print(f"原始片段: {original_start}秒 - {original_end}秒")
        print(f"新片段: {new_start}秒 - {new_end}秒")
        
        if new_start < original_start or new_end > original_end or new_start >= new_end:
            error_msg = f'Invalid time range: new({new_start}-{new_end}) vs original({original_start}-{original_end})'
            print(f"错误: {error_msg}")
            return jsonify({'error': error_msg}), 400
        
        # 使用ffmpeg截取新片段
        input_video = video_dir / f"{scene_id}.mp4"
        input_audio = video_dir / f"{scene_id}.m4a"
        temp_video = video_dir / f"{scene_id}_new.mp4"
        temp_audio = video_dir / f"{scene_id}_new.m4a"
        
        print(f"输入视频: {input_video}")
        print(f"临时视频: {temp_video}")
        
        # 截取视频
        if input_video.exists():
            print(f"开始截取视频...")
            # 使用重新编码模式，避免copy模式导致的损坏
            cmd_video = [
                'ffmpeg', '-i', str(input_video),
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c:v', 'libx264',      # 重新编码视频
                '-preset', 'medium',     # 编码质量（medium是平衡选项）
                '-crf', '23',           # 质量因子（18-28，越小质量越好）
                '-c:a', 'aac',          # 音频编码
                '-b:a', '128k',         # 音频比特率
                str(temp_video),
                '-y'
            ]
            print(f"FFmpeg命令: {' '.join(cmd_video)}")
            result = subprocess.run(cmd_video, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg错误 (视频): {result.stderr}")
                return jsonify({'error': f'FFmpeg video error: {result.stderr}'}), 500
            print(f"✓ 视频截取成功")
            
            # 检查生成的文件大小
            if temp_video.exists():
                file_size = temp_video.stat().st_size
                print(f"  生成的视频文件大小: {file_size} 字节 ({file_size/1024:.2f} KB)")
                if file_size < 10000:  # 小于10KB肯定有问题
                    print(f"  错误: 生成的视频文件太小，截取失败")
                    return jsonify({'error': 'Generated video file is too small'}), 500
        else:
            print(f"警告: 视频文件不存在: {input_video}")
        
        # 截取音频
        if input_audio.exists():
            # 检查原始音频文件是否有效（不是损坏的小文件）
            audio_size = input_audio.stat().st_size
            print(f"检查原始音频文件: {audio_size} 字节 ({audio_size/1024:.2f} KB)")
            
            if audio_size < 10000:  # 小于10KB可能是损坏的
                print(f"警告: 原始音频文件太小（可能损坏），尝试从原始音频重新截取...")
                # 从scene_info获取原始音频源
                if target_scene and 'original-audio' in target_scene:
                    original_audio_path = OUTPUT_CLIPS_DIR.parent / target_scene['original-audio']
                    if original_audio_path.exists():
                        print(f"  使用原始音频: {original_audio_path}")
                        # 计算在原始音频中的绝对时间
                        original_start = target_scene['start-time']
                        abs_start = original_start + start_time
                        abs_duration = end_time - start_time
                        
                        cmd_audio = [
                            'ffmpeg', '-i', str(original_audio_path),
                            '-ss', str(abs_start),
                            '-t', str(abs_duration),
                            '-c:a', 'aac',
                            '-b:a', '128k',
                            str(temp_audio),
                            '-y'
                        ]
                        print(f"  从原始音频截取: {abs_start}s, 时长: {abs_duration}s")
                    else:
                        print(f"  错误: 原始音频文件不存在: {original_audio_path}")
                        return jsonify({'error': 'Original audio file not found'}), 500
                else:
                    print(f"  错误: scene_info中没有original-audio信息")
                    return jsonify({'error': 'No original-audio info in scene_info'}), 500
            else:
                # 原始音频正常，从当前片段截取
                print(f"开始截取音频...")
                cmd_audio = [
                    'ffmpeg', '-i', str(input_audio),
                    '-ss', str(start_time),
                    '-t', str(end_time - start_time),
                    '-c:a', 'aac',          # 重新编码音频
                    '-b:a', '128k',         # 音频比特率
                    str(temp_audio),
                    '-y'
                ]
            
            print(f"FFmpeg命令: {' '.join(cmd_audio)}")
            result = subprocess.run(cmd_audio, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"FFmpeg错误 (音频): {result.stderr}")
                return jsonify({'error': f'FFmpeg audio error: {result.stderr}'}), 500
            print(f"✓ 音频截取成功")
            
            # 检查生成的文件大小
            if temp_audio.exists():
                file_size = temp_audio.stat().st_size
                print(f"  生成的音频文件大小: {file_size} 字节 ({file_size/1024:.2f} KB)")
                # 如果生成的音频太小，报错
                if file_size < 10000:
                    print(f"  错误: 生成的音频文件太小，可能截取失败")
                    return jsonify({'error': 'Generated audio file is too small'}), 500
        else:
            print(f"警告: 音频文件不存在: {input_audio}")
        
        # 删除原文件并重命名新文件
        print(f"替换原文件...")
        if input_video.exists():
            input_video.unlink()
            print(f"✓ 删除原视频文件")
        if input_audio.exists():
            input_audio.unlink()
            print(f"✓ 删除原音频文件")
        
        if temp_video.exists():
            temp_video.rename(input_video)
            print(f"✓ 重命名新视频文件")
        if temp_audio.exists():
            temp_audio.rename(input_audio)
            print(f"✓ 重命名新音频文件")
        
        # 更新场景信息
        print(f"更新场景信息...")
        scenes[target_index]['start-time'] = new_start
        scenes[target_index]['end-time'] = new_end
        scenes[target_index]['durations'] = f"{new_end - new_start:.1f}s"
        
        # 计算帧数（如果有fps信息）
        if 'fps' in target_scene:
            fps = target_scene['fps']
            scenes[target_index]['start-frame'] = int(new_start * fps)
            scenes[target_index]['end-frame'] = int(new_end * fps)
        
        # 保存更新的场景信息
        with open(scene_info_path, 'w', encoding='utf-8') as f:
            for scene in scenes:
                f.write(json.dumps(scene) + '\n')
        print(f"✓ scene_info.json 已更新")
        
        # 更新video_scece_info.txt
        video_scene_info_path = video_dir / 'video_scece_info.txt'
        if video_scene_info_path.exists():
            with open(video_scene_info_path, 'w', encoding='utf-8') as f:
                for i, scene in enumerate(scenes, 1):
                    start_time_str = seconds_to_time_str(scene['start-time'])
                    end_time_str = seconds_to_time_str(scene['end-time'])
                    f.write(f"scene {i} infos: start_time {start_time_str}, end_time {end_time_str}\n")
            print(f"✓ video_scece_info.txt 已更新")
        
        print(f"=== 截取成功 ===\n")
        return jsonify({'success': True, 'updatedScene': scenes[target_index]})
    except subprocess.CalledProcessError as e:
        error_msg = f'FFmpeg error: {e.stderr.decode() if e.stderr else str(e)}'
        print(f"错误: {error_msg}")
        return jsonify({'error': error_msg}), 500
    except Exception as e:
        import traceback
        error_msg = f'{str(e)}\n{traceback.format_exc()}'
        print(f"错误: {error_msg}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print(f"Output clips directory: {OUTPUT_CLIPS_DIR}")
    print(f"Server starting on http://localhost:5000")
    app.run(debug=True, port=5000)
