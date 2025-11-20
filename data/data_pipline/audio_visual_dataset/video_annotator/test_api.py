#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的API测试脚本
用于验证后端服务是否正常工作
"""

import requests
import json

API_BASE = 'http://localhost:5000/api'

def test_api():
    print("=" * 50)
    print("视频片段管理工具 - API测试")
    print("=" * 50)
    print()
    
    # 测试1: 获取分类
    print("测试1: 获取分类列表")
    print("-" * 50)
    try:
        response = requests.get(f'{API_BASE}/categories')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 成功! 找到 {len(data['categories'])} 个分类:")
            for cat in data['categories']:
                print(f"  - {cat}")
        else:
            print(f"✗ 失败! 状态码: {response.status_code}")
    except Exception as e:
        print(f"✗ 错误: {e}")
    print()
    
    # 测试2: 获取Lecture分类下的视频
    print("测试2: 获取Lecture分类下的视频")
    print("-" * 50)
    try:
        response = requests.get(f'{API_BASE}/videos/Lecture')
        if response.status_code == 200:
            data = response.json()
            print(f"✓ 成功! 找到 {len(data['videos'])} 个视频:")
            for video in data['videos'][:5]:  # 只显示前5个
                print(f"  - {video['id']} ({video['sceneCount']} 个片段)")
        else:
            print(f"✗ 失败! 状态码: {response.status_code}")
    except Exception as e:
        print(f"✗ 错误: {e}")
    print()
    
    # 测试3: 获取第一个视频的场景
    print("测试3: 获取第一个视频的场景信息")
    print("-" * 50)
    try:
        # 先获取视频列表
        response = requests.get(f'{API_BASE}/videos/Lecture')
        if response.status_code == 200:
            data = response.json()
            if len(data['videos']) > 0:
                first_video = data['videos'][0]
                
                # 获取场景信息
                response = requests.get(f"{API_BASE}/video/{first_video['path']}/scenes")
                if response.status_code == 200:
                    scene_data = response.json()
                    print(f"✓ 成功! 视频 {first_video['id']} 有 {len(scene_data['scenes'])} 个场景:")
                    for scene in scene_data['scenes'][:3]:  # 只显示前3个
                        print(f"  - {scene['id']}: {scene['durations']}")
                else:
                    print(f"✗ 失败! 状态码: {response.status_code}")
            else:
                print("✗ 没有找到视频")
        else:
            print(f"✗ 失败! 状态码: {response.status_code}")
    except Exception as e:
        print(f"✗ 错误: {e}")
    print()
    
    print("=" * 50)
    print("测试完成!")
    print("=" * 50)
    print()
    print("如果所有测试都通过，说明后端API工作正常。")
    print("现在可以在浏览器中打开 index.html 使用完整功能。")
    print()

if __name__ == '__main__':
    print()
    print("确保后端服务器正在运行（python server.py）")
    print()
    input("按Enter键开始测试...")
    print()
    test_api()
