# 截取功能故障排查指南

## 常见问题及解决方案

### 1. 检查FFmpeg是否安装

**测试命令:**
```bash
ffmpeg -version
```

**预期输出:**
应该显示FFmpeg版本信息

**如果失败:**
- Windows: 下载FFmpeg并添加到PATH
- 下载地址: https://ffmpeg.org/download.html

### 2. 检查后端服务器日志

重启后端服务器后，会看到详细的日志输出：

**正常流程:**
```
=== 截取片段请求 ===
视频路径: Lecture/_jcW-ZgpRbM
场景ID: scene_2
起始时间: 5.0秒
结束时间: 15.0秒
原始片段: 183.128秒 - 208.128秒
新片段: 188.128秒 - 198.128秒
输入视频: D:\...\output_clips\Lecture\_jcW-ZgpRbM\scene_2.mp4
临时视频: D:\...\output_clips\Lecture\_jcW-ZgpRbM\scene_2_new.mp4
开始截取视频...
FFmpeg命令: ffmpeg -i ... -ss 5.0 -t 10.0 -c copy ... -y
✓ 视频截取成功
开始截取音频...
✓ 音频截取成功
✓ 删除原视频文件
✓ 删除原音频文件
✓ 重命名新视频文件
✓ 重命名新音频文件
✓ scene_info.json 已更新
✓ video_scece_info.txt 已更新
=== 截取成功 ===
```

**查找错误信息:**
在日志中查找：
- "错误:" 开头的行
- "FFmpeg错误" 信息
- Python traceback

### 3. 常见错误类型

#### 错误A: FFmpeg未找到
**症状:**
```
FFmpeg error: 'ffmpeg' 不是内部或外部命令
```

**解决:**
1. 确认FFmpeg已安装
2. 添加到系统PATH环境变量
3. 重启命令提示符和后端服务器

#### 错误B: 视频文件不存在
**症状:**
```
警告: 视频文件不存在: D:\...\scene_2.mp4
```

**解决:**
检查文件是否真实存在，路径是否正确

#### 错误C: 权限问题
**症状:**
```
Permission denied
```

**解决:**
1. 检查文件是否被其他程序占用
2. 以管理员权限运行
3. 检查文件夹权限

#### 错误D: 时间范围无效
**症状:**
```
Invalid time range: new(188-198) vs original(183-208)
```

**原因:**
截取的时间超出了原片段范围

**解决:**
确保：
- 起始时间 >= 0
- 结束时间 <= 片段总长度
- 起始时间 < 结束时间

#### 错误E: FFmpeg编码问题
**症状:**
```
FFmpeg error: [codec] ...
```

**解决:**
可能需要重新编码而不是直接复制流。修改server.py中的FFmpeg命令：

将:
```python
'-c', 'copy',
```

改为:
```python
'-c:v', 'libx264', '-c:a', 'aac',
```

但这会慢很多。

### 4. 调试步骤

#### 步骤1: 查看浏览器控制台
1. 按F12打开开发者工具
2. 切换到Console标签
3. 查看"发送截取请求"日志
4. 查看"截取响应"日志

#### 步骤2: 查看后端日志
在运行`python server.py`的终端窗口查看详细日志

#### 步骤3: 手动测试FFmpeg
尝试手动运行FFmpeg命令：

```bash
cd "D:\Projects\Exploration\3DTalkingHeadCodeBase\data\data_pipline\audio_visual_dataset\output_clips\Lecture\_jcW-ZgpRbM"

ffmpeg -i scene_2.mp4 -ss 5.0 -t 10.0 -c copy test_output.mp4 -y
```

如果这个命令失败，说明是FFmpeg或视频文件的问题。

#### 步骤4: 检查文件路径
确保路径中没有特殊字符，特别是：
- 空格
- 中文字符
- 特殊符号

### 5. 临时解决方案

如果FFmpeg的 `-c copy` 模式不工作，可以使用重新编码模式：

**修改 server.py 第223-229行和235-241行:**

```python
# 视频截取 - 使用重新编码
cmd_video = [
    'ffmpeg', '-i', str(input_video),
    '-ss', str(start_time),
    '-t', str(end_time - start_time),
    '-c:v', 'libx264',  # 重新编码视频
    '-preset', 'fast',  # 快速编码
    str(temp_video),
    '-y'
]

# 音频截取 - 使用重新编码
cmd_audio = [
    'ffmpeg', '-i', str(input_audio),
    '-ss', str(start_time),
    '-t', str(end_time - start_time),
    '-c:a', 'aac',      # 重新编码音频
    str(temp_audio),
    '-y'
]
```

**注意:** 这会慢很多，但更可靠。

### 6. 完整测试流程

1. **测试FFmpeg:**
   ```bash
   ffmpeg -version
   ```

2. **重启后端服务器:**
   ```bash
   cd video_annotator
   python server.py
   ```

3. **在浏览器中测试:**
   - 打开index.html
   - 切换到本地片段模式
   - 选择一个视频和片段
   - 点击"截取新片段"
   - 设置起止时间（例如从5秒到15秒）
   - 点击"确认截取并删除原片段"

4. **查看日志:**
   - 浏览器控制台（F12）
   - 后端服务器终端

5. **验证结果:**
   - 视频是否重新加载？
   - 时长是否正确？
   - scene_info.json是否更新？

## 需要提供的信息

如果问题仍然存在，请提供：

1. **浏览器控制台的完整错误信息**
2. **后端服务器终端的完整日志**
3. **FFmpeg版本信息** (`ffmpeg -version`)
4. **操作系统版本**
5. **你设置的时间** (起始和结束)
6. **当前片段的总时长**

有了这些信息，我可以提供更准确的解决方案。
