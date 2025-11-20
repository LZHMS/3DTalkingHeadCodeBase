# 快速开始指南

## 安装步骤

### 1. 安装Python依赖
```bash
cd video_annotator
pip install -r requirements.txt
```

### 2. 安装FFmpeg（如果还没安装）

**Windows:**
1. 从 https://ffmpeg.org/download.html 下载ffmpeg
2. 解压到一个文件夹（如 C:\ffmpeg）
3. 将 C:\ffmpeg\bin 添加到系统环境变量PATH中

**验证安装:**
```bash
ffmpeg -version
```

## 启动应用

### 方法1: 使用批处理文件（Windows）
双击 `start_server.bat`

### 方法2: 手动启动
```bash
cd video_annotator
python server.py
```

然后在浏览器中打开 `index.html` 文件

## 使用流程

### 本地片段模式（主要功能）

1. **浏览视频**
   - 点击"本地片段"标签
   - 选择分类（Lecture、Speech等）
   - 选择要查看的视频
   - 系统会自动加载所有场景片段

2. **播放片段**
   - 片段会自动播放
   - 使用"上一个"/"下一个"按钮切换片段
   - 使用播放控制按钮调整速度

3. **删除片段**
   - 播放到要删除的片段
   - 点击"🗑️ 删除当前片段"
   - 确认删除
   - 系统会自动：
     * 删除视频和音频文件
     * 更新 scene_info.json
     * 更新 video_scece_info.txt

4. **截取新片段**
   - 点击"✂️ 截取新片段"
   - 在当前片段中移动到起始位置，点击"设置"
   - 移动到结束位置，点击"设置"
   - 点击"确认截取并删除原片段"
   - 系统会：
     * 使用FFmpeg截取新片段
     * 删除原片段
     * 更新所有信息文件

### URL模式（原有功能）

1. 点击"URL模式"标签
2. 上传包含YouTube链接的txt文件
3. 选择视频并标注片段
4. 导出JSON文件

## 常见问题

### Q: 服务器启动失败
A: 检查是否已安装flask和flask-cors：
```bash
pip install flask flask-cors
```

### Q: 截取功能不工作
A: 确保已正确安装ffmpeg并添加到PATH

### Q: 无法加载视频列表
A: 检查：
1. 后端服务器是否在运行（localhost:5000）
2. output_clips文件夹路径是否正确
3. 浏览器控制台是否有错误信息

### Q: CORS错误
A: 确保使用正确的方式打开HTML：
- 推荐：通过本地web服务器
- 或直接在浏览器中打开文件

## 文件说明

- `index.html` - 前端页面
- `app.js` - 前端逻辑
- `styles.css` - 样式表
- `server.py` - 后端API服务器
- `requirements.txt` - Python依赖
- `start_server.bat` - Windows启动脚本

## 技术支持

如遇到问题，请检查：
1. 浏览器控制台（F12）的错误信息
2. 后端服务器终端的错误输出
3. scene_info.json 文件格式是否正确
