# 视频片段标注与管理工具

这是一个用于视频片段标注和管理的Web应用，支持两种模式：

## 功能特性

### 1. URL模式（原有功能）
- 上传包含YouTube链接的txt文件
- 在线播放YouTube视频
- 设置起止点标注视频片段
- 导出标注结果为JSON

### 2. 本地片段模式（新功能）
- 浏览output_clips文件夹下的本地视频片段
- 按分类和视频名称组织
- 依次播放视频的多个场景片段
- 删除不需要的片段
- 从现有片段中截取新片段并删除原片段
- 自动更新scene_info.json和video_scece_info.txt文件

## 使用方法

### 安装依赖

```bash
pip install flask flask-cors
```

确保系统已安装ffmpeg（用于视频截取功能）：
- Windows: 下载ffmpeg并添加到PATH
- Linux: `sudo apt-get install ffmpeg`
- Mac: `brew install ffmpeg`

### 启动服务

1. 启动后端服务器：
```bash
cd video_annotator
python server.py
```

2. 在浏览器中打开 `index.html` 文件

### URL模式使用步骤

1. 点击"URL模式"按钮
2. 上传包含YouTube链接的txt文件
3. 选择要标注的视频
4. 使用快捷键或按钮设置起止点：
   - `I` 键：设置起始时间
   - `O` 键：设置结束时间
   - `空格`：播放/暂停
   - `Enter`：添加片段
5. 导出JSON文件

### 本地片段模式使用步骤

1. 点击"本地片段"按钮
2. 从下拉菜单选择分类（如Lecture、Speech）
3. 选择要查看的视频
4. 应用会自动加载该视频的所有场景片段

#### 浏览和播放
- 使用"上一个"/"下一个"按钮在片段间切换
- 当前片段会自动播放

#### 删除片段
1. 播放到要删除的片段
2. 点击"删除当前片段"按钮
3. 确认删除
4. scene_info.json和video_scece_info.txt会自动更新

#### 截取新片段
1. 播放到要截取的片段
2. 点击"截取新片段"按钮
3. 在视频中定位新片段的起止点：
   - 点击"设置"按钮标记起始和结束时间
4. 点击"确认截取并删除原片段"
5. 应用会：
   - 使用ffmpeg从原片段中截取新片段
   - 删除原片段文件
   - 用新片段替换
   - 更新scene_info.json和video_scece_info.txt

## 文件结构

```
output_clips/
├── Lecture/                    # 分类1
│   └── _jcW-ZgpRbM/           # 视频名称
│       ├── scene_1.mp4        # 场景视频文件
│       ├── scene_1.m4a        # 场景音频文件
│       ├── scene_2.mp4
│       ├── scene_2.m4a
│       ├── ...
│       ├── scene_info.json    # 场景元数据
│       └── video_scece_info.txt  # 场景时间信息
└── Speech/                     # 分类2
    └── ...
```

## 快捷键

- `空格`：播放/暂停
- `I`：设置起始时间（URL模式）
- `O`：设置结束时间（URL模式）
- `Enter`：添加片段（URL模式）

## 注意事项

1. 后端服务必须在localhost:5000端口运行
2. 删除和截取操作不可恢复，请谨慎操作
3. 确保有ffmpeg用于视频处理
4. scene_info.json使用每行一个JSON对象的格式
5. 删除或修改片段后会自动更新两个信息文件

## 技术栈

- 前端：HTML5、CSS3、JavaScript
- 后端：Python Flask
- 视频处理：FFmpeg
- 视频播放：YouTube IFrame API（URL模式）、HTML5 Video（本地模式）

## API端点

- `GET /api/categories` - 获取所有分类
- `GET /api/videos/<category>` - 获取分类下的视频列表
- `GET /api/video/<path>/scenes` - 获取视频的场景列表
- `GET /api/video/<path>` - 获取视频文件
- `POST /api/clip/delete` - 删除片段
- `POST /api/clip/extract` - 截取新片段
