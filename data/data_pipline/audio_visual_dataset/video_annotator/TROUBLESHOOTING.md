# 问题修复说明

## 已修复的问题

### 1. loadLocalVideo函数中的event错误
**问题:** 函数中使用了 `event.target`，但没有正确传递event对象
**修复:** 修改函数签名，通过参数传递点击的元素

### 2. 视频路径问题
**问题:** video-path包含 'output_clips/' 前缀，但API不需要这个前缀
**修复:** 在播放时自动移除 'output_clips/' 前缀

### 3. 后端服务器未运行
**问题:** 加载视频失败是因为后端服务器没有启动
**解决方案:** 必须先启动后端服务器

## 使用步骤（重要！）

### 第一步：启动后端服务器

打开终端/命令提示符，执行：

```bash
cd d:\Projects\Exploration\3DTalkingHeadCodeBase\data\data_pipline\audio_visual_dataset\video_annotator
python server.py
```

或者双击 `start_server.bat`

你应该看到类似这样的输出：
```
Output clips directory: D:\Projects\Exploration\3DTalkingHeadCodeBase\data\data_pipline\audio_visual_dataset\output_clips
Server starting on http://localhost:5000
 * Serving Flask app 'server'
 * Debug mode: on
WARNING: This is a development server. Do not use it in a production deployment.
 * Running on http://127.0.0.1:5000
```

**保持这个窗口打开！不要关闭！**

### 第二步：打开前端页面

在浏览器中打开：
```
d:\Projects\Exploration\3DTalkingHeadCodeBase\data\data_pipline\audio_visual_dataset\video_annotator\index.html
```

### 第三步：切换到本地片段模式

1. 点击左上角的"本地片段"按钮
2. 从下拉菜单选择分类（如 Lecture 或 Speech）
3. 点击任意视频
4. 视频片段应该开始播放

## 如何验证后端服务器是否运行

打开浏览器，访问：
```
http://localhost:5000/api/categories
```

如果看到 JSON 数据（如 `{"categories":["Lecture","Speech"]}`），说明服务器正常运行。

## 常见错误及解决方案

### 错误1: "无法连接到服务器"
**原因:** 后端服务器未启动
**解决:** 按照"第一步"启动服务器

### 错误2: "Failed to load because no supported source was found"
**原因:** 
- 后端服务器未运行，或
- 视频文件路径错误

**解决:** 
1. 确保后端服务器正在运行
2. 检查浏览器控制台（F12）看实际请求的URL
3. 检查视频文件是否存在

### 错误3: "TypeError: Cannot read properties of undefined"
**原因:** 已修复 - 更新app.js后应该不会再出现

**解决:** 刷新浏览器页面（Ctrl+F5强制刷新）

### 错误4: Python模块未找到
**原因:** 缺少依赖

**解决:**
```bash
pip install flask flask-cors
```

## 调试技巧

1. **查看浏览器控制台**
   - 按 F12 打开开发者工具
   - 切换到 Console 标签
   - 查看红色错误信息

2. **查看网络请求**
   - 按 F12 打开开发者工具
   - 切换到 Network 标签
   - 重新加载页面
   - 查看所有请求是否成功（状态码200）

3. **查看后端日志**
   - 在运行 `python server.py` 的终端窗口
   - 查看是否有错误信息

## 测试API

可以使用提供的测试脚本：

```bash
cd video_annotator
python test_api.py
```

这会测试所有API端点是否正常工作。

## 完整的启动检查清单

- [ ] Python已安装
- [ ] Flask和Flask-CORS已安装 (`pip install -r requirements.txt`)
- [ ] FFmpeg已安装（用于截取功能）
- [ ] 后端服务器正在运行（`python server.py`）
- [ ] 浏览器中打开了index.html
- [ ] 切换到"本地片段"模式
- [ ] 选择了分类
- [ ] 点击视频列表中的视频

如果所有步骤都正确，视频应该能正常播放！
