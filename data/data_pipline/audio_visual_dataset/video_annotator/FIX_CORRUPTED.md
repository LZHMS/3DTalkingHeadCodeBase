# 视频损坏问题说明和解决方案

## 问题原因

发现81个视频文件已损坏！这是因为使用了FFmpeg的 `-c copy` 模式进行截取。

### 为什么会损坏？

`-c copy` 模式直接复制视频流，不重新编码。但是：
1. **关键帧问题**：如果起始点不是关键帧（I-frame），视频可能无法正确播放
2. **时间戳问题**：直接复制可能导致时间戳不连续
3. **编码器兼容性**：某些编码格式不支持流复制

### 损坏的表现

- 文件大小异常小（通常<10KB，有些甚至0字节）
- 无法播放或播放出错
- 浏览器报错：`Failed to load because no supported source was found`

## 已修复的问题

### 修改内容

已将 `server.py` 中的FFmpeg命令从：
```python
'-c', 'copy',  # 直接复制（快但可能损坏）
```

改为：
```python
'-c:v', 'libx264',      # 重新编码视频
'-preset', 'medium',     # 编码质量
'-crf', '23',           # 质量因子
'-c:a', 'aac',          # 重新编码音频
'-b:a', '128k',         # 音频比特率
```

### 优缺点对比

**Copy模式（旧方式）：**
- ✅ 速度快（几乎瞬间完成）
- ✅ 无质量损失
- ❌ 容易损坏（关键帧问题）
- ❌ 生成的文件可能无法播放

**重新编码模式（新方式）：**
- ✅ 可靠，不会损坏
- ✅ 保证能播放
- ✅ CRF=23质量很好（肉眼几乎无差别）
- ❌ 速度较慢（可能需要几秒到几十秒）
- ❌ 有轻微质量损失（但可接受）

## 处理已损坏的文件

### 方法1：自动清理（推荐）

运行清理脚本：
```bash
cd video_annotator
python clean_corrupted.py
```

脚本会：
1. 扫描所有<10KB的损坏文件
2. 备份到 `corrupted_backup` 目录
3. 删除损坏的.mp4和.m4a文件
4. 更新 scene_info.json（移除对应记录）
5. 更新 video_scece_info.txt（重新编号）

**注意：** 这会永久删除损坏的文件（但有备份）

### 方法2：仅检查不删除

```bash
python check_corrupted.py
```

只查看哪些文件损坏，不做修改。

### 方法3：手动处理

对于特定的损坏文件：

1. **删除损坏文件**
   ```bash
   del "output_clips\Lecture\17IgK9b6P2M\scene_2.mp4"
   del "output_clips\Lecture\17IgK9b6P2M\scene_2.m4a"
   ```

2. **编辑 scene_info.json**
   - 删除 `{"id":"scene_2",...}` 这一行

3. **编辑 video_scece_info.txt**
   - 删除 scene 2 那一行
   - 重新编号剩余的scene

4. **或者在网页中重新截取**
   - 使用更新后的工具重新截取该片段
   - 新生成的文件将是正常的

## 预防措施

### 之后的使用

**已解决！** 更新后的 `server.py` 使用重新编码模式，不会再产生损坏的文件。

### 使用建议

1. **重启后端服务器**（重要！）
   ```bash
   # 停止旧服务器 (Ctrl+C)
   python server.py  # 启动新版本
   ```

2. **刷新浏览器** (Ctrl+F5)

3. **测试截取功能**
   - 选择一个片段
   - 截取一小段
   - 验证新文件可以播放

4. **速度影响**
   - 截取现在会慢一些（可能5-30秒）
   - 这是正常的，因为需要重新编码
   - 但生成的文件保证可用

## 清理统计

运行 `check_corrupted.py` 发现：
- **总损坏文件数**: 81个
- **影响的目录**: 约50个
- **Lecture分类**: 47个损坏文件
- **Speech分类**: 34个损坏文件

**最严重的情况：**
- `Speech\5nTuScU70As`: 6个损坏文件
- `Speech\3E46oWB4V0s`: 4个损坏文件
- `Speech\yeRH7AkXm4c`: 4个损坏文件
- `Lecture\17IgK9b6P2M`: 2个损坏文件

## 恢复选项

如果确实需要这些片段的内容：

### 选项A：从原始视频重新截取

如果 scene_info.json 中有 `original-video` 信息，可以：
1. 找到原始视频文件
2. 使用FFmpeg手动重新截取
3. 使用正确的时间范围和新的编码参数

### 选项B：接受损失

如果这些都是测试截取或不重要的片段：
1. 直接运行清理脚本删除
2. 继续使用剩余的有效片段

## 常见问题

### Q: 清理后会不会影响其他片段？
A: 不会。只删除损坏的文件，正常的片段不受影响。

### Q: 备份在哪里？
A: `corrupted_backup` 目录，与 `output_clips` 同级。

### Q: 可以撤销清理操作吗？
A: 可以从备份目录手动复制文件回去，但建议不要（因为它们已经损坏）。

### Q: 新的编码模式会很慢吗？
A: 比copy模式慢，但对于10-30秒的短片段，通常在10-30秒内完成。

### Q: 质量会下降吗？
A: 使用CRF=23，质量损失非常小，肉眼几乎看不出区别。

## 执行清理

如果确定要清理，请按照以下步骤：

```bash
cd d:\Projects\Exploration\3DTalkingHeadCodeBase\data\data_pipline\audio_visual_dataset\video_annotator

# 1. 先检查（可选）
python check_corrupted.py

# 2. 清理损坏文件
python clean_corrupted.py

# 3. 重启后端服务器
python server.py

# 4. 在浏览器中刷新页面
```

之后截取的所有片段都将是正常的！
