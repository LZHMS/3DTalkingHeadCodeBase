const express = require('express');
const fs = require('fs').promises;
const fsSync = require('fs');
const path = require('path');
const fsExtra = require('fs-extra');

const app = express();
const PORT = 3000;
const sourcepath = '../output';

app.use(express.json());
app.use(express.static(__dirname));

// 设置视频流中间件（支持Range请求）
app.get('/videos/*', (req, res, next) => {
    const filePath = path.join(__dirname, sourcepath, req.params[0]);
    
    // 检查文件是否存在
    if (!fsSync.existsSync(filePath)) {
        return res.status(404).json({ error: '文件不存在' });
    }
    
    const stat = fsSync.statSync(filePath);
    const fileSize = stat.size;
    
    // 设置正确的MIME类型和响应头
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Accept-Ranges', 'bytes');
    // 移除Cache-Control避免304问题
    res.setHeader('Cache-Control', 'no-cache, no-store, must-revalidate');
    res.setHeader('Pragma', 'no-cache');
    res.setHeader('Expires', '0');
    
    // 处理Range请求（用于快进/快退）
    const range = req.headers.range;
    if (range) {
        const parts = range.replace(/bytes=/, '').split('-');
        const start = parseInt(parts[0], 10);
        const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
        const chunksize = end - start + 1;
        
        res.status(206);
        res.setHeader('Content-Range', `bytes ${start}-${end}/${fileSize}`);
        res.setHeader('Content-Length', chunksize);
        
        const stream = fsSync.createReadStream(filePath, { start, end });
        stream.pipe(res);
    } else {
        res.setHeader('Content-Length', fileSize);
        const stream = fsSync.createReadStream(filePath);
        stream.pipe(res);
    }
});

const CATEGORY_FILE = path.join(__dirname, 'categories.json');
const SKIPPED_FILE = path.join(__dirname, 'skipped_videos.json');

// 初始化分类数据
async function initCategories() {
    try {
        await fs.access(CATEGORY_FILE);
        const data = await fs.readFile(CATEGORY_FILE, 'utf-8');
        return JSON.parse(data);
    } catch {
        const defaultCategories = {
            '1': '访谈交流',
            '2': '公开演讲',
            '3': '课程教学',
            '4': '新闻节目'
        };
        await fs.writeFile(CATEGORY_FILE, JSON.stringify(defaultCategories, null, 2));
        return defaultCategories;
    }
}

// 获取所有类别
app.get('/api/categories', async (req, res) => {
    try {
        const categories = await initCategories();
        res.json(categories);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// 新建类别
app.post('/api/categories', async (req, res) => {
    try {
        const { name } = req.body;
        
        if (!name || name.trim() === '') {
            return res.status(400).json({ error: '类别名称不能为空' });
        }
        
        const categories = await initCategories();
        
        // 检查类别是否已存在
        const existingCategory = Object.values(categories).find(
            cat => cat === name.trim()
        );
        if (existingCategory) {
            return res.status(400).json({ error: '类别已存在' });
        }
        
        // 获取下一个编号
        const maxId = Math.max(...Object.keys(categories).map(Number));
        const newId = String(maxId + 1);
        
        categories[newId] = name.trim();
        await fs.writeFile(CATEGORY_FILE, JSON.stringify(categories, null, 2));
        
        // 创建对应的文件夹
        const categoryPath = path.join(__dirname, 'data', name.trim());
        await fsExtra.ensureDir(categoryPath);
        
        res.json({ success: true, id: newId, name: name.trim() });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// 获取所有视频文件夹列表
app.get('/api/videos', async (req, res) => {
    try {
        const outputPath = path.join(__dirname, sourcepath);
        const sourceCategories = await fs.readdir(outputPath);
        
        const videoFolders = [];
        for (const sourceCategory of sourceCategories) {
            const categoryPath = path.join(outputPath, sourceCategory);
            const stat = await fs.stat(categoryPath);
            
            if (!stat.isDirectory()) continue;
            
            const videoNames = await fs.readdir(categoryPath);
            
            for (const videoName of videoNames) {
                const videoPath = path.join(categoryPath, videoName);
                const videoStat = await fs.stat(videoPath);
                
                if (videoStat.isDirectory()) {
                    const files = await fs.readdir(videoPath);
                    const mergedVideo = files.find(f => f.endsWith('.mp4'));
                    
                    if (mergedVideo) {
                        const videoUrl = `/videos/${sourceCategory}/${videoName}/${mergedVideo}`;
                        videoFolders.push({
                            sourceCategory: sourceCategory,
                            name: videoName,
                            videoFile: mergedVideo,
                            path: `${sourceCategory}/${videoName}/${mergedVideo}`,
                            fullPath: `${sourcepath}/${sourceCategory}/${videoName}`,
                            url: videoUrl
                        });
                    }
                }
            }
        }
        
        res.json(videoFolders);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// 获取上一个视频
app.post('/api/videos/previous', async (req, res) => {
    try {
        const { currentSourceCategory, currentFolderName } = req.body;
        
        const outputPath = path.join(__dirname, sourcepath);
        const sourceCategories = await fs.readdir(outputPath);
        
        const videoFolders = [];
        for (const sourceCategory of sourceCategories) {
            const categoryPath = path.join(outputPath, sourceCategory);
            const stat = await fs.stat(categoryPath);
            
            if (!stat.isDirectory()) continue;
            
            const videoNames = await fs.readdir(categoryPath);
            
            for (const videoName of videoNames) {
                const videoPath = path.join(categoryPath, videoName);
                const videoStat = await fs.stat(videoPath);
                
                if (videoStat.isDirectory()) {
                    const files = await fs.readdir(videoPath);
                    const mergedVideo = files.find(f => f.endsWith('.mp4'));
                    
                    if (mergedVideo) {
                        videoFolders.push({
                            sourceCategory: sourceCategory,
                            name: videoName,
                            videoFile: mergedVideo,
                            path: `${sourceCategory}/${videoName}/${mergedVideo}`,
                            fullPath: `${sourcepath}/${sourceCategory}/${videoName}`
                        });
                    }
                }
            }
        }
        
        // 找到当前视频的索引
        const currentIndex = videoFolders.findIndex(
            v => v.sourceCategory === currentSourceCategory && v.name === currentFolderName
        );
        
        if (currentIndex > 0) {
            res.json(videoFolders[currentIndex - 1]);
        } else {
            res.status(404).json({ error: '没有上一个视频' });
        }
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// 分类并拷贝文件夹
app.post('/api/classify', async (req, res) => {
    try {
        const { sourceCategory, folderName, category } = req.body;
        const categories = await initCategories();
        const categoryName = categories[category];
        
        if (!categoryName) {
            return res.status(400).json({ error: '无效的分类' });
        }
        
        const sourcePath = path.join(__dirname, sourcepath, sourceCategory, folderName);
        const destPath = path.join(__dirname, 'data', categoryName, folderName);
        
        // 确保源文件夹存在
        try {
            await fs.access(sourcePath);
        } catch {
            return res.status(404).json({ error: '源文件夹不存在' });
        }
        
        // 确保目标目录存在
        await fsExtra.ensureDir(path.join(__dirname, 'data', categoryName));
        
        // 检查目标是否已存在
        try {
            await fs.access(destPath);
            return res.status(400).json({ error: '目标文件夹已存在，请先删除或重命名' });
        } catch {
            // 目标不存在，可以继续
        }
        
        await fsExtra.copy(sourcePath, destPath);
        
        res.json({ 
            success: true, 
            message: `已从 ${sourceCategory} 拷贝到 data/${categoryName}` 
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// 记录跳过的视频
app.post('/api/skip', async (req, res) => {
    try {
        const { sourceCategory, folderName, videoFile, fullPath } = req.body;
        
        // 读取现有的跳过记录
        let skippedVideos = [];
        try {
            const data = await fs.readFile(SKIPPED_FILE, 'utf-8');
            skippedVideos = JSON.parse(data);
        } catch {
            // 文件不存在，使用空数组
        }
        
        // 添加新的跳过记录
        const skipRecord = {
            sourceCategory,
            folderName,
            videoFile,
            fullPath,
            timestamp: new Date().toISOString(),
            date: new Date().toLocaleString('zh-CN')
        };
        
        skippedVideos.push(skipRecord);
        
        // 保存到文件
        await fs.writeFile(SKIPPED_FILE, JSON.stringify(skippedVideos, null, 2));
        
        res.json({ 
            success: true, 
            message: `已跳过视频: ${folderName}`,
            totalSkipped: skippedVideos.length
        });
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// 获取跳过的视频列表
app.get('/api/skipped', async (req, res) => {
    try {
        let skippedVideos = [];
        try {
            const data = await fs.readFile(SKIPPED_FILE, 'utf-8');
            skippedVideos = JSON.parse(data);
        } catch {
            // 文件不存在，返回空数组
        }
        res.json(skippedVideos);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// 添加视频信息检查接口
app.get('/api/video-info/:sourceCategory/:folderName/:videoFile', async (req, res) => {
    try {
        const { sourceCategory, folderName, videoFile } = req.params;
        const videoPath = path.join(__dirname, sourcepath, sourceCategory, folderName, videoFile);
        
        // 检查文件是否存在
        const stat = await fs.stat(videoPath);
        
        console.log(`视频信息: ${videoPath}`);
        console.log(`文件大小: ${(stat.size / 1024 / 1024).toFixed(2)} MB`);
        console.log(`修改时间: ${stat.mtime}`);
        
        res.json({
            exists: true,
            size: stat.size,
            sizeInMB: (stat.size / 1024 / 1024).toFixed(2),
            mtime: stat.mtime,
            isFile: stat.isFile(),
            path: videoPath
        });
    } catch (error) {
        res.status(404).json({ 
            exists: false, 
            error: error.message 
        });
    }
});

// 添加MIME类型支持
app.use((req, res, next) => {
    if (req.url.endsWith('.mp4')) {
        res.type('video/mp4');
    }
    next();
});

// 添加视频调试页面
app.get('/debug/video', (req, res) => {
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>视频调试</title>
            <style>
                body { font-family: Arial; margin: 20px; }
                video { width: 100%; max-width: 800px; background: #000; margin: 20px 0; }
                .info { background: #f0f0f0; padding: 10px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <h1>视频播放调试</h1>
            <div class="info" id="videoList"></div>
            <video id="testVideo" controls></video>
            <div class="info" id="status"></div>
            
            <script>
                const statusEl = document.getElementById('status');
                const video = document.getElementById('testVideo');
                
                // 获取视频列表
                fetch('/api/videos')
                    .then(r => r.json())
                    .then(videos => {
                        if (videos.length > 0) {
                            const firstVideo = videos[0];
                            document.getElementById('videoList').innerHTML = 
                                '<strong>找到 ' + videos.length + ' 个视频</strong><br>' +
                                '第一个视频: ' + firstVideo.name + '<br>' +
                                'URL: ' + firstVideo.url;
                            
                            video.src = firstVideo.url;
                            updateStatus();
                        }
                    })
                    .catch(e => statusEl.innerHTML = '加载视频列表失败: ' + e);
                
                function updateStatus() {
                    statusEl.innerHTML = '<strong>视频状态：</strong><br>' +
                        'readyState: ' + video.readyState + '<br>' +
                        'networkState: ' + video.networkState + '<br>' +
                        'currentTime: ' + video.currentTime + '<br>' +
                        'duration: ' + video.duration + '<br>' +
                        'paused: ' + video.paused + '<br>' +
                        'error: ' + (video.error ? video.error.message : 'none');
                }
                
                video.addEventListener('loadstart', updateStatus);
                video.addEventListener('loadedmetadata', updateStatus);
                video.addEventListener('canplay', updateStatus);
                video.addEventListener('error', updateStatus);
                setInterval(updateStatus, 500);
            </script>
        </body>
        </html>
    `);
});

app.listen(PORT, () => {
    console.log(`服务器运行在 http://localhost:${PORT}`);
    console.log(`视频文件夹: ${sourcepath}`);
    console.log(`调试页面: http://localhost:${PORT}/debug/video`);
});
