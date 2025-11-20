// 应用模式: 'url' 或 'local'
let appMode = 'url';

// YouTube API 和应用状态
let player;
let currentVideoId = '';
let videoList = [];
let segments = [];
let startTime = null;
let endTime = null;

// 本地视频模式状态
let localPlayer;
let currentCategory = '';
let currentVideoPath = '';
let currentClips = [];
let currentClipIndex = 0;
let extractStartTime = null;
let extractEndTime = null;

// API基础URL
const API_BASE_URL = 'http://localhost:5000/api';

// 从 public_speaking.txt 读取的视频列表 (手动复制或通过文件上传)
let VIDEO_URLS = [
    'https://www.youtube.com/watch?v=8S0FDjFBj8o',
    'https://www.youtube.com/watch?v=5MgBikgcWnY',
    'https://www.youtube.com/watch?v=2jHWxUovYRg',
];

// 初始化
function init() {
    localPlayer = document.getElementById('localPlayer');
    loadVideoList();
    setupEventListeners();
    loadYouTubeAPI();
    setupFileUpload();
    setupModeSwitch();
    setupHelpPanel();
    loadCategories();
}

// 设置帮助面板
function setupHelpPanel() {
    const toggleBtn = document.getElementById('toggleHelpBtn');
    const closeBtn = document.getElementById('closeHelpBtn');
    const helpPanel = document.getElementById('helpPanel');
    
    toggleBtn.onclick = () => {
        const isVisible = helpPanel.style.display !== 'none';
        helpPanel.style.display = isVisible ? 'none' : 'block';
    };
    
    closeBtn.onclick = () => {
        helpPanel.style.display = 'none';
    };
    
    // 按H键切换帮助
    document.addEventListener('keydown', (e) => {
        if (e.key.toLowerCase() === 'h' && e.target.tagName !== 'INPUT' && e.target.tagName !== 'TEXTAREA') {
            e.preventDefault();
            const isVisible = helpPanel.style.display !== 'none';
            helpPanel.style.display = isVisible ? 'none' : 'block';
        }
    });
}

// 设置模式切换
function setupModeSwitch() {
    const urlModeBtn = document.getElementById('urlModeBtn');
    const localModeBtn = document.getElementById('localModeBtn');
    
    urlModeBtn.onclick = () => switchMode('url');
    localModeBtn.onclick = () => switchMode('local');
}

function switchMode(mode) {
    appMode = mode;
    
    // 更新按钮状态
    document.getElementById('urlModeBtn').classList.toggle('active', mode === 'url');
    document.getElementById('localModeBtn').classList.toggle('active', mode === 'local');
    
    // 显示/隐藏相应面板
    document.getElementById('urlModePanel').classList.toggle('active', mode === 'url');
    document.getElementById('localModePanel').classList.toggle('active', mode === 'local');
    
    // 显示/隐藏播放器
    document.getElementById('player').style.display = mode === 'url' ? 'block' : 'none';
    document.getElementById('localPlayer').style.display = mode === 'local' ? 'block' : 'none';
    
    // 显示/隐藏功能面板
    document.getElementById('annotationPanel').style.display = mode === 'url' ? 'block' : 'none';
    document.getElementById('segmentsList').style.display = mode === 'url' ? 'block' : 'none';
    document.getElementById('localClipActions').style.display = mode === 'local' ? 'block' : 'none';
    document.getElementById('clipInfo').style.display = mode === 'local' ? 'block' : 'none';
    
    // 切换帮助面板显示的快捷键
    document.getElementById('urlModeHelp').style.display = mode === 'url' ? 'block' : 'none';
    document.getElementById('localModeHelp').style.display = mode === 'local' ? 'block' : 'none';
}

// 加载分类列表
async function loadCategories() {
    try {
        const response = await fetch(`${API_BASE_URL}/categories`);
        const data = await response.json();
        
        const select = document.getElementById('categorySelect');
        select.innerHTML = '<option value="">选择分类...</option>';
        
        data.categories.forEach(category => {
            const option = document.createElement('option');
            option.value = category;
            option.textContent = category;
            select.appendChild(option);
        });
        
        select.onchange = () => loadVideosInCategory(select.value);
    } catch (error) {
        console.error('加载分类失败:', error);
        alert('无法连接到服务器，请确保后端服务已启动 (python server.py)');
    }
}

// 加载分类下的视频
async function loadVideosInCategory(category) {
    if (!category) {
        document.getElementById('localVideoList').innerHTML = '';
        return;
    }
    
    currentCategory = category;
    
    try {
        const response = await fetch(`${API_BASE_URL}/videos/${category}`);
        const data = await response.json();
        
        const container = document.getElementById('localVideoList');
        container.innerHTML = '';
        
        if (data.videos.length === 0) {
            container.innerHTML = '<p style="padding: 10px; text-align: center;">此分类下没有视频</p>';
            return;
        }
        
        data.videos.forEach((video, index) => {
            const item = document.createElement('div');
            item.className = 'video-item';
            item.innerHTML = `
                <div>${video.id}</div>
                <small>${video.sceneCount} 个片段</small>
            `;
            item.onclick = (e) => loadLocalVideo(category, video, item);
            container.appendChild(item);
        });
    } catch (error) {
        console.error('加载视频列表失败:', error);
        alert('加载视频列表失败');
    }
}

// 加载本地视频
async function loadLocalVideo(category, video, clickedItem) {
    try {
        const response = await fetch(`${API_BASE_URL}/video/${video.path}/scenes`);
        const data = await response.json();
        
        currentVideoPath = video.path;
        currentClips = data.scenes;
        currentClipIndex = 0;
        
        if (currentClips.length > 0) {
            playLocalClip(0);
        }
        
        // 更新激活状态
        document.querySelectorAll('#localVideoList .video-item').forEach((item) => {
            item.classList.remove('active');
        });
        if (clickedItem) {
            clickedItem.classList.add('active');
        }
    } catch (error) {
        console.error('加载视频片段失败:', error);
        alert('加载视频片段失败: ' + error.message);
    }
}

// 播放本地片段
function playLocalClip(index) {
    if (index < 0 || index >= currentClips.length) return;
    
    currentClipIndex = index;
    const clip = currentClips[index];
    
    // 更新片段信息
    document.getElementById('clipTitle').textContent = `${clip.id} - ${clip.durations}`;
    document.getElementById('clipPosition').textContent = `${index + 1} / ${currentClips.length}`;
    
    // 加载视频 - 从video-path中移除output_clips/前缀
    let videoPath = clip['video-path'];
    if (videoPath.startsWith('output_clips/')) {
        videoPath = videoPath.substring('output_clips/'.length);
    }
    const videoUrl = `${API_BASE_URL}/video/${videoPath}`;
    
    console.log('Loading video:', videoUrl); // 调试信息
    
    localPlayer.src = videoUrl;
    localPlayer.load();
    localPlayer.play().catch(err => {
        console.error('播放失败:', err);
        alert('视频播放失败: ' + err.message);
    });
    
    // 更新导航按钮状态
    document.getElementById('prevClipBtn').disabled = index === 0;
    document.getElementById('nextClipBtn').disabled = index === currentClips.length - 1;
}

// 删除当前片段
async function deleteCurrentClip() {
    if (!currentVideoPath || currentClips.length === 0) return;
    
    const clip = currentClips[currentClipIndex];
    
    if (!confirm(`确定要删除片段 ${clip.id} 吗？此操作不可恢复。`)) {
        return;
    }
    
    await performDelete();
}

// 快速删除（无确认）
async function deleteCurrentClipQuick() {
    if (!currentVideoPath || currentClips.length === 0) return;
    
    console.log('快速删除片段:', currentClips[currentClipIndex].id);
    await performDelete();
}

// 执行删除操作
async function performDelete() {
    const clip = currentClips[currentClipIndex];
    
    try {
        const response = await fetch(`${API_BASE_URL}/clip/delete`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                videoPath: currentVideoPath,
                sceneId: clip.id
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            console.log(`✓ 片段 ${clip.id} 已删除`);
            
            // 从列表中移除
            currentClips.splice(currentClipIndex, 1);
            
            // 如果还有片段，播放下一个或上一个
            if (currentClips.length > 0) {
                if (currentClipIndex >= currentClips.length) {
                    currentClipIndex = currentClips.length - 1;
                }
                playLocalClip(currentClipIndex);
            } else {
                localPlayer.src = '';
                document.getElementById('clipTitle').textContent = '没有更多片段';
            }
        } else {
            console.error('删除失败:', data.error);
            alert('删除失败: ' + data.error);
        }
    } catch (error) {
        console.error('删除片段失败:', error);
        alert('删除片段失败');
    }
}

// 显示截取面板
function showExtractPanel() {
    const panel = document.getElementById('extractPanel');
    panel.style.display = panel.style.display === 'none' ? 'block' : 'none';
    extractStartTime = null;
    extractEndTime = null;
    updateExtractInputs();
}

// 设置截取起止点
function setExtractStart() {
    extractStartTime = localPlayer.currentTime;
    updateExtractInputs();
}

function setExtractEnd() {
    extractEndTime = localPlayer.currentTime;
    updateExtractInputs();
}

function updateExtractInputs() {
    document.getElementById('extractStart').value = extractStartTime !== null ? formatTime(extractStartTime) : '';
    document.getElementById('extractEnd').value = extractEndTime !== null ? formatTime(extractEndTime) : '';
}

// 确认截取
async function confirmExtract() {
    if (extractStartTime === null || extractEndTime === null) {
        alert('请设置截取的起始和结束时间');
        return;
    }
    
    if (extractStartTime >= extractEndTime) {
        alert('起始时间必须小于结束时间');
        return;
    }
    
    const clip = currentClips[currentClipIndex];
    const duration = extractEndTime - extractStartTime;
    
    if (!confirm(`确定要截取新片段并删除原片段 ${clip.id} 吗？\n\n新片段时长: ${formatTime(duration)}\n起始: ${formatTime(extractStartTime)}\n结束: ${formatTime(extractEndTime)}`)) {
        return;
    }
    
    await performExtract();
}

// 快速截取（无确认）
async function confirmExtractQuick() {
    if (extractStartTime === null || extractEndTime === null) {
        console.error('截取失败: 未设置起止时间');
        return;
    }
    
    if (extractStartTime >= extractEndTime) {
        console.error('截取失败: 起始时间必须小于结束时间');
        return;
    }
    
    console.log('快速截取:', formatTime(extractStartTime), '-', formatTime(extractEndTime));
    await performExtract();
}

// 执行截取操作
async function performExtract() {
    const clip = currentClips[currentClipIndex];
    
    try {
        console.log('发送截取请求:', {
            videoPath: currentVideoPath,
            sceneId: clip.id,
            startTime: extractStartTime,
            endTime: extractEndTime
        });
        
        const response = await fetch(`${API_BASE_URL}/clip/extract`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                videoPath: currentVideoPath,
                sceneId: clip.id,
                startTime: extractStartTime,
                endTime: extractEndTime
            })
        });
        
        const data = await response.json();
        console.log('截取响应:', data);
        
        if (data.success) {
            console.log(`✓ 片段 ${clip.id} 已截取并更新`);
            
            // 更新当前片段信息
            currentClips[currentClipIndex] = data.updatedScene;
            
            // 重新加载视频
            playLocalClip(currentClipIndex);
            
            // 隐藏截取面板并重置
            document.getElementById('extractPanel').style.display = 'none';
            extractStartTime = null;
            extractEndTime = null;
            updateExtractInputs();
        } else {
            const errorMsg = data.error || '未知错误';
            console.error('截取失败:', errorMsg);
            alert('截取失败: ' + errorMsg + '\n\n请查看后端服务器日志了解详细信息');
        }
    } catch (error) {
        console.error('截取片段失败:', error);
        alert('截取片段失败: ' + error.message + '\n\n请确保:\n1. 后端服务器正在运行\n2. FFmpeg已安装\n3. 视频文件可访问');
    }
}

// 取消截取
function cancelExtract() {
    document.getElementById('extractPanel').style.display = 'none';
    extractStartTime = null;
    extractEndTime = null;
    updateExtractInputs();
    console.log('已取消截取');
}


function setupFileUpload() {
    const fileInput = document.getElementById('fileInput');
    const fileInfo = document.getElementById('fileInfo');
    
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        
        if (!file) return;
        
        if (!file.name.endsWith('.txt')) {
            fileInfo.textContent = '错误: 请上传 .txt 文件';
            fileInfo.className = 'file-info error';
            return;
        }
        
        const reader = new FileReader();
        
        reader.onload = function(event) {
            try {
                const content = event.target.result;
                const urls = parseVideoUrls(content);
                
                if (urls.length === 0) {
                    fileInfo.textContent = '错误: 文件中没有找到有效的 YouTube 链接';
                    fileInfo.className = 'file-info error';
                    return;
                }
                
                VIDEO_URLS = urls;
                loadVideoList();
                
                // 加载第一个视频
                if (player && player.loadVideoById) {
                    const firstVideoId = extractVideoId(VIDEO_URLS[0]);
                    player.loadVideoById(firstVideoId);
                    currentVideoId = firstVideoId;
                }
                
                fileInfo.textContent = `✓ 成功加载 ${urls.length} 个视频链接`;
                fileInfo.className = 'file-info';
                
                // 清空当前标注
                segments = [];
                renderSegments();
                startTime = null;
                endTime = null;
                updateTimeInputs();
                
            } catch (error) {
                fileInfo.textContent = '错误: 文件解析失败';
                fileInfo.className = 'file-info error';
                console.error('文件解析错误:', error);
            }
        };
        
        reader.onerror = function() {
            fileInfo.textContent = '错误: 文件读取失败';
            fileInfo.className = 'file-info error';
        };
        
        reader.readAsText(file);
    });
}

// 解析视频 URL
function parseVideoUrls(content) {
    const lines = content.split('\n');
    const urls = [];
    
    lines.forEach(line => {
        const trimmed = line.trim();
        // 匹配 YouTube 链接
        if (trimmed && (trimmed.includes('youtube.com/watch') || trimmed.includes('youtu.be/'))) {
            urls.push(trimmed);
        }
    });
    
    return urls;
}

// 加载 YouTube IFrame API
function loadYouTubeAPI() {
    const tag = document.createElement('script');
    tag.src = 'https://www.youtube.com/iframe_api';
    const firstScriptTag = document.getElementsByTagName('script')[0];
    firstScriptTag.parentNode.insertBefore(tag, firstScriptTag);
}

// YouTube API 就绪回调
window.onYouTubeIframeAPIReady = function() {
    player = new YT.Player('player', {
        height: '480',
        width: '854',
        videoId: extractVideoId(VIDEO_URLS[0]),
        playerVars: {
            'controls': 1,
            'modestbranding': 1
        },
        events: {
            'onReady': onPlayerReady,
            'onStateChange': onPlayerStateChange
        }
    });
};

function onPlayerReady(event) {
    currentVideoId = extractVideoId(VIDEO_URLS[0]);
    updateTimeDisplay();
    drawTimeline();
}

function onPlayerStateChange(event) {
    if (event.data === YT.PlayerState.PLAYING) {
        startTimeUpdate();
    } else {
        stopTimeUpdate();
    }
}

// 提取视频 ID
function extractVideoId(url) {
    const match = url.match(/[?&]v=([^&]+)/);
    return match ? match[1] : '';
}

// 加载视频列表
function loadVideoList() {
    const container = document.getElementById('videoListContainer');
    container.innerHTML = '';
    
    videoList = VIDEO_URLS.map((url, index) => {
        const id = extractVideoId(url);
        return { id, url, index };
    });

    videoList.forEach((video, index) => {
        const item = document.createElement('div');
        item.className = 'video-item' + (index === 0 ? ' active' : '');
        item.textContent = `视频 ${index + 1}: ${video.id}`;
        item.onclick = () => loadVideo(video);
        container.appendChild(item);
    });
}

// 加载视频
function loadVideo(video) {
    if (player && player.loadVideoById) {
        player.loadVideoById(video.id);
        currentVideoId = video.id;
        
        // 更新激活状态
        document.querySelectorAll('.video-item').forEach((item, idx) => {
            item.classList.toggle('active', idx === video.index);
        });

        // 清空当前标注
        segments = [];
        renderSegments();
        startTime = null;
        endTime = null;
        updateTimeInputs();
    }
}

// 时间格式化
function formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
}

// 更新本地播放器时间显示
function updateLocalTimeDisplay() {
    const current = localPlayer.currentTime;
    const duration = localPlayer.duration;
    
    document.getElementById('currentTime').textContent = formatTime(current);
    document.getElementById('duration').textContent = formatTime(duration || 0);
    
    updateTimelineMarkers(current, duration || 0);
}

// 更新时间显示
let timeUpdateInterval;
function startTimeUpdate() {
    timeUpdateInterval = setInterval(() => {
        if (player && player.getCurrentTime) {
            updateTimeDisplay();
        }
    }, 100);
}

function stopTimeUpdate() {
    clearInterval(timeUpdateInterval);
}

function updateTimeDisplay() {
    if (!player || !player.getCurrentTime) return;
    
    const current = player.getCurrentTime();
    const duration = player.getDuration();
    
    document.getElementById('currentTime').textContent = formatTime(current);
    document.getElementById('duration').textContent = formatTime(duration);
    
    updateTimelineMarkers(current, duration);
}

// 设置事件监听
function setupEventListeners() {
    // 播放控制
    document.getElementById('playPauseBtn').onclick = () => {
        if (appMode === 'url') {
            if (player.getPlayerState() === YT.PlayerState.PLAYING) {
                player.pauseVideo();
            } else {
                player.playVideo();
            }
        } else {
            if (localPlayer.paused) {
                localPlayer.play();
            } else {
                localPlayer.pause();
            }
        }
    };

    document.getElementById('slowDown').onclick = () => {
        if (appMode === 'url') {
            player.setPlaybackRate(0.5);
        } else {
            localPlayer.playbackRate = 0.5;
        }
    };
    
    document.getElementById('normalSpeed').onclick = () => {
        if (appMode === 'url') {
            player.setPlaybackRate(1);
        } else {
            localPlayer.playbackRate = 1;
        }
    };
    
    document.getElementById('speedUp').onclick = () => {
        if (appMode === 'url') {
            player.setPlaybackRate(1.5);
        } else {
            localPlayer.playbackRate = 1.5;
        }
    };

    // 时间点设置 (URL模式)
    document.getElementById('setStartBtn').onclick = setStartPoint;
    document.getElementById('setEndBtn').onclick = setEndPoint;
    document.getElementById('jumpStartBtn').onclick = () => jumpToTime(startTime);
    document.getElementById('jumpEndBtn').onclick = () => jumpToTime(endTime);

    // 片段操作 (URL模式)
    document.getElementById('addSegmentBtn').onclick = addSegment;
    document.getElementById('previewSegmentBtn').onclick = previewSegment;
    
    // 导出和清空 (URL模式)
    document.getElementById('exportJsonBtn').onclick = exportToJSON;
    document.getElementById('clearAllBtn').onclick = clearAll;

    // 本地片段导航
    document.getElementById('prevClipBtn').onclick = () => playLocalClip(currentClipIndex - 1);
    document.getElementById('nextClipBtn').onclick = () => playLocalClip(currentClipIndex + 1);
    
    // 本地片段操作
    document.getElementById('deleteClipBtn').onclick = deleteCurrentClip;
    document.getElementById('extractClipBtn').onclick = showExtractPanel;
    document.getElementById('setExtractStartBtn').onclick = setExtractStart;
    document.getElementById('setExtractEndBtn').onclick = setExtractEnd;
    document.getElementById('confirmExtractBtn').onclick = confirmExtract;
    document.getElementById('cancelExtractBtn').onclick = () => {
        document.getElementById('extractPanel').style.display = 'none';
    };

    // 时间轴点击
    document.getElementById('timelineCanvas').onclick = onTimelineClick;

    // 本地播放器时间更新
    localPlayer.addEventListener('timeupdate', () => {
        if (appMode === 'local') {
            updateLocalTimeDisplay();
        }
    });

    // 键盘快捷键
    document.addEventListener('keydown', handleKeyboard);
}

function handleKeyboard(e) {
    // 如果焦点在输入框，不处理快捷键
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch(e.key.toLowerCase()) {
        case ' ':
            e.preventDefault();
            document.getElementById('playPauseBtn').click();
            break;
            
        // URL模式快捷键
        case 'i':
            if (appMode === 'url') {
                e.preventDefault();
                setStartPoint();
            }
            break;
        case 'o':
            if (appMode === 'url') {
                e.preventDefault();
                setEndPoint();
            }
            break;
        case 'enter':
            if (appMode === 'url') {
                e.preventDefault();
                addSegment();
            }
            break;
            
        // 本地片段模式快捷键
        case 'd':
            if (appMode === 'local' && currentClips.length > 0) {
                e.preventDefault();
                deleteCurrentClipQuick();  // 快速删除，无确认
            }
            break;
        case 'f':
            if (appMode === 'local') {
                e.preventDefault();
                playLocalClip(currentClipIndex + 1);  // 下一个片段
            }
            break;
        case 'a':
            if (appMode === 'local') {
                e.preventDefault();
                playLocalClip(currentClipIndex - 1);  // 上一个片段
            }
            break;
        case 's':
            if (appMode === 'local') {
                e.preventDefault();
                setExtractStart();  // 设置截取起点
            }
            break;
        case 'e':
            if (appMode === 'local') {
                e.preventDefault();
                setExtractEnd();  // 设置截取终点
            }
            break;
        case 'x':
            if (appMode === 'local' && extractStartTime !== null && extractEndTime !== null) {
                e.preventDefault();
                confirmExtractQuick();  // 快速截取，无确认
            }
            break;
        case 'c':
            if (appMode === 'local') {
                e.preventDefault();
                cancelExtract();  // 取消截取
            }
            break;
            
        // 播放速度控制
        case '1':
            e.preventDefault();
            if (appMode === 'url' && player) {
                player.setPlaybackRate(0.5);
            } else if (appMode === 'local') {
                localPlayer.playbackRate = 0.5;
            }
            break;
        case '2':
            e.preventDefault();
            if (appMode === 'url' && player) {
                player.setPlaybackRate(1);
            } else if (appMode === 'local') {
                localPlayer.playbackRate = 1;
            }
            break;
        case '3':
            e.preventDefault();
            if (appMode === 'url' && player) {
                player.setPlaybackRate(1.5);
            } else if (appMode === 'local') {
                localPlayer.playbackRate = 1.5;
            }
            break;
            
        // 左右箭头：快进快退
        case 'arrowleft':
            e.preventDefault();
            if (appMode === 'url' && player) {
                player.seekTo(Math.max(0, player.getCurrentTime() - 5), true);
            } else if (appMode === 'local') {
                localPlayer.currentTime = Math.max(0, localPlayer.currentTime - 5);
            }
            break;
        case 'arrowright':
            e.preventDefault();
            if (appMode === 'url' && player) {
                player.seekTo(player.getCurrentTime() + 5, true);
            } else if (appMode === 'local') {
                localPlayer.currentTime = Math.min(localPlayer.duration, localPlayer.currentTime + 5);
            }
            break;
    }
}

// 设置起止点
function setStartPoint() {
    if (appMode === 'url') {
        startTime = player.getCurrentTime();
    } else {
        startTime = localPlayer.currentTime;
    }
    updateTimeInputs();
    drawTimeline();
}

function setEndPoint() {
    if (appMode === 'url') {
        endTime = player.getCurrentTime();
    } else {
        endTime = localPlayer.currentTime;
    }
    updateTimeInputs();
    drawTimeline();
}

function updateTimeInputs() {
    document.getElementById('startTime').value = startTime !== null ? formatTime(startTime) : '';
    document.getElementById('endTime').value = endTime !== null ? formatTime(endTime) : '';
}

function jumpToTime(time) {
    if (time !== null) {
        if (appMode === 'url' && player) {
            player.seekTo(time, true);
        } else if (appMode === 'local') {
            localPlayer.currentTime = time;
        }
    }
}

// 添加片段
function addSegment() {
    if (startTime === null || endTime === null) {
        alert('请先设置起始和结束时间点');
        return;
    }
    
    if (startTime >= endTime) {
        alert('起始时间必须小于结束时间');
        return;
    }

    const label = document.getElementById('segmentLabel').value || `片段 ${segments.length + 1}`;
    
    segments.push({
        id: Date.now(),
        videoId: currentVideoId,
        start: startTime,
        end: endTime,
        label: label,
        duration: endTime - startTime
    });

    renderSegments();
    
    // 清空输入
    document.getElementById('segmentLabel').value = '';
    startTime = null;
    endTime = null;
    updateTimeInputs();
    drawTimeline();
}

// 预览片段
function previewSegment() {
    if (startTime !== null && endTime !== null) {
        player.seekTo(startTime, true);
        player.playVideo();
        
        setTimeout(() => {
            player.pauseVideo();
        }, (endTime - startTime) * 1000);
    }
}

// 渲染片段列表
function renderSegments() {
    const container = document.getElementById('segmentsListContainer');
    container.innerHTML = '';
    
    segments.forEach(segment => {
        const item = document.createElement('div');
        item.className = 'segment-item';
        item.innerHTML = `
            <div class="segment-info">
                <div class="segment-time">
                    ${formatTime(segment.start)} - ${formatTime(segment.end)}
                    (时长: ${formatTime(segment.duration)})
                </div>
                <div class="segment-label">${segment.label}</div>
            </div>
            <div class="segment-actions">
                <button class="play-btn" onclick="playSegment(${segment.id})">播放</button>
                <button class="delete-btn" onclick="deleteSegment(${segment.id})">删除</button>
            </div>
        `;
        container.appendChild(item);
    });
}

// 播放片段
window.playSegment = function(id) {
    const segment = segments.find(s => s.id === id);
    if (segment) {
        player.seekTo(segment.start, true);
        player.playVideo();
        
        setTimeout(() => {
            player.pauseVideo();
        }, segment.duration * 1000);
    }
};

// 删除片段
window.deleteSegment = function(id) {
    segments = segments.filter(s => s.id !== id);
    renderSegments();
};

// 导出 JSON
function exportToJSON() {
    const data = {
        videoId: currentVideoId,
        videoUrl: `https://www.youtube.com/watch?v=${currentVideoId}`,
        totalSegments: segments.length,
        segments: segments.map(s => ({
            label: s.label,
            start: s.start,
            end: s.end,
            duration: s.duration,
            startFormatted: formatTime(s.start),
            endFormatted: formatTime(s.end)
        }))
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `annotations_${currentVideoId}_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
}

// 清空所有
function clearAll() {
    if (confirm('确定要清空所有标注吗?')) {
        segments = [];
        renderSegments();
        startTime = null;
        endTime = null;
        updateTimeInputs();
        drawTimeline();
    }
}

// 绘制时间轴
function drawTimeline() {
    const canvas = document.getElementById('timelineCanvas');
    const ctx = canvas.getContext('2d');
    let duration = 100;
    
    if (appMode === 'url' && player) {
        duration = player.getDuration() || 100;
    } else if (appMode === 'local') {
        duration = localPlayer.duration || 100;
    }
    
    canvas.width = canvas.offsetWidth;
    canvas.height = 80;
    
    // 背景
    ctx.fillStyle = '#ecf0f1';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制片段 (仅URL模式)
    if (appMode === 'url') {
        segments.forEach(segment => {
            const startX = (segment.start / duration) * canvas.width;
            const width = ((segment.end - segment.start) / duration) * canvas.width;
            
            ctx.fillStyle = 'rgba(52, 152, 219, 0.5)';
            ctx.fillRect(startX, 0, width, canvas.height);
            
            ctx.strokeStyle = '#2980b9';
            ctx.lineWidth = 2;
            ctx.strokeRect(startX, 0, width, canvas.height);
        });
    }
    
    // 绘制当前标记
    if (startTime !== null) {
        const x = (startTime / duration) * canvas.width;
        ctx.strokeStyle = '#27ae60';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    
    if (endTime !== null) {
        const x = (endTime / duration) * canvas.width;
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, canvas.height);
        ctx.stroke();
    }
    
    // 绘制截取标记 (本地模式)
    if (appMode === 'local') {
        if (extractStartTime !== null) {
            const x = (extractStartTime / duration) * canvas.width;
            ctx.strokeStyle = '#27ae60';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
        
        if (extractEndTime !== null) {
            const x = (extractEndTime / duration) * canvas.width;
            ctx.strokeStyle = '#e74c3c';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(x, 0);
            ctx.lineTo(x, canvas.height);
            ctx.stroke();
        }
    }
}

function updateTimelineMarkers(current, duration) {
    drawTimeline();
    
    // 绘制当前播放位置
    const canvas = document.getElementById('timelineCanvas');
    const ctx = canvas.getContext('2d');
    const x = (current / duration) * canvas.width;
    
    ctx.strokeStyle = '#f39c12';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.stroke();
}

function onTimelineClick(e) {
    const canvas = document.getElementById('timelineCanvas');
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    let duration = 100;
    
    if (appMode === 'url' && player) {
        duration = player.getDuration();
    } else if (appMode === 'local') {
        duration = localPlayer.duration || 100;
    }
    
    const time = (x / canvas.width) * duration;
    
    if (appMode === 'url' && player) {
        player.seekTo(time, true);
    } else if (appMode === 'local') {
        localPlayer.currentTime = time;
    }
}

// 启动应用
init();
