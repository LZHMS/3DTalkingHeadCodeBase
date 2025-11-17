// YouTube API 和应用状态
let player;
let currentVideoId = '';
let videoList = [];
let segments = [];
let startTime = null;
let endTime = null;

// 从 public_speaking.txt 读取的视频列表 (手动复制或通过文件上传)
let VIDEO_URLS = [
    'https://www.youtube.com/watch?v=8S0FDjFBj8o',
    'https://www.youtube.com/watch?v=5MgBikgcWnY',
    'https://www.youtube.com/watch?v=2jHWxUovYRg',
];

// 初始化
function init() {
    loadVideoList();
    setupEventListeners();
    loadYouTubeAPI();
    setupFileUpload();
}

// 设置文件上传
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
        if (player.getPlayerState() === YT.PlayerState.PLAYING) {
            player.pauseVideo();
        } else {
            player.playVideo();
        }
    };

    document.getElementById('slowDown').onclick = () => player.setPlaybackRate(0.5);
    document.getElementById('normalSpeed').onclick = () => player.setPlaybackRate(1);
    document.getElementById('speedUp').onclick = () => player.setPlaybackRate(1.5);

    // 时间点设置
    document.getElementById('setStartBtn').onclick = setStartPoint;
    document.getElementById('setEndBtn').onclick = setEndPoint;
    document.getElementById('jumpStartBtn').onclick = () => jumpToTime(startTime);
    document.getElementById('jumpEndBtn').onclick = () => jumpToTime(endTime);

    // 片段操作
    document.getElementById('addSegmentBtn').onclick = addSegment;
    document.getElementById('previewSegmentBtn').onclick = previewSegment;
    
    // 导出和清空
    document.getElementById('exportJsonBtn').onclick = exportToJSON;
    document.getElementById('clearAllBtn').onclick = clearAll;

    // 时间轴点击
    document.getElementById('timelineCanvas').onclick = onTimelineClick;

    // 键盘快捷键
    document.addEventListener('keydown', handleKeyboard);
}

function handleKeyboard(e) {
    if (e.target.tagName === 'INPUT') return;
    
    switch(e.key) {
        case ' ':
            e.preventDefault();
            document.getElementById('playPauseBtn').click();
            break;
        case 'i':
        case 'I':
            setStartPoint();
            break;
        case 'o':
        case 'O':
            setEndPoint();
            break;
        case 'Enter':
            addSegment();
            break;
    }
}

// 设置起止点
function setStartPoint() {
    startTime = player.getCurrentTime();
    updateTimeInputs();
    drawTimeline();
}

function setEndPoint() {
    endTime = player.getCurrentTime();
    updateTimeInputs();
    drawTimeline();
}

function updateTimeInputs() {
    document.getElementById('startTime').value = startTime !== null ? formatTime(startTime) : '';
    document.getElementById('endTime').value = endTime !== null ? formatTime(endTime) : '';
}

function jumpToTime(time) {
    if (time !== null && player) {
        player.seekTo(time, true);
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
    const container = document.getElementById('segmentsList');
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
    const duration = player ? player.getDuration() : 100;
    
    canvas.width = canvas.offsetWidth;
    canvas.height = 80;
    
    // 背景
    ctx.fillStyle = '#ecf0f1';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // 绘制片段
    segments.forEach(segment => {
        const startX = (segment.start / duration) * canvas.width;
        const width = ((segment.end - segment.start) / duration) * canvas.width;
        
        ctx.fillStyle = 'rgba(52, 152, 219, 0.5)';
        ctx.fillRect(startX, 0, width, canvas.height);
        
        ctx.strokeStyle = '#2980b9';
        ctx.lineWidth = 2;
        ctx.strokeRect(startX, 0, width, canvas.height);
    });
    
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
    const duration = player.getDuration();
    const time = (x / canvas.width) * duration;
    
    player.seekTo(time, true);
}

// 启动应用
init();
