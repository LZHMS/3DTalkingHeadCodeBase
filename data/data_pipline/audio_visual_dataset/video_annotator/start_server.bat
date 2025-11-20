@echo off
chcp 65001 >nul
echo ====================================
echo 视频片段标注与管理工具
echo ====================================
echo.
echo 正在启动后端服务器...
echo 服务器地址: http://localhost:5000
echo.
echo 请在浏览器中打开 index.html 文件
echo.
echo 按 Ctrl+C 停止服务器
echo ====================================
echo.

python server.py
