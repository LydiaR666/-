@echo off
chcp 65001 >nul
echo ========================================
echo   巨潮资讯并购公告爬虫 - 一键运行
echo ========================================
echo.

REM 设置变量（请根据实际情况修改）
set INPUT_FILE=C:\Users\10657\Desktop\并购公告\并购公告下载.xlsx
set OUTPUT_DIR=C:\Users\10657\Desktop\并购公告\爬取结果

echo 输入文件: %INPUT_FILE%
echo 输出目录: %OUTPUT_DIR%
echo.
echo 按任意键开始运行，或关闭窗口取消...
pause >nul

echo.
echo 正在运行爬虫程序...
echo ========================================
echo.

python main.py --input "%INPUT_FILE%" --output "%OUTPUT_DIR%"

echo.
echo ========================================
echo 运行完成！
echo 结果保存在: %OUTPUT_DIR%
echo ========================================
echo.
echo 按任意键关闭窗口...
pause >nul
