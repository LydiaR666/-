@echo off
chcp 65001 >nul
echo ========================================
echo   步骤1: 转换Excel格式
echo ========================================
echo.
echo 这个工具会自动将你的Excel文件转换为程序需要的格式
echo 转换内容包括:
echo   - 列名标准化 (event_id → EventID)
echo   - 日期格式标准化 (2007/11/22 → 2007-11-22)
echo   - 股票代码补齐6位
echo.
echo 按任意键开始...
pause >nul
echo.

python 格式转换工具.py

echo.
pause
