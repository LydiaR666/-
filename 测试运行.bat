@echo off
chcp 65001 >nul
echo ========================================
echo   巨潮资讯并购公告爬虫 - 测试运行
echo   （仅处理前2条数据）
echo ========================================
echo.

REM 设置变量（请根据实际情况修改）
set INPUT_FILE=C:\Users\10657\Desktop\并购公告\并购公告下载.xlsx
set OUTPUT_DIR=C:\Users\10657\Desktop\并购公告\测试结果

echo 输入文件: %INPUT_FILE%
echo 输出目录: %OUTPUT_DIR%
echo 测试模式: 仅处理前2条数据
echo.
echo 按任意键开始测试，或关闭窗口取消...
pause >nul

echo.
echo 正在运行测试...
echo ========================================
echo.

python main.py --input "%INPUT_FILE%" --output "%OUTPUT_DIR%" --sample 2

echo.
echo ========================================
echo 测试完成！
echo 结果保存在: %OUTPUT_DIR%
echo.
echo 如果测试成功，请运行"一键运行.bat"处理完整数据
echo ========================================
echo.
echo 按任意键关闭窗口...
pause >nul
