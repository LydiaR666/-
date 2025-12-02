# 快速入门指南

## 1. 环境准备

### 安装Python依赖

```bash
pip install -r requirements.txt
```

### 检查Python版本
要求 Python 3.7+

```bash
python --version
```

## 2. 准备数据

### 方式一：使用示例数据（快速测试）

```bash
# 生成示例数据
python example_ma_data.py

# 使用示例数据运行（仅处理前2条测试）
python main.py --input example_ma_data.xlsx --output ./test_output --sample 2
```

### 方式二：使用自己的并购简表数据

将国泰安并购简表导出为Excel文件，必须包含以下字段：

- `Symbol`: 证券代码（如 000001）
- `FirstDeclareDate`: 首次公告日期（格式：YYYY-MM-DD）
- `EventID`: 事件ID（用于文件命名和记录关联）

可选字段：
- `LatestDeclareDate`: 最新公告日期
- `FinishDeclareDate`: 完成公告日期
- 其他字段会被保留用于记录

## 3. 运行爬虫

### 基本用法

```bash
python main.py --input ma_data.xlsx --output ./announcements
```

### 常用参数

```bash
# 指定日期范围（只处理2020-2023年的并购事件）
python main.py --input ma_data.xlsx --output ./announcements \
    --start-date 2020-01-01 --end-date 2023-12-31

# 调整并发数（加快下载速度，但注意不要太高）
python main.py --input ma_data.xlsx --output ./announcements --workers 5

# 测试模式：只处理前10条数据
python main.py --input ma_data.xlsx --output ./announcements --sample 10

# 仅验证已下载的PDF，不重新下载
python main.py --input ma_data.xlsx --output ./announcements --validate-only
```

## 4. 查看结果

运行完成后，会在输出目录生成以下文件：

```
output/
├── announcements/              # 下载的PDF文件
│   ├── 20210001_000001_20210315_xxx_平安银行....pdf
│   ├── 20210001_000001_20210520_xxx_平安银行....pdf
│   └── ...
├── download_log.xlsx           # 完整下载记录
├── valid_announcements.xlsx    # 通过验证的有效公告
└── invalid_announcements.xlsx  # 未通过验证的公告
```

### 验证标准

公告会被自动验证是否包含以下4类关键信息：

1. **企业动机**：并购动机、战略目的等
2. **标的资产和业务**：标的资产、业务范围、经营情况等
3. **并购目的**：交易目的、收购目的等
4. **对收购方影响**：对公司影响、交易影响等

默认配置：至少匹配 **3/4** 类才算有效公告

## 5. 调整配置

编辑 `config.py` 可以调整：

```python
# 修改验证阈值（4类中至少匹配几类才算有效）
VALIDATION_THRESHOLD = 3  # 改为2会降低标准，改为4会提高标准

# 修改请求延迟（秒）
REQUEST_CONFIG = {
    'request_delay': 1,  # 增大可减少对服务器压力
    'timeout': 30,       # 超时时间
    'retry_times': 3,    # 重试次数
}

# 添加自定义关键词
CONTENT_VALIDATION = {
    'motive': [
        '并购动机', '收购动机',
        # 在这里添加更多关键词...
    ],
    # ...
}
```

## 6. 常见问题

### Q: 下载速度很慢怎么办？

A: 可以适当增加并发数，但不建议超过5：
```bash
python main.py --input ma_data.xlsx --output ./announcements --workers 5
```

### Q: 某些公告下载失败？

A: 程序会自动重试3次。如果仍然失败，可能是：
1. 公告已被删除或移动
2. 网络问题
3. 巨潮资讯网站的反爬虫机制

建议：增大请求延迟，或稍后重试

### Q: 如何只重新下载失败的公告？

A: 程序会自动记录已下载的公告，重新运行时会跳过已下载的。
如果想重新下载某些公告，删除对应的PDF文件即可。

### Q: 很多公告被标记为"无效"？

A: 可能原因：
1. 验证阈值设置过高
2. 关键词不够全面
3. PDF文本提取失败

解决方法：
1. 降低 `config.py` 中的 `VALIDATION_THRESHOLD`
2. 在 `config.py` 中添加更多关键词
3. 检查 `invalid_announcements.xlsx` 了解具体原因

### Q: 如何确保不遗漏非首次公告日的公告？

A: 程序会自动：
1. 在首次公告日前后各扩展7天搜索
2. 搜索到最新公告日期和完成公告日期
3. 使用并购关键词过滤相关公告

## 7. 进阶用法

### 批量处理多个Excel文件

```bash
# 使用shell脚本
for file in data/*.xlsx; do
    python main.py --input "$file" --output "./output/$(basename $file .xlsx)"
done
```

### 仅验证特定目录下的PDF

```python
from content_validator import ContentValidator
import os

validator = ContentValidator()
pdf_dir = './announcements'

for filename in os.listdir(pdf_dir):
    if filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_dir, filename)
        is_valid, result = validator.validate_pdf(pdf_path)
        print(f"{filename}: {'有效' if is_valid else '无效'}")
        print(validator.get_validation_summary(result))
```

## 8. 注意事项

1. **遵守法律法规**：使用本工具时请遵守《网络安全法》等相关法规
2. **尊重robots.txt**：不要过于频繁地请求服务器
3. **数据使用**：下载的公告仅供学术研究使用，不得用于商业目的
4. **备份数据**：建议定期备份下载的公告和记录文件

## 9. 技术支持

如遇到问题：
1. 查看 `README.md` 详细文档
2. 检查 `download_log.xlsx` 中的错误信息
3. 查看程序运行时的输出日志

## 10. 许可证

MIT License - 详见 LICENSE 文件
