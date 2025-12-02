# 巨潮资讯并购公告爬虫

从巨潮资讯网站批量爬取国泰安并购简表相关的并购公告，支持智能过滤和内容验证。

## 功能特点

1. **智能公告匹配**：根据国泰安并购简表的数据（证券代码、日期范围）精准匹配公告
2. **完整性保障**：自动爬取首次公告日及后续所有相关公告，不遗漏任何更新
3. **内容验证**：自动识别公告是否包含关键信息（企业动机、标的资产业务范围、并购目的、对收购方影响）
4. **去重过滤**：避免下载无效和重复公告
5. **断点续传**：支持中断后继续下载

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备并购简表数据

将国泰安并购简表导出为Excel或CSV文件，必须包含以下字段：
- `Symbol`: 证券代码（如 000001）
- `FirstDeclareDate`: 首次公告日期
- `LatestDeclareDate`: 最新公告日期（可选）
- `FinishDeclareDate`: 完成公告日期（可选）
- `EventID`: 事件ID（用于记录关联）

示例格式（Excel/CSV）：
```
EventID,Symbol,FirstDeclareDate,LatestDeclareDate,Buyer,Seller,Underlying
20000001,000001,2020-01-15,2020-03-20,平安银行,某公司,某资产
20000002,000002,2020-02-10,2020-02-10,万科A,某公司,某股权
```

### 2. 运行爬虫

```bash
# 基本用法
python main.py --input ma_data.xlsx --output ./announcements

# 指定日期范围
python main.py --input ma_data.xlsx --output ./announcements --start-date 2020-01-01 --end-date 2023-12-31

# 只验证内容，不下载
python main.py --input ma_data.xlsx --output ./announcements --validate-only

# 设置并发数
python main.py --input ma_data.xlsx --output ./announcements --workers 5
```

### 3. 查看结果

程序会生成以下文件：
- `announcements/`: 下载的PDF公告文件
- `download_log.xlsx`: 下载记录，包含文件名、下载状态、验证结果
- `valid_announcements.xlsx`: 通过验证的公告列表
- `invalid_announcements.xlsx`: 未通过验证的公告列表

## 配置选项

在 `config.py` 中可以配置：
- 并发数量
- 重试次数
- 超时时间
- 关键词列表（用于内容验证）
- 下载延迟

## 公告内容验证规则

程序会检查公告是否包含以下关键信息：

1. **企业动机**：并购动机、战略目的、发展战略等
2. **标的资产和业务**：标的资产、标的公司、业务范围、主营业务等
3. **并购目的**：交易目的、本次交易目的、收购目的等
4. **对收购方影响**：对公司影响、对上市公司影响、交易影响等

## 注意事项

1. 请遵守巨潮资讯网站的使用条款和robots.txt规定
2. 建议设置适当的请求延迟，避免对服务器造成压力
3. 首次运行时建议先用小数据集测试
4. 网络不稳定时程序会自动重试

## 许可证

MIT License
