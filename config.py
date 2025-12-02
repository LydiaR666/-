"""
配置文件：巨潮资讯并购公告爬虫配置
"""

# 巨潮资讯API配置
CNINFO_API = {
    'base_url': 'http://www.cninfo.com.cn',
    'search_url': 'http://www.cninfo.com.cn/new/fulltextSearch/full',
    'announcement_url': 'http://www.cninfo.com.cn/new/disclosure/detail',
    'download_url': 'http://static.cninfo.com.cn/',
}

# 请求配置
REQUEST_CONFIG = {
    'timeout': 30,  # 请求超时时间（秒）
    'retry_times': 3,  # 重试次数
    'retry_delay': 2,  # 重试延迟（秒）
    'request_delay': 1,  # 请求间隔（秒），避免请求过快
    'max_workers': 3,  # 并发下载数量
}

# 请求头配置
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/javascript, */*; q=0.01',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate',
    'X-Requested-With': 'XMLHttpRequest',
    'Referer': 'http://www.cninfo.com.cn/',
}

# 并购公告关键词（用于搜索和过滤）
MA_KEYWORDS = [
    '资产收购', '资产购买', '收购资产',
    '股权收购', '股权转让', '股权购买',
    '资产重组', '重大资产重组',
    '吸收合并', '合并',
    '要约收购',
    '资产出售', '资产剥离', '资产转让',
    '资产置换',
]

# 公告类型代码（巨潮资讯分类）
ANNOUNCEMENT_TYPES = [
    'category_scgkfx_szsh',  # 首次公开发行及上市
    'category_ndbg_szsh',     # 年度报告
    'category_bndbg_szsh',    # 半年度报告
    'category_yjdbg_szsh',    # 一季度报告
    'category_sjdbg_szsh',    # 三季度报告
    'category_scgkfx_szsh',   # 首次公开发行
    'category_zzcggg_szsh',   # 重组公告
]

# 内容验证关键词配置
CONTENT_VALIDATION = {
    # 企业动机相关关键词
    'motive': [
        '并购动机', '收购动机', '交易动机',
        '战略目的', '战略意图', '发展战略',
        '交易背景', '交易原因',
        '符合公司', '有利于公司', '提升公司',
        '增强', '完善', '优化', '拓展', '延伸',
    ],

    # 标的资产和业务范围
    'target_business': [
        '标的资产', '标的公司', '交易标的',
        '业务范围', '经营范围', '主营业务',
        '资产情况', '资产构成', '资产明细',
        '业务情况', '经营情况',
        '标的', '拟购买', '拟收购', '拟出售',
    ],

    # 并购目的
    'purpose': [
        '交易目的', '本次交易目的', '收购目的', '并购目的',
        '通过本次', '本次交易旨在', '本次交易有利于',
        '为了', '以便', '实现', '达到',
    ],

    # 对收购方影响
    'impact': [
        '对公司影响', '对上市公司影响', '对本公司影响',
        '交易影响', '收购影响', '并购影响',
        '对公司的影响', '交易对公司',
        '影响分析', '影响说明',
        '有利影响', '不利影响', '重大影响',
        '财务影响', '经营影响', '业绩影响',
    ],
}

# 验证阈值：至少需要匹配的类别数
VALIDATION_THRESHOLD = 3  # 4个类别中至少匹配3个

# 无效公告过滤规则
INVALID_PATTERNS = [
    '更正公告', '补充公告', '致歉公告',
    '澄清公告', '风险提示', '停牌公告',
    '复牌公告', '问询函', '关注函',
    # 但不排除包含实质内容的更正/补充公告
]

# 文件保存配置
SAVE_CONFIG = {
    'pdf_folder': 'announcements',  # PDF保存目录
    'log_file': 'download_log.xlsx',  # 下载日志
    'valid_file': 'valid_announcements.xlsx',  # 有效公告列表
    'invalid_file': 'invalid_announcements.xlsx',  # 无效公告列表
    'temp_folder': 'temp',  # 临时文件目录
}

# 日期格式
DATE_FORMATS = ['%Y-%m-%d', '%Y/%m/%d', '%Y%m%d']

# 证券代码格式化（补齐到6位）
def format_stock_code(code):
    """格式化股票代码为6位"""
    code = str(code).strip()
    # 移除可能的市场后缀（.SZ, .SH等）
    if '.' in code:
        code = code.split('.')[0]
    # 补齐到6位
    return code.zfill(6)

# 日期范围扩展（天数）
# 为了不遗漏公告，在首次公告日前后扩展搜索范围
DATE_RANGE_EXTEND_DAYS = 7
