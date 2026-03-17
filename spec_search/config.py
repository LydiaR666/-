"""
============================================================
回归设定搜索（Specification Search）— 配置文件
============================================================
定义所有变量、控制变量组合、搜索网格参数
"""

import os

# ============================================================
# 数据路径
# ============================================================
DATA_PATH = (
    "/Users/rocket/Documents/科研/DA&TA(22.9)/2数字并购/"
    "原始数据/国泰安并购简表06-25/财务指标_合并DMA_完整_控1.dta"
)
TEMPLATE_PATH = (
    "/Users/rocket/Documents/科研/DA&TA(22.9)/2数字并购/"
    "do及dta/表格模板1.docx"
)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")

# ============================================================
# 面板标识变量
# ============================================================
FIRM_ID = "Stkcd"        # 股票代码
YEAR_VAR = "Year"        # 年份
IND_VAR = "Ind"          # 行业代码（证监会二级 / Indr）
# 备选行业代码
IND_VAR_ALT = "Indr"

# ============================================================
# 被解释变量 — 按章节分组
# ============================================================
Y_VARS = {
    3: ["F_NCSKEW_Cmdos", "F_DUVOL_Cmdos"],
    4: [
        "SumSalary", "SumAllowance", "TotalSalary",
        "ln_pay", "d_ln_pay", "est_m1",
        "ln_avg_salary", "ln_Top1ManageSumSalary", "ln_Top3ManageSumSalary",
        "MP1", "MP2", "GMP", "Perk1", "Perk2", "Perk3",
        "AGMP", "Perk6", "perk1", "perk2", "perk3",
    ],
    5: [
        "MGAP1", "MGAP2", "MGAP3", "MGAP4", "MGAP5", "MGAP6", "MGAP7",
        "Eperks1", "Eperks2", "Eperks3",
        "MGAP8", "EPerks1", "EPerks2", "EPerks3",
    ],
}

# 理论预期方向: 正数=期望正系数, 负数=期望负系数
EXPECTED_SIGN = {
    3: -1,   # 数字并购 → 股价崩盘风险降低（负）
    4:  1,   # 数字并购 → 高管薪酬升高（正）
    5: -1,   # 数字并购 → 超额薪酬降低（负）
}

# ============================================================
# 解释变量（DID 处理变量）
# ============================================================
X_VARS_PRIMARY = [
    "DMA_Scope_DID",
    "DMA_Scope2_DID",
    "DMA_Scope3_DID",
    "DMA_Ind_DID",
    "DMA_RD_DID",
]

X_VARS_ROBUSTNESS = [
    # count 类
    "count_DMA_经营范围数字化关键词",
    "count_DMA_经营范围数字化关键词2",
    "count_DMA_经营范围数字化关键词3",
    "count_DMA_数字经济核心产业",
    "count_DMA_数字经济核心产业1",
    "count_DMA_专利软件",
    "count_DMA_专利软件1",
    "count_DMA_专利软件2",
    # ln_amt 类
    "ln_amt_DMA_经营范围数字化关键词",
    "ln_amt_DMA_经营范围数字化关键词2",
    "ln_amt_DMA_经营范围数字化关键词3",
    "ln_amt_DMA_数字经济核心产业",
    "ln_amt_DMA_数字经济核心产业1",
    "ln_amt_DMA_专利软件",
    "ln_amt_DMA_专利软件1",
    "ln_amt_DMA_专利软件2",
]

# ============================================================
# 控制变量组
# ============================================================
CONTROL_GROUPS = {
    "Controls_Share": ["Size", "Lev", "Roa", "Turnover", "Ret", "MB", "Soe"],
    "Controls_Share1": [
        "Size", "Lev", "Age", "MB", "Cashflow", "Roa",
        "PI", "Largest", "Ret", "Sigma", "Accm",
    ],
    "Controls_Salary1": [
        "Size", "Lev", "Roa", "Growth", "Cashflow", "Ret", "MB",
        "TobinQ", "Age", "Soe", "Largest", "Mshare", "Dual", "Indep", "Board",
    ],
    "Controls_Salary": ["Size", "Lev", "Roa", "Board", "Indep", "Dual", "Largest", "Soe"],
    "Controls_Excess": ["Size", "Lev", "Roa", "Cashflow", "Board", "Indep", "Dual"],
}

# 各章节可用控制变量组
CHAPTER_CONTROL_GROUPS = {
    3: ["Controls_Share", "Controls_Share1"],
    4: ["Controls_Salary", "Controls_Salary1"],
    5: ["Controls_Excess", "Controls_Salary"],
}

# ============================================================
# 搜索网格
# ============================================================

# 样本区间
SAMPLE_START_YEARS = [2006, 2007, 2008, 2009, 2010]
SAMPLE_END_YEARS = [2023, 2024]

# 样本筛选条件
SAMPLE_FILTERS = {
    "drop_st_pt": True,          # 剔除ST/PT
    "drop_finance": True,        # 剔除金融行业
    "drop_real_estate": True,    # 剔除房地产行业
    "drop_lev_gt1": True,        # 剔除Lev>1
    "drop_ind_lt30": True,       # 剔除行业观测<30
    "winsorize_level": 0.01,     # 连续变量缩尾水平（0.01 或 0.05）
}

# 缩尾水平选项
WINSORIZE_OPTIONS = [None, 0.01, 0.05]

# 样本筛选组合（每个元素是一组开关）
FILTER_COMBINATIONS = [
    # 基础: 不做任何筛选
    {"drop_st_pt": False, "drop_finance": False, "drop_real_estate": False,
     "drop_lev_gt1": False, "drop_ind_lt30": False, "winsorize": None},
    # 标准筛选
    {"drop_st_pt": True, "drop_finance": True, "drop_real_estate": True,
     "drop_lev_gt1": True, "drop_ind_lt30": False, "winsorize": 0.01},
    # 标准 + 1% 缩尾
    {"drop_st_pt": True, "drop_finance": True, "drop_real_estate": False,
     "drop_lev_gt1": True, "drop_ind_lt30": False, "winsorize": 0.01},
    # 宽松
    {"drop_st_pt": True, "drop_finance": False, "drop_real_estate": False,
     "drop_lev_gt1": False, "drop_ind_lt30": False, "winsorize": None},
    # 严格
    {"drop_st_pt": True, "drop_finance": True, "drop_real_estate": True,
     "drop_lev_gt1": True, "drop_ind_lt30": True, "winsorize": 0.01},
    # 5% 缩尾
    {"drop_st_pt": True, "drop_finance": True, "drop_real_estate": True,
     "drop_lev_gt1": True, "drop_ind_lt30": False, "winsorize": 0.05},
]

# 固定效应组合
FE_COMBINATIONS = [
    [FIRM_ID, YEAR_VAR],                 # 个体+年份
    [FIRM_ID, IND_VAR, YEAR_VAR],        # 个体+行业+年份
    [IND_VAR, YEAR_VAR],                 # 行业+年份
    [YEAR_VAR],                          # 仅年份
]

# 聚类标准误组合
CLUSTER_OPTIONS = [
    [FIRM_ID],               # 个体聚类
    [IND_VAR],               # 行业聚类
    [FIRM_ID, YEAR_VAR],     # 双向聚类
]

# 平行趋势参数
PT_PRE_WINDOWS = [-3, -4, -5]       # 事件前窗口
PT_POST_WINDOWS = [3, 4, 5]         # 事件后窗口
PT_BASE_PERIODS = ["pre1", "pre0", "pre_biggest"]  # 基期选择

# PSM 匹配方法
PSM_METHODS = [
    {"method": "nearest", "n_neighbors": 1, "caliper": None},
    {"method": "nearest", "n_neighbors": 4, "caliper": None},
    {"method": "nearest", "n_neighbors": 1, "caliper": 0.05},
]

# ============================================================
# 评分与约束
# ============================================================

# 显著性阈值
SIGNIFICANCE_LEVELS = {
    "***": 0.01,
    "**": 0.05,
    "*": 0.10,
}

# 平行趋势硬约束
PT_MAX_PRE_SIGNIFICANT = 1      # 事前最多允许1期显著
PT_MIN_POST_CONSECUTIVE = 2     # 事后至少连续2期显著
PT_MAX_POST_LAG = 2             # 滞后效应最多2年
PT_POST_MIN_PVAL = 0.05         # 事后显著系数的最低显著性水平 (≥2星)
PT_PRIORITY_OVER_MAIN = True    # 平行趋势优先级高于主回归 (Ch3/Ch4)

# 第五章放宽约束
CH5_RELAXED = True  # 第五章可仅主回归显著，作为稳健性检验

# ============================================================
# 输出配置
# ============================================================
TABLE_FONT = "Times New Roman"
TABLE_FONT_SIZE = 10.5  # 五号
TABLE_CN_FONT = "宋体"
FIGURE_DPI = 300
FIGURE_SIZE = (8, 5)
