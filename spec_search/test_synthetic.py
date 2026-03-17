"""
用合成数据测试完整 pipeline。
模拟 DID 面板数据，验证搜索流程和输出。
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 修改 config 使用临时路径
from spec_search import config

np.random.seed(42)

# ============================================================
# 生成合成面板数据
# ============================================================
N_FIRMS = 300
YEARS = list(range(2007, 2024))
N_YEARS = len(YEARS)

firms = list(range(1, N_FIRMS + 1))
panel = pd.DataFrame(
    [(f, y) for f in firms for y in YEARS],
    columns=["Stkcd", "Year"],
)

# 行业代码
panel["Ind"] = np.random.choice(["C27", "C35", "C39", "C40", "I65", "M73"], len(panel))

# 处理组: 30% 的企业在 2015 年开始被处理
treated_firms = set(np.random.choice(firms, int(N_FIRMS * 0.30), replace=False))
panel["DMA_Scope_DID"] = 0
panel.loc[
    (panel["Stkcd"].isin(treated_firms)) & (panel["Year"] >= 2015),
    "DMA_Scope_DID",
] = 1

# 同样逻辑生成其他 DID 变量
for xvar in ["DMA_Scope2_DID", "DMA_Scope3_DID", "DMA_Ind_DID", "DMA_RD_DID"]:
    t_firms = set(np.random.choice(firms, int(N_FIRMS * 0.25), replace=False))
    panel[xvar] = 0
    panel.loc[
        (panel["Stkcd"].isin(t_firms)) & (panel["Year"] >= 2015),
        xvar,
    ] = 1

# 控制变量
panel["Size"] = np.random.normal(22, 1.5, len(panel))
panel["Lev"] = np.random.uniform(0.1, 0.9, len(panel))
panel["Roa"] = np.random.normal(0.05, 0.03, len(panel))
panel["Board"] = np.random.choice([5, 7, 9, 11, 13], len(panel))
panel["Indep"] = np.random.uniform(0.3, 0.6, len(panel))
panel["Dual"] = np.random.choice([0, 1], len(panel))
panel["Largest"] = np.random.uniform(0.1, 0.7, len(panel))
panel["Soe"] = np.random.choice([0, 1], len(panel))
panel["Turnover"] = np.random.uniform(0.3, 2.0, len(panel))
panel["Ret"] = np.random.normal(0.1, 0.3, len(panel))
panel["MB"] = np.random.uniform(1, 8, len(panel))
panel["Age"] = np.random.randint(3, 30, len(panel))
panel["Cashflow"] = np.random.normal(0.05, 0.05, len(panel))
panel["Growth"] = np.random.normal(0.1, 0.2, len(panel))
panel["TobinQ"] = np.random.uniform(1, 5, len(panel))
panel["Mshare"] = np.random.uniform(0, 0.5, len(panel))
panel["PI"] = np.random.normal(0, 0.02, len(panel))
panel["Sigma"] = np.random.uniform(0.01, 0.1, len(panel))
panel["Accm"] = np.random.normal(0, 0.05, len(panel))

# Y 变量 — 带有真实的处理效应
# 第3章: 负效应 (DID → 崩盘风险↓)
panel["F_NCSKEW_Cmdos"] = (
    -0.05 + 0.3 * panel["Size"] / 22 - 0.15 * panel["DMA_Scope_DID"]
    + np.random.normal(0, 0.5, len(panel))
)
panel["F_DUVOL_Cmdos"] = (
    -0.03 + 0.2 * panel["Size"] / 22 - 0.10 * panel["DMA_Scope_DID"]
    + np.random.normal(0, 0.4, len(panel))
)

# 第4章: 正效应 (DID → 薪酬↑)
panel["ln_pay"] = (
    14.0 + 0.8 * panel["Size"] / 22 + 0.12 * panel["DMA_Scope_DID"]
    + np.random.normal(0, 0.3, len(panel))
)
panel["TotalSalary"] = np.exp(panel["ln_pay"]) + np.random.normal(0, 10000, len(panel))
panel["SumSalary"] = panel["TotalSalary"] * np.random.uniform(0.6, 1.0, len(panel))
panel["SumAllowance"] = panel["TotalSalary"] * np.random.uniform(0.0, 0.2, len(panel))

# 第5章: 负效应 (DID → 超额薪酬↓)
panel["MGAP1"] = (
    0.10 - 0.08 * panel["DMA_Scope_DID"]
    + np.random.normal(0, 0.2, len(panel))
)
panel["Eperks1"] = (
    0.05 - 0.04 * panel["DMA_Scope_DID"]
    + np.random.normal(0, 0.15, len(panel))
)

# ST 标识
panel["ST"] = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], len(panel))

# 保存到临时文件
tmpdir = tempfile.mkdtemp()
data_path = os.path.join(tmpdir, "test_data.dta")
panel.to_stata(data_path, write_index=False, version=118)
print(f"合成数据已保存: {data_path}")
print(f"数据维度: {panel.shape}")

# ============================================================
# 覆盖 config 路径
# ============================================================
config.DATA_PATH = data_path
config.OUTPUT_DIR = os.path.join(tmpdir, "output")

# 只保留合成数据有的 Y 变量
config.Y_VARS = {
    3: ["F_NCSKEW_Cmdos", "F_DUVOL_Cmdos"],
    4: ["ln_pay", "TotalSalary", "SumSalary"],
    5: ["MGAP1", "Eperks1"],
}

# 减少搜索规模以加速测试
config.SAMPLE_START_YEARS = [2007, 2009]
config.SAMPLE_END_YEARS = [2023]
config.FE_COMBINATIONS = [
    ["Stkcd", "Year"],
    ["Ind", "Year"],
]
config.CLUSTER_OPTIONS = [
    ["Stkcd"],
]
config.FILTER_COMBINATIONS = [
    {"drop_st_pt": True, "drop_finance": False, "drop_real_estate": False,
     "drop_lev_gt1": False, "drop_ind_lt30": False, "winsorize": 0.01},
]
config.PT_PRE_WINDOWS = [-3]
config.PT_POST_WINDOWS = [3]
config.PT_BASE_PERIODS = ["pre1"]

# ============================================================
# 运行搜索
# ============================================================
from spec_search.main import main

print("\n" + "=" * 70)
print("开始合成数据测试...")
print("=" * 70)

main()

# 验证输出
print("\n" + "=" * 70)
print("验证输出文件:")
print("=" * 70)

output_dir = config.OUTPUT_DIR
for root, dirs, files in os.walk(output_dir):
    for f in files:
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        relpath = os.path.relpath(fpath, output_dir)
        print(f"  {relpath}: {size:,} bytes")

print("\n测试完成！")
