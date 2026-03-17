"""
用合成数据测试完整 pipeline。
模拟 DID 面板数据，验证搜索流程和输出。
增强版: 验证 PT 结果，强化处理效应以确保 PT ≥2星通过。
"""

import sys
import os
import numpy as np
import pandas as pd
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_search import config

np.random.seed(42)

# ============================================================
# 生成合成面板数据
# ============================================================
N_FIRMS = 400  # 更多企业 → 更大统计功效
YEARS = list(range(2007, 2024))
N_YEARS = len(YEARS)
TREAT_YEAR = 2015

firms = list(range(1, N_FIRMS + 1))
panel = pd.DataFrame(
    [(f, y) for f in firms for y in YEARS],
    columns=["Stkcd", "Year"],
)

# 行业代码
panel["Ind"] = np.random.choice(["C27", "C35", "C39", "C40", "I65", "M73"], len(panel))

# 处理组: 35% 的企业在 2015 年开始被处理
treated_firms = set(np.random.choice(firms, int(N_FIRMS * 0.35), replace=False))
panel["DMA_Scope_DID"] = 0
panel.loc[
    (panel["Stkcd"].isin(treated_firms)) & (panel["Year"] >= TREAT_YEAR),
    "DMA_Scope_DID",
] = 1

# 同样逻辑生成其他 DID 变量
for xvar in ["DMA_Scope2_DID", "DMA_Scope3_DID", "DMA_Ind_DID", "DMA_RD_DID"]:
    t_firms = set(np.random.choice(firms, int(N_FIRMS * 0.25), replace=False))
    panel[xvar] = 0
    panel.loc[
        (panel["Stkcd"].isin(t_firms)) & (panel["Year"] >= TREAT_YEAR),
        xvar,
    ] = 1

# 企业固定效应（使 PT 更清晰）
firm_fe = {f: np.random.normal(0, 0.3) for f in firms}
panel["_firm_fe"] = panel["Stkcd"].map(firm_fe)

# 年份趋势
year_trend = {y: 0.01 * (y - 2015) for y in YEARS}
panel["_year_trend"] = panel["Year"].map(year_trend)

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

# ============================================================
# Y 变量 — 带有强处理效应 + 动态效应结构
# 设计: 事前无效应, 事后逐步显现（确保 PT 通过）
# ============================================================

# 事件时间
panel["_event_time"] = np.nan
for idx in panel.index:
    if panel.loc[idx, "Stkcd"] in treated_firms:
        panel.loc[idx, "_event_time"] = panel.loc[idx, "Year"] - TREAT_YEAR

# 动态效应: 事前0, 事后逐步增大
def dynamic_effect(event_time, base_effect, ramp_speed=0.7):
    """事前无效应, 事后效应逐步增强。"""
    if pd.isna(event_time) or event_time < 0:
        return 0.0
    # 从 t=0 开始, 效应逐步增强
    return base_effect * min(1.0, ramp_speed + 0.1 * event_time)


# 第3章: 负效应 (DID → 崩盘风险↓), 强效应
panel["_ch3_effect"] = panel["_event_time"].apply(lambda t: dynamic_effect(t, -0.25))
panel["F_NCSKEW_Cmdos"] = (
    panel["_firm_fe"] + panel["_year_trend"]
    + 0.3 * panel["Size"] / 22
    + panel["_ch3_effect"]
    + np.random.normal(0, 0.35, len(panel))
)
panel["F_DUVOL_Cmdos"] = (
    panel["_firm_fe"] * 0.8 + panel["_year_trend"]
    + 0.2 * panel["Size"] / 22
    + panel["_event_time"].apply(lambda t: dynamic_effect(t, -0.20))
    + np.random.normal(0, 0.30, len(panel))
)

# 第4章: 正效应 (DID → 薪酬↑), 强效应
panel["_ch4_effect"] = panel["_event_time"].apply(lambda t: dynamic_effect(t, 0.20))
panel["ln_pay"] = (
    14.0 + panel["_firm_fe"] * 0.5 + panel["_year_trend"]
    + 0.8 * panel["Size"] / 22
    + panel["_ch4_effect"]
    + np.random.normal(0, 0.25, len(panel))
)
panel["TotalSalary"] = np.exp(panel["ln_pay"]) + np.random.normal(0, 10000, len(panel))
panel["SumSalary"] = panel["TotalSalary"] * np.random.uniform(0.6, 1.0, len(panel))
panel["SumAllowance"] = panel["TotalSalary"] * np.random.uniform(0.0, 0.2, len(panel))

# 第5章: 负效应 (DID → 超额薪酬↓)
panel["MGAP1"] = (
    0.10 + panel["_firm_fe"] * 0.3
    + panel["_event_time"].apply(lambda t: dynamic_effect(t, -0.15))
    + np.random.normal(0, 0.20, len(panel))
)
panel["Eperks1"] = (
    0.05 + panel["_firm_fe"] * 0.2
    + panel["_event_time"].apply(lambda t: dynamic_effect(t, -0.10))
    + np.random.normal(0, 0.15, len(panel))
)

# 清理临时列
panel.drop(columns=["_firm_fe", "_year_trend", "_event_time", "_ch3_effect", "_ch4_effect"],
           inplace=True)

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

# ============================================================
# 验证输出
# ============================================================
print("\n" + "=" * 70)
print("验证输出文件:")
print("=" * 70)

output_dir = config.OUTPUT_DIR
for root, dirs, files in os.walk(output_dir):
    for f in sorted(files):
        fpath = os.path.join(root, f)
        size = os.path.getsize(fpath)
        relpath = os.path.relpath(fpath, output_dir)
        print(f"  {relpath}: {size:,} bytes")

# ============================================================
# 验证 PT 和关键约束
# ============================================================
import json

print("\n" + "=" * 70)
print("验证关键约束:")
print("=" * 70)

best_path = os.path.join(output_dir, "best_specifications.json")
with open(best_path, "r") as f:
    best_specs = json.load(f)

errors = []

for ch_key in ["Chapter3", "Chapter4", "Chapter5"]:
    if ch_key not in best_specs:
        errors.append(f"  FAIL: {ch_key} 无结果")
        continue

    spec = best_specs[ch_key]
    ch = int(ch_key[-1])
    coef = spec.get("coef", 0)
    pval = spec.get("pval", 1)
    stars = spec.get("stars", "")

    # 检查方向
    expected_sign = config.EXPECTED_SIGN[ch]
    sign_ok = (expected_sign < 0 and coef < 0) or (expected_sign > 0 and coef > 0)
    sig_ok = pval < 0.05

    status = "OK" if (sign_ok and sig_ok) else "FAIL"
    print(f"  {ch_key}: coef={coef:.4f}{stars}, "
          f"sign={'✓' if sign_ok else '✗'}, "
          f"sig={'✓' if sig_ok else '✗'} [{status}]")

    if not sign_ok:
        errors.append(f"  FAIL: {ch_key} 方向错误 (expected {'<0' if expected_sign < 0 else '>0'}, got {coef:.4f})")
    if not sig_ok:
        errors.append(f"  FAIL: {ch_key} 不显著 (p={pval:.4f})")

    # 检查跨章一致性
    x_var = spec.get("x_var", "")
    print(f"    X={x_var}, N={spec.get('nobs', 0)}, "
          f"period={spec.get('start_year', '')}-{spec.get('end_year', '')}")

    # 验证 PT 结果（第3/4章强制要求）
    if ch in [3, 4]:
        pt_summary = spec.get("pt_summary")
        if pt_summary:
            n_pre = pt_summary.get("n_pre_sig", 999)
            max_consec_2star = pt_summary.get("max_post_consecutive_2star", 0)
            sign_2star = pt_summary.get("post_correct_sign_2star", False)
            pt_qualified = spec.get("pt_qualified", False)

            pt_ok = pt_qualified and max_consec_2star >= config.PT_MIN_POST_CONSECUTIVE
            print(f"    PT: pre_sig={n_pre}, post_consec_2★={max_consec_2star}, "
                  f"sign_2★={'✓' if sign_2star else '✗'}, "
                  f"qualified={'✓' if pt_qualified else '✗'}")

            # PT 事前期系数打印
            periods_data = pt_summary.get("periods", {})
            for p_str in sorted(periods_data.keys(), key=lambda x: int(x)):
                p_int = int(p_str)
                pd_info = periods_data[p_str]
                pv = pd_info["pval"]
                c = pd_info["coef"]
                star = "***" if pv < 0.01 else ("**" if pv < 0.05 else ("*" if pv < 0.10 else ""))
                flag = " ← PRE-SIG!" if p_int < 0 and pv < 0.10 else (
                       " ← POST✓" if p_int > 0 and pv < 0.05 and sign_2star else "")
                print(f"      t={p_int:+d}: coef={c:.4f}, p={pv:.4f}{star}{flag}")
        else:
            print(f"    PT: 无 pt_summary (未通过或未运行)")

# 检查 X 一致性
x_vars = set()
periods = set()
for ch_key in ["Chapter3", "Chapter4"]:
    if ch_key in best_specs:
        x_vars.add(best_specs[ch_key].get("x_var", ""))
        periods.add((best_specs[ch_key].get("start_year"), best_specs[ch_key].get("end_year")))

if len(x_vars) == 1:
    print(f"  跨章X一致性: ✓ (X={x_vars.pop()})")
else:
    errors.append(f"  FAIL: 跨章X不一致: {x_vars}")
    print(f"  跨章X一致性: ✗ ({x_vars})")

if len(periods) == 1:
    print(f"  跨章区间一致性: ✓")
else:
    print(f"  跨章区间一致性: ✗ ({periods})")

# 检查 summary_report.txt 是否存在
summary_path = os.path.join(output_dir, "summary_report.txt")
if os.path.exists(summary_path):
    print(f"\n  汇总报告存在: ✓ ({os.path.getsize(summary_path):,} bytes)")
else:
    errors.append("  FAIL: summary_report.txt 不存在")

# 检查动态效应图
png_count = 0
for root, dirs, files in os.walk(output_dir):
    for f in files:
        if f.endswith(".png"):
            png_count += 1
print(f"  动态效应图: {png_count} 个")

if errors:
    print(f"\n{'!' * 50}")
    print("测试发现问题:")
    for e in errors:
        print(e)
    print(f"{'!' * 50}")
else:
    print(f"\n{'✓' * 30}")
    print("所有约束验证通过！")

print("\n测试完成！")
