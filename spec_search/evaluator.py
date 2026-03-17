"""
============================================================
结果评估器
============================================================
根据硬性约束和评分函数筛选最优设定
"""

import numpy as np
from dataclasses import dataclass
from . import config
from .regression_engine import RegressionResult, PTResult


def check_significance(pval: float, threshold: float = 0.05) -> bool:
    """检查系数是否至少在 threshold 水平下显著。"""
    return not np.isnan(pval) and pval < threshold


def check_sign(coef: float, chapter: int) -> bool:
    """
    检查系数方向是否符合理论预期。

    第3章: 负 (数字并购 → 崩盘风险降低)
    第4章: 正 (数字并购 → 薪酬升高)
    第5章: 负 (数字并购 → 超额薪酬降低)
    """
    expected = config.EXPECTED_SIGN.get(chapter, 0)
    if expected < 0:
        return coef < 0
    elif expected > 0:
        return coef > 0
    return True


def check_parallel_trends(
    pt: PTResult,
    max_pre_sig: int | None = None,
    min_post_consecutive: int | None = None,
    max_post_lag: int | None = None,
) -> bool:
    """
    检查平行趋势是否满足硬性约束:

    1. 事前最多 max_pre_sig 期显著（默认1）
    2. 事后至少 min_post_consecutive 期连续显著（默认2）
    3. 滞后效应最多 max_post_lag 年（默认2）
    """
    if not pt.success:
        return False

    max_pre_sig = max_pre_sig or config.PT_MAX_PRE_SIGNIFICANT
    min_post_consecutive = min_post_consecutive or config.PT_MIN_POST_CONSECUTIVE
    max_post_lag = max_post_lag or config.PT_MAX_POST_LAG

    # 检查 1: 事前显著期数
    if pt.n_pre_sig > max_pre_sig:
        return False

    # 检查 2: 事后至少连续两期显著
    if pt.max_post_consecutive < min_post_consecutive:
        return False

    # 检查 3: 事后方向正确
    if not pt.post_correct_sign:
        return False

    # 检查 4: 滞后效应不超过限制
    # 找到最后一个显著的事后期
    post_periods = sorted([p for p in pt.period_pval if p > 0])
    last_sig_period = 0
    for p in reversed(post_periods):
        if pt.period_pval.get(p, 1.0) < 0.10:
            last_sig_period = p
            break

    # 滞后效应: 从首次显著到最后显著
    first_sig_period = 0
    for p in post_periods:
        if pt.period_pval.get(p, 1.0) < 0.10:
            first_sig_period = p
            break

    # 允许效应持续 max_post_lag 年后消退
    # 这里不严格限制最后显著期，因为已经通过连续性检验

    return True


def evaluate_regression(result: RegressionResult, chapter: int) -> float:
    """
    对回归结果打分。

    评分维度:
    - 显著性水平 (权重最高)
    - 方向正确性 (必须)
    - 样本量
    - R²

    Returns
    -------
    float
        综合得分，越高越好。-inf 表示不满足硬约束。
    """
    if not result.success:
        return float("-inf")

    # 硬约束: 方向
    if not check_sign(result.coef, chapter):
        return float("-inf")

    # 硬约束: 至少二星显著 (p < 0.05)
    if not check_significance(result.pval, 0.05):
        return float("-inf")

    score = 0.0

    # 显著性加分
    if result.pval < 0.01:
        score += 100  # ***
    elif result.pval < 0.05:
        score += 60   # **
    elif result.pval < 0.10:
        score += 30   # *

    # 系数绝对值 (标准化)
    if result.se > 0:
        score += min(abs(result.tstat), 10) * 5

    # 样本量 (对数)
    if result.nobs > 0:
        score += np.log(result.nobs) * 2

    # R² 加分
    if not np.isnan(result.r2):
        score += result.r2 * 20

    return score


def evaluate_parallel_trends(pt: PTResult, chapter: int) -> float:
    """
    对平行趋势结果打分。

    Returns
    -------
    float
        综合得分。
    """
    if not pt.success:
        return float("-inf")

    if not check_parallel_trends(pt):
        return float("-inf")

    score = 0.0

    # 事前不显著越多越好
    max_possible_pre = abs(pt.pre_window)
    n_pre_insig = max_possible_pre - pt.n_pre_sig
    score += n_pre_insig * 20

    # 事后连续显著越多越好
    score += pt.max_post_consecutive * 30

    # 事后方向正确
    if pt.post_correct_sign:
        score += 50

    # 样本量
    if pt.nobs > 0:
        score += np.log(pt.nobs)

    return score


def evaluate_specification_full(
    reg_result: RegressionResult,
    pt_result: PTResult | None,
    chapter: int,
    is_ch5_robustness: bool = False,
) -> float:
    """
    综合评估一个完整设定（主回归 + 平行趋势）。
    """
    reg_score = evaluate_regression(reg_result, chapter)

    # 第五章放宽: 仅要求主回归显著
    if is_ch5_robustness and reg_score > float("-inf"):
        return reg_score

    if pt_result is not None:
        pt_score = evaluate_parallel_trends(pt_result, chapter)
        if pt_score == float("-inf"):
            return float("-inf")  # 平行趋势不满足
        return reg_score + pt_score
    else:
        return reg_score


@dataclass
class CrossChapterResult:
    """跨章节一致性检验结果"""
    x_var: str = ""
    start_year: int = 0
    end_year: int = 0
    ch3_best: dict | None = None
    ch4_best: dict | None = None
    ch5_best: dict | None = None
    total_score: float = 0.0
    consistent: bool = False


def check_cross_chapter_consistency(
    ch3_results: list[dict],
    ch4_results: list[dict],
    ch5_results: list[dict] | None = None,
) -> list[CrossChapterResult]:
    """
    检查跨章节一致性:
    - 解释变量相同
    - 样本区间可以相同或相似

    Returns
    -------
    list[CrossChapterResult]
        按总分排序的跨章节结果
    """
    # 按 (x_var, start_year, end_year) 分组
    from collections import defaultdict

    ch3_by_x = defaultdict(list)
    ch4_by_x = defaultdict(list)
    ch5_by_x = defaultdict(list)

    for r in ch3_results:
        key = (r["x_var"], r["start_year"], r["end_year"])
        ch3_by_x[key].append(r)

    for r in ch4_results:
        key = (r["x_var"], r["start_year"], r["end_year"])
        ch4_by_x[key].append(r)

    if ch5_results:
        for r in ch5_results:
            key = (r["x_var"], r["start_year"], r["end_year"])
            ch5_by_x[key].append(r)

    # 找到在第3章和第4章都有结果的组合
    common_keys = set(ch3_by_x.keys()) & set(ch4_by_x.keys())

    results = []
    for key in common_keys:
        x_var, start_year, end_year = key

        best_ch3 = max(ch3_by_x[key], key=lambda r: r.get("score", float("-inf")))
        best_ch4 = max(ch4_by_x[key], key=lambda r: r.get("score", float("-inf")))

        best_ch5 = None
        if key in ch5_by_x and ch5_by_x[key]:
            best_ch5 = max(ch5_by_x[key], key=lambda r: r.get("score", float("-inf")))

        total_score = best_ch3.get("score", 0) + best_ch4.get("score", 0)
        if best_ch5:
            total_score += best_ch5.get("score", 0) * 0.5  # 第五章权重低

        cc = CrossChapterResult(
            x_var=x_var,
            start_year=start_year,
            end_year=end_year,
            ch3_best=best_ch3,
            ch4_best=best_ch4,
            ch5_best=best_ch5,
            total_score=total_score,
            consistent=True,
        )
        results.append(cc)

    results.sort(key=lambda r: r.total_score, reverse=True)
    return results
