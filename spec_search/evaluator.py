"""
============================================================
结果评估器
============================================================
根据硬性约束和评分函数筛选最优设定

优先级（第3/4章）:
  1. 平行趋势通过（事后≥2期连续≥2星显著 + 方向正确）
  2. 主回归≥2星显著 + 方向正确
  3. 跨章一致性（同X同区间）
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
    chapter: int = 3,
    max_pre_sig: int | None = None,
    min_post_consecutive: int | None = None,
    require_post_2star: bool = True,
) -> bool:
    """
    检查平行趋势是否满足硬性约束:

    1. 事前最多 max_pre_sig 期显著（默认1）
    2. 事后至少 min_post_consecutive 期连续显著（默认2）
    3. 事后显著系数方向正确
    4. 事后显著系数 ≥2星 (p < 0.05)（Ch3/Ch4 强制要求）
    """
    if not pt.success:
        return False

    max_pre_sig = max_pre_sig or config.PT_MAX_PRE_SIGNIFICANT
    min_post_consecutive = min_post_consecutive or config.PT_MIN_POST_CONSECUTIVE

    # 检查 1: 事前显著期数
    if pt.n_pre_sig > max_pre_sig:
        return False

    # 检查 2: 事后方向正确
    if not pt.post_correct_sign:
        return False

    # 检查 3: 事后连续显著（≥2星, p < 0.05）
    # 重新计算：只计 p < 0.05 的为"显著"
    post_pval_threshold = config.PT_POST_MIN_PVAL if require_post_2star else 0.10
    post_periods = sorted([p for p in pt.period_pval if p > 0])

    # 计算连续 ≥2星 显著期数
    max_consecutive_2star = 0
    current_streak = 0
    expected_sign = config.EXPECTED_SIGN.get(chapter, 0)

    for p in post_periods:
        pval = pt.period_pval.get(p, 1.0)
        coef = pt.period_coefs.get(p, 0.0)
        sign_ok = (expected_sign < 0 and coef < 0) or \
                  (expected_sign > 0 and coef > 0) or \
                  expected_sign == 0

        if pval < post_pval_threshold and sign_ok:
            current_streak += 1
            max_consecutive_2star = max(max_consecutive_2star, current_streak)
        else:
            current_streak = 0

    if max_consecutive_2star < min_post_consecutive:
        return False

    return True


def evaluate_regression(result: RegressionResult, chapter: int) -> float:
    """
    对回归结果打分。

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


def evaluate_regression_relaxed(result: RegressionResult, chapter: int) -> float:
    """
    对回归结果打分（放宽版: 允许1星通过，用于 Phase 2 → Phase 3 筛选）。
    仅要求方向正确 + p < 0.10。
    """
    if not result.success:
        return float("-inf")
    if not check_sign(result.coef, chapter):
        return float("-inf")
    if not check_significance(result.pval, 0.10):
        return float("-inf")

    score = 0.0
    if result.pval < 0.01:
        score += 100
    elif result.pval < 0.05:
        score += 60
    elif result.pval < 0.10:
        score += 30

    if result.se > 0:
        score += min(abs(result.tstat), 10) * 5
    if result.nobs > 0:
        score += np.log(result.nobs) * 2
    if not np.isnan(result.r2):
        score += result.r2 * 20

    return score


def evaluate_parallel_trends(pt: PTResult, chapter: int) -> float:
    """
    对平行趋势结果打分。
    要求事后系数≥2星显著 + 方向正确。

    Returns
    -------
    float
        综合得分。
    """
    if not pt.success:
        return float("-inf")

    if not check_parallel_trends(pt, chapter=chapter, require_post_2star=True):
        return float("-inf")

    score = 0.0

    # 事前不显著越多越好
    max_possible_pre = abs(pt.pre_window)
    n_pre_insig = max_possible_pre - pt.n_pre_sig
    score += n_pre_insig * 20

    # 事后连续≥2星显著期数（重新计算）
    post_pval_threshold = config.PT_POST_MIN_PVAL
    post_periods = sorted([p for p in pt.period_pval if p > 0])
    expected_sign = config.EXPECTED_SIGN.get(chapter, 0)

    max_consecutive_2star = 0
    current_streak = 0
    total_2star = 0

    for p in post_periods:
        pval = pt.period_pval.get(p, 1.0)
        coef = pt.period_coefs.get(p, 0.0)
        sign_ok = (expected_sign < 0 and coef < 0) or \
                  (expected_sign > 0 and coef > 0) or \
                  expected_sign == 0

        if pval < post_pval_threshold and sign_ok:
            current_streak += 1
            total_2star += 1
            max_consecutive_2star = max(max_consecutive_2star, current_streak)
        else:
            current_streak = 0

    # 连续≥2星显著越多越好（核心指标）
    score += max_consecutive_2star * 40

    # 总≥2星显著期数
    score += total_2star * 15

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

    优先级（Ch3/Ch4）: PT通过 > 主回归显著
    """
    reg_score = evaluate_regression(reg_result, chapter)

    # 第五章放宽: 仅要求主回归显著
    if is_ch5_robustness and reg_score > float("-inf"):
        return reg_score

    if pt_result is not None:
        pt_score = evaluate_parallel_trends(pt_result, chapter)
        if pt_score > float("-inf"):
            # PT 通过: 给予巨大加成，确保 PT-qualified 始终排在前面
            return reg_score + pt_score + 500
        else:
            # PT 未通过：仅保留主回归分数（无加成）
            return reg_score
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
    ch3_pt_pass: bool = False
    ch4_pt_pass: bool = False


def check_cross_chapter_consistency(
    ch3_results: list[dict],
    ch4_results: list[dict],
    ch5_results: list[dict] | None = None,
) -> list[CrossChapterResult]:
    """
    检查跨章节一致性:
    - 解释变量相同
    - 样本区间相同
    - 优先选择 Ch3+Ch4 都通过平行趋势的组合

    Returns
    -------
    list[CrossChapterResult]
        按总分排序（PT双通过优先）
    """
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

        # 优先选择 PT-qualified 的结果
        ch3_list = ch3_by_x[key]
        ch4_list = ch4_by_x[key]

        # Ch3: 优先 PT 通过的
        ch3_pt = [r for r in ch3_list if r.get("pt_qualified", False)]
        best_ch3 = (
            max(ch3_pt, key=lambda r: r.get("score", float("-inf")))
            if ch3_pt else
            max(ch3_list, key=lambda r: r.get("score", float("-inf")))
        )

        # Ch4: 优先 PT 通过的
        ch4_pt = [r for r in ch4_list if r.get("pt_qualified", False)]
        best_ch4 = (
            max(ch4_pt, key=lambda r: r.get("score", float("-inf")))
            if ch4_pt else
            max(ch4_list, key=lambda r: r.get("score", float("-inf")))
        )

        best_ch5 = None
        if key in ch5_by_x and ch5_by_x[key]:
            best_ch5 = max(ch5_by_x[key], key=lambda r: r.get("score", float("-inf")))

        ch3_has_pt = best_ch3.get("pt_qualified", False)
        ch4_has_pt = best_ch4.get("pt_qualified", False)

        total_score = best_ch3.get("score", 0) + best_ch4.get("score", 0)
        if best_ch5:
            total_score += best_ch5.get("score", 0) * 0.5

        # 双PT通过 → 巨大加成
        if ch3_has_pt and ch4_has_pt:
            total_score += 1000
        elif ch3_has_pt or ch4_has_pt:
            total_score += 400

        cc = CrossChapterResult(
            x_var=x_var,
            start_year=start_year,
            end_year=end_year,
            ch3_best=best_ch3,
            ch4_best=best_ch4,
            ch5_best=best_ch5,
            total_score=total_score,
            consistent=True,
            ch3_pt_pass=ch3_has_pt,
            ch4_pt_pass=ch4_has_pt,
        )
        results.append(cc)

    results.sort(key=lambda r: r.total_score, reverse=True)
    return results
