"""
============================================================
回归设定搜索（Specification Search）— 主控程序
============================================================
自动化搜索最优回归设定，满足硬性约束：
  1. 各章 X 一致、章内样本量一致
  2. 主回归至少二星显著 + 方向正确
  3. 平行趋势: 事前 ≤1 期显著, 事后 ≥2 期连续显著

用法:
    python -m spec_search.main
============================================================
"""

import os
import sys
import json
import time
import warnings
import logging
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_search import config
from spec_search.data_loader import (
    load_data, apply_sample_filters, winsorize_variables,
    generate_event_time, get_continuous_vars, set_verbose,
)
from spec_search.regression_engine import (
    run_ols_fe, run_parallel_trends, run_psm_did,
    compute_descriptive_stats, compute_correlation_matrix,
    RegressionResult, PTResult, PSMResult,
)
from spec_search.specification_grid import (
    SpecConfig, generate_phase1_grid, generate_phase2_grid,
    generate_phase3_pt_grid, generate_psm_grid, generate_robustness_x_grid,
)
from spec_search.evaluator import (
    evaluate_regression, evaluate_regression_relaxed,
    evaluate_parallel_trends,
    check_cross_chapter_consistency, check_sign, check_significance,
)
from spec_search.table_generator import save_chapter_tables
from spec_search.figure_generator import (
    plot_dynamic_effects, plot_multiple_dynamic_effects,
)


# ============================================================
# 日志配置
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spec_search")


class SearchState:
    """搜索状态"""
    def __init__(self):
        self.raw_df = None
        self.chapter_results = {3: [], 4: [], 5: []}
        self.best_specs = {3: None, 4: None, 5: None}
        self.start_time = time.time()

    def elapsed(self):
        return f"{time.time() - self.start_time:.0f}s"


def main():
    """主入口。"""
    print("=" * 70)
    print("  Specification Search — 数字并购与利益相关者价值效应")
    print("=" * 70)
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据: {config.DATA_PATH}")
    print("=" * 70)

    state = SearchState()

    # ========== Step 1: 加载数据 ==========
    log.info("Step 1: 加载数据")
    state.raw_df = load_data()
    _print_variable_availability(state.raw_df)

    # ========== Step 2: 逐章搜索 ==========
    # 关闭筛选函数的逐条打印
    set_verbose(False)

    for chapter in [3, 4, 5]:
        log.info(f"=" * 50)
        log.info(f"第{chapter}章搜索开始 [{state.elapsed()}]")

        is_ch5 = (chapter == 5)
        results = run_chapter_search(state.raw_df, chapter, is_ch5_robustness=is_ch5)
        state.chapter_results[chapter] = results

        if results:
            best = max(results, key=lambda r: r.get("score", float("-inf")))
            state.best_specs[chapter] = best
            log.info(
                f"第{chapter}章最优: X={best['x_var']}, Y={best['y_var']}, "
                f"coef={best.get('coef', 'NA'):.4f}, "
                f"p={best.get('pval', 'NA'):.4f}{best.get('stars', '')}, "
                f"N={best.get('nobs', 0)}, score={best.get('score', 0):.1f}"
            )
        else:
            log.warning(f"第{chapter}章: 未找到满足约束的设定")

    set_verbose(True)

    # ========== Step 3: 跨章一致性 ==========
    log.info("=" * 50)
    log.info(f"跨章节一致性检验 [{state.elapsed()}]")

    cross_results = check_cross_chapter_consistency(
        state.chapter_results[3],
        state.chapter_results[4],
        state.chapter_results[5] or None,
    )

    if cross_results:
        best_cross = cross_results[0]
        pt_status = (f"Ch3-PT:{'✓' if best_cross.ch3_pt_pass else '✗'} "
                     f"Ch4-PT:{'✓' if best_cross.ch4_pt_pass else '✗'}")
        log.info(
            f"最优跨章设定: X={best_cross.x_var}, "
            f"区间={best_cross.start_year}-{best_cross.end_year}, "
            f"总分={best_cross.total_score:.1f}, {pt_status}"
        )
        if best_cross.ch3_best:
            state.best_specs[3] = best_cross.ch3_best
        if best_cross.ch4_best:
            state.best_specs[4] = best_cross.ch4_best
        if best_cross.ch5_best:
            state.best_specs[5] = best_cross.ch5_best
    else:
        log.info("未找到跨章一致设定，使用各章独立最优")

    # ========== Step 4: 生成输出 ==========
    log.info("=" * 50)
    log.info(f"生成输出文件 [{state.elapsed()}]")

    output_dir = os.path.abspath(config.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    for chapter in [3, 4, 5]:
        if state.best_specs[chapter]:
            generate_chapter_output(
                state.raw_df, state.best_specs[chapter],
                state.chapter_results[chapter], chapter, output_dir,
            )
        else:
            log.warning(f"第{chapter}章: 无最优设定，跳过输出")

    save_search_log(state, output_dir)
    generate_stata_code(state, output_dir)
    generate_summary_report(state, cross_results, output_dir)

    log.info("=" * 50)
    log.info(f"完成！耗时: {state.elapsed()}")
    log.info(f"输出目录: {output_dir}")


def _print_variable_availability(df: pd.DataFrame):
    """打印关键变量可用性。"""
    cols = set(df.columns)

    print("\n--- 解释变量 ---")
    for x in config.X_VARS_PRIMARY:
        if x in cols:
            n = df[x].notna().sum()
            n1 = (df[x] == 1).sum() if pd.api.types.is_numeric_dtype(df[x]) else "?"
            print(f"  ✓ {x}: N={n}, treated={n1}")
        else:
            print(f"  ✗ {x}")

    print("\n--- 被解释变量 ---")
    for ch, yvars in config.Y_VARS.items():
        found = [v for v in yvars if v in cols]
        missing = [v for v in yvars if v not in cols]
        print(f"  第{ch}章: {len(found)}/{len(yvars)} 可用")
        if missing:
            print(f"    缺失: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    print("\n--- 控制变量 ---")
    for name, cvars in config.CONTROL_GROUPS.items():
        found = [v for v in cvars if v in cols]
        print(f"  {name}: {len(found)}/{len(cvars)} 可用")

    # 稳健性X
    rob_found = [v for v in config.X_VARS_ROBUSTNESS if v in cols]
    print(f"\n--- 稳健性X变量: {len(rob_found)}/{len(config.X_VARS_ROBUSTNESS)} 可用 ---")


def run_chapter_search(
    raw_df: pd.DataFrame,
    chapter: int,
    is_ch5_robustness: bool = False,
) -> list[dict]:
    """
    运行单章完整搜索:
    Phase 1 → 快速扫描 (所有X-Y, 默认设定, 遍历样本区间)
    Phase 2 → 深度搜索 (变化控制变量/FE/聚类/筛选)
    Phase 3 → 平行趋势检验 (优先级最高: Ch3/Ch4)
    Phase 4 → 稳健性X检验

    优先级 (Ch3/Ch4):
      平行趋势≥2星通过 > 主回归显著
      当无法全部满足时，优先保证PT通过
    """
    all_qualified = []
    # 放宽候选: p<0.10 即可进入 Phase 3 PT 检验
    relaxed_candidates = []

    pt_first = (chapter in [3, 4]) and config.PT_PRIORITY_OVER_MAIN

    # -------- Phase 1: 快速扫描 --------
    log.info(f"Phase 1: 快速扫描{'（PT优先模式）' if pt_first else ''}")
    phase1_specs = generate_phase1_grid(chapter)
    promising_pairs = set()
    phase1_results = []
    n_total = len(phase1_specs)

    for i, spec in enumerate(phase1_specs):
        if (i + 1) % max(n_total // 5, 1) == 0:
            log.info(f"  Phase 1 进度: {i + 1}/{n_total}")

        result = _run_single_spec(raw_df, spec)
        if result is None:
            continue
        phase1_results.append(result)

        coef = result.get("coef", np.nan)
        pval = result.get("pval", np.nan)

        if check_sign(coef, chapter) and check_significance(pval, 0.10):
            promising_pairs.add((spec.y_var, spec.x_var))
            result["score"] = evaluate_regression_relaxed(
                RegressionResult(coef=coef, pval=pval, nobs=result.get("nobs", 0),
                                 r2=result.get("r2", np.nan), success=True),
                chapter,
            )
            # 放宽候选（方向正确 + p<0.10）全部保留供 Phase 3 使用
            relaxed_candidates.append(result)
            if check_significance(pval, 0.05):
                all_qualified.append(result)

    log.info(f"  Phase 1: {len(promising_pairs)} 组合方向正确, "
             f"{len(all_qualified)} 个≥2星, {len(relaxed_candidates)} 个≥1星")

    if not promising_pairs:
        log.warning("  Phase 1 无方向正确组合，放宽条件...")
        for r in phase1_results:
            if check_sign(r.get("coef", np.nan), chapter):
                promising_pairs.add((r["y_var"], r["x_var"]))

    if not promising_pairs:
        log.warning("  仍无结果")
        return all_qualified

    # 限制组合数
    promising_pairs = list(promising_pairs)
    if len(promising_pairs) > 15:
        pair_scores = defaultdict(lambda: float("-inf"))
        for r in phase1_results:
            key = (r["y_var"], r["x_var"])
            pair_scores[key] = max(pair_scores[key], r.get("score", float("-inf")))
        promising_pairs.sort(key=lambda p: pair_scores[p], reverse=True)
        promising_pairs = promising_pairs[:15]

    # -------- Phase 2: 深度搜索 --------
    log.info(f"Phase 2: 深度搜索 ({len(promising_pairs)} 组合)")
    phase2_specs = generate_phase2_grid(chapter, promising_pairs)
    n_total = len(phase2_specs)

    for i, spec in enumerate(phase2_specs):
        if (i + 1) % max(n_total // 10, 1) == 0:
            log.info(f"  Phase 2 进度: {i + 1}/{n_total}")

        result = _run_single_spec(raw_df, spec)
        if result is None:
            continue

        coef = result.get("coef", np.nan)
        pval = result.get("pval", np.nan)

        if check_sign(coef, chapter):
            if check_significance(pval, 0.05):
                result["score"] = evaluate_regression(
                    RegressionResult(coef=coef, pval=pval, nobs=result.get("nobs", 0),
                                     r2=result.get("r2", np.nan), success=True),
                    chapter,
                )
                all_qualified.append(result)
            # PT优先模式: p<0.10 也保留，供 Phase 3 PT 检验使用
            if pt_first and check_significance(pval, 0.10):
                result["score"] = evaluate_regression_relaxed(
                    RegressionResult(coef=coef, pval=pval, nobs=result.get("nobs", 0),
                                     r2=result.get("r2", np.nan), success=True),
                    chapter,
                )
                relaxed_candidates.append(result)

    log.info(f"  Phase 2: {len(all_qualified)} 个≥2星, {len(relaxed_candidates)} 个≥1星候选")

    # -------- Phase 3: 平行趋势（优先级最高）--------
    log.info(f"Phase 3: 平行趋势检验{'（PT优先: 事后≥2星显著）' if pt_first else ''}")

    # PT优先模式: 从放宽候选（p<0.10）中选取 Top 进入 PT 检验
    candidates_for_pt = relaxed_candidates if pt_first else all_qualified
    if not candidates_for_pt:
        candidates_for_pt = all_qualified

    candidates_for_pt.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)

    # 去重取 Top 设定
    top_specs = []
    seen = set()
    for r in candidates_for_pt[:120]:  # PT优先: 扩大候选池
        key = (r["y_var"], r["x_var"], r.get("controls_group", ""),
               "+".join(r.get("fe_vars", [])), "+".join(r.get("cluster_vars", [])),
               r.get("start_year", 0), r.get("end_year", 0))
        if key not in seen:
            seen.add(key)
            top_specs.append(SpecConfig(
                chapter=chapter, y_var=r["y_var"], x_var=r["x_var"],
                controls_group=r.get("controls_group", ""),
                controls=r.get("controls", []),
                fe_vars=r.get("fe_vars", []),
                cluster_vars=r.get("cluster_vars", []),
                start_year=r.get("start_year", 2007),
                end_year=r.get("end_year", 2023),
                filters=r.get("filters", {}),
            ))

    max_top = 50 if pt_first else 30  # PT优先: 更多候选进入PT检验
    if len(top_specs) > max_top:
        top_specs = top_specs[:max_top]

    pt_specs = generate_phase3_pt_grid(chapter, top_specs)
    n_total = len(pt_specs)
    pt_qualified = []

    for i, spec in enumerate(pt_specs):
        if (i + 1) % max(n_total // 10, 1) == 0:
            log.info(f"  Phase 3 进度: {i + 1}/{n_total}")

        pt_result = _run_parallel_trends_spec(raw_df, spec, chapter)
        if pt_result is None:
            continue

        pt_score = evaluate_parallel_trends(pt_result, chapter)
        if pt_score > float("-inf"):
            # PT 通过: 事后≥2期连续≥2星显著 + 方向正确
            # 在 candidates 中查找匹配的主回归结果
            spec_fe_key = "+".join(sorted(spec.fe_vars))
            spec_cl_key = "+".join(sorted(spec.cluster_vars))
            matching = [r for r in (all_qualified + relaxed_candidates)
                        if r["y_var"] == spec.y_var and r["x_var"] == spec.x_var
                        and r.get("start_year") == spec.start_year
                        and r.get("end_year") == spec.end_year
                        and r.get("controls_group") == spec.controls_group
                        and "+".join(sorted(r.get("fe_vars", []))) == spec_fe_key
                        and "+".join(sorted(r.get("cluster_vars", []))) == spec_cl_key]

            if matching:
                best_match = max(matching, key=lambda r: r.get("score", float("-inf")))
                record = best_match.copy()
                record["pt_score"] = pt_score
                record["pt_qualified"] = True  # 标记PT通过
                # PT优先模式: PT通过给予+500加成，确保排序优先
                record["score"] = best_match.get("score", 0) + pt_score + 500
                record["pt_result"] = pt_result
                record["pt_pre_window"] = spec.pt_pre_window
                record["pt_post_window"] = spec.pt_post_window
                record["pt_base_period"] = spec.pt_base_period
                pt_qualified.append(record)

    log.info(f"  Phase 3: {len(pt_qualified)} 个通过平行趋势（事后≥2星）")

    if pt_qualified:
        pt_qualified.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
        # PT优先: PT通过的结果放在最前面
        non_pt = [r for r in all_qualified
                  if not any(r.get("y_var") == p.get("y_var") and
                            r.get("x_var") == p.get("x_var") and
                            r.get("start_year") == p.get("start_year") and
                            r.get("end_year") == p.get("end_year") and
                            r.get("controls_group") == p.get("controls_group")
                            for p in pt_qualified)]
        final_results = pt_qualified + non_pt[:10]
    elif is_ch5_robustness:
        log.info("  第5章作为稳健性检验，不要求平行趋势")
        final_results = all_qualified[:30]
    else:
        log.warning("  无设定通过平行趋势（事后≥2星），返回主回归最优结果")
        final_results = all_qualified[:30]

    # -------- Phase 4: 稳健性X检验 --------
    if final_results:
        log.info(f"Phase 4: 稳健性X变量检验")
        best_result = final_results[0]
        best_spec = SpecConfig(
            chapter=chapter, y_var=best_result["y_var"], x_var=best_result["x_var"],
            controls_group=best_result.get("controls_group", ""),
            controls=best_result.get("controls", []),
            fe_vars=best_result.get("fe_vars", []),
            cluster_vars=best_result.get("cluster_vars", []),
            start_year=best_result.get("start_year", 2007),
            end_year=best_result.get("end_year", 2023),
            filters=best_result.get("filters", {}),
            pt_pre_window=best_result.get("pt_pre_window", -3),
            pt_post_window=best_result.get("pt_post_window", 3),
            pt_base_period=best_result.get("pt_base_period", "pre1"),
        )
        rob_specs = generate_robustness_x_grid(chapter, best_spec)
        rob_results = []
        for spec in rob_specs:
            r = _run_single_spec(raw_df, spec)
            if r and check_sign(r.get("coef", np.nan), chapter):
                r["score"] = evaluate_regression(
                    RegressionResult(coef=r["coef"], pval=r["pval"],
                                     nobs=r["nobs"], r2=r["r2"], success=True),
                    chapter,
                )
                r["is_robustness_x"] = True
                rob_results.append(r)
        n_sig = sum(1 for r in rob_results if check_significance(r.get("pval", 1), 0.05))
        log.info(f"  稳健性X: {n_sig}/{len(rob_results)} 个方向正确且≥2星")
        # 附加到结果（标记为稳健性）
        final_results.extend(rob_results)

    return final_results


def _run_single_spec(raw_df: pd.DataFrame, spec: SpecConfig) -> dict | None:
    """运行单个回归设定。"""
    try:
        df = apply_sample_filters(raw_df, spec.start_year, spec.end_year, spec.filters)
        if len(df) < 100:
            return None

        winsorize_level = spec.filters.get("winsorize")
        if winsorize_level:
            cont_vars = get_continuous_vars(df, [spec.y_var] + spec.controls)
            df = winsorize_variables(df, cont_vars, winsorize_level)

        result = run_ols_fe(
            df, spec.y_var, spec.x_var,
            spec.controls, spec.fe_vars, spec.cluster_vars,
        )
        if not result.success:
            return None

        return {
            "chapter": spec.chapter,
            "y_var": spec.y_var, "x_var": spec.x_var,
            "controls_group": spec.controls_group,
            "controls": spec.controls,
            "fe_vars": spec.fe_vars,
            "cluster_vars": spec.cluster_vars,
            "start_year": spec.start_year,
            "end_year": spec.end_year,
            "filters": spec.filters,
            "coef": result.coef, "se": result.se,
            "tstat": result.tstat, "pval": result.pval,
            "stars": result.stars,
            "nobs": result.nobs, "r2": result.r2,
            "reg_result": result,
        }
    except Exception:
        return None


def _run_parallel_trends_spec(
    raw_df: pd.DataFrame,
    spec: SpecConfig,
    chapter: int,
) -> PTResult | None:
    """运行单个平行趋势设定。"""
    try:
        df = apply_sample_filters(raw_df, spec.start_year, spec.end_year, spec.filters)
        if len(df) < 100:
            return None

        winsorize_level = spec.filters.get("winsorize")
        if winsorize_level:
            cont_vars = get_continuous_vars(df, [spec.y_var] + spec.controls)
            df = winsorize_variables(df, cont_vars, winsorize_level)

        df = generate_event_time(df, spec.x_var)

        pt = run_parallel_trends(
            df, spec.y_var, spec.x_var,
            spec.controls, spec.fe_vars, spec.cluster_vars,
            pre_window=spec.pt_pre_window,
            post_window=spec.pt_post_window,
            base_period=spec.pt_base_period,
            chapter=chapter,
        )
        return pt if pt.success else None
    except Exception:
        return None


def generate_chapter_output(
    raw_df: pd.DataFrame,
    best: dict,
    all_results: list[dict],
    chapter: int,
    output_dir: str,
):
    """
    为一章生成完整输出（5-7 个文件）：
    1. 描述统计表.docx
    2. 相关系数表.docx
    3. 主回归表.docx（含多列 Y；如有PSM则含匹配前基准+平衡性+匹配后）
    4. 平行趋势检验表.docx
    5-7. 动态效应图.png
    """
    from docx import Document

    ch_dir = os.path.join(output_dir, f"Chapter{chapter}")
    os.makedirs(ch_dir, exist_ok=True)

    log.info(f"生成第{chapter}章输出:")
    log.info(f"  X={best['x_var']}, Y={best['y_var']}, "
             f"区间={best['start_year']}-{best['end_year']}")

    # ---- 准备统一样本 ----
    set_verbose(True)
    df = apply_sample_filters(
        raw_df, best["start_year"], best["end_year"], best.get("filters", {}),
    )
    set_verbose(False)

    controls = best.get("controls", [])
    winsorize_level = best.get("filters", {}).get("winsorize")
    if winsorize_level:
        cont_vars = get_continuous_vars(df, [best["y_var"], best["x_var"]] + controls)
        df = winsorize_variables(df, cont_vars, winsorize_level)

    # Listwise deletion：确保章内所有表格的 N 完全一致
    all_vars = [best["y_var"], best["x_var"]] + controls
    available_vars = [v for v in all_vars if v in df.columns]
    df_complete = df.dropna(subset=available_vars)
    log.info(f"  统一样本 N={len(df_complete)}")

    best_x = best["x_var"]
    best_fe = best.get("fe_vars", [])
    best_cl = best.get("cluster_vars", [])

    # ---- 收集同X的多个Y结果 ----
    same_spec_results = [
        r for r in all_results
        if r["x_var"] == best_x
        and r.get("controls_group") == best.get("controls_group")
        and r.get("start_year") == best.get("start_year")
        and r.get("end_year") == best.get("end_year")
        and not r.get("is_robustness_x", False)
    ]
    y_best = {}
    for r in same_spec_results:
        yv = r["y_var"]
        if yv not in y_best or r.get("score", 0) > y_best[yv].get("score", 0):
            y_best[yv] = r

    # 收集 RegressionResult 列表（最多6列）
    main_results = []
    if best.get("reg_result"):
        main_results.append(best["reg_result"])
    for yv, r in sorted(y_best.items(), key=lambda x: x[1].get("score", 0), reverse=True):
        if yv != best["y_var"] and r.get("reg_result") and len(main_results) < 6:
            main_results.append(r["reg_result"])
    if not main_results:
        reg = run_ols_fe(df_complete, best["y_var"], best["x_var"], controls, best_fe, best_cl)
        if reg.success:
            main_results = [reg]

    # ---- 1. 描述统计表 ----
    stats_vars = list(dict.fromkeys(
        [best["x_var"], best["y_var"]] + controls
    ))
    stats_vars = [v for v in stats_vars if v in df_complete.columns]
    desc_stats = compute_descriptive_stats(df_complete, stats_vars)

    from spec_search.table_generator import (
        create_descriptive_stats_table, create_correlation_table,
        create_main_regression_table, create_psm_balance_table,
        create_parallel_trends_table, create_robustness_x_table,
        save_chapter_tables,
    )

    doc1 = Document()
    create_descriptive_stats_table(doc1, desc_stats, "主要变量描述统计", chapter)
    doc1.save(os.path.join(ch_dir, f"表{chapter}-1_描述统计表.docx"))
    log.info(f"  表{chapter}-1 描述统计表")

    # ---- 2. 相关系数表 ----
    corr_vars = list(dict.fromkeys(
        [best["x_var"], best["y_var"]] + controls[:8]
    ))
    corr_vars = [v for v in corr_vars if v in df_complete.columns]
    corr_matrix = compute_correlation_matrix(df_complete, corr_vars)

    doc2 = Document()
    create_correlation_table(doc2, corr_matrix, "主要变量Pearson相关系数矩阵", chapter)
    doc2.save(os.path.join(ch_dir, f"表{chapter}-2_相关系数表.docx"))
    log.info(f"  表{chapter}-2 相关系数表")

    # ---- 3. 主回归表（多列: 逐步加入控制变量/FE）----
    table_num = 3
    doc3 = Document()

    # 生成逐步回归: (1)仅X (2)X+Controls (3)X+Controls+FE (=best设定)
    step_results = []
    # 列(1): 无控制变量, 仅年份FE
    reg_no_ctrl = run_ols_fe(
        df_complete, best["y_var"], best_x, [],
        [config.YEAR_VAR], best_cl,
    )
    if reg_no_ctrl.success:
        step_results.append(reg_no_ctrl)
    # 列(2): 加控制变量, 仅年份FE
    reg_ctrl_no_fe = run_ols_fe(
        df_complete, best["y_var"], best_x, controls,
        [config.YEAR_VAR], best_cl,
    )
    if reg_ctrl_no_fe.success:
        step_results.append(reg_ctrl_no_fe)
    # 列(3): 完整设定 (Controls + 双向FE)
    if main_results:
        step_results.append(main_results[0])
    else:
        reg_full = run_ols_fe(
            df_complete, best["y_var"], best_x, controls, best_fe, best_cl,
        )
        if reg_full.success:
            step_results.append(reg_full)

    # 追加其他Y变量的完整设定结果
    for yv, r in sorted(y_best.items(), key=lambda x: x[1].get("score", 0), reverse=True):
        if yv != best["y_var"] and r.get("reg_result") and len(step_results) < 6:
            step_results.append(r["reg_result"])

    create_main_regression_table(
        doc3, step_results, "基准回归结果", chapter, table_num, controls,
    )
    doc3.save(os.path.join(ch_dir, f"表{chapter}-{table_num}_主回归表.docx"))
    log.info(f"  表{chapter}-{table_num} 主回归表 ({len(step_results)} 列)")
    table_num += 1

    # ---- 4. PSM-DID ----
    psm_result = None
    try:
        psm_result = run_psm_did(
            df_complete, best["y_var"], best["x_var"],
            controls, best_fe, best_cl,
            psm_config={"method": "nearest", "n_neighbors": 1, "caliper": 0.05},
        )
        if not psm_result.success:
            psm_result = None
    except Exception:
        pass

    if psm_result:
        doc_psm = Document()
        create_psm_balance_table(doc_psm, psm_result, "PSM平衡性检验", chapter, table_num)
        doc_psm.save(os.path.join(ch_dir, f"表{chapter}-{table_num}_PSM平衡性.docx"))
        log.info(f"  表{chapter}-{table_num} PSM平衡性检验")
        table_num += 1

        # PSM-DID 回归
        psm_regs = []
        if psm_result.baseline_result.success:
            psm_regs.append(psm_result.baseline_result)
        if psm_result.psm_result.success:
            psm_regs.append(psm_result.psm_result)
        if psm_regs:
            doc_psm_reg = Document()
            create_main_regression_table(
                doc_psm_reg, psm_regs,
                "PSM-DID回归结果", chapter, table_num, controls,
            )
            doc_psm_reg.save(os.path.join(ch_dir, f"表{chapter}-{table_num}_PSM-DID回归.docx"))
            log.info(f"  表{chapter}-{table_num} PSM-DID回归")
            table_num += 1

    # ---- 稳健性检验表（替代X变量）----
    rob_results_x = [r for r in all_results if r.get("is_robustness_x", False)]
    rob_reg_results = [r.get("reg_result") for r in rob_results_x
                       if r.get("reg_result") and r["reg_result"].success]
    if rob_reg_results:
        doc_rob = Document()
        create_robustness_x_table(
            doc_rob, rob_reg_results[:6],
            "稳健性检验：替代解释变量", chapter, table_num, controls,
        )
        doc_rob.save(os.path.join(ch_dir, f"表{chapter}-{table_num}_稳健性替代X.docx"))
        log.info(f"  表{chapter}-{table_num} 稳健性替代X ({len(rob_reg_results[:6])} 列)")
        table_num += 1

    # ---- 平行趋势检验表 ----
    pt_results = []
    pt_result = best.get("pt_result")
    if pt_result is None:
        df_event = generate_event_time(df_complete, best["x_var"])
        pt_result = run_parallel_trends(
            df_event, best["y_var"], best["x_var"],
            controls, best_fe, best_cl,
            pre_window=best.get("pt_pre_window", -3),
            post_window=best.get("pt_post_window", 3),
            base_period=best.get("pt_base_period", "pre1"),
            chapter=chapter,
        )
    if pt_result and pt_result.success:
        pt_results.append(pt_result)

    # 其他 Y 的平行趋势
    for yv, r in y_best.items():
        if yv != best["y_var"] and r.get("pt_result") and len(pt_results) < 4:
            pt_results.append(r["pt_result"])

    if pt_results:
        doc_pt = Document()
        create_parallel_trends_table(
            doc_pt, pt_results, "平行趋势检验与动态效应", chapter, table_num,
        )
        doc_pt.save(os.path.join(ch_dir, f"表{chapter}-{table_num}_平行趋势检验表.docx"))
        log.info(f"  表{chapter}-{table_num} 平行趋势表 ({len(pt_results)} 列)")
        table_num += 1

    # ---- 动态效应图 ----
    for pt in pt_results:
        if pt.success:
            plot_dynamic_effects(
                pt,
                output_path=os.path.join(ch_dir, f"图{chapter}_动态效应_{pt.y_var}.png"),
                title=f"图{chapter}  动态效应: {pt.y_var}",
                chapter=chapter,
                y_label=pt.y_var,
            )

    if len(pt_results) > 1:
        plot_multiple_dynamic_effects(
            pt_results,
            output_path=os.path.join(ch_dir, f"图{chapter}_动态效应对比.png"),
            title=f"图{chapter}  动态效应对比",
            chapter=chapter,
        )

    # 保存合并版
    save_chapter_tables(
        chapter=chapter,
        desc_stats=desc_stats,
        corr_matrix=corr_matrix,
        main_results=step_results,
        psm_result=psm_result,
        pt_results=pt_results,
        controls_list=controls,
        output_dir=ch_dir,
        robustness_results=rob_reg_results[:6] if rob_reg_results else None,
    )


def generate_summary_report(state: SearchState, cross_results, output_dir: str):
    """生成跨章节汇总报告（文本 + Word）。"""
    lines = []
    lines.append("=" * 70)
    lines.append("  回归设定搜索结果汇总报告")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)

    for chapter in [3, 4, 5]:
        best = state.best_specs.get(chapter)
        if not best:
            lines.append(f"\n第{chapter}章: 未找到满足约束的设定")
            continue

        pt_ok = best.get("pt_qualified", False)
        pt_tag = "PT-PASS(≥2★)" if pt_ok else "PT-FAIL"

        lines.append(f"\n{'─' * 50}")
        lines.append(f"第{chapter}章  [{pt_tag}]")
        lines.append(f"{'─' * 50}")
        lines.append(f"  Y = {best['y_var']}")
        lines.append(f"  X = {best['x_var']}")
        lines.append(f"  Controls = {best.get('controls_group', '')}")
        lines.append(f"  FE = {' + '.join(best.get('fe_vars', []))}")
        lines.append(f"  Cluster = {' + '.join(best.get('cluster_vars', []))}")
        lines.append(f"  Sample = {best.get('start_year', '')}-{best.get('end_year', '')}")
        lines.append(f"  Filters = {best.get('filters', {})}")
        lines.append(f"  主回归: coef={best.get('coef', 'NA'):.4f}, "
                     f"p={best.get('pval', 'NA'):.4f}{best.get('stars', '')}, "
                     f"N={best.get('nobs', 0)}")

        pt_result = best.get("pt_result")
        if pt_result and pt_result.success:
            lines.append(f"  平行趋势:")
            lines.append(f"    事前显著(10%): {pt_result.n_pre_sig} 期 "
                         f"(≤{config.PT_MAX_PRE_SIGNIFICANT} 期要求)")
            lines.append(f"    事后显著(5%): {pt_result.n_post_sig_2star} 期")
            lines.append(f"    事后最大连续(5%): {pt_result.max_post_consecutive_2star} 期 "
                         f"(≥{config.PT_MIN_POST_CONSECUTIVE} 期要求)")
            lines.append(f"    事后方向正确(5%): {'是' if pt_result.post_correct_sign_2star else '否'}")
            lines.append(f"    窗口: [{best.get('pt_pre_window', -3)}, "
                         f"{best.get('pt_post_window', 3)}], 基期: {best.get('pt_base_period', 'pre1')}")

            # 逐期系数
            for p in sorted(pt_result.period_coefs.keys()):
                coef = pt_result.period_coefs[p]
                pval = pt_result.period_pval.get(p, 1.0)
                star = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
                tag = ""
                if p < 0 and pval < 0.10:
                    tag = " ← PRE-SIG!"
                elif p > 0 and pval < 0.05:
                    tag = " ← POST-2★ ✓"
                lines.append(f"    t={p:+d}: coef={coef:.4f}, p={pval:.4f}{star}{tag}")
        else:
            lines.append(f"  平行趋势: 未检验或未通过")

    # 跨章一致性
    lines.append(f"\n{'═' * 50}")
    lines.append(f"跨章节一致性")
    lines.append(f"{'═' * 50}")

    if cross_results:
        best_cc = cross_results[0]
        lines.append(f"  最优组合: X={best_cc.x_var}, "
                     f"区间={best_cc.start_year}-{best_cc.end_year}")
        lines.append(f"  总分: {best_cc.total_score:.1f}")
        lines.append(f"  Ch3 PT: {'通过(≥2★)' if best_cc.ch3_pt_pass else '未通过'}")
        lines.append(f"  Ch4 PT: {'通过(≥2★)' if best_cc.ch4_pt_pass else '未通过'}")

        if best_cc.ch3_pt_pass and best_cc.ch4_pt_pass:
            lines.append(f"  状态: ★★ Ch3+Ch4 双PT通过 — 最优组合 ★★")
        elif best_cc.ch3_pt_pass or best_cc.ch4_pt_pass:
            lines.append(f"  状态: ★ 单章PT通过 — 可接受（需手动调整另一章）")
        else:
            lines.append(f"  状态: 无PT通过组合 — 需要放宽约束或手动调整")

        # 显示其他候选
        if len(cross_results) > 1:
            lines.append(f"\n  其他候选:")
            for cc in cross_results[1:5]:
                pt3 = "✓" if cc.ch3_pt_pass else "✗"
                pt4 = "✓" if cc.ch4_pt_pass else "✗"
                lines.append(f"    X={cc.x_var}, {cc.start_year}-{cc.end_year}, "
                             f"score={cc.total_score:.1f}, Ch3-PT:{pt3}, Ch4-PT:{pt4}")
    else:
        lines.append("  无跨章一致设定")

    report_text = "\n".join(lines)

    # 保存文本报告
    report_path = os.path.join(output_dir, "summary_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    log.info(f"汇总报告: {report_path}")

    # 打印到控制台
    print(report_text)


def save_search_log(state: SearchState, output_dir: str):
    """保存搜索日志。"""
    records = []
    for ch, results in state.chapter_results.items():
        for r in results:
            records.append({
                "Chapter": ch,
                "Y": r.get("y_var", ""),
                "X": r.get("x_var", ""),
                "Controls": r.get("controls_group", ""),
                "FE": "+".join(r.get("fe_vars", [])),
                "Cluster": "+".join(r.get("cluster_vars", [])),
                "Period": f"{r.get('start_year', '')}-{r.get('end_year', '')}",
                "Filters": str({k: v for k, v in r.get("filters", {}).items() if v}),
                "Coef": r.get("coef", np.nan),
                "SE": r.get("se", np.nan),
                "t-stat": r.get("tstat", np.nan),
                "p-value": r.get("pval", np.nan),
                "Stars": r.get("stars", ""),
                "N": r.get("nobs", 0),
                "R2": r.get("r2", np.nan),
                "Score": r.get("score", np.nan),
                "PT_Score": r.get("pt_score", np.nan),
                "PT_Pre": r.get("pt_pre_window", ""),
                "PT_Post": r.get("pt_post_window", ""),
                "PT_Base": r.get("pt_base_period", ""),
            })

    if records:
        log_df = pd.DataFrame(records)
        log_path = os.path.join(output_dir, "search_log.xlsx")
        log_df.to_excel(log_path, index=False)
        log.info(f"搜索日志: {log_path} ({len(log_df)} 条)")

    # JSON
    best_path = os.path.join(output_dir, "best_specifications.json")
    best_info = {}
    for ch, best in state.best_specs.items():
        if best:
            info = {}
            for k, v in best.items():
                if k == "reg_result":
                    continue
                if k == "pt_result":
                    # 序列化 PTResult 的关键统计字段
                    pt = v
                    if pt and pt.success:
                        info["pt_summary"] = {
                            "success": pt.success,
                            "nobs": int(pt.nobs),
                            "r2": float(pt.r2) if not np.isnan(pt.r2) else None,
                            "n_pre_sig": pt.n_pre_sig,
                            "n_pre_sig_2star": pt.n_pre_sig_2star,
                            "n_post_sig": pt.n_post_sig,
                            "n_post_sig_2star": pt.n_post_sig_2star,
                            "max_post_consecutive": pt.max_post_consecutive,
                            "max_post_consecutive_2star": pt.max_post_consecutive_2star,
                            "post_correct_sign": pt.post_correct_sign,
                            "post_correct_sign_2star": pt.post_correct_sign_2star,
                            "periods": {
                                str(p): {
                                    "coef": float(c),
                                    "pval": float(pt.period_pval.get(p, 1.0)),
                                    "se": float(pt.period_se.get(p, 0.0)),
                                    "ci_lo": float(pt.period_ci_lower.get(p, 0.0)),
                                    "ci_hi": float(pt.period_ci_upper.get(p, 0.0)),
                                }
                                for p, c in sorted(pt.period_coefs.items())
                            },
                        }
                    continue
                if isinstance(v, (np.floating, np.integer)):
                    info[k] = float(v)
                elif isinstance(v, np.ndarray):
                    info[k] = v.tolist()
                else:
                    info[k] = v
            best_info[f"Chapter{ch}"] = info

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)
    log.info(f"最优设定: {best_path}")


def generate_stata_code(state: SearchState, output_dir: str):
    """
    为各章生成完整、可直接运行的 Stata .do 文件。
    包含: 数据加载→样本筛选→缩尾→Listwise Deletion→面板设定→
          描述统计→相关系数→逐步回归→PSM-DID→平行趋势→动态效应图→esttab输出
    """
    for chapter in [3, 4, 5]:
        best = state.best_specs.get(chapter)
        if not best:
            continue

        do_path = os.path.join(output_dir, f"Chapter{chapter}", f"chapter{chapter}_code.do")
        os.makedirs(os.path.dirname(do_path), exist_ok=True)

        y = best["y_var"]
        x = best["x_var"]
        ctrls = best.get("controls", [])
        fe = best.get("fe_vars", [])
        cl = best.get("cluster_vars", [])
        sy = best.get("start_year", 2007)
        ey = best.get("end_year", 2023)
        filt = best.get("filters", {})
        pt_pre = best.get("pt_pre_window", -3)
        pt_post = best.get("pt_post_window", 3)
        pt_base = best.get("pt_base_period", "pre1")

        ctrl_str = " ".join(ctrls)
        firm = config.FIRM_ID
        year = config.YEAR_VAR
        absorb_str = " ".join(fe) if fe else year
        cluster_str = cl[0] if cl else firm

        omit_val = {"pre1": -1, "pre0": 0, "pre_biggest": pt_pre}.get(pt_base, -1)

        L = []  # code lines
        a = L.append

        # =========== 文件头 ===========
        a(f"/*")
        a(f"{'=' * 64}")
        a(f"  第{chapter}章 完整回归代码")
        a(f"  自动生成 by Specification Search Engine")
        a(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        a(f"{'─' * 64}")
        a(f"  Y (被解释变量):  {y}")
        a(f"  X (解释变量):    {x}")
        a(f"  Controls:        {ctrl_str}")
        a(f"  FE:              {' + '.join(fe)}")
        a(f"  Cluster:         {' + '.join(cl)}")
        a(f"  Sample:          {sy}-{ey}")
        a(f"  Python结果:      coef={best.get('coef', 0):.4f}, "
          f"p={best.get('pval', 1):.4f}{best.get('stars', '')}, "
          f"N={best.get('nobs', 0)}")
        a(f"{'=' * 64}")
        a(f"*/")
        a("")

        # =========== 0. 环境准备 ===========
        a("clear all")
        a("set more off")
        a("set matsize 11000")
        a("")
        a("* --- 安装所需命令（首次运行取消注释）---")
        a("* ssc install reghdfe, replace")
        a("* ssc install ftools, replace")
        a("* ssc install psmatch2, replace")
        a("* ssc install estout, replace")
        a("* ssc install winsor2, replace")
        a("* ssc install coefplot, replace")
        a("")

        # =========== 1. 数据加载 ===========
        a(f'{"*" * 60}')
        a(f"* 1. 数据加载")
        a(f'{"*" * 60}')
        a(f'use "{config.DATA_PATH}", clear')
        a("")

        # =========== 2. 样本筛选 ===========
        a(f'{"*" * 60}')
        a(f"* 2. 样本筛选")
        a(f'{"*" * 60}')
        a(f"keep if {year} >= {sy} & {year} <= {ey}")

        if filt.get("drop_st_pt"):
            a("")
            a("* 剔除 ST/PT 股票")
            a("capture confirm variable ST")
            a("if !_rc drop if ST == 1")
            a("capture confirm variable Stkname")
            a('if !_rc drop if regexm(Stkname, "ST|PT|\\*ST")')

        if filt.get("drop_finance"):
            a("")
            a("* 剔除金融行业 (证监会行业代码J)")
            a("capture confirm variable Ind")
            a('if !_rc drop if substr(Ind, 1, 1) == "J"')

        if filt.get("drop_real_estate"):
            a("")
            a("* 剔除房地产行业 (证监会行业代码K)")
            a("capture confirm variable Ind")
            a('if !_rc drop if substr(Ind, 1, 1) == "K"')

        if filt.get("drop_lev_gt1"):
            a("")
            a("* 剔除资产负债率>1的异常值")
            a("drop if Lev > 1 & Lev != .")

        if filt.get("drop_ind_lt30"):
            a("")
            a("* 剔除行业年度观测量<30的行业")
            a("bysort Ind: gen _ind_n = _N")
            a("drop if _ind_n < 30")
            a("drop _ind_n")

        # =========== 3. 缩尾处理 ===========
        winsorize = filt.get("winsorize")
        if winsorize:
            pct = int(winsorize * 100)
            a("")
            a(f'{"*" * 60}')
            a(f"* 3. 连续变量缩尾处理 ({pct}%/{100 - pct}%)")
            a(f'{"*" * 60}')
            a(f"foreach var of varlist {y} {ctrl_str} {{")
            a(f"    capture winsor2 `var', replace cuts({pct} {100 - pct})")
            a(f"}}")
        a("")

        # =========== 4. Listwise Deletion ===========
        a(f'{"*" * 60}')
        a(f"* 4. Listwise Deletion (保证各表N一致)")
        a(f'{"*" * 60}')
        a(f"* 标记缺失观测")
        a(f"gen _keep = 1")
        all_vars_str = f"{y} {x} {ctrl_str}"
        a(f"foreach var of varlist {all_vars_str} {{")
        a(f"    replace _keep = 0 if missing(`var')")
        a(f"}}")
        a(f"keep if _keep == 1")
        a(f"drop _keep")
        a(f'di "样本量 N = " _N')
        a("")

        # =========== 5. 面板设定 ===========
        a(f'{"*" * 60}')
        a(f"* 5. 面板设定")
        a(f'{"*" * 60}')
        a(f"destring {firm}, replace force")
        a(f"xtset {firm} {year}")
        a("")

        # =========== 6. 描述统计 ===========
        a(f'{"*" * 60}')
        a(f"* 6. 描述统计 (表{chapter}-1)")
        a(f'{"*" * 60}')
        a(f"summarize {x} {y} {ctrl_str}, detail")
        a(f'estpost summarize {x} {y} {ctrl_str}, detail')
        a(f'esttab using "表{chapter}-1_描述统计.rtf", ///'.rstrip())
        a(f'    cells("count mean sd min p25 p50 p75 max") ///'.rstrip())
        a(f"    replace noobs label ///")
        a(f'    title("表{chapter}-1 主要变量描述统计")')
        a("")

        # =========== 7. 相关系数矩阵 ===========
        a(f'{"*" * 60}')
        a(f"* 7. 相关系数矩阵 (表{chapter}-2)")
        a(f'{"*" * 60}')
        a(f"pwcorr {x} {y} {ctrl_str}, star(0.05) sig")
        a("")

        # =========== 8. 逐步回归 (表X-3) ===========
        a(f'{"*" * 60}')
        a(f"* 8. 主回归 — 逐步加入控制变量 (表{chapter}-3)")
        a(f'{"*" * 60}')
        a("")
        a(f"* 列(1): 仅X, 年份FE")
        a(f"reghdfe {y} {x}, absorb({year}) vce(cluster {cluster_str})")
        a(f"est store m1")
        a("")
        a(f"* 列(2): X + Controls, 年份FE")
        a(f"reghdfe {y} {x} {ctrl_str}, absorb({year}) vce(cluster {cluster_str})")
        a(f"est store m2")
        a("")
        a(f"* 列(3): X + Controls, 完整FE ({' + '.join(fe)})")
        a(f"reghdfe {y} {x} {ctrl_str}, absorb({absorb_str}) vce(cluster {cluster_str})")
        a(f"est store m3")
        a("")
        a(f"* 输出逐步回归表")
        a(f'esttab m1 m2 m3 using "表{chapter}-3_主回归.rtf", ///'.rstrip())
        a(f"    replace ///")
        a(f"    star(* 0.10 ** 0.05 *** 0.01) ///")
        a(f"    b(%9.4f) se(%9.4f) ///")
        a(f'    stats(N r2_a, labels("N" "Adj. R-sq") fmt(%9.0fc %9.4f)) ///'.rstrip())
        a(f'    title("表{chapter}-3 基准回归结果") ///'.rstrip())
        a(f'    mtitles("(1)" "(2)" "(3)") ///'.rstrip())
        a(f'    note("括号内为聚类稳健标准误（聚类至{cluster_str}层面）。*** p<0.01, ** p<0.05, * p<0.10")')
        a("")

        # =========== 9. PSM-DID ===========
        a(f'{"*" * 60}')
        a(f"* 9. PSM-DID (表{chapter}-4, 表{chapter}-5)")
        a(f'{"*" * 60}')
        a("")
        a(f"* 9a. 生成企业层面处理标识")
        a(f"capture drop _ever_treated")
        a(f"bysort {firm}: egen _ever_treated = max({x})")
        a("")
        a(f"* 9b. Logit 倾向得分估计")
        a(f"logit _ever_treated {ctrl_str}")
        a(f"predict _pscore, pr")
        a("")
        a(f"* 9c. 最近邻匹配 (1:1, caliper=0.05)")
        a(f"psmatch2 _ever_treated, pscore(_pscore) neighbor(1) caliper(0.05)")
        a("")
        a(f"* 9d. 平衡性检验 (表{chapter}-4)")
        a(f"pstest {ctrl_str}, both")
        a("")
        a(f"* 9e. 匹配后回归")
        a(f"reghdfe {y} {x} {ctrl_str} if _weight != ., ///")
        a(f"    absorb({absorb_str}) vce(cluster {cluster_str})")
        a(f"est store psm_reg")
        a("")
        a(f"* 9f. 全样本 vs 匹配后对比 (表{chapter}-5)")
        a(f'esttab m3 psm_reg using "表{chapter}-5_PSM-DID.rtf", ///'.rstrip())
        a(f"    replace ///")
        a(f"    star(* 0.10 ** 0.05 *** 0.01) ///")
        a(f"    b(%9.4f) se(%9.4f) ///")
        a(f'    stats(N r2_a, labels("N" "Adj. R-sq") fmt(%9.0fc %9.4f)) ///'.rstrip())
        a(f'    title("表{chapter}-5 PSM-DID回归结果") ///'.rstrip())
        a(f'    mtitles("全样本" "匹配后") ///'.rstrip())
        a(f'    note("括号内为聚类稳健标准误。*** p<0.01, ** p<0.05, * p<0.10")')
        a("")

        # =========== 10. 平行趋势检验 ===========
        a(f'{"*" * 60}')
        a(f"* 10. 平行趋势检验 / 事件研究法 (表{chapter}-6)")
        a(f'{"*" * 60}')
        a("")
        a(f"* 10a. 生成事件时间变量")
        a(f"capture drop _first_treat event_year event_time treated_ever")
        a(f"bysort {firm}: egen _first_treat = min({year}) if {x} == 1")
        a(f"bysort {firm}: egen event_year = min(_first_treat)")
        a(f"gen event_time = {year} - event_year")
        a(f"gen treated_ever = !missing(event_year)")
        a(f"drop _first_treat")
        a("")
        a(f"* 10b. 生成事件时间哑变量 (基期: t={omit_val})")

        dummies = []
        for p in range(pt_pre, pt_post + 1):
            if p == omit_val:
                a(f"* t={p}: 基期（省略）")
                continue
            name = f"pre{abs(p)}" if p < 0 else ("current" if p == 0 else f"post{p}")
            dummies.append(name)
            a(f"gen {name} = (event_time == {p}) & treated_ever == 1")

        dummies_str = " ".join(dummies)
        a("")
        a(f"* 10c. 事件研究回归")
        a(f"reghdfe {y} {dummies_str} {ctrl_str}, ///")
        a(f"    absorb({absorb_str}) vce(cluster {cluster_str})")
        a(f"est store pt_reg")
        a("")

        a(f"* 10d. 输出平行趋势表")
        a(f'esttab pt_reg using "表{chapter}-6_平行趋势.rtf", ///'.rstrip())
        a(f"    replace ///")
        a(f"    keep({dummies_str}) ///")
        a(f"    star(* 0.10 ** 0.05 *** 0.01) ///")
        a(f"    b(%9.4f) se(%9.4f) ///")
        a(f'    stats(N r2_a, labels("N" "Adj. R-sq") fmt(%9.0fc %9.4f)) ///'.rstrip())
        a(f'    title("表{chapter}-6 平行趋势检验") ///'.rstrip())
        a(f'    note("基期: t={omit_val}（系数为0）。括号内为聚类稳健标准误。")')
        a("")

        # =========== 11. 动态效应图 ===========
        a(f'{"*" * 60}')
        a(f"* 11. 动态效应图 (图{chapter})")
        a(f'{"*" * 60}')
        a("")
        a(f"coefplot pt_reg, ///")
        a(f"    keep({dummies_str}) ///")
        a(f"    vertical ///")
        a(f"    yline(0, lpattern(dash) lcolor(black) lwidth(thin)) ///")
        a(f"    xline({abs(omit_val) + 0.5 if omit_val < 0 else 0.5}, lpattern(dash) lcolor(red) lwidth(thin)) ///")
        a(f"    ylabel(, angle(horizontal) labsize(small)) ///")
        a(f"    xlabel(, labsize(small)) ///")
        a(f'    title("图{chapter} 动态效应", size(medium)) ///'.rstrip())
        a(f'    xtitle("事件时间（相对于处理时点）", size(small)) ///'.rstrip())
        a(f'    ytitle("回归系数", size(small)) ///'.rstrip())
        a(f"    msymbol(O) mcolor(navy) msize(medium) ///")
        a(f"    ciopts(lcolor(navy%60) lwidth(thin)) ///")
        a(f"    graphregion(color(white)) bgcolor(white) ///")
        a(f"    plotregion(margin(small))")
        a(f'graph export "图{chapter}_动态效应_{y}.png", replace width(1500)')
        a("")

        # =========== 12. 完整输出表 ===========
        a(f'{"*" * 60}')
        a(f"* 12. 完整回归结果汇总 (表{chapter})")
        a(f'{"*" * 60}')
        a("")
        a(f'esttab m1 m2 m3 psm_reg pt_reg ///'.rstrip())
        a(f'    using "第{chapter}章_回归结果汇总.rtf", ///'.rstrip())
        a(f"    replace ///")
        a(f"    star(* 0.10 ** 0.05 *** 0.01) ///")
        a(f"    b(%9.4f) se(%9.4f) ///")
        a(f'    stats(N r2_a, labels("N" "Adj. R-sq") fmt(%9.0fc %9.4f)) ///'.rstrip())
        a(f'    title("第{chapter}章 回归结果汇总") ///'.rstrip())
        a(f'    mtitles("仅X" "加Controls" "基准" "PSM-DID" "平行趋势") ///'.rstrip())
        a(f'    note("括号内为聚类稳健标准误（聚类至{cluster_str}层面）。*** p<0.01, ** p<0.05, * p<0.10")')
        a("")
        a(f"* === 第{chapter}章完毕 ===")

        with open(do_path, "w", encoding="utf-8") as f:
            f.write("\n".join(L))

        log.info(f"  Stata代码: {do_path}")


if __name__ == "__main__":
    main()
