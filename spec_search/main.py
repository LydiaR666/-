"""
============================================================
回归设定搜索（Specification Search）— 主控程序
============================================================
自动化搜索最优回归设定，满足硬性约束：
  1. 各章 X 一致、章内样本量一致
  2. 主回归至少二星显著 + 方向正确
  3. 平行趋势: 事前 ≤1 期显著, 事后 ≥2 期连续显著
============================================================
用法:
    python -m spec_search.main
    或
    python spec_search/main.py
============================================================
"""

import os
import sys
import json
import time
import warnings
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 确保可以 import spec_search
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_search import config
from spec_search.data_loader import (
    load_data, apply_sample_filters, winsorize_variables,
    generate_event_time, get_continuous_vars, prepare_panel_data,
)
from spec_search.regression_engine import (
    run_ols_fe, run_parallel_trends, run_psm_did,
    compute_descriptive_stats, compute_correlation_matrix,
    RegressionResult, PTResult, PSMResult,
)
from spec_search.specification_grid import (
    SpecConfig, generate_phase1_grid, generate_phase2_grid,
    generate_phase3_pt_grid, generate_psm_grid,
)
from spec_search.evaluator import (
    evaluate_regression, evaluate_parallel_trends,
    evaluate_specification_full, check_cross_chapter_consistency,
    check_sign, check_significance,
)
from spec_search.table_generator import save_chapter_tables
from spec_search.figure_generator import (
    plot_dynamic_effects, plot_multiple_dynamic_effects,
)


# ============================================================
# 全局状态
# ============================================================
class SearchState:
    """搜索状态跟踪"""
    def __init__(self):
        self.raw_df = None
        self.chapter_results = {3: [], 4: [], 5: []}
        self.best_specs = {3: None, 4: None, 5: None}
        self.search_log = []
        self.start_time = time.time()

    def elapsed(self):
        return f"{time.time() - self.start_time:.1f}s"


def main():
    """主入口：执行完整的回归设定搜索流程。"""
    print("=" * 70)
    print("  回归设定搜索（Specification Search）")
    print("  数字并购与企业内部利益相关者价值效应")
    print("=" * 70)
    print(f"  开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据路径: {config.DATA_PATH}")
    print("=" * 70)

    state = SearchState()

    # ============================================================
    # Step 1: 加载数据
    # ============================================================
    print("\n" + "=" * 50)
    print("Step 1: 加载数据")
    print("=" * 50)

    state.raw_df = load_data()

    # 打印变量信息
    cols = state.raw_df.columns.tolist()
    print(f"\n总变量数: {len(cols)}")
    print(f"总样本量: {len(state.raw_df)}")

    # 检查关键变量
    _print_variable_availability(state.raw_df)

    # ============================================================
    # Step 2: 逐章搜索
    # ============================================================
    for chapter in [3, 4, 5]:
        print(f"\n{'=' * 70}")
        print(f"  第{chapter}章 搜索开始 [{state.elapsed()}]")
        print(f"{'=' * 70}")

        is_ch5 = (chapter == 5)
        results = run_chapter_search(state.raw_df, chapter, is_ch5_robustness=is_ch5)
        state.chapter_results[chapter] = results

        if results:
            best = max(results, key=lambda r: r.get("score", float("-inf")))
            state.best_specs[chapter] = best
            print(f"\n  [第{chapter}章最优] X={best['x_var']}, Y={best['y_var']}, "
                  f"coef={best.get('coef', 'N/A')}, p={best.get('pval', 'N/A')}, "
                  f"score={best.get('score', 'N/A'):.1f}")
        else:
            print(f"\n  [第{chapter}章] 未找到满足约束的设定")

    # ============================================================
    # Step 3: 跨章节一致性检验
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  跨章节一致性检验 [{state.elapsed()}]")
    print(f"{'=' * 70}")

    cross_results = check_cross_chapter_consistency(
        state.chapter_results[3],
        state.chapter_results[4],
        state.chapter_results[5] if state.chapter_results[5] else None,
    )

    if cross_results:
        best_cross = cross_results[0]
        print(f"\n  最优跨章节设定:")
        print(f"    X变量: {best_cross.x_var}")
        print(f"    样本区间: {best_cross.start_year}-{best_cross.end_year}")
        print(f"    总分: {best_cross.total_score:.1f}")

        # 更新各章最优为一致的设定
        if best_cross.ch3_best:
            state.best_specs[3] = best_cross.ch3_best
        if best_cross.ch4_best:
            state.best_specs[4] = best_cross.ch4_best
        if best_cross.ch5_best:
            state.best_specs[5] = best_cross.ch5_best
    else:
        print("  未找到跨章节一致设定，使用各章独立最优")

    # ============================================================
    # Step 4: 生成输出
    # ============================================================
    print(f"\n{'=' * 70}")
    print(f"  生成输出文件 [{state.elapsed()}]")
    print(f"{'=' * 70}")

    output_dir = os.path.abspath(config.OUTPUT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    for chapter in [3, 4, 5]:
        if state.best_specs[chapter]:
            generate_chapter_output(
                state.raw_df, state.best_specs[chapter], chapter, output_dir
            )
        else:
            print(f"  第{chapter}章: 无最优设定，跳过输出")

    # 保存搜索日志
    save_search_log(state, output_dir)

    # 生成 Stata 代码
    generate_stata_code(state, output_dir)

    print(f"\n{'=' * 70}")
    print(f"  搜索完成！总耗时: {state.elapsed()}")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 70}")


def _print_variable_availability(df: pd.DataFrame):
    """打印关键变量可用性。"""
    cols = set(df.columns)

    print("\n--- 解释变量可用性 ---")
    for x in config.X_VARS_PRIMARY:
        status = "✓" if x in cols else "✗"
        if x in cols:
            n_nonmiss = df[x].notna().sum()
            n_ones = (df[x] == 1).sum() if df[x].dtype in [float, int, np.float64, np.int64] else "?"
            print(f"  {status} {x}: N={n_nonmiss}, treated={n_ones}")
        else:
            print(f"  {status} {x}: 不存在")

    print("\n--- 被解释变量可用性 ---")
    for ch, yvars in config.Y_VARS.items():
        found = [v for v in yvars if v in cols]
        print(f"  第{ch}章: {len(found)}/{len(yvars)} 可用")
        for v in found[:5]:  # 只显示前5个
            print(f"    ✓ {v}: N={df[v].notna().sum()}")
        if len(found) > 5:
            print(f"    ... 及其他 {len(found) - 5} 个")


def run_chapter_search(
    raw_df: pd.DataFrame,
    chapter: int,
    is_ch5_robustness: bool = False,
) -> list[dict]:
    """
    运行单章的完整搜索流程。

    Phase 1: 快速扫描所有 X-Y 组合
    Phase 2: 深度搜索有希望的组合
    Phase 3: 平行趋势验证
    Phase 4: PSM-DID（如适用）

    Returns
    -------
    list[dict]
        所有满足约束的结果记录
    """
    all_qualified = []

    # -------- Phase 1: 快速扫描 --------
    print(f"\n  --- Phase 1: 快速扫描 ---")
    phase1_specs = generate_phase1_grid(chapter)
    promising_pairs = set()
    phase1_results = []

    for i, spec in enumerate(phase1_specs):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{len(phase1_specs)}")

        result = _run_single_spec(raw_df, spec)
        if result is None:
            continue

        phase1_results.append(result)

        # 检查方向和显著性
        coef = result.get("coef", np.nan)
        pval = result.get("pval", np.nan)

        if check_sign(coef, chapter) and check_significance(pval, 0.10):
            promising_pairs.add((spec.y_var, spec.x_var))
            result["score"] = evaluate_regression(
                RegressionResult(coef=coef, pval=pval, nobs=result.get("nobs", 0),
                                 r2=result.get("r2", np.nan), success=True),
                chapter,
            )

            # 如果已经二星显著，加入候选
            if check_significance(pval, 0.05):
                all_qualified.append(result)

    print(f"    Phase 1 完成: {len(promising_pairs)} 个有希望的 (Y, X) 组合")
    print(f"    已找到 {len(all_qualified)} 个二星以上结果")

    if not promising_pairs:
        print("    [警告] Phase 1 未找到任何方向正确的组合，尝试放宽条件...")
        # 放宽: 只要求方向正确
        for r in phase1_results:
            if check_sign(r.get("coef", np.nan), chapter):
                promising_pairs.add((r["y_var"], r["x_var"]))

    if not promising_pairs:
        print("    [警告] 仍然没有找到，返回空")
        return all_qualified

    # 限制组合数量
    promising_pairs = list(promising_pairs)
    if len(promising_pairs) > 20:
        # 按 Phase 1 得分排序，取 Top 20
        pair_scores = defaultdict(float)
        for r in phase1_results:
            key = (r["y_var"], r["x_var"])
            if key in promising_pairs:
                pair_scores[key] = max(pair_scores[key], r.get("score", float("-inf")))
        promising_pairs.sort(key=lambda p: pair_scores[p], reverse=True)
        promising_pairs = promising_pairs[:20]

    # -------- Phase 2: 深度搜索 --------
    print(f"\n  --- Phase 2: 深度搜索 ({len(promising_pairs)} 组合) ---")
    phase2_specs = generate_phase2_grid(chapter, promising_pairs)

    for i, spec in enumerate(phase2_specs):
        if (i + 1) % 200 == 0:
            print(f"    进度: {i + 1}/{len(phase2_specs)}")

        result = _run_single_spec(raw_df, spec)
        if result is None:
            continue

        coef = result.get("coef", np.nan)
        pval = result.get("pval", np.nan)

        if check_sign(coef, chapter) and check_significance(pval, 0.05):
            result["score"] = evaluate_regression(
                RegressionResult(coef=coef, pval=pval, nobs=result.get("nobs", 0),
                                 r2=result.get("r2", np.nan), success=True),
                chapter,
            )
            all_qualified.append(result)

    print(f"    Phase 2 完成: 共 {len(all_qualified)} 个合格结果")

    if not all_qualified:
        print("    [警告] 未找到任何满足约束的设定")
        return all_qualified

    # -------- Phase 3: 平行趋势 --------
    print(f"\n  --- Phase 3: 平行趋势检验 ---")

    # 取 Top 设定
    all_qualified.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
    top_specs = []
    seen = set()
    for r in all_qualified[:50]:
        key = (r["y_var"], r["x_var"], r.get("controls_group", ""),
               r.get("start_year", 0), r.get("end_year", 0))
        if key not in seen:
            seen.add(key)
            spec = SpecConfig(
                chapter=chapter,
                y_var=r["y_var"],
                x_var=r["x_var"],
                controls_group=r.get("controls_group", ""),
                controls=r.get("controls", []),
                fe_vars=r.get("fe_vars", []),
                cluster_vars=r.get("cluster_vars", []),
                start_year=r.get("start_year", 2007),
                end_year=r.get("end_year", 2023),
                filters=r.get("filters", {}),
            )
            top_specs.append(spec)

    if len(top_specs) > 20:
        top_specs = top_specs[:20]

    pt_specs = generate_phase3_pt_grid(chapter, top_specs)

    pt_qualified = []
    for i, spec in enumerate(pt_specs):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{len(pt_specs)}")

        pt_result = _run_parallel_trends_spec(raw_df, spec, chapter)
        if pt_result is None:
            continue

        pt_score = evaluate_parallel_trends(pt_result, chapter)
        if pt_score > float("-inf"):
            # 找到对应的主回归结果
            matching = [r for r in all_qualified
                        if r["y_var"] == spec.y_var and r["x_var"] == spec.x_var
                        and r.get("start_year") == spec.start_year
                        and r.get("end_year") == spec.end_year]

            if matching:
                best_match = max(matching, key=lambda r: r.get("score", float("-inf")))
                combined_score = best_match.get("score", 0) + pt_score

                record = best_match.copy()
                record["pt_score"] = pt_score
                record["score"] = combined_score
                record["pt_result"] = pt_result
                record["pt_pre_window"] = spec.pt_pre_window
                record["pt_post_window"] = spec.pt_post_window
                record["pt_base_period"] = spec.pt_base_period
                pt_qualified.append(record)

    print(f"    Phase 3 完成: {len(pt_qualified)} 个通过平行趋势检验")

    # 如果有通过平行趋势的结果，优先使用
    if pt_qualified:
        pt_qualified.sort(key=lambda r: r.get("score", float("-inf")), reverse=True)
        return pt_qualified
    else:
        # 如果是第五章且为稳健性，不要求平行趋势
        if is_ch5_robustness:
            print("    [第五章] 作为稳健性检验，不要求平行趋势通过")
            return all_qualified[:20]

        print("    [警告] 无设定通过平行趋势检验")
        print("    返回主回归最优结果（附注：需补充平行趋势）")
        return all_qualified[:20]


def _run_single_spec(raw_df: pd.DataFrame, spec: SpecConfig) -> dict | None:
    """
    运行单个回归设定。

    Returns
    -------
    dict or None
        结果字典
    """
    try:
        # 应用样本筛选
        df = apply_sample_filters(
            raw_df, spec.start_year, spec.end_year, spec.filters,
        )

        if len(df) < 100:
            return None

        # 缩尾
        winsorize_level = spec.filters.get("winsorize")
        if winsorize_level:
            cont_vars = get_continuous_vars(df, [spec.y_var] + spec.controls)
            df = winsorize_variables(df, cont_vars, winsorize_level)

        # 运行回归
        result = run_ols_fe(
            df, spec.y_var, spec.x_var,
            spec.controls, spec.fe_vars, spec.cluster_vars,
        )

        if not result.success:
            return None

        return {
            "chapter": spec.chapter,
            "y_var": spec.y_var,
            "x_var": spec.x_var,
            "controls_group": spec.controls_group,
            "controls": spec.controls,
            "fe_vars": spec.fe_vars,
            "cluster_vars": spec.cluster_vars,
            "start_year": spec.start_year,
            "end_year": spec.end_year,
            "filters": spec.filters,
            "coef": result.coef,
            "se": result.se,
            "tstat": result.tstat,
            "pval": result.pval,
            "stars": result.stars,
            "nobs": result.nobs,
            "r2": result.r2,
            "reg_result": result,
        }

    except Exception as e:
        return None


def _run_parallel_trends_spec(
    raw_df: pd.DataFrame,
    spec: SpecConfig,
    chapter: int,
) -> PTResult | None:
    """运行单个平行趋势设定。"""
    try:
        df = apply_sample_filters(
            raw_df, spec.start_year, spec.end_year, spec.filters,
        )

        if len(df) < 100:
            return None

        winsorize_level = spec.filters.get("winsorize")
        if winsorize_level:
            cont_vars = get_continuous_vars(df, [spec.y_var] + spec.controls)
            df = winsorize_variables(df, cont_vars, winsorize_level)

        # 生成事件时间
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
    chapter: int,
    output_dir: str,
):
    """
    为一章生成全部输出: 表格 + 图形。
    """
    ch_dir = os.path.join(output_dir, f"Chapter{chapter}")
    os.makedirs(ch_dir, exist_ok=True)

    print(f"\n  --- 生成第{chapter}章输出 ---")
    print(f"    X: {best['x_var']}, Y: {best['y_var']}")
    print(f"    区间: {best['start_year']}-{best['end_year']}")
    print(f"    控制变量组: {best.get('controls_group', 'N/A')}")

    # 应用同样的样本筛选
    df = apply_sample_filters(
        raw_df, best["start_year"], best["end_year"], best.get("filters", {}),
    )

    winsorize_level = best.get("filters", {}).get("winsorize")
    controls = best.get("controls", [])
    if winsorize_level:
        cont_vars = get_continuous_vars(df, [best["y_var"], best["x_var"]] + controls)
        df = winsorize_variables(df, cont_vars, winsorize_level)

    # 确保样本量一致: 先做 listwise deletion
    all_vars = [best["y_var"], best["x_var"]] + controls
    available_vars = [v for v in all_vars if v in df.columns]
    df_complete = df.dropna(subset=available_vars)

    print(f"    完整样本量: {len(df_complete)}")

    # 1. 描述统计
    stats_vars = [best["x_var"], best["y_var"]] + controls
    stats_vars = [v for v in stats_vars if v in df_complete.columns]
    desc_stats = compute_descriptive_stats(df_complete, stats_vars)

    # 2. 相关系数
    corr_vars = [best["x_var"], best["y_var"]] + controls[:8]  # 限制变量数
    corr_vars = [v for v in corr_vars if v in df_complete.columns]
    corr_matrix = compute_correlation_matrix(df_complete, corr_vars)

    # 3. 主回归
    main_result = best.get("reg_result")
    if main_result is None:
        main_result = run_ols_fe(
            df_complete, best["y_var"], best["x_var"],
            controls, best.get("fe_vars", []), best.get("cluster_vars", []),
        )

    # 尝试 PSM-DID
    psm_result = None
    try:
        psm_result = run_psm_did(
            df_complete, best["y_var"], best["x_var"],
            controls, best.get("fe_vars", []), best.get("cluster_vars", []),
            psm_config={"method": "nearest", "n_neighbors": 1, "caliper": 0.05},
        )
        if not psm_result.success:
            psm_result = None
    except Exception:
        psm_result = None

    # 4. 平行趋势
    pt_result = best.get("pt_result")
    if pt_result is None:
        df_event = generate_event_time(df_complete, best["x_var"])
        pt_result = run_parallel_trends(
            df_event, best["y_var"], best["x_var"],
            controls, best.get("fe_vars", []), best.get("cluster_vars", []),
            pre_window=best.get("pt_pre_window", -3),
            post_window=best.get("pt_post_window", 3),
            base_period=best.get("pt_base_period", "pre1"),
            chapter=chapter,
        )

    # 生成表格文档
    save_chapter_tables(
        chapter=chapter,
        desc_stats=desc_stats,
        corr_matrix=corr_matrix,
        main_results=[main_result] if isinstance(main_result, RegressionResult) else [],
        psm_result=psm_result,
        pt_results=[pt_result] if pt_result and pt_result.success else [],
        controls_list=controls,
        output_dir=ch_dir,
    )

    # 生成动态效应图
    if pt_result and pt_result.success:
        plot_dynamic_effects(
            pt_result,
            output_path=os.path.join(ch_dir, f"动态效应图_{best['y_var']}.png"),
            title=f"第{chapter}章 动态效应图: {best['y_var']}",
            chapter=chapter,
            y_label=best["y_var"],
        )


def save_search_log(state: SearchState, output_dir: str):
    """保存完整搜索日志。"""
    log_path = os.path.join(output_dir, "search_log.xlsx")

    all_records = []
    for ch, results in state.chapter_results.items():
        for r in results:
            record = {
                "Chapter": ch,
                "Y": r.get("y_var", ""),
                "X": r.get("x_var", ""),
                "Controls": r.get("controls_group", ""),
                "FE": "+".join(r.get("fe_vars", [])),
                "Cluster": "+".join(r.get("cluster_vars", [])),
                "Period": f"{r.get('start_year', '')}-{r.get('end_year', '')}",
                "Coef": r.get("coef", np.nan),
                "SE": r.get("se", np.nan),
                "t-stat": r.get("tstat", np.nan),
                "p-value": r.get("pval", np.nan),
                "Stars": r.get("stars", ""),
                "N": r.get("nobs", 0),
                "R2": r.get("r2", np.nan),
                "Score": r.get("score", np.nan),
                "PT_Score": r.get("pt_score", np.nan),
            }
            all_records.append(record)

    if all_records:
        log_df = pd.DataFrame(all_records)
        log_df.to_excel(log_path, index=False)
        print(f"\n  搜索日志已保存: {log_path}")
        print(f"  共 {len(log_df)} 条记录")
    else:
        print("\n  [警告] 无搜索记录可保存")

    # 最优设定 JSON
    best_path = os.path.join(output_dir, "best_specifications.json")
    best_info = {}
    for ch, best in state.best_specs.items():
        if best:
            best_copy = {k: v for k, v in best.items()
                         if k not in ["reg_result", "pt_result"]}
            # 转换 numpy 类型
            for k, v in best_copy.items():
                if isinstance(v, (np.floating, np.integer)):
                    best_copy[k] = float(v)
                elif isinstance(v, np.ndarray):
                    best_copy[k] = v.tolist()
            best_info[f"Chapter{ch}"] = best_copy

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(best_info, f, ensure_ascii=False, indent=2)
    print(f"  最优设定已保存: {best_path}")


def generate_stata_code(state: SearchState, output_dir: str):
    """
    生成等价的 Stata .do 代码，确保可复现。
    """
    for chapter in [3, 4, 5]:
        best = state.best_specs.get(chapter)
        if not best:
            continue

        do_path = os.path.join(output_dir, f"Chapter{chapter}", f"chapter{chapter}_code.do")
        os.makedirs(os.path.dirname(do_path), exist_ok=True)

        y_var = best["y_var"]
        x_var = best["x_var"]
        controls = best.get("controls", [])
        fe_vars = best.get("fe_vars", [])
        cluster_vars = best.get("cluster_vars", [])
        start_year = best.get("start_year", 2007)
        end_year = best.get("end_year", 2023)
        filters = best.get("filters", {})
        pt_pre = best.get("pt_pre_window", -3)
        pt_post = best.get("pt_post_window", 3)
        pt_base = best.get("pt_base_period", "pre1")
        ctrl_str = " ".join(controls)

        # FE 和 聚类
        fe_cmd = ""
        absorb_vars = []
        for fv in fe_vars:
            absorb_vars.append(fv)
        absorb_str = " ".join(absorb_vars) if absorb_vars else "Year"

        cluster_str = cluster_vars[0] if cluster_vars else "Stkcd"

        # 生成 do 文件
        code = f"""/*
============================================================
第{chapter}章 回归代码 — 自动生成
数字并购与企业内部利益相关者价值效应
============================================================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Y变量: {y_var}
X变量: {x_var}
控制变量: {ctrl_str}
固定效应: {' + '.join(fe_vars)}
聚类: {' + '.join(cluster_vars)}
样本区间: {start_year}-{end_year}
============================================================
*/

clear all
set more off
set matsize 10000

* ============================================================
* 1. 数据加载
* ============================================================
use "{config.DATA_PATH}", clear

* ============================================================
* 2. 样本筛选
* ============================================================
* 样本区间
keep if {config.YEAR_VAR} >= {start_year} & {config.YEAR_VAR} <= {end_year}
"""
        # 样本筛选条件
        if filters.get("drop_st_pt"):
            code += """
* 剔除ST/PT
* 方法1: 如果有ST标识变量
capture drop if ST == 1
* 方法2: 如果有股票简称
capture drop if regexm(Stkname, "ST|PT|\\*ST")
"""
        if filters.get("drop_finance"):
            code += """
* 剔除金融行业 (证监会J门类)
capture drop if substr(string(Ind), 1, 1) == "J"
capture drop if substr(Ind, 1, 1) == "J"
"""
        if filters.get("drop_real_estate"):
            code += """
* 剔除房地产行业 (证监会K门类)
capture drop if substr(string(Ind), 1, 1) == "K"
capture drop if substr(Ind, 1, 1) == "K"
"""
        if filters.get("drop_lev_gt1"):
            code += """
* 剔除资产负债率>1
drop if Lev > 1 & Lev != .
"""
        if filters.get("drop_ind_lt30"):
            code += """
* 剔除行业观测<30的行业
bysort Ind: gen _ind_n = _N
drop if _ind_n < 30
drop _ind_n
"""
        # 缩尾
        winsorize = filters.get("winsorize")
        if winsorize:
            pct = int(winsorize * 100)
            code += f"""
* 连续变量缩尾 ({pct}%)
foreach var of varlist {y_var} {ctrl_str} {{
    capture winsor2 `var', replace cuts({pct} {100 - pct})
}}
"""

        code += f"""
* ============================================================
* 3. 描述统计
* ============================================================
* 确保样本一致: 删除关键变量缺失值
drop if missing({y_var})
drop if missing({x_var})
foreach var of varlist {ctrl_str} {{
    drop if missing(`var')
}}

summarize {x_var} {y_var} {ctrl_str}, detail

* ============================================================
* 4. 相关系数矩阵
* ============================================================
pwcorr {x_var} {y_var} {ctrl_str}, star(0.05)

* ============================================================
* 5. 主回归 (固定效应)
* ============================================================
* 方法: reghdfe (需安装: ssc install reghdfe)
reghdfe {y_var} {x_var} {ctrl_str}, absorb({absorb_str}) vce(cluster {cluster_str})

* 保存结果
est store main_reg
"""
        # PSM-DID
        code += f"""
* ============================================================
* 6. PSM-DID
* ============================================================
* Step 1: 倾向得分匹配
* 构造企业层面处理组标识
bysort {config.FIRM_ID}: egen _ever_treated = max({x_var})
logit _ever_treated {ctrl_str}
predict _pscore, pr

* Step 2: 最近邻匹配 (1:1, caliper=0.05)
* 需要安装 psmatch2: ssc install psmatch2
psmatch2 _ever_treated, pscore(_pscore) neighbor(1) caliper(0.05)

* Step 3: 平衡性检验
pstest {ctrl_str}, both

* Step 4: 匹配后回归
reghdfe {y_var} {x_var} {ctrl_str} if _weight != ., absorb({absorb_str}) vce(cluster {cluster_str})
est store psm_reg
"""
        # 平行趋势
        if pt_base == "pre1":
            omit_label = "pre1"
            omit_val = -1
        elif pt_base == "pre0":
            omit_label = "current"
            omit_val = 0
        else:
            omit_label = f"pre{abs(pt_pre)}"
            omit_val = pt_pre

        code += f"""
* ============================================================
* 7. 平行趋势检验 (事件研究法)
* ============================================================
* 生成事件时间变量
bysort {config.FIRM_ID}: egen _first_treat_year = min({config.YEAR_VAR}) if {x_var} == 1
bysort {config.FIRM_ID}: egen event_year = min(_first_treat_year)
gen event_time = {config.YEAR_VAR} - event_year

* 生成事件时间哑变量
"""
        for p in range(pt_pre, pt_post + 1):
            if p == omit_val:
                continue
            if p < 0:
                name = f"pre{abs(p)}"
            elif p == 0:
                name = "current"
            else:
                name = f"post{p}"
            code += f'gen {name} = (event_time == {p}) & !missing(event_year)\n'

        dummies = []
        for p in range(pt_pre, pt_post + 1):
            if p == omit_val:
                continue
            if p < 0:
                dummies.append(f"pre{abs(p)}")
            elif p == 0:
                dummies.append("current")
            else:
                dummies.append(f"post{p}")
        dummies_str = " ".join(dummies)

        code += f"""
* 回归 (基期: {omit_label})
reghdfe {y_var} {dummies_str} {ctrl_str}, absorb({absorb_str}) vce(cluster {cluster_str})
est store pt_reg

* ============================================================
* 8. 动态效应图
* ============================================================
coefplot, keep({dummies_str}) ///
    vertical ///
    yline(0, lpattern(dash) lcolor(black)) ///
    xline({abs(omit_val) + 1}, lpattern(dash) lcolor(red)) ///
    ylabel(, angle(horizontal)) ///
    xlabel(, angle(0)) ///
    title("第{chapter}章 动态效应图") ///
    xtitle("Event Time") ///
    ytitle("Coefficient") ///
    msymbol(O) mcolor(navy) ///
    ciopts(lcolor(navy) lwidth(thin)) ///
    graphregion(color(white)) bgcolor(white)

graph export "动态效应图_{y_var}.png", replace width(1200)

* ============================================================
* 9. 输出回归结果表
* ============================================================
* 需要安装: ssc install esttab
esttab main_reg psm_reg pt_reg using "第{chapter}章_回归结果.rtf", ///
    replace ///
    star(* 0.10 ** 0.05 *** 0.01) ///
    b(%9.3f) se(%9.3f) ///
    stats(N r2_a, labels("Observations" "Adj. R²") fmt(%9.0fc %9.3f)) ///
    title("第{chapter}章 回归结果") ///
    mtitles("OLS" "PSM-DID" "平行趋势") ///
    note("括号内为聚类稳健标准误; *** p<0.01, ** p<0.05, * p<0.10")

* ============================================================
* End of Chapter {chapter}
* ============================================================
"""
        with open(do_path, "w", encoding="utf-8") as f:
            f.write(code)
        print(f"  Stata代码已保存: {do_path}")


if __name__ == "__main__":
    main()
