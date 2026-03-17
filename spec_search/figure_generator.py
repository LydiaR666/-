"""
============================================================
图形生成器 — 动态效应图 / 平行趋势图
============================================================
增强版: 事后≥2星显著期用实心标记突出显示
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from . import config
from .regression_engine import PTResult


def setup_matplotlib():
    """配置 matplotlib 中文字体和样式"""
    for font in ["SimSun", "SimHei", "STSong", "Arial Unicode MS", "PingFang SC",
                  "WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]:
        try:
            rcParams["font.sans-serif"] = [font] + rcParams.get("font.sans-serif", [])
            break
        except Exception:
            continue

    rcParams["axes.unicode_minus"] = False
    rcParams["figure.dpi"] = config.FIGURE_DPI
    rcParams["savefig.dpi"] = config.FIGURE_DPI
    rcParams["figure.figsize"] = config.FIGURE_SIZE


def _get_period_style(p: int, pval: float, chapter: int, coef: float):
    """
    根据期间和显著性返回样式。

    事前不显著: 空心灰色 (期望)
    事前显著: 空心红色 (不期望)
    事后≥2星+方向正确: 实心深蓝 (期望)
    事后1星: 半透明蓝
    事后不显著: 空心灰色
    """
    expected = config.EXPECTED_SIGN.get(chapter, 0)
    sign_ok = (expected < 0 and coef < 0) or (expected > 0 and coef > 0) or expected == 0

    if p < 0:
        # 事前: 不显著最好
        if pval < 0.10:
            return {"color": "#d62728", "marker": "o", "facecolors": "none",
                    "edgecolors": "#d62728", "s": 50, "zorder": 5}
        return {"color": "#999999", "marker": "o", "facecolors": "none",
                "edgecolors": "#999999", "s": 50, "zorder": 5}
    elif p == 0:
        return {"color": "#555555", "marker": "D", "facecolors": "none",
                "edgecolors": "#555555", "s": 55, "zorder": 5}
    else:
        # 事后: ≥2星+方向正确最好
        if pval < 0.05 and sign_ok:
            return {"color": "#1f77b4", "marker": "o", "facecolors": "#1f77b4",
                    "edgecolors": "#1f77b4", "s": 70, "zorder": 6}
        elif pval < 0.10 and sign_ok:
            return {"color": "#1f77b4", "marker": "o", "facecolors": "#1f77b4",
                    "edgecolors": "#1f77b4", "s": 50, "zorder": 5, "alpha": 0.5}
        return {"color": "#999999", "marker": "o", "facecolors": "none",
                "edgecolors": "#999999", "s": 50, "zorder": 5}


def plot_dynamic_effects(
    pt_result: PTResult,
    output_path: str,
    title: str = "动态效应图",
    chapter: int = 3,
    y_label: str = "Coefficient",
    show_ci: bool = True,
) -> str:
    """
    绘制事件研究法动态效应图（增强版）。

    - 事后≥2星显著期: 实心大标记
    - 事前显著期: 红色空心标记（不期望）
    - 95% CI 阴影
    - PT 检验状态注释
    """
    setup_matplotlib()

    if not pt_result.success or not pt_result.period_coefs:
        print(f"[图形] 无法生成: {pt_result.error_msg}")
        return ""

    periods = sorted(pt_result.period_coefs.keys())
    coefs = [pt_result.period_coefs[p] for p in periods]
    ci_lower = [pt_result.period_ci_lower.get(p, c - 1.96 * pt_result.period_se.get(p, 0))
                for p, c in zip(periods, coefs)]
    ci_upper = [pt_result.period_ci_upper.get(p, c + 1.96 * pt_result.period_se.get(p, 0))
                for p, c in zip(periods, coefs)]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    # 95% CI 阴影
    if show_ci:
        ax.fill_between(periods, ci_lower, ci_upper, alpha=0.15, color="steelblue",
                        label="95% CI")

    # 连线（淡灰色）
    ax.plot(periods, coefs, "-", color="#bbbbbb", linewidth=1.2, zorder=3)

    # 误差线
    if show_ci:
        yerr_lower = [c - cl for c, cl in zip(coefs, ci_lower)]
        yerr_upper = [cu - c for c, cu in zip(coefs, ci_upper)]
        ax.errorbar(periods, coefs, yerr=[yerr_lower, yerr_upper],
                    fmt="none", ecolor="steelblue", capsize=3, alpha=0.5, zorder=4)

    # 逐点绘制（根据显著性差异化）
    for p, c in zip(periods, coefs):
        pval = pt_result.period_pval.get(p, 1.0)
        style = _get_period_style(p, pval, chapter, c)
        alpha = style.pop("alpha", 1.0)
        ax.scatter([p], [c], alpha=alpha, **style)

    # 参考线
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=-0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.4,
               label="Treatment")

    # 基期标注
    base_period = pt_result.base_period
    if base_period == "pre1":
        base_x = -1
    elif base_period == "pre0":
        base_x = 0
    elif base_period == "pre_biggest":
        base_x = pt_result.pre_window
    else:
        base_x = -1

    if base_x in periods:
        idx = periods.index(base_x)
        ax.annotate("Base", (base_x, coefs[idx]),
                     textcoords="offset points", xytext=(0, 15),
                     ha="center", fontsize=9, color="gray",
                     arrowprops=dict(arrowstyle="-", color="gray", alpha=0.3))

    # 显著性星号
    for p, c in zip(periods, coefs):
        pval = pt_result.period_pval.get(p, 1.0)
        if pval < 0.01:
            marker = "***"
        elif pval < 0.05:
            marker = "**"
        elif pval < 0.10:
            marker = "*"
        else:
            continue
        color = "#d62728" if p < 0 else "#1f77b4"
        ax.annotate(marker, (p, c),
                     textcoords="offset points",
                     xytext=(0, -15 if c > 0 else 10),
                     ha="center", fontsize=8, fontweight="bold", color=color)

    # PT 状态注释
    n_pre = pt_result.n_pre_sig
    n_post_2star = pt_result.n_post_sig_2star
    max_consec_2star = pt_result.max_post_consecutive_2star
    pt_status = (f"Pre-sig(10%): {n_pre}  |  "
                 f"Post-sig(5%): {n_post_2star}  |  "
                 f"Max consec(5%): {max_consec_2star}")
    ax.text(0.02, 0.02, pt_status, transform=ax.transAxes,
            fontsize=7.5, color="#666666", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="#cccccc"))

    # 格式化
    ax.set_xlabel("Event Time (relative to treatment)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(periods)

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
               markeredgecolor='#1f77b4', markersize=8, label='Post ≥2★ (correct sign)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#999999', markersize=7, label='Not significant'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#d62728', markersize=7, label='Pre-period sig (undesired)'),
        plt.Rectangle((0, 0), 1, 1, fc='steelblue', alpha=0.15, label='95% CI'),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[图形] 已保存: {output_path}")
    return output_path


def plot_multiple_dynamic_effects(
    pt_results: list[PTResult],
    output_path: str,
    title: str = "动态效应图",
    chapter: int = 3,
    labels: list[str] | None = None,
) -> str:
    """
    多子图动态效应对比（增强版）。
    """
    setup_matplotlib()

    valid_results = [pt for pt in pt_results if pt.success and pt.period_coefs]
    if not valid_results:
        print("[图形] 无有效结果")
        return ""

    n_plots = len(valid_results)
    fig, axes = plt.subplots(
        1, n_plots,
        figsize=(5 * n_plots, 5),
        squeeze=False,
    )

    colors = ["steelblue", "coral", "forestgreen", "purple"]

    for idx, pt in enumerate(valid_results):
        ax = axes[0, idx]
        periods = sorted(pt.period_coefs.keys())
        coefs = [pt.period_coefs[p] for p in periods]
        ci_lower = [pt.period_ci_lower.get(p, c - 1.96 * pt.period_se.get(p, 0))
                    for p, c in zip(periods, coefs)]
        ci_upper = [pt.period_ci_upper.get(p, c + 1.96 * pt.period_se.get(p, 0))
                    for p, c in zip(periods, coefs)]

        base_color = colors[idx % len(colors)]

        # CI shade
        ax.fill_between(periods, ci_lower, ci_upper, alpha=0.15, color=base_color)

        # Line
        ax.plot(periods, coefs, "-", color="#bbbbbb", linewidth=1.2, zorder=3)

        # Error bars
        yerr_lower = [c - cl for c, cl in zip(coefs, ci_lower)]
        yerr_upper = [cu - c for c, cu in zip(coefs, ci_upper)]
        ax.errorbar(periods, coefs, yerr=[yerr_lower, yerr_upper],
                    fmt="none", ecolor=base_color, capsize=3, alpha=0.5, zorder=4)

        # Points with differentiated styling
        for p, c in zip(periods, coefs):
            pval = pt.period_pval.get(p, 1.0)
            expected = config.EXPECTED_SIGN.get(chapter, 0)
            sign_ok = (expected < 0 and c < 0) or (expected > 0 and c > 0) or expected == 0

            if p > 0 and pval < 0.05 and sign_ok:
                ax.scatter([p], [c], marker="o", s=70, color=base_color, zorder=6)
            elif p < 0 and pval < 0.10:
                ax.scatter([p], [c], marker="o", s=50, facecolors="none",
                          edgecolors="#d62728", zorder=5)
            else:
                ax.scatter([p], [c], marker="o", s=50, facecolors="none",
                          edgecolors="#999999", zorder=5)

        # Stars
        for p, c in zip(periods, coefs):
            pval = pt.period_pval.get(p, 1.0)
            if pval < 0.01:
                star = "***"
            elif pval < 0.05:
                star = "**"
            elif pval < 0.10:
                star = "*"
            else:
                continue
            ax.annotate(star, (p, c), textcoords="offset points",
                       xytext=(0, -13 if c > 0 else 9),
                       ha="center", fontsize=7, fontweight="bold",
                       color="#d62728" if p < 0 else base_color)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=-0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.4)

        label = labels[idx] if labels and idx < len(labels) else pt.y_var
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Event Time", fontsize=10)
        ax.set_ylabel("Coefficient", fontsize=10)
        ax.set_xticks(periods)
        ax.grid(True, alpha=0.3, linestyle=":")

        # PT status
        status = f"Post≥2★: {pt.max_post_consecutive_2star} consec"
        ax.text(0.02, 0.02, status, transform=ax.transAxes,
                fontsize=7, color="#666666", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                         alpha=0.8, edgecolor="#cccccc"))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[图形] 已保存: {output_path}")
    return output_path
