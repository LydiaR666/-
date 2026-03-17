"""
============================================================
图形生成器 — 动态效应图 / 平行趋势图
============================================================
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
    # 尝试中文字体
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


def plot_dynamic_effects(
    pt_result: PTResult,
    output_path: str,
    title: str = "动态效应图",
    chapter: int = 3,
    y_label: str = "Coefficient",
    show_ci: bool = True,
) -> str:
    """
    绘制事件研究法动态效应图。

    X轴: 事件时间 (相对于处理年份)
    Y轴: 系数估计值
    误差线: 95% 置信区间
    虚线: t=0 垂直线, y=0 水平线

    Parameters
    ----------
    pt_result : PTResult
    output_path : str
    title : str
    chapter : int
    y_label : str
    show_ci : bool

    Returns
    -------
    str
        保存的文件路径
    """
    setup_matplotlib()

    if not pt_result.success or not pt_result.period_coefs:
        print(f"[图形] 无法生成: {pt_result.error_msg}")
        return ""

    # 提取数据
    periods = sorted(pt_result.period_coefs.keys())
    coefs = [pt_result.period_coefs[p] for p in periods]
    ci_lower = [pt_result.period_ci_lower.get(p, c - 1.96 * pt_result.period_se.get(p, 0))
                for p, c in zip(periods, coefs)]
    ci_upper = [pt_result.period_ci_upper.get(p, c + 1.96 * pt_result.period_se.get(p, 0))
                for p, c in zip(periods, coefs)]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    # 绘制置信区间（灰色阴影）
    if show_ci:
        ax.fill_between(periods, ci_lower, ci_upper, alpha=0.2, color="steelblue",
                        label="95% CI")

    # 绘制系数点和连线
    ax.plot(periods, coefs, "o-", color="steelblue", markersize=6, linewidth=1.5,
            label="Coefficient", zorder=5)

    # 误差线
    if show_ci:
        yerr_lower = [c - cl for c, cl in zip(coefs, ci_lower)]
        yerr_upper = [cu - c for c, cu in zip(coefs, ci_upper)]
        ax.errorbar(periods, coefs, yerr=[yerr_lower, yerr_upper],
                    fmt="none", ecolor="steelblue", capsize=3, alpha=0.6)

    # 参考线
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    # 标注基期
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
                     ha="center", fontsize=9, color="gray")

    # 标注显著性
    for p, c in zip(periods, coefs):
        pval = pt_result.period_pval.get(p, 1.0)
        if pval < 0.01:
            marker = "***"
        elif pval < 0.05:
            marker = "**"
        elif pval < 0.10:
            marker = "*"
        else:
            marker = ""
        if marker:
            ax.annotate(marker, (p, c),
                         textcoords="offset points",
                         xytext=(0, -15 if c > 0 else 10),
                         ha="center", fontsize=8, color="darkred")

    # 格式化
    ax.set_xlabel("Event Time (relative to treatment)", fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(periods)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, linestyle=":")

    plt.tight_layout()

    # 保存
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
    在同一张图上绘制多个被解释变量的动态效应。
    适用于同一章多个Y变量的对比。
    """
    setup_matplotlib()

    valid_results = [pt for pt in pt_results if pt.success and pt.period_coefs]
    if not valid_results:
        print("[图形] 无有效结果")
        return ""

    fig, axes = plt.subplots(
        1, len(valid_results),
        figsize=(5 * len(valid_results), 5),
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

        color = colors[idx % len(colors)]

        ax.fill_between(periods, ci_lower, ci_upper, alpha=0.2, color=color)
        ax.plot(periods, coefs, "o-", color=color, markersize=5, linewidth=1.5)

        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.axvline(x=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

        label = labels[idx] if labels and idx < len(labels) else pt.y_var
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Event Time", fontsize=10)
        ax.set_ylabel("Coefficient", fontsize=10)
        ax.set_xticks(periods)
        ax.grid(True, alpha=0.3, linestyle=":")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[图形] 已保存: {output_path}")
    return output_path
