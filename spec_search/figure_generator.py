"""
============================================================
图形生成器 — 动态效应图 / 平行趋势图（学术论文规范）
============================================================
规范:
  - 白底, 无网格或极浅网格
  - 事前期: 空心灰色圆点, 事后≥2★显著: 实心深蓝圆点
  - 基期标注, 95% CI 误差线 + 阴影
  - 中文标题与轴标签, 英文字体 Times New Roman
  - 300 DPI, 适合直接插入论文
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from . import config
from .regression_engine import PTResult


def setup_matplotlib():
    """配置 matplotlib 学术论文样式"""
    # 尝试可用中文字体
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
    rcParams["font.family"] = "sans-serif"
    rcParams["mathtext.fontset"] = "stix"
    # 学术论文偏好
    rcParams["axes.linewidth"] = 0.8
    rcParams["xtick.major.width"] = 0.6
    rcParams["ytick.major.width"] = 0.6


def _period_style(p: int, pval: float, chapter: int, coef: float):
    """
    返回散点样式:
      事前不显著: 空心灰色
      事前显著: 空心红色 (不期望)
      事后≥2★+方向正确: 实心深蓝
      事后1★: 半实心蓝
      基期(coef==0): 菱形
    """
    expected = config.EXPECTED_SIGN.get(chapter, 0)
    sign_ok = (expected < 0 and coef < 0) or (expected > 0 and coef > 0) or expected == 0

    if p == 0 and coef == 0:
        # 基期
        return dict(color="#333333", marker="D", facecolors="none",
                    edgecolors="#333333", s=55, zorder=6, linewidths=1.2)
    if p < 0:
        if pval < 0.10:
            return dict(color="#d62728", marker="o", facecolors="none",
                        edgecolors="#d62728", s=55, zorder=5, linewidths=1.2)
        return dict(color="#888888", marker="o", facecolors="none",
                    edgecolors="#888888", s=50, zorder=5, linewidths=1.0)
    else:
        if pval < 0.05 and sign_ok:
            return dict(color="#1f4e79", marker="o", facecolors="#1f4e79",
                        edgecolors="#1f4e79", s=65, zorder=6, linewidths=1.0)
        elif pval < 0.10 and sign_ok:
            return dict(color="#4472c4", marker="o", facecolors="#4472c4",
                        edgecolors="#4472c4", s=55, zorder=5, linewidths=1.0,
                        alpha=0.6)
        return dict(color="#888888", marker="o", facecolors="none",
                    edgecolors="#888888", s=50, zorder=5, linewidths=1.0)


def plot_dynamic_effects(
    pt_result: PTResult,
    output_path: str,
    title: str = "动态效应图",
    chapter: int = 3,
    y_label: str = "系数估计值",
    show_ci: bool = True,
) -> str:
    """
    绘制事件研究法动态效应图（学术论文规范）。

    - 95% CI 阴影 + 误差线
    - 事后≥2★显著期: 实心深蓝大标记
    - 事前显著期: 红色空心标记
    - PT 诊断信息注释
    """
    setup_matplotlib()

    if not pt_result.success or not pt_result.period_coefs:
        return ""

    periods = sorted(pt_result.period_coefs.keys())
    coefs = [pt_result.period_coefs[p] for p in periods]
    ses = [pt_result.period_se.get(p, 0) for p in periods]
    ci_lo = [pt_result.period_ci_lower.get(p, c - 1.96 * s)
             for p, c, s in zip(periods, coefs, ses)]
    ci_hi = [pt_result.period_ci_upper.get(p, c + 1.96 * s)
             for p, c, s in zip(periods, coefs, ses)]

    fig, ax = plt.subplots(figsize=config.FIGURE_SIZE)

    # 白底
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # 95% CI 阴影
    if show_ci:
        ax.fill_between(periods, ci_lo, ci_hi, alpha=0.12, color="#4472c4",
                        label="95% CI")

    # 连线
    ax.plot(periods, coefs, "-", color="#b0b0b0", linewidth=1.0, zorder=3)

    # 误差线
    if show_ci:
        yerr_lo = [c - cl for c, cl in zip(coefs, ci_lo)]
        yerr_hi = [cu - c for c, cu in zip(coefs, ci_hi)]
        ax.errorbar(periods, coefs, yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor="#4472c4", capsize=3, alpha=0.5,
                    elinewidth=0.8, zorder=4)

    # 逐点绘制
    for p, c in zip(periods, coefs):
        pval = pt_result.period_pval.get(p, 1.0)
        style = _period_style(p, pval, chapter, c)
        alpha = style.pop("alpha", 1.0)
        lw = style.pop("linewidths", 1.0)
        ax.scatter([p], [c], alpha=alpha, linewidths=lw, **style)

    # 参考线
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
    ax.axvline(x=-0.5, color="#c00000", linestyle="--", linewidth=0.8, alpha=0.5)

    # 基期标注
    base_map = {"pre1": -1, "pre0": 0, "pre_biggest": pt_result.pre_window}
    base_x = base_map.get(pt_result.base_period, -1)
    if base_x in periods:
        idx = periods.index(base_x)
        ax.annotate("基期", (base_x, coefs[idx]),
                    textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=8.5, color="#555555",
                    arrowprops=dict(arrowstyle="-", color="#999999", alpha=0.4))

    # 显著性星号
    for p, c in zip(periods, coefs):
        pval = pt_result.period_pval.get(p, 1.0)
        if pval < 0.01:
            star = "***"
        elif pval < 0.05:
            star = "**"
        elif pval < 0.10:
            star = "*"
        else:
            continue
        color = "#d62728" if p < 0 else "#1f4e79"
        offset_y = -14 if c > 0 else 10
        ax.annotate(star, (p, c),
                    textcoords="offset points", xytext=(0, offset_y),
                    ha="center", fontsize=7.5, fontweight="bold", color=color)

    # PT 诊断注释
    n_pre = pt_result.n_pre_sig
    max_c2 = pt_result.max_post_consecutive_2star
    status = f"事前显著(10%): {n_pre}期 | 事后连续显著(5%): {max_c2}期"
    ax.text(0.02, 0.02, status, transform=ax.transAxes,
            fontsize=7, color="#666666", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      alpha=0.9, edgecolor="#dddddd", linewidth=0.5))

    # 轴标签
    ax.set_xlabel("事件时间（相对于处理时点）", fontsize=11)
    ax.set_ylabel("回归系数", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(periods)
    ax.tick_params(axis='both', labelsize=9.5)

    # 图例
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f4e79',
               markeredgecolor='#1f4e79', markersize=8, label='事后≥5%显著'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#888888', markersize=7, label='不显著'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor='#d62728', markersize=7, label='事前显著(不期望)'),
        plt.Rectangle((0, 0), 1, 1, fc='#4472c4', alpha=0.12, label='95%置信区间'),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=8, framealpha=0.9,
              edgecolor="#cccccc")
    ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.5)

    # 去除右侧和顶部轴线
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path


def plot_multiple_dynamic_effects(
    pt_results: list[PTResult],
    output_path: str,
    title: str = "动态效应对比",
    chapter: int = 3,
    labels: list[str] | None = None,
) -> str:
    """多子图动态效应对比（学术论文规范）。"""
    setup_matplotlib()

    valid = [pt for pt in pt_results if pt.success and pt.period_coefs]
    if not valid:
        return ""

    n = len(valid)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    palette = ["#1f4e79", "#c55a11", "#2e7d32", "#7b2d8e"]

    for idx, pt in enumerate(valid):
        ax = axes[0, idx]
        ax.set_facecolor("white")

        periods = sorted(pt.period_coefs.keys())
        coefs = [pt.period_coefs[p] for p in periods]
        ses = [pt.period_se.get(p, 0) for p in periods]
        ci_lo = [pt.period_ci_lower.get(p, c - 1.96 * s)
                 for p, c, s in zip(periods, coefs, ses)]
        ci_hi = [pt.period_ci_upper.get(p, c + 1.96 * s)
                 for p, c, s in zip(periods, coefs, ses)]

        clr = palette[idx % len(palette)]

        # CI shade
        ax.fill_between(periods, ci_lo, ci_hi, alpha=0.12, color=clr)
        # Line
        ax.plot(periods, coefs, "-", color="#b0b0b0", linewidth=1.0, zorder=3)
        # Error bars
        ax.errorbar(periods, coefs,
                    yerr=[[c - cl for c, cl in zip(coefs, ci_lo)],
                          [cu - c for c, cu in zip(coefs, ci_hi)]],
                    fmt="none", ecolor=clr, capsize=3, alpha=0.5,
                    elinewidth=0.8, zorder=4)

        # Points
        expected = config.EXPECTED_SIGN.get(chapter, 0)
        for p, c in zip(periods, coefs):
            pval = pt.period_pval.get(p, 1.0)
            sign_ok = (expected < 0 and c < 0) or (expected > 0 and c > 0) or expected == 0
            if p > 0 and pval < 0.05 and sign_ok:
                ax.scatter([p], [c], marker="o", s=65, color=clr, zorder=6)
            elif p < 0 and pval < 0.10:
                ax.scatter([p], [c], marker="o", s=50, facecolors="none",
                           edgecolors="#d62728", linewidths=1.2, zorder=5)
            else:
                ax.scatter([p], [c], marker="o", s=50, facecolors="none",
                           edgecolors="#888888", linewidths=1.0, zorder=5)

        # Stars
        for p, c in zip(periods, coefs):
            pval = pt.period_pval.get(p, 1.0)
            star = "***" if pval < 0.01 else ("**" if pval < 0.05 else ("*" if pval < 0.10 else ""))
            if not star:
                continue
            ax.annotate(star, (p, c), textcoords="offset points",
                        xytext=(0, -12 if c > 0 else 9),
                        ha="center", fontsize=7, fontweight="bold",
                        color="#d62728" if p < 0 else clr)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.6, alpha=0.4)
        ax.axvline(x=-0.5, color="#c00000", linestyle="--", linewidth=0.8, alpha=0.5)

        lbl = labels[idx] if labels and idx < len(labels) else pt.y_var
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.set_xlabel("事件时间", fontsize=10)
        ax.set_ylabel("回归系数", fontsize=10)
        ax.set_xticks(periods)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(True, alpha=0.15, linestyle=":", linewidth=0.5)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        # PT status
        ax.text(0.02, 0.02, f"连续≥5%显著: {pt.max_post_consecutive_2star}期",
                transform=ax.transAxes, fontsize=7, color="#666666", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.9, edgecolor="#dddddd", linewidth=0.5))

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.patch.set_facecolor("white")
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return output_path
