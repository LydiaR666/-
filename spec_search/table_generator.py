"""
============================================================
表格生成器 — Word 文档输出（严格学术三线表模板）
============================================================
格式规范:
  - 三线表: 顶线(1.5pt) + 表头下线(0.75pt) + 底线(1.5pt)
  - 系数加星号, 括号内为标准误 (非t统计量)
  - 中文宋体 + 英文/数字 Times New Roman, 五号(10.5pt)
  - 主回归表: 逐步加入控制变量 + FE 多列
  - 平行趋势表: 事件时间哑变量系数 + 标准误
  - 尾注: 聚类层级 + 显著性水平说明
"""

import os
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from . import config
from .regression_engine import RegressionResult, PTResult, PSMResult


# ============================================================
# 基础工具
# ============================================================

def _set_cell(cell, text, font_en="Times New Roman", font_cn=None,
              size=10.5, bold=False, align=WD_ALIGN_PARAGRAPH.CENTER,
              italic=False):
    """设置单元格内容和格式 (宋体中文 + TNR英文)。"""
    font_cn = font_cn or config.TABLE_CN_FONT
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = align
    # 段前段后 0
    pf = p.paragraph_format
    pf.space_before = Pt(1)
    pf.space_after = Pt(1)
    pf.line_spacing = Pt(14)

    run = p.add_run(str(text))
    run.font.name = font_en
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run._element.rPr.rFonts.set(qn('w:eastAsia'), font_cn)


def _add_three_line_border(table):
    """
    严格三线表:
      顶线 1.5pt, 表头下线 0.75pt, 底线 1.5pt
      其他边框全部无
    """
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else tbl._add_tblPr()

    # 清除所有默认边框
    borders = OxmlElement('w:tblBorders')
    for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'none')
        el.set(qn('w:sz'), '0')
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), '000000')
        borders.append(el)
    tblPr.append(borders)

    # 顶线 1.5pt (12 half-points)
    _set_row_border(table.rows[0], 'top', '12')
    # 表头下线 0.75pt (6 half-points)
    _set_row_border(table.rows[0], 'bottom', '6')
    # 底线 1.5pt
    _set_row_border(table.rows[-1], 'bottom', '12')


def _set_row_border(row, edge, sz):
    """为单行设置指定边框。"""
    for cell in row.cells:
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        borders = tcPr.find(qn('w:tcBorders'))
        if borders is None:
            borders = OxmlElement('w:tcBorders')
            tcPr.append(borders)
        el = OxmlElement(f'w:{edge}')
        el.set(qn('w:val'), 'single')
        el.set(qn('w:sz'), sz)
        el.set(qn('w:space'), '0')
        el.set(qn('w:color'), '000000')
        borders.append(el)


def _fmt(val, d=4):
    """格式化数值 (4位小数)。"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    return f"{val:.{d}f}"


def _fmt_coef(coef, pval, d=4):
    """系数 + 星号。"""
    if coef is None or (isinstance(coef, float) and np.isnan(coef)):
        return ""
    stars = ""
    if not np.isnan(pval):
        if pval < 0.01:
            stars = "***"
        elif pval < 0.05:
            stars = "**"
        elif pval < 0.10:
            stars = "*"
    return f"{coef:.{d}f}{stars}"


def _fmt_se(se, d=4):
    """(标准误) 括号格式。"""
    if se is None or (isinstance(se, float) and np.isnan(se)):
        return ""
    return f"({se:.{d}f})"


def _add_title(doc, text, size=12):
    """居中加粗标题。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pf = p.paragraph_format
    pf.space_before = Pt(6)
    pf.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)


def _add_note(doc, text):
    """表格注释 (小五号, 左对齐)。"""
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf = p.paragraph_format
    pf.space_before = Pt(2)
    run = p.add_run(text)
    run.font.name = "Times New Roman"
    run.font.size = Pt(9)
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)


def _set_col_widths(table, widths_cm):
    """设置列宽 (cm 列表)。"""
    for row in table.rows:
        for i, w in enumerate(widths_cm):
            if i < len(row.cells):
                row.cells[i].width = Cm(w)


# ============================================================
# 1. 描述统计表
# ============================================================

def create_descriptive_stats_table(
    doc: Document,
    stats_df: pd.DataFrame,
    title: str = "描述统计",
    chapter: int = 3,
) -> None:
    """
    表 X-1 描述统计表
    列: Variable | N | Mean | Std | Min | P25 | Median | P75 | Max
    """
    _add_title(doc, f"表{chapter}-1  {title}")

    if len(stats_df) == 0:
        doc.add_paragraph("（无数据）")
        return

    cols = ["Variable", "N", "Mean", "Std", "Min", "P25", "Median", "P75", "Max"]
    n_rows = len(stats_df) + 1
    table = doc.add_table(rows=n_rows, cols=len(cols))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for j, col in enumerate(cols):
        _set_cell(table.rows[0].cells[j], col, bold=True)

    # 数据
    for i, (_, row) in enumerate(stats_df.iterrows()):
        _set_cell(table.rows[i + 1].cells[0], row["Variable"],
                  align=WD_ALIGN_PARAGRAPH.LEFT)
        _set_cell(table.rows[i + 1].cells[1], f"{int(row['N']):,}")
        for j, col in enumerate(cols[2:], start=2):
            _set_cell(table.rows[i + 1].cells[j], _fmt(row[col], 4))

    _add_three_line_border(table)
    _add_note(doc, f"注：样本量N={int(stats_df['N'].iloc[0]):,}。连续变量在1%和99%水平上进行了缩尾处理。")


# ============================================================
# 2. 相关系数矩阵
# ============================================================

def create_correlation_table(
    doc: Document,
    corr_df: pd.DataFrame,
    title: str = "相关系数矩阵",
    chapter: int = 3,
) -> None:
    """
    表 X-2 Pearson相关系数矩阵 (下三角)
    """
    _add_title(doc, f"表{chapter}-2  {title}")

    if corr_df.empty:
        doc.add_paragraph("（无数据）")
        return

    n_vars = len(corr_df)
    var_names = corr_df.index.tolist()

    table = doc.add_table(rows=n_vars + 1, cols=n_vars + 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    _set_cell(table.rows[0].cells[0], "", bold=True)
    for j in range(n_vars):
        _set_cell(table.rows[0].cells[j + 1], f"({j + 1})", bold=True, size=9)

    # 数据
    for i in range(n_vars):
        _set_cell(table.rows[i + 1].cells[0],
                  f"({i + 1}){var_names[i]}",
                  align=WD_ALIGN_PARAGRAPH.LEFT, size=9)
        for j in range(n_vars):
            if j <= i:
                val = corr_df.iloc[i, j]
                _set_cell(table.rows[i + 1].cells[j + 1], str(val), size=9)
            else:
                _set_cell(table.rows[i + 1].cells[j + 1], "")

    _add_three_line_border(table)
    _add_note(doc, "注：***、**、*分别表示在1%、5%、10%水平上显著。下三角为Pearson相关系数。")


# ============================================================
# 3. 主回归结果表（学术标准: 系数+星号, 括号内标准误）
# ============================================================

def create_main_regression_table(
    doc: Document,
    results: list[RegressionResult],
    title: str = "基准回归结果",
    chapter: int = 3,
    table_num: int = 3,
    controls_list: list[str] | None = None,
    cluster_desc: str = "",
    show_controls_coefs: bool = False,
) -> None:
    """
    学术三线表格式:
      第一行: Y变量名(被解释变量)
      第二行: 列编号 (1) (2) ...
      X系数行: 系数+星号
      SE行: (标准误)
      [可选] 控制变量系数行
      Controls: Yes/No
      FE: 具体内容
      Cluster: 具体内容
      N
      R² / Adj. R²
    """
    _add_title(doc, f"表{chapter}-{table_num}  {title}")

    if not results:
        doc.add_paragraph("（无结果）")
        return

    n_cols = len(results)
    rows_data = []

    # --- 表头行 ---
    # Y 变量名
    rows_data.append(("header", [""] + [r.y_var for r in results]))
    # 列编号
    rows_data.append(("header", [""] + [f"({i + 1})" for i in range(n_cols)]))

    # --- X 变量系数 ---
    x_var = results[0].x_var
    rows_data.append(("coef", [x_var] +
                      [_fmt_coef(r.coef, r.pval) for r in results]))
    rows_data.append(("se", [""] +
                      [_fmt_se(r.se) for r in results]))

    # --- 控制变量系数（可选，默认不展示避免冗长）---
    if show_controls_coefs and controls_list:
        for ctrl in controls_list:
            ctrl_coef = [ctrl]
            ctrl_se = [""]
            for r in results:
                if ctrl in r.all_coefs:
                    ctrl_coef.append(_fmt_coef(
                        r.all_coefs[ctrl], r.all_pval.get(ctrl, 1)))
                    ctrl_se.append(_fmt_se(r.all_se.get(ctrl, np.nan)))
                else:
                    ctrl_coef.append("")
                    ctrl_se.append("")
            rows_data.append(("coef", ctrl_coef))
            rows_data.append(("se", ctrl_se))

    # --- 固定部分 ---
    # Controls
    rows_data.append(("info", ["Controls"] +
                      ["Yes" if r.controls else "No" for r in results]))

    # FE (显示具体内容)
    def _fe_label(r):
        if not r.fe_vars:
            return "No"
        labels = []
        for v in r.fe_vars:
            if v == config.FIRM_ID:
                labels.append("个体")
            elif v == config.YEAR_VAR:
                labels.append("年份")
            elif v in (config.IND_VAR, config.IND_VAR_ALT):
                labels.append("行业")
            else:
                labels.append(v)
        return "、".join(labels)

    rows_data.append(("info", ["固定效应"] +
                      [_fe_label(r) for r in results]))

    # Cluster
    def _cl_label(r):
        if not r.cluster_vars:
            return "—"
        labels = []
        for v in r.cluster_vars:
            if v == config.FIRM_ID:
                labels.append("个体")
            elif v == config.YEAR_VAR:
                labels.append("年份")
            elif v in (config.IND_VAR, config.IND_VAR_ALT):
                labels.append("行业")
            else:
                labels.append(v)
        return "、".join(labels)

    rows_data.append(("info", ["聚类层级"] +
                      [_cl_label(r) for r in results]))

    # N
    rows_data.append(("info", ["N"] + [f"{r.nobs:,}" for r in results]))

    # R²
    rows_data.append(("info", ["R²"] + [_fmt(r.r2, 4) for r in results]))

    # Adj R² / Within R²
    rows_data.append(("info", ["Adj. R²"] +
                      [_fmt(r.r2_within, 4) for r in results]))

    # --- 渲染表格 ---
    table = doc.add_table(rows=len(rows_data), cols=n_cols + 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, (row_type, row_data) in enumerate(rows_data):
        for j, val in enumerate(row_data):
            is_bold = (row_type == "header")
            is_italic = (row_type == "se")
            align = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            _set_cell(table.rows[i].cells[j], val, bold=is_bold,
                      italic=is_italic, align=align, size=10.5)

    _add_three_line_border(table)

    # 尾注
    cl_desc = cluster_desc or _cl_label(results[0])
    _add_note(doc,
              f"注：括号内为聚类稳健标准误（聚类至{cl_desc}层面）。"
              "***、**、*分别表示在1%、5%、10%水平上显著。")


# ============================================================
# 4. PSM 平衡性检验表
# ============================================================

def create_psm_balance_table(
    doc: Document,
    psm_result: PSMResult,
    title: str = "PSM平衡性检验",
    chapter: int = 3,
    table_num: int = 4,
) -> None:
    """
    PSM平衡性检验表: 匹配前后均值 + 标准化偏差 + t检验
    """
    _add_title(doc, f"表{chapter}-{table_num}  {title}")

    before = psm_result.balance_before
    after = psm_result.balance_after
    if before.empty:
        doc.add_paragraph("（无数据）")
        return

    # 表头
    header1 = ["", "匹配前", "", "", "", "匹配后", "", "", ""]
    header2 = ["变量", "处理组", "对照组", "偏差(%)", "t值",
               "处理组", "对照组", "偏差(%)", "t值"]

    n_vars = len(before)
    table = doc.add_table(rows=n_vars + 2, cols=9)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 合并表头第一行的匹配前/后
    for j, val in enumerate(header1):
        _set_cell(table.rows[0].cells[j], val, bold=True, size=10)
    for j, val in enumerate(header2):
        _set_cell(table.rows[1].cells[j], val, bold=True, size=9.5)

    # 数据行
    for i in range(n_vars):
        row_b = before.iloc[i]
        _set_cell(table.rows[i + 2].cells[0], row_b["Variable"],
                  align=WD_ALIGN_PARAGRAPH.LEFT, size=9.5)
        _set_cell(table.rows[i + 2].cells[1], _fmt(row_b["Mean_Treated"], 4), size=9.5)
        _set_cell(table.rows[i + 2].cells[2], _fmt(row_b["Mean_Control"], 4), size=9.5)
        _set_cell(table.rows[i + 2].cells[3], _fmt(row_b["Std_Bias(%)"], 2), size=9.5)
        _set_cell(table.rows[i + 2].cells[4], _fmt(row_b["t_stat"], 3), size=9.5)

        if not after.empty and i < len(after):
            row_a = after.iloc[i]
            _set_cell(table.rows[i + 2].cells[5], _fmt(row_a["Mean_Treated"], 4), size=9.5)
            _set_cell(table.rows[i + 2].cells[6], _fmt(row_a["Mean_Control"], 4), size=9.5)
            _set_cell(table.rows[i + 2].cells[7], _fmt(row_a["Std_Bias(%)"], 2), size=9.5)
            _set_cell(table.rows[i + 2].cells[8], _fmt(row_a["t_stat"], 3), size=9.5)

    _add_three_line_border(table)
    _add_note(doc, f"注：处理组{psm_result.n_treated}家，匹配后对照组{psm_result.n_control}家。"
              f"匹配方法：{psm_result.psm_method}。标准化偏差(%)衡量匹配质量，越小越好。")


# ============================================================
# 5. PSM-DID 回归结果表
# ============================================================

def create_psm_did_table(
    doc: Document,
    baseline: RegressionResult,
    psm_reg: RegressionResult,
    title: str = "PSM-DID回归结果",
    chapter: int = 3,
    table_num: int = 5,
    controls_list: list[str] | None = None,
) -> None:
    """
    PSM-DID表: 列(1)全样本基准, 列(2)匹配后样本
    """
    results = []
    if baseline.success:
        results.append(baseline)
    if psm_reg.success:
        results.append(psm_reg)

    if not results:
        _add_title(doc, f"表{chapter}-{table_num}  {title}")
        doc.add_paragraph("（无结果）")
        return

    create_main_regression_table(
        doc, results, title, chapter, table_num, controls_list,
    )


# ============================================================
# 6. 平行趋势检验表（严格学术格式）
# ============================================================

def create_parallel_trends_table(
    doc: Document,
    pt_results: list[PTResult],
    title: str = "平行趋势检验",
    chapter: int = 3,
    table_num: int = 6,
) -> None:
    """
    平行趋势检验表 (事件研究法):
      行: Pre_n, ..., Pre_1(基期), Current, Post_1, ..., Post_n
      系数+星号, 括号内标准误
      底部: Controls, FE, Cluster, N, R²
    """
    _add_title(doc, f"表{chapter}-{table_num}  {title}")

    if not pt_results:
        doc.add_paragraph("（无结果）")
        return

    # 确定所有期
    all_periods = set()
    for pt in pt_results:
        all_periods.update(pt.period_coefs.keys())
    all_periods = sorted(all_periods)

    n_cols = len(pt_results)
    rows_data = []

    # 表头
    rows_data.append(("header", [""] + [pt.y_var for pt in pt_results]))
    rows_data.append(("header", [""] + [f"({i + 1})" for i in range(n_cols)]))

    # 期间行
    for p in all_periods:
        if p < 0:
            label = f"Pre_{abs(p)}"
        elif p == 0:
            label = "Current"
        else:
            label = f"Post_{p}"

        coef_row = [label]
        se_row = [""]

        for pt in pt_results:
            if p in pt.period_coefs:
                coef = pt.period_coefs[p]
                pval = pt.period_pval.get(p, 1.0)
                se = pt.period_se.get(p, 0)

                if se == 0 and pval == 1.0:
                    # 基期
                    coef_row.append("Base")
                    se_row.append("")
                else:
                    coef_row.append(_fmt_coef(coef, pval))
                    se_row.append(_fmt_se(se))
            else:
                coef_row.append("")
                se_row.append("")

        rows_data.append(("coef", coef_row))
        rows_data.append(("se", se_row))

    # Controls / FE / N / R²
    rows_data.append(("info", ["Controls"] + ["Yes"] * n_cols))
    rows_data.append(("info", ["固定效应"] + ["个体、年份"] * n_cols))
    rows_data.append(("info", ["N"] + [f"{pt.nobs:,}" for pt in pt_results]))
    rows_data.append(("info", ["R²"] + [_fmt(pt.r2, 4) for pt in pt_results]))

    # 渲染
    table = doc.add_table(rows=len(rows_data), cols=n_cols + 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, (row_type, row_data) in enumerate(rows_data):
        for j, val in enumerate(row_data):
            is_bold = (row_type == "header") or (val == "Base")
            is_italic = (row_type == "se")
            align = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            _set_cell(table.rows[i].cells[j], val, bold=is_bold,
                      italic=is_italic, align=align, size=10.5)

    _add_three_line_border(table)

    # 尾注
    base_desc = "t=-1" if pt_results[0].base_period == "pre1" else pt_results[0].base_period
    _add_note(doc,
              f"注：基期为{base_desc}（系数标准化为0）。"
              "括号内为聚类稳健标准误。"
              "***、**、*分别表示在1%、5%、10%水平上显著。")


# ============================================================
# 7. 稳健性检验表（替代解释变量）
# ============================================================

def create_robustness_x_table(
    doc: Document,
    results: list[RegressionResult],
    title: str = "稳健性检验：替代解释变量",
    chapter: int = 3,
    table_num: int = 7,
    controls_list: list[str] | None = None,
) -> None:
    """
    稳健性检验表: 每列使用不同 X 变量, 显示各自系数+标准误
    """
    _add_title(doc, f"表{chapter}-{table_num}  {title}")

    if not results:
        doc.add_paragraph("（无结果）")
        return

    n_cols = len(results)
    rows_data = []

    # Y 变量
    rows_data.append(("header", [""] + [r.y_var for r in results]))
    rows_data.append(("header", [""] + [f"({i + 1})" for i in range(n_cols)]))

    # 每列的 X 变量不同
    # 先收集所有不重复的 X
    all_x = list(dict.fromkeys(r.x_var for r in results))
    for x_var in all_x:
        coef_row = [x_var]
        se_row = [""]
        for r in results:
            if r.x_var == x_var:
                coef_row.append(_fmt_coef(r.coef, r.pval))
                se_row.append(_fmt_se(r.se))
            else:
                coef_row.append("")
                se_row.append("")
        rows_data.append(("coef", coef_row))
        rows_data.append(("se", se_row))

    # 底部信息
    rows_data.append(("info", ["Controls"] + ["Yes"] * n_cols))
    rows_data.append(("info", ["固定效应"] + ["个体、年份"] * n_cols))
    rows_data.append(("info", ["N"] + [f"{r.nobs:,}" for r in results]))
    rows_data.append(("info", ["R²"] + [_fmt(r.r2, 4) for r in results]))

    # 渲染
    table = doc.add_table(rows=len(rows_data), cols=n_cols + 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, (row_type, row_data) in enumerate(rows_data):
        for j, val in enumerate(row_data):
            is_bold = (row_type == "header")
            is_italic = (row_type == "se")
            align = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            sz = 9.5 if n_cols > 4 else 10.5
            _set_cell(table.rows[i].cells[j], val, bold=is_bold,
                      italic=is_italic, align=align, size=sz)

    _add_three_line_border(table)
    _add_note(doc,
              "注：括号内为聚类稳健标准误。"
              "***、**、*分别表示在1%、5%、10%水平上显著。"
              "各列使用不同的解释变量（DID处理变量的替代度量）。")


# ============================================================
# 合并文档
# ============================================================

def save_chapter_tables(
    chapter: int,
    desc_stats: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    main_results: list[RegressionResult],
    psm_result: PSMResult | None,
    pt_results: list[PTResult],
    controls_list: list[str],
    output_dir: str,
    robustness_results: list[RegressionResult] | None = None,
) -> str:
    """保存一章全部表格到合并Word文档。"""
    os.makedirs(output_dir, exist_ok=True)
    doc = Document()

    # A4 页面
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)

    tnum = 1

    # 1. 描述统计
    create_descriptive_stats_table(doc, desc_stats, "主要变量描述统计", chapter)
    tnum += 1

    # 2. 相关系数
    create_correlation_table(doc, corr_matrix, "主要变量Pearson相关系数矩阵", chapter)
    tnum += 1

    # 3. 主回归
    if psm_result and psm_result.success:
        # 3a. 全样本基准
        create_main_regression_table(
            doc, [psm_result.baseline_result] if psm_result.baseline_result.success else main_results,
            "基准回归结果（全样本）", chapter, tnum, controls_list,
        )
        tnum += 1
        # 3b. 平衡性检验
        create_psm_balance_table(doc, psm_result, "PSM平衡性检验", chapter, tnum)
        tnum += 1
        # 3c. 匹配后回归
        if psm_result.psm_result.success:
            create_main_regression_table(
                doc, [psm_result.psm_result],
                "PSM-DID回归结果（匹配后）", chapter, tnum, controls_list,
            )
            tnum += 1
    else:
        create_main_regression_table(
            doc, main_results, "基准回归结果", chapter, tnum, controls_list,
        )
        tnum += 1

    # 4. 稳健性
    if robustness_results:
        create_robustness_x_table(
            doc, robustness_results, "稳健性检验：替代解释变量",
            chapter, tnum, controls_list,
        )
        tnum += 1

    # 5. 平行趋势
    if pt_results:
        create_parallel_trends_table(
            doc, pt_results, "平行趋势检验与动态效应", chapter, tnum,
        )
        tnum += 1

    filepath = os.path.join(output_dir, f"第{chapter}章_回归结果.docx")
    doc.save(filepath)
    return filepath
