"""
============================================================
表格生成器 — Word 文档输出
============================================================
生成描述统计表、相关系数表、主回归表、PSM平衡性检验表、平行趋势表
"""

import os
import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from . import config
from .regression_engine import RegressionResult, PTResult, PSMResult


def _set_cell_font(cell, text, font_name="Times New Roman", font_size=10.5,
                   bold=False, alignment=WD_ALIGN_PARAGRAPH.CENTER):
    """设置单元格字体格式"""
    cell.text = ""
    p = cell.paragraphs[0]
    p.alignment = alignment
    run = p.add_run(str(text))
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    # 中文字体
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)


def _add_table_border(table):
    """为表格添加三线表边框"""
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else tbl._add_tblPr()

    borders = tblPr.makeelement(qn('w:tblBorders'), {})

    for edge in ['top', 'bottom']:
        element = borders.makeelement(
            qn(f'w:{edge}'),
            {qn('w:val'): 'single', qn('w:sz'): '12',
             qn('w:space'): '0', qn('w:color'): '000000'}
        )
        borders.append(element)

    # 表头下方细线
    element = borders.makeelement(
        qn('w:insideH'),
        {qn('w:val'): 'single', qn('w:sz'): '4',
         qn('w:space'): '0', qn('w:color'): '000000'}
    )
    borders.append(element)

    tblPr.append(borders)


def _format_number(val, decimals=3):
    """格式化数字"""
    if pd.isna(val) or val is None:
        return ""
    if isinstance(val, (int, np.integer)):
        return f"{val:,}"
    return f"{val:.{decimals}f}"


def _format_coef_with_stars(coef, pval, decimals=3):
    """格式化系数并加星号"""
    if pd.isna(coef) or pd.isna(pval):
        return ""
    stars = ""
    if pval < 0.01:
        stars = "***"
    elif pval < 0.05:
        stars = "**"
    elif pval < 0.10:
        stars = "*"
    return f"{coef:.{decimals}f}{stars}"


def create_descriptive_stats_table(
    doc: Document,
    stats_df: pd.DataFrame,
    title: str = "描述统计",
    chapter: int = 3,
) -> None:
    """
    生成描述统计表。

    Parameters
    ----------
    doc : Document
    stats_df : pd.DataFrame
        compute_descriptive_stats 输出
    title : str
    chapter : int
    """
    # 标题
    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(f"表 {chapter}-1 {title}")
    run.font.name = config.TABLE_CN_FONT
    run.font.size = Pt(12)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)

    if len(stats_df) == 0:
        doc.add_paragraph("（无数据）")
        return

    # 表格
    cols = ["Variable", "N", "Mean", "Std", "Min", "P25", "Median", "P75", "Max"]
    table = doc.add_table(rows=len(stats_df) + 1, cols=len(cols))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for j, col in enumerate(cols):
        _set_cell_font(table.rows[0].cells[j], col, bold=True)

    # 数据行
    for i, (_, row) in enumerate(stats_df.iterrows()):
        _set_cell_font(table.rows[i + 1].cells[0], row["Variable"],
                       alignment=WD_ALIGN_PARAGRAPH.LEFT)
        _set_cell_font(table.rows[i + 1].cells[1], f"{int(row['N']):,}")
        for j, col in enumerate(cols[2:], start=2):
            _set_cell_font(table.rows[i + 1].cells[j], _format_number(row[col], 3))

    _add_table_border(table)
    doc.add_paragraph("")  # 空行


def create_correlation_table(
    doc: Document,
    corr_df: pd.DataFrame,
    title: str = "相关系数矩阵",
    chapter: int = 3,
) -> None:
    """
    生成相关系数矩阵表（下三角）。
    """
    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(f"表 {chapter}-2 {title}")
    run.font.name = config.TABLE_CN_FONT
    run.font.size = Pt(12)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)

    if corr_df.empty:
        doc.add_paragraph("（无数据）")
        return

    n_vars = len(corr_df)
    var_names = corr_df.index.tolist()

    # 编号
    table = doc.add_table(rows=n_vars + 1, cols=n_vars + 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头: 编号
    _set_cell_font(table.rows[0].cells[0], "", bold=True)
    for j in range(n_vars):
        _set_cell_font(table.rows[0].cells[j + 1], f"({j + 1})", bold=True)

    # 数据行
    for i in range(n_vars):
        _set_cell_font(table.rows[i + 1].cells[0], f"({i + 1}){var_names[i]}",
                       alignment=WD_ALIGN_PARAGRAPH.LEFT, font_size=9)
        for j in range(n_vars):
            if j <= i:
                val = corr_df.iloc[i, j]
                _set_cell_font(table.rows[i + 1].cells[j + 1], str(val), font_size=9)
            else:
                _set_cell_font(table.rows[i + 1].cells[j + 1], "")

    _add_table_border(table)
    doc.add_paragraph("")


def create_main_regression_table(
    doc: Document,
    results: list[RegressionResult],
    title: str = "基准回归结果",
    chapter: int = 3,
    table_num: int = 3,
    controls_list: list[str] | None = None,
) -> None:
    """
    生成主回归结果表。

    每列一个回归结果，行包括：
    - X 系数 + 星号
    - (t统计量)
    - 控制变量系数
    - 控制变量 Yes/No
    - 固定效应 Yes/No
    - N, R²
    """
    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(f"表 {chapter}-{table_num} {title}")
    run.font.name = config.TABLE_CN_FONT
    run.font.size = Pt(12)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)

    if not results:
        doc.add_paragraph("（无结果）")
        return

    n_cols = len(results)

    # 行结构
    rows_data = []

    # Y 变量名
    rows_data.append([""] + [r.y_var for r in results])
    rows_data.append([""] + [f"({i + 1})" for i in range(n_cols)])

    # X 系数
    x_var = results[0].x_var
    rows_data.append([x_var] + [_format_coef_with_stars(r.coef, r.pval) for r in results])
    # t 统计量
    rows_data.append([""] + [f"({r.tstat:.3f})" if not np.isnan(r.tstat) else "" for r in results])

    # 控制变量
    if controls_list:
        for ctrl in controls_list:
            coefs_row = [ctrl]
            tstats_row = [""]
            for r in results:
                if ctrl in r.all_coefs:
                    coefs_row.append(_format_coef_with_stars(r.all_coefs[ctrl], r.all_pval.get(ctrl, 1)))
                    tstat = r.all_coefs[ctrl] / r.all_se.get(ctrl, 1) if r.all_se.get(ctrl, 0) > 0 else np.nan
                    tstats_row.append(f"({tstat:.3f})" if not np.isnan(tstat) else "")
                else:
                    coefs_row.append("")
                    tstats_row.append("")
            rows_data.append(coefs_row)
            rows_data.append(tstats_row)

    # 控制变量 Yes/No
    rows_data.append(["Controls"] + ["Yes" if r.controls else "No" for r in results])

    # 固定效应
    fe_labels = []
    for r in results:
        fe_str = "+".join(r.fe_vars) if r.fe_vars else "None"
        fe_labels.append(fe_str)
    rows_data.append(["FE"] + fe_labels)

    # N
    rows_data.append(["N"] + [f"{r.nobs:,}" for r in results])

    # R²
    rows_data.append(["R²"] + [_format_number(r.r2, 3) for r in results])

    # 创建表格
    table = doc.add_table(rows=len(rows_data), cols=n_cols + 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_data in enumerate(rows_data):
        for j, val in enumerate(row_data):
            is_bold = (i < 2)  # 前两行加粗
            align = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            _set_cell_font(table.rows[i].cells[j], val, bold=is_bold,
                           alignment=align, font_size=10)

    _add_table_border(table)

    # 注释
    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = note.add_run("注：括号内为t统计量。***、**、*分别表示在1%、5%、10%水平上显著。")
    run.font.size = Pt(9)
    run.font.name = config.TABLE_CN_FONT
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)

    doc.add_paragraph("")


def create_psm_balance_table(
    doc: Document,
    psm_result: PSMResult,
    title: str = "PSM 平衡性检验",
    chapter: int = 3,
    table_num: int = 4,
) -> None:
    """
    生成 PSM 平衡性检验表。
    包含匹配前后的均值差异和标准化偏差。
    """
    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(f"表 {chapter}-{table_num} {title}")
    run.font.name = config.TABLE_CN_FONT
    run.font.size = Pt(12)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)

    if psm_result.balance_before.empty:
        doc.add_paragraph("（无数据）")
        return

    # 合并匹配前后
    cols = ["Variable", "匹配前_处理组均值", "匹配前_对照组均值", "匹配前_标准化偏差(%)",
            "匹配前_t值", "匹配后_处理组均值", "匹配后_对照组均值",
            "匹配后_标准化偏差(%)", "匹配后_t值"]

    before = psm_result.balance_before
    after = psm_result.balance_after

    merged = before[["Variable", "Mean_Treated", "Mean_Control", "Std_Bias(%)", "t_stat"]].copy()
    merged.columns = ["Variable", "匹配前_处理组均值", "匹配前_对照组均值", "匹配前_标准化偏差(%)", "匹配前_t值"]

    if not after.empty:
        after_sub = after[["Variable", "Mean_Treated", "Mean_Control", "Std_Bias(%)", "t_stat"]].copy()
        after_sub.columns = ["Variable", "匹配后_处理组均值", "匹配后_对照组均值", "匹配后_标准化偏差(%)", "匹配后_t值"]
        merged = merged.merge(after_sub, on="Variable", how="left")

    table = doc.add_table(rows=len(merged) + 1, cols=len(merged.columns))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for j, col in enumerate(merged.columns):
        _set_cell_font(table.rows[0].cells[j], col, bold=True, font_size=9)

    # 数据
    for i, (_, row) in enumerate(merged.iterrows()):
        for j, col in enumerate(merged.columns):
            val = row[col]
            if isinstance(val, (float, np.floating)):
                text = _format_number(val, 3)
            else:
                text = str(val)
            _set_cell_font(table.rows[i + 1].cells[j], text, font_size=9)

    _add_table_border(table)
    doc.add_paragraph("")


def create_parallel_trends_table(
    doc: Document,
    pt_results: list[PTResult],
    title: str = "平行趋势检验",
    chapter: int = 3,
    table_num: int = 5,
) -> None:
    """
    生成平行趋势检验表。

    行: 各事件时间期 (pre3, pre2, ..., current, post1, post2, ...)
    列: 不同的Y变量或设定
    """
    heading = doc.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(f"表 {chapter}-{table_num} {title}")
    run.font.name = config.TABLE_CN_FONT
    run.font.size = Pt(12)
    run.font.bold = True
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)

    if not pt_results:
        doc.add_paragraph("（无结果）")
        return

    # 确定所有期
    all_periods = set()
    for pt in pt_results:
        all_periods.update(pt.period_coefs.keys())
    all_periods = sorted(all_periods)

    n_cols = len(pt_results)

    # 行: 期标签 + 系数 + t统计量
    rows_data = []

    # 表头
    rows_data.append([""] + [pt.y_var for pt in pt_results])
    rows_data.append(["Period"] + [f"({i + 1})" for i in range(n_cols)])

    for p in all_periods:
        # 期标签
        if p < 0:
            label = f"pre{abs(p)}"
        elif p == 0:
            label = "current"
        else:
            label = f"post{p}"

        # 系数行
        coef_row = [label]
        tstat_row = [""]

        for pt in pt_results:
            if p in pt.period_coefs:
                coef = pt.period_coefs[p]
                pval = pt.period_pval.get(p, 1.0)
                se = pt.period_se.get(p, 0)
                tstat = coef / se if se > 0 else 0

                coef_row.append(_format_coef_with_stars(coef, pval))
                tstat_row.append(f"({tstat:.3f})")
            else:
                coef_row.append("")
                tstat_row.append("")

        rows_data.append(coef_row)
        rows_data.append(tstat_row)

    # N, R²
    rows_data.append(["N"] + [f"{pt.nobs:,}" for pt in pt_results])
    rows_data.append(["R²"] + [_format_number(pt.r2, 3) for pt in pt_results])

    # 创建表格
    table = doc.add_table(rows=len(rows_data), cols=n_cols + 1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, row_data in enumerate(rows_data):
        for j, val in enumerate(row_data):
            is_bold = (i < 2)
            align = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            _set_cell_font(table.rows[i].cells[j], val, bold=is_bold,
                           alignment=align, font_size=10)

    _add_table_border(table)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.LEFT
    run = note.add_run("注：括号内为t统计量。***、**、*分别表示在1%、5%、10%水平上显著。"
                       "基期已省略（系数为0）。")
    run.font.size = Pt(9)
    run.font.name = config.TABLE_CN_FONT
    run._element.rPr.rFonts.set(qn('w:eastAsia'), config.TABLE_CN_FONT)

    doc.add_paragraph("")


def save_chapter_tables(
    chapter: int,
    desc_stats: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    main_results: list[RegressionResult],
    psm_result: PSMResult | None,
    pt_results: list[PTResult],
    controls_list: list[str],
    output_dir: str,
) -> str:
    """
    保存一章的所有表格到一个 Word 文档。

    Returns
    -------
    str
        输出文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    doc = Document()

    # 页面设置
    section = doc.sections[0]
    section.page_width = Cm(21)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)

    # 章标题
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(f"第{chapter}章 回归结果")
    run.font.size = Pt(16)
    run.font.bold = True

    table_num = 1

    # 1. 描述统计
    create_descriptive_stats_table(doc, desc_stats, "主要变量描述统计", chapter)
    table_num += 1

    # 2. 相关系数
    create_correlation_table(doc, corr_matrix, "主要变量Pearson相关系数矩阵", chapter)
    table_num += 1

    # 3. 主回归
    # 分为匹配前和匹配后（如果有PSM）
    if psm_result and psm_result.success:
        # 匹配前基准回归
        create_main_regression_table(
            doc, [psm_result.baseline_result],
            "基准回归结果（全样本）", chapter, table_num, controls_list
        )
        table_num += 1

        # PSM 平衡性检验
        create_psm_balance_table(doc, psm_result, "PSM平衡性检验", chapter, table_num)
        table_num += 1

        # 匹配后回归
        create_main_regression_table(
            doc, [psm_result.psm_result],
            "PSM-DID回归结果（匹配后样本）", chapter, table_num, controls_list
        )
        table_num += 1
    else:
        create_main_regression_table(
            doc, main_results,
            "基准回归结果", chapter, table_num, controls_list
        )
        table_num += 1

    # 4. 平行趋势检验
    if pt_results:
        create_parallel_trends_table(
            doc, pt_results, "平行趋势检验与动态效应", chapter, table_num
        )
        table_num += 1

    # 保存
    filepath = os.path.join(output_dir, f"第{chapter}章_回归结果.docx")
    doc.save(filepath)
    print(f"[输出] 已保存: {filepath}")
    return filepath
