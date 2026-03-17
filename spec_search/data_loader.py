"""
============================================================
数据加载与预处理模块
============================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import mstats
from . import config

# 控制批量搜索期间的打印输出
_VERBOSE = True


def set_verbose(v: bool):
    global _VERBOSE
    _VERBOSE = v


def _log(msg: str):
    if _VERBOSE:
        print(msg)


def load_data(path: str | None = None) -> pd.DataFrame:
    """读取 Stata .dta 数据文件。"""
    path = path or config.DATA_PATH
    print(f"[数据加载] 正在读取: {path}")
    df = pd.read_stata(path, convert_categoricals=False)
    print(f"[数据加载] 原始样本量: {len(df)} 行 × {len(df.columns)} 列")

    # 检测关键变量
    _check_key_variables(df)

    # 确保 Year 为整数
    year_col = config.YEAR_VAR
    if year_col in df.columns:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.dropna(subset=[year_col])
        df[year_col] = df[year_col].astype(int)
        print(f"[数据加载] 年份范围: {df[year_col].min()} – {df[year_col].max()}")

    return df


def _check_key_variables(df: pd.DataFrame) -> None:
    """检查关键变量是否存在。"""
    cols = set(df.columns)

    for v in [config.FIRM_ID, config.YEAR_VAR]:
        if v not in cols:
            matches = [c for c in cols if c.lower() == v.lower()]
            if matches:
                print(f"[警告] 变量 '{v}' 未找到，发现近似: {matches}")
            else:
                print(f"[警告] 关键变量 '{v}' 未找到！")

    found_x = [v for v in config.X_VARS_PRIMARY if v in cols]
    print(f"[数据加载] 主要解释变量: {len(found_x)}/{len(config.X_VARS_PRIMARY)} 可用")

    for ch, yvars in config.Y_VARS.items():
        found_y = [v for v in yvars if v in cols]
        print(f"[数据加载] 第{ch}章 Y变量: {len(found_y)}/{len(yvars)} 可用")


def apply_sample_filters(
    df: pd.DataFrame,
    start_year: int = 2007,
    end_year: int = 2023,
    filters: dict | None = None,
) -> pd.DataFrame:
    """
    应用样本筛选条件。

    Parameters
    ----------
    df : pd.DataFrame
    start_year, end_year : int
    filters : dict
        键: drop_st_pt, drop_finance, drop_real_estate,
            drop_lev_gt1, drop_ind_lt30, winsorize
    """
    if filters is None:
        filters = {}

    result = df.copy()
    year_col = config.YEAR_VAR

    # 1. 样本区间
    if year_col in result.columns:
        result = result[(result[year_col] >= start_year) & (result[year_col] <= end_year)]

    # 2. 剔除 ST/PT
    if filters.get("drop_st_pt", False):
        n_before = len(result)
        # 尝试ST标识变量
        for st_col in ["ST", "st", "Is_ST", "is_st", "ST_PT", "Trdsta"]:
            if st_col in result.columns:
                result = result[result[st_col] != 1]
                break
        # 尝试通过股票简称
        for name_col in ["Stkname", "ShortName", "股票简称"]:
            if name_col in result.columns:
                mask = result[name_col].astype(str).str.contains(r"ST|PT|\*ST", na=False)
                result = result[~mask]
                break
        _log(f"  剔除ST/PT: {n_before} → {len(result)}")

    # 3. 剔除金融行业
    if filters.get("drop_finance", False):
        n_before = len(result)
        ind_col = _find_ind_col(result)
        if ind_col:
            mask = result[ind_col].astype(str).str.upper().str.startswith("J")
            result = result[~mask]
            _log(f"  剔除金融行业: {n_before} → {len(result)}")

    # 4. 剔除房地产行业
    if filters.get("drop_real_estate", False):
        n_before = len(result)
        ind_col = _find_ind_col(result)
        if ind_col:
            mask = result[ind_col].astype(str).str.upper().str.startswith("K")
            result = result[~mask]
            _log(f"  剔除房地产行业: {n_before} → {len(result)}")

    # 5. 剔除 Lev > 1
    if filters.get("drop_lev_gt1", False):
        if "Lev" in result.columns:
            n_before = len(result)
            result = result[(result["Lev"] <= 1) | result["Lev"].isna()]
            _log(f"  剔除Lev>1: {n_before} → {len(result)}")

    # 6. 剔除行业观测数 < 30
    if filters.get("drop_ind_lt30", False):
        ind_col = _find_ind_col(result)
        if ind_col:
            n_before = len(result)
            ind_counts = result.groupby(ind_col)[config.FIRM_ID].transform("count")
            result = result[ind_counts >= 30]
            _log(f"  剔除行业<30: {n_before} → {len(result)}")

    return result


def winsorize_variables(
    df: pd.DataFrame,
    variables: list[str],
    level: float = 0.01,
) -> pd.DataFrame:
    """对连续变量进行双侧缩尾处理。"""
    result = df.copy()
    for var in variables:
        if var in result.columns:
            col = result[var]
            if pd.api.types.is_numeric_dtype(col):
                valid_mask = col.notna()
                if valid_mask.sum() > 10:
                    vals = col[valid_mask].values.copy()
                    winsorized = mstats.winsorize(vals, limits=[level, level])
                    result.loc[valid_mask, var] = winsorized
    return result


def generate_event_time(
    df: pd.DataFrame,
    treatment_var: str,
    year_col: str | None = None,
    firm_col: str | None = None,
) -> pd.DataFrame:
    """
    根据 DID 处理变量生成事件时间变量。

    处理组: 曾经 treatment_var==1 的企业, event_time = Year - 首次处理年份
    对照组: 从未被处理的企业, event_time = NaN (在回归中自动为0)

    新增列: event_year, event_time, treated_ever
    """
    year_col = year_col or config.YEAR_VAR
    firm_col = firm_col or config.FIRM_ID

    result = df.copy()

    # 清理已有列（避免重复merge）
    for c in ["event_year", "event_time", "treated_ever"]:
        if c in result.columns:
            result = result.drop(columns=[c])

    if treatment_var not in result.columns:
        _log(f"[警告] 处理变量 '{treatment_var}' 不存在")
        result["event_year"] = np.nan
        result["event_time"] = np.nan
        result["treated_ever"] = 0
        return result

    # 找到每个处理组企业首次被处理的年份
    treated_obs = result[result[treatment_var] == 1]
    if len(treated_obs) == 0:
        _log(f"[警告] '{treatment_var}' 无处理组观测")
        result["event_year"] = np.nan
        result["event_time"] = np.nan
        result["treated_ever"] = 0
        return result

    first_treat = (
        treated_obs.groupby(firm_col)[year_col]
        .min()
        .reset_index()
        .rename(columns={year_col: "event_year"})
    )

    result = result.merge(first_treat, on=firm_col, how="left")
    result["treated_ever"] = result["event_year"].notna().astype(int)
    result["event_time"] = result[year_col] - result["event_year"]

    n_treated_firms = first_treat[firm_col].nunique()
    n_control_firms = result.loc[result["treated_ever"] == 0, firm_col].nunique()
    _log(f"[事件时间] 处理组: {n_treated_firms} 家, 对照组: {n_control_firms} 家")

    return result


def get_continuous_vars(df: pd.DataFrame, variables: list[str]) -> list[str]:
    """筛选出连续变量（数值型且非二值变量）。"""
    continuous = []
    for var in variables:
        if var in df.columns and pd.api.types.is_numeric_dtype(df[var]):
            if df[var].dropna().nunique() > 2:
                continuous.append(var)
    return continuous


def _find_ind_col(df: pd.DataFrame) -> str | None:
    """查找行业代码列。"""
    for col in [config.IND_VAR, config.IND_VAR_ALT, "Ind", "Indr", "IndustryCode",
                "ind", "industry", "行业代码"]:
        if col in df.columns:
            return col
    return None


def prepare_panel_data(
    df: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
) -> pd.DataFrame:
    """准备面板回归数据：删除关键变量缺失值。"""
    all_vars = list(dict.fromkeys(
        [y_var, x_var] + controls + fe_vars + cluster_vars
    ))
    available = [v for v in all_vars if v in df.columns]
    return df[available].dropna()
