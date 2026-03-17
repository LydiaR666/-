"""
============================================================
数据加载与预处理模块
============================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import mstats
from . import config


def load_data(path: str | None = None) -> pd.DataFrame:
    """
    读取 Stata .dta 数据文件。

    Parameters
    ----------
    path : str, optional
        数据文件路径，默认使用 config.DATA_PATH

    Returns
    -------
    pd.DataFrame
    """
    path = path or config.DATA_PATH
    print(f"[数据加载] 正在读取: {path}")
    df = pd.read_stata(path, convert_categoricals=False)
    print(f"[数据加载] 原始样本量: {len(df)} 行 × {len(df.columns)} 列")

    # 统一列名大小写处理 —— 保留原始列名
    # 检测关键变量是否存在
    _check_key_variables(df)

    return df


def _check_key_variables(df: pd.DataFrame) -> None:
    """检查关键变量是否存在，并打印缺失变量警告。"""
    cols = set(df.columns)

    # 面板标识
    for v in [config.FIRM_ID, config.YEAR_VAR]:
        if v not in cols:
            # 尝试大小写模糊匹配
            matches = [c for c in cols if c.lower() == v.lower()]
            if matches:
                print(f"[警告] 变量 '{v}' 未找到，但发现近似匹配: {matches}")
            else:
                print(f"[警告] 关键变量 '{v}' 未找到！")

    # 解释变量
    found_x = [v for v in config.X_VARS_PRIMARY if v in cols]
    print(f"[数据加载] 找到 {len(found_x)}/{len(config.X_VARS_PRIMARY)} 个主要解释变量")

    # 被解释变量
    for ch, yvars in config.Y_VARS.items():
        found_y = [v for v in yvars if v in cols]
        print(f"[数据加载] 第{ch}章: 找到 {len(found_y)}/{len(yvars)} 个被解释变量")


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
    start_year : int
        样本起始年份
    end_year : int
        样本结束年份
    filters : dict
        筛选开关字典，键包括:
        - drop_st_pt, drop_finance, drop_real_estate,
          drop_lev_gt1, drop_ind_lt30, winsorize

    Returns
    -------
    pd.DataFrame
        筛选后的数据
    """
    if filters is None:
        filters = {}

    n0 = len(df)
    result = df.copy()

    # 1. 样本区间
    year_col = config.YEAR_VAR
    if year_col in result.columns:
        result = result[(result[year_col] >= start_year) & (result[year_col] <= end_year)]
        print(f"  样本区间 [{start_year}, {end_year}]: {n0} → {len(result)}")

    # 2. 剔除 ST / PT
    if filters.get("drop_st_pt", False):
        n_before = len(result)
        # 尝试多种可能的ST标识列
        for st_col in ["ST", "st", "Is_ST", "is_st", "ST_PT"]:
            if st_col in result.columns:
                result = result[result[st_col] == 0]
                break
        # 也尝试通过股票名称筛选
        for name_col in ["Stkname", "ShortName", "股票简称"]:
            if name_col in result.columns:
                mask = result[name_col].astype(str).str.contains(r"ST|PT|\*ST", na=False)
                result = result[~mask]
                break
        if len(result) < n_before:
            print(f"  剔除ST/PT: {n_before} → {len(result)}")

    # 3. 剔除金融行业
    if filters.get("drop_finance", False):
        n_before = len(result)
        ind_col = _find_ind_col(result)
        if ind_col:
            # 证监会行业分类中金融业代码通常以 J 开头
            mask = result[ind_col].astype(str).str.startswith("J")
            result = result[~mask]
            if len(result) < n_before:
                print(f"  剔除金融行业: {n_before} → {len(result)}")

    # 4. 剔除房地产行业
    if filters.get("drop_real_estate", False):
        n_before = len(result)
        ind_col = _find_ind_col(result)
        if ind_col:
            mask = result[ind_col].astype(str).str.startswith("K")
            result = result[~mask]
            if len(result) < n_before:
                print(f"  剔除房地产行业: {n_before} → {len(result)}")

    # 5. 剔除 Lev > 1
    if filters.get("drop_lev_gt1", False):
        if "Lev" in result.columns:
            n_before = len(result)
            result = result[result["Lev"] <= 1]
            print(f"  剔除Lev>1: {n_before} → {len(result)}")

    # 6. 剔除行业观测数 < 30
    if filters.get("drop_ind_lt30", False):
        ind_col = _find_ind_col(result)
        if ind_col:
            n_before = len(result)
            ind_counts = result.groupby(ind_col)[config.FIRM_ID].transform("count")
            result = result[ind_counts >= 30]
            print(f"  剔除行业观测<30: {n_before} → {len(result)}")

    print(f"  最终样本量: {len(result)}")
    return result


def winsorize_variables(
    df: pd.DataFrame,
    variables: list[str],
    level: float = 0.01,
) -> pd.DataFrame:
    """
    对连续变量进行双侧缩尾处理。

    Parameters
    ----------
    df : pd.DataFrame
    variables : list[str]
        需要缩尾的变量列表
    level : float
        缩尾比例（单侧），如 0.01 表示上下各1%

    Returns
    -------
    pd.DataFrame
    """
    result = df.copy()
    for var in variables:
        if var in result.columns:
            col = result[var]
            if col.dtype in [np.float64, np.float32, np.int64, np.int32, float, int]:
                numeric = col.dropna()
                if len(numeric) > 0:
                    result[var] = mstats.winsorize(col, limits=[level, level])
    return result


def generate_event_time(
    df: pd.DataFrame,
    treatment_var: str,
    year_col: str | None = None,
    firm_col: str | None = None,
) -> pd.DataFrame:
    """
    根据 DID 处理变量生成事件时间变量。

    对于每个企业，找到处理变量从 0 变为 1 的首次年份作为事件年份，
    然后计算 event_time = Year - event_year。

    Parameters
    ----------
    df : pd.DataFrame
    treatment_var : str
        DID 处理变量（0/1）
    year_col : str
        年份变量名
    firm_col : str
        个体标识变量名

    Returns
    -------
    pd.DataFrame
        新增 'event_year' 和 'event_time' 列
    """
    year_col = year_col or config.YEAR_VAR
    firm_col = firm_col or config.FIRM_ID

    result = df.copy()

    if treatment_var not in result.columns:
        print(f"[警告] 处理变量 '{treatment_var}' 不在数据中")
        result["event_year"] = np.nan
        result["event_time"] = np.nan
        return result

    # 找到每个企业首次处理年份
    treated = result[result[treatment_var] == 1].copy()
    if len(treated) == 0:
        print(f"[警告] 处理变量 '{treatment_var}' 无处理组观测")
        result["event_year"] = np.nan
        result["event_time"] = np.nan
        return result

    first_treat = (
        treated.groupby(firm_col)[year_col]
        .min()
        .reset_index()
        .rename(columns={year_col: "event_year"})
    )

    result = result.merge(first_treat, on=firm_col, how="left")
    result["event_time"] = result[year_col] - result["event_year"]

    # 对于从未被处理的企业, event_time 设为 NaN（对照组）
    n_treated = result["event_year"].notna().sum()
    n_control = result["event_year"].isna().sum()
    print(f"[事件时间] 处理组观测: {n_treated}, 对照组观测: {n_control}")

    return result


def get_continuous_vars(df: pd.DataFrame, variables: list[str]) -> list[str]:
    """
    从变量列表中筛选出连续变量（数值型且不是虚拟变量）。
    """
    continuous = []
    for var in variables:
        if var in df.columns:
            col = df[var].dropna()
            if len(col) > 0 and col.dtype in [np.float64, np.float32, np.int64, np.int32]:
                nunique = col.nunique()
                if nunique > 2:  # 非虚拟变量
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
    """
    准备面板回归数据：删除关键变量缺失值，确保所有变量可用。

    Returns
    -------
    pd.DataFrame
        仅包含完整观测的子集
    """
    all_vars = [y_var, x_var] + controls + fe_vars + cluster_vars
    # 去重
    all_vars = list(dict.fromkeys(all_vars))

    # 检查哪些变量存在
    available = [v for v in all_vars if v in df.columns]
    missing = [v for v in all_vars if v not in df.columns]
    if missing:
        print(f"[警告] 以下变量不在数据中: {missing}")

    # 仅保留存在的变量，并删除缺失值
    subset = df[available].dropna()
    return subset
