"""
============================================================
回归引擎模块
============================================================
核心回归函数：OLS/FE、PSM-DID、平行趋势检验、描述统计
支持 linearmodels.PanelOLS（快速吸收FE）和 statsmodels OLS 降级方案
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ============================================================
# 数据容器
# ============================================================

@dataclass
class RegressionResult:
    """回归结果"""
    y_var: str = ""
    x_var: str = ""
    controls: list = field(default_factory=list)
    fe_vars: list = field(default_factory=list)
    cluster_vars: list = field(default_factory=list)
    coef: float = np.nan
    se: float = np.nan
    tstat: float = np.nan
    pval: float = np.nan
    nobs: int = 0
    r2: float = np.nan
    r2_within: float = np.nan
    f_stat: float = np.nan
    all_coefs: dict = field(default_factory=dict)
    all_se: dict = field(default_factory=dict)
    all_pval: dict = field(default_factory=dict)
    success: bool = False
    error_msg: str = ""
    stars: str = ""

    def __post_init__(self):
        if not np.isnan(self.pval):
            if self.pval < 0.01:
                self.stars = "***"
            elif self.pval < 0.05:
                self.stars = "**"
            elif self.pval < 0.10:
                self.stars = "*"
            else:
                self.stars = ""


@dataclass
class PTResult:
    """平行趋势检验结果"""
    y_var: str = ""
    x_var: str = ""
    base_period: str = "pre1"
    pre_window: int = -3
    post_window: int = 3
    period_coefs: dict = field(default_factory=dict)
    period_se: dict = field(default_factory=dict)
    period_pval: dict = field(default_factory=dict)
    period_ci_lower: dict = field(default_factory=dict)
    period_ci_upper: dict = field(default_factory=dict)
    nobs: int = 0
    r2: float = np.nan
    success: bool = False
    error_msg: str = ""
    n_pre_sig: int = 0
    n_post_sig: int = 0
    max_post_consecutive: int = 0
    post_correct_sign: bool = False


@dataclass
class PSMResult:
    """PSM-DID 结果"""
    psm_method: str = ""
    n_treated: int = 0
    n_control: int = 0
    balance_before: pd.DataFrame = field(default_factory=pd.DataFrame)
    balance_after: pd.DataFrame = field(default_factory=pd.DataFrame)
    baseline_result: RegressionResult = field(default_factory=RegressionResult)
    psm_result: RegressionResult = field(default_factory=RegressionResult)
    success: bool = False


# ============================================================
# 固定效应回归
# ============================================================

def run_ols_fe(
    df: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
) -> RegressionResult:
    """
    运行固定效应面板回归。
    优先 linearmodels.PanelOLS，失败时降级到 reghdfe 风格的组内变换，
    最终降级到 statsmodels + 哑变量。
    """
    from . import config

    result = RegressionResult(
        y_var=y_var, x_var=x_var, controls=list(controls),
        fe_vars=list(fe_vars), cluster_vars=list(cluster_vars),
    )

    try:
        # 准备数据
        all_needed = list(dict.fromkeys(
            [y_var, x_var] + controls + fe_vars + cluster_vars
        ))
        available = [v for v in all_needed if v in df.columns]
        missing = set(all_needed) - set(available)
        if y_var in missing or x_var in missing:
            result.error_msg = f"关键变量缺失: {missing}"
            return result

        # 仅保留有效的控制变量
        valid_controls = [c for c in controls if c in df.columns]

        panel = df[available].dropna().copy()
        if len(panel) < 50:
            result.error_msg = f"样本量不足: {len(panel)}"
            return result

        # 尝试 PanelOLS
        firm_col = config.FIRM_ID
        year_col = config.YEAR_VAR

        if firm_col in panel.columns and year_col in panel.columns:
            try:
                return _run_panel_ols(
                    panel, y_var, x_var, valid_controls, fe_vars,
                    cluster_vars, result, firm_col, year_col
                )
            except Exception:
                pass

        # 降级到 OLS + 哑变量
        return _run_ols_dummies(
            panel, y_var, x_var, valid_controls, fe_vars, cluster_vars, result
        )

    except Exception as e:
        result.error_msg = str(e)
        return result


def _run_panel_ols(
    panel: pd.DataFrame,
    y_var: str, x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
    result: RegressionResult,
    firm_col: str, year_col: str,
) -> RegressionResult:
    """使用 linearmodels.PanelOLS（高效 FE 吸收）。"""
    from linearmodels.panel import PanelOLS
    from . import config as cfg

    panel_indexed = panel.set_index([firm_col, year_col])

    y = panel_indexed[y_var]
    exog_cols = [x_var] + [c for c in controls
                           if c in panel_indexed.columns and c not in [firm_col, year_col]]
    X = panel_indexed[exog_cols].copy()

    # 检查 X 是否全为常数或存在共线性
    if X[x_var].nunique() < 2:
        raise ValueError(f"{x_var} 无变异")

    entity_effects = firm_col in fe_vars
    time_effects = year_col in fe_vars

    # 行业 FE 通过哑变量
    ind_col = None
    for ic in [cfg.IND_VAR, cfg.IND_VAR_ALT]:
        if ic in fe_vars and ic in panel_indexed.columns:
            ind_col = ic
            break

    if ind_col:
        ind_dummies = pd.get_dummies(
            panel_indexed[ind_col], prefix="ind", drop_first=True, dtype=float
        )
        X = pd.concat([X, ind_dummies], axis=1)

    mod = PanelOLS(
        y, X,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
        check_rank=False,
    )

    # 聚类标准误
    if len(cluster_vars) >= 2 and firm_col in cluster_vars and year_col in cluster_vars:
        fit = mod.fit(cov_type="clustered", cluster_entity=True, cluster_time=True)
    elif len(cluster_vars) >= 1 and cluster_vars[0] == firm_col:
        fit = mod.fit(cov_type="clustered", cluster_entity=True)
    else:
        fit = mod.fit(cov_type="clustered", cluster_entity=True)

    if x_var not in fit.params.index:
        result.error_msg = f"{x_var} 被吸收"
        return result

    result.coef = float(fit.params[x_var])
    result.se = float(fit.std_errors[x_var])
    result.tstat = float(fit.tstats[x_var])
    result.pval = float(fit.pvalues[x_var])
    result.nobs = int(fit.nobs)
    result.r2 = float(fit.rsquared)
    result.r2_within = float(getattr(fit, "rsquared_within", fit.rsquared))
    try:
        result.f_stat = float(fit.f_statistic.stat)
    except Exception:
        pass

    for var in fit.params.index:
        if not var.startswith("ind_"):
            result.all_coefs[var] = float(fit.params[var])
            result.all_se[var] = float(fit.std_errors[var])
            result.all_pval[var] = float(fit.pvalues[var])

    result.success = True
    result.__post_init__()
    return result


def _run_ols_dummies(
    panel: pd.DataFrame,
    y_var: str, x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
    result: RegressionResult,
) -> RegressionResult:
    """使用 statsmodels OLS + 哑变量（降级方案）。"""
    import statsmodels.api as sm

    y = panel[y_var]
    exog_vars = [x_var] + [c for c in controls if c in panel.columns]
    X = panel[exog_vars].copy()

    # 限制哑变量数量——如果FE变量类别>500则跳过以避免内存问题
    for fe_var in fe_vars:
        if fe_var in panel.columns:
            n_cats = panel[fe_var].nunique()
            if n_cats > 500:
                continue  # 跳过过大的FE
            dummies = pd.get_dummies(panel[fe_var], prefix=fe_var, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    cluster_col = cluster_vars[0] if cluster_vars and cluster_vars[0] in panel.columns else None

    mod = sm.OLS(y, X, missing="drop")
    if cluster_col is not None:
        fit = mod.fit(cov_type="cluster", cov_kwds={"groups": panel[cluster_col]})
    else:
        fit = mod.fit(cov_type="HC1")

    if x_var not in fit.params.index:
        result.error_msg = f"{x_var} 不在结果中"
        return result

    result.coef = float(fit.params[x_var])
    result.se = float(fit.bse[x_var])
    result.tstat = float(fit.tvalues[x_var])
    result.pval = float(fit.pvalues[x_var])
    result.nobs = int(fit.nobs)
    result.r2 = float(fit.rsquared)
    result.r2_within = float(fit.rsquared_adj)
    result.f_stat = float(fit.fvalue) if fit.fvalue is not None else np.nan

    for var in [x_var] + exog_vars:
        if var in fit.params.index:
            result.all_coefs[var] = float(fit.params[var])
            result.all_se[var] = float(fit.bse[var])
            result.all_pval[var] = float(fit.pvalues[var])

    result.success = True
    result.__post_init__()
    return result


# ============================================================
# 平行趋势检验（事件研究法）
# ============================================================

def run_parallel_trends(
    df: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
    pre_window: int = -3,
    post_window: int = 3,
    base_period: str = "pre1",
    chapter: int = 3,
) -> PTResult:
    """
    事件研究法 / 动态效应估计。

    base_period:
        "pre1" → 以 t=-1 为基期
        "pre0" → 以 t=0 为基期
        "pre_biggest" → 以最远事前期为基期
    """
    from . import config as cfg

    pt = PTResult(
        y_var=y_var, x_var=x_var,
        base_period=base_period,
        pre_window=pre_window,
        post_window=post_window,
    )

    try:
        firm_col = cfg.FIRM_ID
        year_col = cfg.YEAR_VAR

        if x_var not in df.columns or y_var not in df.columns:
            pt.error_msg = f"变量缺失"
            return pt

        data = df.copy()

        # 生成事件时间（如果尚未生成）
        if "event_time" not in data.columns:
            from .data_loader import generate_event_time
            data = generate_event_time(data, x_var, year_col, firm_col)

        if "event_time" not in data.columns or data["event_time"].isna().all():
            pt.error_msg = "无法生成事件时间"
            return pt

        # 基期确定
        if base_period == "pre1":
            omit_period = -1
        elif base_period == "pre0":
            omit_period = 0
        elif base_period == "pre_biggest":
            omit_period = pre_window
        else:
            omit_period = -1

        # 生成事件时间哑变量
        # 对于处理组: event_time == p → 1
        # 对于对照组(event_year is NaN): 始终为 0
        periods_to_include = [p for p in range(pre_window, post_window + 1) if p != omit_period]

        dummy_names = {}
        for p in periods_to_include:
            if p < 0:
                name = f"_et_pre{abs(p)}"
            elif p == 0:
                name = "_et_current"
            else:
                name = f"_et_post{p}"
            dummy_names[p] = name
            is_treated = data.get("treated_ever", data["event_year"].notna().astype(int)) == 1
            data[name] = ((data["event_time"] == p) & is_treated).astype(float)

        # 准备回归
        event_dummy_cols = [dummy_names[p] for p in periods_to_include]
        valid_controls = [c for c in controls if c in data.columns]
        all_exog = event_dummy_cols + valid_controls

        reg_vars = list(dict.fromkeys(
            [y_var] + all_exog + fe_vars + cluster_vars
        ))
        available = [v for v in reg_vars if v in data.columns]
        panel = data[available].dropna()

        if len(panel) < 50:
            pt.error_msg = f"样本量不足: {len(panel)}"
            return pt

        # 使用 PanelOLS 或 OLS
        try:
            pt = _run_pt_panel_ols(
                panel, y_var, event_dummy_cols, valid_controls,
                fe_vars, cluster_vars, periods_to_include,
                omit_period, chapter, pt, firm_col, year_col
            )
        except Exception:
            pt = _run_pt_ols_dummies(
                panel, y_var, event_dummy_cols, valid_controls,
                fe_vars, cluster_vars, periods_to_include,
                omit_period, chapter, pt
            )

    except Exception as e:
        pt.error_msg = str(e)

    return pt


def _run_pt_panel_ols(
    panel, y_var, event_dummy_cols, controls, fe_vars, cluster_vars,
    periods, omit_period, chapter, pt, firm_col, year_col,
):
    """平行趋势的 PanelOLS 实现。"""
    from linearmodels.panel import PanelOLS
    from . import config as cfg

    panel_idx = panel.set_index([firm_col, year_col])
    y = panel_idx[y_var]

    exog_cols = [c for c in event_dummy_cols + controls
                 if c in panel_idx.columns and c not in [firm_col, year_col]]
    X = panel_idx[exog_cols].copy()

    entity_effects = firm_col in fe_vars
    time_effects = year_col in fe_vars

    mod = PanelOLS(y, X, entity_effects=entity_effects, time_effects=time_effects,
                   drop_absorbed=True, check_rank=False)
    fit = mod.fit(cov_type="clustered", cluster_entity=True)

    return _extract_pt_results(
        fit, event_dummy_cols, periods, omit_period, chapter, pt
    )


def _run_pt_ols_dummies(
    panel, y_var, event_dummy_cols, controls, fe_vars, cluster_vars,
    periods, omit_period, chapter, pt,
):
    """平行趋势的 OLS+哑变量 实现。"""
    import statsmodels.api as sm

    y = panel[y_var]
    exog = [c for c in event_dummy_cols + controls if c in panel.columns]
    X = panel[exog].copy()

    for fe_var in fe_vars:
        if fe_var in panel.columns and panel[fe_var].nunique() <= 500:
            dummies = pd.get_dummies(panel[fe_var], prefix=fe_var, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)
    cluster_col = cluster_vars[0] if cluster_vars and cluster_vars[0] in panel.columns else None

    mod = sm.OLS(y, X, missing="drop")
    if cluster_col:
        fit = mod.fit(cov_type="cluster", cov_kwds={"groups": panel[cluster_col]})
    else:
        fit = mod.fit(cov_type="HC1")

    return _extract_pt_results(
        fit, event_dummy_cols, periods, omit_period, chapter, pt
    )


def _extract_pt_results(fit, event_dummy_cols, periods, omit_period, chapter, pt):
    """从回归结果中提取平行趋势系数。"""
    from . import config as cfg

    pt.r2 = float(getattr(fit, "rsquared", np.nan))
    pt.nobs = int(getattr(fit, "nobs", 0))

    expected_sign = cfg.EXPECTED_SIGN.get(chapter, -1)
    n_pre_sig = 0
    n_post_sig = 0
    post_sig_consecutive = 0
    max_post_consecutive = 0
    post_signs_correct = True

    for p, dummy_name in zip(periods, event_dummy_cols):
        if dummy_name in fit.params.index:
            coef = float(fit.params[dummy_name])
            se_val = float(fit.std_errors[dummy_name]) if hasattr(fit, 'std_errors') else float(fit.bse[dummy_name])
            pv = float(fit.pvalues[dummy_name])
            try:
                ci = fit.conf_int()
                if dummy_name in ci.index:
                    ci_row = ci.loc[dummy_name]
                    ci_lo = float(ci_row.iloc[0])
                    ci_hi = float(ci_row.iloc[1])
                else:
                    ci_lo = coef - 1.96 * se_val
                    ci_hi = coef + 1.96 * se_val
            except Exception:
                ci_lo = coef - 1.96 * se_val
                ci_hi = coef + 1.96 * se_val

            pt.period_coefs[p] = coef
            pt.period_se[p] = se_val
            pt.period_pval[p] = pv
            pt.period_ci_lower[p] = ci_lo
            pt.period_ci_upper[p] = ci_hi

            is_sig = pv < 0.10
            if p < 0:
                if is_sig:
                    n_pre_sig += 1
            elif p > 0:
                if is_sig:
                    n_post_sig += 1
                    post_sig_consecutive += 1
                    max_post_consecutive = max(max_post_consecutive, post_sig_consecutive)
                    if expected_sign < 0 and coef > 0:
                        post_signs_correct = False
                    if expected_sign > 0 and coef < 0:
                        post_signs_correct = False
                else:
                    post_sig_consecutive = 0

    # 基期
    pt.period_coefs[omit_period] = 0.0
    pt.period_se[omit_period] = 0.0
    pt.period_pval[omit_period] = 1.0
    pt.period_ci_lower[omit_period] = 0.0
    pt.period_ci_upper[omit_period] = 0.0

    pt.n_pre_sig = n_pre_sig
    pt.n_post_sig = n_post_sig
    pt.max_post_consecutive = max_post_consecutive
    pt.post_correct_sign = post_signs_correct
    pt.success = True

    return pt


# ============================================================
# PSM-DID
# ============================================================

def run_psm_did(
    df: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
    psm_config: dict | None = None,
) -> PSMResult:
    """
    PSM-DID 回归:
    1. Logit 估计倾向得分
    2. 最近邻匹配
    3. 平衡性检验
    4. 匹配后 DID 回归
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.spatial import KDTree
    from . import config as cfg

    psm = PSMResult()
    if psm_config is None:
        psm_config = {"method": "nearest", "n_neighbors": 1, "caliper": None}

    psm.psm_method = (
        f"{psm_config['method']}_k{psm_config.get('n_neighbors', '')}"
        f"_c{psm_config.get('caliper', '')}"
    )

    try:
        firm_col = cfg.FIRM_ID

        # 基准回归（全样本）
        psm.baseline_result = run_ols_fe(df, y_var, x_var, controls, fe_vars, cluster_vars)

        if x_var not in df.columns:
            return psm

        # 企业层面处理标识
        treat_firm = df.groupby(firm_col)[x_var].max().reset_index()
        treat_firm.columns = [firm_col, "treated"]

        available_controls = [c for c in controls if c in df.columns]
        if not available_controls:
            return psm

        firm_chars = df.groupby(firm_col)[available_controls].mean().reset_index()
        psm_data = firm_chars.merge(treat_firm, on=firm_col, how="inner").dropna()

        n_treated = int(psm_data["treated"].sum())
        n_control = int((psm_data["treated"] == 0).sum())
        if n_treated < 10 or n_control < 10:
            return psm

        # Logit 倾向得分
        X_psm = psm_data[available_controls].values
        y_psm = psm_data["treated"].values.astype(int)

        logit = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
        logit.fit(X_psm, y_psm)
        psm_data = psm_data.copy()
        psm_data["pscore"] = logit.predict_proba(X_psm)[:, 1]

        # 匹配
        treated_mask = psm_data["treated"] == 1
        control_mask = psm_data["treated"] == 0

        treated_scores = psm_data.loc[treated_mask, "pscore"].values.reshape(-1, 1)
        control_scores = psm_data.loc[control_mask, "pscore"].values.reshape(-1, 1)
        control_firms = psm_data.loc[control_mask, firm_col].values

        tree = KDTree(control_scores)
        n_neighbors = psm_config.get("n_neighbors") or 1
        k = min(n_neighbors, len(control_scores))
        distances, indices = tree.query(treated_scores, k=k)

        if distances.ndim == 1:
            distances = distances.reshape(-1, 1)
            indices = indices.reshape(-1, 1)

        caliper = psm_config.get("caliper")
        matched_control_ids = set()
        for i in range(len(treated_scores)):
            for j in range(distances.shape[1]):
                if caliper is None or distances[i, j] <= caliper:
                    matched_control_ids.add(control_firms[indices[i, j]])

        treated_firms = set(psm_data.loc[treated_mask, firm_col].values)
        psm.n_treated = len(treated_firms)
        psm.n_control = len(matched_control_ids)

        # 平衡性检验
        psm.balance_before = _balance_test(psm_data, available_controls, "treated")
        matched_firms = treated_firms | matched_control_ids
        matched_psm = psm_data[psm_data[firm_col].isin(matched_firms)]
        psm.balance_after = _balance_test(matched_psm, available_controls, "treated")

        # 匹配后回归
        matched_panel = df[df[firm_col].isin(matched_firms)].copy()
        psm.psm_result = run_ols_fe(
            matched_panel, y_var, x_var, controls, fe_vars, cluster_vars
        )
        psm.success = psm.psm_result.success

    except Exception as e:
        psm.baseline_result.error_msg = str(e)

    return psm


def _balance_test(data: pd.DataFrame, covariates: list[str], treat_col: str) -> pd.DataFrame:
    """PSM 平衡性检验。"""
    from scipy.stats import ttest_ind

    results = []
    for var in covariates:
        if var not in data.columns:
            continue
        treated = data.loc[data[treat_col] == 1, var].dropna()
        control = data.loc[data[treat_col] == 0, var].dropna()
        if len(treated) < 2 or len(control) < 2:
            continue

        mean_t, mean_c = treated.mean(), control.mean()
        std_t, std_c = treated.std(), control.std()
        pooled_std = np.sqrt((std_t**2 + std_c**2) / 2)
        sbias = (mean_t - mean_c) / pooled_std * 100 if pooled_std > 0 else 0
        t_stat, p_val = ttest_ind(treated, control, equal_var=False)

        results.append({
            "Variable": var,
            "Mean_Treated": round(mean_t, 4),
            "Mean_Control": round(mean_c, 4),
            "Std_Bias(%)": round(sbias, 2),
            "t_stat": round(t_stat, 3),
            "p_value": round(p_val, 3),
        })

    return pd.DataFrame(results)


# ============================================================
# 描述统计 & 相关系数
# ============================================================

def compute_descriptive_stats(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """计算描述统计量: N, Mean, Std, Min, P25, Median, P75, Max。"""
    stats = []
    for var in variables:
        if var not in df.columns:
            continue
        col = df[var].dropna()
        if len(col) == 0:
            continue
        stats.append({
            "Variable": var,
            "N": len(col),
            "Mean": col.mean(),
            "Std": col.std(),
            "Min": col.min(),
            "P25": col.quantile(0.25),
            "Median": col.median(),
            "P75": col.quantile(0.75),
            "Max": col.max(),
        })
    return pd.DataFrame(stats)


def compute_correlation_matrix(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """计算 Pearson 相关系数矩阵（带显著性星号）。"""
    from scipy.stats import pearsonr

    available = [v for v in variables if v in df.columns]
    data = df[available].dropna()
    n = len(available)

    corr = pd.DataFrame(np.eye(n), index=available, columns=available)
    pval = pd.DataFrame(np.zeros((n, n)), index=available, columns=available)

    for i in range(n):
        for j in range(i + 1, n):
            valid = data[[available[i], available[j]]].dropna()
            if len(valid) > 2:
                r, p = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
                corr.iloc[i, j] = r
                corr.iloc[j, i] = r
                pval.iloc[i, j] = p
                pval.iloc[j, i] = p

    formatted = corr.copy().astype(str)
    for i in range(n):
        for j in range(n):
            if i != j:
                r = corr.iloc[i, j]
                p = pval.iloc[i, j]
                stars = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
                formatted.iloc[i, j] = f"{r:.3f}{stars}"
            else:
                formatted.iloc[i, j] = "1.000"

    return formatted
