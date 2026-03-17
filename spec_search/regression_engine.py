"""
============================================================
回归引擎模块
============================================================
核心回归函数：OLS/FE、PSM-DID、平行趋势检验、描述统计
"""

import warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


@dataclass
class RegressionResult:
    """回归结果容器"""
    y_var: str = ""
    x_var: str = ""
    controls: list = field(default_factory=list)
    fe_vars: list = field(default_factory=list)
    cluster_vars: list = field(default_factory=list)
    coef: float = np.nan            # X 的系数
    se: float = np.nan              # 标准误
    tstat: float = np.nan           # t 统计量
    pval: float = np.nan            # p 值
    nobs: int = 0                   # 观测数
    r2: float = np.nan              # R²
    r2_within: float = np.nan       # 组内 R²
    f_stat: float = np.nan          # F 统计量
    all_coefs: dict = field(default_factory=dict)  # 所有系数
    all_se: dict = field(default_factory=dict)
    all_pval: dict = field(default_factory=dict)
    success: bool = False
    error_msg: str = ""
    stars: str = ""                 # 显著性星号

    def __post_init__(self):
        if not np.isnan(self.pval):
            if self.pval < 0.01:
                self.stars = "***"
            elif self.pval < 0.05:
                self.stars = "**"
            elif self.pval < 0.10:
                self.stars = "*"


@dataclass
class PTResult:
    """平行趋势检验结果"""
    y_var: str = ""
    x_var: str = ""
    base_period: str = "pre1"
    pre_window: int = -3
    post_window: int = 3
    # 各期系数、标准误、p值
    period_coefs: dict = field(default_factory=dict)   # {-3: coef, -2: coef, ...}
    period_se: dict = field(default_factory=dict)
    period_pval: dict = field(default_factory=dict)
    period_ci_lower: dict = field(default_factory=dict)
    period_ci_upper: dict = field(default_factory=dict)
    nobs: int = 0
    r2: float = np.nan
    success: bool = False
    error_msg: str = ""
    # 评估指标
    n_pre_sig: int = 0        # 事前显著期数
    n_post_sig: int = 0       # 事后显著期数
    max_post_consecutive: int = 0  # 事后最大连续显著期数
    post_correct_sign: bool = False  # 事后系数方向是否正确


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


def run_ols_fe(
    df: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
) -> RegressionResult:
    """
    运行固定效应面板回归 (absorbing FE via demeaning)。

    使用 linearmodels.PanelOLS 或降级到 statsmodels OLS + dummies。

    Parameters
    ----------
    df : pd.DataFrame
    y_var, x_var : str
    controls : list[str]
    fe_vars : list[str]
        固定效应变量（如 ['Stkcd', 'Year']）
    cluster_vars : list[str]
        聚类变量

    Returns
    -------
    RegressionResult
    """
    from . import config

    result = RegressionResult(
        y_var=y_var, x_var=x_var, controls=controls,
        fe_vars=fe_vars, cluster_vars=cluster_vars,
    )

    try:
        # 准备变量
        all_needed = [y_var, x_var] + controls + fe_vars + cluster_vars
        all_needed = list(dict.fromkeys(all_needed))
        available = [v for v in all_needed if v in df.columns]
        missing = set(all_needed) - set(available)
        if missing:
            result.error_msg = f"缺失变量: {missing}"
            return result

        panel = df[available].dropna().copy()
        if len(panel) < 50:
            result.error_msg = f"样本量不足: {len(panel)}"
            return result

        result.nobs = len(panel)

        # 尝试使用 linearmodels.PanelOLS
        try:
            return _run_panel_ols(panel, y_var, x_var, controls, fe_vars, cluster_vars, result)
        except Exception:
            # 降级到 statsmodels 哑变量方法
            return _run_ols_dummies(panel, y_var, x_var, controls, fe_vars, cluster_vars, result)

    except Exception as e:
        result.error_msg = str(e)
        return result


def _run_panel_ols(
    panel: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
    result: RegressionResult,
) -> RegressionResult:
    """使用 linearmodels.PanelOLS 运行面板回归。"""
    from linearmodels.panel import PanelOLS
    from . import config

    firm_col = config.FIRM_ID
    year_col = config.YEAR_VAR

    # 设置多级索引
    if firm_col in panel.columns and year_col in panel.columns:
        panel = panel.set_index([firm_col, year_col])
    else:
        raise ValueError("面板标识变量不可用，降级到 OLS")

    y = panel[y_var]
    exog_vars = [x_var] + [c for c in controls if c in panel.columns
                            and c not in [firm_col, year_col]]
    X = panel[exog_vars]

    # 确定固定效应
    # EntityEffects = Stkcd FE, TimeEffects = Year FE
    entity_effects = firm_col in fe_vars
    time_effects = year_col in fe_vars

    # 行业固定效应需要通过哑变量加入
    from . import config as cfg
    ind_col = cfg.IND_VAR if cfg.IND_VAR in panel.columns else (
        cfg.IND_VAR_ALT if cfg.IND_VAR_ALT in panel.columns else None
    )
    if ind_col and ind_col in fe_vars and ind_col in panel.columns:
        ind_dummies = pd.get_dummies(panel[ind_col], prefix="ind", drop_first=True, dtype=float)
        X = pd.concat([X, ind_dummies], axis=1)

    # 构建模型
    mod = PanelOLS(
        y, X,
        entity_effects=entity_effects,
        time_effects=time_effects,
        drop_absorbed=True,
        check_rank=False,
    )

    # 确定聚类
    if len(cluster_vars) == 1 and cluster_vars[0] == firm_col:
        cov_type = "clustered"
        cov_kwds = {"cluster_entity": True}
    elif len(cluster_vars) == 1 and cluster_vars[0] in [cfg.IND_VAR, cfg.IND_VAR_ALT]:
        # 行业聚类需要额外处理
        cov_type = "clustered"
        cov_kwds = {"cluster_entity": True}  # 近似
    elif len(cluster_vars) == 2:
        cov_type = "clustered"
        cov_kwds = {"cluster_entity": True, "cluster_time": True}
    else:
        cov_type = "clustered"
        cov_kwds = {"cluster_entity": True}

    fit = mod.fit(cov_type=cov_type, **cov_kwds)

    # 提取结果
    if x_var in fit.params.index:
        result.coef = fit.params[x_var]
        result.se = fit.std_errors[x_var]
        result.tstat = fit.tstats[x_var]
        result.pval = fit.pvalues[x_var]
    else:
        result.error_msg = f"解释变量 {x_var} 被吸收或不在回归中"
        return result

    result.nobs = int(fit.nobs)
    result.r2 = fit.rsquared
    result.r2_within = fit.rsquared_within if hasattr(fit, 'rsquared_within') else fit.rsquared
    try:
        result.f_stat = fit.f_statistic.stat
    except Exception:
        pass

    # 所有系数
    for var in fit.params.index:
        result.all_coefs[var] = fit.params[var]
        result.all_se[var] = fit.std_errors[var]
        result.all_pval[var] = fit.pvalues[var]

    result.success = True
    result.__post_init__()
    return result


def _run_ols_dummies(
    panel: pd.DataFrame,
    y_var: str,
    x_var: str,
    controls: list[str],
    fe_vars: list[str],
    cluster_vars: list[str],
    result: RegressionResult,
) -> RegressionResult:
    """使用 statsmodels OLS + 哑变量运行回归（降级方案）。"""
    import statsmodels.api as sm
    from . import config

    y = panel[y_var]
    exog_vars = [x_var] + [c for c in controls if c in panel.columns]

    X = panel[exog_vars].copy()

    # 添加固定效应哑变量
    for fe_var in fe_vars:
        if fe_var in panel.columns:
            dummies = pd.get_dummies(panel[fe_var], prefix=fe_var, drop_first=True, dtype=float)
            X = pd.concat([X, dummies], axis=1)

    X = sm.add_constant(X)

    # 聚类标准误
    cluster_col = cluster_vars[0] if cluster_vars and cluster_vars[0] in panel.columns else None

    try:
        mod = sm.OLS(y, X, missing="drop")
        if cluster_col is not None:
            groups = panel[cluster_col]
            fit = mod.fit(cov_type="cluster", cov_kwds={"groups": groups})
        else:
            fit = mod.fit(cov_type="HC1")

        if x_var in fit.params.index:
            result.coef = fit.params[x_var]
            result.se = fit.bse[x_var]
            result.tstat = fit.tvalues[x_var]
            result.pval = fit.pvalues[x_var]
        else:
            result.error_msg = f"解释变量 {x_var} 不在结果中"
            return result

        result.nobs = int(fit.nobs)
        result.r2 = fit.rsquared
        result.r2_within = fit.rsquared_adj
        result.f_stat = fit.fvalue

        for var in [x_var] + exog_vars:
            if var in fit.params.index:
                result.all_coefs[var] = fit.params[var]
                result.all_se[var] = fit.bse[var]
                result.all_pval[var] = fit.pvalues[var]

        result.success = True
        result.__post_init__()
    except Exception as e:
        result.error_msg = str(e)

    return result


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
    运行平行趋势检验（事件研究法 / 动态效应估计）。

    生成事件时间虚拟变量，与处理组交互，估计各期系数。

    Parameters
    ----------
    base_period : str
        基期选择:
        - "pre1": 事前一年 (t=-1)
        - "pre0": 冲击当年 (t=0)
        - "pre_biggest": 事前最远年份 (t=pre_window)
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
            pt.error_msg = f"变量缺失: {x_var} 或 {y_var}"
            return pt

        data = df.copy()

        # 生成事件时间
        if "event_time" not in data.columns or "event_year" not in data.columns:
            from .data_loader import generate_event_time
            data = generate_event_time(data, x_var, year_col, firm_col)

        if data["event_time"].isna().all():
            pt.error_msg = "无法生成事件时间"
            return pt

        # 确定基期
        if base_period == "pre1":
            omit_period = -1
        elif base_period == "pre0":
            omit_period = 0
        elif base_period == "pre_biggest":
            omit_period = pre_window
        else:
            omit_period = -1

        # 生成事件时间哑变量
        periods = list(range(pre_window, post_window + 1))
        periods_to_include = [p for p in periods if p != omit_period]

        for p in periods_to_include:
            col_name = f"pre{abs(p)}" if p < 0 else (f"current" if p == 0 else f"post{p}")
            if p < 0:
                col_name = f"pre{abs(p)}"
            elif p == 0:
                col_name = "current"
            else:
                col_name = f"post{p}"

            # 对于处理组: 事件时间 == p 时为1, 否则为0
            # 对于对照组: 始终为0
            is_treated = data["event_year"].notna()
            is_period = data["event_time"] == p
            data[col_name] = (is_treated & is_period).astype(float)

        # 准备回归变量
        event_dummies = []
        for p in periods_to_include:
            if p < 0:
                event_dummies.append(f"pre{abs(p)}")
            elif p == 0:
                event_dummies.append("current")
            else:
                event_dummies.append(f"post{p}")

        all_exog = event_dummies + [c for c in controls if c in data.columns]
        all_needed = [y_var] + all_exog + fe_vars + cluster_vars
        all_needed = list(dict.fromkeys(all_needed))
        available = [v for v in all_needed if v in data.columns]

        panel = data[available].dropna()
        if len(panel) < 50:
            pt.error_msg = f"样本量不足: {len(panel)}"
            return pt

        pt.nobs = len(panel)

        # 运行回归
        import statsmodels.api as sm

        y = panel[y_var]
        exog_available = [v for v in all_exog if v in panel.columns]
        X = panel[exog_available].copy()

        # 固定效应哑变量
        for fe_var in fe_vars:
            if fe_var in panel.columns:
                dummies = pd.get_dummies(panel[fe_var], prefix=fe_var, drop_first=True, dtype=float)
                X = pd.concat([X, dummies], axis=1)

        X = sm.add_constant(X)

        cluster_col = cluster_vars[0] if cluster_vars and cluster_vars[0] in panel.columns else None

        mod = sm.OLS(y, X, missing="drop")
        if cluster_col is not None:
            fit = mod.fit(cov_type="cluster", cov_kwds={"groups": panel[cluster_col]})
        else:
            fit = mod.fit(cov_type="HC1")

        pt.r2 = fit.rsquared
        pt.nobs = int(fit.nobs)

        # 提取各期系数
        expected_sign = cfg.EXPECTED_SIGN.get(chapter, -1)
        n_pre_sig = 0
        n_post_sig = 0
        post_sig_consecutive = 0
        max_post_consecutive = 0
        post_signs_correct = True

        for i, p in enumerate(periods_to_include):
            dummy_name = event_dummies[i]
            if dummy_name in fit.params.index:
                coef = fit.params[dummy_name]
                se = fit.bse[dummy_name]
                pv = fit.pvalues[dummy_name]
                ci = fit.conf_int().loc[dummy_name]

                pt.period_coefs[p] = coef
                pt.period_se[p] = se
                pt.period_pval[p] = pv
                pt.period_ci_lower[p] = ci[0]
                pt.period_ci_upper[p] = ci[1]

                is_sig = pv < 0.10

                if p < 0 and p != omit_period:
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

        # 添加基期 (系数=0, se=0)
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

    except Exception as e:
        pt.error_msg = str(e)

    return pt


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
    运行 PSM-DID 回归。

    步骤：
    1. 估计倾向得分 (Logit)
    2. 根据指定方法匹配
    3. 平衡性检验
    4. 在匹配样本上运行 DID 回归

    Parameters
    ----------
    psm_config : dict
        匹配方法配置:
        - method: "nearest" 或 "kernel"
        - n_neighbors: 匹配邻居数 (nearest)
        - caliper: 卡尺 (nearest)
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.spatial import KDTree

    psm = PSMResult()

    if psm_config is None:
        psm_config = {"method": "nearest", "n_neighbors": 1, "caliper": None}

    psm.psm_method = f"{psm_config['method']}_k{psm_config.get('n_neighbors', '')}_c{psm_config.get('caliper', '')}"

    try:
        from . import config as cfg

        firm_col = cfg.FIRM_ID
        year_col = cfg.YEAR_VAR

        # --- Step 0: 基准回归（匹配前） ---
        psm.baseline_result = run_ols_fe(df, y_var, x_var, controls, fe_vars, cluster_vars)

        # --- Step 1: 倾向得分估计 ---
        # 使用基期截面数据估计
        # 找到处理变量: 需要一个时不变的处理组标识
        if x_var not in df.columns:
            psm.success = False
            return psm

        # 构造企业层面处理标识
        treat_firm = df.groupby(firm_col)[x_var].max().reset_index()
        treat_firm.columns = [firm_col, "treated"]

        # 用控制变量的均值作为匹配依据
        available_controls = [c for c in controls if c in df.columns]
        firm_chars = df.groupby(firm_col)[available_controls].mean().reset_index()
        psm_data = firm_chars.merge(treat_firm, on=firm_col, how="inner").dropna()

        if psm_data["treated"].sum() < 10 or (1 - psm_data["treated"]).sum() < 10:
            psm.success = False
            psm.baseline_result.error_msg = "处理组或对照组企业数不足"
            return psm

        X_psm = psm_data[available_controls].values
        y_psm = psm_data["treated"].values

        # Logit 模型估计倾向得分
        logit = LogisticRegression(max_iter=1000, solver="lbfgs")
        logit.fit(X_psm, y_psm)
        pscore = logit.predict_proba(X_psm)[:, 1]
        psm_data["pscore"] = pscore

        # --- Step 2: 匹配 ---
        treated_idx = psm_data["treated"] == 1
        control_idx = psm_data["treated"] == 0

        treated_scores = psm_data.loc[treated_idx, "pscore"].values.reshape(-1, 1)
        control_scores = psm_data.loc[control_idx, "pscore"].values.reshape(-1, 1)

        if psm_config["method"] in ["nearest", "kernel"]:
            tree = KDTree(control_scores)
            n_neighbors = psm_config.get("n_neighbors", 1) or 1
            distances, indices = tree.query(treated_scores, k=min(n_neighbors, len(control_scores)))

            # 应用卡尺
            caliper = psm_config.get("caliper")
            matched_control_ids = []
            control_firms = psm_data.loc[control_idx, firm_col].values

            if distances.ndim == 1:
                distances = distances.reshape(-1, 1)
                indices = indices.reshape(-1, 1)

            for i in range(len(treated_scores)):
                for j in range(distances.shape[1]):
                    if caliper is None or distances[i, j] <= caliper:
                        matched_control_ids.append(control_firms[indices[i, j]])

        matched_control_ids = list(set(matched_control_ids))
        treated_firms = psm_data.loc[treated_idx, firm_col].values.tolist()

        psm.n_treated = len(treated_firms)
        psm.n_control = len(matched_control_ids)

        # --- Step 3: 平衡性检验 ---
        psm.balance_before = _balance_test(psm_data, available_controls, "treated")

        matched_firms = set(treated_firms + matched_control_ids)
        matched_psm_data = psm_data[psm_data[firm_col].isin(matched_firms)]
        psm.balance_after = _balance_test(matched_psm_data, available_controls, "treated")

        # --- Step 4: 匹配后回归 ---
        matched_panel = df[df[firm_col].isin(matched_firms)].copy()
        psm.psm_result = run_ols_fe(
            matched_panel, y_var, x_var, controls, fe_vars, cluster_vars
        )

        psm.success = psm.psm_result.success

    except Exception as e:
        psm.success = False
        psm.baseline_result.error_msg = str(e)

    return psm


def _balance_test(data: pd.DataFrame, covariates: list[str], treat_col: str) -> pd.DataFrame:
    """
    PSM 平衡性检验: 计算处理组和对照组各协变量的均值、标准差、标准化偏差。
    """
    from scipy.stats import ttest_ind

    results = []
    for var in covariates:
        if var not in data.columns:
            continue
        treated = data.loc[data[treat_col] == 1, var].dropna()
        control = data.loc[data[treat_col] == 0, var].dropna()

        if len(treated) < 2 or len(control) < 2:
            continue

        mean_t = treated.mean()
        mean_c = control.mean()
        std_t = treated.std()
        std_c = control.std()

        # 标准化偏差 (Standardized Bias)
        pooled_std = np.sqrt((std_t**2 + std_c**2) / 2)
        sbias = (mean_t - mean_c) / pooled_std * 100 if pooled_std > 0 else 0

        # t 检验
        t_stat, p_val = ttest_ind(treated, control, equal_var=False)

        results.append({
            "Variable": var,
            "Mean_Treated": mean_t,
            "Mean_Control": mean_c,
            "Std_Treated": std_t,
            "Std_Control": std_c,
            "Std_Bias(%)": sbias,
            "t_stat": t_stat,
            "p_value": p_val,
        })

    return pd.DataFrame(results)


def compute_descriptive_stats(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    计算描述统计量。

    Returns
    -------
    pd.DataFrame
        包含 N, Mean, Std, Min, P25, Median, P75, Max
    """
    stats_list = []
    for var in variables:
        if var not in df.columns:
            continue
        col = df[var].dropna()
        if len(col) == 0:
            continue
        stats_list.append({
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

    return pd.DataFrame(stats_list)


def compute_correlation_matrix(
    df: pd.DataFrame,
    variables: list[str],
) -> pd.DataFrame:
    """
    计算 Pearson 相关系数矩阵，标注显著性。

    Returns
    -------
    pd.DataFrame
        相关系数矩阵（下三角 Pearson, 对角线为 1）
    """
    from scipy.stats import pearsonr

    available = [v for v in variables if v in df.columns]
    data = df[available].dropna()

    n = len(available)
    corr_matrix = pd.DataFrame(np.eye(n), index=available, columns=available)
    pval_matrix = pd.DataFrame(np.zeros((n, n)), index=available, columns=available)

    for i in range(n):
        for j in range(i + 1, n):
            x = data[available[i]]
            y = data[available[j]]
            valid = x.notna() & y.notna()
            if valid.sum() > 2:
                r, p = pearsonr(x[valid], y[valid])
                corr_matrix.iloc[i, j] = r
                corr_matrix.iloc[j, i] = r
                pval_matrix.iloc[i, j] = p
                pval_matrix.iloc[j, i] = p

    # 格式化: 添加星号
    formatted = corr_matrix.copy().astype(str)
    for i in range(n):
        for j in range(n):
            if i != j:
                r = corr_matrix.iloc[i, j]
                p = pval_matrix.iloc[i, j]
                stars = ""
                if p < 0.01:
                    stars = "***"
                elif p < 0.05:
                    stars = "**"
                elif p < 0.10:
                    stars = "*"
                formatted.iloc[i, j] = f"{r:.3f}{stars}"
            else:
                formatted.iloc[i, j] = "1.000"

    return formatted
