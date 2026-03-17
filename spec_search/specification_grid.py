"""
============================================================
设定搜索网格生成器
============================================================
管理组合爆炸：分阶段搜索策略
"""

import itertools
from dataclasses import dataclass, field
from typing import Any
from . import config


@dataclass
class SpecConfig:
    """单个回归设定配置"""
    chapter: int = 3
    y_var: str = ""
    x_var: str = ""
    controls_group: str = ""
    controls: list = field(default_factory=list)
    fe_vars: list = field(default_factory=list)
    cluster_vars: list = field(default_factory=list)
    start_year: int = 2007
    end_year: int = 2023
    filters: dict = field(default_factory=dict)
    psm_config: dict | None = None
    # 平行趋势参数
    pt_pre_window: int = -3
    pt_post_window: int = 3
    pt_base_period: str = "pre1"

    @property
    def spec_id(self) -> str:
        """唯一标识"""
        fe_str = "+".join(self.fe_vars)
        cl_str = "+".join(self.cluster_vars)
        filt_str = "_".join(f"{k}{v}" for k, v in sorted(self.filters.items()) if v)
        return (f"ch{self.chapter}_{self.y_var}_{self.x_var}_"
                f"{self.controls_group}_{fe_str}_{cl_str}_"
                f"{self.start_year}-{self.end_year}_{filt_str}")


def generate_phase1_grid(chapter: int) -> list[SpecConfig]:
    """
    Phase 1: 快速扫描 — 默认设定测试所有 X-Y 组合。

    使用标准控制变量、Stkcd+Year FE、Stkcd 聚类。
    遍历所有样本区间。
    """
    specs = []
    y_vars = config.Y_VARS.get(chapter, [])
    ctrl_groups = config.CHAPTER_CONTROL_GROUPS.get(chapter, [])

    default_ctrl_group = ctrl_groups[0] if ctrl_groups else "Controls_Share"
    default_controls = config.CONTROL_GROUPS.get(default_ctrl_group, [])

    default_fe = [config.FIRM_ID, config.YEAR_VAR]
    default_cluster = [config.FIRM_ID]

    # 标准筛选
    default_filters = {
        "drop_st_pt": True,
        "drop_finance": True,
        "drop_real_estate": True,
        "drop_lev_gt1": True,
        "drop_ind_lt30": False,
        "winsorize": 0.01,
    }

    for y_var in y_vars:
        for x_var in config.X_VARS_PRIMARY:
            for start_year in config.SAMPLE_START_YEARS:
                for end_year in config.SAMPLE_END_YEARS:
                    specs.append(SpecConfig(
                        chapter=chapter,
                        y_var=y_var,
                        x_var=x_var,
                        controls_group=default_ctrl_group,
                        controls=default_controls,
                        fe_vars=default_fe,
                        cluster_vars=default_cluster,
                        start_year=start_year,
                        end_year=end_year,
                        filters=default_filters.copy(),
                    ))

    print(f"[Phase 1] 第{chapter}章: 生成 {len(specs)} 个快速扫描设定")
    return specs


def generate_phase2_grid(
    chapter: int,
    promising_pairs: list[tuple[str, str]],
) -> list[SpecConfig]:
    """
    Phase 2: 针对有希望的 (X, Y) 组合，变化控制变量、FE、聚类等。

    Parameters
    ----------
    promising_pairs : list of (y_var, x_var) tuples
        Phase 1 中通过方向检验的组合
    """
    specs = []
    ctrl_groups = config.CHAPTER_CONTROL_GROUPS.get(chapter, [])

    for y_var, x_var in promising_pairs:
        for ctrl_name in ctrl_groups:
            controls = config.CONTROL_GROUPS.get(ctrl_name, [])
            for fe_vars in config.FE_COMBINATIONS:
                for cluster_vars in config.CLUSTER_OPTIONS:
                    for filters in config.FILTER_COMBINATIONS:
                        for start_year in config.SAMPLE_START_YEARS:
                            for end_year in config.SAMPLE_END_YEARS:
                                specs.append(SpecConfig(
                                    chapter=chapter,
                                    y_var=y_var,
                                    x_var=x_var,
                                    controls_group=ctrl_name,
                                    controls=controls,
                                    fe_vars=fe_vars,
                                    cluster_vars=cluster_vars,
                                    start_year=start_year,
                                    end_year=end_year,
                                    filters=filters.copy(),
                                ))

    print(f"[Phase 2] 第{chapter}章: 生成 {len(specs)} 个深度搜索设定")

    # 如果组合太多，采样
    if len(specs) > 5000:
        import random
        random.seed(42)
        specs = random.sample(specs, 5000)
        print(f"[Phase 2] 采样至 5000 个设定")

    return specs


def generate_phase3_pt_grid(
    chapter: int,
    best_specs: list[SpecConfig],
) -> list[SpecConfig]:
    """
    Phase 3: 对 Top 设定进行平行趋势检验搜索。

    遍历不同的事前事后窗口和基期选择。
    """
    specs = []

    for base_spec in best_specs:
        for pre_w in config.PT_PRE_WINDOWS:
            for post_w in config.PT_POST_WINDOWS:
                for base_p in config.PT_BASE_PERIODS:
                    new_spec = SpecConfig(
                        chapter=base_spec.chapter,
                        y_var=base_spec.y_var,
                        x_var=base_spec.x_var,
                        controls_group=base_spec.controls_group,
                        controls=base_spec.controls,
                        fe_vars=base_spec.fe_vars,
                        cluster_vars=base_spec.cluster_vars,
                        start_year=base_spec.start_year,
                        end_year=base_spec.end_year,
                        filters=base_spec.filters.copy(),
                        pt_pre_window=pre_w,
                        pt_post_window=post_w,
                        pt_base_period=base_p,
                    )
                    specs.append(new_spec)

    print(f"[Phase 3] 第{chapter}章: 生成 {len(specs)} 个平行趋势设定")
    return specs


def generate_psm_grid(
    chapter: int,
    best_specs: list[SpecConfig],
) -> list[SpecConfig]:
    """
    为 Top 设定生成 PSM-DID 变体。
    """
    specs = []

    for base_spec in best_specs:
        for psm_cfg in config.PSM_METHODS:
            new_spec = SpecConfig(
                chapter=base_spec.chapter,
                y_var=base_spec.y_var,
                x_var=base_spec.x_var,
                controls_group=base_spec.controls_group,
                controls=base_spec.controls,
                fe_vars=base_spec.fe_vars,
                cluster_vars=base_spec.cluster_vars,
                start_year=base_spec.start_year,
                end_year=base_spec.end_year,
                filters=base_spec.filters.copy(),
                psm_config=psm_cfg.copy(),
            )
            specs.append(new_spec)

    print(f"[PSM] 第{chapter}章: 生成 {len(specs)} 个PSM-DID设定")
    return specs
