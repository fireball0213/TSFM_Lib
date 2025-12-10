import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from Model_Path.model_zoo_config import Model_zoo_details, Model_abbrev_map, All_model_names
from selector.selector_config import Selector_zoo_details
from utils.check_tools import (
    check_results_file,
    standardize_model_names,
    calculate_order_metrics,
)


def process_results(file_path, model_name, common_datasets, verbose=False):
    """加载并处理数据文件的通用函数"""
    if not file_path.exists():
        print(f"❌ 文件缺失: {file_path}")
        return None

    df = check_results_file(file_path, verbose)
    if df is None:
        return None

    df["model"] = model_name
    df[['ds_key', 'ds_freq', 'term']] = df['dataset'].str.extract(r'^(.*?)/([^/]+)/([^/]+)$')
    if 'model_order' in df.columns:
        df['model_order'] = df['model_order'].apply(
            lambda x: x.tolist() if hasattr(x, 'tolist') else
            [int(i) for i in x.strip('[]').split()] if isinstance(x, str) else x
        )
    df_return=df[df['dataset'].isin(common_datasets)].copy()
    return df_return


def caculate_combined_rank(
    combined_df: pd.DataFrame,
    zoo_model_name: str,
    verbose: bool = False,
    first_col_prefix: str | None = None,
    rank_base: str = "MASE",
    include_selector_in_rank: bool = False,
):
    """
    阶段二：对 baseline + selector 合并后的 DataFrame 做 rank 计算与 selector 汇总。

    1）根据 rank_base 指定的指标计算每个 dataset 内的模型 rank，可以是 'MASE' / 'sMAPE' / 'CRPS'

    2）是否将 selector 一起参与 rank 的计算由 include_selector_in_rank 控制：
        - include_selector_in_rank = True：
            所有模型（baseline + selector）一起 groupby('dataset') 排名
            baseline 的 rank 会随着 selector 表现变化而变化
        - include_selector_in_rank = False：
            先只对 baseline 排名，再把 selector 按“有多少 baseline 比它好”插入排名
            baseline 的 rank 完全不受 selector 变化影响（稳定）
    """
    df = combined_df.copy()

    if rank_base not in df.columns:
        raise ValueError(f"rank_base='{rank_base}' 不在 DataFrame 列中，无法计算 RANK")

    # 计算 RANK 列
    ranked_df = df.copy()
    rank_col = "RANK"

    # 情况一：不区分特殊模型，所有模型一起按指标排名
    if zoo_model_name is None or include_selector_in_rank:
        ranked_df[rank_col] = ranked_df.groupby("dataset")[rank_base].rank(
            method="min", ascending=True
        )
    else:
        # 情况二：baseline 与 selector 分开处理，保证 baseline 的 rank 稳定
        special_mask = ranked_df["model"] == zoo_model_name
        special_rows = ranked_df[special_mask].copy()
        other_rows = ranked_df[~special_mask].copy()

        # 先对 baseline（other_rows）按指标排名
        other_rows[rank_col] = other_rows.groupby("dataset")[rank_base].rank(
            method="min", ascending=True
        )

        EPS_REL = 5e-3  # 相对阈值：0.5% 之内视为Rank相同
        final_dfs = []
        for dataset, group in other_rows.groupby("dataset"):
            dataset_special = special_rows[special_rows["dataset"] == dataset].copy()

            if not dataset_special.empty:
                special_val = dataset_special[rank_base].values[0]
                # baseline_val < special_val * (1 - EPS_REL) 才算“严格优于 selector”
                better_mask = group[rank_base] < special_val * (1.0 - EPS_REL)
                rank_pos = int(better_mask.sum()) + 1
                dataset_special[rank_col] = rank_pos

                # ⭐ Real 专用 debug：查看该 dataset 下所有 baseline + Real 的指标和 rank,检查Rank计算异常
                if (
                        verbose
                        and zoo_model_name.startswith("Real")
                        and rank_pos != 1
                ):
                    debug_df = pd.concat([group, dataset_special], ignore_index=True)
                    keep_cols = ["dataset", "model", rank_base, rank_col]
                    keep_cols_exist = [c for c in keep_cols if c in debug_df.columns]
                    debug_df = debug_df[keep_cols_exist].copy()
                    debug_df = debug_df.sort_values(by=rank_base, ascending=True)

                    print(f"\n⚠️ [DEBUG-Real-RankStep] dataset = {dataset}")
                    print(
                        tabulate(
                            debug_df,
                            headers="keys",
                            tablefmt="plain",
                            floatfmt=".6f",
                            numalign="decimal",
                            stralign="left",
                        )
                    )

                final_dfs.append(pd.concat([group, dataset_special]))
            else:
                final_dfs.append(group)

        if not final_dfs and not special_rows.empty:
            # 只有 selector 没有 baseline 的极端情况
            special_rows[rank_col] = 1
            ranked_df = special_rows
        else:
            ranked_df = pd.concat(final_dfs)

    df = ranked_df.sort_index()

    # 1）全局（所有模型）的平均指标，可选打印
    metrics_to_show = ["sMAPE", "MASE", "CRPS", "RANK"]
    metrics_exist = [m for m in metrics_to_show if m in df.columns]
    if metrics_exist:
        global_avg = df.groupby("model")[metrics_exist].mean().T
        global_avg = global_avg.reindex(metrics_exist).round(4)

        if verbose:
            n_ds = df["dataset"].nunique()
            print(f"共有数据集: {n_ds}, "
                  f"rank_base={rank_base}, 动态Rank={include_selector_in_rank}")

            data = global_avg if isinstance(global_avg, pd.DataFrame) else global_avg.to_frame().T
            cols = list(data.columns)
            if first_col_prefix:
                first_cols = [c for c in cols if str(c).startswith(first_col_prefix)]
                other_cols = [c for c in cols if c not in first_cols]
                cols = first_cols + other_cols
                data = data[cols]

            data_print = data.copy()
            data_print.columns = [Model_abbrev_map.get(str(c), str(c)) for c in data_print.columns]

            data_print = data_print.reset_index().rename(columns={"index": "Metrics"})
            print(
                tabulate(
                    data_print,
                    headers="keys",
                    tablefmt="plain",
                    floatfmt=".3f",
                    numalign="decimal",
                    stralign="left",
                )
            )

    # 2）仅构造 selector 对应的一列汇总（Rank + 指标均值）
    filtered = df[df["model"] == zoo_model_name]

    table = pd.DataFrame(index=[], columns=[zoo_model_name])

    # Rank 行：selector 的平均 RANK
    table.loc["Rank", zoo_model_name] = filtered["RANK"].mean()

    # rank_base 行：selector 的 rank_base 指标平均值
    if rank_base in filtered.columns:
        table.loc[rank_base, zoo_model_name] = filtered[rank_base].mean()

    # 额外附加一个常用指标：sMAPE（如果存在）
    if "sMAPE" in filtered.columns:
        table.loc["sMAPE", zoo_model_name] = filtered["sMAPE"].mean()

    table = table.round(2)

    rank_summary = {"RANK": table}
    return rank_summary



def add_order_metrics(
    baseline_subset: pd.DataFrame,
    subset_df: pd.DataFrame,
    model_name: str,
    rank_summary_all: dict,
    add_index: int = 0,
    verbose: bool = True,
    df_real: pd.DataFrame | None = None,
    k_order=None,
    rank_base: str = "MASE",
    include_selector_in_rank: bool = False,
):

    combined_df = pd.concat([baseline_subset, subset_df], ignore_index=True)
    rank_summary = caculate_combined_rank(combined_df, zoo_model_name=model_name, verbose=verbose,rank_base=rank_base,include_selector_in_rank=include_selector_in_rank,)

    for rank_type in rank_summary_all:
        if rank_type in rank_summary and model_name not in rank_summary_all[rank_type].columns:
            rank_summary_all[rank_type].insert(add_index, model_name, rank_summary[rank_type][model_name])

            # 对 All_* 列不计算 order 的 5 个指标
            skip_order_for_this_model = str(model_name).startswith("All_")

            # 添加order指标计算结果
            if (
                not skip_order_for_this_model
                and df_real is not None
                and "model_order" in subset_df.columns
                and "model_order" in df_real.columns
            ):
                metrics = calculate_order_metrics(df_real, subset_df, k_order)
                for metric_name, value in metrics.items():
                    if metric_name not in rank_summary_all[rank_type].index:
                        rank_summary_all[rank_type].loc[metric_name] = np.nan
                    rank_summary_all[rank_type].loc[metric_name, model_name] = value
        elif model_name in rank_summary_all[rank_type].columns:
            print(f"⚠️ 已存在 '{model_name}' 列，跳过插入")

    return rank_summary_all

def parse_seed_list(seed_str: str):
    """✅ 将 '2024,2025,2026' 这类字符串解析为 [2024,2025,2026]"""
    seeds = []
    for part in seed_str.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            seeds.append(int(part))
        except ValueError:
            print(f"⚠️ 无法解析随机种子 '{part}'，已跳过")
    return seeds

# 构造带默认参数的 selector 路径 builder
def make_selector_path_builder(
    results_dir: Path,
    current_zoo_num: int,
    zoo_total_num: int,
    ensemble_size: int,
    default_real_metric: str,
):
    """
    返回一个内嵌的 build(selector_name, seed=None, real_order_metric=None) 函数。

    - 公共参数（results_dir / current_zoo_num / zoo_total_num / ensemble_size / default_real_metric）
      只在这里写一次，后面调用时只需要关心 selector_name、seed、real_order_metric。
    """
    def build(selector_name: str, seed: int | None = None, real_order_metric: str | None = None) -> Path:
        cfg = Selector_zoo_details[selector_name]
        tpl = cfg["csv_name_tpl"]

        fname = tpl.format(
            current_zoo_num=current_zoo_num,
            zoo_total_num=zoo_total_num,
            ensemble_size=ensemble_size,
            real_order_metric=real_order_metric or default_real_metric,# Real_Select 用到
            seed=seed if seed is not None else 0,# Random_Select 用 seed
        )
        return results_dir / selector_name / fname

    return build


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",type=str,default="results",help="结果根目录",)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--context_len', type=int, default=512, help='模型预测所需的输入长度context length')
    parser.add_argument('--zoo_total_num', type=int, default=4)
    parser.add_argument('--ensemble_size', type=int, default=1)
    parser.add_argument("--rank_base",type=str,default="MASE",choices=["MASE", "sMAPE", "CRPS"],help="用于计算 RANK 的指标",)
    parser.add_argument('--real_order_metric', type=str, default='MASE', help='用于计算真实order的评估指标，options: [sMAPE, MASE]')
    parser.add_argument('--real_world_mode', action='store_true', default=False, help='是否使用增量模型库运行,Fasle时使用单一select_date')
    parser.add_argument("--random_seeds",type=str,default="1",help="Random_Select 使用的随机种子列表，例如 '2024,2025,2026'",)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    args.zoo_total_num = sum(len(sizes) for sizes in Model_zoo_details.values())

    # =========================
    # 1️⃣ baseline 结果
    # =========================

    # 获取全部 baseline TSFM模型结果
    baseline_data = []
    for model_name in All_model_names:
        file_path = Path(results_dir) / model_name / f"cl_{args.context_len}" /  "all_results.csv"
        if file_path.exists():
            print(f"\n🔹加载 baseline : {model_name}",end=" ")
            df = check_results_file(file_path, args.verbose)
            if df is not None:
                df["model"] = model_name
                baseline_data.append(df)
        else:
            print(f"❌ 未找到original文件: {file_path}\n")


    baseline_df_all = standardize_model_names(baseline_data)

    # 根据 release_date 构造「按发布时间排序」的模型缩写列表
    model_release_list = []
    for family, sizes in Model_zoo_details.items():
        for size, details in sizes.items():
            full_name = f"{family}_{size}"
            abbrev = details.get("abbreviation", Model_abbrev_map.get(full_name, full_name))
            rel = details.get("release_date", "2026-01-01")
            model_release_list.append((rel, abbrev))

    # 按发布日期排序
    model_release_list = sorted(model_release_list, key=lambda x: x[0])
    ordered_model_names = [abbrev for _, abbrev in model_release_list]
    args.zoo_total_num = len(ordered_model_names) # 以「按发布日期排序后的模型数」为准

    # =========================
    # 2️⃣ selector 对比
    # =========================
    rank_summary_all = {"RANK": pd.DataFrame()}
    random_seeds = parse_seed_list(args.random_seeds)


    if args.real_world_mode:
        current_zoo_nums = range(args.ensemble_size + 1, args.zoo_total_num + 1)
    else:# 非 real_world_mode：只看最完整zoo的 selector 表现
        current_zoo_nums = [args.zoo_total_num]

    k_order = [1,3]

    for current_zoo_num in current_zoo_nums:
        current_model_names = ordered_model_names[:current_zoo_num]
        baseline_df = baseline_df_all[baseline_df_all["model"].isin(current_model_names)].copy()
        baseline_datasets = set(baseline_df["dataset"].unique())

        # 使用 builder，把公共参数固化在这里
        build_sel_path = make_selector_path_builder(
            results_dir=results_dir,
            current_zoo_num=current_zoo_num,
            zoo_total_num=args.zoo_total_num,
            ensemble_size=args.ensemble_size,
            default_real_metric=args.real_order_metric,
        )

        print(
            f"\n{'=' * 60}\n🎯 对比 selector（zoo{current_zoo_num}-{args.zoo_total_num}, "
            f"ensemble_size={args.ensemble_size}, rank_base={args.rank_base})\n{'=' * 60}"
        )

        # ---------- 先尝试加载 Real_Select，作为 order 指标的“真值” ----------
        real_model_name = f"Real-{args.real_order_metric}_z{current_zoo_num}-{args.zoo_total_num}"

        real_path = build_sel_path(
            selector_name="Real_Select",
            seed=0,
            real_order_metric=args.real_order_metric,
        )

        real_raw = None
        real_datasets = set()

        if real_path.exists():
            real_raw = check_results_file(real_path, args.verbose)
            if real_raw is not None:
                real_datasets = set(real_raw["dataset"].unique())
                print(f"✅ 加载 Real_Select 标记: {real_path}")
        else:
            print(f"⚠️ Real_Select 文件缺失: {real_path}（无法计算 order 指标，仅比较 sMAPE/MASE/Rank）")

        # ---------- 加载 所有Select方法 ----------
        selector_tasks = []

        # 1) All_Select
        all_path = build_sel_path("All_Select")
        selector_tasks.append(("All_Select", None, all_path, f"All_z{current_zoo_num}-{args.zoo_total_num}"))

        # 2) Recent_Select
        recent_path = build_sel_path("Recent_Select")
        selector_tasks.append(("Recent_Select",None,recent_path,f"Recent_z{current_zoo_num}-{args.zoo_total_num}",))

        # 3) Real_Select
        if real_raw is not None:
            selector_tasks.append(("Real_Select",None,real_path,real_model_name,))

        # 4) Random_Select（可能有多个 seed）
        for seed in random_seeds:
            rand_path = build_sel_path("Random_Select", seed=seed)
            selector_tasks.append(("Random_Select",seed,rand_path,f"Random_s{seed}_z{current_zoo_num}-{args.zoo_total_num}",))

        # 5) TODO:参照四个baseline_selector，添加其他 selector 方法的结果汇总
            # selector_tasks.append(...)


        # ---------- 逐个 selector 汇总结果 ----------
        for selector_name, seed, sel_path, model_col_name in selector_tasks:
            if not sel_path.exists():
                print(f"⚠️ {selector_name} 文件不存在: {sel_path}")
                continue

            print(f"\n🔹 加载 {selector_name} 结果: {sel_path}",end=" ")

            sel_raw = check_results_file(sel_path, args.verbose)
            if sel_raw is None:
                continue
            sel_datasets = set(sel_raw["dataset"].unique())

            # 确定本 selector 下使用的公共数据集
            if real_raw is not None:
                common_datasets = baseline_datasets & sel_datasets & real_datasets
            else:
                common_datasets = baseline_datasets & sel_datasets

            if not common_datasets:
                print(f"⚠️ {model_col_name} 在 baseline/Real 中没有共同数据集，跳过。")
                continue

            baseline_subset = baseline_df[baseline_df["dataset"].isin(common_datasets)].copy()

            subset_df = process_results(
                sel_path, model_col_name, common_datasets, verbose=args.verbose
            )

            if subset_df is None or subset_df.empty:
                print(f"⚠️ {model_col_name} 过滤后无数据，跳过。")
                continue

            # Real_Select 真值（给 order 指标用）
            df_real = None
            if real_raw is not None:
                df_real = process_results(
                    real_path, real_model_name, common_datasets, verbose=False
                )

            rank_summary_all = add_order_metrics(
                baseline_subset=baseline_subset,
                subset_df=subset_df,
                model_name=model_col_name,
                rank_summary_all=rank_summary_all,
                add_index=0,
                k_order=k_order,
                df_real=df_real,
                rank_base=args.rank_base,
                include_selector_in_rank=False, #确保基线TSFM的Rank不会被Selector影响
            )
            
    # =========================
    # 3️⃣ 汇总打印
    # =========================

    print("\n" + "=" * 60 + "\n📊 Selector 对比汇总表格\n" + "=" * 60)
    for rank_type, df_summary in rank_summary_all.items():
        print(f"\n📈 Rank Type: {args.rank_base}-{rank_type}")

        # 隐藏随机结果的详细列，只保留平均值列（如果你后续按 seed 做聚合，可以保留）
        random_cols = [
            col for col in df_summary.columns if col.startswith("Rt") and not col.endswith("m")
        ]
        cols_to_show = [col for col in df_summary.columns if col not in random_cols]
        df_summary_to_print = df_summary[cols_to_show]

        print(
            tabulate(
                df_summary_to_print,
                headers="keys",
                tablefmt="plain",
                floatfmt=".3f",
                numalign="decimal",
                stralign="left",
            )
        )


