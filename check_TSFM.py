import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from Model_Path.model_zoo_config import Model_zoo_details, Model_abbrev_map, All_model_names
from utils.check_tools import (
    check_dataset_completeness,
    check_duplicate_results,
    check_model_naming,
    analyze_model_results,
    standardize_model_names,
    format_rank_summary_all,
    calculate_order_metrics,
)



def check_results_file(csv_file_path,verbose=False):
    """检查结果文件的完整性和一致性"""
    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        print(f"无法读取CSV文件: {e}")
        return None

    if verbose:
        print(f"\n{'=' * 50}\n检查结果文件: {csv_file_path}")

    # 1. 数据集完整性
    check_dataset_completeness(df, verbose)
    # 2. 去重
    df = check_duplicate_results(df, csv_file_path, verbose)
    # 3. 模型命名检查
    check_model_naming(df, verbose)
    # 4. 打印全局指标（只读）
    analyze_model_results(df, verbose)

    return df

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
    df = df.rename(columns={
        'eval_metrics/MASE[0.5]': 'MASE',
        'eval_metrics/mean_weighted_sum_quantile_loss': 'CRPS',
        'eval_metrics/sMAPE[0.5]': 'sMAPE',
    })
    if 'model_order' in df.columns:
        df['model_order'] = df['model_order'].apply(
            lambda x: x.tolist() if hasattr(x, 'tolist') else
            [int(i) for i in x.strip('[]').split()] if isinstance(x, str) else x
        )
    df_return=df[df['dataset'].isin(common_datasets)].copy()
    return df_return


def summarize_baselines(baseline_df: pd.DataFrame, rank_base_list=None):
    """
    阶段一：仅对 baseline (cl_original) 做汇总。

    功能：
    - 对 baseline 内部按 rank_base 做 dataset 内排名，得到平均 RANK
    - 打印 sMAPE / MASE / CRPS / RANK 的全局均值表（列为模型，使用缩写）
    """
    if rank_base_list is None:
        rank_base_list = ["MASE", "sMAPE", "CRPS"]

    # 统一指标命名，兼容 eval_metrics 前缀
    df = baseline_df.copy()
    df = df.rename(
        columns={
            "eval_metrics/MASE[0.5]": "MASE",
            "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
            "eval_metrics/sMAPE[0.5]": "sMAPE",
        }
    )

    for rank_base in rank_base_list:
        if rank_base not in df.columns:
            print(f"⚠️ baseline 中不存在 rank_base='{rank_base}'，跳过该指标")
            continue

        df_ranked = df.copy()
        df_ranked["RANK"] = df_ranked.groupby("dataset")[rank_base].rank(
            method="min", ascending=True
        )

        metrics_to_show = ["sMAPE", "MASE", "CRPS", "RANK"]
        metrics_exist = [m for m in metrics_to_show if m in df_ranked.columns]
        if not metrics_exist:
            print(f"⚠️ baseline 中没有可用指标，跳过 rank_base={rank_base}")
            continue

        global_avg = df_ranked.groupby("model")[metrics_exist].mean().T
        global_avg = global_avg.reindex(metrics_exist).round(3)

        # 打印用列名替换为缩写
        print_df = global_avg.copy()
        print_df.columns = [
            Model_abbrev_map.get(str(c), str(c)) for c in print_df.columns
        ]

        print("\n" + "=" * 60)
        print(f"📊 Baseline 汇总表（rank_base = {rank_base}）")
        print("=" * 60)
        print(
            tabulate(
                print_df,
                headers="keys",
                tablefmt="plain",
                floatfmt=".3f",
                numalign="decimal",
                stralign="left",
            )
        )


def summarize_selectors(
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

    df = df.rename(
        columns={
            "eval_metrics/MASE[0.5]": "MASE",
            "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
            "eval_metrics/sMAPE[0.5]": "sMAPE",
        }
    )

    # 支持传原始列名，统一映射到 MASE/CRPS/sMAPE
    rank_base_map = {
        "eval_metrics/MASE[0.5]": "MASE",
        "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
        "eval_metrics/sMAPE[0.5]": "sMAPE",
    }
    rank_base = rank_base_map.get(rank_base, rank_base)
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

        final_dfs = []
        for dataset, group in other_rows.groupby("dataset"):
            dataset_special = special_rows[special_rows["dataset"] == dataset].copy()

            if not dataset_special.empty:
                special_val = dataset_special[rank_base].values[0]
                # baseline 中有多少模型比它好（指标更小）
                rank_pos = (group[rank_base] < special_val).sum() + 1
                dataset_special[rank_col] = rank_pos
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
            print(f"\n全局平均值共有数据集: {n_ds}, "
                  f"rank_base={rank_base}, include_selector_in_rank={include_selector_in_rank}")

            data = global_avg if isinstance(global_avg, pd.DataFrame) else global_avg.to_frame().T
            cols = list(data.columns)
            if first_col_prefix:
                first_cols = [c for c in cols if str(c).startswith(first_col_prefix)]
                other_cols = [c for c in cols if c not in first_cols]
                cols = first_cols + other_cols
                data = data[cols]

            header = ["Index"] + [str(c) for c in data.columns]
            print("\t".join(header))
            for idx in data.index:
                row = [str(idx)]
                for col in data.columns:
                    value = data.at[idx, col]
                    if pd.isna(value):
                        row.append("")
                    else:
                        row.append(f"{value:.3f}")
                print("\t".join(row))

    # 2）仅构造 selector 对应的一列汇总（Rank + 指标均值）
    filtered = df[df["model"] == zoo_model_name]
    table = pd.DataFrame(index=[], columns=[zoo_model_name])

    # Global 行：selector 的平均 RANK
    table.loc["Global", zoo_model_name] = filtered["RANK"].mean()

    # rank_base 行：selector 的 rank_base 指标平均值
    if rank_base in filtered.columns:
        table.loc[rank_base, zoo_model_name] = filtered[rank_base].mean()

    # 额外附加一个常用指标：sMAPE（如果存在）
    if "sMAPE" in filtered.columns:
        table.loc["sMAPE", zoo_model_name] = filtered["sMAPE"].mean()

    table = table.round(2)

    rank_summary = {"RANK": table}
    return rank_summary



def add_selector_rank(
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
    rank_summary = summarize_selectors(combined_df, zoo_model_name=model_name, verbose=verbose,rank_base=rank_base,include_selector_in_rank=include_selector_in_rank,)

    for rank_type in rank_summary_all:
        if rank_type in rank_summary and model_name not in rank_summary_all[rank_type].columns:
            rank_summary_all[rank_type].insert(add_index, model_name, rank_summary[rank_type][model_name])
            # 添加order指标计算结果
            if df_real is not None and 'model_order' in subset_df.columns and 'model_order' in df_real.columns:
                metrics = calculate_order_metrics(df_real, subset_df, k_order)
                for metric_name, value in metrics.items():
                    if metric_name not in rank_summary_all[rank_type].index:
                        rank_summary_all[rank_type].loc[metric_name] = np.nan
                    rank_summary_all[rank_type].loc[metric_name, model_name] = value
        elif model_name in rank_summary_all[rank_type].columns:
            print(f"⚠️ 已存在 '{model_name}' 列，跳过插入")

    return rank_summary_all




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    results_dir = "results/"
    # =========================
    # 1️⃣ baseline 结果汇总
    # =========================

    baseline_rank_summary_all = {"RANK": pd.DataFrame()}

    for context_len in [512, 'original']:
        baseline_data_print = []
        for model_name in ['chronos_bolt_tiny','moirai_small','timesfm_2.5','sundial_base']:
        # for model_name in All_model_names:
            file_path = Path(results_dir) / model_name / f"cl_{context_len}" / "all_results.csv"
            if not file_path.exists():
                print(f"⚠️ baseline 缺少文件: {file_path}")
                continue

            df_print = check_results_file(file_path, verbose=args.verbose)
            if df_print is None:
                continue

            df_print["model"] = model_name
            baseline_data_print.append(df_print)

        if not baseline_data_print:
            print(f"⚠️ context_len={context_len} 下没有任何可用的 baseline 结果，跳过该参数\n")
            continue
        baseline_df_print = pd.concat(baseline_data_print, ignore_index=True)

        # 1. 打印分表
        summarize_baselines(baseline_df_print, rank_base_list=["MASE"])

        # 2. 保存结果，用于打印跨参数对比表
        df = baseline_df_print.rename(
            columns={
                "eval_metrics/MASE[0.5]": "MASE",
                "eval_metrics/mean_weighted_sum_quantile_loss": "CRPS",
                "eval_metrics/sMAPE[0.5]": "sMAPE",
            }
        )

        df["RANK"] = df.groupby("dataset")["MASE"].rank(method="min", ascending=True)

        avg_rank = df.groupby("model")["RANK"].mean()
        avg_mase = df.groupby("model")["MASE"].mean()
        avg_smape = df.groupby("model")["sMAPE"].mean() if "sMAPE" in df.columns else None

        for model_name in avg_rank.index:
            abbrev = Model_abbrev_map.get(model_name, model_name)

            model_column_name = f"{abbrev}_c_{context_len}"# 使用缩写 + context_len 作为列名

            if model_column_name not in baseline_rank_summary_all["RANK"].columns:
                baseline_rank_summary_all["RANK"][model_column_name] = np.nan

            baseline_rank_summary_all["RANK"].loc["Global", model_column_name] = avg_rank[model_name]
            baseline_rank_summary_all["RANK"].loc["MASE", model_column_name] = avg_mase[model_name]
            if avg_smape is not None:
                baseline_rank_summary_all["RANK"].loc["sMAPE", model_column_name] = avg_smape[model_name]

    print("\n" + "=" * 60 + "\n📊 Baseline 所有参数取值对比汇总表\n" + "=" * 60)
    print(
        tabulate(
            baseline_rank_summary_all["RANK"],
            headers="keys",
            tablefmt="plain",
            floatfmt=".3f",
            numalign="decimal",
            stralign="left",
        )
    )

    # =========================
    # 2️⃣ selector 对比
    # =========================

    # # 获取全部 baseline TSFM模型结果
    # baseline_model_folders = All_model_names
    # print("baseline_model_folders", baseline_model_folders)
    #
    #
    # baseline_data = []
    # for model_name in baseline_model_folders:
    #     file_path = Path(results_dir) / model_name / "cl_512" /  "all_results.csv"
    #     if file_path.exists():
    #         print(f"\n🔹加载 baseline : {model_name}",end=" ")
    #         df = check_results_file(file_path, args.verbose)
    #         if df is not None:
    #             df["model"] = model_name
    #             baseline_data.append(df)
    #     else:
    #         print(f"❌ 未找到original文件: {file_path}\n")
    #
    #
    # baseline_df = standardize_model_names(baseline_data)
    #
    # rank_summary_all = {"RANK": pd.DataFrame()}
    #
    # # 遍历各个模型和参数下的结果
    # # for model_size in ['chronos_bolt_tiny','moirai_small','timesfm_1.0','timesfm_2.0','visionts_base','sundial_base']:
    # for model_size in ['timesfm_1.0','timesfm_2.0','timesfm_2.5']:
    # #     for context_len in [36,96,512,1024,2048,4096]:
    #     for context_len in [96,256,512,1024,2048,5000,'original']:
    #
    #         result_file = Path(results_dir) / model_size / f"cl_{context_len}" / f"all_results.csv"
    #         if not result_file.exists():
    #             print(f"❌ 文件缺失: {result_file}")
    #             continue
    #         result_df = check_results_file(result_file, args.verbose)
    #         if result_df is None:
    #             continue
    #         result_datasets = set(result_df['dataset'].unique())
    #         baseline_datasets = set(baseline_df['dataset'].unique())
    #         common_datasets = result_datasets & baseline_datasets
    #         # print(f"📊 与 baseline 重合数据集数量: {len(common_datasets)}")
    #         baseline_subset = baseline_df[baseline_df['dataset'].isin(common_datasets)].copy()
    #
    #         # 模型缩写
    #         model_family, model_variant = model_size.split('_', 1)
    #         abbreviation = Model_zoo_details[model_family][model_variant]["abbreviation"]
    #         model_column_name = f"{abbreviation}_c_{context_len}"
    #
    #         zoo_subset = process_results(result_file, model_column_name, common_datasets, verbose=True)
    #
    #         # 统计每个 group 的数据集数量
    #         dataset_counts_by_group = {}
    #         zoo_grouped = zoo_subset.copy()
    #
    #         for group_name in ['ds_freq', 'term', 'domain', 'num_variates']:
    #             if group_name == 'num_variates':
    #                 zoo_grouped['num_variates_group'] = zoo_grouped['num_variates'].apply(
    #                     lambda x: '=1' if x == 1 else '>1')
    #                 for group_val in ['=1', '>1']:
    #                     subset = zoo_grouped[zoo_grouped['num_variates_group'] == group_val]
    #                     dataset_counts_by_group[f"{group_name}:{group_val}"] = subset['dataset'].nunique()
    #             else:
    #                 for group_val in sorted(zoo_grouped[group_name].dropna().unique()):
    #                     subset = zoo_grouped[zoo_grouped[group_name] == group_val]
    #                     dataset_counts_by_group[f"{group_name}:{group_val}"] = subset['dataset'].nunique()
    #
    #         dataset_counts_by_group['Global'] = zoo_grouped['dataset'].nunique()
    #         dataset_counts_by_group['MASE'] = zoo_grouped['MASE'].nunique()
    #         dataset_counts_by_group['sMAPE'] = zoo_grouped['sMAPE'].nunique()
    #
    #
    #
    #         rank_summary_all = add_selector_rank(
    #             baseline_subset=baseline_subset,
    #             subset_df=zoo_subset,
    #             model_name=model_column_name,
    #             rank_summary_all=rank_summary_all,
    #             add_index=0,  # 插入位置
    #             k_order=5, #计算order指标的Topk值
    #             df_real=None,
    #             rank_base="MASE",# rank_base 可以自由切换：'MASE' / 'sMAPE' / 'CRPS'
    #             include_selector_in_rank=False,  # 控制 selector 是否参与 baseline 排名
    #         )
    #
    # # 最终格式化并打印汇总表
    # rank_summary_all = format_rank_summary_all(rank_summary_all, dataset_counts_by_group)
    #
    # print("\n" + "=" * 60 + "\n📊 对比汇总表格\n" + "=" * 60)
    # for rank_type, df_summary in rank_summary_all.items():
    #     print(f"\n📈 Rank Type: {rank_type}")
    #
    #     # 隐藏随机结果的详细列，只保留平均值列
    #     random_cols = [col for col in df_summary.columns if col.startswith('Rt') and not col.endswith('m')]
    #     cols_to_show = [col for col in df_summary.columns if col not in random_cols]
    #     df_summary_to_print = df_summary[cols_to_show]
    #
    #     # 5. 打印表格
    #     print(tabulate(
    #         df_summary_to_print,
    #         headers="keys",
    #         tablefmt="plain",  # 比 markdown 更紧凑，无任何边框或分隔符
    #         floatfmt=".3f",
    #         numalign="decimal",  # 数字对齐小数点，更美观
    #         stralign="left"
    #     ))


