import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

from Model_Path.model_zoo_config import Model_zoo_details, Model_abbrev_map, All_model_names
from utils.check_tools import check_results_file


def summarize_baselines(context_len_lst=None):
    """
    功能：
    - 对 baseline 内部按 rank_base 做 dataset 内排名，得到平均 RANK
    - 打印 sMAPE / MASE / CRPS / RANK 的全局均值表（列为模型，使用缩写）
    """

    baseline_rank_summary_all = {"RANK": pd.DataFrame()}

    for context_len in context_len_lst:
        baseline_data_print = []
        for model_name in All_model_names:
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

        # =========================
        # 1️⃣ 打印baseline 结果分表
        # =========================
        rank_base_list=["MASE"]
        df = baseline_df_print.copy()

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

        # =========================
        # 2️⃣ 打印跨参数对比表
        # =========================
        df = baseline_df_print
        df["RANK"] = df.groupby("dataset")["MASE"].rank(method="min", ascending=True)

        avg_rank = df.groupby("model")["RANK"].mean()
        avg_mase = df.groupby("model")["MASE"].mean()
        avg_smape = df.groupby("model")["sMAPE"].mean() if "sMAPE" in df.columns else None

        for model_name in avg_rank.index:
            abbrev = Model_abbrev_map.get(model_name, model_name)

            model_column_name = f"{abbrev}_c_{context_len}"  # 使用缩写 + context_len 作为列名

            if model_column_name not in baseline_rank_summary_all["RANK"].columns:
                baseline_rank_summary_all["RANK"][model_column_name] = np.nan

            baseline_rank_summary_all["RANK"].loc["Rank", model_column_name] = avg_rank[model_name]
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",type=str,default="results",help="结果根目录",)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    context_len_lst=[512] # TODO:评估更多 context_len 取值

    summarize_baselines(context_len_lst=context_len_lst)




