import os
import csv
import json
import time
import warnings
import random

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from datetime import datetime

from gluonts.model import evaluate_forecasts
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MASE,
    SMAPE,
    MeanWeightedSumQuantileLoss,
)

from data import Dataset
from utils.debug import debug_check_input_nan, debug_print_test_input, debug_forecasts

from Model_Path.model_zoo_config import Model_zoo_details, MULTIVAR_TSFM_PREFIXES

warnings.filterwarnings("ignore")
load_dotenv()  # 加载环境变量

# ====================== 全局配置与常量 ======================

# 数据集属性（domain、变量维度、freq 等）
dataset_properties_map = json.load(open("Dataset_Path/dataset_properties.json", encoding="utf-8"))

# 评估指标集合
metrics = [
    MASE(),
    SMAPE(),
    MeanWeightedSumQuantileLoss(
        quantile_levels=[0.1 * i for i in range(1, 10)]
    ),
]

# 数据集重命名
pretty_names = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


class BaseModel:
    def __init__(self, model_name, args, output_dir=None):
        self.args = args
        self.model_name = model_name
        if output_dir is None:
            self.output_dir = args.output_dir
        self.batch_size = args.batch_size

        self.get_save_path()
        print('Save Path: ',self.csv_file_path)

        self.done_datasets = []
        if self.args.skip_saved:
            if os.path.exists(self.csv_file_path):
                df_res = pd.read_csv(self.csv_file_path)
                if "dataset" in df_res.columns:
                    self.done_datasets = df_res["dataset"].values

                    print(f"Done {len(self.done_datasets)} datasets")
            else:
                print(f"[skip_saved] 结果文件不存在，忽略已完成数据集检测：{self.csv_file_path}")

    # ==============================================================
    # 路径构造与 CSV 头部
    # ==============================================================
    def get_save_path(self):

        os.makedirs(self.output_dir, exist_ok=True)

        if self.args.fix_context_len:
            self.output_dir = os.path.join(self.output_dir, f"cl_{self.args.context_len}")
        else:
            self.output_dir = os.path.join(self.output_dir, f"cl_original")

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.csv_file_path = os.path.join(self.output_dir, "all_results.csv")

        header = [
            "dataset",
            "model",
            "eval_metrics/MASE[0.5]",
            "eval_metrics/sMAPE[0.5]",
            "eval_metrics/mean_weighted_sum_quantile_loss",
            "domain",
            "num_variates",
            "model_order"
        ]

        if not os.path.exists(self.csv_file_path):
            with open(self.csv_file_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

    def get_predictor(self, dataset, batch_size):
        """
        子类实现：加载模型，返回 predictor 对象
        """
        raise NotImplementedError("子类必须实现 get_predictor 方法")

    def _build_ds_meta(self, ds_name, term):
        """统一解析 ds_name，返回 ds_key, ds_freq, ds_config, dataset_name"""
        if "/" in ds_name:
            ds_key_raw, ds_freq = ds_name.split("/")
            ds_key = pretty_names.get(ds_key_raw.lower(), ds_key_raw.lower())
        else:
            ds_key_raw = ds_name
            ds_key = pretty_names.get(ds_key_raw.lower(), ds_key_raw.lower())
            ds_freq = dataset_properties_map[ds_key]["frequency"]

        ds_config = f"{ds_key}/{ds_freq}/{term}"
        dataset_name = f"{ds_key}_{ds_freq}_{term}"
        return ds_key, ds_freq, ds_config, dataset_name

    def _decide_univariate(self, ds_name, term):
        """根据模型类型和数据维度，决定是否转为单变量"""
        prefix = self.model_name.split("_")[0].lower()
        if (
            prefix in MULTIVAR_TSFM_PREFIXES
            or self.model_name.split("_")[-1] == "Select"
        ):
            return False

        # 先用 to_univariate=False 探测原始 target_dim 决定是否需要转为单变量
        tmp_ds = Dataset(name=ds_name, term=term, to_univariate=False)
        return tmp_ds.target_dim != 1

    def _make_forecasts(self, dataset, dataset_name, ds_config, fixed_model_order, debug_mode):
        """
        统一的预测入口：
        - 选择器：返回 (forecasts, model_order)
        - 非选择器：内部处理 OOM 重试、debug 打印、NaN 检查和噪声注入
        """
        model_order = None

        batch_size = self.batch_size
        while True:
            try:
                predictor = self.get_predictor(dataset, batch_size)
                test_input = dataset.test_data.input

                if debug_mode:
                    # 打印数据格式
                    debug_print_test_input(dataset)
                    debug_check_input_nan(test_input)

                input_data = test_input
                forecasts = list(
                    tqdm(
                        predictor.predict(input_data),
                        total=len(dataset.test_data.input),
                        desc=f"Predicting {ds_config}",
                    )
                )
                break
            except torch.cuda.OutOfMemoryError:
                print(
                    f"⚠️ OOM at batch_size {batch_size}, "
                    f"reducing to {batch_size // 2}"
                )
                batch_size //= 2

        if debug_mode:
            debug_forecasts(forecasts)

        return forecasts, model_order

    # ==============================================================
    # 主运行流程
    # ==============================================================

    def run(self):
        total_time = 0
        total_memory = 0
        max_memory = 0
        fixed_model_order = None
        print(f"🚀 Running {self.model_name}", )

        if len(self.done_datasets) > 0 and self.args.skip_saved:
            print(f"✅  Skipping...✅  Done with {len(self.done_datasets)} datasets. ")
        self.all_data_configs = []
        for ds_name in self.args.all_datasets:
            terms = ["short", "medium", "long"]
            for term in terms:
                # 中长 term 只对指定数据集生效
                if (term in ["medium", "long"]) and (ds_name not in self.args.med_long_datasets.split()):
                    continue

                # 统一构造 ds_key / ds_freq / ds_config / dataset_name
                ds_key, ds_freq, ds_config, dataset_name = self._build_ds_meta(ds_name, term)
                self.all_data_configs.append(ds_config)

                if ds_config in self.done_datasets and getattr(self.args, "skip_saved", False):
                    print(f"{ds_config}.", end=" ✅  ")
                    continue
                else:
                    print(f"\n🚀 Dataset: [{ds_config}]",
                          f"Model: {self.model_name}",
                          'GPU:',os.environ.get('CUDA_VISIBLE_DEVICES', 'None'),
                          'Batch_size:', self.batch_size,
                          'num_workers:', self.args.num_workers
                          )
                # ---------- 是否转为单变量 ----------
                to_univariate = self._decide_univariate(ds_name, term)
                dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)
                start_time = time.time()

                forecasts, model_order = self._make_forecasts(
                    dataset=dataset,
                    dataset_name=dataset_name,
                    ds_config=ds_config,
                    fixed_model_order=fixed_model_order,
                    debug_mode=self.args.debug_mode,
                )
                res = evaluate_forecasts(
                    forecasts=forecasts,
                    test_data=dataset.test_data,
                    metrics=metrics,
                    batch_size=1024,
                    axis=None,
                    mask_invalid_label=True,
                    allow_nan_forecast=False,
                    seasonality=get_seasonality(dataset.freq),
                )

                # ===================== 记录耗时和显存 =====================
                end_time = time.time()
                elapsed = end_time - start_time
                reserved = torch.cuda.memory_reserved() / 1024 ** 2
                allocated = torch.cuda.memory_allocated() / 1024 ** 2
                memory_used = reserved + allocated

                max_memory= max(max_memory, memory_used)
                total_memory += memory_used
                total_time += elapsed

                print(f"time cost 🧭 {elapsed:.2f}s",
                      f"memory-use {memory_used:.0f} MB", end=' ')

                self.save_results(res, forecasts, ds_config, dataset_name, ds_key, elapsed, memory_used,dataset, model_order)

        # ===================== 运行结束后：统计总体性能并检查结果文件 =====================
        num_ds = len(self.all_data_configs)
        if num_ds-len(self.done_datasets) > 0 and self.args.save_pred:
            # 计算平均耗时，保留整数

            average_time = total_time / max(num_ds, 1)
            average_memory = total_memory / max(num_ds, 1)

            print(f"\n🧭 已运行{num_ds}个数据集，total_time:",f"{total_time:.2f}s","average_time:",f"{average_time:.2f}s",
                  "max_memory:",f"{max_memory:.0f} MB","average_memory:",f"{average_memory:.0f} MB \n",)


            # 保存整体时间统计到 CSV 文件
            time_save_filename = "results/runtime-TSFM.csv"

            if self.args.fix_context_len:
                context_tag = self.args.context_len
            else:
                context_tag = "original"
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file_exists = os.path.isfile(time_save_filename)

            row = {
                "model_name": self.model_name,
                "context_length": context_tag,
                "dataset_num": num_ds,  # 数据集数量
                "total_time_s": round(total_time, 0),  # 总耗时（秒）
                "average_time_s": round(average_time, 2),  # 平均耗时（秒）
                "average_memory_MB": round(average_memory, 0),  # 平均内存占用（MB）
                "timestamp": timestamp,  # 新增：时间戳
            }

            # 追加模式
            with open(time_save_filename, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=row.keys())
                if not file_exists:
                    writer.writeheader()  # 首次写文件时写入列名
                writer.writerow(row)

    # ==============================================================
    # 结果保存逻辑
    # ==============================================================
    def save_results(self, res, forecasts, ds_config, dataset_name, ds_key, elapsed, memory_used, dataset=None, model_order=None):
        if self.args.save_pred:
            formatted_model_order = '[' + " ".join(map(str, model_order)) + ']' if model_order is not None else ""

            row = [
                ds_config,
                self.model_name,
                res["MASE[0.5]"][0],
                res["sMAPE[0.5]"][0],
                res["mean_weighted_sum_quantile_loss"][0],
                dataset_properties_map[ds_key]["domain"],
                dataset_properties_map[ds_key]["num_variates"],
                formatted_model_order
            ]

            with open(self.csv_file_path, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row)

        if res is not None:
            print(
                f"{self.model_name}",
                f"MASE: {res['MASE[0.5]'][0]:.2f}",
                f"sMAPE: {res['sMAPE[0.5]'][0]:.2f}",
                f"CRPS: {res['mean_weighted_sum_quantile_loss'][0]:.2f}")

        # 保存预测结果到npy和json
        if self.args.save_pred and self.model_name.split("_")[-1] != "Select":
            # 1) 样本数组：shape = (num_series, num_samples, pred_len, num_channels)
            if hasattr(forecasts[0], "samples"):
                arrs = [fc.samples for fc in forecasts]
            elif hasattr(forecasts[0], "forecast_array"):
                arrs = [fc.forecast_array for fc in forecasts]
            else:
                print(f"forecasts[0] attributes: {dir(forecasts[0])}")
                raise ValueError("forecasts[0] does not have 'samples' or 'forecast_array' attribute")
            samples = np.stack(arrs, axis=0)

            samples_path = os.path.join(self.output_dir, f"{dataset_name}_samples.npy")

            np.save(samples_path, samples)


            # 2) 保存元数据 + 性能指标
            meta = {
                "performance": {
                    "runtime_seconds": elapsed,
                    "memory_use_mb": memory_used,
                    "batch_size": self.batch_size,
                },
                "entries": [
                    {
                        "item_id": fc.item_id,
                        "start_date": str(fc.start_date)
                    }
                    for fc in forecasts
                ]
            }
            meta_path = os.path.join(self.output_dir, f"{dataset_name}_meta.json")
            with open(meta_path, "w") as fp:
                json.dump(meta, fp)
            print(f"👉 预测结果npy和json元数据保存到 {self.output_dir}")
