# Copyright (c) 2023, Salesforce, Inc.
# Copyright (c) 2025 fireball0213, LAMDA, Nanjing University
# SPDX-License-Identifier: Apache-2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE:
#   This file is based on Salesforce's original implementation and has been
#   modified by fireball0213 (LAMDA, Nanjing University). The added utilities
#   include GluonTS <-> NumPy conversion functions and local forecast loading.

import os
import math
from functools import cached_property
from enum import Enum
from pathlib import Path
from typing import Iterable, Iterator, List, Tuple
from gluonts.model.forecast import SampleForecast, QuantileForecast
import json
import numpy as np
import pandas as pd
import datasets
from dotenv import load_dotenv
from gluonts.dataset import DataEntry
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import TestData, TrainingDataset, split
from gluonts.itertools import Map
from gluonts.time_feature import norm_freq_str
from gluonts.transform import Transformation
from pandas.tseries.frequencies import to_offset
import pyarrow.compute as pc
from toolz import compose

# ====================== 常量定义 ======================

# 测试集比例，用于滑窗数量计算
TEST_SPLIT = 0.1

# 最大窗口数（防止窗口数量过大）
MAX_WINDOW = 20

# M4 数据集的预测长度映射
M4_PRED_LENGTH_MAP = {
    "A": 6,
    "Q": 8,
    "M": 18,
    "W": 13,
    "D": 14,
    "H": 48,
}

# 通用预测长度映射（非 M4）
PRED_LENGTH_MAP = {
    "M": 12,
    "W": 8,
    "D": 30,
    "H": 48,
    "T": 48,
    "S": 60,
}

# TFB 相关的预测长度映射
TFB_PRED_LENGTH_MAP = {
    "A": 6,
    "H": 48,
    "Q": 8,
    "D": 14,
    "M": 18,
    "W": 13,
    "U": 8,
    "T": 8,
}


class Term(Enum):
    """预测 term（短 / 中 / 长期），通过 multiplier 控制预测长度倍数。"""
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"

    @property
    def multiplier(self) -> int:
        if self == Term.SHORT:
            return 1
        elif self == Term.MEDIUM:
            return 10
        elif self == Term.LONG:
            return 15


def itemize_start(data_entry: DataEntry) -> DataEntry:
    data_entry["start"] = data_entry["start"].item()
    return data_entry


def maybe_reconvert_freq(freq: str) -> str:
    """
    如果 freq 属于新版 pandas 的频率字符串，则转换为旧频率表示。

    主要用于兼容一些老代码/库中使用的 A/Q/M/T/U 等频率缩写。
    """
    deprecated_map = {
        "Y": "A",
        "YE": "A",
        "QE": "Q",
        "ME": "M",
        "h": "H",
        "min": "T",
        "s": "S",
        "us": "U",
    }
    if freq in deprecated_map:
        return deprecated_map[freq]
    return freq


class MultivariateToUnivariate(Transformation):
    """
    将多变量 target 转换为多条单变量序列：
    - 输入：某条样本 target 形状为 (C, T)
    - 输出：拆分为 C 条样本，每条 target 形状为 (T,) 或 (1, T)，item_id 加上 "_dim{i}"
    """
    def __init__(self, field):
        self.field = field

    def __call__(
            self, data_it: Iterable[DataEntry], is_train: bool = False
    ) -> Iterator:
        for data_entry in data_it:
            item_id = data_entry["item_id"]
            val_ls = list(data_entry[self.field])
            for id, val in enumerate(val_ls):
                univariate_entry = data_entry.copy()
                univariate_entry[self.field] = val
                univariate_entry["item_id"] = item_id + "_dim" + str(id)
                yield univariate_entry


class Dataset:
    """
    name:
        本地 datasets 的子目录名，即 load_from_disk(storage_path / name) 中的 name。
    term:
        预测 term，可为 Term 枚举或字符串 "short"/"medium"/"long"。
    to_univariate:
        是否将多变量序列拆成多条单变量序列。
    storage_env_var:
        环境变量名，用于指定项目根路径，例如 "Project_Path"。
        最终数据集路径为: $Project_Path/Dataset_Path/{name}
    """
    def __init__(
            self,
            name: str,
            term: Term | str = Term.SHORT,
            to_univariate: bool = False,
            storage_env_var: str = "Project_Path",
    ):
        load_dotenv()
        storage_path = Path(os.getenv(storage_env_var)) / "Dataset_Path"
        self.hf_dataset = datasets.load_from_disk(str(storage_path / name)).with_format(
            "numpy"
        )
        process = ProcessDataEntry(
            self.freq,
            one_dim_target=self.target_dim == 1,
        )

        self.gluonts_dataset = Map(compose(process, itemize_start), self.hf_dataset)
        if to_univariate:
            self.gluonts_dataset = MultivariateToUnivariate("target").apply(
                self.gluonts_dataset
            )

        self.term = Term(term)
        self.name = name

    @cached_property
    def prediction_length(self) -> int:
        """
        推断单步预测长度（基础值），再乘以 term.multiplier 得到最终 pred_len。
        """
        freq = norm_freq_str(to_offset(self.freq).name)
        freq = maybe_reconvert_freq(freq)
        pred_len = (
            M4_PRED_LENGTH_MAP[freq] if "m4" in self.name else PRED_LENGTH_MAP[freq]
        )
        return self.term.multiplier * pred_len

    @cached_property
    def freq(self) -> str:
        """数据集频率字符串（例如 'H', '15T', 'D' 等）。"""
        return self.hf_dataset[0]["freq"]

    @cached_property
    def target_dim(self) -> int:
        """目标维度：多变量时为通道数 C，单变量时为 1。"""
        target = self.hf_dataset[0]["target"]
        return target.shape[0] if target.ndim > 1 else 1

    @cached_property
    def past_feat_dynamic_real_dim(self) -> int:
        """历史动态特征的维度数（如果不存在则为 0）。"""
        first = self.hf_dataset[0]
        if "past_feat_dynamic_real" not in first:
            return 0

        past_feat_dynamic_real = first["past_feat_dynamic_real"]
        return past_feat_dynamic_real.shape[0] if past_feat_dynamic_real.ndim > 1 else 1

    @cached_property
    def windows(self) -> int:
        """
        在测试区间内可生成的滑窗数量。
        - M4 默认只取 1 个 window；
        - 其他数据集根据 TEST_SPLIT 和 pred_len 自动估计，但不会超过 MAX_WINDOW。
        """
        if "m4" in self.name:
            return 1
        w = math.ceil(TEST_SPLIT * self._min_series_length / self.prediction_length)
        return min(max(1, w), MAX_WINDOW)

    @cached_property
    def _min_series_length(self) -> int:
        """所有序列中最短的长度，用于估计可用窗口数。"""
        column = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(column, 0, 1))
            )
        else:
            lengths = pc.list_value_length(column)
        return int(min(lengths.to_numpy()))

    @cached_property
    def sum_series_length(self) -> int:
        """所有序列长度之和（用于统计/日志）。"""
        column = self.hf_dataset.data.column("target")
        if self.hf_dataset[0]["target"].ndim > 1:
            lengths = pc.list_value_length(pc.list_flatten(column))
        else:
            lengths = pc.list_value_length(column)
        return int(sum(lengths.to_numpy()))

    # --------- GluonTS split 接口 ---------

    @property
    def training_dataset(self) -> TrainingDataset:
        """训练集：去掉 windows+1 个预测窗口，保留前面的历史。"""
        training_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * (self.windows + 1),
        )
        return training_dataset

    @property
    def validation_dataset(self) -> TrainingDataset:
        """验证集：在最后 windows 个预测窗口之前切分。"""
        validation_dataset, _ = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        return validation_dataset

    @property
    def test_data(self) -> TestData:
        """
        测试数据：基于最后 windows 个窗口，生成测试实例。
        每个实例的 prediction_length 与 Dataset.prediction_length 一致。
        """
        _, test_template = split(
            self.gluonts_dataset,
            offset=-self.prediction_length * self.windows,
        )
        test_data = test_template.generate_instances(
            prediction_length=self.prediction_length,
            windows=self.windows,
            distance=self.prediction_length,
        )
        return test_data

# ====================== GluonTS <-> NumPy 工具 （added by fireball0213）======================

def gluonts_to_numpy(input_dataset: List[SampleForecast]):
    """
    将 GluonTS 格式的数据集 (list of dicts, target.shape=(C, T))
    转换成 numpy 格式 (N, T, C)
    """
    data_list: List[np.ndarray] = []
    for forecast in input_dataset:
        # forecast.samples: (C, T) -> 转为 (T, C)
        data_list.append(forecast.samples.T)

    data_array = np.stack(data_list, axis=0)  # (N, T, C)
    # print(f"✅ 转换完成，输出形状: {data_array.shape}")
    return data_array

def load_forecasts_from_npy(
    samples_path: str,
    meta_path: str,
    freq: str,
) -> List[SampleForecast]:
    """
    从 samples.npy + meta.json 恢复 SampleForecast 列表。

    参数
    ----
    samples_path: .npy 文件路径
    meta_path: .json 元数据文件路径
    freq: 时间频率字符串，例如 "15T", "H", "D" 等。

    返回
    ----
    List[SampleForecast]
    """
    samples = np.load(samples_path)  # (N_series, N_samples, pred_len, C)

    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    if len(meta) != samples.shape[0]:
        raise ValueError("样本数与元数据条目数不匹配")

    forecasts: List[SampleForecast] = []
    for idx, info in enumerate(meta):
        item_id = info["item_id"]
        start_date = pd.Period(info["start_date"], freq=freq)
        sample_arr = samples[idx]  # (num_samples, pred_len, C)

        sf = SampleForecast(
            samples=sample_arr,
            start_date=start_date,
            item_id=item_id,
        )
        forecasts.append(sf)

    return forecasts


def load_gluonts_pred(base_path: str, model_name: str, model_cl_name: str, dataset_name: str, pred_len, channels, windows, verobse=False):
    """
    从指定文件夹加载样本和元数据，恢复为 GluonTS 的 SampleForecast 列表。
    """
    # 1) 构建路径
    model_folder = os.path.join(base_path, model_name)
    model_cl_folder = os.path.join(model_folder, model_cl_name)
    samples_path = os.path.join(model_cl_folder, f"{dataset_name}_samples.npy")
    meta_path = os.path.join(model_cl_folder, f"{dataset_name}_meta.json")

    # 2) 载入样本数组
    samples = np.load(samples_path)
    if verobse:
        print(f"Load {model_name} pred: {samples.shape}", end=" ")

    # 3) 载入元数据
    with open(meta_path, "r") as fp:
        meta = json.load(fp)

    entries = meta.get("entries", None)
    performance = meta.get("performance", None) or {}
    runtime_seconds = performance.get("runtime_seconds", None)

    # 4) 对样本数组取中位数，作为最终预测，注意不是0.5分位
    median_forecast = np.median(samples, axis=1)  # (N_series *N_channels, pred_len)

    if verobse:
        print('median_forecast:', median_forecast.shape, end=" ")

    if median_forecast.ndim == 2:
        # 单通道情况，自动补一维
        # median_forecast = median_forecast[..., np.newaxis]
        #
        # #仅已知C的情况下，将（N * C，T）安全转换为（N，T，C），且按通道优先排布
        # N = median_forecast.shape[0] // channels
        # median_forecast = np.concatenate(
        #     [median_forecast[c * N: (c + 1) * N][..., None] for c in range(channels)],
        #     axis=-1)
        # median_forecast=median_forecast[:,:,0,:]# → (N, T, C)

        # 错误转换
        # median_forecast = median_forecast.reshape(-1, median_forecast.shape[1], channels)  # (N_series, pred_len, C)

        # 正确转换，考虑series、windows等信息
        median_forecast = median_forecast.reshape(-1, channels, windows, pred_len)  # [series, variates, windows, pred_len]
        median_forecast = median_forecast.transpose(0, 2, 3, 1)  # → [series, windows, pred_len, variates]
        median_forecast = median_forecast.reshape(-1, pred_len, channels)  # → (N_series, pred_len, C)

    elif median_forecast.ndim == 3:  # moirai 多变量情况下直接保持 (N_series, pred_len, C)
        pass
    if verobse:
        print('to ', median_forecast.shape, end=" ")
        # print(median_forecast[:,:,0])#第一个channel

    # 5) 推断频率
    freq = dataset_name.split("_")[-2]

    # 6) 构造 SampleForecast 列表
    pred_dataset: List[SampleForecast] = []
    n_series = median_forecast.shape[0]

    for idx in range(n_series):
        if entries is None or idx >= len(entries):
            raise ValueError(f"Entry 数量不足，无法匹配样本 {idx}。")

        info = entries[idx]
        item_id = str(info["item_id"])
        start = pd.Period(info["start_date"], freq=freq)

        target_array = median_forecast[idx]  # (pred_len, C)
        if target_array.ndim != 2:
            raise ValueError(f"预期 target_array 为 2 维，实际为: {target_array.shape}")

        # GluonTS 约定：(C, T)
        target = target_array.T

        forecast = SampleForecast(
            samples=target,
            start_date=start,
            item_id=item_id,
        )
        pred_dataset.append(forecast)

    if verobse and pred_dataset:
        print(
            f"恢复为：样本数 {len(pred_dataset)}，"
            f"target shape: {pred_dataset[0].samples.shape}"
        )

    return pred_dataset, samples, runtime_seconds


def numpy_to_gluonts(data_array, template_dataset):
    """
    将 NumPy 数组 (N, T, C) 转换回 SampleForecast 列表。
    """
    forecasts: List[SampleForecast] = []
    N, T, C = data_array.shape

    if len(template_dataset) != N:
        raise ValueError(
            f"template_dataset 数量 ({len(template_dataset)}) 与 N ({N}) 不匹配。"
        )

    for i in range(N):
        base_sample = template_dataset[i]
        item_id = base_sample.item_id
        start_date = base_sample.start_date

        # data_array[i]: (T, C)
        sample_array = data_array[i]

        # GluonTS 中 SampleForecast.samples 通常为 (num_samples, T) 或 (num_samples, T, C)
        if C > 1:
            sample_array = sample_array.reshape(1, T, C)
        else:
            sample_array = sample_array.reshape(1, T)

        forecast = SampleForecast(
            samples=sample_array.astype(np.float32),
            start_date=start_date,
            item_id=item_id,
        )
        forecasts.append(forecast)

    # print(f"✅ 逆转换完成，生成 SampleForecast 数量: {len(forecasts)}")
    return forecasts
