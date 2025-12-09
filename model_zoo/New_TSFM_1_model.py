# -*- coding: utf-8 -*-
"""

可参考 Chronos / Moirai / Sundial / TimesFM 等已有模型的写法，在此基础上补充一个新的 TSFM 模型。
TODO：将模型文件名 New_TSFM_1_model.py 和类名 NewModel,NewModelPredictor 修改为实际的模型名字

整体结构约定：
- NewModel(BaseModel)    : 负责接收全局 args / 路径，并实例化 NewModelPredictor
- NewModelPredictor      : 负责具体的预测逻辑（加载模型、处理 context / pred_len / freq / NaN 等）

"""

import os
import logging
from typing import List

import numpy as np
import torch
from torch import cuda
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gluonts.model.forecast import QuantileForecast, SampleForecast
from gluonts.itertools import batcher

from model_zoo.base_model import BaseModel
from utils.missing import fill_missing

# TODO: （可选）如果新模型需要自己的工具函数或源代码，请在 model_zoo.TSFM_src 下创建对应文件，然后在这里导入，例如：
# from model_zoo.TSFM_src.timesfm.configs import ForecastConfig
# from model_zoo.TSFM_src.chronos_utils import SeriesDataset, identity_collate


# ====================== GluonTS 日志过滤 ======================

class WarningFilter(logging.Filter):

    def __init__(self, text_to_filter: str):
        super().__init__()
        self.text_to_filter = text_to_filter

    def filter(self, record):
        return self.text_to_filter not in record.getMessage()


gts_logger = logging.getLogger("gluonts.model.forecast")
gts_logger.addFilter(
    WarningFilter("The mean prediction is not stored in the forecast data")
)


# ====================== 模型封装层：NewModel ======================

class NewModel(BaseModel):
    """
    通用模型封装：
    - 负责接收 args / module_name / model_name / model_local_path
    - 负责在 get_predictor 中实例化 NewModelPredictor，并把通用参数传进去
    """

    def __init__(self, args, module_name, model_name, model_local_path):
        self.args = args
        self.module_name = module_name
        self.model_name = model_name
        self.model_local_path = model_local_path
        self.output_dir = os.path.join(self.args.output_dir, self.model_name)

        super().__init__(self.model_name, args, self.output_dir)

    def get_predictor(self, dataset, batch_size):
        """
        dataset: 数据集对象，通常包含：
            - prediction_length
            - freq等

        这里仅负责把 dataset 中的关键信息传递给 Predictor，具体如何使用由 NewModelPredictor 决定。
        """
        predictor = NewModelPredictor(
            config=self.args,
            batch_size=batch_size,
            model_path=self.model_local_path,
            prediction_length=dataset.prediction_length,
            ds_freq=dataset.freq,
            # TODO: （可选）如有需要，可传递更多 dataset 属性
        )
        return predictor


# ====================== 预测器层：NewModelPredictor ======================

class NewModelPredictor:
    """
    该类负责：
    - 加载具体的 TSFM 模型（从本地 model_path）
    - 在 __init__ 中处理：
        * device / GPU 设置
        * pred_len (prediction_length)
        * context_len（是否使用 args.fix_context_len）
        * freq 映射 / 其他模型特有超参数
    - 在 predict 中处理：
        * 批量遍历 test_data_input
        * 对每条序列进行：
            - context 截断（依据 context_len）
            - 缺失值 / NaN 处理（如需）
            - freq 处理（如需）
        * 调用底层模型进行预测，并将结果封装为 GluonTS 的 Forecast 对象
    """

    def __init__(
        self,
        config,
        batch_size: int,
        model_path: str,
        prediction_length: int,
        ds_freq: str,
        target_dim: int = 1,
        past_feat_dynamic_real_dim: int = 0,
        *args,
        **kwargs,
    ):
        # ===== 1) 通用属性保存 =====
        self.config = config               # 全局运行配置（包含 fix_context_len / context_len / num_workers 等）
        self.batch_size = batch_size
        self.model_path = model_path       # 本地模型权重路径（离线环境）
        self.prediction_length = prediction_length
        self.ds_freq = ds_freq             # 数据集频率字符串（如 "H" / "15min" 等）
        self.target_dim = target_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim

        # 设备选择：默认优先用 GPU
        self.device = torch.device("cuda" if cuda.is_available() else "cpu")

        # ===== 2) 模型加载 =====
        self.model = None  # TODO: 加载具体模型实现，并移动到 self.device

        # ===== 3) context_len 处理 =====
        if getattr(self.config, "fix_context_len", False):
            self.context_length = self.config.context_len
        else:
            self.context_length = 2048  # TODO: 根据模型默认的 context 长度进行设定，例如 4000 / 2048 / 15360 等

        # ===== 4) freq / 缺失值 / 量化分位点等，可以在此统一设定 =====
        # TODO: （可选）如果模型需要 freq 映射（类似 TimesFM），在这里将 ds_freq -> 内部 freq
        # TODO: （可选）如果模型输出的是 quantile 预测，请在这里定义 quantiles 列表，并在最后使用 QuantileForecast
        #
        # 示例：
        #   self.freq = some_freq_mapping(self.ds_freq)
        #   self.quantiles = [0.1 * i for i in range(1, 10)]
        #
        self.freq = ds_freq
        self.quantiles = None

        context_info = (
            self.context_length
            if getattr(self.config, "fix_context_len", False)
            else "full_history"
        )
        print(
            f"[NewModel] context_len={context_info}, "
            f"freq_in={self.ds_freq}, "
            f"impute_missing=TODO"   # 提示：是否进行了缺失值处理
        )

    # =========================================================================
    # 预测主函数：需要补充内部逻辑
    # =========================================================================
    def predict(self, test_data_input: List[dict], batch_size: int = None) -> List:
        """
        参数说明：
        ----------
        test_data_input:
            形如 List[{"start": <Period>, "target": np.ndarray}, ...] 的列表。
        batch_size:
            可选，若为 None，则使用初始化时传入的 self.batch_size。

        返回值：
        ----------
        List[Forecast]，其中 Forecast 通常为：
            - SampleForecast（样本形式）
            - 或 QuantileForecast（分位数形式）
        """
        if batch_size is None:
            batch_size = self.batch_size

        # TODO: 1) 使用 gluonts.itertools.batcher 进行分批，可以参考 Chronos / Sundial / TimesFM 的实现方式

        forecasts = []  # 最终返回的 GluonTS Forecast 列表

        # 示例：
        for batch in tqdm(
                batcher(test_data_input, batch_size=self.batch_size),
                total=len(test_data_input) // self.batch_size,
                desc="New Model Predict"):

            # --------------------------------------------------
            # TODO: 2) 逐样本处理 target：
            #   - 处理缺失值 / NaN：
            #       * 可调用已实现的工具函数 fill_missing
            #   - 按照 context_length 截取最后 context_length 步
            # --------------------------------------------------

            # 示例占位代码：
            context_batch = []
            for entry in batch:
                raw = np.array(entry["target"], dtype=float)

                # 处理缺失值示例
                raw = fill_missing(
                                    raw,
                                    all_nan_strategy_1d="linspace",
                                    interp_kind_1d="nearest",
                                    add_noise_1d=True,
                                    noise_ratio_1d=0.01,
                                )

                # context 截取示例
                if getattr(self.config, "fix_context_len", False):
                    arr = raw[-self.context_length :]
                else:
                    arr = raw

                context_batch.append(arr)

            # --------------------------------------------------
            # TODO: 3) 调用底层模型进行预测：
            #   - 将 context_batch 转为 torch.Tensor 并移动到 self.device
            #   - 调用 self.model 进行前向推理，得到未来 prediction_length 步的预测
            #   - 注意 pred_len / horizon 的对齐
            # --------------------------------------------------

            # 示例占位代码：
            #   context_tensor = torch.tensor(context_batch, device=self.device, dtype=torch.float32)
            #   model_outputs = self.model(...)

            model_outputs = None  # TODO: 替换为真实预测结果（形状参考你选用的 Forecast 类型）

            # --------------------------------------------------
            # TODO: 4) 将模型输出转换为 GluonTS Forecast 对象：
            #   - 如果是样本形式：[B, num_samples, H] -> 使用 SampleForecast
            #   - 如果是分位数形式：[B, Q, H]      -> 使用 QuantileForecast
            #   - 注意设置正确的 start_date：ts["start"] + len(ts["target"])
            #   - （可选）对预测结果做 NaN 检查，及时报错定位问题
            # --------------------------------------------------

            # 示例占位逻辑（需要学生根据模型输出改写）：
            for i, ts in enumerate(batch):
                forecast_start_date = ts["start"] + len(ts["target"])

                # TODO: 4.1 从 model_outputs 中取出第 i 条的预测数组 pred_i
                pred_i = None  # np.ndarray

                # TODO: 4.2 NaN 检查
                # if np.isnan(pred_i).any():
                #     raise ValueError(f"[NaN DEBUG] NewModel 第 {global_idx}-th 序列预测含 NaN")

                # TODO: 4.3 根据模型实际能力，选择 Forecast 类型
                # 示例一：分位数形式
                # forecasts.append(
                #     QuantileForecast(
                #         forecast_arrays=pred_i,
                #         forecast_keys=list(map(str, self.quantiles)),
                #         start_date=forecast_start_date,
                #     )
                # )
                #
                # 示例二：样本形式
                # forecasts.append(
                #     SampleForecast(
                #         samples=pred_i,
                #         start_date=forecast_start_date,
                #     )
                # )

        return forecasts
