# run_model_zoo.py
import argparse
import numpy as np
import random
import torch
from torch.backends import cudnn
import os
import sys

sys.path.append(os.path.dirname(__file__))

import importlib
import warnings

warnings.filterwarnings("ignore")


def set_seed(seed):
    np.random.seed(seed=seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


from Dataset_Path.dataset_config import Med_long_Fast_datasets, Short_Fast_datasets
from Model_Path.model_zoo_config import Model_zoo_details, All_model_names
from selector.selector_config import Selector_zoo_details
from utils.check_tools import filter_models_by_key


def run_select(args):

    # 按release_date字段筛选出 <= select_date 的模型，并按照日期排序、编号 id
    Model_zoo_current, sorted_filtered_models = filter_models_by_key(Model_zoo_details, args.select_date, select_key="release_date")
    args.current_zoo_num = sum(len(sizes) for sizes in Model_zoo_current.values())  # 筛选后模型数
    print(f"日期{args.select_date}之前的模型族总数：{args.current_zoo_num} / {args.zoo_total_num}")

    select_name = args.models

    cfg = Selector_zoo_details.get(select_name, None)
    if cfg is None:
        raise ValueError(f"⚠️ 未知选择器 {select_name}，请在 selector_config.py 中补充配置")

    # 动态 import
    module = importlib.import_module(cfg["model_module"])
    SelectorClass = getattr(module, cfg["model_class"])
    model = SelectorClass(args, model_name=select_name, Model_zoo_current=Model_zoo_current, )

    model.run()


def main():
    parser = argparse.ArgumentParser(description="遍历模型和数据集")
    parser.add_argument('--seed', type=int, default=2025, help='Random seed')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--source_data', type=str, default=None, help='dataset type')
    parser.add_argument('--target_data', type=str, default=None, help='dataset type')
    parser.add_argument('--root_path', type=str, default=None, help='root path of the data file')
    parser.add_argument('--data_path', type=str, default=None, help='data file')
    parser.add_argument('--target', type=str, default='OT', help='name of target column')
    parser.add_argument('--scale', type=bool, default=True, help='scale the time series with sklearn.StandardScale()')
    parser.add_argument('--output_dir', type=str, default='results/', help='output dir')

    # model zoo
    parser.add_argument('--run_mode', type=str, default='zoo', help='运行模式，zoo / select')
    parser.add_argument('--context_len', type=int, default=512, help='模型预测所需的输入长度context length')
    parser.add_argument('--fix_context_len', action='store_true', help='设置允许模型使用的context_len最大值，用于公平比较，否则使用模型原始的')
    parser.add_argument("--save_pred", default=True, action="store_true", help="是否保存TSFM的预测结果")
    parser.add_argument("--skip_saved", action="store_true", help="是否跳过已保存结果的数据集")
    parser.add_argument("--debug_mode", action="store_true", help="是否采用 try-except 运行模型，调试时使用")
    parser.add_argument('--zoo_total_num', type=int, default=4, help='model zoo中包含的模型总数')

    # model zoo selector
    parser.add_argument('--select_mode', type=str, default='Recent', help='选择模型的方式')
    parser.add_argument('--random_times', type=int, default=10, help='随机集成次数')
    parser.add_argument('--ensemble_size', type=int, default=0, help='集成的TopK模型数量')

    # model zoo 增量版
    parser.add_argument('--real_world_mode', action='store_true', default=False, help='是否使用增量模型库运行,Fasle时使用单一select_date')
    parser.add_argument('--select_date', type=str, help='选择截止使用的模型发布日期，模拟真实的模型发布状态，年月日格式', default='2025-08-01')
    parser.add_argument('--current_zoo_num', type=int, default=0, help='当前模型总数量')
    parser.add_argument('--real_order_metric', type=str, default='sMAPE', help='用于计算真实order的评估指标，options: [sMAPE, MASE]')

    parser.add_argument(
        "--models", type=str, default="all_zoo",
        help=(
            "选择要运行的模型，逗号分隔的模型名列表 "
            "(如 moirai,chronos)；"
            "all_zoo=遍历所有在 Model_sizes 中启用的模型"
        ),
    )
    parser.add_argument(
        "--size_mode", type=str, default="all_size",
        help=(
            "选择 size 模式："
            "all_size=遍历 Model_sizes 中该模型的所有 size；"
            "first_size（默认）=只遍历该模型第一个 size"
        ),
    )

    args = parser.parse_args()
    set_seed(args.seed)

    args.all_datasets = sorted(set(Short_Fast_datasets.split() + Med_long_Fast_datasets.split()))
    args.med_long_datasets = Med_long_Fast_datasets

    if args.run_mode == "zoo":
        if args.models == "all_zoo":
            families = list(Model_zoo_details.keys())
        else:
            requested = [m.strip() for m in args.models.split(",")]
            families = [m for m in requested if m in Model_zoo_details]
            missing = set(requested) - set(families)
            if missing:
                print(f"\n ⚠️ 下列模型未启用或不存在，将被忽略：{missing} \n ")

        print('运行模型族:', families)
        for family in families:
            sizes_dict = Model_zoo_details[family]

            if not sizes_dict:
                print(f"\n ⚠️ 模型族 '{family}' 在 Model_zoo_details 中未定义任何版本，跳过 \n")
                continue
            if args.size_mode == "all_size":
                sizes = list(sizes_dict.keys())
            elif args.size_mode == "first_size":
                sizes = [next(iter(sizes_dict.keys()))]
            else:
                all_sizes = [s.strip() for s in args.size_mode.split(",")]
                sizes = [s for s in all_sizes if s in sizes_dict]
                if len(all_sizes) - len(sizes) > 0:
                    raise ValueError(f"⚠️ size 模式 {args.size_mode} 中的 size 在 {family} 中不存在")
            print(f"模型族 {family} 将运行的 size 列表: {sizes}")

            if not sizes:
                sizes = [None]

            for size in sizes:
                variant_cfg = sizes_dict[size]

                # 动态 import 对应模型类
                ModelModule = importlib.import_module(variant_cfg["model_module"])
                ModelClass = getattr(ModelModule, variant_cfg["model_class"])

                model = ModelClass(
                    args,
                    module_name=variant_cfg["module_name"],
                    model_name=f"{family}_{size}",
                    model_local_path=variant_cfg["model_local_path"],
                )

                print(f"\n=== 开始运行 {family} [{size}] ===")

                model.run()

    elif args.run_mode == "select":
        args.zoo_total_num = sum(len(sizes) for sizes in Model_zoo_details.values())

        if args.real_world_mode:
            # Real_world增量模型库模式
            all_models = [
                details
                for family in Model_zoo_details.values()
                for details in family.values()
            ]
            # 按模型的发布日期排列
            sorted_models = sorted(all_models, key=lambda x: x["release_date"])
            all_zoo_release_list = [model["release_date"] for model in sorted_models]

            assert args.ensemble_size + 1 <= len(all_zoo_release_list), "ensemble_size must < current_zoo_num)"
            # for current_zoo_num in range(args.ensemble_size+1,len(all_zoo_release_list)+1):
            for current_zoo_num in range(len(all_zoo_release_list), len(all_zoo_release_list) + 1):
                current_zoo_release_list = all_zoo_release_list[args.ensemble_size:current_zoo_num]
                args.select_date = current_zoo_release_list[-1]
                print(f"\n🚀 🚀 🚀 Real_world增量模型库模式，{args.select_date}之前模型数量: "
                      f"{current_zoo_num} / {len(all_zoo_release_list)}, ensemble_size={args.ensemble_size}")
                run_select(args)
        else:
            # 指定日期的固定模型库模式，如args.select_date = '2025-08-01'
            run_select(args)

    else:
        raise ValueError('⚠️ 未知运行模式，仅支持 zoo / select')


if __name__ == "__main__":
    main()
