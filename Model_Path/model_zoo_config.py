
'''
Model_zoo_details 填写说明：
-----------------------
这个字典用于统一配置“模型族 + 具体规模(size)”的所有信息，便于框架代码自动加载和遍历。
配置完成后，无需修改其他文件，就可以通过命令行参数 --models New_TSFM_1 来运行你的模型。
-----------------------

层级结构：
  - 最外层 key：模型族 family 名（如 "sundial" / "chronos" / "moirai" / "timesfm" / "New_TSFM_1"）
  - 第二层 key：该模型族下的具体 size / variant 名（如 "base" / "bolt_tiny" / "small" / "2.5"）
  - 第二层 value：一个描述该 size 的配置字典，字段含义如下：

    * "name"：
        - 含义：该模型在官方仓库或论文中的具体版本名称。
        - 从何处获得：通常来自官方模型发布页 / HuggingFace 模型名
          例如：
            - Chronos 使用 "bolt-tiny"
            - Moirai 使用 "1.0-R-small"

    * "abbreviation"：
        - 从何处获得：由你自己设计，要求简短且不冲突；
          例如：
            - "Sun.B" 表示 Sundial Base
            - "Chr.bT" 表示 Chronos bolt-tiny

    * "model_module"：
        - 含义：对应模型封装类所在的 Python 模块路径（用于 importlib.import_module）。
        - 从何处获得：由你在 model_zoo 目录中创建的模型封装文件决定，例如：
            - "model_zoo.Sundial_model" 对应文件 model_zoo/Sundial_model.py
            - "model_zoo.Chronos_model" 对应文件 model_zoo/Chronos_model.py

    * "model_class"：
        - 含义：封装该模型的 Python 类名（需要继承 BaseModel）。
        - 从何处获得：你在对应的 model_module 文件中定义的类名，例如：
            - Sundial_model.py 中的 SundialModel
            - Chronos_model.py 中的 ChronosModel

    * "module_name"：
        - 含义：模型的在线地址 。
        - 从何处获得：
            - 通常是官方模型在 HuggingFace / GitHub 中的名称；
            - 示例：
                - "Salesforce/Sundial-base_128m"
                - "Salesforce/chronos-bolt-tiny"

    * "model_local_path"：
        - 含义：本地离线模型权重所在的路径（相对于项目根目录）。
        - 从何处获得：
            - 由你手动将HuggingFace官方模型下载到本地，然后根据实际目录填写；
            - 示例：
                - "Model_Path/sundial-models/Sundial-base_128m"
                - "Model_Path/chronos-models/chronos-bolt-tiny"

    * "release_date"：
        - 含义：模型的发布日期。
        - 从何处获得：
            - HuggingFace官方模型的 Files and versions 页面

'''


Model_zoo_details = {
    "sundial": {
        "base": {
            "name": "base",
            "abbreviation": "Sun.B",
            "model_module": "model_zoo.Sundial_model",
            "model_class": "SundialModel",
            "module_name": "thuml/Sundial-base_128m",
            "model_local_path": "Model_Path/sundial-models/Sundial-base_128m",
            "release_date": "2025-05-14",
        },
    },
    "chronos": {
        "bolt_tiny": {
            "name": "bolt-tiny",
            "abbreviation": "Chr.bT",
            "model_module": "model_zoo.Chronos_model",
            "model_class": "ChronosModel",
            "module_name": "amazon/chronos-bolt-tiny",
            "model_local_path": "Model_Path/chronos-models/chronos-bolt-tiny",
            "release_date": "2024-11-10",
        },
    },
    "moirai": {
        "small": {
            "name": "1.0-R-small",
            "abbreviation": "Moi.S",
            "model_module": "model_zoo.Moirai_model",
            "model_class": "MoiraiModel",
            "module_name": "Salesforce/moirai-1.0-R-small",
            "model_local_path": "Model_Path/moirai-models/moirai-1.0-R-small",
            "release_date": "2024-03-19",
        },
    },

    "timesfm": {
        "2.5": {
            "name": "2.5-200m-pytorch",
            "abbreviation": "TFM.25",
            "model_module": "model_zoo.TimesFM_model",
            "model_class": "TimesFMModel",
            "module_name": "google/timesfm-2.5-200m-pytorch",
            "model_local_path": "Model_Path/timesfm-models/timesfm-2.5-200m-pytorch",
            "release_date": "2025-10-1",
        },
    },

    "New_TSFM_1": {
        # TODO：对应 New_TSFM_1_model.py
        # 示例：
        # "small": {
        #     "name": "v1-small",
        #     "abbreviation": "N1.S",
        #     "model_module": "model_zoo.New_TSFM_1_model",
        #     "model_class": "NewModel",
        #     "module_name": "YourOrg/New_TSFM_1-v1-small",
        #     "model_local_path": "Model_Path/new-tsfm-1-models/New_TSFM_1-v1-small",
        #     "release_date": "2025-12-1",
        # },
    },

    "New_TSFM_2": {
        # TODO：对应 New_TSFM_2_model.py
        # 示例：
        # "base": {
        #     "name": "v1-base",
        #     "abbreviation": "N2.B",
        #     "model_module": "model_zoo.New_TSFM_2_model",
        #     "model_class": "NewModel",
        #     "module_name": "YourOrg/New_TSFM_2-v1-base",
        #     "model_local_path": "Model_Path/new-tsfm-2-models/New_TSFM_2-v1-base",
        #     "release_date": "2025-12-2",
        # },
    },
}



Model_abbrev_map = {
    f"{family}_{variant}": info["abbreviation"]
    for family, variants in Model_zoo_details.items()
    for variant, info in variants.items()
    if "abbreviation" in info
}

All_model_names = [
    f"{family}_{variant}"
    for family, variants in Model_zoo_details.items()
    for variant, info in variants.items()
    if "abbreviation" in info
]



MULTIVAR_TSFM_PREFIXES = [
    "moirai",
    # TODO：如果后续有新的多变量 TSFM，请在此追加模型族的 family 名（对应Model_zoo_details的最外层 key）
]
