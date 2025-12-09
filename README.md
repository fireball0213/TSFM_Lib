# TSFM_Lib
第一次作业我们使用了传统的深度学习代码框架，在完全固定的框架下体验了时序分析中的基本技巧和基本设定。在第二次作业中，我们将难度和复杂性全面升级，改用了最新的 TSFM（Time Series Foundation Models）模型进行 zero-shot 预测，采用 GIFT-Eval 框架，完成半开放式的复现任务。

TSFM 的核心特点是：在大规模多源时序数据上进行预训练，具备较强的 **zero-shot 迁移能力**，能够在“未见过的下游数据集”上直接给出具有竞争力的预测结果。近两年，TSFM 逐渐成为时序预测领域的主流方向，各大公司和高校都在持续推出新的 TSFM 模型。与此同时，GIFT-Eval 提供了一个公认的、统一的 **时序基础模型排行榜**，从多个领域、多种频率、多步预测角度系统评估 TSFM 的 zero-shot 能力。

本次作业的目标是：在一个统一的框架和环境下，构建一个由 TSFM 组成的 **model zoo**，让同学们：

- 熟悉最新 TSFM 的 zero-shot 预测能力；
- 了解其中关键参数（如 `context_len`、`prediction_length`、批大小等）对模型性能与稳定性的影响；
- 学会在一个统一代码框架中整合与复现前沿模型；
- 熟悉 GIFT-Eval 这一最新的时序基础模型 benchmark。



## 作业概览

**任务一：在统一框架下运行多种 TSFM，检查他们的结果（10 pts）**

1. 在本地配置好环境，完成 **离线** 数据与模型准备；
2. 运行现有的示例 TSFM（Chronos / Moirai / TimesFM / Sundial），验证框架跑通。

**任务二：复现新的 TSFM 方法到框架代码中（50 pts）**

1. 在 GIFT-Eval 排行榜中随机选择 **2 个排名靠前的 TSFM**，将其整合到本框架中；
2. 检查新 TSFM 的结果和性能，确保指标范围合理，复现成功；
3. 统一运行 model zoo 中所有模型，比较它们在不同 `context_len` 下的性能变化。

**任务三：复现新的 selector 方法（40 pts，README 待补充）**

实现一个新的 selector 方法，对不同的下游预测任务推荐不同的模型进行预测，并与基线 selector 进行性能对比。该部分的具体要求和接口说明将在后续补充。

> ⚠️ 本作业 **推荐采用本地离线方式**：  
> 所有数据和模型文件从 HuggingFace 或其他来源预先下载到本地，运行时 **不再从网络在线拉取**，框架直接从 `Dataset_Path/` 和 `Model_Path/` 目录加载。

---



## 任务一： 代码部署与测试(10 pts)

> 任务一中，主要进行环境准备和部署测试，你并不需要修改任何代码。

### 1. 环境准备与项目克隆

本项目支持 **Linux** 与 **Windows**，建议在Linux环境下使用GPU运行，使用 Anaconda / Miniconda 管理 Python 环境，注意将路径替换为你自己的实际路径。

```
# 1）克隆仓库
git clone https://github.com/fireball0213/TSFM_Lib.git
cd TSFM-Lib

# 创建并激活环境
conda create --name myenv python=3.10
conda activate myenv

# 安装核心依赖
pip install -r requirements_core.txt

# 手动安装可能存在依赖冲突的包版本，可忽略冲突警告
pip install gluonts==0.15.1
pip install transformers==4.40.1
```



### 2. 一键配置项目路径（.env）

本项目通过根目录下的 `.env` 文件来获取项目根路径。
 **数据和模型都放在项目根路径下的子目录（`Dataset_Path/` 与 `Model_Path/`）**，因此只需要设置项目绝对路径：

##### 2.1 Linux 

```
cd /path/to/TSFM-Lib   # 请替换为你的真实路径

# 生成或覆盖 .env 文件，仅包含项目根路径
echo "Project_Path=$(pwd)" > .env
```

生成的 `.env` 类似：

```
Project_Path=/home/yourname/projects/TSFM-Lib
```

##### 2.2 Windows （CMD）

```
cd C:\path\to\TSFM-Lib   :: 请替换为你的真实路径

:: 生成或覆盖 .env 文件，仅包含项目根路径
echo Project_Path=%cd%> .env
```



### 3. 准备数据与预训练模型（离线方式）

本作业统一采用 **本地离线** 的方式加载数据与模型，运行时 **不会** 再从 HuggingFace 等网站自动下载，请提前从https://box.nju.edu.cn/d/275f6b45548c4763a72f/ 中下载下述文件：

- `Dataset.zip`：包含 GIFT-Eval 所需的所有数据集（ett1、ett2、m4、solar 等），下载后将其内容解压到`TSFM-Lib/Dataset_Path`中
- `Model.zip`：包含预先下载好的 TSFM 预训练模型（Chronos、Moirai、TimesFM、Sundial 等），下载后将其内容解压到`TSFM-Lib/Model_Path`中

成功后，应当看到如下目录：

```
TSFM-Lib/
  ├── Dataset_Path/
  │   ├── dataset_config.py # 数据集信息
  │   ├── ett1/
  │   ├── ett2/
  │   ├── m4_hourly/
  │   ├── m4_weekly/
  │   ├── solar/
  │   └── ...  # 其他数据集
  └── Model_Path/
      ├── model_zoo_config.py # 模型库信息
      ├── chronos-models/chronos-bolt-tiny/
      ├── moirai-models/moirai-1.0-R-small/
      ├── sundial-models/Sundial-base_128m/
      └── timesfm-models/timesfm-2.5-200m-pytorch/
```

### 4. 运行现有 TSFM 模型，验证环境

##### 4.1 Linux

```
cd /path/to/TSFM-Lib
conda activate myenv

# 单个模型运行：使用 GPU 0 运行 Chronos bolt_tiny，context_len 采用默认设置
CUDA_VISIBLE_DEVICES=0 python run_model_zoo.py \
  --models "chronos" \
  --size_mode "bolt_tiny" \
  --batch_size 128

# 一键全部运行：使用 GPU 0 运行全部模型的首个 size，context_len 固定为 512
CUDA_VISIBLE_DEVICES=0 python run_model_zoo.py \
  --models "all_zoo" \
  --size_mode "first_size" \
  --batch_size 128 \
  --fix_context_len \
  --context_len 512
```

##### 4.2 Windows（CMD）

```
cd C:\path\to\TSFM-Lib
conda activate myenv

:: 使用 GPU 0 运行 Chronos bolt_tiny，context_len 采用默认设置
set CUDA_VISIBLE_DEVICES=0
python run_model_zoo.py --models "chronos" --size_mode "bolt_tiny" --batch_size 128

:: 使用 GPU 0 运行全部模型的首个 size，context_len 固定为 512
set CUDA_VISIBLE_DEVICES=0
python run_model_zoo.py --models "all_zoo" --size_mode "first_size" --batch_size 128 --fix_context_len --context_len 512

```

运行成功后，各模型的结果会自动写入 `results/` 目录（按模型名、context_len 等分子目录存放）。



## 5. 使用 `check_TSFM.py` 检查运行结果

本次作业选取了50个运行速度较快的数据集configurations，数据集的具体细节详见 GIFT-Eval 论文（https://arxiv.org/pdf/2410.10393 ）中的附录 E。

现有模型运行完成后，可以使用 `check_TSFM.py` 对结果进行汇总和检查。

```
python check_TSFM.py
```

脚本会：

- 读取 `results/` 目录下的结果文件（如 `all_results.csv` 等）；
- 检查缺失或异常文件；
- 汇总各模型在不同数据集上的性能指标，打印对比表格。



## 任务二：复现两个新的TSFM (50 pts)

> 任务二中，你需要修改 `Model_Path/model_zoo_config.py` / `model_zoo/New_TSFM_1_model.py` / `model_zoo/New_TSFM_2_model.py` 这三个文件

本任务是本次作业的核心部分，要求同学 **从 GIFT-Eval 排行榜中选择两个新的 TSFM 模型，并完整复现到当前框架中**，包括：

- 找到官方代码和 HuggingFace 模型仓库；
- 按官方示例跑通模型（先在原生环境中成功运行）；
- 将运行环境、关键参数、数据处理逻辑与本框架对齐；
- 以 **离线、本地加载** 的方式加入 `model_zoo`，并在统一脚本 `run_model_zoo.py` 中参与对比。

下面给出推荐步骤与注意事项。

### 6. 选择模型并收集信息

1. 打开 GIFT-Eval 排行榜，在 overall 榜单下全选所有模型：

   > https://huggingface.co/spaces/Salesforce/GIFT-Eval

2. 从排行榜的 **前 20 名** 中挑选**尚未出现**在当前 model zoo 中的**两个 TSFM**（相同模型的**不同size视作同一个**TSFM）
    已经包含的示例模型包括：

   - `Chronos_model.py`
   - `Moirai_model.py`
   - `TimesFM_model.py`
   - `Sundial_model.py`

3. 对每个新模型，至少记录以下信息，后续在报告中使用：

   - HuggingFace 模型仓库链接；
   - 官方 GitHub 代码链接；
   - 论文标题、会议/期刊；
   - 是否支持多变量预测（multivariate）；
   - 输入/输出格式定义（`context_len`、`prediction_length`、特征维度等）；
   - 是否提供不同 size（如 small / base / large）

4. 对每个新模型，建议选择 **发布时间最新、参数量最小的版本**，选择一个版本即可

------

### 7. 在原始仓库中跑通官方示例代码

在将模型移植到本框架之前，**务必先在官方代码环境中跑通一次示例**，理解基本调用方式。建议独立创建一个临时目录或 conda 环境，用于测试原始代码，避免破坏本作业环境。

1. **克隆官方 GitHub 仓库**，或在本地新建一个简单脚本，按照官方 README 中的示例调用 HuggingFace 模型；
2. 在官方示例中重点观察以下参数的 **传递方式与含义**：
   - `context_len`（历史窗口长度 / 输入长度）；
   - `prediction_length`（预测步数，通常与 GIFT-Eval 配置相关）；
   - `batch_size`（一次前向的序列数）；
   - 数据的时间频率（`freq`，例如 "H"、"D"、"W"），是否需要显式指定；
   - 模型对缺失值 / `NaN` 的处理机制（是否需要在输入前手动填补缺失）。
3. **注意部分模型可能默认“在线下载”权重**，代码中可能出现：
   - `from_pretrained("xxx/yyy")`
   - `force_download=True`
   - `revision="main"` 等参数
   - 在本作业中，我们强调 **本地离线** 调用，即模型文件预先下载到 `Model_Path/` 中再加载，而不是运行时联网下载(速度较慢且不稳定)，如果官方示例代码默认在线调用，且未显示保留支持本地调用的接口，则可能需要在官方源代码中寻找上述参数进行手动修改
   - 本框架中，会在 `Model_Path/model_zoo_config.py` 中为每个模型记录 `model_local_path`，统一管理，自动加载

------

### 8. 将模型整合进框架代码：关键步骤与注意事项

整合过程大致分为三步：

1. 在 `Model_Path/` 中放置模型文件；
2. 在 `Model_Path/model_zoo_config.py` 中注册模型；
3. 在 `model_zoo/New_TSFM_1_model.py` / `model_zoo/New_TSFM_2_model.py` 中实现封装类。

#### 步骤 1：整理模型文件到 `Model_Path/`

假设你选中的第一个模型名称为 `new-tsfm-base`，建议在 `Model_Path/` 下创建结构类似：

```
Model_Path/
  ├── NewTSFM1-models/
  │   └── new-tsfm-base/          # 从 HuggingFace 下载解压后的模型文件和配置文件
  └── NewTSFM2-models/
      └── ...
```

> 命名规则和层次结构可以参考已有模型，如 `chronos-models/chronos-bolt-tiny/`。确保在后续配置中使用一致的路径。

------

#### 步骤 2：在 `Model_zoo_details` 中注册新模型

打开 `Model_Path/model_zoo_config.py`，按照文件中已有的示例，找到并修改：

- `Model_zoo_details`：模型详细信息总表；
- `MULTIVAR_TSFM_PREFIXES`：支持“原生多变量预测”的模型前缀列表。

为你的两个模型分别在`Model_zoo_details`中新增一条配置，详细指南参见 `Model_Path/model_zoo_config.py`

> ✅ 如果模型 **原生支持多变量预测**，即可以一次性接收多通道输入并联合建模（典型如 `Moirai`、`VisionTS`），请将其前缀加入 `dataset_config.py` 中的 `MULTIVAR_TSFM_PREFIXES`。
>  若模型只能逐通道预测，则不要加入该列表。

------

#### 步骤 3：在 `New_TSFM_1_model.py` / `New_TSFM_2_model.py` 中实现封装

打开：

- `model_zoo/New_TSFM_1_model.py`
- `model_zoo/New_TSFM_2_model.py`

代码中已经预留好类定义与 TODO 注释，在实现过程中，重点注意以下几点：

1. **`context_len` 的处理**
   - 不同模型对最大 `context_len` 可能有软 / 硬限制，为保证模型间的公平比较，建议开启 `fix_context_len` 模式，手动维护模型能够接受命令行传入的 `--context_len` 覆盖默认值；
   - 确保模型在调用时，使用的是修改后的 `context_len` 
2. **`freq` 映射**
   - GIFT-Eval 数据的频率等信息存储在 `Dataset_Path/dataset_properties.json`；
   - 部分模型（如 `TimesFM_model.py`）要求显式传入时间频率（如 `"H"`、`"D"`），而且不同项目对频率的字符串写法可能略有区别；
   - 请参考已有模型的写法，确保读取的 freq 能正确传入新模型。
3. **`prediction_length`**
   - GIFT-Eval 每个配置指定了预测长度 `term`，在本框架中由 `data.py` 和 `Dataset` 类统一管理；
   - 在封装新模型时，要确保模型的 `prediction_length` 与该配置严格一致；
   - 若模型只能输出固定长度（比如 96），且与 GIFT-Eval 的 `prediction_length` 不一致，需要在报告中说明，并尽量处理为可比较的形式（比如截断）。
4. **缺失值 (`NaN`) 处理**
   - 有些模型（例如TimesFM）对输入中的 `NaN` 非常敏感，一旦出现就会输出 `NaN`；
   - 本框架中提供了 `utils/missing.py`，可以参考已有模型（如 `TimesFM_model.py`）中如何调用 `fill_missing` 函数；
   - 请确保在送入模型之前，输入张量中不存在未处理的 `NaN`。
5. **统一输出格式**
   - 所有模型预测最终都需要产生统一结构的 `GluonTS Forecast` 对象输出，用来支持统一的评估流程，具体格式和实现方式参考示例文件
   - 需要根据模型是否具备多分位的输出能力，决定是将输出转换为`QuantileForecast` 还是 `SampleForecast`

------

### 9. 结果检查与评估

在完成新模型集成后，请至少进行如下检查：

1. **基础跑通检查**

   - 使用你在任务一中已经跑通的命令，替换 `--models` 参数为新模型名称，运行若干个配置；
   - 确认不会出现明显的报错（例如路径找不到、维度不匹配、`NaN` 传播到输出）。

2. **多种 `context_len` 的性能对比**

   - 设置`--models "all_zoo"`，在 `--fix_context_len` 模式下，尝试几个不同的 `context_len`（默认值为512）；
   - 运行`python check_TSFM.py`，检查新模型在不同长度下的 sMAPE、MASE、Rank 等指标是否有合理变化，总结你发现的规律：
   - 如果结果过于夸张（如 sMAPE=10^5、MASE 为负数或极大值）通常说明维度对齐存在问题，请优先检查对数据集多个channels (variates)、series、windows的对齐方式；

   



## 任务三：复现一个新的模型选择方法 (40 pts)





## Submission

**1. Modified Code**

- 提交完成任务所需的全部修改代码（包括新增或修改的 `.py` 文件）；
- 在项目根目录中提供一个 `README.md` 文件，内容包括：
  - 如何安装环境与依赖；
  - 如何运行代码与脚本复现报告中的主要结果；
  - 新增 TSFM / selector 方法对应的 GitHub、HuggingFace、论文链接。

**2. PDF Report**

- 提交一篇详细的 PDF 报告，结构至少包含：
  - 任务一（运行结果）；
  - 任务二（两个 TSFM 的复现与对比分析）；
  - 任务三（新的 selector 方法设计与实验结果）；
  - 结果分析与讨论。

**3. Submission Format**

- 在完成所有代码与实验后，**删除掉所有数据和模型文件**（`Dataset_Path/` 与 `Model_Path/` 中的大文件不要打包），保留：
  - 完整的代码与脚本；
  - `README.md`；
  - PDF 报告。
- 将上述内容打包为 `.zip` 压缩包提交。

**4. Submission Deadline:** 2025-12-30 23:55
