# Research Data Analysis — Python Scripts

科研数据分析脚本集合，覆盖氨基酸传感、拉曼光谱、异常检测等方向。

## 📁 项目结构

```
.
├── amino\_acid/                          # 氨基酸传感 DNA-SWCNT 研究（核心模块）
│   ├── data\_processing/                 # 原始数据预处理（NS Super 数据流程）
│   │   ├── hcl\_baseline/                # 盐酸基线处理子流程
│   │   └── ...
│   ├── preprocessing/                   # 机器学习前期特征工程
│   │   ├── KMeans.py                    # K-Means 聚类探索
│   │   ├── PCA\_projection\_overlay.py    # PCA 投影叠加可视化
│   │   ├── 2d\_visualization.py          # 二维降维可视化
│   │   ├── dimensionality\_reduction.py  # 整体降维分析
│   │   ├── feature\_selection.py         # 特征筛选
│   │   ├── separability\_evaluation.py   # 可分性评估
│   │   └── classification\_modeling.py   # 初步分类建模
│   ├── classification/                  # 分类模型训练与验证
│   │   ├── hierarchical/                # 分层分类器（DNA-SWCNT）
│   │   ├── spectral\_analysis\_tool.py    # 光谱数据分析工具（GUI）
│   │   ├── spectral\_prediction\_tool.py  # 光谱数据预测工具（GUI）
│   │   ├── regression\_quantification.py # 回归定量
│   │   ├── incremental\_learning.py      # 增量学习工具
│   │   ├── data\_remerge\_learning.py     # 数据重新合并学习
│   │   ├── independent\_cv.py            # 独立交叉验证
│   │   ├── permutation\_importance.py    # 置换重要性 + SHAP
│   │   └── bootstrap\_cv.py              # 重复随机子抽样验证
│   ├── visualization/                   # 绘图脚本
│   │   ├── ROC.py                       # ROC 曲线
│   │   ├── raw\_spectra.py               # 原始光谱图
│   │   ├── response\_concentration.py    # 响应-浓度曲线
│   │   ├── average\_heatmap.py           # 平均热图
│   │   ├── mechanism\_map.py             # 机理可视化
│   │   ├── query\_all\_amino\_acids.py     # 查全氨基酸
│   │   ├── heatmap.py                   # 热图
│   │   ├── feature\_value\_plot.py        # 特征值作图
│   │   ├── color\_palette.py             # 颜色配置
│   │   └── conc\_response\_format.py      # 浓度响应格式转化
│   ├── hot\_map/                         # 双层面板指纹图热图
│   └── amino\_acid\_discrimination/       # 氨基酸鉴别集成脚本（LDA + KMeans）
│       └── lda\_kmeans\_model/            # 模型文件与辅助工具
│
├── raman/                               # 拉曼光谱分析
│   ├── peak\_532.py                      # 532nm 峰值处理
│   ├── area\_532.py                      # 532nm 面积计算
│   ├── raman\_785.py                     # 785nm 处理
│   ├── PCA.py                           # PCA 分析
│   └── swcnt.py                         # 纯碳管分析
│
├── ad/                                  # 异常检测 / 原始数据汇总
│   └── raw\_data\_summary.py
│
├── reagent\_manager/                     # 试剂库管理（Streamlit 应用）
│   ├── app.py                           # 主程序（streamlit run app.py）
│   └── data/
│       └── 试剂总库\_不要删除.xlsx       # 主数据库（勿删）
│
└── titanic/                             # Titanic 竞赛练习（ML 入门）
    ├── titanic\_survival\_prediction.py
    ├── train.csv
    └── test.csv
```

## 🚀 快速开始

### 环境依赖

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl streamlit
```

> 部分脚本使用 `tkinter`（Python 自带），无需额外安装。

### 运行试剂管理应用

```bash
cd reagent\_manager
streamlit run app.py
```

### 运行氨基酸分析脚本

各脚本均可独立运行，启动后通过文件对话框选择数据文件：

```bash
python amino\_acid/classification/spectral\_analysis\_tool.py
```

## 📌 各模块说明

|模块|功能|关键技术|
|-|-|-|
|`amino\_acid/data\_processing`|NS Super 仪器原始数据清洗与格式转换|pandas, CSV 转换|
|`amino\_acid/preprocessing`|降维、聚类、可分性探索|PCA, KMeans, t-SNE|
|`amino\_acid/classification`|多分类器训练、验证、SHAP 解释|RF, SVM, LDA, SHAP|
|`amino\_acid/visualization`|科研绘图|matplotlib, seaborn|
|`amino\_acid/amino\_acid\_discrimination`|LDA + KMeans 集成鉴别流程|sklearn|
|`raman`|拉曼峰值/面积提取与 PCA 分析|scipy, sklearn|
|`ad`|原始数据汇总整合|pandas|
|`reagent\_manager`|实验室试剂库 CRUD 管理|Streamlit|
|`titanic`|ML 练习（Stacking 集成学习）|sklearn|

## \---

## 

### \## 📊 成果可视化示意图展示 (Scientific Visualization Demo)



> 本模块展示的图表仅作项目功能与科研绘图质量的演示。

\### 核心成果图谱集成 (Integrated Results Gallery)



下列四幅图展示了项目在数据分析管线中的关键产出：



1\. \*\*PCA 聚类分析 (PCA Cluster Analysis)\*\*：展示原始光谱数据在降维后的空间分布。

2\. \*\*特征重要性排名 (SHAP Feature Importance)\*\*：直观展示不同传感通道对模型决策的贡献度排名。

3\. \*\*多模型分类验证 (ROC Curves Plot)\*\*：提供不同参数、不同分类模型（如分层分类器）的 ROC 曲线对比。

4\. \*\*标准化响应热图 (Normalized Response Heatmap)\*\*：直观呈现传感器阵列对不同样本的标准化响应指纹图谱。



!\[Academic Project Visualization Demo](demo\_plots/academic\_demo.png)



\---

## 📝 文件命名说明

* 版本迭代文件（`V18`、`V19`）保留在 `classification/hierarchical/` 下，以最新版为主
* `bug.py`（已清除，原为 tkinter 测试残留）
* `02 NS Super Data NlR..py`（已清除，空文件）

## 🗂 数据文件

* 原始数据文件（`.xlsx`, `.csv`）**不纳入版本控制**，请参考 `.gitignore`
* 唯一例外：`reagent\_manager/data/试剂总库\_不要删除.xlsx` 为应用主数据库，建议手动备份

\---

*研究方向：sc-SWCNT 功能化 · 氨基酸传感 · 拉曼光谱 · 机器学习辅助材料研究*

