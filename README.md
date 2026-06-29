# Research Data Analysis — Python Scripts

科研数据分析脚本集合，覆盖氨基酸传感、拉曼光谱、异常检测等方向。

## 📁 项目结构

```
.
├── amino_acid/                          # 氨基酸传感 DNA-SWCNT 研究（核心模块）
│   ├── data_processing/                 # 原始数据预处理（NS Super 数据流程）
│   │   ├── hcl_baseline/                # 盐酸基线处理子流程
│   │   └── ...
│   ├── preprocessing/                   # 机器学习前期特征工程
│   │   ├── KMeans.py                    # K-Means 聚类探索
│   │   ├── PCA_projection_overlay.py    # PCA 投影叠加可视化
│   │   ├── 2d_visualization.py          # 二维降维可视化
│   │   ├── dimensionality_reduction.py  # 整体降维分析
│   │   ├── feature_selection.py         # 特征筛选
│   │   ├── separability_evaluation.py   # 可分性评估
│   │   └── classification_modeling.py   # 初步分类建模
│   ├── classification/                  # 分类模型训练与验证
│   │   ├── hierarchical/                # 分层分类器（DNA-SWCNT）
│   │   ├── spectral_analysis_tool.py    # 光谱数据分析工具（GUI）
│   │   ├── spectral_prediction_tool.py  # 光谱数据预测工具（GUI）
│   │   ├── regression_quantification.py # 回归定量
│   │   ├── incremental_learning.py      # 增量学习工具
│   │   ├── data_remerge_learning.py     # 数据重新合并学习
│   │   ├── independent_cv.py            # 独立交叉验证
│   │   ├── permutation_importance.py    # 置换重要性 + SHAP
│   │   └── bootstrap_cv.py              # 重复随机子抽样验证
│   ├── visualization/                   # 绘图脚本
│   │   ├── ROC.py                       # ROC 曲线
│   │   ├── raw_spectra.py               # 原始光谱图
│   │   ├── response_concentration.py    # 响应-浓度曲线
│   │   ├── average_heatmap.py           # 平均热图
│   │   ├── mechanism_map.py             # 机理可视化
│   │   ├── query_all_amino_acids.py     # 查全氨基酸
│   │   ├── heatmap.py                   # 热图
│   │   ├── feature_value_plot.py        # 特征值作图
│   │   ├── color_palette.py             # 颜色配置
│   │   └── conc_response_format.py      # 浓度响应格式转化
│   ├── hot_map/                         # 双层面板指纹图热图
│   └── amino_acid_discrimination/       # 氨基酸鉴别集成脚本（LDA + KMeans）
│       └── lda_kmeans_model/            # 模型文件与辅助工具
│
├── raman/                               # 拉曼光谱分析
│   ├── peak_532.py                      # 532nm 峰值处理
│   ├── area_532.py                      # 532nm 面积计算
│   ├── raman_785.py                     # 785nm 处理
│   ├── PCA.py                           # PCA 分析
│   └── swcnt.py                         # 纯碳管分析
│
├── ad/                                  # 异常检测 / 原始数据汇总
│   └── raw_data_summary.py
│
├── reagent_manager/                     # 试剂库管理（Streamlit 应用）
│   ├── app.py                           # 主程序（streamlit run app.py）
│   └── data/
│       └── 试剂总库_不要删除.xlsx       # 主数据库（勿删）
│
└── titanic/                             # Titanic 竞赛练习（ML 入门）
    ├── titanic_survival_prediction.py
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
cd reagent_manager
streamlit run app.py
```

### 运行氨基酸分析脚本

各脚本均可独立运行，启动后通过文件对话框选择数据文件：

```bash
python amino_acid/classification/spectral_analysis_tool.py
```

## 📌 各模块说明

| 模块 | 功能 | 关键技术 |
|------|------|----------|
| `amino_acid/data_processing` | NS Super 仪器原始数据清洗与格式转换 | pandas, CSV 转换 |
| `amino_acid/preprocessing` | 降维、聚类、可分性探索 | PCA, KMeans, t-SNE |
| `amino_acid/classification` | 多分类器训练、验证、SHAP 解释 | RF, SVM, LDA, SHAP |
| `amino_acid/visualization` | 科研绘图 | matplotlib, seaborn |
| `amino_acid/amino_acid_discrimination` | LDA + KMeans 集成鉴别流程 | sklearn |
| `raman` | 拉曼峰值/面积提取与 PCA 分析 | scipy, sklearn |
| `ad` | 原始数据汇总整合 | pandas |
| `reagent_manager` | 实验室试剂库 CRUD 管理 | Streamlit |
| `titanic` | ML 练习（Stacking 集成学习） | sklearn |

## 📝 文件命名说明

- 版本迭代文件（`V18`、`V19`）保留在 `classification/hierarchical/` 下，以最新版为主
- `bug.py`（已清除，原为 tkinter 测试残留）
- `02 NS Super Data NlR..py`（已清除，空文件）

## 🗂 数据文件

- 原始数据文件（`.xlsx`, `.csv`）**不纳入版本控制**，请参考 `.gitignore`
- 唯一例外：`reagent_manager/data/试剂总库_不要删除.xlsx` 为应用主数据库，建议手动备份

---

*研究方向：sc-SWCNT 功能化 · 氨基酸传感 · 拉曼光谱 · 机器学习辅助材料研究*
