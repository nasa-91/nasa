ENSO贝叶斯不对称性分析系统

<p align="center">
  <strong>基于贝叶斯隐马尔可夫模型的厄尔尼诺-南方涛动不对称性分析工具</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License" />
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg" alt="Status" />
</p>

-----------------------------------------------------------------------

项目简介

这是一个基于Python的贝叶斯隐马尔可夫模型分析工具，用于分析ENSO（厄尔尼诺-南方涛动）不对称性分析系统，采用贝叶斯隐马尔可夫模型进行状态识别和统计推断。

核心特性

完整贝叶斯框架   Gibbs采样MCMC + 后验推断
多维不对称性分析   转移概率、持续时间、振幅、演变速度
模型选择准则   BIC/AIC/AICc自动计算
传统方法对比  ONI阈值法基线
专业可视化   7子图综合报告
鲁棒性设计   Student-t分布处理厚尾数据
多格式支持   CSV、NOAA ASCII、NetCDF
敏感性分析   Bootstrap不确定性量化

-----------------------------------------------------------------------

环境要求

Python 3.11 或更高版本
Windows/Linux/macOS

安装步骤

1. 克隆仓库
   ```bash
   git clone https://github.com/nasa-91/nasa.git
   cd nasa
   ```

项目链接: [https://github.com/nasa-91/nasa](https://github.com/nasa-91/nasa)

2. 创建虚拟环境
   ```bash
   python -m venv venv_env
   
   # Windows
   .\venv_env\Scripts\activate
   
   # Linux/macOS
   source venv_env/bin/activate
   ```

3. 安装依赖
   ```bash
   pip install numpy pandas scipy matplotlib netCDF4
   ```

4. 准备数据

   将NINO3.4数据保存为CSV格式（`data_nino.csv`）：

   ```csv
   date,nino34
   1950-01,-0.45
   1950-02,-0.52
   1950-03,-0.38
   ...
   ```

5. 运行分析

   ```bash
   python core/main.py
   ```

---

使用指南

基础用法

```python
from core.main import ENSODataLoader, RobustBayesianHMM

# 1. 加载数据
loader = ENSODataLoader()
data = loader.load_data('data_nino.csv')

# 2. 运行贝叶斯HMM分析
hmm = RobustBayesianHMM(
    n_states=3,           # 状态数：拉尼娜、中性、厄尔尼诺
    emission_dist='student_t',  # 使用Student-t分布
    random_seed=42
)

# 3. 拟合模型
results = hmm.fit(
    data['standardized_nino34'],
    n_iterations=5000,    # MCMC迭代次数
    burn_in=2000,         # 预烧期
    n_chains=4,           # MCMC链数
    verbose=True
)

# 4. 获取结果
print(results['state_sequence'])      # 状态序列
print(results['transition_matrix'])   # 转移概率矩阵
print(results['asymmetry_metrics'])   # 不对称性指标
```

### 高级配置

```python
# 自定义参数
hmm = RobustBayesianHMM(
    n_states=4,                    # 更多状态
    emission_dist='gaussian',      # 高斯分布
    use_time_covariates=True       # 使用时间协变量
)

results = hmm.fit(
    y,
    n_iterations=10000,            # 更多迭代提高精度
    burn_in=3000,
    n_chains=6,                    # 更多链提高稳定性
    thin=10,                       # 更大稀疏间隔
    nu_prior={'mean': 5, 'std': 2} # 自定义先验
)
```

---

##  项目结构

```
enso-bayesian-analysis/
├── core/                      # 核心算法模块
│   └── main.py               # 主程序（数据加载+HMM实现）
├── tools/                     # 数据处理工具
│   ├── download_data.py      # NOAA数据下载器
│   └── generate_sample_data.py # 示例数据生成器
├── tests/                     # 测试脚本
│   ├── basic_test.py         # 基础功能测试
│   ├── full_test.py          # 完整流程测试
│   └── advanced_test.py      # 高级功能测试
├── docs/                      # 文档说明
│   ├── quick_start.md        # 快速开始指南
│   ├── data_prep.md          # 数据准备指南
│   └── project_overview.md   # 项目详细说明
├── outputs/                   # 输出结果（自动生成）
│   ├── analysis_results.png  # 分析结果图
│   └── *.txt                 # 文本输出
├── venv_env/                  # Python虚拟环境（不提交到Git）
├── .gitignore                # Git忽略规则
├── README.md                 # 项目说明（本文件）
└── requirements.txt          # 依赖列表
```

---

技术架构

核心算法

1. **贝叶斯隐马尔可夫模型 (BHMM)**
   - 状态空间：K个隐藏状态（拉尼娜/中性/厄尔尼诺）
   - 观测模型：Student-t分布或高斯分布
   - 推断方法：Gibbs采样MCMC

2. **不对称性度量**
   - 转移概率不对称性：P(El Niño→La Niña) vs P(La Niña→El Niño)
   - 持续时间不对称性：平均持续时间差异
   - 振幅不对称性：极端事件强度对比
   - 演变速度不对称性：建立/衰减速率比较

3. **模型评估**
   - BIC/AIC/AICc模型选择
   - Gelman-Rubin收敛诊断
   - 后验预测检验
   - Bootstrap不确定性量化

### 技术栈

| 类别 | 技术 |
|------|------|
| 语言 | Python 3.11+ |
| 数值计算 | NumPy, SciPy |
| 数据处理 | Pandas |
| 可视化 | Matplotlib, Plotly |
| 统计推断 | MCMC (Gibbs Sampling) |
| 分布假设 | Student-t / Gaussian |

---

##  输出示例

系统会自动生成以下输出：

1. **可视化图表** (`analysis_results.png`)
   - NINO3.4时间序列及状态标注
   - 状态转移路径图
   - 后验参数分布密度图
   - 转移概率矩阵热力图
   - 持续时间分布直方图
   - 不对称性指标对比图
   - 收敛诊断图

2. **文本报告** (`*.txt`)
   - 模型参数后验估计
   - 事件检测列表
   - 统计显著性检验
   - 模型拟合优度指标

---

##  测试运行

```bash
# 运行基础测试
python tests/basic_test.py

# 运行完整测试（包含所有功能）
python tests/full_test.py

# 运行高级测试（Bootstrap敏感性分析）
python tests/advanced_test.py
```

---

##  数据来源

推荐使用以下公开数据集：

- **NOAA CPC NINO3.4指数**: https://www.cpc.ncep.noaa.gov/data/indices/
- **ERSST v5**: https://www.ncei.noaa.gov/data/sea-surface-temp-optimum-interpolation/
- **HadISST**: https://www.metoffice.gov.uk/hadobs/hadisst/

---

##  贡献指南

欢迎提交Issue和Pull Request！

1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

---

##  作者

开发者: 何旭
邮箱: h2078309440@163.com

---

致谢

感谢所有开源贡献者
数据来源于NOAA/NCEI等机构
算法灵感来自气候动力学前沿研究

---

<p align="center">
  <strong>如果这个项目对您有帮助，请给一个Star！</strong>
</p>
