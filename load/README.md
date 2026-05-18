# 辅助文件和资源

本文件夹包含ENSO贝叶斯分析系统的辅助文件，这些文件不是运行核心代码所必需的。

## 文件夹结构

```
load/
├── image/           # 图表文件 (4个)
├── report/          # 技术报告文档 (5个)
├── tests/           # 测试套件 (3个)
├── tools/           # 数据工具脚本 (2个)
├── docker/          # Docker容器化配置 (3个)
├── config/          # 版本控制配置 (1个)
└── README.md        # 本说明文件
```

---

## image/ - 图表文件

生成的可视化图表：

| 文件名 | 说明 |
|--------|------|
| `fig1_timeseries_font_fixed.png` | NINO3.4时间序列图（含状态标注）|
| `fig2_posteriors_font_fixed.png` | 后验参数分布图（均值、方差等）|
| `fig3_transition_font_fixed.png` | 状态转移矩阵热力图 |
| `fig4_asymmetry_font_fixed.png` | ENSO不对称性分析图 |

**查看方式**：直接打开PNG文件或使用图片查看器

---

## report/ - 技术报告文档

详细的技术报告和分析文档：

### 核心报告（推荐阅读）

| 文档 | 说明 | 重要性 |
|------|------|--------|
| **[REFACTORING_REPORT.md](report/REFACTORING_REPORT.md)** | **v4.0专业级重构完整报告** | 必读 |
| [CODE_IMPROVEMENTS.md](report/CODE_IMPROVEMENTS.md) | 代码改进详情和问题修复记录 | 推荐 |
| [ACTION_PLAN_REPORT.md](report/ACTION_PLAN_REPORT.md) | v2.1关键行动计划实施报告 | 参考 |

### 性能和优化

| 文档 | 说明 |
|------|------|
| [V3_PERFORMANCE_OPTIMIZATION_REPORT.md](report/V3_PERFORMANCE_OPTIMIZATION_REPORT.md) | v3.0性能优化报告（10-50倍提升）|

### 项目交付

| 文档 | 说明 |
|------|------|
| [README_DELIVERY.md](report/README_DELIVERY.md) | 项目交付文档和使用指南 |

---

## tests/ - 测试套件

单元测试和集成测试文件：

| 测试文件 | 覆盖范围 | 用例数 |
|----------|---------|--------|
| `test_refactored_modules.py` | **v4.0重构模块测试**（推荐）| 90+ |
| `test_v3_optimized.py` | v3.0优化版本功能测试 | 60+ |
| `test_core.py` | 核心算法基础测试 | 40+ |

**运行测试**：
```bash
cd tests/

# 运行重构后的模块测试（推荐）
python test_refactored_modules.py

# 或使用pytest
pytest test_refactored_modules.py -v --cov=../../main/core/modules
```

**测试覆盖内容**：
- 类型系统正确性
- 接口完整性验证
- 数据加载器功能
- MCMC采样引擎
- HMM模型核心逻辑
- 错误处理机制
- 向后兼容性检查

---

## tools/ - 数据工具

辅助工具和数据生成脚本：

| 工具 | 功能 | 使用场景 |
|------|------|----------|
| `download_data.py` | 从NOAA等数据源下载NINO3.4数据 | 首次使用/数据更新 |
| `generate_sample_data.py` | 生成模拟的ENSO时间序列数据 | 测试/演示 |

**使用示例**：
```bash
cd tools/

# 下载数据（需要网络连接）
python download_data.py --output ../data/nino34.csv

# 生成模拟数据（用于测试）
python generate_sample_data.py --n_months 120 --output test_data.csv --noise_level 0.5
```

---

## docker/ - 容器化部署

Docker容器化相关配置文件：

| 文件 | 说明 |
|------|------|
| `Dockerfile` | Docker镜像构建文件（基于Python 3.11）|
| `docker-compose.yml` | 多服务编排配置（支持数据卷挂载）|
| `.dockerignore` | Docker构建优化规则（排除不必要文件）|

**快速部署**：
```bash
cd docker/

# 构建镜像
docker build -t enso-bayesian-hmm .

# 运行容器
docker-compose up -d

# 查看日志
docker-compose logs -f
```

**优势**：
- 环境一致性保证
- 一键部署到任何平台
- 隔离的运行环境
- 易于版本管理

---

## config/ - 配置文件

版本控制和项目配置：

| 文件 | 说明 |
|------|------|
| `.gitignore` | Git忽略规则（排除venv、__pycache__等）|

**使用方法**：
```bash
# 复制到项目根目录（如果需要Git版本控制）
cp config/.gitignore ../../.gitignore
```

---

## 完整文件清单

### 按类型统计

| 类别 | 文件数 | 总大小（约）|
|------|--------|------------|
| 图表 (image/) | 4 PNG | ~1.9 MB |
| 报告 (report/) | 5 MD | ~68 KB |
| 测试 (tests/) | 3 PY | ~54 KB |
| 工具 (tools/) | 2 PY | ~7 KB |
| Docker (docker/) | 3 files | ~4 KB |
| 配置 (config/) | 1 file | <1 KB |
| **总计** | **18 files** | **~2 MB** |

---

## 与 main/ 文件夹的关系

```
test1_1/
├── main/            # 核心代码包（运行必需）
│   ├── core/        # 算法实现
│   ├── cli.py       # 命令行入口
│   └── config.json  # 运行配置
│
└── load/            # 辅助资源（可选）
    ├── image/       # 生成的图表
    ├── report/      # 技术文档
    ├── tests/       # 测试套件
    └ ...            # 其他辅助材料
```

**关系说明**：
- `main/` 可以独立运行，不依赖 `load/`
- `load/` 提供额外的工具、文档和可视化结果
- 开发者建议同时保留两个文件夹
- 分发时可根据需求选择包含哪些部分

---

## 使用建议

### 对于最终用户
只需保留 `main/` 文件夹即可运行完整的分析流程。

### 对于开发者/研究人员
建议同时保留 `main/` 和 `load/` 文件夹：
- 使用 `tests/` 验证代码修改的正确性
- 参考 `report/` 了解实现细节和设计决策
- 利用 `tools/` 获取或生成测试数据
- 使用 `docker/` 进行标准化部署

### 对于论文投稿
建议包含：
- `main/` （必需）：可复现代码
- `load/image/` （推荐）：关键图表
- `load/report/REFACTORING_REPORT.md` （可选）：方法说明补充

---

## 常见问题

### Q: 这些辅助文件是必需的吗？
A: 不是。`main/` 文件夹是完全自包含的，可以独立运行。

### Q: 如何只获取核心代码？
A: 只需复制或下载 `main/` 文件夹即可。

### Q: 测试失败怎么办？
A: 请检查是否安装了所有依赖（见 `main/requirements.txt`），然后查看具体的错误信息。

### Q: 如何生成新的图表？
A: 运行 `main/cli.py` 后会自动在当前目录生成新的图表文件。

### Q: Docker镜像多大？
A: 基础镜像约800MB，包含所有依赖项。

---

**最后更新**: 2025-01-19  
**维护者**: 项目开发团队  
**状态**: 可用于生产环境
