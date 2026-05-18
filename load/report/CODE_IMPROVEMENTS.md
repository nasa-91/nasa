# ENSO贝叶斯不对称性分析系统 - 代码改进报告

**版本**: v2.0  
**日期**: 2026-01-18  
**状态**: 已完成8/10项关键改进

---

## 改进概览

| # | 问题 | 状态 | 优先级 |
|---|------|------|--------|
| 1 | Student-t尺度参数解释修正 | [OK] 完成 | 高 |
| 2 | 标签交换问题解决 | [OK] 完成 | 高 |
| 3 | 不对称性统计检验更正 | [OK] 完成 | 高 |
| 4 | WAIC和LPPD正确实现 | [OK] 完成 | 高 |
| 5 | MCMC效率优化 (Numba) | [PENDING] 待完成 | 中 |
| 6 | 统一先验为逆伽马分布 | [PENDING] 待完成 | 中 |
| 7 | 可移植性改进 | [OK] 完成 | 高 |
| 8 | 增加单元测试 | [OK] 完成 | 高 |
| 9 | 补全/移除use_time_covariates | [OK] 完成 | 中 |
| 10 | 文档更新 | [OK] 完成 | 高 |

---

## 详细修改内容

### 1. Student-t尺度参数修正

**问题**: 
- `_log_emission_pdf`中Student-t分支的`sigma`参数含义不明确
- 输出结果未转换为标准差，导致可解释性问题

**解决方案**:
- 明确文档说明：Student-t分布使用位置-尺度参数化
- `sigma` = 尺度参数(scale)，实际标准差 = `sigma * sqrt(nu/(nu-2))`
- 添加转换方法：
  - `_scale_to_std(sigma, nu)` → 转换为标准差
  - `_std_to_scale(std, nu)` → 从标准差反推
- 在输出时自动转换，确保用户看到的是实际标准差

**修改文件**: 
- [main.py](core/main.py) - `_log_emission_pdf()`, `_scale_to_std()`, `_std_to_scale()`

---

### 2. 标签交换问题解决

**问题**:
- 贝叶斯HMM存在状态标签对称性
- MCMC链可能在不同迭代间交换状态标签
- 导致后验均值等统计量无意义

**解决方案**:
- 实现基于K-means聚类的后验标签匹配算法
- 新增方法：
  - `_resolve_label_switching(all_mu_samples)`: 检测并解决标签交换
  - `_apply_label_consistency(chain_results)`: 应用跨链一致性校正
  - `_compute_optimal_mapping(sample, reference)`: 计算最优排列映射
- 在`fit()`函数末尾自动调用标签校正

**算法流程**:
1. 对每条链的后验均值进行K-means聚类
2. 根据聚类中心排序统一标签顺序
3. 使用匈牙利算法计算最优状态映射
4. 重排所有链的参数样本

**修改文件**:
- [main.py](core/main.py) - `fit()`, `_resolve_label_switching()`, `_apply_label_consistency()`, `_compute_optimal_mapping()`

---

### 3. 不对称性统计检验更正

**问题**:
- 原实现使用t检验，假设数据正态分布
- ENSO事件持续时间等指标通常不满足正态假设
- 缺少峰值振幅分析

**解决方案**:
- **替换检验方法**: t检验 → Mann-Whitney U检验
  - 非参数检验，不假设正态分布
  - 更适合小样本和厚尾分布
  - 对异常值更鲁棒
  
- **新增效应量**: Cliff's Delta
  - 量化差异的实际大小
  - 解释：0.147=小, 0.33=中, 0.474=大
  
- **新增峰值振幅分析**:
  - 提取每个事件期间的极值（最大值/最小值）
  - 单独进行Mann-Whitney U检验
  - 反映极端事件强度的不对称性

**修改的函数**:
- `_analyze_duration_asymmetry()` 
- `_analyze_amplitude_asymmetry()` - 新增峰值振幅子分析
- `_analyze_evolution_speed_asymmetry()`

**输出示例**:
```
持续时间不对称性:
  Mann-Whitney U: 45.0, P值: 0.0321 [显著]
  Cliff's Delta (效应量): 0.284
  95% CI: [0.12, 4.56]

振幅不对称性:
  [新增] 峰值振幅不对称性:
    厄尔尼诺峰值: 2.34°C
    拉尼娜峰值: 1.89°C
    P值: 0.0156 [显著]
```

**修改文件**:
- [main.py](core/main.py) - 三个不对称性分析函数

---

### 4. WAIC和LPPD正确实现

**问题**:
- 原LPPD实现未正确考虑状态混合(marginalization over states)
- WAIC惩罚项计算过于简化
- 缺少逐点对数似然存储

**解决方案**:

**LPPD修正**:
```
原实现: lppd_t = log(Σ_k p(y_t|θ_{k,s}))  # 错误：只对状态求和
修正后: lppd_t = log(1/S * Σ_s [Σ_k π_{k,s} * p(y_t|μ_{k,s},σ_{k,s})])
        # 正确：先对状态混合，再对后验样本平均
```

关键改进：
- 对每个时间点t：
  1. 遍历所有S个后验样本s
  2. 对每个样本，对K个状态加权求和（权重=π_{k,s}）
  3. 使用log-sum-exp技巧数值稳定地计算点lppd
  4. 最后对所有T个时间点求和

**WAIC惩罚项改进**:
- 使用逐点方差法：`p_waic = Σ_t Var_s[log p(y_t|θ_s)]`
- 可选存储完整逐点对数似然矩阵（`_pointwise_log_likelihood`）

**修改文件**:
- [main.py](core/main.py) - `_compute_lppd()`, `_compute_waic_penalty()`

---

### 7. 可移植性改进

**问题**:
- 数据路径硬编码在代码中
- 无法通过命令行运行
- 缺少配置文件支持

**解决方案**:

**新建命令行接口** ([cli.py](cli.py)):
```bash
python cli.py --data nino.csv --states 4 --iterations 8000
python cli.py --config config.json
```

功能特性：
- 完整的参数解析（数据、模型、MCMC、输出）
- JSON配置文件支持
- 自动创建输出目录
- 详细进度显示
- 错误处理和帮助信息

**配置文件示例** ([config.json](config.json)):
```json
{
  "data": {
    "file_path": "data/nino34.csv",
    "format": "auto"
  },
  "model": {
    "n_states": 3,
    "emission_distribution": "student_t"
  },
  "mcmc": {
    "n_iterations": 5000,
    "n_chains": 4
  }
}
```

---

### 8. 单元测试

**新增测试套件** ([tests/test_core.py](tests/test_core.py))：

**测试覆盖范围**:

1. **TestLogEmissionPDF** (6个测试)
   - 高斯PDF形状和数值正确性
   - Student-t PDF形状和厚尾特性
   - 无效分布类型异常处理
   - sigma和nu的自动限制

2. **TestScaleParameterConversion** (4个测试)
   - scale→std转换正确性
   - std→scale转换正确性
   - 往返转换一致性
   - nu≤2时的错误处理

3. **TestForwardBackwardAlgorithm** (2个测试)
   - 前向算法输出形状和对数似然有限性
   - 后向算法输出形状

4. **TestStateSampling** (2个测试)
   - 状态序列长度和范围验证
   - 转移矩阵随机性和行归一化

5. **TestDataLoading** (2个测试)
   - CSV数据加载正确性
   - 标准化后的均值和方差验证

6. **TestLabelSwitchingResolution** (2个测试)
   - 基本标签交换解决功能
   - 两状态的简单排序逻辑

**运行方法**:
```bash
python -m pytest tests/test_core.py -v
```

---

### 9. use_time_covariates清理

**问题**:
- 构造函数接受`use_time_covariates`参数
- 但该功能从未实现
- 可能误导用户认为已可用

**解决方案**:
- 移除构造函数中的`use_time_covariates`参数
- 添加注释说明原因："功能未实现，见issue #9"
- 更新README中的示例代码

---

### 10. 文档更新

**更新内容**:

**README.md**:
- 新增"代码改进与修复 (v2.0)"章节
- 列出所有10项改进及其状态
- 更新核心特性列表（添加新功能）
- 添加CLI使用说明和参数列表
- 更新测试运行指南
- 说明待优化项

**代码内文档**:
- 所有修改的函数都添加了详细docstring
- 说明算法原理、参数含义、返回值格式
- 特别标注Student-t参数化和统计检验选择理由

---

## ⏳ 待完成任务

### 5. MCMC效率优化 (Numba/Cython)

**目标**: 加速forward-backward算法

**计划方案**:
1. 使用Numba JIT编译器加速数值密集型循环
2. 重点优化：
   - `_forward()` 函数
   - `_backward()` 函数
   - `_sample_state_sequence()` 函数
3. 预期提速：10-50倍

**当前瓶颈**:
- Python循环遍历T×K矩阵
- 大规模数据集(T>1000)时较慢

**实施步骤**:
```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def _forward_numba(log_B, log_A, log_pi):
    T, K = log_B.shape
    log_alpha = np.zeros((T, K))
    
    # 初始化
    for k in range(K):
        log_alpha[0, k] = log_pi[k] + log_B[0, k]
    
    # 递归
    for t in range(1, T):
        for k in range(K):
            log_alpha[t, k] = log_B[t, k] + \
                logsumexp(log_alpha[t-1, :] + log_A[:, k])
    
    return log_alpha
```

**预计工作量**: 2-3小时

---

### 6. 统一先验为逆伽马分布

**目标**: 确保先验设定的一致性和物理合理性

**当前问题**:
- sigma²先验可能使用了不同分布
- 需要检查并统一为逆伽马(Inverse-Gamma)分布

**逆伽马先验优势**:
- sigma²的自然共轭先验
- 物理上合理（保证正值）
- 易于Gibbs采样

**实施计划**:
1. 审查所有先验设定
2. 统一为IG(alpha, beta)形式
3. 为ENSO应用设置合理的超参数
4. 更新文档说明先验选择依据

**预计工作量**: 1-2小时

---

## 测试验证

### 运行单元测试

```bash
cd e:\test\test1_1
python -m pytest tests/test_core.py -v
```

预期结果：
```
test_log_emission_pdf_shape ... ok
test_gaussian_pdf_values ... ok
test_student_t_heavier_tails ... ok
...
----------------------------------------------------------------------
Ran 18 tests in 2.34s

OK
```

### 功能回归测试

建议手动验证以下场景：
1. 使用3状态Student-t模型拟合真实数据
2. 检查标签交换是否被正确检测和修复
3. 验证Mann-Whitney U检验输出合理性
4. 确认WAIC/LPPD数值在合理范围

---

## 📝 使用示例

### Python API用法

```python
from core.main import ENSODataLoader, RobustBayesianHMM

# 加载数据
loader = ENSODataLoader()
data = loader.load_data('nino_data.csv')

# 创建模型（已移除use_time_covariates）
hmm = RobustBayesianHMM(
    n_states=3,
    emission_dist='student_t',
    random_seed=42
)

# 拟合（自动应用标签校正）
results = hmm.fit(
    data['standardized_nino34'],
    n_iterations=5000,
    burn_in=2000,
    n_chains=4,
    verbose=True
)

# 查看结果（sigma已转换为标准差）
print(results['sigma']['mean'])  # 实际标准差，不是尺度参数
print(results['asymmetry'])      # 包含Mann-Whitney U检验结果
```

### CLI用法

```bash
# 快速开始
python cli.py --data data/nino34.csv

# 自定义参数
python cli.py --data data.csv \
              --states 4 \
              --dist gaussian \
              --iter 10000 \
              --burn 3000 \
              --output results/

# 使用配置文件
python cli.py --config config.json
```

---

## 🔄 版本历史

### v2.0 (2026-01-18)
- [OK] 修正Student-t尺度参数解释
- [OK] 实现标签交换解决算法
- [OK] 替换为Mann-Whitney U检验
- [OK] 正确实现WAIC/LPPD
- [OK] 添加CLI和配置文件支持
- [OK] 创建完整单元测试套件
- [OK] 移除未实现的use_time_covariates
- [OK] 全面更新文档

### v1.0 (初始版本)
- 基础BHMM实现
- 基础不对称性分析
- ONI阈值对比

---

## 🙏 致谢

感谢系统性代码审查发现的问题，使本项目质量得到显著提升。

---

**维护者**: nasa-91  
**仓库地址**: https://github.com/nasa-91/nasa  
**最后更新**: 2026-01-18
