"""
ENSO贝叶斯分析系统 - 重构后的HMM模型核心模块

本模块提供基于新架构的HMM模型实现，整合了：
- 类型安全的参数管理
- 可插拔的采样策略
- 完整的错误处理和验证
- 向后兼容的API接口

设计模式应用：
1. 工厂模式: RobustBayesianHMM.create() 创建实例
2. 策略模式: 可替换的分布类型和采样器
3. 建造者模式: 链式配置
4. 适配器模式: 兼容旧的调用方式

作者: nasa-91
版本: 4.0 (Refactored)
日期: 2026-05-18
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings
import time

from .types import (
    HMMParameters, MCMCConfig, PosteriorSummary,
    ForecastResult, AsymmetryMetrics, DataContainer,
    EmissionDistribution, ModelCriteria, ConvergenceStatus,
    ValidationResult, Array1D
)
from .interfaces import BaseHMMModel, BaseMCMCSampler
from .sampler import GibbsSampler


@dataclass
class ModelConfig:
    """
    模型配置数据类
    
    封装所有模型级别的配置参数，便于管理和验证。
    
    Attributes:
        n_states (int): 隐藏状态数
        emission_dist (str): 发射分布类型
        random_seed (int): 随机种子
        use_optimized_core (bool): 是否使用Numba优化
        mcmc_config (MCMCConfig): MCMC采样配置
    """
    n_states: int = 3
    emission_dist: str = 'student_t'
    random_seed: int = 42
    use_optimized_core: bool = True
    mcmc_config: MCMCConfig = field(default_factory=MCMCConfig)
    
    def validate(self) -> ValidationResult:
        """验证配置有效性"""
        errors = []
        
        if self.n_states < 2:
            errors.append(f"n_states必须>=2，得到{self.n_states}")
        
        if self.emission_dist not in ['gaussian', 'student_t']:
            errors.append(
                f"emission_dist必须是'gaussian'或'student_t'，"
                f"得到'{self.emission_dist}'"
            )
        
        # 验证MCMC配置
        mcmc_validation = self.mcmc_config.validate()
        
        return ValidationResult(
            is_valid=(len(errors) == 0 and mcmc_validation.is_valid),
            errors=errors + mcmc_validation.errors,
            warnings=mcmc_validation.warnings
        )


class GaussianDistribution:
    """
    高斯（正态）发射分布实现
    
    策略模式的具体策略类。
    提供标准正态分布的PDF、采样等功能。
    """
    
    name = 'gaussian'
    n_parameters = 2  # mu, sigma
    
    @staticmethod
    def log_pdf(x: np.ndarray, mu: float, sigma: float, **kwargs) -> np.ndarray:
        """计算对数概率密度"""
        sigma = max(sigma, 1e-8)
        return -0.5 * ((x - mu) / sigma)**2 - np.log(sigma) - 0.5 * np.log(2*np.pi)
    
    @staticmethod
    def sample(mu: float, sigma: float, size: int = 1,
               rng: Optional[np.random.RandomState] = None, **kwargs) -> np.ndarray:
        """从高斯分布采样"""
        if rng is None:
            rng = np.random.RandomState()
        return rng.normal(mu, max(sigma, 1e-8), size=size)
    
    @staticmethod
    def fit_mle(x: np.ndarray, **kwargs) -> Dict[str, float]:
        """最大似然估计"""
        return {
            'mu': float(np.mean(x)),
            'sigma': float(np.std(x, ddof=1))
        }


class StudentTDistribution:
    """
    Student-t发射分布实现
    
    策略模式的具体策略类。
    提供厚尾鲁棒的Student-t分布功能。
    """
    
    name = 'student_t'
    n_parameters = 3  # mu, sigma, nu
    
    @staticmethod
    def log_pdf(x: np.ndarray, mu: float, sigma: float, 
                nu: float = 5.0, **kwargs) -> np.ndarray:
        """计算对数概率密度"""
        from scipy.special import gammaln
        
        sigma = max(sigma, 1e-8)
        nu = max(nu, 2.01)  # 确保有限方差
        
        return (
            gammaln((nu + 1) / 2) - gammaln(nu / 2)
            - 0.5 * np.log(nu * np.pi) - np.log(sigma)
            - (nu + 1) / 2 * np.log(1 + ((x - mu) / sigma)**2 / nu)
        )
    
    @staticmethod
    def sample(mu: float, sigma: float, size: int = 1,
               rng: Optional[np.random.RandomState] = None,
               nu: float = 5.0, **kwargs) -> np.ndarray:
        """从Student-t分布采样"""
        if rng is None:
            rng = np.random.RandomState()
        
        # 使用标准t分布然后缩放
        z = rng.standard_t(max(nu, 2.01), size=size)
        return mu + sigma * z
    
    @staticmethod
    def fit_mle(x: np.ndarray, **kwargs) -> Dict[str, float]:
        """最大似然估计（近似）"""
        from scipy.stats import t as student_t
        from scipy.optimize import minimize_scalar
        
        # 固定mu和sigma，优化nu
        mu_est = np.median(x)  # 使用中位数更鲁棒
        sigma_est = np.std(x, ddof=1)
        
        def neg_log_lik(nu):
            nu = max(nu, 2.01)
            return -np.sum(student_t.logpdf(x, df=nu, loc=mu_est, scale=sigma_est))
        
        result = minimize_scalar(neg_log_lik, bounds=(2.01, 100), method='bounded')
        
        return {
            'mu': mu_est,
            'sigma': sigma_est,
            'nu': result.x if result.success else 5.0
        }


class DistributionFactory:
    """
    分布工厂类（工厂模式）
    
    根据名称创建对应的分布对象。
    支持运行时注册自定义分布。
    """
    
    _distributions = {
        'gaussian': GaussianDistribution,
        'student_t': StudentTDistribution,
    }
    
    @classmethod
    def create(cls, name: str):
        """
        创建分布实例
        
        Args:
            name: 分布名称 ('gaussian', 'student_t')
            
        Returns:
            分布类实例
            
        Raises:
            ValueError: 不支持的分布类型
        """
        name_lower = name.lower()
        
        if name_lower not in cls._distributions:
            available = list(cls._distributions.keys())
            raise ValueError(
                f"不支持的分布类型 '{name}'。可用选项: {available}"
            )
        
        return cls._distributions[name_lower]()
    
    @classmethod
    def register(cls, name: str, distribution_class: type) -> None:
        """
        注册自定义分布
        
        Args:
            name: 分布名称
            distribution_class: 分布类（需实现log_pdf, sample等方法）
        """
        cls._distributions[name.lower()] = distribution_class
    
    @classmethod
    def available_distributions(cls) -> List[str]:
        """获取所有可用的分布名称列表"""
        return list(cls._distributions.keys())


class RobustBayesianHMM(BaseHMMModel):
    """
    鲁棒贝叶斯隐马尔可夫模型 - 重构版 v4.0
    
    这是ENSO分析系统的核心模型类。采用全新的模块化架构，
    整合了类型安全、策略模式和工厂模式等软件工程最佳实践。
    
    主要特性：
    1. 类型安全的参数管理（HMMParameters数据类）
    2. 可插拔的发射分布（策略模式）
    3. 灵活的MCMC采样器（支持Gibbs、MH等）
    4. 完整的输入验证和错误处理
    5. 高内聚低耦合的代码结构
    6. 向后兼容的API接口
    
    设计模式：
    - 工厂模式: create() 类方法创建实例
    - 策略模式: 发射分布和采样器可替换
    - 建造者模式: 配置链式调用
    - 适配器模式: 兼容v3.x的调用方式
    
    Usage:
        >>> # 方式1：工厂方法创建（推荐）
        >>> model = RobustBayesianHMM.create(n_states=3, emission_dist='student_t')
        >>> result = model.fit(data)
        >>>
        >>> # 方式2：传统构造函数（向后兼容）
        >>> model = RobustBayesianHMM(n_states=3, emission_dist='student_t')
        >>> result = model.fit(data, n_iterations=5000)
        >>>
        >>> # 方式3：建造者模式配置
        >>> model = (RobustBayesianHMM.builder()
        ...     .with_n_states(3)
        ...     .with_emission_distribution('student_t')
        ...     .with_random_seed(42)
        ...     .build())
    """
    
    def __init__(
        self,
        n_states: int = 3,
        random_seed: int = 42,
        emission_dist: str = 'student_t',
        use_optimized_core: bool = True,
        **kwargs
    ) -> None:
        """
        初始化鲁棒贝叶斯HMM模型
        
        Args:
            n_states (int): 隐藏状态数K (必须>=2)
            random_seed (int): 随机种子，确保可复现性
            emission_dist (str): 发射分布类型 ('gaussian' 或 'student_t')
            use_optimized_core (bool): 是否使用Numba优化核心 (默认True)
            **kwargs: 其他兼容性参数（忽略）
            
        Raises:
            ValueError: 如果参数值无效
            TypeError: 如果参数类型错误
        """
        # 参数验证
        validation = self._validate_init_params(
            n_states, random_seed, emission_dist
        )
        validation.raise_if_invalid()
        
        # 存储配置
        self.config = ModelConfig(
            n_states=n_states,
            emission_dist=emission_dist,
            random_seed=random_seed,
            use_optimized_core=use_optimized_core
        )
        
        # 创建组件（依赖注入）
        self.distribution = DistributionFactory.create(emission_dist)
        self.sampler = GibbsSampler(
            emission_dist=emission_dist,
            use_optimized_core=use_optimized_core
        )
        
        # 内部状态
        self._is_fitted: bool = False
        self._posterior_summary: Optional[PosteriorSummary] = None
        self._parameters: Optional[HMMParameters] = None
        self._chains_results: List[Dict[str, Any]] = []
        
        # 创建独立的RandomState
        self._rng = np.random.RandomState(random_seed)
        
        print(f"\n{'='*70}")
        print(f"鲁棒贝叶斯隐马尔可夫模型 v4.0 (Refactored Professional)")
        print(f"{'='*70}")
        print(f"  状态数 K={n_states}")
        print(f"  发射分布: {emission_dist}")
        print(f"  采样算法: {self.sampler.algorithm_name}")
        print(f"  随机种子: {random_seed}")
        print(f"{'='*70}\n")
    
    @classmethod
    def create(
        cls,
        n_states: int = 3,
        emission_dist: Union[str, EmissionDistribution] = EmissionDistribution.STUDENT_T,
        **kwargs
    ) -> 'RobustBayesianHMM':
        """
        工厂方法：创建并配置模型实例
        
        推荐使用此方法创建模型实例，提供更清晰的API。
        
        Args:
            n_states: 隐藏状态数
            emission_dist: 发射分布类型（字符串或枚举）
            **kwargs: 传递给构造函数的其他参数
            
        Returns:
            配置好的RobustBayesianHMM实例
            
        Example:
            >>> model = RobustBayesianHMM.create(
            ...     n_states=3,
            ...     emission_dist=EmissionDistribution.STUDENT_T,
            ...     random_seed=42
            ... )
        """
        if isinstance(emission_dist, EmissionDistribution):
            emission_dist = emission_dist.value
        
        return cls(
            n_states=n_states,
            emission_dist=emission_dist,
            **kwargs
        )
    
    @classmethod
    def builder(cls) -> 'HMMBuilder':
        """
        创建建造者实例（建造者模式）
        
        Returns:
            HMMBuilder对象用于链式配置
            
        Example:
            >>> model = (RobustBayesianHMM.builder()
            ...     .with_n_states(4)
            ...     .with_emission_distribution('gaussian')
            ...     .with_mcmc_iterations(10000)
            ...     .build())
        """
        return HMMBuilder()
    
    def _validate_init_params(
        self,
        n_states: int,
        random_seed: int,
        emission_dist: str
    ) -> ValidationResult:
        """验证初始化参数"""
        errors = []
        
        if not isinstance(n_states, int) or n_states < 2:
            errors.append(f"n_states必须是整数且>=2，得到{n_states}")
        
        if not isinstance(random_seed, int):
            errors.append(f"random_seed必须是整数，得到{type(random_seed).__name__}")
        
        if emission_dist not in ['gaussian', 'student_t']:
            errors.append(
                f"emission_dist必须是'gaussian'或'student_t'，得到'{emission_dist}'"
            )
        
        return ValidationResult(is_valid=len(errors)==0, errors=errors)
    
    def fit(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        **kwargs
    ) -> PosteriorSummary:
        """
        拟合模型到观测数据
        
        这是主要的模型训练方法。执行完整的MCMC推断流程：
        1. 数据验证
        2. 初始化多条MCMC链
        3. 并行/串行运行Gibbs采样
        4. 合并后验样本
        5. 计算诊断指标
        
        Args:
            y: 观测时间序列 (T,)，NINO3.4指数
            dates: 时间戳索引（可选，用于季节性模型）
            **kwargs: 覆盖默认MCMC配置的参数：
                - n_iterations: 迭代次数
                - burn_in: 预烧期
                - n_chains: 链数
                - thin: 稀疏间隔
                - verbose: 是否显示进度
                
        Returns:
            PosteriorSummary: 后验推断结果摘要
            
        Raises:
            ValueError: 输入数据无效
            RuntimeError: MCMC采样失败
            
        Example:
            >>> model = RobustBayesianHMM.create(n_states=3)
            >>> result = model.fit(data, n_iterations=5000, burn_in=2000, n_chains=4)
            >>> print(result.parameter_means['mu'])
        """
        # 参数提取和验证
        start_time = time.perf_counter()
        
        mcmc_config = MCMCConfig(**{
            **self.config.mcmc_config.__dict__,
            **{k: v for k, v in kwargs.items() 
               if k in ['n_iterations', 'burn_in', 'n_chains', 'thinning_interval',
                        'random_seed', 'store_samples', 'compression', 'verbose']}
        })
        
        # 更新随机种子（每条链不同）
        base_seed = mcmc_config.random_seed
        
        # 数据验证
        validation = self._validate_input_data(y)
        if not validation.is_valid:
            raise ValueError("\n".join(validation.errors))
        
        T = len(y)
        K = self.config.n_states
        
        print(f"[输入验证] [OK] 所有检查通过")
        print(f"  数据长度: {T}个月 (~{T/12:.1f}年)")
        print(f"  数据范围: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  缺失值: {int(np.sum(np.isnan(y)))}个")
        
        # 运行多条MCMC链
        self._chains_results = []
        
        for chain_idx in range(mcmc_config.n_chains):
            chain_seed = base_seed + chain_idx * 1000
            chain_config = MCMCConfig(
                **mcmc_config.__dict__,
                random_seed=chain_seed
            )
            
            # 初始化链
            initial_state = self.sampler.initialize_chain(
                y, K, chain_config
            )
            initial_state['chain_state'].chain_id = chain_idx
            
            # 定义进度回调
            def make_callback(chain_id):
                def callback(iteration, chain_state):
                    if iteration % 1000 == 0 and mcmc_config.verbose:
                        pass  # 可以添加实时监控
                return callback
            
            # 运行链
            try:
                chain_result = self.sampler.run_chain(
                    y,
                    initial_state,
                    chain_config,
                    chain_id=chain_idx,
                    callback=make_callback(chain_idx)
                )
                
                self._chains_results.append(chain_result)
                
            except Exception as e:
                warnings.warn(
                    f"链{chain_idx+1}运行失败: {str(e)}",
                    RuntimeWarning
                )
        
        # 检查是否有成功的链
        if len(self._chains_results) == 0:
            raise RuntimeError("所有MCMC链都失败了，无法完成拟合")
        
        # 合并链结果
        self._posterior_summary = self.sampler.combine_chains(
            self._chains_results,
            mcmc_config
        )
        
        # 计算模型准则
        criteria = self.compute_model_criteria()
        self._posterior_summary.model_criteria = criteria
        
        # 提取最终参数
        if len(self._posterior_summary.parameter_means) > 0:
            self._parameters = HMMParameters(
                n_states=K,
                mu=self._posterior_summary.parameter_means.get('mu', np.zeros(K)),
                sigma=self._posterior_summary.parameter_means.get('sigma', np.ones(K)),
                nu=self._posterior_summary.parameter_means.get('nu'),
                transition_matrix=self._posterior_summary.parameter_means.get('transition_matrix'),
                initial_distribution=None
            )
        
        # 标记为已拟合
        self._is_fitted = True
        
        # 输出结果摘要
        elapsed_time = time.perf_counter() - start_time
        self._print_fit_summary(elapsed_time)
        
        return self._posterior_summary
    
    def _validate_input_data(self, y: np.ndarray) -> ValidationResult:
        """验证输入数据的有效性"""
        errors = []
        warnings_list = []
        
        if y is None:
            errors.append("输入数据y不能为None")
            return ValidationResult(is_valid=False, errors=errors)
        
        if not isinstance(y, np.ndarray):
            try:
                y = np.array(y, dtype=float)
            except Exception:
                errors.append("无法将y转换为numpy数组")
                return ValidationResult(is_valid=False, errors=errors)
        
        if y.ndim != 1:
            errors.append(f"y必须是一维数组，得到{y.ndim}维")
        
        if len(y) < 24:
            warnings_list.append(
                f"数据长度较短({len(y)}个月)，建议至少24个月以获得可靠结果"
            )
        
        missing_count = int(np.sum(np.isnan(y)))
        if missing_count > 0:
            ratio = missing_count / len(y)
            if ratio > 0.2:
                errors.append(f"缺失值过多({ratio:.1%})，可能严重影响结果")
            else:
                warnings_list.append(f"存在{missing_count}个缺失值({ratio:.1%})，将进行插补")
        
        inf_count = int(np.sum(np.isinf(y)))
        if inf_count > 0:
            errors.append(f"存在{inf_count}个无穷值，请检查数据源")
        
        return ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings_list
        )
    
    def _print_fit_summary(self, elapsed_time: float) -> None:
        """打印拟合结果摘要"""
        summary = self._posterior_summary
        
        print(f"\n{'='*70}")
        print(f"[拟合完成] 后验推断结果摘要")
        print(f"{'='*70}")
        
        if 'mu' in summary.parameter_means:
            mu_mean = summary.parameter_means['mu']
            mu_std = summary.parameter_stds.get('mu', np.zeros_like(mu_mean))
            print(f"\n  状态均值 (mu):")
            for k in range(len(mu_mean)):
                ci_low, ci_high = summary.parameter_ci.get('mu', (mu_mean, mu_mean))
                print(f"    状态{k}: {mu_mean[k]:.3f} ± {mu_std[k]:.3f} "
                      f"[{ci_low[k]:.3f}, {ci_high[k]:.3f}]")
        
        if summary.model_criteria:
            print(f"\n  模型选择准则:")
            for crit_name, crit_value in summary.model_criteria.items():
                print(f"    {crit_name}: {crit_value:.2f}")
        
        conv_info = summary.convergence_diagnostics
        if conv_info:
            status_str = conv_info.get('convergence_status_per_chain', {})
            print(f"\n  收敛诊断:")
            print(f"    收敛链数: {conv_info.get('converged_chains', 0)}/{conv_info.get('total_chains', 0)}")
            print(f"    总有效样本: {conv_info.get('effective_sample_size_total', 0)}")
        
        print(f"\n  运行时间: {elapsed_time:.1f}s")
        print(f"{'='*70}\n")
    
    def predict(
        self,
        y: np.ndarray,
        n_ahead: int = 12,
        n_scenarios: int = 1000,
        confidence_levels: Optional[List[float]] = None,
        **kwargs
    ) -> ForecastResult:
        """
        生成前瞻预测
        
        基于后验样本使用蒙特卡洛方法生成概率预测。
        
        Args:
            y: 历史观测数据 (T,)
            n_ahead: 前瞻月数
            n_scenarios: 蒙特卡洛场景数量
            confidence_levels: 置信水平列表 [0.05, 0.25, 0.5, 0.75, 0.95]
            **kwargs: 额外参数
            
        Returns:
            ForecastResult: 预测结果对象
            
        Raises:
            RuntimeError: 模型尚未拟合
        """
        if not self._is_fitted or self._posterior_summary is None:
            raise RuntimeError("模型尚未拟合。请先调用fit()方法。")
        
        if confidence_levels is None:
            confidence_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        K = self.config.n_states
        T_history = len(y)
        
        # 从后验中采样参数集
        scenarios = np.zeros((n_scenarios, n_ahead))
        state_probs_all = np.zeros((n_ahead, K))
        
        # 获取后验样本
        all_mu_samples = []
        all_sigma_samples = []
        all_P_samples = []
        
        for chain_result in self._chains_results:
            chain_state = chain_result['chain_state']
            if len(chain_state.samples_history['mu']) > 0:
                all_mu_samples.extend(chain_state.samples_history['mu'])
                all_sigma_samples.extend(chain_state.samples_history['sigma'])
                all_P_samples.extend(chain_state.samples_history['transition_matrix'])
        
        if len(all_mu_samples) == 0:
            # 如果没有存储样本，使用点估计
            all_mu_samples = [self._parameters.mu] * n_scenarios
            all_sigma_samples = [self._parameters.sigma] * n_scenarios
            all_P_samples = [self._parameters.transition_matrix] * n_scenarios
        
        n_available = len(all_mu_samples)
        
        # 生成场景
        for s in range(n_scenarios):
            idx = s % n_available
            mu_s = all_mu_samples[idx]
            sigma_s = all_sigma_samples[idx]
            P_s = all_P_samples[idx] if len(all_P_samples) > 0 else None
            
            current_state = K // 2  # 从中间状态开始
            scenario_values = np.zeros(n_ahead)
            state_counts = np.zeros(K)
            
            for t in range(n_ahead):
                # 状态转移
                if P_s is not None:
                    probs = P_s[current_state]
                    current_state = self._rng.choice(K, p=probs)
                
                # 从发射分布采样
                value = self.distribution.sample(
                    mu=mu_s[current_state],
                    sigma=sigma_s[current_state],
                    size=1,
                    rng=self._rng
                )[0]
                
                scenario_values[t] = value
                state_counts[current_state] += 1
            
            scenarios[s] = scenario_values
            state_probs_all += state_counts / n_ahead
        
        # 归一化状态概率
        state_probs_all /= n_scenarios
        
        # 计算统计量
        forecast_mean = np.mean(scenarios, axis=0)
        forecast_median = np.median(scenarios, axis=0)
        
        confidence_bands = {}
        for level in confidence_levels:
            lower = np.percentile(scenarios, level * 100, axis=0)
            upper = np.percentile(scenarios, (1 - level) * 100, axis=0)
            confidence_bands[level] = (lower, upper)
        
        return ForecastResult(
            mean=forecast_mean,
            median=forecast_median,
            confidence_bands=confidence_bands,
            scenarios=scenarios,
            state_probabilities=state_probs_all,
            metadata={
                'model_type': 'RobustBayesianHMM',
                'n_scenarios': n_scenarios,
                'method': 'monte_carlo_posterior',
                'emission_dist': self.config.emission_dist
            }
        )
    
    def compute_model_criteria(
        self,
        criteria: Optional[List[ModelCriteria]] = None
    ) -> Dict[str, float]:
        """
        计算模型选择准则
        
        支持多种信息准则的计算：
        - BIC/AIC/AICc: 基于样本量的惩罚项
        - WAIC/LPPD: 贝叶斯准则（需要逐点对数似然）
        
        Args:
            criteria: 要计算的准则列表（None表示全部）
            
        Returns:
            {准则名: 准则值} 字典
        """
        if not self._is_fitted or self._posterior_summary is None:
            warnings.warn("模型尚未拟合，返回空字典", UserWarning)
            return {}
        
        if criteria is None:
            criteria = list(ModelCriteria)
        
        results = {}
        T = len(self._posterior_summary.state_sequence) if \
            self._posterior_summary.state_sequence is not None else 0
        K = self.config.n_states
        
        # 参数数量（每个状态：mu, sigma, (+nu), 转移矩阵K*(K-1), 初始分布K-1）
        if self.config.emission_dist == 'student_t':
            n_params = K * 3 + K * (K - 1) + (K - 1)
        else:
            n_params = K * 2 + K * (K - 1) + (K - 1)
        
        log_lik = self._posterior_summary.log_likelihood
        
        for criterion in criteria:
            if criterion == ModelCriteria.BIC:
                results['BIC'] = -2 * log_lik + n_params * np.log(T) if T > 0 else 0
                
            elif criterion == ModelCriteria.AIC:
                results['AIC'] = -2 * log_lik + 2 * n_params
                
            elif criterion == ModelCriteria.AICC:
                if T > n_params + 1:
                    correction = 2 * n_params * (n_params + 1) / (T - n_params - 1)
                    results['AICc'] = -2 * log_lik + 2 * n_params + correction
                else:
                    results['AICc'] = float('inf')
                    
            elif criterion in [ModelCriteria.WAIC, ModelCriteria.LPPD]:
                # 简化的WAIC计算（完整版本需要逐点似然）
                if hasattr(self, '_compute_waic_penalty'):
                    lppd = log_lik  # 近似
                    p_waic = self._compute_waic_penalty()
                    waic = -2 * (lppd - p_waic)
                    results['WAIC'] = waic
                    results['LPPD'] = lppd
                else:
                    results['WAIC'] = None
                    results['LPPD'] = log_lik
        
        return {k: v for k, v in results.items() if v is not None}
    
    def get_parameters(self) -> HMMParameters:
        """
        获取当前模型参数
        
        Returns:
            HMMParameters数据类实例
            
        Raises:
            RuntimeError: 模型尚未拟合
        """
        if not self._is_fitted or self._parameters is None:
            raise RuntimeError("模型尚未拟合。请先调用fit()方法。")
        
        return self._parameters
    
    def set_parameters(self, params: HMMParameters) -> None:
        """
        设置模型参数（用于预训练或手动指定）
        
        Args:
            params: 新的参数集
        """
        self._parameters = params
        self.config.n_states = params.n_states
        
        # 如果有转移矩阵，标记为已初始化
        if params.transition_matrix is not None:
            self._is_fitted = True
    
    @property
    def is_fitted(self) -> bool:
        """模型是否已完成拟合"""
        return self._is_fitted
    
    @property
    def posterior_summary(self) -> Optional[PosteriorSummary]:
        """获取后验摘要"""
        return self._posterior_summary


class HMMBuilder:
    """
    HMM模型建造者（建造者模式）
    
    提供流畅的链式API用于配置和构建HMM模型。
    
    Example:
        >>> model = (HMMBuilder()
        ...     .with_n_states(3)
        ...     .with_emission_distribution('student_t')
        ...     .with_random_seed(42)
        ...     .with_numba_optimization(True)
        ...     .with_mcmc_iterations(5000)
        ...     .with_burn_in(2000)
        ...     .build())
    """
    
    def __init__(self) -> None:
        """初始化建造者"""
        self._n_states = 3
        self._emission_dist = 'student_t'
        self._random_seed = 42
        self._use_optimized_core = True
        self._mcmc_config = MCMCConfig()
    
    def with_n_states(self, n_states: int) -> 'HMMBuilder':
        """设置隐藏状态数"""
        self._n_states = n_states
        return self
    
    def with_emission_distribution(self, dist: str) -> 'HMMBuilder':
        """设置发射分布类型"""
        self._emission_dist = dist
        return self
    
    def with_random_seed(self, seed: int) -> 'HMMBuilder':
        """设置随机种子"""
        self._random_seed = seed
        return self
    
    def with_numba_optimization(self, enabled: bool) -> 'HMMBuilder':
        """设置是否启用Numba优化"""
        self._use_optimized_core = enabled
        return self
    
    def with_mcmc_iterations(self, n: int) -> 'HMMBuilder':
        """设置MCMC迭代次数"""
        self._mcmc_config.n_iterations = n
        return self
    
    def with_burn_in(self, burn_in: int) -> 'HMMBuilder':
        """设置预烧期"""
        self._mcmc_config.burn_in = burn_in
        return self
    
    def with_n_chains(self, n: int) -> 'HMMBuilder':
        """设置MCMC链数"""
        self._mcmc_config.n_chains = n
        return self
    
    def build(self) -> RobustBayesianHMM:
        """构建并返回配置好的模型实例"""
        model = RobustBayesianHMM(
            n_states=self._n_states,
            emission_dist=self._emission_dist,
            random_seed=self._random_seed,
            use_optimized_core=self._use_optimized_core
        )
        model.config.mcmc_config = self._mcmc_config
        return model


__all__ = [
    # 配置类
    'ModelConfig',
    
    # 分布实现
    'GaussianDistribution',
    'StudentTDistribution',
    'DistributionFactory',
    
    # 主模型类
    'RobustBayesianHMM',
    'HMMBuilder',
]
