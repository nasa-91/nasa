"""
ENSO贝叶斯分析系统 - 基础类型定义模块

本模块定义了系统中使用的所有核心数据类型、类型别名和数据结构。
遵循PEP 484类型注解规范，提供完整的类型支持。

设计原则：
- 使用TypeDict和NamedTuple定义复合数据结构
- 提供清晰的类型别名增强可读性
- 支持Optional和Union处理缺失值和多态
- 所有公共类型都导出供其他模块使用

作者: nasa-91
版本: 4.0 (Refactored)
日期: 2026-05-18
"""

from typing import (
    Dict, List, Optional, Tuple, Union, Any, 
    Callable, Iterator, TypeVar, Generic,
    NamedTuple, TypedDict
)
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pandas as pd


class EmissionDistribution(Enum):
    """
    发射分布类型枚举
    
    定义HMM模型支持的发射概率分布类型：
    - GAUSSIAN: 标准高斯分布（正态分布）
    - STUDENT_T: Student-t分布（厚尾鲁棒）
    """
    GAUSSIAN = 'gaussian'
    STUDENT_T = 'student_t'


class ModelCriteria(Enum):
    """
    模型选择准则枚举
    
    定义用于模型比较的信息准则：
    - BIC: 贝叶斯信息准则（惩罚复杂度）
    - AIC: 赤池信息准则
    - AICC: 修正AIC（小样本）
    - WAIC: 广义瓦坦abe-赤池信息准则（贝叶斯）
    - LPPD: 对数预测密度
    """
    BIC = 'bic'
    AIC = 'aic'
    AICC = 'aicc'
    WAIC = 'waic'
    LPPD = 'lppd'


class ConvergenceStatus(Enum):
    """
    MCMC收敛状态枚举
    
    定义链的收敛诊断结果：
    - CONVERGED: 收敛良好（R-hat < 1.1, ESS > 100）
    - WARNING: 轻微问题（R-hat < 1.2, ESS > 50）
    - NOT_CONVERGED: 未收敛
    - INSUFFICIENT_DATA: 数据不足无法判断
    """
    CONVERGED = auto()
    WARNING = auto()
    NOT_CONVERGED = auto()
    INSUFFICIENT_DATA = auto()


@dataclass
class HMMParameters:
    """
    HMM模型参数数据类
    
    存储完整的HMM参数集，包括发射参数、转移矩阵等。
    
    Attributes:
        n_states (int): 隐藏状态数K
        mu (np.ndarray): 状态均值向量 (K,)
        sigma (np.ndarray): 状态标准差向量 (K,)
        nu (np.ndarray): Student-t自由度向量 (K,)，仅Student-t分布使用
        transition_matrix (np.ndarray): 转移概率矩阵 (K, K)
        initial_distribution (np.ndarray): 初始状态分布 (K,)
    
    Example:
        >>> params = HMMParameters(
        ...     n_states=3,
        ...     mu=np.array([-1.0, 0.0, 1.0]),
        ...     sigma=np.array([0.3, 0.2, 0.5]),
        ...     transition_matrix=np.array([[0.9, 0.08, 0.02], ...])
        ... )
    """
    n_states: int
    mu: np.ndarray
    sigma: np.ndarray
    nu: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None
    initial_distribution: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """验证参数维度一致性"""
        if len(self.mu) != self.n_states:
            raise ValueError(f"mu长度({len(self.mu)})必须等于n_states({self.n_states})")
        if len(self.sigma) != self.n_states:
            raise ValueError(f"sigma长度({len(self.sigma)})必须等于n_states({self.n_states})")
        if self.nu is not None and len(self.nu) != self.n_states:
            raise ValueError(f"nu长度({len(self.nu)})必须等于n_states({self.n_states})")


@dataclass
class MCMCConfig:
    """
    MCMC采样配置数据类
    
    封装所有MCMC采样相关的配置参数。
    
    Attributes:
        n_iterations (int): 每条链的总迭代次数
        burn_in (int): 预烧期迭代数（丢弃）
        n_chains (int): 并行MCMC链数
        thinning_interval (int): 稀疏间隔（保留每thin个样本）
        random_seed (int): 基础随机种子
        store_samples (bool): 是否存储后验样本
        compression (str): 样本压缩方式 ('none', 'float16')
        verbose (bool): 是否输出详细进度
    
    Example:
        >>> config = MCMCConfig(
        ...     n_iterations=5000,
        ...     burn_in=2000,
        ...     n_chains=4,
        ...     thinning_interval=5
        ... )
    """
    n_iterations: int = 5000
    burn_in: int = 2000
    n_chains: int = 4
    thinning_interval: int = 5
    random_seed: int = 42
    store_samples: bool = True
    compression: str = 'none'
    verbose: bool = True
    
    def validate(self) -> None:
        """验证配置参数的合理性"""
        if self.burn_in >= self.n_iterations:
            raise ValueError(f"burn_in({self.burn_in})必须小于n_iterations({self.n_iterations})")
        if self.thinning_interval < 1:
            raise ValueError(f"thinning_interval必须>=1，得到{self.thinning_interval}")
        if self.n_chains < 1:
            raise ValueError(f"n_chains必须>=1，得到{self.n_chains}")
        
        effective_samples = (self.n_iterations - self.burn_in) // self.thinning_interval
        if effective_samples < 10:
            raise ValueError(
                f"有效样本数过少({effective_samples})。"
                f"请增加n_iterations或减少burn_in/thinning_interval"
            )


@dataclass
class PosteriorSummary:
    """
    后验推断摘要数据类
    
    存储MCMC后验分析的结果摘要。
    
    Attributes:
        parameter_means (Dict[str, np.ndarray]): 参数后验均值
        parameter_stds (Dict[str, np.ndarray]): 参数后验标准差
        parameter_ci (Dict[str, Tuple[np.ndarray, np.ndarray]]): 95%置信区间
        state_sequence (np.ndarray): 最可能状态序列 (T,)
        log_likelihood (float): 对数似然值
        model_criteria (Dict[str, float]): 各信息准则值
        convergence_diagnostics (Dict[str, Any]): 收敛诊断指标
        pointwise_log_lik (Optional[np.ndarray]): 逐点对数似然 (S, T)
    
    Example:
        >>> summary = PosteriorSummary(
        ...     parameter_means={'mu': np.array([-1.0, 0.0, 1.0])},
        ...     log_likelihood=-1234.5,
        ...     state_sequence=np.array([0, 0, 1, 1, 2, ...])
        ... )
    """
    parameter_means: Dict[str, np.ndarray] = field(default_factory=dict)
    parameter_stds: Dict[str, np.ndarray] = field(default_factory=dict)
    parameter_ci: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    state_sequence: Optional[np.ndarray] = None
    log_likelihood: float = 0.0
    model_criteria: Dict[str, float] = field(default_factory=dict)
    convergence_diagnostics: Dict[str, Any] = field(default_factory=dict)
    pointwise_log_lik: Optional[np.ndarray] = None
    
    @property
    def n_states(self) -> int:
        """推断隐藏状态数"""
        if 'mu' in self.parameter_means:
            return len(self.parameter_means['mu'])
        return 0
    
    def get_parameter(self, name: str) -> Tuple[np.ndarray, np.ndarray, Tuple]:
        """
        获取指定参数的后验统计量
        
        Args:
            name: 参数名称 ('mu', 'sigma', 'nu', 'transition_matrix'等)
            
        Returns:
            Tuple of (mean, std, (ci_lower, ci_upper))
            
        Raises:
            KeyError: 如果参数不存在
        """
        if name not in self.parameter_means:
            raise KeyError(f"参数'{name}'不在后验摘要中")
        
        mean = self.parameter_means[name]
        std = self.parameter_stds.get(name, np.zeros_like(mean))
        ci = self.parameter_ci.get(name, (mean - 1.96*std, mean + 1.96*std))
        
        return mean, std, ci


@dataclass
class ForecastResult:
    """
    概率预测结果数据类
    
    存储蒙特卡洛概率预测的完整结果。
    
    Attributes:
        mean (np.ndarray): 预测均值序列 (n_ahead,)
        median (np.ndarray): 预测中位数序列 (n_ahead,)
        confidence_bands (Dict[float, Tuple[np.ndarray, np.ndarray]]): 置信区间
        scenarios (np.ndarray): 所有场景路径 (n_scenarios, n_ahead)
        state_probabilities (np.ndarray): 状态概率 (n_ahead, K)
        crps_scores (Optional[np.ndarray]): CRPS评分序列
        prediction_dates (Optional[pd.DatetimeIndex]): 预测时间点
        metadata (Dict[str, Any]): 元数据（方法、参数等）
    
    Example:
        >>> forecast = ForecastResult(
        ...     mean=np.array([0.5, 0.8, 1.0]),
        ...     scenarios=np.random.randn(1000, 12),
        ...     n_scenarios=1000,
        ...     n_ahead=12
        ... )
    """
    mean: np.ndarray
    median: np.ndarray
    confidence_bands: Dict[float, Tuple[np.ndarray, np.ndarray]]
    scenarios: np.ndarray
    state_probabilities: np.ndarray
    crps_scores: Optional[np.ndarray] = None
    prediction_dates: Optional[pd.DatetimeIndex] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_ahead(self) -> int:
        """前瞻期数"""
        return len(self.mean)
    
    @property
    def n_scenarios(self) -> int:
        """场景数量"""
        return self.scenarios.shape[0]
    
    def get_quantile(self, q: float) -> np.ndarray:
        """
        获取指定分位数的预测值
        
        Args:
            q: 分位数 (0-1)
            
        Returns:
            分位数序列
        """
        return np.quantile(self.scenarios, q, axis=0)


@dataclass
class AsymmetryMetrics:
    """
    不对称性度量数据类
    
    存储ENSO不对称性分析的完整结果。
    
    Attributes:
        duration_asymmetry (Dict): 持续时间不对称性
        amplitude_asymmetry (Dict): 振幅不对称性
        frequency_asymmetry (Dict): 频率不对称性
        transition_asymmetry (Dict): 转移概率不对称性
        phased_speed_asymmetry (Dict): 分阶段速度不对称性
        overall_significance (bool): 总体显著性
        confidence_level (float): 置信水平
    """
    duration_asymmetry: Dict[str, Any] = field(default_factory=dict)
    amplitude_asymmetry: Dict[str, Any] = field(default_factory=dict)
    frequency_asymmetry: Dict[str, Any] = field(default_factory=dict)
    transition_asymmetry: Dict[str, Any] = field(default_factory=dict)
    phased_speed_asymmetry: Dict[str, Any] = field(default_factory=dict)
    overall_significance: bool = False
    confidence_level: float = 0.95
    
    def summary(self) -> str:
        """生成不对称性分析摘要文本"""
        lines = [
            "=" * 60,
            "ENSO不对称性分析结果",
            "=" * 60,
            f"总体显著性: {'显著' if self.overall_significance else '不显著'} "
            f"(置信水平: {self.confidence_level:.1%})"
        ]
        
        for name, metrics in [
            ("持续时间", self.duration_asymmetry),
            ("振幅", self.amplitude_asymmetry),
            ("频率", self.frequency_asymmetry),
            ("转移概率", self.transition_asymmetry),
            ("分阶段速度", self.phased_speed_asymmetry)
        ]:
            if metrics:
                sig = " [显著]" if metrics.get('is_significant', False) else ""
                lines.append(f"\n{name}不对称性:{sig}")
                for key, value in metrics.items():
                    if key != 'is_significant':
                        lines.append(f"  {key}: {value}")
        
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class DataContainer:
    """
    数据容器数据类
    
    统一封装ENSO分析所需的所有数据。
    
    Attributes:
        dates (pd.DatetimeIndex): 时间戳索引
        raw_nino34 (np.ndarray): 原始NINO3.4指数
        standardized_nino34 (np.ndarray): 标准化后的NINO3.4指数
        dates_index (pd.DatetimeIndex): 完整日期索引
        metadata (Dict[str, Any]): 数据元数据
    """
    dates: pd.DatetimeIndex
    raw_nino34: np.ndarray
    standardized_nino34: np.ndarray
    dates_index: pd.DatetimeIndex
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def T(self) -> int:
        """数据长度"""
        return len(self.dates)
    
    @property
    def date_range(self) -> str:
        """日期范围描述"""
        return f"{self.dates.iloc[0].strftime('%Y-%m')} 至 {self.dates.iloc[-1].strftime('%Y-%m')}"
    
    def get_period(self, start_date: str, end_date: str) -> 'DataContainer':
        """
        提取指定时间段的数据
        
        Args:
            start_date: 开始日期 ('YYYY-MM')
            end_date: 结束日期 ('YYYY-MM')
            
        Returns:
            新的DataContainer实例
        """
        mask = (self.dates >= start_date) & (self.dates <= end_date)
        return DataContainer(
            dates=self.dates[mask],
            raw_nino34=self.raw_nino34[mask],
            standardized_nino34=self.standardized_nino34[mask],
            dates_index=self.dates_index[mask],
            metadata=self.metadata.copy()
        )


# 类型别名定义（增强可读性）
Array1D = np.ndarray
Array2D = np.ndarray
Array3D = np.ndarray
DateIndex = pd.DatetimeIndex
Series = pd.Series
DataFrame = pd.DataFrame

# 泛型类型变量（用于工厂模式等）
T = TypeVar('T')
ModelType = TypeVar('ModelType', bound='BaseHMMModel')
DistributionType = TypeVar('DistributionType', bound='BaseDistribution')


class ValidationResult:
    """
    输入验证结果类
    
    封装参数验证的结果，包括成功/失败状态和详细错误信息。
    
    Attributes:
        is_valid (bool): 验证是否通过
        errors (List[str]): 错误消息列表
        warnings (List[str]): 警告消息列表
        validated_params (Dict[str, Any]): 验证后的参数值
    """
    
    def __init__(
        self, 
        is_valid: bool = True,
        errors: Optional[List[str]] = None,
        warnings: Optional[List[str]] = None,
        validated_params: Optional[Dict[str, Any]] = None
    ) -> None:
        self.is_valid = is_valid
        self.errors = errors or []
        self.warnings = warnings or []
        self.validated_params = validated_params or {}
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """合并两个验证结果"""
        return ValidationResult(
            is_valid=self.is_valid and other.is_valid,
            errors=self.errors + other.errors,
            warnings=self.warnings + other.warnings,
            validated_params={**self.validated_params, **other.validated_params}
        )
    
    def raise_if_invalid(self) -> None:
        """如果验证失败则抛出异常"""
        if not self.is_valid:
            error_msg = "\n".join([f"  - {e}" for e in self.errors])
            raise ValueError(f"参数验证失败:\n{error_msg}")


__all__ = [
    # 枚举类型
    'EmissionDistribution',
    'ModelCriteria', 
    'ConvergenceStatus',
    
    # 数据类
    'HMMParameters',
    'MCMCConfig',
    'PosteriorSummary',
    'ForecastResult',
    'AsymmetryMetrics',
    'DataContainer',
    'ValidationResult',
    
    # 类型别名
    'Array1D', 'Array2D', 'Array3D',
    'DateIndex', 'Series', 'DataFrame',
    
    # 泛型变量
    'T', 'ModelType', 'DistributionType',
]
