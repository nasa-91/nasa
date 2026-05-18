"""
ENSO贝叶斯分析系统 - 接口定义模块

本模块定义了系统中所有核心组件的抽象基类和接口规范。
遵循依赖倒置原则（DIP），所有高层模块依赖于这些抽象接口而非具体实现。

设计模式应用：
- 策略模式：发射分布、MCMC采样器
- 工厂模式：模型创建
- 模板方法模式：算法骨架

作者: nasa-91
版本: 4.0 (Refactored)
日期: 2026-05-18
"""

from abc import ABC, abstractmethod
from typing import (
    Dict, List, Optional, Tuple, Any, 
    Union, Protocol, runtime_checkable
)
import numpy as np
import pandas as pd

from .types import (
    HMMParameters, MCMCConfig, PosteriorSummary,
    ForecastResult, AsymmetryMetrics, DataContainer,
    EmissionDistribution, ModelCriteria, ConvergenceStatus,
    ValidationResult
)


class BaseDataLoader(ABC):
    """
    数据加载器抽象基类
    
    定义数据加载和预处理的统一接口。
    所有数据源实现（CSV、NetCDF等）都应继承此类。
    
    职责：
    - 从不同格式加载数据
    - 基本的数据验证和清洗
    - 标准化输出格式为DataContainer
    
    设计原则：
    - 单一职责：只负责数据I/O
    - 开闭原则：易于扩展新数据源
    - 依赖倒置：上层代码依赖此接口
    """
    
    @abstractmethod
    def load(
        self, 
        filepath: str, 
        **kwargs
    ) -> DataContainer:
        """
        加载数据文件
        
        Args:
            filepath: 数据文件路径
            **kwargs: 格式特定的参数
            
        Returns:
            DataContainer: 标准化的数据容器
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 数据格式错误或内容无效
        """
        pass
    
    @abstractmethod
    def validate(self, data: DataContainer) -> ValidationResult:
        """
        验证数据质量
        
        Args:
            data: 待验证的数据容器
            
        Returns:
            ValidationResult: 验证结果，包含错误/警告信息
        """
        pass
    
    @abstractmethod
    def preprocess(
        self, 
        data: DataContainer,
        remove_trend: bool = False,
        remove_seasonal_cycle: bool = False,
        standardize: bool = True
    ) -> DataContainer:
        """
        数据预处理
        
        Args:
            data: 原始数据
            remove_trend: 是否去除长期趋势
            remove_seasonal_cycle: 是否去除季节循环
            standardize: 是否标准化
            
        Returns:
            处理后的DataContainer
        """
        pass


class BaseDistribution(ABC):
    """
    概率分布抽象基类 (策略模式)
    
    定义HMM发射概率分布的统一接口。
    支持高斯分布、Student-t分布等多种实现。
    
    使用场景：
    - HMM模型的发射概率计算
    - 对数似然评估
    - 参数先验定义
    
    策略模式优势：
    - 运行时切换分布类型
    - 易于添加新分布类型
    - 分布特定逻辑封装在子类中
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """分布名称标识符"""
        pass
    
    @property
    @abstractmethod
    def n_parameters(self) -> int:
        """每个状态需要的参数数量"""
        pass
    
    @abstractmethod
    def log_pdf(
        self,
        x: np.ndarray,
        mu: float,
        sigma: float,
        **params
    ) -> np.ndarray:
        """
        计算对数概率密度函数
        
        Args:
            x: 观测值 (T,) 或 (T, K)
            mu: 位置参数（均值）
            sigma: 尺度参数（标准差）
            **params: 分布特定参数（如nu for Student-t）
            
        Returns:
            对数概率密度值
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        mu: float,
        sigma: float,
        size: int = 1,
        rng: Optional[np.random.RandomState] = None,
        **params
    ) -> np.ndarray:
        """
        从分布中采样
        
        Args:
            mu: 位置参数
            sigma: 尺度参数
            size: 采样数量
            rng: 随机数生成器（确保可复现性）
            **params: 分布特定参数
            
        Returns:
            采样值数组
        """
        pass
    
    @abstractmethod
    def fit_mle(
        self,
        x: np.ndarray,
        initial_params: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        最大似然估计拟合
        
        Args:
            x: 观测数据
            initial_params: 初始参数估计
            
        Returns:
            MLE参数估计字典
        """
        pass
    
    def validate_parameters(
        self,
        mu: float,
        sigma: float,
        **params
    ) -> ValidationResult:
        """验证参数有效性"""
        errors = []
        
        if not np.isfinite(mu):
            errors.append(f"mu必须有限，得到{mu}")
        if sigma <= 0:
            errors.append(f"sigma必须>0，得到{sigma}")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )


class BaseMCMCSampler(ABC):
    """
    MCMC采样器抽象基类 (策略模式)
    
    定义马尔可夫链蒙特卡洛采样的统一接口。
    支持Gibbs采样、Metropolis-Hastings、Hamiltonian Monte Carlo等算法。
    
    职责：
    - 执行后验采样
    - 管理多条链的并行运行
    - 监控收敛诊断
    - 存储和管理后验样本
    
    设计原则：
    - 策略模式：可替换不同采样算法
    - 单一职责：只负责采样逻辑
    - 可观测性：提供进度回调和诊断指标
    """
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """采样算法名称"""
        pass
    
    @abstractmethod
    def initialize_chain(
        self,
        data: np.ndarray,
        n_states: int,
        config: MCMCConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        初始化单条MCMC链的状态
        
        Args:
            data: 观测数据 (T,)
            n_states: 状态数
            config: MCMC配置
            **kwargs: 模型特定参数
            
        Returns:
            初始状态字典（包含所有参数的初始值）
        """
        pass
    
    @abstractmethod
    def run_chain(
        self,
        data: np.ndarray,
        initial_state: Dict[str, Any],
        config: MCMCConfig,
        chain_id: int = 0,
        callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        运行单条MCMC链
        
        Args:
            data: 观测数据
            initial_state: 初始状态（来自initialize_chain）
            config: MCMC配置
            chain_id: 链标识符（用于多链并行）
            callback: 进度回调函数
            
        Returns:
            链结果字典，包含：
            - samples: 后验样本字典
            - diagnostics: 收敛诊断指标
            - log_likelihoods: 对数似然序列
            - pointwise_log_lik: 逐点对数似然（可选）
        """
        pass
    
    @abstractmethod
    def assess_convergence(
        self,
        chains_results: List[Dict[str, Any]]
    ) -> Dict[int, ConvergenceStatus]:
        """
        评估多条链的收敛情况
        
        Args:
            chains_results: 所有链的结果列表
            
        Returns:
            字典：{chain_id: ConvergenceStatus}
        """
        pass
    
    @abstractmethod
    def combine_chains(
        self,
        chains_results: List[Dict[str, Any]],
        config: MCMCConfig
    ) -> PosteriorSummary:
        """
        合并多条链的结果生成后验摘要
        
        Args:
            chains_results: 所有链的结果列表
            config: MCMC配置（用于确定稀疏等）
            
        Returns:
            完整的后验摘要
        """
        pass


class BaseHMMModel(ABC):
    """
    HMM模型抽象基类 (工厂模式 + 模板方法)
    
    定义隐马尔可夫模型的统一接口和核心算法框架。
    这是整个系统的核心抽象，所有具体HMM实现都必须继承此类。
    
    核心职责：
    - 定义模型结构（状态空间、转移矩阵等）
    - 提供fit/predict标准接口
    - 协调子组件（分布、采样器）的工作流程
    
    设计模式：
    - 工厂模式：通过create方法创建实例
    - 模板方法：定义算法骨架，子类实现细节
    - 组合模式：包含Distribution和Sampler对象
    
    使用示例：
        >>> model = RobustBayesianHMM.create(n_states=3, emission_dist='student_t')
        >>> result = model.fit(data, n_iterations=5000)
        >>> forecast = model.forecast(data, n_ahead=12)
    """
    
    @classmethod
    @abstractmethod
    def create(
        cls,
        n_states: int = 3,
        emission_dist: Union[str, EmissionDistribution] = EmissionDistribution.STUDENT_T,
        **kwargs
    ) -> 'BaseHMMModel':
        """
        工厂方法：创建模型实例
        
        Args:
            n_states: 隐藏状态数
            emission_dist: 发射分布类型
            **kwargs: 其他模型特定参数
            
        Returns:
            配置好的模型实例
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        y: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        **kwargs
    ) -> PosteriorSummary:
        """
        拟合模型到数据
        
        Args:
            y: 观测时间序列 (T,)
            dates: 时间戳（可选，用于季节性模型）
            **kwargs: 拟合参数（覆盖默认配置）
            
        Returns:
            后验推断摘要
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        y: np.ndarray,
        n_ahead: int = 1,
        **kwargs
    ) -> ForecastResult:
        """
        生成前瞻预测
        
        Args:
            y: 历史观测数据
            n_ahead: 前瞻步数
            **kwargs: 预测参数
            
        Returns:
            预测结果对象
        """
        pass
    
    @abstractmethod
    def compute_model_criteria(
        self,
        criteria: Optional[List[ModelCriteria]] = None
    ) -> Dict[str, float]:
        """
        计算模型选择准则
        
        Args:
            criteria: 要计算的准则列表（None表示全部）
            
        Returns:
            {准则名称: 准则值} 字典
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> HMMParameters:
        """
        获取当前模型参数
        
        Returns:
            HMMParameters数据类实例
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: HMMParameters) -> None:
        """
        设置模型参数
        
        Args:
            params: 新的参数集
        """
        pass
    
    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """模型是否已完成拟合"""
        pass
    
    @property
    @abstractmethod
    def posterior_summary(self) -> Optional[PosteriorSummary]:
        """获取后验摘要（未拟合时返回None）"""
        pass


class BaseAsymmetryAnalyzer(ABC):
    """
    不对称性分析器抽象基类
    
    定义ENSO不对称性分析的统一接口。
    
    分析维度：
    - 持续时间不对称性（El Nino vs La Nina持续时间差异）
    - 振幅不对称性（峰值强度对比）
    - 频率不对称性（发生频率差异）
    - 转移概率不对称性（状态转换倾向）
    - 分阶段速度不对称性（建立/衰减速度差异）
    """
    
    @abstractmethod
    def analyze(
        self,
        state_sequence: np.ndarray,
        data: np.ndarray,
        dates: pd.DatetimeIndex,
        **kwargs
    ) -> AsymmetryMetrics:
        """
        执行完整的不对称性分析
        
        Args:
            state_sequence: HMM推断的状态序列 (T,)
            data: 原始NINO3.4数据 (T,)
            dates: 时间戳
            **kwargs: 分析参数
            
        Returns:
            完整的不对称性度量结果
        """
        pass
    
    @abstractmethod
    def analyze_duration(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析持续时间不对称性"""
        pass
    
    @abstractmethod
    def analyze_amplitude(
        self,
        events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析振幅不对称性"""
        pass
    
    @abstractmethod
    def analyze_transition(
        self,
        transition_matrix: np.ndarray
    ) -> Dict[str, Any]:
        """分析转移概率不对称性"""
        pass
    
    @abstractmethod
    def analyze_phased_speed(
        self,
        events: List[Dict[str, Any]],
        data: np.ndarray,
        dates: pd.DatetimeIndex
    ) -> Dict[str, Any]:
        """分析分阶段演化速度不对称性"""
        pass


class BaseEventDetector(ABC):
    """
    ENSO事件检测器抽象基类
    
    定义ENSO事件（El Nino/La Nina）检测的标准接口。
    
    检测方法：
    - ONI阈值法（传统气象标准）
    - HMM状态法（基于隐状态序列）
    - 统计阈值法（基于分位数）
    """
    
    @abstractmethod
    def detect_events(
        self,
        data: np.ndarray,
        dates: pd.DatetimeIndex,
        method: str = 'threshold',
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        检测ENSO事件
        
        Args:
            data: NINO3.4指数序列
            dates: 时间戳
            method: 检测方法 ('threshold', 'hmm', 'statistical')
            **kwargs: 方法特定参数
            
        Returns:
            事件列表，每个事件是包含以下信息的字典：
            - type: 事件类型 ('el_nino', 'la_nina')
            - start_date: 开始日期
            - end_date: 结束日期
            - peak_value: 峰值
            - peak_date: 峰值日期
            - duration: 持续月数
            - intensity: 平均强度
        """
        pass


class BaseVisualizer(ABC):
    """
    可视化器抽象基类
    
    定义ENSO分析和HMM结果可视化的统一接口。
    
    图表类型：
    - 时间序列图（带状态标注）
    - 后验分布图
    - 转移矩阵热力图
    - 不对称性对比图
    - 收敛诊断图
    - 预测置信区间图
    """
    
    @abstractmethod
    def plot_timeseries(
        self,
        data: np.ndarray,
        dates: pd.DatetimeIndex,
        state_sequence: Optional[np.ndarray] = None,
        events: Optional[List[Dict]] = None,
        **kwargs
    ) -> plt.Figure:
        """绘制时间序列分析图"""
        pass
    
    @abstractmethod
    def plot_posteriors(
        self,
        posterior_summary: PosteriorSummary,
        **kwargs
    ) -> plt.Figure:
        """绘制后验参数分布图"""
        pass
    
    @abstractmethod
    def plot_transition_matrix(
        self,
        transition_matrix: np.ndarray,
        state_labels: Optional[List[str]] = None,
        **kwargs
    ) -> plt.Figure:
        """绘制转移概率矩阵热力图"""
        pass
    
    @abstractmethod
    def plot_asymmetry(
        self,
        metrics: AsymmetryMetrics,
        **kwargs
    ) -> plt.Figure:
        """绘制不对称性综合分析图"""
        pass
    
    @abstractmethod
    def plot_forecast(
        self,
        forecast_result: ForecastResult,
        historical_data: Optional[np.ndarray] = None,
        **kwargs
    ) -> plt.Figure:
        """绘制概率预测图"""
        pass
    
    @abstractmethod
    def save_figure(
        self,
        figure: plt.Figure,
        filepath: str,
        dpi: int = 300,
        format: str = 'png'
    ) -> None:
        """
        保存图表到文件
        
        Args:
            figure: matplotlib Figure对象
            filepath: 输出文件路径
            dpi: 分辨率
            format: 文件格式 ('png', 'pdf', 'svg')
        """
        pass


@runtime_checkable
class ModelSelectionCriterion(Protocol):
    """
    模型选择准则协议（Protocol类型）
    
    定义模型选择准则的计算接口。
    支持BIC、AIC、WAIC等多种准则的统一调用。
    """
    
    def __call__(
        self,
        model: BaseHMMModel,
        data: np.ndarray
    ) -> float:
        """
        计算准则值
        
        Args:
            model: 已拟合的模型
            data: 用于计算的数据
            
        Returns:
            准则值（越小越好）
        """
        ...


class ConfigValidator(ABC):
    """
    配置验证器抽象基类
    
    定义配置参数验证的统一接口。
    用于验证MCMC配置、模型参数等的合理性。
    """
    
    @abstractmethod
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        验证配置字典
        
        Args:
            config: 待验证的配置字典
            
        Returns:
            验证结果
        """
        pass


__all__ = [
    # 抽象基类
    'BaseDataLoader',
    'BaseDistribution',
    'BaseMCMCSampler',
    'BaseHMMModel',
    'BaseAsymmetryAnalyzer',
    'BaseEventDetector',
    'BaseVisualizer',
    'ConfigValidator',
    
    # 协议类型
    'ModelSelectionCriterion',
]
