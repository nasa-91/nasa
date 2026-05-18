"""
ENSO贝叶斯分析系统 - 模块包初始化

本包提供ENSO（厄尔尼诺-南方涛动）贝叶斯分析的核心功能。
采用专业级软件工程架构，遵循设计模式和最佳实践。

模块结构：
- types: 基础类型定义和数据类
- interfaces: 抽象接口和协议定义
- data_loader: 数据加载和预处理
- sampler: MCMC采样引擎
- hmm: HMM模型核心实现
- analysis: 不对称性和事件分析
- forecast: 概率预测系统

设计模式：
- 策略模式: 发射分布、采样算法
- 工厂模式: 模型创建
- 外观模式: 统一API入口
- 观察者模式: 进度监控

作者: nasa-91
版本: 4.0 (Professional Refactored)
日期: 2026-05-18
"""

from .types import (
    # 枚举类型
    EmissionDistribution,
    ModelCriteria,
    ConvergenceStatus,
    
    # 数据类
    HMMParameters,
    MCMCConfig,
    PosteriorSummary,
    ForecastResult,
    AsymmetryMetrics,
    DataContainer,
    ValidationResult,
    
    # 类型别名
    Array1D, Array2D, Array3D,
    DateIndex, Series, DataFrame,
)

from .interfaces import (
    # 抽象基类
    BaseDataLoader,
    BaseDistribution,
    BaseMCMCSampler,
    BaseHMMModel,
    BaseAsymmetryAnalyzer,
    BaseEventDetector,
    BaseVisualizer,
)

from .data_loader import ENSODataLoader
from .sampler import GibbsSampler

__version__ = '4.0.0'
__author__ = 'nasa-91'
__email__ = 'h2078309440@163.com'

__all__ = [
    # 版本信息
    '__version__', '__author__', '__email__',
    
    # 类型系统
    'EmissionDistribution', 'ModelCriteria', 'ConvergenceStatus',
    'HMMParameters', 'MCMCConfig', 'PosteriorSummary',
    'ForecastResult', 'AsymmetryMetrics', 'DataContainer',
    'ValidationResult',
    
    # 接口
    'BaseDataLoader', 'BaseDistribution', 'BaseMCMCSampler',
    'BaseHMMModel', 'BaseAsymmetryAnalyzer',
    'BaseEventDetector', 'BaseVisualizer',
    
    # 核心组件
    'ENSODataLoader', 'GibbsSampler',
]
