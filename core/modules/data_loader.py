"""
ENSO贝叶斯分析系统 - 数据加载器模块

本模块提供数据加载、验证和预处理功能。
支持多种数据格式（CSV、NOAA ASCII、NetCDF），遵循单一职责原则。

设计模式：
- 策略模式：不同格式的加载器
- 工厂方法：根据文件扩展名自动选择加载器
- 模板方法：统一的预处理流程

作者: nasa-91
版本: 4.0 (Refactored)
日期: 2026-05-18
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Type
from abc import ABC, abstractmethod
from pathlib import Path
import warnings

from .types import (
    DataContainer, ValidationResult,
    DateIndex, Array1D
)
from .interfaces import BaseDataLoader


class DataLoaderStrategy(ABC):
    """
    数据加载策略抽象基类（策略模式）
    
    定义特定格式数据加载的接口。
    每种数据格式（CSV、NetCDF等）对应一个具体策略类。
    """
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """格式名称"""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """支持的文件扩展名列表"""
        pass
    
    @abstractmethod
    def load(
        self,
        filepath: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        加载数据文件
        
        Args:
            filepath: 文件路径
            **kwargs: 格式特定参数
            
        Returns:
            包含'dates'和'nino34'的字典
        """
        pass
    
    def validate_file(self, filepath: str) -> ValidationResult:
        """验证文件是否存在和可读"""
        errors = []
        warnings_list = []
        
        path = Path(filepath)
        
        if not path.exists():
            errors.append(f"文件不存在: {filepath}")
            return ValidationResult(is_valid=False, errors=errors)
        
        if not path.is_file():
            errors.append(f"路径不是文件: {filepath}")
            return ValidationResult(is_valid=False, errors=errors)
        
        if not path.suffix.lower() in self.supported_extensions:
            warnings_list.append(
                f"文件扩展名'{path.suffix}'不在推荐列表中: {self.supported_extensions}"
            )
        
        # 检查文件大小
        file_size = path.stat().st_size
        if file_size == 0:
            errors.append("文件为空")
            return ValidationResult(is_valid=False, errors=errors)
        
        if file_size < 100:  # 太小可能有问题
            warnings_list.append(f"文件大小异常小 ({file_size} bytes)")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings_list
        )


class CSVDataLoader(DataLoaderStrategy):
    """
    CSV格式数据加载器
    
    支持标准CSV格式的NINO3.4数据文件。
    
    预期格式示例：
    ```csv
    date,nino34
    1950-01,-0.45
    1950-02,-0.52
    ```
    
    Attributes:
        default_date_col: 默认日期列索引（从0开始）
        default_value_col: 默认值列索引
        supported_date_formats: 支持的日期格式列表
    """
    
    def __init__(self) -> None:
        self.default_date_col = 0
        self.default_value_col = 1
        self.supported_date_formats = [
            '%Y-%m-%d',
            '%Y-%m', 
            '%Y/%m/%d',
            '%Y%m%d'
        ]
    
    @property
    def format_name(self) -> str:
        return 'csv'
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.csv', '.txt']
    
    def load(
        self,
        filepath: str,
        date_col: Optional[int] = None,
        value_col: Optional[int] = None,
        date_format: Optional[str] = None,
        header: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        加载CSV格式数据
        
        Args:
            filepath: CSV文件路径
            date_col: 日期列位置（默认自动检测）
            value_col: 数值列位置（默认自动检测）
            date_format: 日期解析格式（None表示自动推断）
            header: 表头行数
            
        Returns:
            {'dates': DatetimeIndex, 'nino34': ndarray}
            
        Raises:
            ValueError: 列数不足或数据格式错误
        """
        # 使用默认值
        date_col = date_col or self.default_date_col
        value_col = value_col or self.default_value_col
        
        # 读取CSV
        try:
            df = pd.read_csv(filepath, header=header)
        except Exception as e:
            raise ValueError(f"CSV读取失败: {str(e)}")
        
        # 验证列数
        if len(df.columns) <= max(date_col, value_col):
            raise ValueError(
                f"CSV只有{len(df.columns)}列，但需要访问第{max(date_col, value_col)+1}列"
            )
        
        # 解析日期
        date_strings = df.iloc[:, date_col].astype(str).str.strip()
        
        if date_format:
            dates = pd.to_datetime(date_strings, format=date_format)
        else:
            # 尝试多种格式
            dates = None
            for fmt in self.supported_date_formats:
                try:
                    dates = pd.to_datetime(date_strings, format=fmt)
                    break
                except (ValueError, TypeError):
                    continue
            
            if dates is None:
                # 最后尝试pandas自动解析
                try:
                    dates = pd.to_datetime(date_strings, format='mixed')
                except Exception as e:
                    raise ValueError(f"无法解析日期列: {str(e)}")
        
        # 提取数值
        nino34 = df.iloc[:, value_col].values.astype(float)
        
        return {
            'dates': dates,
            'nino34': nino34,
            'source': filepath,
            'format': 'csv'
        }


class NOAAASCIIDataLoader(DataLoaderStrategy):
    """
    NOAA ASCII格式数据加载器
    
    支持NOAA标准的ASCII文本格式。
    
    预期格式示例：
    ```
    1950  1 -0.45
    1950  2 -0.52
    1950  3 -0.38
    ```
    
    每行包含：年 月 值
    """
    
    def __init__(self) -> None:
        self.default_skiprows = 0
    
    @property
    def format_name(self) -> str:
        return 'noaa_ascii'
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.txt', '.asc', '.dat']
    
    def load(
        self,
        filepath: str,
        skiprows: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        加载NOAA ASCII格式数据
        
        Args:
            filepath: 文件路径
            skiprows: 跳过的开头行数
            
        Returns:
            {'dates': DatetimeIndex, 'nino34': ndarray}
        """
        skiprows = skiprows or self.default_skiprows
        
        data_lines = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines[skiprows:]:
            parts = line.split()
            if len(parts) >= 3:
                try:
                    year = int(parts[0])
                    month = int(parts[1])
                    value = float(parts[2])
                    data_lines.append((year, month, value))
                except (ValueError, IndexError):
                    continue  # 跳过无效行
        
        if len(data_lines) == 0:
            raise ValueError(f"未能从{filepath}中解析出有效数据")
        
        years, months, values = zip(*data_lines)
        
        date_strings = [f"{y}-{m:02d}-01" for y, m in zip(years, months)]
        dates = pd.to_datetime(date_strings)
        nino34 = np.array(values, dtype=float)
        
        return {
            'dates': dates,
            'nino34': nino34,
            'source': filepath,
            'format': 'noaa_ascii'
        }


class NetCDFDataLoader(DataLoaderStrategy):
    """
    NetCDF格式数据加载器
    
    支持NetCDF/NetCDF4格式的气候数据文件。
    需要安装netCDF4库。
    """
    
    def __init__(self) -> None:
        self._check_dependency()
    
    def _check_dependency(self) -> None:
        """检查netCDF4是否可用"""
        try:
            import netCDF4
            self._nc_module = netCDF4
        except ImportError:
            raise ImportError(
                "需要安装netCDF4库才能读取NetCDF文件。\n"
                "请运行: pip install netCDF4"
            )
    
    @property
    def format_name(self) -> str:
        return 'netcdf'
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.nc', '.nc4', '.nc3']
    
    def load(
        self,
        filepath: str,
        variable_name: str = 'nino34',
        time_var: str = 'time',
        **kwargs
    ) -> Dict[str, Any]:
        """
        加载NetCDF格式数据
        
        Args:
            filepath: 文件路径
            variable_name: NINO3.4变量名
            time_var: 时间变量名
            
        Returns:
            {'dates': DatetimeIndex, 'nino34': ndarray}
        """
        try:
            ds = self._nc_module.Dataset(filepath, 'r')
        except Exception as e:
            raise ValueError(f"NetCDF文件打开失败: {str(e)}")
        
        try:
            # 提取时间
            time_var_obj = ds.variables[time_var]
            dates = pd.to_datetime(time_var_obj[:].astype(str))
            
            # 提取NINO3.4值
            var_obj = ds.variables[variable_name]
            nino34 = np.array(var_obj[:]).flatten().astype(float)
            
            # 处理填充值
            if hasattr(var_obj, '_FillValue'):
                fill_value = var_obj._FillValue
                mask = (nino34 == fill_value) | ~np.isfinite(nino34)
                nino34[mask] = np.nan
                
        finally:
            ds.close()
        
        return {
            'dates': dates,
            'nino34': nino34,
            'source': filepath,
            'format': 'netcdf'
        }


class ENSODataLoader(BaseDataLoader):
    """
    ENSO数据统一加载器（外观模式 + 工厂方法）
    
    提供统一的加载数据接口，自动选择合适的加载策略。
    对外隐藏不同格式的复杂性，提供简洁易用的API。
    
    设计特点：
    - 外观模式：统一入口，简化复杂子系统
    - 工厂方法：根据文件扩展名自动选择策略
    - 单一职责：只负责协调，不包含业务逻辑
    
    Attributes:
        _strategies: 已注册的数据加载策略字典
        _default_strategy: 默认使用的策略名称
    
    Usage:
        >>> loader = ENSODataLoader()
        >>> data = loader.load('data/nino34.csv')
        >>> print(data.T)  # 数据长度
        >>> print(data.date_range)  # 时间范围
    """
    
    def __init__(self) -> None:
        """
        初始化数据加载器并注册所有支持的格式策略
        """
        self._strategies: Dict[str, DataLoaderStrategy] = {}
        self._register_default_strategies()
        self.data_info: Dict[str, Any] = {}
    
    def _register_default_strategies(self) -> None:
        """注册内置的数据加载策略"""
        strategies = [
            CSVDataLoader(),
            NOAAASCIIDataLoader(),
        ]
        
        # 尝试注册NetCDF（可选依赖）
        try:
            strategies.append(NetCDFDataLoader())
        except ImportError:
            pass
        
        for strategy in strategies:
            for ext in strategy.supported_extensions:
                self._strategies[ext.lower()] = strategy
    
    def register_strategy(
        self, 
        strategy: DataLoaderStrategy,
        extensions: Optional[List[str]] = None
    ) -> None:
        """
        注册自定义数据加载策略
        
        Args:
            strategy: 数据加载器策略实例
            extensions: 要关联的文件扩展名（None使用策略默认值）
        """
        exts = extensions or strategy.supported_extensions
        for ext in exts:
            self._strategies[ext.lower()] = strategy
    
    def _detect_format(self, filepath: str) -> str:
        """
        根据文件扩展名检测数据格式
        
        Args:
            filepath: 文件路径
            
        Returns:
            格式标识符字符串
        """
        ext = Path(filepath).suffix.lower()
        
        if ext in self._strategies:
            return self._strategies[ext].format_name
        
        # 默认返回CSV
        return 'csv'
    
    def _get_strategy(self, filepath: str) -> DataLoaderStrategy:
        """
        获取适合指定文件的加载策略
        
        Args:
            filepath: 文件路径
            
        Returns:
            数据加载策略实例
            
        Raises:
            ValueError: 不支持的文件格式
        """
        ext = Path(filepath).suffix.lower()
        
        if ext not in self._strategies:
            raise ValueError(
                f"不支持的文件格式 '{ext}'。"
                f"支持格式: {list(set(s.format_name for s in self._strategies.values()))}"
            )
        
        return self._strategies[ext]
    
    def load(
        self,
        filepath: str,
        format_type: str = 'auto',
        **kwargs
    ) -> DataContainer:
        """
        加载数据文件（主入口方法）
        
        这是数据加载的主要公共API。自动处理格式检测、
        数据验证和标准化输出。
        
        Args:
            filepath: 数据文件路径（绝对或相对路径）
            format_type: 数据格式 ('auto'=自动检测, 'csv', 'noaa_ascii', 'nc')
            **kwargs: 传递给具体加载器的额外参数
            
        Returns:
            DataContainer: 标准化的数据容器对象
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 格式不支持或数据无效
            
        Example:
            >>> loader = ENSODataLoader()
            >>> data = loader.load('nino34.csv')
            >>> print(f"加载了{data.T}个月的数据")
            >>> print(f"时间范围: {data.date_range}")
        """
        # 格式检测
        if format_type == 'auto':
            format_type = self._detect_format(filepath)
        
        print(f"[数据加载] 正在加载数据: {filepath}")
        print(f"          检测到格式: {format_type}")
        
        # 获取加载策略
        strategy = self._get_strategy(filepath)
        
        # 验证文件
        validation = strategy.validate_file(filepath)
        if not validation.is_valid:
            raise FileNotFoundError("\n".join(validation.errors))
        
        if validation.warnings:
            for w in validation.warnings:
                warnings.warn(w, UserWarning)
        
        # 执行加载
        raw_data = strategy.load(filepath, **kwargs)
        
        # 创建DataContainer并进行预处理
        container = self._create_container(raw_data)
        container = self.preprocess(container)
        
        # 更新元信息
        self.data_info.update({
            'source': filepath,
            'format': format_type,
            'n_observations': container.T,
            'date_range': container.date_range,
            'missing_values': int(np.sum(np.isnan(container.raw_nino34))),
            'loader_type': type(strategy).__name__
        })
        
        print(f"[数据加载] [OK] 成功加载 {container.T} 个月的数据")
        print(f"          时间范围: {container.date_range}")
        print(f"          缺失值数量: {self.data_info['missing_values']}")
        
        return container
    
    def validate(self, data: DataContainer) -> ValidationResult:
        """
        验证数据质量
        
        检查项：
        - 数据长度是否足够（>=24个月）
        - 缺失值比例是否过高（>10%）
        - 是否存在极端异常值（>5个标准差）
        - 时间序列是否连续（无大间隔）
        
        Args:
            data: 待验证的数据容器
            
        Returns:
            ValidationResult: 包含所有问题的详细报告
        """
        errors = []
        warnings_list = []
        
        T = data.T
        
        # 检查最小长度
        if T < 24:
            errors.append(f"数据长度过短({T}个月)，建议至少24个月")
        
        # 检查缺失值
        missing_count = int(np.sum(np.isnan(data.raw_nino34)))
        missing_ratio = missing_count / T
        if missing_count > 0:
            if missing_ratio > 0.1:
                errors.append(f"缺失值过多({missing_ratio:.1%})，可能影响分析质量")
            else:
                warnings_list.append(f"存在{missing_count}个缺失值({missing_ratio:.1%})")
        
        # 检查有效数据量
        valid_data = data.raw_nino34[~np.isnan(data.raw_nino34)]
        if len(valid_data) < 12:
            errors.append(f"有效数据点过少({len(valid_data)}个)")
        
        # 检查异常值
        if len(valid_data) > 0:
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            if std_val > 0:
                z_scores = np.abs((valid_data - mean_val) / std_val)
                extreme_count = np.sum(z_scores > 5)
                if extreme_count > 0:
                    warnings_list.append(
                        f"发现{extreme_count}个极端异常值(|z|>5)，请检查数据质量"
                    )
        
        # 检查时间连续性
        if T > 1:
            time_gaps = np.diff(data.dates.values.astype('datetime64[D]').astype(int))
            median_gap = np.median(time_gaps)
            max_gap = np.max(time_gaps)
            
            # 允许30天左右的正常间隔（月度数据）
            expected_gap_days = 30
            if median_gap > expected_gap_days * 2:
                warnings_list.append(
                    f"数据间隔不均匀（中位数{median_gap:.0f}天），可能是非月度数据"
                )
            
            # 检查是否有大的时间间断
            if max_gap > 365:  # 超过1年的间隔
                gap_months = max_gap / 30
                warnings_list.append(
                    f"发现最大时间间断约{gap_months:.1f}个月，请注意"
                )
        
        result = ValidationResult(
            is_valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings_list
        )
        
        if result.warnings and result.is_valid:
            print(f"[数据验证] 通过（有{len(result.warnings)}条警告）")
            for w in result.warnings[:3]:  # 只显示前3条
                print(f"           [WARN] {w}")
        elif result.is_valid:
            print("[数据验证] [OK] 完全通过")
        else:
            print(f"[数据验证] [FAIL] 发现{len(errors)}个错误:")
            for e in errors:
                print(f"           [ERROR] {e}")
        
        return result
    
    def preprocess(
        self,
        data: DataContainer,
        remove_trend: bool = False,
        remove_seasonal_cycle: bool = False,
        standardize: bool = True,
        **kwargs
    ) -> DataContainer:
        """
        数据预处理流水线
        
        按顺序执行以下步骤：
        1. 缺失值插补（线性插值）
        2. 可选：去除长期趋势
        3. 可选：去除季节循环
        4. 标准化（z-score）
        
        Args:
            data: 原始数据容器
            remove_trend: 是否去除线性趋势
            remove_seasonal_cycle: 是否去除季节性周期
            standardize: 是否进行z-score标准化
            
        Returns:
            处理后的新DataContainer实例
        """
        processed_nino34 = data.raw_nino34.copy()
        
        # 步骤1：缺失值插补
        nan_mask = np.isnan(processed_nino34)
        if np.any(nan_mask):
            print(f"[预处理] 正在插补{np.sum(nan_mask)}个缺失值...")
            valid_indices = np.where(~nan_mask)[0]
            invalid_indices = np.where(nan_mask)[0]
            
            if len(valid_indices) > 1:
                processed_nino34[nan_mask] = np.interp(
                    invalid_indices,
                    valid_indices,
                    processed_nino34[valid_indices]
                )
        
        # 步骤2：去除长期趋势（可选）
        if remove_trend:
            print("[预处理] 正在去除长期趋势...")
            T = len(processed_nino34)
            x = np.arange(T)
            slope, intercept = np.polyfit(x, processed_nino34, 1)
            trend = slope * x + intercept
            processed_nino34 = processed_nino34 - trend
            print(f"         趋势斜率: {slope:.6f} /月")
        
        # 步骤3：去除季节循环（可选）
        if remove_seasonal_cycle:
            print("[预处理] 正在去除季节循环...")
            # 计算各月均值
            months = data.dates.month
            monthly_means = np.zeros(12)
            for m in range(1, 13):
                mask = (months == m)
                if np.sum(mask) > 0:
                    monthly_means[m-1] = np.mean(processed_nino34[mask])
            
            # 减去对应的月均值
            seasonal_cycle = np.array([monthly_means[m-1] for m in months])
            processed_nino34 = processed_nino34 - seasonal_cycle
            print(f"         季节振幅范围: [{monthly_means.min():.3f}, {monthly_means.max():.3f}]")
        
        # 步骤4：标准化
        if standardize:
            mean_val = np.mean(processed_nino34)
            std_val = np.std(processed_nino34)
            
            if std_val > 0:
                processed_nino34 = (processed_nino34 - mean_val) / std_val
            else:
                warnings.warn("数据标准差为0，跳过标准化", UserWarning)
            
            print(f"[预处理] 标准化完成 (mean={mean_val:.3f}, std={std_val:.3f})")
        
        # 返回新的DataContainer
        return DataContainer(
            dates=data.dates,
            raw_nino34=data.raw_nino34,
            standardized_nino34=processed_nino34,
            dates_index=data.dates_index,
            metadata={
                **data.metadata,
                'preprocessing_applied': {
                    'remove_trend': remove_trend,
                    'remove_seasonal_cycle': remove_seasonal_cycle,
                    'standardized': standardize,
                    'missing_values_imputed': int(np.any(nan_mask)) if 'nan_mask' in dir() else False
                }
            }
        )
    
    def _create_container(self, raw_data: Dict[str, Any]) -> DataContainer:
        """
        从原始数据创建标准化的DataContainer
        
        Args:
            raw_data: 包含'dates'和'nino34'的字典
            
        Returns:
            DataContainer实例
        """
        dates = raw_data['dates']
        nino34 = raw_data['nino34'].astype(float)
        
        # 初始时 standardized 和 raw 相同
        return DataContainer(
            dates=dates,
            raw_nino34=nino34,
            standardized_nino34=nino34.copy(),
            dates_index=dates,
            metadata={
                'source': raw_data.get('source', ''),
                'format': raw_data.get('format', 'unknown'),
                'loaded_at': pd.Timestamp.now().isoformat()
            }
        )


__all__ = [
    # 抽象基类
    'DataLoaderStrategy',
    
    # 具体策略
    'CSVDataLoader',
    'NOAAASCIIDataLoader',
    'NetCDFDataLoader',
    
    # 统一加载器（主要使用这个）
    'ENSODataLoader',
]
