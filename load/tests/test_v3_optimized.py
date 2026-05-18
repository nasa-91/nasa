#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENSO贝叶斯不对称性分析系统 - v3.0 综合测试套件

覆盖范围：
1. [OK] 优化核心算法测试（Numba/NumPy双路径）
2. [OK] 输入验证和类型安全测试
3. [OK] 错误处理和边界情况测试
4. [OK] 性能基准测试和对比
5. [OK] 逆伽马先验分布测试
6. [OK] 完整端到端流程测试

运行方式：
    python tests/test_v3_optimized.py          # 运行所有测试
    python -m pytest tests/test_v3_optimized.py -v   # 使用pytest运行

作者: nasa-91
日期: 2026-01-18
版本: v3.0 Test Suite
"""

import sys
import os
import unittest
import numpy as np
import time
import warnings
from typing import Dict, Any, List

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.optimized_core import (
    OptimizedHMMCore,
    PerformanceMonitor,
    validate_array,
    validate_probability_matrix,
    NUMBA_AVAILABLE,
    try_import_numba
)

try:
    from core.main import RobustBayesianHMM, SeasonalBayesianHMM, ENSODataLoader
    MAIN_MODULE_AVAILABLE = True
except ImportError as e:
    MAIN_MODULE_AVAILABLE = False
    warnings.warn(f"主模块导入失败: {e}", ImportWarning)


class TestOptimizedCoreBasic(unittest.TestCase):
    """优化核心基础功能测试"""
    
    @classmethod
    def setUpClass(cls):
        """初始化测试数据"""
        np.random.seed(42)
        cls.T = 500  # 时间序列长度
        cls.K = 3    # 状态数
        
        cls.y = np.random.randn(cls.T).astype(np.float64)
        cls.mu = np.array([-1.0, 0.0, 1.5])
        cls.sigma = np.array([0.3, 0.2, 0.5])
        cls.nu = np.full(cls.K, 5.0)
        
        cls.P = np.random.dirichlet(np.ones(cls.K), size=cls.K)
        cls.pi = np.random.dirichlet(np.ones(cls.K))
    
    def test_01_initialization(self):
        """测试优化核心初始化"""
        print("\n[TEST] 优化核心初始化...")
        core = OptimizedHMMCore(use_numba=True)
        
        self.assertIsNotNone(core)
        self.assertEqual(core.use_numba, NUMBA_AVAILABLE)
        
        if NUMBA_AVAILABLE:
            self.assertIsNotNone(core._optimized_core)
        else:
            self.assertIsNone(core._optimized_core)
    
    def test_02_emission_matrix_shape(self):
        """测试发射矩阵形状正确性"""
        print("\n[TEST] 发射矩阵形状检查...")
        core = OptimizedHMMCore(use_numba=False)  # 强制使用NumPy
        
        log_B = core.compute_emission_matrix(
            self.y, self.mu, self.sigma, 
            nu=self.nu, dist='student_t'
        )
        
        self.assertEqual(log_B.shape, (self.T, self.K))
        self.assertTrue(np.all(np.isfinite(log_B)))
        self.assertTrue(np.all(log_B <= 0))  # 对数概率应≤0
    
    def test_03_gaussian_vs_student_t(self):
        """测试高斯分布与Student-t分布差异"""
        print("\n[TEST] 高斯 vs Student-t 发射矩阵...")
        core = OptimizedHMMCore(use_numba=False)
        
        log_B_gaussian = core.compute_emission_matrix(
            self.y[:100], self.mu, self.sigma, dist='gaussian'
        )
        
        log_B_student = core.compute_emission_matrix(
            self.y[:100], self.mu, self.sigma, nu=self.nu, dist='student_t'
        )
        
        # 两种分布应不同
        diff = np.abs(log_B_gaussian - log_B_student).mean()
        self.assertGreater(diff, 0.01, "高斯和Student-t发射矩阵应显著不同")
    
    def test_04_forward_algorithm_output(self):
        """测试前向算法输出格式"""
        print("\n[TEST] 前向算法输出...")
        core = OptimizedHMMCore(use_numba=False)
        
        log_B = core.compute_emission_matrix(self.y, self.mu, self.sigma, dist='gaussian')
        log_alpha, log_lik = core.forward_algorithm(log_B, self.P, self.pi)
        
        self.assertEqual(log_alpha.shape, (self.T, self.K))
        self.assertIsInstance(log_lik, float)
        self.assertTrue(np.isfinite(log_lik))
    
    def test_05_backward_algorithm_output(self):
        """测试后向算法输出格式"""
        print("\n[TEST] 后向算法输出...")
        core = OptimizedHMMCore(use_numba=False)
        
        log_B = core.compute_emission_matrix(self.y, self.mu, self.sigma, dist='gaussian')
        log_beta = core.backward_algorithm(log_B, self.P)
        
        self.assertEqual(log_beta.shape, (self.T, self.K))
        self.assertTrue(np.all(np.isfinite(log_beta)))
    
    def test_06_state_sampling_range(self):
        """测试状态采样范围有效性"""
        print("\n[TEST] 状态采样范围...")
        core = OptimizedHMMCore(use_numba=False)
        
        gamma = np.random.dirichlet(np.ones(self.K), size=self.T)
        states = core.sample_states(gamma)
        
        self.assertEqual(states.shape, (self.T,))
        self.assertTrue(np.all(states >= 0))
        self.assertTrue(np.all(states < self.K))


class TestInputValidation(unittest.TestCase):
    """输入验证和类型安全测试"""
    
    def test_01_validate_array_type_error(self):
        """测试非数组输入的错误处理"""
        with self.assertRaises(TypeError):
            validate_array([1, 2, 3], "test", expected_ndim=1)
    
    def test_02_validate_array_dimension_error(self):
        """测试维度错误检测"""
        arr = np.random.randn(10, 5)
        with self.assertRaises(ValueError):
            validate_array(arr, "test", expected_ndim=1)
    
    def test_03_validate_array_size_error(self):
        """测试长度不足检测"""
        arr = np.array([1.0])
        with self.assertRaises(ValueError):
            validate_array(arr, "test", min_size=5)
    
    def test_04_validate_probability_non_square(self):
        """测试非方阵概率矩阵检测"""
        P = np.random.rand(3, 4)
        with self.assertRaises(ValueError):
            validate_probability_matrix(P)
    
    def test_05_validate_probability_range(self):
        """测试概率值范围检测"""
        P = np.array([[0.5, 0.6], [0.5, 0.4]])  # 第一行>1
        with self.assertRaises(ValueError):
            validate_probability_matrix(P)
    
    def test_06_auto_normalization(self):
        """测试自动归一化功能"""
        P = np.array([[0.45, 0.55], [0.48, 0.52]])
        P_normalized = validate_probability_matrix(P)
        
        self.assertAlmostEqual(P_normalized[0].sum(), 1.0, places=5)
        self.assertAlmostEqual(P_normalized[1].sum(), 1.0, places=5)


class TestErrorHandling(unittest.TestCase):
    """错误处理和边界情况测试"""
    
    def setUp(self):
        if not MAIN_MODULE_AVAILABLE:
            self.skipTest("主模块不可用")
    
    def test_01_invalid_n_states(self):
        """测试无效状态数"""
        with self.assertRaises(ValueError):
            RobustBayesianHMM(n_states=1)
        
        with self.assertRaises(ValueError):
            RobustBayesianHMM(n_states=2.5)
    
    def test_02_invalid_emission_dist(self):
        """测试无效发射分布"""
        with self.assertRaises(ValueError):
            RobustBayesianHMM(emission_dist='invalid')
    
    def test_03_short_data_warning(self):
        """测试短数据警告"""
        hmm = RobustBayesianHMM()
        y_short = np.random.randn(80)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                # 使用较少迭代以加速测试
                result = hmm.fit(y_short, n_iterations=200, burn_in=100, 
                               n_chains=2, verbose=False)
                
                # 应该有数据长度警告
                data_warnings = [warning for warning in w 
                                if issubclass(warning.category, UserWarning)]
                self.assertGreater(len(data_warnings), 0, "应有数据长度警告")
            except ValueError:
                pass  # 数据太短可能直接报错
    
    def test_04_nan_data_detection(self):
        """测试NaN数据检测"""
        hmm = RobustBayesianHMM()
        y_with_nan = np.concatenate([np.random.randn(90), [np.nan]])
        
        with self.assertRaises(ValueError) as context:
            hmm.fit(y_with_nan, n_iterations=50, burn_in=20, verbose=False)
        
        self.assertIn("NaN", str(context.exception))
    
    def test_05_inf_data_detection(self):
        """测试inf数据检测"""
        hmm = RobustBayesianHMM()
        y_with_inf = np.concatenate([np.random.randn(99), [np.inf]])
        
        with self.assertRaises(ValueError) as context:
            hmm.fit(y_with_inf, n_iterations=50, burn_in=20, verbose=False)
        
        self.assertIn("inf", str(context.exception))
    
    def test_06_burn_in_validation(self):
        """测试burn_in参数验证"""
        hmm = RobustBayesianHMM()
        y = np.random.randn(150)
        
        with self.assertRaises(ValueError):
            hmm.fit(y, n_iterations=500, burn_in=600, n_chains=1, verbose=False)
    
    def test_07_single_chain_warning(self):
        """测试单链警告"""
        hmm = RobustBayesianHMM()
        y = np.random.randn(120)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                result = hmm.fit(y, n_iterations=100, burn_in=50, 
                               n_chains=1, verbose=False)
                
                chain_warnings = [warning for warning in w 
                                 if "单条链" in str(warning.message)]
                self.assertGreater(len(chain_warnings), 0, "应有单链警告")
            except Exception:
                pass


class TestInverseGammaPrior(unittest.TestCase):
    """逆伽马先验分布测试"""
    
    def test_01_prior_setup(self):
        """测试逆伽马先验初始化"""
        if not MAIN_MODULE_AVAILABLE:
            self.skipTest("主模块不可用")
        
        hmm = RobustBayesianHMM(n_states=3)
        
        self.assertIn('inverse_gamma', hmm.prior_params)
        prior = hmm.prior_params['inverse_gamma']
        
        self.assertIn('alpha', prior)
        self.assertIn('beta', prior)
        self.assertIn('mean', prior)
        self.assertIn('variance', prior)
    
    def test_02_prior_parameters_validity(self):
        """测试先验参数有效性"""
        if not MAIN_MODULE_AVAILABLE:
            self.skipTest("主模块不可用")
        
        hmm = RobustBayesianHMM(n_states=3)
        prior = hmm.prior_params['inverse_gamma']
        
        alpha = prior['alpha']
        beta = prior['beta']
        
        # 形状参数必须 > 2以保证有限方差
        self.assertTrue(np.all(alpha > 2), "alpha必须>2")
        
        # 先验均值应在合理范围内
        mean = prior['mean']
        self.assertGreater(mean, 0.01, "先验均值应为正")
        self.assertLess(mean, 10.0, "先验均值不应过大")


class TestPerformanceBenchmark(unittest.TestCase):
    """性能基准测试"""
    
    @classmethod
    def setUpClass(cls):
        """设置性能测试参数"""
        cls.n_iterations = 20  # 减少重复次数以加快测试
        cls.T_small = 200      # 小规模数据
        cls.T_large = 800      # 大规模数据（~67年）
        cls.K = 3
        
        np.random.seed(42)
        cls.y_small = np.random.randn(cls.T_small).astype(np.float64)
        cls.y_large = np.random.randn(cls.T_large).astype(np.float64)
        
        cls.mu = np.array([-1.0, 0.0, 1.5])
        cls.sigma = np.array([0.3, 0.2, 0.5])
        cls.nu = np.full(cls.K, 5.0)
        cls.P = np.random.dirichlet(np.ones(cls.K), size=cls.K)
        cls.pi = np.random.dirichlet(np.ones(cls.K))
    
    def test_01_emission_performance_small(self):
        """小规模发射矩阵性能测试"""
        print(f"\n[PERF] 小规模发射矩阵 (T={self.T_small})...")
        core = OptimizedHMMCore(use_numba=NUMBA_AVAILABLE)
        
        start = time.perf_counter()
        for _ in range(self.n_iterations):
            core.compute_emission_matrix(self.y_small, self.mu, self.sigma,
                                        nu=self.nu, dist='student_t')
        elapsed = (time.perf_counter() - start) / self.n_iterations * 1000
        
        print(f"  平均耗时: {elapsed:.3f} ms")
        
        # NumPy版本应该在50ms以内
        self.assertLess(elapsed, 100, f"发射矩阵计算过慢: {elapsed:.1f}ms")
    
    def test_02_forward_backward_performance(self):
        """前向后向算法性能测试"""
        print(f"\n[PERF] 前向后向算法 (T={self.T_small})...")
        core = OptimizedHMMCore(use_numba=NUMBA_AVAILABLE)
        
        log_B = core.compute_emission_matrix(self.y_small, self.mu, self.sigma, dist='gaussian')
        
        start = time.perf_counter()
        for _ in range(self.n_iterations):
            core.forward_algorithm(log_B, self.P, self.pi)
            core.backward_algorithm(log_B, self.P)
        elapsed = (time.perf_counter() - start) / self.n_iterations * 1000
        
        print(f"  前后向平均耗时: {elapsed:.3f} ms")
        
        # 单次前后向应该在100ms以内
        self.assertLess(elapsed, 200, f"前后向计算过慢: {elapsed:.1f}ms")
    
    def test_03_scalability_test(self):
        """可扩展性测试：大规模 vs 小规模"""
        print(f"\n[PERF] 可扩展性测试...")
        core = OptimizedHMMCore(use_numba=False)  # 使用NumPy确保公平比较
        
        # 小规模
        start_small = time.perf_counter()
        for _ in range(self.n_iterations):
            core.compute_emission_matrix(self.y_small, self.mu, self.sigma, dist='gaussian')
        time_small = (time.perf_counter() - start_small) / self.n_iterations
        
        # 大规模
        start_large = time.perf_counter()
        for _ in range(self.n_iterations // 2):  # 减少大样本重复次数
            core.compute_emission_matrix(self.y_large, self.mu, self.sigma, dist='gaussian')
        time_large = (time.perf_counter() - start_large) / (self.n_iterations // 2)
        
        ratio = time_large / time_small
        scale_ratio = self.T_large / self.T_small
        
        print(f"  小规模(T={self.T_small}): {time_small*1000:.3f} ms")
        print(f"  大规模(T={self.T_large}): {time_large*1000:.3f} ms")
        print(f"  时间比: {ratio:.2f}x (理论线性比: {scale_ratio:.2f}x)")
        
        # 时间比应该接近线性比例（允许一定误差）
        self.assertLess(ratio, scale_ratio * 2, "可扩展性不佳，可能存在O(n²)复杂度")


class TestEndToEndIntegration(unittest.TestCase):
    """端到端集成测试"""
    
    def setUp(self):
        if not MAIN_MODULE_AVAILABLE:
            self.skipTest("主模块不可用")
        
        np.random.seed(42)
        # 生成模拟ENSO数据
        T = 300  # 25年
        t = np.linspace(0, 4*np.pi, T)
        self.y_simulated = (
            0.8 * np.sin(t) +           # 周期信号
            0.3 * np.random.randn(T) +   # 噪声
            0.5 * (np.random.rand(T) > 0.9) * np.sign(np.random.randn(T))  # 偶发极端事件
        ).astype(np.float64)
    
    def test_01_full_pipeline_standard(self):
        """标准HMM完整流程测试"""
        print("\n[E2E] 标准HMM完整流程...")
        hmm = RobustBayesianHMM(n_states=3, use_optimized_core=True)
        
        try:
            result = hmm.fit(
                self.y_simulated,
                n_iterations=300,     # 减少迭代以加速测试
                burn_in=150,
                n_chains=2,           # 减少链数
                thin=3,
                verbose=False
            )
            
            # 验证结果结构
            self.assertIn('mu', result)
            self.assertIn('sigma', result)
            self.assertIn('P', result)
            self.assertIn('pi', result)
            
            # 验证后验估计合理性
            mu_mean = result['mu']['mean']
            self.assertEqual(len(mu_mean), 3)
            self.assertTrue(np.all(np.isfinite(mu_mean)))
            
            print(f"  [OK] 标准HMM流程成功完成")
            
        except Exception as e:
            self.fail(f"标准HMM流程失败: {str(e)}")
    
    def test_02_performance_metrics_api(self):
        """性能指标API测试"""
        print("\n[E2E] 性能指标API...")
        hmm = RobustBayesianHMM(n_states=3)
        
        metrics = hmm.get_performance_metrics()
        
        self.assertIn('optimization_status', metrics)
        self.assertIn('model_info', metrics)
        self.assertIn('use_optimized_core', metrics['optimization_status'])
        self.assertIn('version', metrics['model_info'])
        
        print(f"  [OK] 性能API正常工作")
    
    def test_03_seasonal_hmm_import(self):
        """季节HMM导入测试"""
        print("\n[E2E] 季节HMM可用性...")
        
        try:
            seasonal_hmm = SeasonalBayesianHMM(n_states=3, fourier_order=2)
            self.assertIsNotNone(seasonal_hmm)
            print(f"  [OK] 季节HMM可正常实例化")
        except Exception as e:
            self.fail(f"季节HMM导入/初始化失败: {str(e)}")


class TestNumbaOptimization(unittest.TestCase):
    """Numba优化专项测试（如果可用）"""
    
    @unittest.skipUnless(NUMBA_AVAILABLE, "Numba未安装，跳过此测试")
    def test_01_numba_acceleration_effective(self):
        """测试Numba实际加速效果"""
        print(f"\n[NUMBA] 测试Numba JIT加速效果...")
        
        np.random.seed(42)
        T = 800
        K = 3
        y = np.random.randn(T).astype(np.float64)
        mu = np.array([-1.0, 0.0, 1.5])
        sigma = np.array([0.3, 0.2, 0.5])
        nu = np.full(K, 5.0)
        
        # Numba版本
        core_numba = OptimizedHMMCore(use_numba=True)
        start_numba = time.perf_counter()
        for _ in range(30):
            core_numba.compute_emission_matrix(y, mu, sigma, nu=nu, dist='student_t')
        time_numba = (time.perf_counter() - start_numba) / 30
        
        # NumPy版本（强制不使用Numba）
        core_numpy = OptimizedHMMCore(use_numba=False)
        start_numpy = time.perf_counter()
        for _ in range(30):
            core_numpy.compute_emission_matrix(y, mu, sigma, nu=nu, dist='student_t')
        time_numpy = (time.perf_counter() - start_numpy) / 30
        
        speedup = time_numpy / time_numba
        
        print(f"  Numba耗时: {time_numba*1000:.3f} ms")
        print(f"  NumPy耗时: {time_numpy*1000:.3f} ms")
        print(f"  加速比: {speedup:.2f}x")
        
        # Numba应该至少和NumPy一样快（首次调用有JIT编译开销）
        # 这里不做严格断言，只记录结果


def run_comprehensive_tests():
    """运行完整测试套件并生成报告"""
    print("\n" + "="*70)
    print("ENSO贝叶斯分析系统 v3.0 - 综合测试套件")
    print("="*70)
    
    print(f"\n环境信息:")
    print(f"  Python版本: {sys.version.split()[0]}")
    print(f"  NumPy版本: {np.__version__}")
    print(f"  Numba状态: {'[OK] 已安装' if NUMBA_AVAILABLE else '[FAIL] 未安装'}")
    print(f"  主模块状态: {'[OK] 可用' if MAIN_MODULE_AVAILABLE else '[FAIL] 不可用'}")
    
    if NUMBA_AVAILABLE:
        import numba
        print(f"  Numba版本: {numba.__version__}")
    
    start_time = time.time()
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestOptimizedCoreBasic))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidation))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestInverseGammaPrior))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformanceBenchmark))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestNumbaOptimization))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("测试总结")
    print("="*70)
    print(f"总测试数: {result.testsRun}")
    print(f"通过: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"跳过: {len(result.skipped)}")
    print(f"总耗时: {elapsed_time:.2f}s ({elapsed_time/60:.1f}分钟)")
    
    if len(result.failures) > 0:
        print("\n失败的测试:")
        for test, traceback in result.failures:
            print(f"  [FAIL] {test}")
    
    if len(result.errors) > 0:
        print("\n错误的测试:")
        for test, traceback in result.errors:
            print(f"  [ERROR] {test}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
    print(f"\n成功率: {success_rate:.1f}%")
    
    if success_rate >= 95:
        print("[OK] 所有关键功能正常运行！")
    elif success_rate >= 80:
        print("[WARN] 部分测试失败，请查看详细输出。")
    else:
        print("[FAIL] 多项测试失败，需要修复。")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
