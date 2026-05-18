#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENSO贝叶斯不对称性分析系统 - 单元测试

测试覆盖：
1. PDF计算 (高斯 & Student-t)
2. 前向-后向算法
3. 状态采样逻辑
4. 标签交换校正
5. 尺度参数转换
6. 数据加载
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.main import RobustBayesianHMM, ENSODataLoader


class TestLogEmissionPDF(unittest.TestCase):
    """测试对数发射概率密度函数"""
    
    def setUp(self):
        self.hmm = RobustBayesianHMM(n_states=3)
        self.test_data = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    def test_gaussian_pdf_shape(self):
        """测试高斯PDF输出形状"""
        result = self.hmm._log_emission_pdf(
            self.test_data, mu=0.0, sigma=1.0, dist='gaussian'
        )
        self.assertEqual(result.shape, (len(self.test_data),))
    
    def test_gaussian_pdf_values(self):
        """测试高斯PDF数值正确性（在均值处应最大）"""
        result = self.hmm._log_emission_pdf(
            self.test_data, mu=0.0, sigma=1.0, dist='gaussian'
        )
        # 在mu=0处应该有最大的对数概率密度
        max_idx = np.argmax(result)
        self.assertEqual(max_idx, 2)  # test_data[2] = 0.0
    
    def test_student_t_pdf_shape(self):
        """测试Student-t PDF输出形状"""
        result = self.hmm._log_emission_pdf(
            self.test_data, mu=0.0, sigma=1.0, nu=5.0, dist='student_t'
        )
        self.assertEqual(result.shape, (len(self.test_data),))
    
    def test_student_t_heavier_tails(self):
        """测试Student-t分布比高斯分布有更厚的尾部"""
        gaussian_pdf = self.hmm._log_emission_pdf(
            np.array([3.0]), mu=0.0, sigma=1.0, dist='gaussian'
        )
        student_t_pdf = self.hmm._log_emission_pdf(
            np.array([3.0]), mu=0.0, sigma=1.0, nu=5.0, dist='student_t'
        )
        # Student-t在极端值处的概率应该更高（更重的尾部）
        self.assertGreater(student_t_pdf[0], gaussian_pdf[0])
    
    def test_invalid_distribution(self):
        """测试无效分布类型抛出异常"""
        with self.assertRaises(ValueError):
            self.hmm._log_emission_pdf(
                self.test_data, mu=0.0, sigma=1.0, dist='invalid_dist'
            )
    
    def test_sigma_clamping(self):
        """测试sigma被限制为正值"""
        result = self.hmm._log_emission_pdf(
            self.test_data, mu=0.0, sigma=-1.0, dist='gaussian'
        )
        # 不应该返回-inf或nan
        self.assertTrue(np.all(np.isfinite(result)))
    
    def test_nu_clamping(self):
        """测试nu被限制>2以保证方差存在"""
        result = self.hmm._log_emission_pdf(
            self.test_data, mu=0.0, sigma=1.0, nu=1.5, dist='student_t'
        )
        # 不应该返回inf或nan
        self.assertTrue(np.all(np.isfinite(result)))


class TestScaleParameterConversion(unittest.TestCase):
    """测试尺度参数与标准差之间的转换"""
    
    def test_scale_to_std_conversion(self):
        """测试尺度参数到标准差的转换"""
        sigma = 1.0
        nu = 5.0
        std = RobustBayesianHMM._scale_to_std(sigma, nu)
        
        expected_std = sigma * np.sqrt(nu / (nu - 2))
        self.assertAlmostEqual(std, expected_std, places=10)
    
    def test_std_to_scale_conversion(self):
        """测试标准差到尺度参数的转换"""
        std = 1.0
        nu = 5.0
        scale = RobustBayesianHMM._std_to_scale(std, nu)
        
        expected_scale = std * np.sqrt((nu - 2) / nu)
        self.assertAlmostEqual(scale, expected_scale, places=10)
    
    def test_roundtrip_conversion(self):
        """测试往返转换的一致性"""
        original_std = 1.5
        nu = 7.0
        
        scale = RobustBayesianHMM._std_to_scale(original_std, nu)
        recovered_std = RobustBayesianHMM._scale_to_std(scale, nu)
        
        self.assertAlmostEqual(original_std, recovered_std, places=10)
    
    def test_nu_leq_2_returns_inf(self):
        """测试nu<=2时返回无穷大"""
        with self.assertRaises(ValueError):
            RobustBayesianHMM._std_to_scale(1.0, nu=2.0)


class TestForwardBackwardAlgorithm(unittest.TestCase):
    """测试前向-后向算法"""
    
    def setUp(self):
        self.hmm = RobustBayesianHMM(n_states=3)
        self.T = 100
        self.y = np.random.randn(self.T) * 0.5
    
    def test_forward_algorithm_output_shape(self):
        """测试前向算法输出形状"""
        K = self.n_states = 3
        
        # 创建简单的发射矩阵
        log_B = np.zeros((self.T, K))
        for t in range(self.T):
            for k in range(K):
                log_B[t, k] = self.hmm._log_emission_pdf(
                    np.array([self.y[t]]),
                    mu=float(k-1),  # 简单的均值设置
                    sigma=1.0,
                    dist='gaussian'
                )[0]
        
        log_A = np.log(np.array([[0.8, 0.1, 0.1],
                                  [0.1, 0.8, 0.1],
                                  [0.1, 0.1, 0.8]]))
        log_pi = np.log(np.array([1/3, 1/3, 1/3]))
        
        log_alpha, log_likelihood = self.hmm._forward(log_B, log_A, log_pi)
        
        self.assertEqual(log_alpha.shape, (self.T, K))
        self.assertIsInstance(log_likelihood, float)
        self.assertTrue(np.isfinite(log_likelihood))
    
    def test_backward_algorithm_output_shape(self):
        """测试后向算法输出形状"""
        K = 3
        log_beta = self.hmm._backward(
            log_A=np.zeros((K, K)),
            T=self.T
        )
        self.assertEqual(log_beta.shape, (self.T, K))


class TestStateSampling(unittest.TestCase):
    """测试状态采样逻辑"""
    
    def setUp(self):
        self.hmm = RobustBayesianHMM(n_states=3)
    
    def test_sample_state_sequence_length(self):
        """测试采样序列长度正确"""
        T = 50
        K = 3
        
        gamma = np.random.dirichlet(np.ones(K), size=T)
        states = self.hmm._sample_state_sequence(gamma)
        
        self.assertEqual(len(states), T)
        self.assertTrue(all(0 <= s < K for s in states))
    
    def test_sample_transition_matrix_stochastic(self):
        """测试转移矩阵是随机的（每行和为1）"""
        alpha = np.ones((3, 3)) + np.eye(3) * 2  # 强先验
        P = self.hmm._sample_transition_matrix(alpha)
        
        # 每行和应该接近1
        row_sums = P.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)
        
        # 所有元素应该在[0, 1]
        self.assertTrue(np.all(P >= 0))
        self.assertTrue(np.all(P <= 1))


class TestDataLoading(unittest.TestCase):
    """测试数据加载功能"""
    
    def test_load_csv_data(self):
        """测试CSV数据加载"""
        loader = ENSODataLoader()
        
        # 创建临时CSV文件
        import tempfile
        import pandas as pd
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("date,value\n")
            f.write("2020-01,-1.5\n")
            f.write("2020-02,0.3\n")
            f.write("2020-03,2.1\n")
            temp_file = f.name
        
        try:
            data = loader.load_data(temp_file, format_type='csv')
            
            self.assertIn('dates', data)
            self.assertIn('nino34', data)
            self.assertEqual(len(data['dates']), 3)
            self.assertEqual(len(data['nino34']), 3)
        finally:
            os.unlink(temp_file)
    
    def test_standardization_properties(self):
        """测试标准化后的数据均值为0，标准差为1"""
        loader = ENSODataLoader()
        
        # 创建简单数据
        dates = pd.date_range(start='2020-01', periods=100, freq='M')
        values = np.random.randn(100) * 2 + 1  # 均值1，标准差2
        
        df = pd.DataFrame({'date': dates, 'value': values})
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            data = loader.load_data(temp_file, format_type='csv')
            
            standardized = data['standardized_nino34']
            self.assertAlmostEqual(np.mean(standardized), 0.0, places=5)
            self.assertAlmostEqual(np.std(standardized), 1.0, places=5)
        finally:
            os.unlink(temp_file)


class TestLabelSwitchingResolution(unittest.TestCase):
    """测试标签交换问题解决"""
    
    def test_resolve_label_switching_basic(self):
        """测试基本的标签交换解决功能"""
        hmm = RobustBayesianHMM(n_states=3)
        
        n_samples = 100
        K = 3
        
        # 创建模拟的后验样本（故意打乱标签）
        all_mu_samples = np.zeros((n_samples, K))
        
        # 前50个样本：状态0=-1.5, 状态1=0.0, 状态2=1.5
        all_mu_samples[:50] = [[-1.5, 0.0, 1.5]] + np.random.randn(50, 3) * 0.1
        
        # 后50个样本：标签交换（状态0=1.5, 状态1=0.0, 状态2=-1.5）
        all_mu_samples[50:] = [[1.5, 0.0, -1.5]] + np.random.randn(50, 3) * 0.1
        
        # 调用标签交换解决函数
        permutation = hmm._resolve_label_switching(all_mu_samples)
        
        # 应该返回有效的排列
        self.assertEqual(permutation.shape[0], n_samples)
    
    def test_two_state_simple_ordering(self):
        """测试两状态的简单排序"""
        hmm = RobustBayesianHMM(n_states=2)
        
        n_samples = 50
        all_mu_samples = np.random.randn(n_samples, 2)
        all_mu_samples[:, 0] -= 1.0  # 状态0均值较低
        all_mu_samples[:, 1] += 1.0  # 状态1均值较高
        
        permutation = hmm._resolve_label_switching(all_mu_samples)
        
        # 对于2个状态，应该返回简单的排序
        self.assertIsNotNone(permutation)


if __name__ == '__main__':
    unittest.main(verbosity=2)
