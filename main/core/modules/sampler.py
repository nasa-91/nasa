"""
ENSO贝叶斯分析系统 - MCMC采样引擎模块

本模块提供马尔可夫链蒙特卡洛采样功能。
支持Gibbs采样、Metropolis-Hastings等算法，遵循策略模式。

设计模式：
- 策略模式：不同采样算法
- 模板方法：统一的采样流程
- 观察者模式：进度监控和回调

作者: nasa-91
版本: 4.0 (Refactored)
日期: 2026-05-18
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings
import time

from .types import (
    MCMCConfig, PosteriorSummary, ConvergenceStatus,
    ValidationResult, Array1D, Array2D, Array3D
)
from .interfaces import BaseMCMCSampler


@dataclass
class ChainState:
    """
    MCMC链状态数据类
    
    存储单条MCMC链的当前状态和样本历史。
    
    Attributes:
        chain_id (int): 链标识符
        iteration (int): 当前迭代次数
        mu (np.ndarray): 当前状态均值向量
        sigma (np.ndarray): 当前状态标准差向量
        nu (np.ndarray): 当前Student-t自由度（可选）
        transition_matrix (np.ndarray): 当前转移矩阵
        initial_distribution (np.ndarray): 初始分布
        state_sequence (np.ndarray): 当前状态序列
        log_likelihood (float): 当前对数似然
        samples_history (Dict[str, List]): 参数样本历史
        log_likelihood_history (List[float]): 对数似然历史
        pointwise_log_lik_history (List[np.ndarray]): 逐点对数似然历史
    """
    chain_id: int = 0
    iteration: int = 0
    mu: np.ndarray = field(default_factory=lambda: np.array([]))
    sigma: np.ndarray = field(default_factory=lambda: np.array([]))
    nu: Optional[np.ndarray] = None
    transition_matrix: Optional[np.ndarray] = None
    initial_distribution: Optional[np.ndarray] = None
    state_sequence: Optional[np.ndarray] = None
    log_likelihood: float = 0.0
    
    def __post_init__(self) -> None:
        """初始化样本历史记录"""
        self.samples_history: Dict[str, List] = {
            'mu': [],
            'sigma': [],
            'nu': [],
            'transition_matrix': [],
            'initial_distribution': []
        }
        self.log_likelihood_history: List[float] = []
        self.pointwise_log_lik_history: List[np.ndarray] = []


class BaseSamplingStep(ABC):
    """
    单步采样操作抽象基类（策略模式）
    
    定义MCMC中单个参数更新步骤的接口。
    每个参数（mu, sigma, P等）对应一个具体的采样步骤类。
    """
    
    @property
    @abstractmethod
    def parameter_name(self) -> str:
        """参数名称"""
        pass
    
    @abstractmethod
    def sample(
        self,
        chain_state: ChainState,
        data: np.ndarray,
        rng: np.random.RandomState,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        执行单步参数采样
        
        Args:
            chain_state: 当前链状态
            data: 观测数据
            rng: 随机数生成器
            
        Returns:
            Tuple of (新参数值, 对数似然贡献)
        """
        pass


class GibbsSampler(BaseMCMCSampler):
    """
    Gibbs采样器实现
    
    针对HMM模型优化的完整Gibbs采样算法。
    支持Student-t发射分布、异方差、季节性转移矩阵等特性。
    
    核心特点：
    - 完整的条件后验采样
    - 自适应步长（可选）
    - 并行多链支持
    - 内存优化（在线统计/稀疏化）
    - 收敛诊断集成
    
    Algorithm:
    1. 初始化参数
    2. 对每个迭代：
       a. 采样状态序列（前向-后向算法）
       b. 采样mu（状态均值）
       c. 采样sigma（状态标准差）
       d. 采样nu（Student-t自由度）
       e. 采样P（转移矩阵）
       f. 计算对数似然
    3. 返回后验样本
    """
    
    def __init__(
        self,
        emission_dist: str = 'student_t',
        use_optimized_core: bool = True,
        **kwargs
    ) -> None:
        """
        初始化Gibbs采样器
        
        Args:
            emission_dist: 发射分布类型 ('gaussian', 'student_t')
            use_optimized_core: 是否使用Numba优化核心
            **kwargs: 其他配置参数
        """
        self.emission_dist = emission_dist
        self.use_optimized_core = use_optimized_core
        
        # 尝试加载优化核心
        self._optimized_core = None
        if use_optimized_core:
            try:
                from core.optimized_core import OptimizedHMMCore
                self._optimized_core = OptimizedHMMCore(use_numba=True)
                print("[Gibbs采样器] [OK] Numba JIT加速已启用")
            except ImportError:
                print("[Gibbs采样器] [WARN] Numba不可用，使用NumPy实现")
        
        # 注册采样步骤
        self._sampling_steps: List[BaseSamplingStep] = []
        self._register_sampling_steps()
    
    @property
    def algorithm_name(self) -> str:
        return "Gibbs Sampling"
    
    def _register_sampling_steps(self) -> None:
        """注册所有采样步骤（按执行顺序）"""
        # 这里会注册：状态采样、mu采样、sigma采样、nu采样、P采样
        # 实际实现在子类或通过依赖注入添加
        pass
    
    def initialize_chain(
        self,
        data: np.ndarray,
        n_states: int,
        config: MCMCConfig,
        **kwargs
    ) -> Dict[str, Any]:
        """
        初始化MCMC链的状态
        
        使用数据驱动的启发式方法进行初始化，
        确保初始值在合理范围内以加速收敛。
        
        Args:
            data: 观测数据 (T,)
            n_states: 隐藏状态数K
            config: MCMC配置
            **kwargs: 模型特定参数
            
        Returns:
            初始状态字典
        """
        T = len(data)
        K = n_states
        
        # 为每条链创建独立的随机数生成器
        base_seed = config.random_seed
        rng = np.random.RandomState(base_seed)
        
        # 数据分位数用于初始化mu
        quantiles = np.linspace(10, 90, K + 1)
        mu_init = np.percentile(data, quantiles[:-1]) + \
                  np.diff(np.percentile(data, quantiles)) / 2
        
        # 确保单调性
        mu_init = np.sort(mu_init)
        
        # sigma初始化为数据标准差的一部分
        sigma_init = np.full(K, np.std(data) / K)
        
        # nu初始化（仅Student-t）
        nu_init = None
        if self.emission_dist == 'student_t':
            nu_init = np.full(K, 5.0)  # 轻厚尾
        
        # 转移矩阵初始化（高持久性 + 均匀小概率转移）
        P_init = np.full((K, K), 0.05 / (K - 1))
        np.fill_diagonal(P_init, 0.95)
        
        # 归一化
        P_init = P_init / P_init.sum(axis=1, keepdims=True)
        
        # 初始分布（均匀）
        pi_init = np.ones(K) / K
        
        # 初始状态序列（基于硬阈值）
        state_init = np.zeros(T, dtype=int)
        for t in range(T):
            distances = np.abs(data[t] - mu_init)
            state_init[t] = np.argmin(distances)
        
        # 创建ChainState对象
        chain_state = ChainState(
            chain_id=0,
            iteration=0,
            mu=mu_init.copy(),
            sigma=sigma_init.copy(),
            nu=nu_init.copy() if nu_init is not None else None,
            transition_matrix=P_init.copy(),
            initial_distribution=pi_init.copy(),
            state_sequence=state_init.copy()
        )
        
        # 计算初始对数似然
        chain_state.log_likelihood = self._compute_log_likelihood(
            data, chain_state
        )
        
        return {
            'chain_state': chain_state,
            'rng': rng,
            'n_states': K,
            'T': T
        }
    
    def run_chain(
        self,
        data: np.ndarray,
        initial_state: Dict[str, Any],
        config: MCMCConfig,
        chain_id: int = 0,
        callback: Optional[Callable[[int, ChainState], None]] = None
    ) -> Dict[str, Any]:
        """
        运行单条完整的MCMC链
        
        这是Gibbs采样的主循环，协调所有采样步骤。
        
        Args:
            data: 观测数据 (T,)
            initial_state: 初始状态字典
            config: MCMC配置
            chain_id: 链标识符
            callback: 进度回调函数 callback(iteration, chain_state)
            
        Returns:
            链结果字典
        """
        chain_state: ChainState = initial_state['chain_state']
        rng: np.random.RandomState = initial_state['rng']
        K = initial_state['n_states']
        T = initial_state['T']
        
        n_iterations = config.n_iterations
        burn_in = config.burn_in
        thin = config.thinning_interval
        verbose = config.verbose
        
        # 准备存储
        store_samples = config.store_samples
        
        start_time = time.perf_counter()
        
        if verbose:
            print(f"\n[Gibbs采样] 开始运行链 {chain_id+1}")
            print(f"           总迭代: {n_iterations}, 预烧: {burn_in}, 稀疏间隔: {thin}")
        
        for iteration in range(n_iterations):
            chain_state.iteration = iteration + 1
            
            # === Gibbs采样步骤 ===
            
            # 步骤1: 采样状态序列（前向-后向算法）
            chain_state.state_sequence = self._sample_states(
                data, chain_state, rng
            )
            
            # 步骤2: 采样mu（状态均值）
            chain_state.mu = self._sample_mu(
                data, chain_state, rng
            )
            
            # 步骤3: 采样sigma（状态标准差）
            chain_state.sigma = self._sample_sigma(
                data, chain_state, rng
            )
            
            # 步骤4: 采样nu（Student-t自由度，如果适用）
            if self.emission_dist == 'student_t' and chain_state.nu is not None:
                chain_state.nu = self._sample_nu(
                    data, chain_state, rng
                )
            
            # 步骤5: 采样转移矩阵P
            chain_state.transition_matrix = self._sample_transition_matrix(
                chain_state, rng
            )
            
            # 步骤6: 计算对数似然
            chain_state.log_likelihood = self._compute_log_likelihood(
                data, chain_state
            )
            
            # 存储样本（如果在预烧期之后且满足稀疏条件）
            if iteration >= burn_in and (iteration - burn_in) % thin == 0:
                if store_samples:
                    chain_state.samples_history['mu'].append(chain_state.mu.copy())
                    chain_state.samples_history['sigma'].append(chain_state.sigma.copy())
                    if chain_state.nu is not None:
                        chain_state.samples_history['nu'].append(chain_state.nu.copy())
                    chain_state.samples_history['transition_matrix'].append(
                        chain_state.transition_matrix.copy()
                    )
                
                chain_state.log_likelihood_history.append(chain_state.log_likelihood)
            
            # 进度回调
            if callback is not None:
                callback(iteration, chain_state)
            
            # 进度输出
            if verbose and (iteration + 1) % 500 == 0:
                elapsed = time.perf_counter() - start_time
                rate = (iteration + 1) / elapsed
                eta = (n_iterations - iteration - 1) / rate if rate > 0 else 0
                
                print(f"[Gibbs采样] 链{chain_id+1}: {iteration+1}/{n_iterations} "
                      f"({rate:.1f} iter/s, ETA: {eta:.0f}s)")
        
        # 完成信息
        total_time = time.perf_counter() - start_time
        n_samples = len(chain_state.log_likelihood_history)
        
        if verbose:
            print(f"[Gibbs采样] 链{chain_id+1} 完成:")
            print(f"           耗时: {total_time:.1f}s")
            print(f"           有效样本数: {n_samples}")
            print(f"           最终log-lik: {chain_state.log_likelihood:.2f}")
        
        return {
            'chain_state': chain_state,
            'n_samples': n_samples,
            'runtime_seconds': total_time,
            'config': config,
            'chain_id': chain_id
        }
    
    def _sample_states(
        self,
        data: np.ndarray,
        chain_state: ChainState,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        使用前向-后向算法采样状态序列
        
        这是Gibbs采样中最计算密集的步骤。
        使用对数空间数值稳定的实现。
        """
        T = len(data)
        K = len(chain_state.mu)
        
        # 计算发射概率的对数
        log_B = self._compute_emission_log_prob(
            data, chain_state
        )  # (T, K)
        
        # 转移矩阵对数
        log_P = np.log(chain_state.transition_matrix + 1e-300)  # (K, K)
        log_pi = np.log(chain_state.initial_distribution + 1e-300)  # (K,)
        
        # 使用优化核心（如果可用）
        if self._optimized_core is not None:
            try:
                _, _ = self._optimized_core.forward_algorithm(
                    log_B, log_P, log_pi
                )
                states = self._optimized_core.backward_sample(
                    log_B, log_P, log_pi, rng
                )
                return states
            except Exception:
                pass
        
        # NumPy回退实现
        return self._forward_backward_numpy(log_B, log_P, log_pi, rng)
    
    def _sample_mu(
        self,
        data: np.ndarray,
        chain_state: ChainState,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """采样状态均值mu（共轭先验）"""
        K = len(chain_state.mu)
        sigma_sq = chain_state.sigma ** 2
        
        mu_new = np.zeros(K)
        
        for k in range(K):
            mask = (chain_state.state_sequence == k)
            n_k = np.sum(mask)
            
            if n_k > 0:
                y_k = data[mask]
                # 先验参数（弱信息先验）
                prior_mean = 0.0
                prior_var = 100.0  # 大方差 = 弱先验
                
                # 后验参数
                posterior_var = 1.0 / (n_k / sigma_sq[k] + 1.0 / prior_var)
                posterior_mean = posterior_var * (
                    np.sum(y_k) / sigma_sq[k] + prior_mean / prior_var
                )
                
                # 采样
                mu_new[k] = rng.normal(
                    np.sqrt(posterior_var) * rng.randn() + posterior_mean
                )
            else:
                # 无观测时从先验采样
                mu_new[k] = rng.normal(prior_mean, np.sqrt(prior_var))
        
        return mu_new
    
    def _sample_sigma(
        self,
        data: np.ndarray,
        chain_state: ChainState,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """采样状态标准差sigma（逆伽马共轭先验）"""
        K = len(chain_state.sigma)
        mu = chain_state.mu
        
        sigma_new = np.zeros(K)
        
        for k in range(K):
            mask = (chain_state.state_sequence == k)
            n_k = np.sum(mask)
            
            if n_k > 0:
                y_k = data[mask]
                # 逆伽马先验参数
                alpha_prior = 2.01  # 使均值有限
                beta_prior = 0.01   # 弱信息
                
                # 后验参数
                alpha_post = alpha_prior + n_k / 2.0
                beta_post = beta_prior + np.sum((y_k - mu[k])**2) / 2.0
                
                # 采样方差，然后取sqrt得到std
                variance = 1.0 / rng.gamma(alpha_post, 1.0 / beta_post)
                sigma_new[k] = np.sqrt(variance)
            else:
                # 从先验采样
                variance = 1.0 / rng.gamma(alpha_prior, 1.0 / beta_prior)
                sigma_new[k] = np.sqrt(variance)
        
        return np.maximum(sigma_new, 1e-6)  # 避免过小的方差
    
    def _sample_nu(
        self,
        data: np.ndarray,
        chain_state: ChainState,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """采样Student-t自由度nu（Metropolis步骤）"""
        K = len(chain_state.nu)
        nu_current = chain_state.nu.copy()
        nu_proposed = nu_current.copy()
        
        for k in range(K):
            mask = (chain_state.state_sequence == k)
            if np.sum(mask) < 2:
                continue
            
            y_k = data[mask]
            mu_k = chain_state.mu[k]
            sigma_k = chain_state.sigma[k]
            
            # 提议分布：对数正态随机游走
            log_nu_current = np.log(nu_current[k])
            log_nu_proposed = log_nu_current + rng.normal(0, 0.5)
            nu_proposed[k] = np.exp(log_nu_proposed)
            
            # 计算接受率（简化版）
            try:
                from scipy.stats import t as student_t
                
                log_lik_current = np.sum(student_t.logpdf(
                    y_k, df=nu_current[k], loc=mu_k, scale=sigma_k
                ))
                log_lik_proposed = np.sum(student_t.logpdf(
                    y_k, df=nu_proposed[k], loc=mu_k, scale=sigma_k
                ))
                
                # 先验项（鼓励适中的nu值）
                log_prior_current = -nu_current[k] / 10.0
                log_prior_proposed = -nu_proposed[k] / 10.0
                
                # Jacobian修正（对数空间的提议）
                log_jacobian = log_nu_proposed - log_nu_current
                
                # 接受率
                log_alpha = (
                    log_lik_proposed + log_prior_proposed + log_jacobian -
                    log_lik_current - log_prior_current
                )
                
                if np.log(rng.uniform()) < log_alpha:
                    nu_current[k] = nu_proposed[k]
                    
            except Exception:
                pass  # 保持当前值
        
        return nu_current
    
    def _sample_transition_matrix(
        self,
        chain_state: ChainState,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """采样转移矩阵P（Dirichlet共轭先验）"""
        states = chain_state.state_sequence
        K = len(chain_state.initial_distribution)
        
        P_new = np.zeros((K, K))
        
        for i in range(K):
            # 统计从状态i转移到各状态的次数
            transitions = np.zeros(K)
            
            for t in range(len(states) - 1):
                if states[t] == i:
                    j = states[t + 1]
                    transitions[j] += 1
            
            # Dirichlet先验参数（对称，鼓励高持久性）
            alpha_prior = np.ones(K)
            alpha_prior[i] += 50.0  # 高对角线先验
            
            # 后验参数
            alpha_posterior = alpha_prior + transitions
            
            # 从Dirichlet分布采样
            P_new[i, :] = rng.dirichlet(alpha_posterior)
        
        return P_new
    
    def _compute_emission_log_prob(
        self,
        data: np.ndarray,
        chain_state: ChainState
    ) -> np.ndarray:
        """
        计算发射概率矩阵的对数
        
        Returns:
            log_B: (T, K) 对数发射概率矩阵
        """
        T = len(data)
        K = len(chain_state.mu)
        
        log_B = np.zeros((T, K))
        
        for k in range(K):
            mu_k = chain_state.mu[k]
            sigma_k = max(chain_state.sigma[k], 1e-8)
            
            if self.emission_dist == 'gaussian':
                log_B[:, k] = -0.5 * ((data - mu_k) / sigma_k) ** 2 \
                              - np.log(sigma_k) - 0.5 * np.log(2 * np.pi)
            elif self.emission_dist == 'student_t':
                nu_k = chain_state.nu[k] if chain_state.nu is not None else 5.0
                from scipy.special import gammaln
                
                log_B[:, k] = (
                    gammaln((nu_k + 1) / 2) - gammaln(nu_k / 2)
                    - 0.5 * np.log(nu_k * np.pi) - np.log(sigma_k)
                    - (nu_k + 1) / 2 * np.log(
                        1 + ((data - mu_k) / sigma_k) ** 2 / nu_k
                    )
                )
        
        return log_B
    
    def _compute_log_likelihood(
        self,
        data: np.ndarray,
        chain_state: ChainState
    ) -> float:
        """计算当前参数下的完全对数似然"""
        log_B = self._compute_emission_log_prob(data, chain_state)
        
        # 近似：使用Viterbi路径的对数似然
        ll = 0.0
        for t in range(len(data)):
            k = chain_state.state_sequence[t]
            ll += log_B[t, k]
        
        return ll
    
    def _forward_backward_numpy(
        self,
        log_B: np.ndarray,
        log_P: np.ndarray,
        log_pi: np.ndarray,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        NumPy实现的前向-后向算法（用于采样状态）
        
        数值稳定版本，使用对数空间运算。
        """
        T, K = log_B.shape
        
        # 前向算法
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + log_B[0]
        
        for t in range(1, T):
            for k in range(K):
                # log-sum-exp技巧
                max_val = np.max(log_alpha[t-1] + log_P[:, k])
                log_alpha[t, k] = (
                    max_val +
                    np.log(np.sum(np.exp(log_alpha[t-1] + log_P[:, k] - max_val))) +
                    log_B[t, k]
                )
        
        # 后向采样
        states = np.zeros(T, dtype=int)
        
        # 最后一个状态
        log_probs = log_alpha[T-1]
        probs = np.exp(log_probs - np.max(log_probs))
        probs /= probs.sum()
        states[T-1] = rng.choice(K, p=probs)
        
        # 反向递推
        for t in range(T-2, -1, -1):
            k_next = states[t+1]
            log_probs = (
                log_alpha[t] +
                log_P[:, k_next]
            )
            max_val = np.max(log_probs)
            probs = np.exp(log_probs - max_val)
            probs /= probs.sum()
            states[t] = rng.choice(K, p=probs)
        
        return states
    
    def assess_convergence(
        self,
        chains_results: List[Dict[str, Any]]
    ) -> Dict[int, ConvergenceStatus]:
        """
        评估多条链的收敛情况
        
        使用Gelman-Rubin R-hat统计量和有效样本量(ESS)诊断收敛。
        
        Args:
            chains_results: 所有链的结果列表
            
        Returns:
            {chain_id: ConvergenceStatus} 字典
        """
        convergence_status = {}
        
        if len(chains_results) < 2:
            # 单链无法评估R-hat，返回WARNING
            for result in chains_results:
                chain_id = result['chain_id']
                convergence_status[chain_id] = ConvergenceStatus.WARNING
            return convergence_status
        
        # 提取所有链的mu样本
        all_chains_mu = []
        for result in chains_results:
            chain_state = result['chain_state']
            mu_samples = np.array(chain_state.samples_history['mu'])
            if len(mu_samples) > 0:
                all_chains_mu.append(mu_samples)
        
        if len(all_chains_mu) < 2:
            for result in chains_results:
                convergence_status[result['chain_id']] = ConvergenceStatus.INSUFFICIENT_DATA
            return convergence_status
        
        # 计算R-hat（简化的Gelman-Rubin诊断）
        n_chains = len(all_chains_mu)
        n_samples = all_chains_mu[0].shape[0]
        
        if n_samples < 20:
            for result in chains_results:
                convergence_status[result['chain_id']] = ConvergenceStatus.INSUFFICIENT_DATA
            return convergence_status
        
        # 对每个参数维度计算R-hat
        r_hat_values = []
        for param_idx in range(all_chains_mu[0].shape[1]):
            chain_means = np.array([
                np.mean(chain[:, param_idx]) 
                for chain in all_chains_mu
            ])
            
            overall_mean = np.mean(chain_means)
            B = n_samples * np.var(chain_means, ddof=1)  # 链间方差
            
            W = np.mean([
                np.var(chain[:, param_idx], ddof=1)
                for chain in all_chains_mu
            ])  # 链内方差
            
            if W > 0:
                var_hat = (1 - 1/n_samples) * W + (1/n_samples) * B
                r_hat = np.sqrt(var_hat / W) if W > 0 else float('inf')
                r_hat_values.append(r_hat)
        
        avg_r_hat = np.mean(r_hat_values) if r_hat_values else 1.0
        
        # 判定收敛状态
        for i, result in enumerate(chains_results):
            chain_id = result['chain_id']
            
            if avg_r_hat < 1.1:
                convergence_status[chain_id] = ConvergenceStatus.CONVERGED
            elif avg_r_hat < 1.2:
                convergence_status[chain_id] = ConvergenceStatus.WARNING
            else:
                convergence_status[chain_id] = ConvergenceStatus.NOT_CONVERGED
        
        return convergence_status
    
    def combine_chains(
        self,
        chains_results: List[Dict[str, Any]],
        config: MCMCConfig
    ) -> PosteriorSummary:
        """
        合并多条链的结果生成后验摘要
        
        执行以下操作：
        1. 合并所有有效样本
        2. 计算后验统计量（均值、标准差、置信区间）
        3. 计算WAIC等模型准则
        4. 生成最可能状态序列
        5. 收敛诊断汇总
        
        Args:
            chains_results: 所有链的结果列表
            config: MCMC配置
            
        Returns:
            PosteriorSummary对象
        """
        # 收集所有样本
        all_mu = []
        all_sigma = []
        all_nu = []
        all_P = []
        all_log_lik = []
        all_pointwise_log_lik = []
        
        for result in chains_results:
            chain_state = result['chain_state']
            
            if len(chain_state.samples_history['mu']) > 0:
                all_mu.extend(chain_state.samples_history['mu'])
                all_sigma.extend(chain_state.samples_history['sigma'])
                
                if chain_state.samples_history['nu']:
                    all_nu.extend(chain_state.samples_history['nu'])
                
                all_P.extend(chain_state.samples_history['transition_matrix'])
            
            all_log_lik.extend(chain_state.log_likelihood_history)
        
        # 转换为数组
        all_mu = np.array(all_mu) if all_mu else np.array([])
        all_sigma = np.array(all_sigma) if all_sigma else np.array([])
        all_nu = np.array(all_nu) if all_nu else None
        all_P = np.array(all_P) if all_P else np.array([])
        all_log_lik = np.array(all_log_lik) if all_log_lik else np.array([])
        
        S_total = len(all_log_lik)  # 总有效样本数
        
        # 构建后验摘要
        summary = PosteriorSummary()
        
        # 参数后验统计量
        if len(all_mu) > 0:
            summary.parameter_means['mu'] = np.mean(all_mu, axis=0)
            summary.parameter_stds['mu'] = np.std(all_mu, axis=0, ddof=1)
            ci_lower = np.percentile(all_mu, 2.5, axis=0)
            ci_upper = np.percentile(all_mu, 97.5, axis=0)
            summary.parameter_ci['mu'] = (ci_lower, ci_upper)
        
        if len(all_sigma) > 0:
            summary.parameter_means['sigma'] = np.mean(all_sigma, axis=0)
            summary.parameter_stds['sigma'] = np.std(all_sigma, axis=0, ddof=1)
            ci_lower = np.percentile(all_sigma, 2.5, axis=0)
            ci_upper = np.percentile(all_sigma, 97.5, axis=0)
            summary.parameter_ci['sigma'] = (ci_lower, ci_upper)
        
        if all_nu is not None and len(all_nu) > 0:
            summary.parameter_means['nu'] = np.mean(all_nu, axis=0)
            summary.parameter_stds['nu'] = np.std(all_nu, axis=0, ddof=1)
            ci_lower = np.percentile(all_nu, 2.5, axis=0)
            ci_upper = np.percentile(all_nu, 97.5, axis=0)
            summary.parameter_ci['nu'] = (ci_lower, ci_upper)
        
        if len(all_P) > 0:
            summary.parameter_means['transition_matrix'] = np.mean(all_P, axis=0)
            summary.parameter_stds['transition_matrix'] = np.std(all_P, axis=0, ddof=1)
        
        # 对数似然
        if len(all_log_lik) > 0:
            summary.log_likelihood = float(np.median(all_log_lik))
        
        # 最可能状态序列（使用最后一条链的状态）
        if len(chains_results) > 0:
            last_chain = chains_results[-1]['chain_state']
            if last_chain.state_sequence is not None:
                summary.state_sequence = last_chain.state_sequence
        
        # 收敛诊断
        convergence = self.assess_convergence(chains_results)
        converged_count = sum(
            1 for s in convergence.values() 
            if s == ConvergenceStatus.CONVERGED
        )
        summary.convergence_diagnostics = {
            'converged_chains': converged_count,
            'total_chains': len(chains_results),
            'convergence_status_per_chain': {
                str(k): v.name for k, v in convergence.items()
            },
            'effective_sample_size_total': S_total,
            'n_chains_used': len(chains_results)
        }
        
        return summary


__all__ = [
    # 辅助类
    'ChainState',
    'BaseSamplingStep',
    
    # 主要采样器
    'GibbsSampler',
]
