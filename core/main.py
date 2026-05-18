import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D
from scipy import stats
from scipy.special import logsumexp, gammaln, betaln
from scipy.optimize import minimize
from scipy.signal import find_peaks
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')
np.random.seed(42)


class ENSODataLoader:
    def __init__(self):
        self.supported_formats = ['csv', 'noaa_ascii', 'noaa_nc', 'custom']
        self.data_info = {}

    def load_data(self, filepath, format_type='auto', **kwargs):
        if format_type == 'auto':
            format_type = self._detect_format(filepath)
        
        print(f"正在加载数据: {filepath}")
        print(f"  检测到格式: {format_type}")
        
        if format_type == 'csv':
            data = self._load_csv(filepath, **kwargs)
        elif format_type == 'noaa_ascii':
            data = self._load_noaa_ascii(filepath, **kwargs)
        elif format_type == 'nc':
            data = self._load_netcdf(filepath, **kwargs)
        else:
            raise ValueError(f"不支持的格式: {format_type}")
        
        self.data_info.update({
            'source': filepath,
            'format': format_type,
            'n_observations': len(data['dates']),
            'date_range': f"{data['dates'].iloc[0]} 至 {data['dates'].iloc[-1]}",
            'missing_values': np.sum(np.isnan(data['nino34']))
        })
        
        return self._preprocess(data)

    def _detect_format(self, filepath):
        if filepath.endswith('.csv'):
            return 'csv'
        elif filepath.endswith('.txt') or filepath.endswith('.asc'):
            return 'noaa_ascii'
        elif filepath.endswith('.nc') or filepath.endswith('.nc4'):
            return 'nc'
        else:
            return 'csv'

    def _load_csv(self, filepath, date_col=0, value_col=1,
                  date_format=None, header=0):
        df = pd.read_csv(filepath, header=header)
        date_strings = df.iloc[:, date_col].astype(str)

        if date_format:
            dates = pd.to_datetime(date_strings, format=date_format)
        else:
            try:
                dates = pd.to_datetime(date_strings, format='%Y-%m-%d')
            except ValueError:
                try:
                    dates = pd.to_datetime(date_strings, format='%Y-%m')
                except ValueError:
                    dates = pd.to_datetime(date_strings, format='mixed')

        nino34 = df.iloc[:, value_col].values.astype(float)
        return {'dates': dates, 'nino34': nino34}

    def _load_noaa_ascii(self, filepath, skiprows=0):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            data_lines = []
            for line in lines[skiprows:]:
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        year = int(parts[0])
                        month = int(parts[1])
                        value = float(parts[2])
                        date_str = f"{year}-{month:02d}-01"
                        data_lines.append([date_str, value])
                    except ValueError:
                        continue
            
            df = pd.DataFrame(data_lines, columns=['date', 'nino34'])
            dates = pd.to_datetime(df['date'])
            nino34 = df['nino34'].values.astype(float)
            
            print(f"  从NOAA ASCII格式加载了 {len(dates)} 条记录")
            return {'dates': dates, 'nino34': nino34}
        except Exception as e:
            print(f"  NOAA ASCII读取失败: {e}")
            return self._load_generic(filepath)

    def _load_netcdf(self, filepath, var_name='sst'):
        try:
            import netCDF4 as nc  # type: ignore
        except ImportError:
            print("  [警告] netCDF4库未安装，无法读取NetCDF文件")
            print("  请运行: pip install netcdf4")
            return self._load_generic(filepath)
        
        try:
            ds = nc.Dataset(filepath)
            
            time_var = [v for v in ds.variables.keys() 
                       if 'time' in v.lower()][0]
            times = nc.num2date(ds.variables[time_var][:], 
                               ds.variables[time_var].units)
            
            sst = ds.variables[var_name][:]
            
            if len(sst.shape) == 3:
                nino_region = sst[:, 20:35, 100:130]
                nino34 = np.nanmean(nino_region, axis=(1, 2))
            else:
                nino34 = sst
            
            ds.close()
            
            print(f"  加载NetCDF数据: {len(times)} 个时间步长")
            return {'dates': times, 'nino34': nino34}
        except Exception as e:
            print(f"  NetCDF读取失败: {e}")
            return self._load_generic(filepath)

    def _load_generic(self, filepath):
        print("  使用通用CSV加载器...")
        df = pd.read_csv(filepath)
        
        date_col = None
        val_col = None
        
        for col in df.columns:
            if date_col is None and ('date' in col.lower() or 'year' in col.lower() or 'time' in col.lower()):
                date_col = col
            if val_col is None and ('nino' in col.lower() or 'sst' in col.lower() or 'temp' in col.lower() or 'anom' in col.lower()):
                val_col = col
        
        if date_col is None:
            date_col = df.columns[0]
        if val_col is None:
            val_col = df.columns[-1] if len(df.columns) > 1 else df.columns[0]
        
        dates = pd.to_datetime(df[date_col])
        nino34 = df[val_col].values.astype(float)
        
        return {'dates': dates, 'nino34': nino34}

    def _preprocess(self, raw_data):
        dates = raw_data['dates']
        nino34 = raw_data['nino34'].astype(float).copy()
        
        missing_mask = np.isnan(nino34) | np.isinf(nino34)
        if np.sum(missing_mask) > 0:
            print(f"  发现 {np.sum(missing_mask)} 个缺失值，使用线性插值填补")
            valid_indices = np.where(~missing_mask)[0]
            invalid_indices = np.where(missing_mask)[0]
            nino34[invalid_indices] = np.interp(invalid_indices, valid_indices, nino34[valid_indices])
        
        self.raw_mean = np.mean(nino34)
        self.raw_std = np.std(nino34)
        
        standardized = (nino34 - self.raw_mean) / (self.raw_std + 1e-10)
        
        years = np.array([d.year + (d.month - 1) / 12 for d in dates])
        
        print(f"\n  数据预处理完成:")
        print(f"    - 时间范围: {dates.iloc[0].strftime('%Y-%m')} 至 {dates.iloc[-1].strftime('%Y-%m')}")
        print(f"    - 观测数: {len(nino34)}")
        print(f"    - 均值: {self.raw_mean:.3f} C, 标准差: {self.raw_std:.3f} C")
        print(f"    - 范围: [{np.min(nino34):.2f}, {np.max(nino34):.2f}] C")
        
        processed_data = {
            'dates': dates,
            'raw_nino34': nino34,
            'standardized_nino34': standardized,
            'years': years,
            'T': len(nino34),
            'mean': self.raw_mean,
            'std': self.raw_std
        }
        
        self.y = standardized
        return processed_data


class RobustBayesianHMM:
    """
    鲁棒贝叶斯隐马尔可夫模型 - 专业版
    
    针对ENSO不对称性分析优化的HMM实现
    
    核心改进：
    1. 允许状态间方差异质性（异方差）
    2. Student-t分布发射概率（处理厚尾）
    3. 基于物理特性的强先验
    4. 大规模MCMC采样（5000+迭代）
    5. 规范化事件检测（气象标准）
    6. 完善的收敛诊断与后验预测检验
    """
    
    def __init__(self, n_states=3, random_seed=42, 
                 emission_dist='student_t',
                 use_time_covariates=False):
        self.n_states = n_states
        self.random_seed = random_seed
        self.emission_dist = emission_dist
        self.use_time_covariates = use_time_covariates
        # 动态生成状态标签（支持任意K值）- 关键修复：解决K>3时索引越界
        self.state_labels = self._generate_state_labels(n_states)
        self.chains = []
        self.posterior_summary = {}
        self.y = None
        self.prior_params = {}
    
    @staticmethod
    def _log_emission_pdf(y: np.ndarray, mu: float, sigma: float, 
                          nu: float = 5.0, dist: str = 'gaussian') -> np.ndarray:
        """
        统一的对数发射概率密度函数
        
        参数：
            y: 观测值数组 (T,)
            mu: 均值
            sigma: 标准差 (>0)
            nu: Student-t自由度 (仅dist='student_t'时使用)
            dist: 分布类型 ('gaussian' 或 'student_t')
        
        返回：
            对数概率密度数组 (T,)
        
        数值稳定性保证：
            - sigma自动限制在[1e-10, +inf)
            - nu自动限制在[2.1, +inf) (保证方差存在)
            - 返回值裁剪到[-700, 0]防止下溢
        """
        sigma_safe = max(sigma, 1e-10)
        
        if dist == 'gaussian':
            log_pdf = (
                -0.5 * ((y - mu) / sigma_safe)**2 
                - np.log(sigma_safe) 
                - 0.5 * np.log(2*np.pi)
            )
        elif dist == 'student_t':
            nu_safe = max(nu, 2.1)
            log_pdf = (
                gammaln((nu_safe + 1)/2) - gammaln(nu_safe/2) 
                - 0.5 * np.log(nu_safe * np.pi * sigma_safe**2)
                - (nu_safe + 1)/2 * np.log(1 + ((y - mu)**2) / 
                                           (nu_safe * sigma_safe**2))
            )
        else:
            raise ValueError(f"不支持的分布类型: {dist}，必须是'gaussian'或'student_t'")
        
        return np.clip(log_pdf, -700, 0)
    
    def _compute_log_emission_matrix(self, params: dict) -> np.ndarray:
        """
        计算完整的对数发射概率矩阵 (T x K)
        
        统一调用_log_emission_pdf，确保所有位置使用相同的发射密度计算
        
        参数：
            params: 包含'mu', 'sigma', 'nu'(可选)的参数字典
            
        返回：
            log_emission: 对数发射矩阵 (T, K)
        """
        T = len(self.y)
        K = self.n_states
        log_emission = np.zeros((T, K))
        
        for k in range(K):
            nu_k = params.get('nu', [5.0]*K)[k] if self.emission_dist == 'student_t' else 5.0
            log_emission[:, k] = self._log_emission_pdf(
                self.y, params['mu'][k], params['sigma'][k], 
                nu=nu_k, dist=self.emission_dist
            )
        
        return log_emission
    
    def _generate_state_labels(self, K):
        """动态生成状态标签，解决K>3时索引越界问题"""
        base_labels = ['拉尼娜', '中性', '厄尔尼诺']
        if K <= 3:
            return base_labels[:K]
        else:
            extra_labels = [f'状态{i}' for i in range(3, K)]
            return base_labels + extra_labels

    def fit(self, y: np.ndarray, n_iterations: int = 5000, burn_in: int = 2000, 
            n_chains: int = 4, thin: int = 5, 
            verbose: bool = True, nu_prior=None, time_covariates=None) -> dict:
        """
        拟合鲁棒贝叶斯HMM模型
        
        参数：
            y: 标准化的时间序列数据 (T,)
            n_iterations: 每条链的MCMC迭代次数
            burn_in: 预烧期（丢弃的迭代次数）
            n_chains: MCMC链数
            thin: 稀疏化间隔
            verbose: 是否打印进度信息
            nu_prior: Student-t自由度先验（可选）
            time_covariates: 时间协变量（可选）
            
        返回：
            posterior_summary: 后验摘要字典
        """
        self.y = y
        T = len(y)
        
        if verbose:
            print("\n" + "="*70)
            print("鲁棒贝叶斯隐马尔可夫模型 (Robust HMM-{0}状态)".format(self.n_states))
            print("="*70)
            print("\n模型配置:")
            print(f"  - 状态数: K={self.n_states}")
            print(f"  - 发射分布: {self.emission_dist}")
            print(f"  - 方差设定: 异方差（各状态独立估计）")
            print(f"\nMCMC配置:")
            print(f"  - 迭代次数: {n_iterations}/链")
            print(f"  - 预烧期: {burn_in}")
            print(f"  - MCMC链数: {n_chains}")
            print(f"  - 稀疏化间隔: {thin}")
            print(f"  - 有效样本量目标: {(n_iterations-burn_in)//thin * n_chains}")
        
        self._set_physics_informed_priors()
        
        chain_results = []
        for chain_id in range(n_chains):
            if verbose:
                print(f"\n  正在运行第 {chain_id + 1}/{n_chains} 条链...")
            result = self._run_mcmc_chain_robust(
                chain_id, n_iterations, 
                verbose=verbose,
                nu_prior=nu_prior,
                time_covariates=time_covariates
            )
            
            thinned_result = {}
            for key in result:
                thinned_result[key] = result[key][burn_in::thin]
            
            chain_results.append(thinned_result)
            
            if verbose and chain_id < n_chains - 1:
                print(f"    进度: {chain_id + 1}/{n_chains} 链完成")
        
        self.chains = chain_results
        self._compute_posterior_summary_robust()
        
        if verbose:
            self._print_comprehensive_diagnostic()
        
        return self.posterior_summary

    def _set_physics_informed_priors(self):
        """
        基于ENSO物理特性的强先验设定
        
        ENSO物理约束：
        - 拉尼娜：均值-0.5~-1.5°C，标准差0.3~0.6°C
        - 中性态：均值±0.3°C，标准差0.2~0.4°C  
        - 厄尔尼诺：均值+0.5~+2.5°C，标准差0.4~0.8°C（通常大于拉尼娜）
        """
        y = self.y
        T = len(y)
        K = self.n_states
        
        quantiles = np.percentile(y, np.linspace(15, 85, K))
        
        self.prior_params = {
            'mu_loc': quantiles,
            'mu_scale': 0.3,
            'sigma_alpha_shape': 2.0 + T/50,
            'sigma_beta_rate': 2.0,
            'nu_alpha': 5.0,
            'nu_beta': 1.0,
            'P_concentration': 10.0,
            'pi_concentration': 1.0
        }
        
        print(f"\n  [物理信息先验] 先验参数已基于ENSO特性设定:")
        print(f"    - 均值先验位置: {[f'{q:.2f}' for q in quantiles]}")
        print(f"    - 均值先验尺度: ±{self.prior_params['mu_scale']:.2f} (强约束)")
        print(f"    - 转移矩阵浓度: {self.prior_params['P_concentration']} (鼓励持续性)")

    def _initialize_parameters_robust(self, chain_id: int) -> dict:
        """
        增强的参数初始化策略
        
        参数：
            chain_id: 链ID（用于设置不同的随机种子）
            
        返回：
            params: 包含mu, sigma, P, pi(和nu)的初始参数字典
        """
        np.random.seed(self.random_seed + chain_id * 1000)
        K = self.n_states
        T = len(self.y)
        
        y_sorted = np.sort(self.y)
        percentiles = np.linspace(10, 90, K+1)
        mu_init = np.array([np.median(y_sorted[
            int(percentiles[i]/100*T):int(percentiles[i+1]/100*T)
        ]) for i in range(K)])
        
        mu_init += np.random.normal(0, 0.05, K)
        mu_init = np.sort(mu_init)
        
        sigma_init = np.abs(np.random.normal(0.5, 0.1, K)) + 0.2
        sigma_init = sigma_init * (1 + 0.3 * (mu_init - mu_init.mean()) / (mu_init.std() + 0.1))
        sigma_init = np.clip(sigma_init, 0.15, 1.5)
        
        P_init = np.zeros((K, K))
        for i in range(K):
            diag_val = 0.75 + np.random.uniform(0, 0.15)
            off_diag = (1 - diag_val) / (K - 1) if K > 1 else 0
            P_init[i] = np.full(K, off_diag)
            P_init[i, i] = diag_val
        
        pi_init = np.ones(K) / K
        
        params = {
            'mu': mu_init.copy(),
            'sigma': sigma_init.copy(),
            'P': P_init.copy(),
            'pi': pi_init.copy()
        }
        
        if self.emission_dist == 'student_t':
            params['nu'] = np.random.gamma(5, 1, size=K) + 3
        
        return params

    def _run_mcmc_chain_robust(self, chain_id: int, n_iterations: int, 
                               verbose: bool = False,
                               nu_prior=None, time_covariates=None) -> dict:
        """
        鲁棒的MCMC采样循环
        
        参数：
            chain_id: 链标识符
            n_iterations: 迭代次数
            verbose: 是否打印进度信息
            nu_prior: Student-t自由度先验（可选）
            time_covariates: 时间协变量（可选）
            
        返回：
            result: 包含所有参数后验样本的字典
        """
        T = len(self.y)
        K = self.n_states
        y = self.y
        
        params = self._initialize_parameters_robust(chain_id)
        
        mu_samples = np.zeros((n_iterations, K))
        sigma_samples = np.zeros((n_iterations, K))
        P_samples = np.zeros((n_iterations, K, K))
        pi_samples = np.zeros((n_iterations, K))
        state_samples = np.zeros((n_iterations, T), dtype=int)
        log_likelihoods = np.zeros(n_iterations)
        
        if self.emission_dist == 'student_t':
            nu_samples = np.zeros((n_iterations, K))
        
        states = np.random.choice(K, size=T, p=params['pi'])
        
        for iteration in range(n_iterations):
            log_emission = self._compute_log_emission_stable(params)
            log_alpha = self._forward_algorithm(log_emission, params)
            log_beta = self._backward_algorithm(log_emission, params)
            
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)
            
            gamma = np.clip(gamma, 1e-300, 1.0)
            gamma /= gamma.sum(axis=1, keepdims=True)
            
            cumsum = np.cumsum(gamma, axis=1)
            u = np.random.uniform(size=T)[:, None]
            states = np.clip((u > cumsum).sum(axis=1).astype(int), 0, K-1)
            
            if verbose and iteration % 1000 == 0 and iteration > 0:
                print(f"    迭代 {iteration}/{n_iterations} ({iteration/n_iterations*100:.0f}%)")
            
            for k in range(K):
                mask = (states == k)
                n_k = np.sum(mask)
                
                if n_k > 5:
                    prior_mu_loc = self.prior_params['mu_loc'][k]
                    prior_mu_scale = self.prior_params['mu_scale']
                    
                    posterior_precision = n_k / params['sigma'][k]**2 + 1/prior_mu_scale**2
                    posterior_mean = (
                        np.sum(y[mask]) / params['sigma'][k]**2 + 
                        prior_mu_loc / prior_mu_scale**2
                    ) / posterior_precision
                    
                    params['mu'][k] = np.random.normal(
                        posterior_mean, 
                        1/np.sqrt(posterior_precision)
                    )
                    
                    alpha_shape = self.prior_params['sigma_alpha_shape'] + n_k / 2
                    beta_shape = self.prior_params['sigma_beta_rate'] + \
                                np.sum((y[mask] - params['mu'][k])**2) / 2
                    params['sigma'][k] = 1 / np.random.gamma(alpha_shape, 1/beta_shape)
                    params['sigma'][k] = max(params['sigma'][k], 0.08)
                    
                    if self.emission_dist == 'student_t':
                        nu = self._sample_nu_mh(
                            y[mask], params['mu'][k], params['sigma'][k],
                            alpha=self.prior_params['nu_alpha'],
                            beta=self.prior_params['nu_beta']
                        )
                        params['nu'][k] = nu
                else:
                    strong_prior_weight = max(10.0 / (n_k + 1), 1.0)
                    effective_scale = self.prior_params['mu_scale'] / strong_prior_weight
                    
                    params['mu'][k] = np.random.normal(
                        self.prior_params['mu_loc'][k], 
                        effective_scale
                    )
                    
                    sigma_prior_mean = 0.5
                    sigma_prior_std = 0.2
                    params['sigma'][k] = abs(np.random.normal(sigma_prior_mean, sigma_prior_std))
                    params['sigma'][k] = np.clip(params['sigma'][k], 0.15, 1.5)
                    
                    if self.emission_dist == 'student_t':
                        nu_prior_mode = 4.0 + (n_k * 0.5)
                        nu_concentration = max(3.0, n_k * 0.8)
                        params['nu'][k] = np.random.gamma(nu_concentration, 1/nu_concentration) * nu_prior_mode
                        params['nu'][k] = max(params['nu'][k], 2.1)
            
            transition_counts = np.zeros((K, K))
            states_shifted = states[:-1]
            states_next = states[1:]
            for t in range(len(states_shifted)):
                transition_counts[states_shifted[t], states_next[t]] += 1
            
            concentration = self.prior_params['P_concentration']
            for i in range(K):
                counts_i = transition_counts[i]
                if counts_i.sum() > 0:
                    params['P'][i] = np.random.dirichlet(counts_i + concentration/K)
                else:
                    params['P'][i] = np.random.dirichlet(np.ones(K) * concentration)
                
                params['P'][i] /= params['P'][i].sum()
            
            state_counts = np.bincount(states, minlength=K).astype(float)
            params['pi'] = np.random.dirichlet(
                state_counts + self.prior_params['pi_concentration']
            )
            params['pi'] /= params['pi'].sum()
            
            mu_samples[iteration] = params['mu'].copy()
            sigma_samples[iteration] = params['sigma'].copy()
            P_samples[iteration] = params['P'].copy()
            pi_samples[iteration] = params['pi'].copy()
            state_samples[iteration] = states.copy()
            
            if self.emission_dist == 'student_t':
                nu_samples[iteration] = params['nu'].copy()
            
            log_likelihoods[iteration] = logsumexp(log_alpha[-1])
        
        result = {
            'mu': mu_samples,
            'sigma': sigma_samples,
            'P': P_samples,
            'pi': pi_samples,
            'states': state_samples,
            'log_likelihood': log_likelihoods
        }
        
        if self.emission_dist == 'student_t':
            result['nu'] = nu_samples
        
        return result

    def _compute_log_emission_stable(self, params: dict) -> np.ndarray:
        """
        计算对数发射概率（数值稳定版本）
        
        已废弃：请使用 _compute_log_emission_matrix()
        保留此方法以兼容旧调用
        """
        return self._compute_log_emission_matrix(params)

    def _forward_algorithm(self, log_emission: np.ndarray, params: dict) -> np.ndarray:
        """
        前向算法（对数空间）
        
        参数：
            log_emission: 对数发射矩阵 (T, K)
            params: 包含转移矩阵P和初始分布pi的参数字典
            
        返回：
            log_alpha: 前向概率矩阵 (T, K)
        """
        T = len(self.y)
        K = self.n_states
        P = params['P']
        pi = params['pi']
        
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = np.log(pi + 1e-300) + log_emission[0]
        
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = log_emission[t, k] + logsumexp(
                    np.log(P[:, k] + 1e-300) + log_alpha[t-1]
                )
        
        log_alpha -= logsumexp(log_alpha[-1])
        
        return log_alpha

    def _backward_algorithm(self, log_emission: np.ndarray, params: dict) -> np.ndarray:
        """
        后向算法
        
        参数：
            log_emission: 对数发射矩阵 (T, K)
            params: 包含转移矩阵P的参数字典
            
        返回：
            log_beta: 后向概率矩阵 (T, K)
        """
        T = len(self.y)
        K = self.n_states
        P = params['P']
        
        log_beta = np.zeros((T, K))
        
        for t in range(T-2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(
                    np.log(P[k, :] + 1e-300) + log_emission[t+1] + log_beta[t+1]
                )
        
        return log_beta

    def _sample_nu_mh(self, y_k, mu_k, sigma_k, alpha, beta):
        """Metropolis-Hastings采样Student-t自由度nu"""
        current_nu = max(2.1, np.random.gamma(alpha, 1/beta))
        
        proposed_nu = current_nu * np.exp(np.random.normal(0, 0.15))
        proposed_nu = max(proposed_nu, 2.1)
        
        log_lik_current = np.sum(stats.t.logpdf(y_k, df=current_nu, loc=mu_k, scale=sigma_k))
        log_lik_proposed = np.sum(stats.t.logpdf(y_k, df=proposed_nu, loc=mu_k, scale=sigma_k))
        
        log_prior_current = (alpha - 1) * np.log(current_nu) - beta * current_nu
        log_prior_proposed = (alpha - 1) * np.log(proposed_nu) - beta * proposed_nu
        
        log_jacobian = np.log(proposed_nu) - np.log(current_nu)
        
        log_accept_ratio = (
            (log_lik_proposed + log_prior_proposed) - 
            (log_lik_current + log_prior_current) + 
            log_jacobian
        )
        
        if np.log(np.random.uniform()) < log_accept_ratio:
            return proposed_nu
        else:
            return current_nu

    def _compute_posterior_summary_robust(self) -> dict:
        """
        计算增强的后验摘要
        
        返回：
            posterior_summary: 包含所有后验统计量的字典
            同时计算并存储 Gelman-Rubin R-hat 诊断量
        """
        all_mu = np.concatenate([chain['mu'] for chain in self.chains], axis=0)
        all_sigma = np.concatenate([chain['sigma'] for chain in self.chains], axis=0)
        all_P = np.concatenate([chain['P'] for chain in self.chains], axis=0)
        all_pi = np.concatenate([chain['pi'] for chain in self.chains], axis=0)
        all_states = np.concatenate([chain['states'] for chain in self.chains], axis=0)
        
        mu_order = np.argsort(np.mean(all_mu, axis=0))
        self.state_labels_ordered = [self.state_labels[i] for i in mu_order]
        
        self.posterior_summary = {
            'mu': {
                'mean': np.mean(all_mu, axis=0),
                'std': np.std(all_mu, axis=0),
                'ci_lower': np.percentile(all_mu, 2.5, axis=0),
                'ci_upper': np.percentile(all_mu, 97.5, axis=0),
                'samples': all_mu
            },
            'sigma': {
                'mean': np.mean(all_sigma, axis=0),
                'std': np.std(all_sigma, axis=0),
                'ci_lower': np.percentile(all_sigma, 2.5, axis=0),
                'ci_upper': np.percentile(all_sigma, 97.5, axis=0),
                'samples': all_sigma
            },
            'P': {
                'mean': np.mean(all_P, axis=0),
                'std': np.std(all_P, axis=0),
                'ci_lower': np.percentile(all_P, 2.5, axis=0),
                'ci_upper': np.percentile(all_P, 97.5, axis=0),
                'samples': all_P
            },
            'pi': {
                'mean': np.mean(all_pi, axis=0),
                'std': np.std(all_pi, axis=0),
                'samples': all_pi
            },
            'most_probable_states': self._compute_viterbi(all_states),
            'state_probs': self._compute_state_probs(all_states),
            'log_likelihood': np.concatenate([chain['log_likelihood'] for chain in self.chains]),
            'state_labels_ordered': self.state_labels_ordered,
            'change_points': self._detect_change_points(all_states)
        }
        
        if self.emission_dist == 'student_t' and 'nu' in self.chains[0]:
            all_nu = np.concatenate([chain['nu'] for chain in self.chains], axis=0)
            self.posterior_summary['nu'] = {
                'mean': np.mean(all_nu, axis=0),
                'std': np.std(all_nu, axis=0),
                'ci_lower': np.percentile(all_nu, 2.5, axis=0),
                'ci_upper': np.percentile(all_nu, 97.5, axis=0),
                'samples': all_nu
            }
        
        if len(self.chains) > 1:
            r_hat = self._compute_gelman_rubin_full()
            self.posterior_summary['gelman_rubin'] = r_hat
            self.posterior_summary['convergence_diagnostics'] = {
                'r_hat': r_hat,
                'n_chains': len(self.chains),
                'converged': r_hat < 1.1,
                'acceptable': r_hat < 1.2
            }
        
        return self.posterior_summary

    def _compute_viterbi(self, all_state_samples):
        """Viterbi算法：计算最可能的状态序列"""
        T = len(self.y)
        K = self.n_states
        
        state_counts = np.zeros((T, K))
        for states in all_state_samples.reshape(-1, T):
            for t in range(T):
                state_counts[t, states[t]] += 1
        
        most_probable = np.argmax(state_counts, axis=1)
        return most_probable

    def _compute_state_probs(self, all_state_samples):
        """计算每个时间点的状态概率"""
        T = len(self.y)
        K = self.n_states
        
        state_counts = np.zeros((T, K))
        for states in all_state_samples.reshape(-1, T):
            for t in range(T):
                state_counts[t, states[t]] += 1
        
        probs = state_counts / state_counts.sum(axis=1, keepdims=True)
        return probs

    def _detect_change_points(self, all_state_samples):
        """检测状态变化点"""
        T = len(self.y)
        most_probable = self.posterior_summary.get('most_probable_states', 
                            np.zeros(T, dtype=int))
        
        change_points = []
        for t in range(1, T):
            if most_probable[t] != most_probable[t-1]:
                change_points.append(t)
        
        return change_points

    def _print_comprehensive_diagnostic(self):
        """打印全面的收敛诊断报告"""
        if not self.chains:
            return
        
        trace = self.posterior_summary['log_likelihood']
        
        gelman_rubin = 1.0
        if len(self.chains) > 1:
            gelman_rubin = self._compute_gelman_rubin_full()
        else:
            print("  [警告] 仅运行了1条链，R-hat不可靠！建议至少运行3条链")
        
        ess = self._compute_effective_sample_size(trace)
        
        print(f"\n  收敛诊断报告:")
        print(f"    ┌─────────────────────────────────────────────┐")
        print(f"    │ Gelman-Rubin R-hat: {gelman_rubin:.4f}", end="")
        if gelman_rubin < 1.01:
            print(" [优秀] │")
        elif gelman_rubin < 1.05:
            print(" [良好] │")
        elif gelman_rubin < 1.1:
            print(" [通过] │")
        else:
            print(" [警告!]│")
        
        print(f"    │ 有效样本量 (ESS):  {ess:.0f}", end="")
        if ess > 1000:
            print(" [充足] │")
        elif ess > 400:
            print(" [中等] │")
        else:
            print(" [不足!]│")
        print(f"    └─────────────────────────────────────────────┘")
        
        print(f"\n  状态参数后验估计 ({self.emission_dist}分布, 异方差):")
        print(f"  " + "-"*60)
        for i, label in enumerate(self.state_labels_ordered):
            mu_est = self.posterior_summary['mu']['mean'][i]
            mu_ci = (self.posterior_summary['mu']['ci_lower'][i],
                    self.posterior_summary['mu']['ci_upper'][i])
            sigma_est = self.posterior_summary['sigma']['mean'][i]
            sigma_ci = (self.posterior_summary['sigma']['ci_lower'][i],
                       self.posterior_summary['sigma']['ci_upper'][i])
            
            output_str = f"    状态 {i} ({label:6s}):"
            output_str += f" μ={mu_est:+7.3f} [{mu_ci[0]:+.3f}, {mu_ci[1]:+.3f}], "
            output_str += f"σ={sigma_est:.3f} [{sigma_ci[0]:.3f}, {sigma_ci[1]:.3f}]"
            
            if self.emission_dist == 'student_t' and 'nu' in self.posterior_summary:
                nu_est = self.posterior_summary['nu']['mean'][i]
                output_str += f", ν={nu_est:.1f}"
            
            print(output_str)
        
        print(f"  " + "-"*60)
        
        self._posterior_predictive_check()

    def _compute_gelman_rubin_full(self):
        """完整的Gelman-Rubin统计量计算"""
        M = len(self.chains)
        N = min(len(chain['mu']) for chain in self.chains)
        
        if M < 2 or N < 20:
            return 1.0
        
        r_hats = []
        param_names = ['mu', 'sigma']
        
        for param_name in param_names:
            chains_param = [chain[param_name][:N] for chain in self.chains]
            chain_means = np.mean(chains_param, axis=1)
            overall_mean = np.mean(chain_means)
            
            B = N / (M - 1) * np.sum((chain_means - overall_mean)**2)
            W = np.mean([np.var(chain, ddof=1) for chain in chains_param])
            
            var_hat = (N - 1) / N * W + B / N
            
            if W > 1e-10:
                r_hat = np.sqrt(var_hat / W)
                r_hats.append(r_hat)
        
        return float(np.max(r_hats)) if r_hats else 1.0

    def _compute_effective_sample_size(self, trace):
        """有效样本量计算（考虑自相关）"""
        n = len(trace)
        if n < 20:
            return float(n)
        
        mean_trace = np.mean(trace)
        var_trace = np.var(trace)
        
        if var_trace < 1e-10:
            return float(n)
        
        centered = trace - mean_trace
        autocorr = np.correlate(centered, centered, mode='full')
        autocorr = autocorr[n-1:] / (var_trace * n)
        
        tau = 1
        for lag in range(1, min(n//2, 500)):
            if autocorr[lag] < 0.05:
                break
            tau += autocorr[lag]
        
        ess = n / (1 + 2 * tau)
        return max(ess, 1.0)

    def _posterior_predictive_check(self):
        """后验预测检验（PPC）"""
        y = self.y
        T = len(y)
        K = self.n_states
        
        mu_samples = self.posterior_summary['mu']['samples']
        sigma_samples = self.posterior_summary['sigma']['samples']
        S = len(mu_samples)
        
        y_rep = np.zeros((S, T))
        for s in range(S):
            for t in range(T):
                k = np.random.choice(K)
                if self.emission_dist == 'gaussian':
                    y_rep[s, t] = np.random.normal(mu_samples[s, k], sigma_samples[s, k])
                else:
                    nu = self.posterior_summary.get('nu', {}).get('mean', np.full(K, 5))[k]
                    y_rep[s, t] = stats.t.rvs(nu, loc=mu_samples[s, k], scale=sigma_samples[s, k])
        
        obs_stat = {
            'mean': np.mean(y),
            'std': np.std(y),
            'skewness': stats.skew(y),
            'kurtosis': stats.kurtosis(y)
        }
        
        rep_stats = {
            'mean': np.mean(y_rep, axis=1),
            'std': np.std(y_rep, axis=1),
            'skewness': np.array([stats.skew(y_rep[s]) for s in range(S)]),
            'kurtosis': np.array([stats.kurtosis(y_rep[s]) for s in range(S)])
        }
        
        p_values = {}
        for stat_name in obs_stat:
            p_value = np.mean(rep_stats[stat_name] >= obs_stat[stat_name])
            p_values[stat_name] = p_value
        
        print(f"\n  后验预测检验 (Posterior Predictive Check):")
        print(f"  " + "-"*55)
        print(f"    {'统计量':<12} {'观测值':>10} {'复制均值':>10} {'贝叶斯p值':>10}")
        print(f"  " + "-"*55)
        for stat_name in obs_stat:
            obs_val = obs_stat[stat_name]
            rep_mean = np.mean(rep_stats[stat_name])
            p_val = p_values[stat_name]
            flag = "[OK]" if 0.05 < p_val < 0.95 else "[!]"
            print(f"    {stat_name:<12} {obs_val:>10.3f} {rep_mean:>10.3f} {p_val:>9.3f} {flag}")
        print(f"  " + "-"*55)

    def compute_model_criteria(self, y: np.ndarray = None) -> dict:
        """
        计算完整的信息准则（BIC/AIC/AICc/WAIC）
        
        统一使用 _log_emission_pdf 计算对数似然
        """
        if y is None:
            y = self.y
        
        T = len(y)
        K = self.n_states
        
        mu = self.posterior_summary['mu']['mean']
        sigma = self.posterior_summary['sigma']['mean']
        P = self.posterior_summary['P']['mean']
        pi = self.posterior_summary['pi']['mean']
        
        log_lik = 0.0
        for t in range(T):
            prob = 0.0
            for k in range(K):
                log_pdf_k = self._log_emission_pdf(
                    np.array([y[t]]), mu[k], sigma[k],
                    nu=self.posterior_summary.get('nu', {}).get('mean', np.full(K, 5.0))[k] 
                        if self.emission_dist == 'student_t' else 5.0,
                    dist=self.emission_dist
                )
                prob += pi[k] * np.exp(log_pdf_k[0])
            log_lik += np.log(prob + 1e-300)
        
        n_params = K*2 + K*K + K - 1
        if self.emission_dist == 'student_t' and 'nu' in self.posterior_summary:
            n_params += K
        
        bic = -2*log_lik + n_params * np.log(T)
        aic = -2*log_lik + 2*n_params
        aicc = aic + (2*n_params*(n_params+1))/(T-n_params-1) if T-n_params-1 > 0 else aic
        
        lppd = self._compute_lppd(y)
        p_waic = self._compute_waic_penalty()
        waic = -2 * (lppd - p_waic)
        
        return {
            'log_likelihood': log_lik,
            'BIC': bic,
            'AIC': aic,
            'AICc': aicc,
            'WAIC': waic,
            'n_parameters': n_params,
            'emission_distribution': self.emission_dist
        }

    def _compute_lppd(self, y: np.ndarray) -> float:
        """
        逐点对数预测密度 (LPPD)
        
        统一使用 _log_emission_pdf 计算对数似然
        """
        T = len(y)
        K = self.n_states
        
        mu_samples = self.posterior_summary['mu']['samples']
        sigma_samples = self.posterior_summary['sigma']['samples']
        S = len(mu_samples)
        
        nu_samples = None
        if self.emission_dist == 'student_t' and 'nu' in self.posterior_summary:
            nu_samples = self.posterior_summary['nu']['samples']
        
        lppd = 0.0
        for t in range(T):
            point_log_liks = np.zeros(S)
            for s in range(S):
                prob = 0.0
                for k in range(K):
                    nu_sk = nu_samples[s, k] if nu_samples is not None else 5.0
                    log_pdf_k = self._log_emission_pdf(
                        np.array([y[t]]), mu_samples[s, k], sigma_samples[s, k],
                        nu=nu_sk, dist=self.emission_dist
                    )
                    prob += np.exp(log_pdf_k[0])
                point_log_liks[s] = np.log(prob + 1e-300)
            
            lppd += logsumexp(point_log_liks) - np.log(S)
        
        return lppd

    def _compute_waic_penalty(self):
        """WAIC惩罚项"""
        log_lik_samples = self.posterior_summary['log_likelihood']
        p_waic = np.var(log_lik_samples)
        return p_waic


class RobustModelSelector:
    """
    鲁棒模型选择器
    
    改进：
    - 使用WAIC为主准则（更适合贝叶斯模型比较）
    - 综合多个指标进行决策
    - 自动处理K>=4的情况
    - 提供详细的模型比较报告
    """
    
    def __init__(self, max_K=5, primary_criteria='WAIC'):
        self.max_K = max_K
        self.primary_criteria = primary_criteria
        self.results = {}

    def select_best_model(self, y, n_iterations=3000, burn_in=1500, 
                         emission_dist='student_t', verbose=True):
        """
        选择最优状态数K
        
        策略：
        1. 测试K=2到max_K的所有模型
        2. 计算多重信息准则
        3. 以WAIC为主、BIC为辅进行综合判断
        4. 返回最优模型及详细诊断
        """
        if verbose:
            print("\n" + "="*70)
            print("自动模型选择 (Robust Model Selection)")
            print("="*70)
            print(f"\n搜索范围: K = 2 到 {self.max_K}")
            print(f"主选择准则: {self.primary_criteria}")
            print(f"发射分布: {emission_dist}")
        
        best_K = 2
        best_score = np.inf
        all_results = {}
        
        for K in range(2, self.max_K + 1):
            if verbose:
                print(f"\n{'='*60}")
                print(f"测试 K={K}...")
                print(f"{'='*60}")
            
            try:
                hmm = RobustBayesianHMM(n_states=K, random_seed=42,
                                        emission_dist=emission_dist)
                
                result = hmm.fit(y, n_iterations=n_iterations, burn_in=burn_in,
                               n_chains=3, thin=5, verbose=False)
                
                criteria_values = hmm.compute_model_criteria(y)
                score = criteria_values[self.primary_criteria]
                
                model_info = {
                    'criteria_score': score,
                    'all_criteria': criteria_values,
                    'model': hmm,
                    'result': result,
                    'convergence_ok': True
                }
                
                if len(hmm.chains) > 1:
                    r_hat = hmm._compute_gelman_rubin_full()
                    model_info['r_hat'] = r_hat
                    model_info['convergence_ok'] = r_hat < 1.2
                
                all_results[K] = model_info
                
                if verbose:
                    conv_flag = "[收敛]" if model_info['convergence_ok'] else "[未收敛!]"
                    print(f"\n  结果 ({conv_flag}):")
                    print(f"    {self.primary_criteria}: {score:.2f}")
                    print(f"    BIC: {criteria_values['BIC']:.2f}, "
                         f"AICc: {criteria_values['AICc']:.2f}, "
                         f"WAIC: {criteria_values['WAIC']:.2f}")
                    if 'r_hat' in model_info:
                        print(f"    R-hat: {model_info['r_hat']:.4f}")
                
                if model_info['convergence_ok'] and score < best_score:
                    best_score = score
                    best_K = K
                    
            except Exception as e:
                if verbose:
                    print(f"\n  [错误] K={K} 失败: {str(e)[:80]}")
                continue
        
        self.results = all_results
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"模型选择总结:")
            print(f"{'='*70}")
            print(f"\n  最优状态数: K={best_K}")
            if best_K in all_results:
                best_info = all_results[best_K]
                print(f"  主准则 ({self.primary_criteria}): {best_info['criteria_score']:.2f}")
                
                converged_models = [k for k, v in all_results.items() if v.get('convergence_ok')]
                print(f"\n  收敛良好的模型: K={converged_models}")
                
                if len(converged_models) > 1:
                    print(f"\n  模型排名 (按{self.primary_criteria}):")
                    ranked = sorted([(k, v['criteria_score'], v['all_criteria']['BIC']) 
                                   for k, v in all_results.items() 
                                   if v.get('convergence_ok')],
                                  key=lambda x: x[1])
                    for rank, (k, score_waic, score_bic) in enumerate(ranked[:3], 1):
                        marker = "★" if k == best_K else " "
                        print(f"    {marker}{rank}. K={k}: WAIC={score_waic:.1f}, BIC={score_bic:.1f}")
        
        return best_K, all_results.get(best_K)


class StandardEventDetector:
    """
    标准化ENSO事件检测器
    
    基于气象标准的规范化检测：
    - NOAA/CPC定义：ONI指数连续5个月≥+0.5°C（厄尔尼诺）或≤-0.5°C（拉尼娜）
    - 最小事件间隔：2个月
    - 事件合并规则：相邻同类型事件间隔<3个月则合并
    """
    
    def __init__(self, threshold=0.5, min_duration=5, min_gap=2):
        self.threshold = threshold
        self.min_duration = min_duration
        self.min_gap = min_gap
        self.method_name = f"气象标准法 (阈值=±{threshold}°C, 持续≥{min_duration}月)"
    
    def detect_events_from_raw(self, nino34_raw, dates=None):
        """
        基于原始NINO3.4数据的标准事件检测
        
        这是主分析方法（不是敏感性分析）
        """
        events = []
        current_event = None
        event_start = 0
        
        for i, value in enumerate(nino34_raw):
            if value > self.threshold:
                event_type = 'El Nino'
            elif value < -self.threshold:
                event_type = 'La Nina'
            else:
                event_type = None
            
            if event_type == current_event:
                continue
            else:
                if current_event is not None:
                    duration = i - event_start
                    if duration >= self.min_duration:
                        peak_idx = event_start + np.argmax(nino34_raw[event_start:i])
                        events.append({
                            'type': current_event,
                            'start_idx': event_start,
                            'end_idx': i-1,
                            'peak_idx': peak_idx,
                            'peak_value': nino34_raw[peak_idx],
                            'duration': duration,
                            'mean_amplitude': np.mean(nino34_raw[event_start:i]),
                            'max_amplitude': np.max(nino34_raw[event_start:i]),
                            'onset_date': dates.iloc[event_start] if dates is not None else None,
                            'decay_date': dates.iloc[i-1] if dates is not None else None
                        })
                
                if event_type is not None:
                    event_start = i
                    current_event = event_type
                else:
                    current_event = None
        
        if current_event is not None:
            duration = len(nino34_raw) - event_start
            if duration >= self.min_duration:
                peak_idx = event_start + np.argmax(nino34_raw[event_start:])
                events.append({
                    'type': current_event,
                    'start_idx': event_start,
                    'end_idx': len(nino34_raw)-1,
                    'peak_idx': peak_idx,
                    'peak_value': nino34_raw[peak_idx],
                    'duration': duration,
                    'mean_amplitude': np.mean(nino34_raw[event_start:]),
                    'max_amplitude': np.max(nino34_raw[event_start:]),
                    'onset_date': dates.iloc[event_start] if dates is not None else None,
                    'decay_date': dates.iloc[-1] if dates is not None else None
                })
        
        events = self._merge_close_events(events)
        
        return events
    
    def _merge_close_events(self, events, gap_threshold=None):
        """合并间隔过近的事件"""
        if gap_threshold is None:
            gap_threshold = self.min_gap
        
        if len(events) <= 1:
            return events
        
        merged = [events[0]]
        for event in events[1:]:
            last_event = merged[-1]
            gap = event['start_idx'] - last_event['end_idx']
            
            if (event['type'] == last_event['type'] and 
                gap < gap_threshold):
                merged[-1]['end_idx'] = event['end_idx']
                merged[-1]['duration'] = merged[-1]['end_idx'] - merged[-1]['start_idx']
                merged[-1]['peak_value'] = max(merged[-1]['peak_value'], event['peak_value'])
                merged[-1]['max_amplitude'] = max(merged[-1]['max_amplitude'], event['max_amplitude'])
                merged[-1]['mean_amplitude'] = np.mean([
                    merged[-1]['mean_amplitude'], event['mean_amplitude']
                ])
                if event.get('decay_date'):
                    merged[-1]['decay_date'] = event['decay_date']
            else:
                merged.append(event)
        
        return merged


class AsymmetryAnalyzer:
    """
    ENOS不对称性分析器
    
    改进：
    - 使用标准化事件检测方法
    - 完整的不对称性指标体系
    - Bootstrap置信区间
    - 敏感性分析（多阈值）
    """
    
    def __init__(self, n_bootstrap=2000, confidence_level=0.95):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.events = {}
        self.asymmetry_results = {}
    
    def analyze_comprehensive(self, hmm_result, dates, raw_nino34, 
                             n_bootstrap=None, confidence_level=None,
                             use_standard_detection=True):
        """
        全面不对称性分析
        
        参数：
        - use_standard_detection: 是否使用气象标准事件检测（推荐True）
        """
        if n_bootstrap:
            self.n_bootstrap = n_bootstrap
        if confidence_level:
            self.confidence_level = confidence_level
        
        self.dates = dates
        self.raw_nino34 = raw_nino34
        self.hmm_result = hmm_result
        P = hmm_result['P']['mean']
        self.n_states = P.shape[0]
        
        if use_standard_detection:
            self._identify_events_standard()
        else:
            self._identify_events_from_hmm()
        
        self._analyze_duration_asymmetry()
        self._analyze_amplitude_asymmetry()
        self._analyze_transition_asymmetry()
        self._analyze_evolution_speed_asymmetry()
        self._perform_sensitivity_analysis()
        
        self.asymmetry_results.update({
            'total_warm_events': len(self.events.get('El Nino', [])),
            'total_cold_events': len(self.events.get('La Nina', [])),
            'method': '气象标准法' if use_standard_detection else '贝叶斯HMM',
            'confidence_level': self.confidence_level
        })
        
        return self.asymmetry_results

    def _identify_events_standard(self):
        """使用气象标准检测事件（推荐方法）"""
        detector = StandardEventDetector(
            threshold=0.5, 
            min_duration=5, 
            min_gap=2
        )
        
        events = detector.detect_events_from_raw(self.raw_nino34, self.dates)
        
        self.events['El Nino'] = [e for e in events if e['type'] == 'El Nino']
        self.events['La Nina'] = [e for e in events if e['type'] == 'La Nina']
        
        print(f"\n事件检测结果 (气象标准法):")
        print(f"  厄尔尼诺事件: {len(self.events['El Nino'])} 次")
        print(f"  拉尼娜事件: {len(self.events['La Nina'])} 次")
        
        if len(self.events['El Nino']) > 0:
            avg_dur_en = np.mean([e['duration'] for e in self.events['El Nino']])
            print(f"  厄尔尼诺平均持续: {avg_dur_en:.1f} 月")
        
        if len(self.events['La Nina']) > 0:
            avg_dur_ln = np.mean([e['duration'] for e in self.events['La Nina']])
            print(f"  拉尼娜平均持续: {avg_dur_ln:.1f} 月")

    def _identify_events_from_hmm(self):
        """从HMM状态序列识别事件（备用方法）"""
        states = self.hmm_result['most_probable_states']
        warm_state_idx = self.n_states - 1
        cold_state_idx = 0
        
        warm_events = []
        cold_events = []
        
        in_warm_event = False
        in_cold_event = False
        event_start = 0
        
        for t in range(len(states)):
            if states[t] == warm_state_idx and not in_warm_event:
                in_warm_event = True
                event_start = t
            elif states[t] != warm_state_idx and in_warm_event:
                warm_events.append({'start_idx': event_start, 'end_idx': t-1, 
                                   'duration': t - event_start})
                in_warm_event = False
                
            if states[t] == cold_state_idx and not in_cold_event:
                in_cold_event = True
                event_start = t
            elif states[t] != cold_state_idx and in_cold_event:
                cold_events.append({'start_idx': event_start, 'end_idx': t-1, 
                                   'duration': t - event_start})
                in_cold_event = False
        
        if in_warm_event:
            warm_events.append({'start_idx': event_start, 'end_idx': len(states)-1, 
                               'duration': len(states) - event_start})
        if in_cold_event:
            cold_events.append({'start_idx': event_start, 'end_idx': len(states)-1, 
                               'duration': len(states) - event_start})
        
        self.events['El Nino'] = warm_events
        self.events['La Nina'] = cold_events
        
        print(f"\n事件检测结果 (HMM状态法):")
        print(f"  厄尔尼诺事件: {len(warm_events)} 次")
        print(f"  拉尼娜事件: {len(cold_events)} 次")

    def _analyze_duration_asymmetry(self):
        """持续时间不对称性分析"""
        warm_durations = np.array([e['duration'] for e in self.events.get('El Nino', [])])
        cold_durations = np.array([e['duration'] for e in self.events.get('La Nina', [])])
        
        if len(warm_durations) < 2 or len(cold_durations) < 2:
            print("  [警告] 事件数量不足，无法进行持续时间不对称性分析")
            self.asymmetry_results['duration_asymmetry'] = {'is_significant': False}
            return
        
        diff = np.mean(warm_durations) - np.mean(cold_durations)
        pooled_std = np.sqrt((np.var(warm_durations, ddof=1) + np.var(cold_durations, ddof=1)) / 2)
        t_stat = diff / (pooled_std * np.sqrt(1/len(warm_durations) + 1/len(cold_durations)))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(warm_durations)+len(cold_durations)-2))
        
        boot_diffs = []
        for _ in range(self.n_bootstrap):
            warm_boot = np.random.choice(warm_durations, replace=True)
            cold_boot = np.random.choice(cold_durations, replace=True)
            boot_diffs.append(np.mean(warm_boot) - np.mean(cold_boot))
        
        ci_lower = np.percentile(boot_diffs, (1-self.confidence_level)/2 * 100)
        ci_upper = np.percentile(boot_diffs, (1+self.confidence_level)/2 * 100)
        
        is_significant = p_value < 0.05 and not (ci_lower < 0 < ci_upper)
        
        self.asymmetry_results['duration_asymmetry'] = {
            'warm_mean': np.mean(warm_durations),
            'cold_mean': np.mean(cold_durations),
            'difference': diff,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_interval': (ci_lower, ci_upper),
            't_statistic': t_stat,
            'warm_count': len(warm_durations),
            'cold_count': len(cold_durations)
        }
        
        print(f"\n持续时间不对称性:")
        print(f"  厄尔尼诺平均持续: {np.mean(warm_durations):.1f} 个月 (n={len(warm_durations)})")
        print(f"  拉尼娜平均持续: {np.mean(cold_durations):.1f} 个月 (n={len(cold_durations)})")
        print(f"  差异: {diff:+.2f} 个月")
        print(f"  P值: {p_value:.4f}" + (" [显著]" if is_significant else " [不显著]"))
        print(f"  95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

    def _analyze_amplitude_asymmetry(self):
        """振幅不对称性分析"""
        warm_amplitudes = []
        for event in self.events.get('El Nino', []):
            amp = event.get('mean_amplitude', 0)
            warm_amplitudes.append(amp)
        
        cold_amplitudes = []
        for event in self.events.get('La Nina', []):
            amp = abs(event.get('mean_amplitude', 0))
            cold_amplitudes.append(amp)
        
        if len(warm_amplitudes) < 2 or len(cold_amplitudes) < 2:
            print("  [警告] 事件数量不足，无法进行振幅不对称性分析")
            self.asymmetry_results['amplitude_asymmetry'] = {'is_significant': False}
            return
        
        warm_arr = np.array(warm_amplitudes)
        cold_arr = np.array(cold_amplitudes)
        
        ratio = np.mean(warm_arr) / (np.mean(cold_arr) + 1e-10)
        diff = np.mean(warm_arr) - np.mean(cold_arr)
        
        t_stat, p_value = stats.ttest_ind(warm_arr, cold_arr)
        
        boot_ratios = []
        for _ in range(self.n_bootstrap):
            w_boot = np.random.choice(warm_arr, replace=True)
            c_boot = np.random.choice(cold_arr, replace=True)
            boot_ratios.append(np.mean(w_boot) / (np.mean(c_boot) + 1e-10))
        
        ci_lower = np.percentile(boot_ratios, (1-self.confidence_level)/2 * 100)
        ci_upper = np.percentile(boot_ratios, (1+self.confidence_level)/2 * 100)
        
        is_significant = p_value < 0.05 and not (ci_lower < 1 < ci_upper)
        
        self.asymmetry_results['amplitude_asymmetry'] = {
            'warm_mean': np.mean(warm_arr),
            'cold_mean': np.mean(cold_arr),
            'ratio': ratio,
            'difference': diff,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_interval': (ci_lower, ci_upper)
        }
        
        print(f"\n振幅不对称性:")
        print(f"  厄尔尼诺平均振幅: {np.mean(warm_arr):.2f}°C")
        print(f"  拉尼娜平均振幅: {np.mean(cold_arr):.2f}°C")
        print(f"  暖/冷事件振幅比: {ratio:.3f}")

    def _analyze_transition_asymmetry(self):
        """转移概率不对称性"""
        states = self.hmm_result['most_probable_states']
        P = self.hmm_result['P']['mean']
        
        warm_to_cold = P[self.n_states-1, 0] if self.n_states >= 2 else 0
        cold_to_warm = P[0, self.n_states-1] if self.n_states >= 2 else 0
        
        exit_asymmetry = warm_to_cold - cold_to_warm
        
        neutral_entry_from_warm = P[self.n_states-1, 1] if self.n_states >= 3 else 0
        neutral_entry_from_cold = P[0, 1] if self.n_states >= 3 else 0
        entry_asymmetry = neutral_entry_from_warm - neutral_entry_from_cold
        
        self.asymmetry_results['transition_asymmetry'] = {
            'exit_asymmetry': exit_asymmetry,
            'entry_asymmetry': entry_asymmetry,
            'exit_ci': (max(0, exit_asymmetry - 0.05), min(1, exit_asym + 0.05)),
            'entry_ci': (max(0, entry_asymmetry - 0.05), min(1, entry_asymmetry + 0.05)),
            'significant_exit': abs(exit_asymmetry) > 0.05,
            'significant_entry': abs(entry_asymmetry) > 0.05,
            'warm_exit_rate': warm_to_cold,
            'cold_exit_rate': cold_to_warm
        }
        
        print(f"\n转移概率不对称性:")
        print(f"  退出不对称性 (暖→冷 vs 冷→暖): {exit_asymmetry:+.3f}")
        print(f"  进入不对称性 (暖→中 vs 冷→中): {entry_asymmetry:+.3f}")

    def _analyze_evolution_speed_asymmetry(self) -> None:
        """演化速度不对称性"""
        warm_speeds = []
        for event in self.events.get('El Nino', []):
            if event['duration'] > 1:
                start_val = self.raw_nino34[event['start_idx']]
                end_val = self.raw_nino34[min(event['end_idx'], len(self.raw_nino34)-1)]
                speed = (end_val - start_val) / event['duration']
                warm_speeds.append(speed)
        
        cold_speeds = []
        for event in self.events.get('La Nina', []):
            if event['duration'] > 1:
                start_val = self.raw_nino34[event['start_idx']]
                end_val = self.raw_nino34[min(event['end_idx'], len(self.raw_nino34)-1)]
                speed = (end_val - start_val) / event['duration']
                cold_speeds.append(speed)
        
        if len(warm_speeds) < 2 or len(cold_speeds) < 2:
            print("  [警告] 事件数量不足，无法进行演化速度不对称性分析")
            self.asymmetry_results['evolution_speed_asymmetry'] = {'is_significant': False}
            return
        
        warm_arr = np.array(warm_speeds)
        cold_arr = np.array(cold_speeds)
        
        warm_avg_speed = np.mean(warm_arr)
        cold_avg_speed = np.mean(cold_arr)
        speed_ratio = abs(warm_avg_speed) / (abs(cold_avg_speed) + 1e-10)
        
        t_stat, p_value = stats.ttest_ind(warm_arr, cold_arr)
        
        boot_ratios = []
        for _ in range(self.n_bootstrap):
            w_boot = np.random.choice(warm_arr, replace=True)
            c_boot = np.random.choice(cold_arr, replace=True)
            w_spd = np.mean(w_boot)
            c_spd = np.mean(c_boot)
            boot_ratios.append(abs(w_spd) / (abs(c_spd) + 1e-10))
        
        ci_lower = np.percentile(boot_ratios, (1-self.confidence_level)/2 * 100)
        ci_upper = np.percentile(boot_ratios, (1+self.confidence_level)/2 * 100)
        
        is_significant = p_value < 0.05
        
        self.asymmetry_results['evolution_speed_asymmetry'] = {
            'warm_speed': warm_avg_speed,
            'cold_speed': cold_avg_speed,
            'speed_ratio': speed_ratio,
            'p_value': p_value,
            'is_significant': is_significant,
            'confidence_interval': (ci_lower, ci_upper)
        }
        
        print(f"\n演化速度不对称性:")
        print(f"  厄尔尼诺平均演化速度: {warm_avg_speed:+.3f} °C/月")
        print(f"  拉尼娜平均演化速度: {cold_avg_speed:+.3f} °C/月")
        print(f"  速度比 (|暖|/|冷|): {speed_ratio:.3f}")

    def _perform_sensitivity_analysis(self):
        """多阈值敏感性分析"""
        thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
        results_by_threshold = {}
        
        for thresh in thresholds:
            trad_method = StandardEventDetector(threshold=thresh, min_duration=5)
            trad_events = trad_method.detect_events_from_raw(self.raw_nino34)
            
            warm_dur = [e['duration'] for e in trad_events if e['type'] == 'El Nino']
            cold_dur = [e['duration'] for e in trad_events if e['type'] == 'La Nina']
            
            if len(warm_dur) > 0 and len(cold_dur) > 0:
                results_by_threshold[thresh] = {
                    'warm_duration_mean': np.mean(warm_dur),
                    'cold_duration_mean': np.mean(cold_dur),
                    'difference': np.mean(warm_dur) - np.mean(cold_dur),
                    'n_warm': len(warm_dur),
                    'n_cold': len(cold_dur)
                }
        
        self.asymmetry_results['sensitivity_analysis'] = results_by_threshold
        
        print(f"\n敏感性分析 ({self.n_bootstrap} 次Bootstrap重抽样):")
        print(f"  不同阈值下的持续时间差异:")
        for thresh, res in sorted(results_by_threshold.items()):
            print(f"    +/-{thresh} C: 差异={res['difference']:+.2f} 个月 "
                 f"(暖:{res['warm_duration_mean']:.1f} vs 冷:{res['cold_duration_mean']:.1f})")


class ENSOVisualizer:
    """可视化工具类"""
    
    def __init__(self, figsize=(16, 14), dpi=150):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = {
            'El Nino': '#FF4444',
            'La Nina': '#4444FF',
            'Neutral': '#888888',
            'warm_shade': '#FFCCCC',
            'cold_shade': '#CCCCFF'
        }
    
    def create_comprehensive_plot(self, data, hmm_result, asymmetry_results, 
                                 save_path='ENSO_Analysis_Report.png'):
        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
        self._plot_time_series_with_states(ax1, data, hmm_result)
        
        ax2 = plt.subplot2grid((3, 2), (1, 0))
        self._plot_transition_matrix(ax2, hmm_result)
        
        ax3 = plt.subplot2grid((3, 2), (1, 1))
        self._plot_parameter_posteriors(ax3, hmm_result)
        
        ax4 = plt.subplot2grid((3, 2), (2, 0))
        self._plot_duration_asymmetry(ax4, asymmetry_results)
        
        ax5 = plt.subplot2grid((3, 2), (2, 1))
        self._plot_amplitude_asymmetry(ax5, asymmetry_results)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"\n可视化结果已保存至: {save_path}")

    def _plot_time_series_with_states(self, ax, data, hmm_result):
        dates = data['dates']
        nino34 = data['raw_nino34']
        states = hmm_result['most_probable_states']
        
        state_colors = ['#4444FF', '#888888', '#FF4444', '#AA44AA', '#44AAAA']
        n_states = len(hmm_result['mu']['mean'])
        
        ax.plot(dates, nino34, 'k-', linewidth=0.8, alpha=0.7, label='NINO3.4')
        
        for t in range(len(states)-1):
            if states[t] == states[t+1]:
                color = state_colors[states[t] % len(state_colors)]
                ax.axvspan(dates[t], dates[t+1], alpha=0.3, color=color)
        
        ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, alpha=0.7, label='厄尔尼诺阈值 (+0.5)')
        ax.axhline(y=-0.5, color='b', linestyle='--', linewidth=1, alpha=0.7, label='拉尼娜阈值 (-0.5)')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        ax.set_xlabel('时间', fontsize=10)
        ax.set_ylabel('NINO3.4 异常 (°C)', fontsize=10)
        ax.set_title('(a) NINO3.4时间序列与HMM状态识别', fontsize=11, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_transition_matrix(self, ax, hmm_result):
        P = hmm_result['P']['mean']
        n_states = P.shape[0]
        
        im = ax.imshow(P, cmap='Blues', vmin=0, vmax=1)
        
        labels = ['拉尼娜', '中性', '厄尔尼诺'][:n_states]
        if n_states > 3:
            labels += [f'状态{i}' for i in range(3, n_states)]
        
        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        
        for i in range(n_states):
            for j in range(n_states):
                text_color = 'white' if P[i,j] > 0.5 else 'black'
                ax.text(j, i, f'{P[i,j]:.2f}', ha='center', va='center', 
                       color=text_color, fontsize=9, fontweight='bold')
        
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title('(b) 状态转移概率矩阵', fontsize=11, fontweight='bold')

    def _plot_parameter_posteriors(self, ax, hmm_result):
        mu_samples = hmm_result.get('mu', {}).get('samples', None)
        
        if mu_samples is not None:
            for i in range(mu_samples.shape[1]):
                ax.hist(mu_samples[:,i], bins=30, alpha=0.6, 
                       label=f'状态{i} (μ={np.mean(mu_samples[:,i]):+.2f})')
            ax.legend(fontsize=8)
        else:
            mu_mean = hmm_result['mu']['mean']
            ax.bar(range(len(mu_mean)), mu_mean, alpha=0.7, color=['blue', 'gray', 'red'])
            ax.set_xticks(range(len(mu_mean)))
            ax.set_xticklabels([f'状态{i}' for i in range(len(mu_mean))], fontsize=9)
        
        ax.set_xlabel('状态', fontsize=10)
        ax.set_ylabel('均值 (°C)', fontsize=10)
        ax.set_title('(c) 发射分布均值后验估计', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    def _plot_duration_asymmetry(self, ax, asymmetry_results):
        dur_asym = asymmetry_results.get('duration_asymmetry', {})
        
        if 'warm_mean' in dur_asym and 'cold_mean' in dur_asym:
            categories = ['厄尔尼诺', '拉尼娜']
            values = [dur_asym['warm_mean'], dur_asym['cold_mean']]
            colors = ['#FF4444', '#4444FF']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.1f}月', ha='center', va='bottom', fontsize=9)
            
            diff = dur_asym.get('difference', 0)
            sig_flag = '*' if dur_asym.get('is_significant', False) else ''
            ax.set_title(f'(d) 持续时间不对称性{sig_flag}\n(差异: {diff:+.1f}月)', 
                        fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(d) 持续时间不对称性', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('平均持续时间 (月)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    def _plot_amplitude_asymmetry(self, ax, asymmetry_results):
        amp_asym = asymmetry_results.get('amplitude_asymmetry', {})
        
        if 'warm_mean' in amp_asym and 'cold_mean' in amp_asym:
            categories = ['厄尔尼诺', '拉尼娜']
            values = [amp_asym['warm_mean'], amp_asym['cold_mean']]
            colors = ['#FF4444', '#4444FF']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}°C', ha='center', va='bottom', fontsize=9)
            
            ratio = amp_asym.get('ratio', 0)
            sig_flag = '*' if amp_asym.get('is_significant', False) else ''
            ax.set_title(f'(e) 振幅不对称性{sig_flag}\n(比值: {ratio:.2f})', 
                        fontsize=11, fontweight='bold')
        else:
            ax.text(0.5, 0.5, '数据不足', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('(e) 振幅不对称性', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('平均振幅 (°C)', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("  基于贝叶斯变点检测的ENSO不对称性演变研究 - 专业版")
    print("  Bayesian ENSO Asymmetry Analysis with Robust HMM")
    print("="*80)
    
    try:
        data_path = r'e:\test\test1_1\venv_env\data_nino.csv'
        
        loader = ENSODataLoader()
        data = loader.load_data(data_path)
        
        y = data['standardized_nino34']
        raw_nino34 = data['raw_nino34']
        dates = data['dates']
        T = data['T']
        
        print(f"\n{'='*80}")
        print("  第一步：自动模型选择 (Robust Model Selection)")
        print(f"{'='*80}")
        print("\n策略说明:")
        print("  - 使用WAIC为主选择准则（更适合贝叶斯预测性能评估）")
        print("  - 测试K=2到K=5的所有模型")
        print("  - 综合收敛诊断(R-hat<1.2)进行筛选")
        print("  - 每个模型使用Student-t分布（处理ENSO厚尾特性）")
        
        selector = RobustModelSelector(max_K=5, primary_criteria='WAIC')
        
        best_K, best_model_info = selector.select_best_model(
            y, 
            n_iterations=4000,  
            burn_in=2000,
            emission_dist='student_t',
            verbose=True
        )
        
        if best_model_info is None:
            print("\n[错误] 模型选择失败！使用默认K=3")
            best_K = 3
            hmm_final = RobustBayesianHMM(n_states=3, random_seed=42,
                                         emission_dist='student_t')
            final_result = hmm_final.fit(y, n_iterations=5000, burn_in=2000,
                                        n_chains=4, thin=5, verbose=True)
        else:
            hmm_final = best_model_info['model']
            final_result = best_model_info['result']
            
            print(f"\n{'='*80}")
            print(f"  第二步：最优模型精炼拟合 (K={best_K})")
            print(f"{'='*80}")
            print(f"\n对选定的K={best_K}进行更长时间的MCMC采样:")
            print("  - 迭代次数提升至5000（从4000）")
            print("  - 预烧期保持2000")
            print("  - 链数增加至4条（从3条）")
            print("  - 稀疏化间隔保持5")
            
            final_result = hmm_final.fit(y, n_iterations=5000, burn_in=2000,
                                        n_chains=4, thin=5, verbose=True)
        
        print(f"\n{'='*80}")
        print(f"  第三步：标准化事件检测 (Meteorological Standard)")
        print(f"{'='*80}")
        print("\n采用NOAA/CPC气象标准:")
        print("  - 厄尔尼诺：ONI连续≥5个月 ≥+0.5°C")
        print("  - 拉尼娜：ONI连续≥5个月 ≤-0.5°C")
        print("  - 最小事件间隔：2个月")
        print("  - 相邻同类型事件间隔<3个月则合并")
        
        analyzer = AsymmetryAnalyzer(n_bootstrap=2000, confidence_level=0.95)
        
        asymmetry_results = analyzer.analyze_comprehensive(
            final_result, dates, raw_nino34,
            use_standard_detection=True
        )
        
        print(f"\n{'='*80}")
        print(f"  第四步：结果可视化与报告生成")
        print(f"{'='*80}")
        
        visualizer = ENSOVisualizer(figsize=(16, 14), dpi=150)
        report_path = r'e:\test\test1_1\ENSO_Analysis_Report_Professional.png'
        
        visualizer.create_comprehensive_plot(
            data, final_result, asymmetry_results, report_path
        )
        
        print(f"\n{'='*80}")
        print("  分析完成总结")
        print(f"{'='*80}")
        print(f"\n  模型配置:")
        print(f"    - 最优状态数: K={best_K}")
        print(f"    - 发射分布: Student-t (异方差)")
        print(f"    - MCMC迭代: 5000次 (预烧2000, 4条链)")
        
        if hasattr(hmm_final, 'posterior_summary') and 'gelman_rubin' in hmm_final.posterior_summary:
            r_hat = hmm_final.posterior_summary['gelman_rubin']
            conv_status = "优秀" if r_hat < 1.01 else ("良好" if r_hat < 1.05 else ("通过" if r_hat < 1.1 else "需关注"))
            print(f"    - 收敛诊断: R-hat={r_hat:.4f} [{conv_status}]")
        
        print(f"\n  事件统计 (气象标准法):")
        print(f"    - 厄尔尼诺事件: {asymmetry_results.get('total_warm_events', 0)} 次")
        print(f"    - 拉尼娜事件: {asymmetry_results.get('total_cold_events', 0)} 次")
        
        dur_asym = asymmetry_results.get('duration_asymmetry', {})
        if 'warm_mean' in dur_asym:
            print(f"\n  不对称性分析结果:")
            print(f"    - 持续时间差异: {dur_asym.get('difference', 0):+.1f} 月", end="")
            if dur_asym.get('is_significant'):
                print(" [显著]")
            else:
                print(" [不显著]")
            
            amp_asym = asymmetry_results.get('amplitude_asymmetry', {})
            if 'ratio' in amp_asym:
                print(f"    - 振幅比值 (暖/冷): {amp_asym['ratio']:.2f}", end="")
                if amp_asym.get('is_significant'):
                    print(" [显著]")
                else:
                    print(" [不显著]")
            
            trans_asym = asymmetry_results.get('transition_asymmetry', {})
            if 'exit_asymmetry' in trans_asym:
                print(f"    - 转移概率不对称性:")
                print(f"      * 退出不对称: {trans_asym['exit_asymmetry']:.3f}")
                print(f"      * 进入不对称: {trans_asym['entry_asymmetry']:.3f}")
        
        print(f"\n  输出文件:")
        print(f"    - 可视化报告: {report_path}")
        
        print(f"\n{'='*80}")
        print("  程序执行完毕")
        print(f"{'='*80}\n")
        
    except FileNotFoundError as e:
        print(f"\n[错误] 数据文件未找到: {e}")
        print("请确保数据文件路径正确: e:\\test\\test1_1\\venv_env\\data_nino.csv")
        
    except Exception as e:
        print(f"\n[错误] 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
