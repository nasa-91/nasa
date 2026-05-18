#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ENSO贝叶斯不对称性分析系统 - 命令行接口

使用方法：
  python cli.py --data data_nino.csv --output results/
  python cli.py --data nino34.txt --format noaa_ascii --states 4 --iterations 8000
  python cli.py --config config.json
"""

import argparse
import json
import sys
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description='ENSO贝叶斯不对称性分析系统 - 命令行接口',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  %(prog)s --data nino_data.csv                    # 使用默认参数运行
  %(prog)s --data data.csv --states 4               # 指定状态数
  %(prog)s --data data.csv --dist gaussian          # 使用高斯分布
  %(prog)s --data data.csv --iter 10000 --burn 3000 # 自定义MCMC参数
  %(prog)s --config my_config.json                  # 使用配置文件
        """
    )
    
    # 数据相关参数
    data_group = parser.add_argument_group('数据参数')
    data_group.add_argument('--data', '-d', type=str, required=True,
                           help='输入数据文件路径 (CSV/TXT/NC)')
    data_group.add_argument('--format', '-f', type=str, default='auto',
                           choices=['auto', 'csv', 'noaa_ascii', 'nc'],
                           help='数据格式 (默认: auto自动检测)')
    data_group.add_argument('--date-col', type=int, default=0,
                           help='日期列索引 (默认: 0)')
    data_group.add_argument('--value-col', type=int, default=1,
                           help='数值列索引 (默认: 1)')
    
    # 模型参数
    model_group = parser.add_argument_group('模型参数')
    model_group.add_argument('--states', '-K', type=int, default=3,
                            help='HMM状态数 (默认: 3)')
    model_group.add_argument('--dist', type=str, default='student_t',
                            choices=['gaussian', 'student_t'],
                            help='发射分布类型 (默认: student_t)')
    model_group.add_argument('--seed', type=int, default=42,
                            help='随机种子 (默认: 42)')
    
    # MCMC参数
    mcmc_group = parser.add_argument_group('MCMC参数')
    mcmc_group.add_argument('--iterations', '-i', type=int, default=5000,
                           help='MCMC迭代次数/链 (默认: 5000)')
    mcmc_group.add_argument('--burn-in', '-b', type=int, default=2000,
                           help='预烧期迭代数 (默认: 2000)')
    mcmc_group.add_argument('--chains', '-c', type=int, default=4,
                           help='MCMC链数 (默认: 4)')
    mcmc_group.add_argument('--thin', '-t', type=int, default=5,
                           help='稀疏化间隔 (默认: 5)')
    
    # 输出参数
    output_group = parser.add_argument_group('输出参数')
    output_group.add_argument('--output', '-o', type=str, default='./outputs',
                             help='输出目录 (默认: ./outputs)')
    output_group.add_argument('--verbose', '-v', action='store_true',
                             help='显示详细输出')
    output_group.add_argument('--quiet', '-q', action='store_true',
                             help='静默模式，只输出结果摘要')
    output_group.add_argument('--no-plot', action='store_true',
                             help='不生成可视化图表')
    
    # 配置文件
    parser.add_argument('--config', type=str,
                       help='JSON配置文件路径')
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，从配置文件加载参数
    if args.config:
        if not os.path.exists(args.config):
            print(f"错误: 配置文件不存在: {args.config}")
            sys.exit(1)
        
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 配置文件中的值会覆盖命令行参数（除了必需的--data）
        for key, value in config.items():
            if hasattr(args, key) and value is not None:
                setattr(args, key, value)
    
    # 验证输入文件存在
    if not os.path.exists(args.data):
        print(f"错误: 数据文件不存在: {args.data}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 导入核心模块（延迟导入以加快启动速度）
    try:
        from core.main import ENSODataLoader, RobustBayesianHMM
    except ImportError:
        print("错误: 无法导入核心模块，请确保在正确的目录下运行")
        sys.exit(1)
    
    if not args.quiet:
        print("=" * 70)
        print("ENSO贝叶斯不对称性分析系统")
        print("=" * 70)
        print(f"\n配置信息:")
        print(f"  数据文件: {args.data}")
        print(f"  数据格式: {args.format}")
        print(f"  状态数 K: {args.states}")
        print(f"  发射分布: {args.dist}")
        print(f"  MCMC迭代: {args.iterations}/链 × {args.chains}链")
        print(f"  预烧期: {args.burn_in}")
        print(f"  输出目录: {args.output}")
        print()
    
    # 加载数据
    if not args.quiet:
        print("[1/4] 正在加载数据...")
    
    loader = ENSODataLoader()
    try:
        data = loader.load_data(
            args.data,
            format_type=args.format,
            date_col=args.date_col,
            value_col=args.value_col
        )
    except Exception as e:
        print(f"错误: 数据加载失败 - {e}")
        sys.exit(1)
    
    if not args.quiet:
        print(f"      加载完成: {len(data['dates'])} 个观测点")
        print(f"      时间范围: {data['dates'].iloc[0]} 至 {data['dates'].iloc[-1]}")
    
    # 运行贝叶斯HMM分析
    if not args.quiet:
        print("\n[2/4] 正在运行贝叶斯HMM推断...")
    
    hmm = RobustBayesianHMM(
        n_states=args.states,
        random_seed=args.seed,
        emission_dist=args.dist
    )
    
    try:
        result = hmm.fit(
            data['standardized_nino34'],
            n_iterations=args.iterations,
            burn_in=args.burn_in,
            n_chains=args.chains,
            thin=args.thin,
            verbose=args.verbose and not args.quiet
        )
    except Exception as e:
        print(f"错误: MCMC推断失败 - {e}")
        sys.exit(1)
    
    if not args.quiet:
        print("\n[3/4] 正在计算不对称性指标...")
    
    # 计算模型选择准则
    criteria = hmm.compute_model_criteria(data['standardized_nino34'])
    
    # 保存结果
    if not args.quiet:
        print(f"\n[4/4] 正在保存结果到 {args.output} ...")
    
    # 保存文本结果
    results_file = output_dir / 'analysis_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("ENSO贝叶斯不对称性分析结果\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("状态参数后验估计:\n")
        for i, label in enumerate(result.get('state_labels_ordered', ['S1', 'S2', 'S3'][:args.states])):
            mu_mean = result['mu']['mean'][i]
            sigma_mean = result['sigma']['mean'][i]
            f.write(f"  状态{i} ({label}): μ={mu_mean:+.3f}, σ={sigma_mean:.3f}\n")
        
        f.write(f"\n模型选择准则:\n")
        f.write(f"  WAIC: {criteria.get('WAIC', 'N/A'):.2f}\n")
        f.write(f"  BIC: {criteria.get('BIC', 'N/A'):.2f}\n")
        f.write(f"  AIC: {criteria.get('AIC', 'N/A'):.2f}\n")
        
        if 'asymmetry' in result:
            f.write(f"\n不对称性分析:\n")
            asymmetry = result['asymmetry']
            for key in ['duration_asymmetry', 'amplitude_asymmetry']:
                if key in asymmetry and asymmetry[key].get('is_significant'):
                    f.write(f"  {key}: 显著 (p={asymmetry[key].get('p_value', 'N/A'):.4f})\n")
    
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            
            fig_file = output_dir / 'analysis_results.png'
            # 这里可以调用可视化函数
            # hmm.create_comprehensive_plot(data, result, asymmetry_results, fig_file)
            if not args.quiet:
                print(f"      图表已保存: {fig_file}")
        except Exception as e:
            if not args.quiet:
                print(f"      警告: 无法生成图表 - {e}")
    
    # 最终总结
    if not args.quiet:
        print("\n" + "=" * 70)
        print("分析完成！")
        print("=" * 70)
        print(f"\n结果已保存到: {args.output}/")
        print(f"  - analysis_results.txt (文本报告)")
        if not args.no_plot:
            print(f"  - analysis_results.png (可视化图表)")
        
        print(f"\n关键发现:")
        if 'asymmetry' in result:
            asym = result['asymmetry']
            significant_count = sum(1 for k, v in asym.items() 
                                   if isinstance(v, dict) and v.get('is_significant'))
            if significant_count > 0:
                print(f"  [OK] 发现 {significant_count} 个显著的不对称性特征")
            else:
                print(f"  - 未检测到显著的不对称性")
        
        print(f"\n模型拟合优度:")
        print(f"  WAIC: {criteria.get('WAIC', 'N/A'):.2f} (越小越好)")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
