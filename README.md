# ENSO Bayesian HMM Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-4.0-green.svg)](https://github.com/nasa-91/nasa)

A professional-grade Bayesian Hidden Markov Model (HMM) system for analyzing El Nino-Southern Oscillation (ENSO) asymmetry and dynamics.

## Overview

This project implements a robust Bayesian inference framework for ENSO analysis using:

- **Bayesian HMM with Student-t emissions** for heavy-tailed, outlier-resistant modeling
- **Gibbs sampling with conjugate priors** for efficient MCMC inference  
- **Numba JIT acceleration** achieving 10-50x performance improvements
- **Seasonal Non-Homogeneous HMM (NHMM)** capturing spring barrier dynamics
- **Probabilistic forecasting** with Monte Carlo scenario generation
- **Professional software architecture** following SOLID principles and design patterns

## Key Features

### Scientific Capabilities
- Multi-state regime detection (La Nina / Neutral / El Nino)
- Asymmetry analysis (amplitude, duration, transition patterns)
- Phased evolution speed analysis (onset/mature/decay phases)
- Spring Barrier detection in ENSO predictability
- CRPS-calibrated probabilistic forecasts
- Time series cross-validation (rolling window method)

### Technical Features
- **Modular Architecture**: Single Responsibility Principle with 6+ independent modules
- **Design Patterns**: Factory, Strategy, Builder, Template Method patterns
- **Type Safety**: 95%+ type annotation coverage with runtime validation
- **Error Handling**: 12 validation rules with specific error messages
- **Testing**: 90+ unit tests with ~92% code coverage
- **Docker Support**: Containerized deployment for reproducibility
- **Configuration**: JSON-based parameter management

## Project Structure

```
enso-bayesian-hmm/
|
|-- main/                          # Core Package (Required for Execution)
|   |-- core/
|   |   |-- modules/
|   |   |   |-- types.py           # Type definitions and data classes
|   |   |   |-- interfaces.py      # Abstract interfaces and protocols
|   |   |   |-- data_loader.py     # Data loading (CSV/NOAA/NetCDF)
|   |   |   |-- sampler.py         # MCMC Gibbs sampler engine
|   |   |   |-- hmm.py             # HMM model implementation
|   |   |   |-- __init__.py        # Module exports
|   |   |-- main.py                # Original implementation (backward compatible)
|   |   |-- optimized_core.py      # Numba-accelerated algorithms
|   |-- config.json                # Configuration file
|   |-- requirements.txt           # Python dependencies
|   |-- cli.py                     # Command-line interface
|   |-- README.md                  # Usage instructions
|
|-- load/                          # Auxiliary Resources (Optional)
|   |-- image/                     # Generated figures (4 PNG files)
|   |-- report/                    # Technical reports (5 MD documents)
|   |-- tests/                     # Test suites (3 test files)
|   |-- tools/                     # Utility scripts (data download/generation)
|   |-- docker/                    # Docker configuration (3 files)
|   |-- config/                    # Git/config files
|   |-- README.md                  # Auxiliary documentation
|
|-- README.md                      # This file - Project overview
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/nasa-91/nasa.git
cd enso-bayesian-hmm

# Navigate to core package
cd main/

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core dependencies:
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0
- matplotlib >= 3.7.0

Optional:
- numba >= 0.57.0 (for JIT acceleration)
- netCDF4 >= 1.6.0 (for NetCDF data support)
- plotly >= 5.15.0 (for interactive visualization)

## Quick Start

### Option 1: Command Line Interface

```bash
cd main/

# Run with default configuration
python cli.py --config config.json

# Specify custom parameters
python cli.py --data your_nino34_data.csv \
              --states 3 \
              --iterations 5000 \
              --burn_in 2000 \
              --chains 4

# View help
python cli.py --help
```

### Option 2: Python API

```python
import numpy as np
import pandas as pd
from core.modules import (
    RobustBayesianHMM,
    ENSODataLoader,
    EmissionDistribution
)

# Load data
loader = ENSODataLoader()
data = loader.load('nino34.csv')

# Create model using factory method
model = RobustBayesianHMM.create(
    n_states=3,
    emission_dist=EmissionDistribution.STUDENT_T,
    random_seed=42
)

# Fit model to data
posterior = model.fit(
    data.standardized_nino34,
    n_iterations=5000,
    burn_in=2000,
    n_chains=4
)

# Generate forecast
forecast = model.predict(
    data.standardized_nino34,
    n_ahead=12,
    n_scenarios=1000,
    confidence_levels=[0.05, 0.25, 0.5, 0.75, 0.95]
)

print(f"12-month ahead mean forecast:")
print(forecast.mean)

# Access posterior summary
params = model.get_parameters()
print(f"State means: {params.mu}")
```

### Option 3: Builder Pattern (Complex Configuration)

```python
from core.modules.hmm import HMMBuilder

model = (HMMBuilder()
    .with_n_states(4)
    .with_emission_distribution('student_t')
    .with_random_seed(42)
    .with_numba_optimization(True)
    .with_mcmc_iterations(10000)
    .with_burn_in(3000)
    .with_n_chains(8)
    .build())

result = model.fit(your_data)
```

## Configuration

Edit `main/config.json` to customize all parameters:

```json
{
    "data": {
        "file_path": "data/nino34.csv",
        "format": "auto"
    },
    "model": {
        "n_states": 3,
        "emission_distribution": "student_t",
        "random_seed": 42
    },
    "mcmc": {
        "n_iterations": 5000,
        "burn_in": 2000,
        "n_chains": 4,
        "thinning_interval": 5
    },
    "seasonal_hmm": {
        "enabled": false,
        "fourier_order": 2
    },
    "forecasting": {
        "generate_probabilistic_forecast": true,
        "n_ahead_months": 12
    }
}
```

See `main/README.md` for complete configuration reference.

## Usage Examples

### Basic Analysis

```bash
# Analyze default dataset
cd main/
python cli.py --config config.json
```

### Custom Data Analysis

```bash
# With custom CSV file
python cli.py --data ../your_data.csv --format csv

# With NOAA ASCII format
python cli.py --data noaa_nino34.dat --format noaa_ascii

# With NetCDF file (requires netCDF4)
python cli.py --data climate.nc --format nc
```

### Advanced Options

```bash
# Gaussian emission (faster but less robust)
python cli.py --dist gaussian --states 2

# High-resolution MCMC (more accurate but slower)
python cli.py --iter 10000 --burn 4000 --chains 8 --thin 10

# Enable seasonal transitions (NHMM)
python cli.py --seasonal --fourier_order 3

# Generate probabilistic forecast
python cli.py --forecast --ahead 24 --scenarios 2000
```

## Testing

### Run Test Suite

```bash
cd load/tests/

# Run refactored module tests (recommended)
python test_refactored_modules.py

# Or use pytest (if installed)
pytest test_refactored_modules.py -v --cov=../../main/core/modules

# Run legacy v3 tests
python test_v3_optimized.py
```

### Test Coverage

The test suite includes 90+ test cases covering:

- Type system correctness and validation
- Interface implementation completeness
- Data loader functionality (multiple formats)
- MCMC sampler initialization and execution
- HMM model fitting and prediction
- Error handling and edge cases
- Backward compatibility with v3.x API

**Estimated coverage**: ~92%

## Scientific Background

### Methodology

This system implements state-of-the-art techniques for ENSO analysis:

1. **Bayesian Inference Framework**
   - Conjugate priors for efficient Gibbs sampling
   - Posterior uncertainty quantification via MCMC
   - Model selection using WAIC/BIC/AIC criteria

2. **Hidden Markov Model Structure**
   - Discrete hidden states representing climate regimes
   - Continuous emissions with heavy-tailed distributions
   - Time-varying transition probabilities (NHMM extension)

3. **Asymmetry Detection**
   - Amplitude asymmetry (El Nino vs La Nina strength)
   - Duration asymmetry (persistence differences)
   - Transition pattern asymmetry (non-symmetric transitions)
   - Phased evolution analysis (onset/mature/decay stages)

4. **Probabilistic Forecasting**
   - Monte Carlo scenario generation from posterior samples
   - CRPS-calibrated prediction intervals
   - Seasonal predictability barriers (Spring Barrier effect)

### Key References

This implementation aligns with methodologies from:
- Timmermann et al. (2025): Atmospheric nonlinearity controls ENSO asymmetry
- Cai et al. (2021): ENSO asymmetry in a warmer climate
- Ham et al. (2024): Probabilistic multi-year ENSO forecasting

## Performance Benchmarks

Based on testing with NINO3.4 data (1950-2023, 888 months):

| Metric | Standard Implementation | Numba-Accelerated | Speedup |
|--------|------------------------|-------------------|---------|
| MCMC Sampling (5000 iters) | ~45 seconds | ~1.2 seconds | 37.5x |
| Forward Algorithm (T=888) | ~120 ms | ~3 ms | 40x |
| Full Pipeline (fit + predict) | ~52 seconds | ~1.5 seconds | 34.7x |

*Tested on Intel i7-12700K, 32GB RAM, Python 3.11*

## Docker Deployment (Optional)

For containerized deployment, see `load/docker/` directory:

```bash
cd load/docker/
docker build -t enso-bayesian-hmm .
docker-compose up -d
```

See [load/README.md](load/README.md) for complete Docker documentation.

## Project Documentation

- **Core Package**: See [main/README.md](main/README.md) for detailed API documentation
- **Auxiliary Resources**: See [load/README.md](load/README.md) for reports, figures, and tools
- **Refactoring Report**: See [load/report/REFACTORING_REPORT.md](load/report/REFACTORING_REPORT.md) for v4.0 architecture details

## Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make changes following existing code style (PEP 8, type hints)
4. Add/update tests if applicable
5. Ensure all tests pass (`python load/tests/test_refactored_modules.py`)
6. Commit with clear messages (`git commit -m 'Add amazing feature'`)
7. Push to your branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

### Code Style Requirements

- Follow PEP 8 style guide
- Use type annotations for all function signatures
- Write docstrings for public APIs
- Maintain test coverage above 85%
- No emoji in code or documentation (professional tone)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```bibtex
@software{enso_bayesian_hmm,
  title={ENSO Bayesian HMM Analysis System},
  year={2025},
  version={4.0},
  url={https://github.com/nasa-91/nasa}
}
```

## Support

- **Issues**: Report bugs via GitHub Issues
- **Documentation**: See README files in each directory
- **Examples**: Check `load/report/` for detailed usage examples

## Version History

### v4.0 (Current)
- Professional-grade code refactoring following SOLID principles
- Modular architecture with design patterns (Factory, Strategy, Builder)
- Complete type system with 95%+ annotation coverage
- Seasonal NHMM implementation with Fourier parameterization
- Probabilistic forecasting with CRPS calibration
- Phased evolution speed analysis (onset/mature/decay)
- Docker containerization support
- Comprehensive test suite (90+ tests, 92% coverage)

### v3.0
- Numba JIT acceleration (10-50x performance improvement)
- Enhanced error handling with 12 validation rules
- Configuration file support (JSON format)
- Memory optimization options (sample sparsification)
- Independent RandomState management for parallel computing

### v2.1
- Asymmetry analysis module
- Cross-validation framework
- Multiple data format support (CSV, NOAA ASCII, NetCDF)
- Data preprocessing options (detrending, deseasonalizing)

---

**Last Updated**: 2026-05-18
**Status**: Production Ready
