# ENSO Bayesian HMM Analysis System - Core Package

## Overview

This is the core package containing all essential code for running the ENSO Bayesian HMM analysis system. This folder is self-contained and can be distributed independently.

## Contents

```
main/
├── core/                        # Core Algorithm Implementation
│   ├── modules/
│   │   ├── types.py            # Type definitions (enums, dataclasses, aliases)
│   │   ├── interfaces.py       # Abstract base classes and protocols
│   │   ├── data_loader.py      # Data loading and preprocessing
│   │   ├── sampler.py          # MCMC Gibbs sampling engine
│   │   ├── hmm.py              # HMM model with factory/builder patterns
│   │   └── __init__.py         # Public API exports
│   ├── main.py                 # Original v3.x implementation (backward compatible)
│   └── optimized_core.py       # Numba JIT-accelerated algorithms
│
├── config.json                  # Configuration file
├── requirements.txt             # Python dependencies
├── cli.py                       # Command-line interface entry point
└── README.md                    # This file
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from core.modules import RobustBayesianHMM; print('OK: Core modules loaded successfully')"
```

### Command Line Usage

```bash
# Run with default configuration
python cli.py --config config.json

# Custom parameters
python cli.py --data your_data.csv \
              --states 3 \
              --iterations 5000 \
              --burn_in 2000 \
              --chains 4

# Help information
python cli.py --help
```

### Python API Usage

#### Basic Example

```python
import numpy as np
from core.modules import (
    RobustBayesianHMM,
    ENSODataLoader,
    EmissionDistribution
)

# Step 1: Load data
loader = ENSODataLoader()
data = loader.load('nino34.csv')

# Step 2: Create and fit model
model = RobustBayesianHMM.create(
    n_states=3,
    emission_dist=EmissionDistribution.STUDENT_T,
    random_seed=42
)

posterior = model.fit(data.standardized_nino34)

# Step 3: Generate forecast
forecast = model.predict(
    data.standardized_nino34,
    n_ahead=12,
    n_scenarios=1000
)

print(f"Forecast mean: {forecast.mean}")
print(f"Confidence intervals available: {list(forecast.confidence_bands.keys())}")
```

#### Advanced Configuration with Builder Pattern

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
params = model.get_parameters()
print(f"Fitted state means: {params.mu}")
print(f"Transition matrix:\n{params.transition_matrix}")
```

#### Using Different Data Formats

```python
loader = ENSODataLoader()

# CSV format (default)
data_csv = loader.load('nino34.csv')

# NOAA ASCII format
data_noaa = loader.load('noaa_data.dat')  # Auto-detected

# NetCDF format (requires netCDF4 package)
data_netcdf = loader.load('climate.nc')   # Auto-detected

# With preprocessing options
data_processed = loader.preprocess(
    data_csv,
    remove_trend=True,
    remove_seasonal_cycle=True,
    standardize=True
)
```

## Configuration Reference

All parameters can be set via `config.json` or passed as keyword arguments:

### Model Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_states` | int | 3 | Number of hidden states (K) |
| `emission_distribution` | str | 'student_t' | Emission distribution type ('gaussian' or 'student_t') |
| `random_seed` | int | 42 | Random seed for reproducibility |
| `use_optimized_core` | bool | True | Enable Numba JIT acceleration |

### MCMC Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_iterations` | int | 5000 | Total MCMC iterations per chain |
| `burn_in` | int | 2000 | Burn-in period (samples to discard) |
| `n_chains` | int | 4 | Number of parallel chains |
| `thinning_interval` | int | 5 | Store every nth sample |
| `store_samples` | bool | True | Store posterior samples in memory |

### Data Preprocessing Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `remove_trend` | bool | False | Remove linear trend from data |
| `remove_seasonal_cycle` | bool | False | Remove seasonal component |
| `standardize` | bool | True | Z-score normalization |

See `config.json` for complete configuration example.

## Module Documentation

### types.py - Type System Foundation

Provides strongly-typed data structures for the entire system:

**Enumerations**:
- `EmissionDistribution`: GAUSSIAN, STUDENT_T
- `ModelCriteria`: BIC, AIC, AICC, WAIC, LPPD
- `ConvergenceStatus`: CONVERGED, NOT_CONVERGED, WARNING, INSUFFICIENT_DATA

**Dataclasses**:
- `HMMParameters`: Complete HMM parameter set (mu, sigma, nu, P, pi)
- `MCMCConfig`: MCMC configuration with validation
- `PosteriorSummary`: Inference results container
- `DataContainer`: Standardized data wrapper
- `ValidationResult`: Error/warning collection

**Usage Example**:
```python
from core.modules.types import HMMParameters, MCMCConfig
import numpy as np

params = HMMParameters(
    n_states=3,
    mu=np.array([-1.0, 0.0, 1.0]),
    sigma=np.array([0.5, 0.5, 0.5])
)
# Automatic dimension validation included
```

### interfaces.py - Abstract Contracts

Defines interfaces for all major components following Dependency Inversion Principle:

**Key Interfaces**:
- `BaseHMMModel`: Factory method + fit/predict API
- `BaseMCMCSampler`: Sampling algorithm abstraction
- `BaseDataLoader`: Multi-format loading strategy
- `BaseDistribution`: Probability distribution interface

### data_loader.py - Data Management

Implements Strategy Pattern for flexible data loading:

**Supported Formats**:
- CSV (comma-separated values)
- NOAA ASCII (year month value format)
- NetCDF (climate data format, optional dependency)

**Features**:
- Automatic format detection by file extension
- Comprehensive data validation
- Preprocessing pipeline (detrending, deseasonalizing, standardization)
- Missing value imputation (linear interpolation)

**Example**:
```python
from core.modules.data_loader import ENSODataLoader

loader = ENSODataLoader()
data = loader.load('your_file.csv')

# Validate data quality
validation = loader.validate(data)
if not validation.is_valid:
    print("Errors:", validation.errors)
    
# Apply custom preprocessing
processed = loader.preprocess(
    data,
    remove_trend=True,
    standardize=True
)
```

### sampler.py - MCMC Engine

Complete Gibbs sampling implementation with production-grade features:

**Components**:
- `GibbsSampler`: Main sampler class
- `ChainState`: Immutable chain state container
- Convergence diagnostics (R-hat, ESS)

**Features**:
- Full Gibbs conditional posteriors
- Independent RandomState per chain
- Configurable burn-in and thinning
- Optional sample storage modes (full/sparse/online)
- Parallel chain execution support

### hmm.py - Core Model

Main model implementation using multiple design patterns:

**Creation Methods**:

1. **Factory Method** (Recommended):
```python
model = RobustBayesianHMM.create(n_states=3, emission_dist='student_t')
```

2. **Builder Pattern** (Complex configs):
```python
model = (HMMBuilder()
    .with_n_states(3)
    .with_mcmc_iterations(10000)
    .build())
```

3. **Traditional Constructor** (Backward compatible):
```python
model = RobustBayesianHMM(n_states=3, random_seed=42)
```

**Distribution Strategies**:
```python
from core.modules.hmm import DistributionFactory

# Built-in distributions
gaussian = DistributionFactory.create('gaussian')
student_t = DistributionFactory.create('student_t')

# Register custom distribution
class MyDistribution:
    name = 'custom'
    @staticmethod
    def log_pdf(x, mu, sigma, **kwargs): ...
    @staticmethod
    def sample(mu, sigma, size, rng=None, **kwargs): ...

DistributionFactory.register('custom', MyDistribution)
```

## Error Handling

The system provides comprehensive error handling with specific messages:

### Input Validation Rules

1. **Type checking**: All inputs validated against expected types
2. **Dimension consistency**: Array shapes verified
3. **Value ranges**: Parameters checked for valid ranges
4. **Missing data**: NaN/Inf detection and reporting
5. **Configuration sanity**: Logical constraints enforced

**Example Output**:
```
[ERROR] n_states must be >= 2, got 1
[WARNING] Data length is short (18 months), recommend at least 24 months
[ERROR] File not found: /path/to/nonexistent.csv
```

### Exception Types

- `ValueError`: Invalid parameter values
- `TypeError`: Wrong input types
- `FileNotFoundError`: Missing data files
- `RuntimeError`: Model state errors (e.g., predict before fit)

## Backward Compatibility

This package maintains 100% backward compatibility with v3.x API:

**v3.x code still works unchanged**:
```python
# Old-style usage (still valid)
from core.main import RobustBayesianHMM
model = RobustBayesianHMM(n_states=3)
result = model.fit(data, n_iterations=5000, burn_in=2000)
```

**New v4.0 features are additive**:
```python
# New features (optional adoption)
from core.modules import RobustBayesianHMM
model = RobustBayesianHMM.create(n_states=3)  # Factory method
params = model.get_parameters()               # Typed access
```

## Performance Notes

### Optimization Status

- **Numba JIT**: Enabled by default (10-50x speedup)
- **Fallback**: Pure NumPy implementation if Numba unavailable
- **Memory**: Configurable storage strategies

### Typical Performance

On standard hardware (i7 CPU, 16GB RAM):
- Small dataset (120 months, K=3, 5000 iter, 4 chains): ~5-15 seconds
- Medium dataset (600 months, K=3, 5000 iter, 4 chains): ~30 seconds - 2 minutes
- Large dataset (2000+ months): Use reduced iterations or enable sparse storage

### Memory Optimization

For large datasets, configure memory-efficient options:
```python
config = MCMCConfig(
    store_samples=False,  # Only compute summary statistics
    compression='float16'  # Reduce precision of stored samples
)
```

## Troubleshooting

### Common Issues

**Issue**: "ModuleNotFoundError: No module named 'core'"
**Solution**: Ensure you're running from the `main/` directory or add it to PYTHONPATH

**Issue**: "Numba not found" warning
**Solution**: Either install Numba (`pip install numba`) or accept slower pure-NumPy performance

**Issue**: Memory error with large datasets
**Solution**: Reduce `n_iterations`, increase `thinning_interval`, or set `store_samples=False`

**Issue**: Poor convergence (high R-hat)
**Solution**: Increase `burn_in`, use more chains (`n_chains=8`), or check data quality

## Dependencies

### Required
- Python >= 3.8
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.10.0

### Recommended
- numba >= 0.57.0 (for JIT acceleration)
- matplotlib >= 3.7.0 (for visualization)

### Optional
- netCDF4 >= 1.6.0 (for NetCDF support)
- plotly >= 5.15.0 (for interactive plots)
- pytest >= 7.0.0 (for running test suite)

## Related Documentation

- **Project README**: [../README.md](../README.md) - Project overview and quick start
- **Auxiliary Resources**: [../load/README.md](../load/README.md) - Tests, reports, tools
- **Refactoring Report**: [../load/report/REFACTORING_REPORT.md](../load/report/REFACTORING_REPORT.md) - Architecture details
- **Performance Report**: [../load/report/V3_PERFORMANCE_OPTIMIZATION_REPORT.md](../load/report/V3_PERFORMANCE_OPTIMIZATION_REPORT.md) - Optimization details

## Version Information

- **Package Version**: 4.0 Professional Refactored
- **API Compatibility**: 100% backward compatible with v3.x
- **Python Support**: 3.8, 3.9, 3.10, 3.11, 3.12
- **Last Updated**: 2026-05-18

---

For questions or issues, please refer to the main project README or contact: h2078309440@163.com
