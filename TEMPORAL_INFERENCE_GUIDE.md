# Temporal Inference with LSTM in MixEHR-SAGE

## Overview

This document describes the temporal inference component that has been added to MixEHR-SAGE. The temporal component uses an LSTM network to model how patient populations evolve over time (e.g., across different age groups).

## What Was Changed

### 1. Fixed Variable Name Conflict
- **Old**: `self.eta = 0.1` (scalar hyperparameter)
- **New**: `self.alpha_prior = 0.1` (renamed to avoid conflict with temporal eta)
- **Impact**: All references updated in `SCVB0_guided()`, `SCVB0_unguided()`, and `get_elbo()`

### 2. Added Temporal Parameters
- `enable_temporal` (bool): Flag to enable/disable temporal inference
- `num_time_steps` (int): Number of time bins (default: 10)
- `self.T`: Stores number of time steps

### 3. Uncommented and Fixed LSTM Architecture
```python
if self.enable_temporal:
    # Temporal parameters
    self.T = num_time_steps
    self.eta = torch.rand(self.T, self.K)  # T x K temporal hyperparameters
    
    # LSTM network
    self.q_eta_map = nn.Linear(self.V[self.guided_modality], self.eta_hidden_size)
    self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers)
    self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K)
    self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K)
```

**Key Fixes**:
- Used `self.V[self.guided_modality]` instead of undefined `self.V`
- Added proper parameter initialization
- Included optimizer setup for LSTM parameters

### 4. Added Helper Methods

#### `alpha_softplus_act()`
Applies softplus activation to eta to ensure positive Dirichlet parameters:
```python
alpha = F.softplus(eta)  # Ensures alpha > 0
```

#### `reparameterize(mu, logvar)`
Implements reparameterization trick for sampling:
```python
z = mu + std * epsilon, where epsilon ~ N(0, 1)
```

#### `encode_temporal_sequence(time_step_data)`
Prepares vocabulary distributions for LSTM input.

#### `infer_eta_variational(time_step_data)`
Main temporal inference method:
1. Encodes temporal sequence through linear layer
2. Passes through LSTM
3. Concatenates with previous eta for autoregression
4. Computes variational parameters (mean and log-variance)
5. Samples eta using reparameterization trick

#### `compute_temporal_kl(mu_eta, logvar_eta)`
Computes KL divergence between variational distribution and prior:
```
KL(q(eta|mu,sigma) || p(eta|0,delta))
```

#### `generate_temporal_word_distributions()`
Placeholder for creating time-varying word distributions.

## Temporal Data Utilities

### `temporal_utils.py`
Provides utilities for temporal data generation and preprocessing:

#### `TemporalDataGenerator`
- Creates age bins for temporal modeling
- Bins patients by age
- Aggregates word distributions per time bin
- Generates synthetic temporal data for testing

#### `TemporalSequencePreprocessor`
- Creates sliding windows for sequence prediction
- Applies smoothing to temporal sequences
- Interpolates missing time bins

## Usage

### Basic Usage (Without Temporal Inference)
```python
from MixEHR_SAGE import MixEHR_SAGE

model = MixEHR_SAGE(
    corpus=corpus,
    seeds_topic_matrix=seeds,
    modality_list=['icd', 'opcs'],
    enable_temporal=False  # Default
)
```

### With Temporal Inference
```python
from MixEHR_SAGE import MixEHR_SAGE
from temporal_utils import TemporalDataGenerator

# Setup temporal data
gen = TemporalDataGenerator(num_time_steps=10, min_age=0, max_age=100)
time_bins = gen.create_temporal_corpus_from_ages(corpus, patient_ages)

# Initialize model with temporal inference
model = MixEHR_SAGE(
    corpus=corpus,
    seeds_topic_matrix=seeds,
    modality_list=['icd', 'opcs'],
    enable_temporal=True,
    num_time_steps=10
)

# Generate temporal word distributions
time_word_dist = gen.aggregate_word_distributions_by_time(
    corpus, time_bins, modality=0
)

# Perform temporal inference
eta_samples, mu_eta, logvar_eta = model.infer_eta_variational(time_word_dist)

# Compute alpha (Dirichlet parameters)
alpha = model.alpha_softplus_act()

# Compute temporal KL divergence
kl_loss = model.compute_temporal_kl(mu_eta, logvar_eta)
```

### Running the Example
```bash
python example_temporal.py
```

This will:
1. Generate synthetic temporal EHR data
2. Initialize MixEHR-SAGE with temporal inference
3. Perform LSTM-based variational inference
4. Visualize temporal topic evolution

## Architecture

### Temporal Model Flow

```
Time-binned Data → Aggregated Word Distributions (T x V)
                                ↓
                    Linear Mapping (V → hidden_size)
                                ↓
                    LSTM (T timesteps)
                                ↓
                    Concatenate with previous eta
                                ↓
            Linear layers (mu and log-variance)
                                ↓
                Reparameterization trick
                                ↓
                    Sampled eta (T x K)
                                ↓
                Softplus → alpha (T x K)
                                ↓
            Used in Dirichlet prior for topics
```

### ELBO with Temporal Component

The ELBO now includes an additional term for temporal modeling:

```
ELBO = E_q[log p(w|z,β,μ,π)] 
       + E_q[log p(z|α)] - E_q[log q(z|γ)]
       - KL(q(η|μ,σ) || p(η|0,δ))  ← New temporal term
```

Where:
- `η (eta)`: Temporal hyperparameters (T x K)
- `α (alpha) = softplus(η)`: Dirichlet parameters
- `δ (delta)`: Prior variance (default: 0.01)

## Hyperparameters

### LSTM Architecture
- `eta_hidden_size`: 200 (hidden units)
- `eta_nlayers`: 3 (LSTM layers)
- `eta_dropout`: 0.0 (dropout rate)

### Optimization
- `lr`: 0.0001 (learning rate)
- `wdecay`: 1.2e-6 (weight decay)

### Temporal Prior
- `delta`: 0.01 (prior variance for eta)
- `max_logsigma_t`: 5.0 (maximum log-variance)
- `min_logsigma_t`: -5.0 (minimum log-variance)

## Known Limitations and Future Work

### Current Limitations
1. **Not Integrated with Main Inference Loop**: The temporal component is implemented but not yet integrated into the main `inference()` method
2. **Placeholder Data Generation**: `generate_temporal_word_distributions()` uses uniform distributions; needs real temporal metadata
3. **No Temporal ELBO Optimization**: KL divergence is computed but not yet used in optimization
4. **Single Modality**: Currently only uses guided modality for temporal inference

### Future Improvements
1. **Full Integration**: Add temporal inference to main training loop
2. **Real Data Support**: Use actual patient ages/timestamps
3. **Joint Optimization**: Optimize both variational and LSTM parameters together
4. **Multi-Modal Temporal**: Extend to all modalities
5. **Adaptive Time Bins**: Automatically determine optimal number of time steps
6. **Evaluation Metrics**: Add metrics for temporal topic coherence

## Testing

### Syntax Check
```bash
python -m py_compile MixEHR_SAGE.py temporal_utils.py example_temporal.py
```

### Run Tests (requires dependencies)
```bash
pip install -r requirements.txt
python temporal_utils.py  # Test data generation
python example_temporal.py  # Full demonstration
```

## References

1. Original MixEHR-SAGE paper for seed-guided topic modeling
2. Dynamic Topic Models (Blei & Lafferty, 2006) for temporal modeling
3. Amortized inference with neural networks (Kingma & Welling, 2013)

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"
Install dependencies:
```bash
pip install -r requirements.txt
```

### "Variable self.eta referenced before assignment"
Make sure `enable_temporal=True` when using temporal methods.

### "CUDA out of memory"
Reduce batch size or use CPU:
```python
device = torch.device("cpu")
```

## Contact

For issues or questions about the temporal inference implementation, please refer to:
- `TEMPORAL_ANALYSIS.md` for detailed problem analysis
- `example_temporal.py` for usage examples
- GitHub issues for the repository
