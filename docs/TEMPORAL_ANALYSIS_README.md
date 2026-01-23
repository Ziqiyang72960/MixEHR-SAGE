# Temporal Analysis in MixEHR-SAGE

This document explains how to use the temporal analysis features of MixEHR-SAGE for longitudinal EHR data analysis.

## Table of Contents
1. [Overview](#overview)
2. [Input Data Format](#input-data-format)
3. [Quick Start](#quick-start)
4. [Temporal Theta Computation](#temporal-theta-computation)
5. [Progression Models](#progression-models)
6. [LSTM Temporal Model](#lstm-temporal-model)
7. [Output Format](#output-format)
8. [Example Workflow](#example-workflow)
9. [API Reference](#api-reference)

## Overview

The temporal analysis module extends MixEHR-SAGE to support longitudinal patient data by:

1. **Temporal Bucketing**: Organizing medical records into discrete time steps (yearly, monthly, quarterly, or by visit)
2. **Per-Time Topic Mixtures (θ_t)**: Computing topic distributions at each time point
3. **Disease Progression Modeling**: Learning temporal dynamics with Markov models or LSTM
4. **Risk Prediction**: Predicting future disease states based on learned dynamics

### Key Features
- **No Data Leakage**: θ_t at time t only depends on records up to time t (cumulative approach)
- **Multiple Bucketing Strategies**: Yearly, monthly, quarterly, or visit-based
- **Backward Compatible**: Non-temporal data still works with the original pipeline
- **Modular Design**: Each component can be used independently

## Input Data Format

### Temporal Data CSV

The temporal data file should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| `SUBJECT_ID` | Patient identifier | `patient_001` |
| `code` | Medical code (ICD, medication, procedure) | `E11.9`, `C03AB01` |
| `timestamp` | Date or visit index | `2015-03-15` or `1` |
| `modality` | Type of medical code | `icd`, `med`, `opcs` |

**Example:**
```csv
SUBJECT_ID,code,timestamp,modality
patient_001,E11.9,2015-03-15,icd
patient_001,I10,2015-03-15,icd
patient_001,C03AB01,1,med
patient_001,E11.9,2016-01-10,icd
patient_002,J44.1,2014-05-10,icd
patient_002,C09AA02,0,med
```

### Timestamp Formats

- **Date-based** (`yearly`, `monthly`, `quarterly`): Use `YYYY-MM-DD` format
- **Visit-based** (`visit`): Use integer visit indices (0, 1, 2, ...)

## Quick Start

### Run the Demo

The fastest way to get started is to run the demo:

```bash
python run_temporal.py demo --output ./demo_results/
```

This will:
1. Generate sample temporal data
2. Create a temporal corpus
3. Generate synthetic theta sequences
4. Train a Markov transition model
5. Predict disease risk
6. Analyze progression patterns

### Full Pipeline

```bash
# Step 1: Compute temporal theta sequences (single file)
python run_temporal.py compute_theta \
    --data ./data/temporal_data.csv \
    --model-path ./results/ \
    --output ./results/temporal/ \
    --bucket-type yearly

# Or use separate modality files
python run_temporal.py compute_theta \
    --icd ./data/temporal_icd.csv \
    --med ./data/temporal_med.csv \
    --opcs ./data/temporal_opcs.csv \
    --model-path ./results/ \
    --output ./results/temporal/ \
    --bucket-type yearly

# Step 2: Train Markov progression model
python run_temporal.py train_markov \
    --theta ./results/temporal/theta_sequences.csv \
    --output ./results/temporal/markov_model.pkl

# Step 3: Predict disease risk
python run_temporal.py predict_risk \
    --model ./results/temporal/markov_model.pkl \
    --patient ./patient_theta.csv \
    --horizon 3
```

## Temporal Theta Computation

### Input Data Formats

**Option 1: Single file with modality column**
```csv
SUBJECT_ID,code,timestamp,modality
patient_001,E11.9,2015-03-15,icd
patient_001,A10BA02,2015-03-15,med
```

**Option 2: Separate files per modality**

Each modality file contains:
```csv
SUBJECT_ID,code,timestamp
patient_001,E11.9,2015-03-15
patient_001,I10,2016-01-10
```

### Bucketing Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `yearly` | Group by calendar year | Long-term trends |
| `monthly` | Group by month | Seasonal patterns |
| `quarterly` | Group by quarter | Medium-term analysis |
| `visit` | Group by visit index | Visit-level dynamics |

### Python API

```python
from temporal_corpus import TemporalCorpus
from MixEHR_SAGE import MixEHR_SAGE

# Option 1: Load from single file
corpus = TemporalCorpus.from_file(
    './data/temporal_data.csv',
    bucket_type='yearly'
)

# Option 2: Load from separate modality files
corpus = TemporalCorpus.from_modality_files(
    {
        'icd': './data/temporal_icd.csv',
        'med': './data/temporal_med.csv',
        'opcs': './data/temporal_opcs.csv'
    },
    bucket_type='yearly'
)

# Set vocabulary mappings
corpus.set_vocab_mappings(vocab_mappings, modality_list)

# Compute theta sequences
theta_sequences = corpus.compute_theta_sequences(
    model,
    num_iterations=10,
    use_cumulative=True,  # Prevents data leakage
    method='variational'
)

# Export results
corpus.export_theta_sequences('./results/theta_sequences.csv', format='csv')
```

### Cumulative vs. Instantaneous

- **Cumulative (default)**: θ_t uses all records from time 0 to t
  - Prevents data leakage
  - Better for risk prediction
  
- **Instantaneous**: θ_t uses only records from time t
  - Captures time-specific patterns
  - Useful for detecting changes

## Progression Models

### First-Order Markov Model

The Markov model learns state transitions P(θ_t | θ_{t-1}).

**Discretization Methods:**
- `dominant`: Assign state based on highest probability topic
- `soft`: Use full probability distribution for soft counts
- `threshold`: Assign to topics above a threshold

```python
from temporal_models import MarkovTransitionModel

# Create and fit model
markov = MarkovTransitionModel(
    num_topics=K,
    discretization='soft'
)
markov.fit(theta_sequences, smoothing=1.0)

# Predict next state
next_dist = markov.predict_next_state(current_theta)

# Predict disease risk at horizon H
risk = markov.predict_disease_risk(current_theta, horizon=3)

# Get stationary distribution
stationary = markov.get_stationary_distribution()
```

### Output: Transition Matrix

The transition matrix `P` has shape `(K, K)` where:
- `P[i,j]` = probability of transitioning from state i to state j
- Row i sums to 1

## LSTM Temporal Model

The LSTM model learns temporal dynamics for the topic prior η.

### Architecture

```
Input: BOW sequence (batch, T, V)
   ↓
Linear projection (V → hidden_size)
   ↓
LSTM (hidden_size, num_layers)
   ↓
μ_t = Linear(hidden + K → K)
log(σ_t) = Linear(hidden + K → K)
   ↓
η_t ~ N(μ_t, σ_t²)
   ↓
α_t = softplus(η_t)  # Topic prior
```

### Training

```bash
python run_temporal.py train_lstm \
    --data ./data/temporal_data.csv \
    --model-path ./results/ \
    --output ./results/temporal_lstm.pt \
    --hidden-size 200 \
    --num-layers 3 \
    --num-epochs 20
```

### Python API

```python
from temporal_models import TemporalLSTMModel, TemporalMixEHR

# Create LSTM model
lstm = TemporalLSTMModel(
    vocab_size=V,
    num_topics=K,
    hidden_size=200,
    num_layers=3
)

# Create temporal MixEHR
temporal_model = TemporalMixEHR(base_model, lstm)

# Train
loss_history = temporal_model.train_temporal(
    corpus, vocab_mappings, modality_list,
    num_epochs=20, batch_size=32
)

# Infer with temporal prior
theta_sequences = temporal_model.infer_temporal_theta(
    corpus, vocab_mappings, modality_list
)
```

## Output Format

### Theta Sequences CSV

```csv
patient_id,time_index,bucket_type,theta_0,theta_1,...,theta_K
patient_001,2015,yearly,0.123,0.045,...,0.032
patient_001,2016,yearly,0.156,0.038,...,0.028
patient_002,2014,yearly,0.089,0.112,...,0.054
```

### Progression Analysis CSV

```csv
patient_id,topic_idx,topic_name,num_time_steps,initial_prob,final_prob,max_prob,min_prob,mean_prob,std_prob,trend,prob_change
patient_001,0,topic_0,3,0.123,0.189,0.189,0.123,0.156,0.027,increasing,0.066
patient_001,1,topic_1,3,0.045,0.038,0.045,0.032,0.038,0.005,stable,-0.007
```

### Risk Prediction JSON

```json
{
  "horizon": 3,
  "current_theta": [0.123, 0.045, ...],
  "next_state_distribution": [0.142, 0.051, ...],
  "risk": {
    "topic_0": 0.167,
    "topic_1": 0.058,
    "max_risk_topic": 5,
    "max_risk": 0.234,
    "entropy": 2.156
  }
}
```

## Example Workflow

### Complete Analysis Pipeline

```python
import pickle
import torch
from corpus import Corpus
from MixEHR_SAGE import MixEHR_SAGE
from temporal_corpus import TemporalCorpus
from temporal_models import MarkovTransitionModel, analyze_disease_progression

# 1. Load trained MixEHR model
corpus = Corpus.read_corpus_from_directory('./store/')
seeds = torch.load('./phecode_mapping/seed_topic_matrix.pt')
model = MixEHR_SAGE.load_trained_model(
    './results/', corpus, seeds, corpus.modalities
)

# 2. Load vocabulary mappings
with open('./mapping/icd_vocab_ids.pkl', 'rb') as f:
    icd_vocab = pickle.load(f)
vocab_mappings = {'icd': icd_vocab}

# 3. Create temporal corpus
temporal_corpus = TemporalCorpus.from_file(
    './data/temporal_data.csv',
    bucket_type='yearly'
)
temporal_corpus.set_vocab_mappings(vocab_mappings, corpus.modalities)

# 4. Compute theta sequences
theta_sequences = temporal_corpus.compute_theta_sequences(
    model, num_iterations=10, use_cumulative=True
)

# 5. Export for downstream use
temporal_corpus.export_theta_sequences('./results/theta_sequences.csv')

# 6. Train progression model
markov = MarkovTransitionModel(num_topics=model.K)
markov.fit(theta_sequences)
markov.save('./results/markov_model.pkl')

# 7. Predict risk for new patient
patient_theta = theta_sequences['patient_001'][-1]
risk = markov.predict_disease_risk(patient_theta, horizon=3)
print(f"3-step risk: {risk}")

# 8. Analyze progression patterns
progression_df = analyze_disease_progression(theta_sequences)
progression_df.to_csv('./results/progression_analysis.csv')
```

### Disease Risk Monitoring

```python
# Monitor disease risk over time
import matplotlib.pyplot as plt

patient_id = 'patient_001'
theta_seq = theta_sequences[patient_id].numpy()

# Plot topic probabilities over time
plt.figure(figsize=(10, 6))
for k in range(min(5, theta_seq.shape[1])):  # Top 5 topics
    plt.plot(theta_seq[:, k], label=f'Topic {k}')
plt.xlabel('Time Step')
plt.ylabel('Topic Probability')
plt.legend()
plt.title(f'Disease Trajectory - {patient_id}')
plt.savefig('./results/trajectory.png')
```

## API Reference

### TemporalCorpus

```python
class TemporalCorpus:
    """Temporal corpus for longitudinal EHR data."""
    
    @classmethod
    def from_file(cls, file_path: str, bucket_type: str = 'yearly', ...) -> 'TemporalCorpus':
        """Load temporal corpus from file."""
        
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, bucket_type: str = 'yearly', ...) -> 'TemporalCorpus':
        """Create temporal corpus from DataFrame."""
        
    def set_vocab_mappings(self, vocab_mappings: Dict, modality_list: List[str]):
        """Set vocabulary mappings."""
        
    def compute_theta_sequences(self, model, num_iterations: int = 10, 
                                use_cumulative: bool = True,
                                method: str = 'variational') -> Dict[str, torch.Tensor]:
        """Compute per-patient per-time theta sequences."""
        
    def export_theta_sequences(self, output_path: str, format: str = 'csv'):
        """Export theta sequences to file."""
```

### MarkovTransitionModel

```python
class MarkovTransitionModel:
    """First-order Markov model for disease progression."""
    
    def __init__(self, num_topics: int, discretization: str = 'dominant'):
        """Initialize model."""
        
    def fit(self, theta_sequences: Dict[str, torch.Tensor], smoothing: float = 1.0):
        """Fit transition matrix from data."""
        
    def predict_next_state(self, current_theta, return_distribution: bool = True):
        """Predict next state distribution."""
        
    def predict_disease_risk(self, current_theta, horizon: int = 1,
                            target_topics: List[int] = None) -> Dict:
        """Predict disease risk at future horizon."""
        
    def get_stationary_distribution(self) -> np.ndarray:
        """Compute stationary distribution."""
```

### TemporalLSTMModel

```python
class TemporalLSTMModel(nn.Module):
    """LSTM-based temporal model for dynamic topic prior."""
    
    def __init__(self, vocab_size: int, num_topics: int,
                 hidden_size: int = 200, num_layers: int = 3,
                 dropout: float = 0.0, delta: float = 0.01):
        """Initialize LSTM model."""
        
    def forward(self, bow_sequence: torch.Tensor, 
                alpha_prev: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass: returns μ and log(σ) for variational distribution."""
        
    def sample_eta(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """Sample η using reparameterization trick."""
        
    def get_alpha(self, eta: torch.Tensor) -> torch.Tensor:
        """Convert η to topic prior α."""
```

## Troubleshooting

### Common Issues

1. **"vocab_mappings not set"**
   - Call `corpus.set_vocab_mappings(vocab_mappings, modality_list)` before computing theta

2. **"No valid patient data found"**
   - Check that codes in your data match the vocabulary
   - Ensure modality names match those in the trained model

3. **CUDA out of memory**
   - Reduce batch size with `--batch-size`
   - Use CPU with `CUDA_VISIBLE_DEVICES=""`

4. **Theta sequences all similar**
   - Increase `--num-iterations`
   - Check if data has enough records per time bucket

### Performance Tips

- Use `method='variational'` for faster inference
- Increase `--batch-size` for GPU efficiency
- Use `--bucket-type visit` for sparse data
- Pre-compute vocabulary mappings once and reuse

## Citation

If you use the temporal analysis features, please cite:

```bibtex
@article{mixehr-sage-temporal,
  title={Temporal Analysis with MixEHR-SAGE},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This code is released under the same license as MixEHR-SAGE.
