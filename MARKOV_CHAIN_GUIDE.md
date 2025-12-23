# Markov Chain Dynamic Topic Model Implementation

## Overview

This implementation introduces **Markov chain-based dynamic topic modeling** where theta (topic distributions) evolves over time for individual patients through multiple visits, following the approach: **p(θ_t | θ_t-1)**.

## Key Differences from Previous Implementation

### Previous Approach (Time-Varying Hyperparameters)
- **Model**: p(θ | α_t) where α_t varies by time bin
- **Scope**: Population-level trends (e.g., age groups)
- **Data**: Aggregated across time bins
- **Use case**: Understanding how disease prevalence changes across age groups

### Current Approach (Markov Chain over Theta)
- **Model**: p(θ_t | θ_t-1) with Markov transitions
- **Scope**: Individual patient trajectories
- **Data**: Sequential visits per patient
- **Use case**: Personalized disease progression modeling

## Architecture

### Markov Chain Prior
```
p(θ_0) ~ Dirichlet(α_prior)
p(θ_t | θ_t-1) ~ N(θ_t-1, σ²I)  for t > 0
```

where σ² is the transition variance (default: 0.01)

### VAE Encoder
```
q(θ_t | X_1...t-1, θ_t-1)
```

The VAE encoder processes:
1. **Sequential observations**: X_1, X_2, ..., X_t-1
2. **Previous theta**: θ_t-1
3. **LSTM hidden states**: Capturing long-term dependencies

### Model Flow
```
Patient Visits → Observation Sequence
                       ↓
              [Visit 1, Visit 2, ..., Visit T]
                       ↓
            Encode each observation (all modalities)
                       ↓
              LSTM processes sequence
                       ↓
         For each time step t:
           - Concatenate LSTM_t with θ_t-1
           - Compute μ_t, σ_t (variational parameters)
           - Sample: θ_t ~ N(μ_t, σ_t²)
           - Apply softmax to get valid distribution
           - Store θ_t
                       ↓
         Compute KL: KL(q(θ_t|...) || p(θ_t|θ_t-1))
```

## Data Structure

### Patient Sequential Data
Each patient has multiple visits over time:
```python
patient_sequences = {
    patient_id: [
        {
            'time': timestamp,
            'words': {
                modality_0: {word_id: frequency, ...},
                modality_1: {word_id: frequency, ...}
            }
        },
        # Next visit...
    ]
}
```

### Temporal Theta Storage
The model stores theta for each patient at each time step:
```python
self.theta_temporal: D x T x K tensor
# D = number of patients
# T = max time steps
# K = number of topics

self.patient_time_mask: D x T boolean tensor
# Tracks which (patient, time) pairs have data
```

## Implementation Details

### Initialization
```python
model = MixEHR_SAGE(
    corpus=corpus,
    seeds_topic_matrix=seeds,
    modality_list=['icd', 'medication', 'procedures'],
    guided_modality=0,  # ICD codes are guided
    enable_temporal=True,
    num_time_steps=10  # Max visits per patient
)
```

### Key Parameters
- `theta_hidden_size`: 200 (LSTM hidden units)
- `theta_nlayers`: 3 (LSTM layers)
- `theta_dropout`: 0.1 (dropout rate)
- `transition_variance`: 0.01 (Markov chain variance)
- `learning_rate`: 0.0001
- `weight_decay`: 1.2e-6

### Inference for a Patient
```python
# Get patient's sequential visits
seq_data = loader.get_patient_sequence_data(patient_id)

# Perform VAE inference
theta_samples, mu_theta, logvar_theta = model.infer_theta_variational(
    seq_data, patient_id
)

# Compute KL divergence
kl_loss = model.compute_markov_chain_kl(theta_samples, mu_theta, logvar_theta)

# Retrieve stored theta
theta_at_visit_3 = model.get_theta_at_time(patient_id, time_step=2)
```

### Saving Results
```python
# Save all temporal theta values
model.save_temporal_theta('./temporal_theta_markov.pt')

# Load later
data = torch.load('./temporal_theta_markov.pt')
theta_temporal = data['theta_temporal']  # D x T x K
patient_time_mask = data['patient_time_mask']  # D x T
```

## Multi-Modality Support

### Question: Should this be applied to all modalities or only seed modality?

**Answer**: The VAE encoder processes **all modalities** simultaneously.

**Rationale**:
1. **Richer observations**: Using all available data (ICD, medications, procedures) provides more complete picture of patient state
2. **Better inference**: More data → better theta estimation
3. **Guided modality**: Seeds only apply to topic inference (SCVB0_guided), not temporal modeling
4. **Implementation**: The encoder concatenates all modality BOWs into single feature vector

```python
# In encode_observation_sequence():
for each time step t:
    combined_features = []
    for modality in [ICD, medication, procedures, ...]:
        combined_features.append(BOW_modality)
    
    combined = concatenate(combined_features)  # Full observation
    encoded[t] = VAE_map(combined)
```

## Time Bins

### Question: Is there a need to make time bins?

**Answer**: **No**, time bins are not needed for Markov chain approach.

**Differences**:

| Aspect | Population-Level (with bins) | Markov Chain (no bins) |
|--------|------------------------------|------------------------|
| Time representation | Discrete bins (e.g., age 0-10, 10-20) | Actual visit sequences |
| Aggregation | Patients grouped by age | Individual trajectories |
| Theta | One per time bin (shared) | One per patient-visit |
| Temporal structure | Independent bins | Sequential dependencies |

**In Markov Chain**:
- Each visit is a time step in patient's sequence
- No need to discretize time into bins
- Visits can be at any time points
- Model learns from sequential order, not absolute time

**Example**:
```python
Patient A visits: [Day 5, Day 30, Day 90, Day 180]
Patient B visits: [Day 10, Day 45, Day 200]

# Markov chain:
Patient A: θ_0 → θ_1 → θ_2 → θ_3
Patient B: θ_0 → θ_1 → θ_2

# Each transition captures progression, regardless of actual time gaps
```

## Usage Example

### 1. Generate Sequential Data
```python
from temporal_markov_utils import PatientSequenceGenerator

gen = PatientSequenceGenerator(
    vocab_sizes=[500, 300],  # ICD, medication
    num_modalities=2
)

patient_sequences, metadata = gen.generate_synthetic_patient_sequences(
    num_patients=100,
    min_visits=3,
    max_visits=10
)
```

### 2. Initialize Model
```python
from MixEHR_SAGE import MixEHR_SAGE

model = MixEHR_SAGE(
    corpus=corpus,
    seeds_topic_matrix=seeds,
    modality_list=['icd', 'medication'],
    enable_temporal=True,
    num_time_steps=10
)
```

### 3. Perform Inference
```python
from temporal_markov_utils import SequenceDataLoader

loader = SequenceDataLoader(patient_sequences, corpus.V)

for patient_id in range(num_patients):
    seq_data = loader.get_patient_sequence_data(patient_id)
    theta_samples, mu, logvar = model.infer_theta_variational(seq_data, patient_id)
    kl_loss = model.compute_markov_chain_kl(theta_samples, mu, logvar)
```

### 4. Analyze Results
```python
# Get patient trajectory
for t in range(num_visits):
    theta_t = model.get_theta_at_time(patient_id, t)
    dominant_topic = torch.argmax(theta_t)
    print(f"Visit {t}: Dominant topic = {dominant_topic}")

# Save results
model.save_temporal_theta('./results/temporal_theta.pt')
```

## ELBO with Markov Chain

The ELBO includes the Markov chain KL term:

```
ELBO = E_q[log p(w|z,β,μ,π)] 
       + E_q[log p(z|θ_t)] - E_q[log q(z|γ)]
       - Σ_t KL(q(θ_t|X,θ_t-1) || p(θ_t|θ_t-1))
```

For each time step t:
- If t=0: KL(q(θ_0) || p(θ_0|α_prior))
- If t>0: KL(q(θ_t) || N(θ_t-1, σ²I))

## Advantages

1. **Personalized**: Models individual patient trajectories
2. **Sequential**: Captures temporal dependencies between visits
3. **Flexible**: Handles variable number of visits per patient
4. **Interpretable**: θ_t shows topic evolution over time
5. **Longitudinal**: Suitable for chronic disease progression

## Limitations and Future Work

1. **Variable time gaps**: Current model doesn't explicitly model time intervals between visits
2. **Missing modalities**: Needs handling when some modalities absent at certain visits
3. **Computational cost**: O(D × T) storage for all patient-time combinations
4. **Integration**: Not yet integrated with main training loop

### Future Enhancements
- Add time interval encoding (e.g., days between visits)
- Implement attention mechanism for long sequences
- Support irregular visit patterns
- Add survival analysis integration
- Develop patient risk stratification metrics

## Files

- `MixEHR_SAGE.py`: Core model with Markov chain implementation
- `temporal_markov_utils.py`: Data generation and loading utilities
- `example_markov_chain.py`: Complete working example
- `MARKOV_CHAIN_GUIDE.md`: This documentation

## References

1. Dynamic Topic Models (Blei & Lafferty, 2006)
2. Variational Autoencoders (Kingma & Welling, 2013)
3. Recurrent Latent Variable Models (Chung et al., 2015)
4. MixEHR-SAGE: Seed-guided topic modeling for EHR
