# Training Guide: Temporal Markov Chain Model with Mixed Granularity Data

## Overview

This guide shows how to train the temporal Markov chain model with real-world mixed temporal granularity data:
- **Medications**: 4 time bins (baseline, 1st visit, 2nd visit, 3rd visit)
- **ICD/OPCS**: Specific dates

The system automatically aligns data to a common temporal grid based on visit dates.

## Quick Start

### 1. Basic Training

```bash
# Train with temporal inference
python train_temporal.py ./example_data/ --enable-temporal --epochs 5

# Output:
# - ./results/temporal_theta.pt (D×T×K tensor)
# - ./results/model.pt (model checkpoint)
# - ./results/patient_theta_results.pkl (detailed results)
```

### 2. Infer Theta for New Patients

```bash
# After training, infer on new patients
python train_temporal.py ./example_data/ --infer-only --model-path ./results/model.pt

# Output:
# - ./patient_theta_inference.json (theta sequences and dominant topics)
```

## Data Format

### Directory Structure

```
data/
├── ukbb_metadata.csv          # Metadata describing all modalities
├── icd_temporal.csv           # ICD codes with specific dates
├── medication_temporal.csv    # Medications with time bins
└── opcs_temporal.csv          # OPCS codes with specific dates
```

### Metadata File (ukbb_metadata.csv)

```csv
index,path,word_column
icd,icd_temporal.csv,code
medication,medication_temporal.csv,code
opcs,opcs_temporal.csv,code
```

**Columns**:
- `index`: Modality name (first one is the guided modality)
- `path`: Path to data file relative to metadata file
- `word_column`: Column name containing codes/words

### Dated Modality Format (ICD, OPCS)

```csv
patient_id,date,code
1001,2020-01-15,J45.0
1001,2020-01-15,E11.9
1001,2020-03-20,I10
1001,2020-03-20,E78.5
1001,2020-07-10,J45.0
```

**Columns**:
- `patient_id`: Patient identifier
- `date`: Visit date (YYYY-MM-DD format)
- `code`: ICD/OPCS code

**Key points**:
- Multiple codes can occur on same date
- Dates define the temporal grid
- Missing dates create separate visit records

### Binned Modality Format (Medications)

```csv
patient_id,time_bin,code
1001,0,N02BE01
1001,0,A10BA02
1001,1,N02BE01
1001,1,A10BA02
1001,1,C03CA01
1001,2,N02BE01
```

**Columns**:
- `patient_id`: Patient identifier
- `time_bin`: Temporal bin (0=baseline, 1=1st visit, 2=2nd visit, 3=3rd visit)
- `code`: Medication code (ATC format)

**Time bin mapping**:
- `0`: Baseline (before any visits)
- `1`: First visit period
- `2`: Second visit period
- `3`: Third visit period

## How Mixed Granularity is Handled

### Alignment Algorithm

1. **Extract visit dates** from ICD/OPCS data
2. **Sort unique dates** to create temporal grid
3. **Map time bins to visits**:
   - Time bin 0 → Visit 0 (or closest date)
   - Time bin 1 → Visit 1
   - Time bin 2 → Visit 2
   - Time bin 3 → Visit 3

### Example

Patient 1001:
- **ICD dates**: 2020-01-15, 2020-03-20, 2020-07-10
- **Medication bins**: 0, 1, 2, 3

**Aligned sequence**:
```
Visit 0 (2020-01-15):
  - ICD: J45.0, E11.9
  - Medication (bin 0): N02BE01, A10BA02
  - OPCS: E85.1

Visit 1 (2020-03-20):
  - ICD: I10, E78.5
  - Medication (bin 1): N02BE01, A10BA02, C03CA01
  - OPCS: E85.1

Visit 2 (2020-07-10):
  - ICD: J45.0, Z23
  - Medication (bin 2): N02BE01, A10BA02, C03CA01
  - OPCS: E85.1

Visit 3 (implied from bin 3):
  - Medication (bin 3): A10BA02, C03CA01
```

## Command-Line Arguments

### Training Mode

```bash
python train_temporal.py <data_dir> [OPTIONS]
```

**Required**:
- `data_dir`: Directory containing data files and ukbb_metadata.csv

**Optional**:
- `--metadata FILE`: Metadata filename (default: ukbb_metadata.csv)
- `--output DIR, -o DIR`: Output directory (default: ./results/)
- `--enable-temporal`: Enable Markov chain temporal inference
- `--num-time-steps N`: Max time steps per patient (default: 10)
- `--epochs N, -e N`: Training epochs (default: 5)
- `--batch-size N, -b N`: Batch size (default: 50)

### Inference Mode

```bash
python train_temporal.py <data_dir> --infer-only --model-path <model.pt>
```

**Required for inference**:
- `--infer-only`: Switch to inference mode
- `--model-path FILE`: Path to trained model

**Optional**:
- `--inference-output FILE`: Output JSON file (default: ./patient_theta_inference.json)

## Complete Examples

### Example 1: Training from Scratch

```bash
# Using provided example data
python train_temporal.py ./example_data/ \
    --enable-temporal \
    --num-time-steps 10 \
    --epochs 5 \
    --batch-size 50 \
    --output ./my_results/

# Output files:
# ./my_results/temporal_theta.pt
# ./my_results/model.pt
# ./my_results/patient_theta_results.pkl
```

### Example 2: Training without Temporal (Baseline)

```bash
# Standard model without temporal dynamics
python train_temporal.py ./example_data/ \
    --epochs 5 \
    --output ./baseline_results/
```

### Example 3: Inference on New Patients

```bash
# Step 1: Train model
python train_temporal.py ./training_data/ \
    --enable-temporal \
    --epochs 10 \
    --output ./trained_model/

# Step 2: Prepare new patient data in same format
# (new_data/ukbb_metadata.csv, new_data/icd_temporal.csv, etc.)

# Step 3: Infer theta for new patients
python train_temporal.py ./new_data/ \
    --infer-only \
    --model-path ./trained_model/model.pt \
    --inference-output ./new_patient_results.json
```

### Example 4: Custom Configuration

```bash
python train_temporal.py ./data/ \
    --metadata custom_metadata.csv \
    --output ./custom_output/ \
    --enable-temporal \
    --num-time-steps 15 \
    --epochs 20 \
    --batch-size 100
```

## Output Files

### temporal_theta.pt

PyTorch tensor file containing:
```python
{
    'theta_temporal': D×T×K tensor,  # Theta for each patient-visit-topic
    'patient_time_mask': D×T tensor,  # Boolean mask of valid data
    'num_time_steps': T,              # Max time steps
    'num_patients': D,                # Number of patients
    'num_topics': K                   # Number of topics
}
```

**Usage**:
```python
import torch

data = torch.load('temporal_theta.pt')
theta = data['theta_temporal']  # D×T×K
mask = data['patient_time_mask']  # D×T

# Get theta for patient 5, visit 2
patient_5_visit_2_theta = theta[5, 2, :]  # K-dimensional vector
```

### model.pt

Model checkpoint containing:
```python
{
    'model_state': OrderedDict,      # Model parameters
    'vocab_sizes': List[int],        # Vocabulary sizes per modality
    'modality_list': List[str],      # Modality names
    'num_topics': int,               # Number of topics
    'patient_metadata': Dict         # Patient visit information
}
```

### patient_theta_results.pkl

Pickle file with detailed results:
```python
{
    'patient_sequences': Dict,        # Original patient visit sequences
    'patient_metadata': Dict,         # Visit counts and times
    'theta_temporal': numpy.array,    # D×T×K array
    'patient_time_mask': numpy.array  # D×T array
}
```

### patient_theta_inference.json (Inference Mode)

JSON file with theta sequences for new patients:
```json
{
  "1001": {
    "theta_sequence": [[0.05, 0.02, ...], [0.04, 0.03, ...], ...],
    "num_visits": 4,
    "dominant_topics": [12, 12, 15, 18],
    "topic_evolution": [...]
  },
  "1002": {
    ...
  }
}
```

**Fields**:
- `theta_sequence`: List of theta vectors (one per visit)
- `num_visits`: Number of visits for this patient
- `dominant_topics`: Top topic ID at each visit
- `topic_evolution`: Full theta evolution (same as theta_sequence)

## Interpreting Results

### Accessing Patient Trajectories

```python
import torch
import pickle

# Load results
with open('./results/patient_theta_results.pkl', 'rb') as f:
    results = pickle.load(f)

theta_temporal = results['theta_temporal']  # D×T×K
mask = results['patient_time_mask']  # D×T
metadata = results['patient_metadata']

# Analyze patient 1001
patient_id = 1001
num_visits = metadata[patient_id]['num_visits']

print(f"Patient {patient_id} has {num_visits} visits")

for t in range(num_visits):
    if mask[patient_id, t]:
        theta_t = theta_temporal[patient_id, t, :]
        dominant_topic = theta_t.argmax()
        top_5_topics = theta_t.argsort()[-5:][::-1]
        
        print(f"Visit {t+1}:")
        print(f"  Dominant topic: {dominant_topic}")
        print(f"  Top 5 topics: {top_5_topics}")
        print(f"  Topic probabilities: {theta_t[top_5_topics]}")
```

### Disease Progression Analysis

```python
import matplotlib.pyplot as plt
import numpy as np

# Track topic evolution for a patient
patient_id = 1001
num_visits = metadata[patient_id]['num_visits']

# Get theta over time
theta_seq = [theta_temporal[patient_id, t, :] for t in range(num_visits)]
theta_seq = np.array(theta_seq)  # T×K

# Plot top 5 topics over time
top_topics = theta_seq.mean(axis=0).argsort()[-5:][::-1]

plt.figure(figsize=(10, 6))
for topic_id in top_topics:
    plt.plot(range(num_visits), theta_seq[:, topic_id], 
             marker='o', label=f'Topic {topic_id}')

plt.xlabel('Visit Number')
plt.ylabel('Topic Probability')
plt.title(f'Patient {patient_id}: Topic Evolution')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('topic_evolution.png')
```

## Troubleshooting

### Issue: "No temporal data found"

**Problem**: Patient has no dated modalities or time bins

**Solution**: Ensure at least one modality has `date` or `time_bin` column

### Issue: "Misaligned time bins"

**Problem**: Medication time bins don't align with visit dates

**Solution**: The algorithm automatically maps bins to nearest visits. Verify time_bin values are 0-3.

### Issue: "Memory error with large datasets"

**Problem**: D×T×K tensor too large

**Solution**: 
- Reduce `--num-time-steps`
- Process patients in smaller batches
- Use `--batch-size` to control memory

### Issue: "Inference gives unexpected theta"

**Problem**: New patient data format differs from training

**Solution**: Ensure new data follows exact same format and column names as training data

## Advanced Usage

### Custom Temporal Alignment

Modify `MixedTemporalDataProcessor.align_temporal_data()` to implement custom alignment logic.

### Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python train_temporal.py ./data/ --enable-temporal
```

### Checkpoint Resume

Save checkpoints during training and resume:
```python
# In training loop, save periodically
if epoch % save_every == 0:
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict()
    }, f'checkpoint_epoch_{epoch}.pt')
```

## Performance Tips

1. **Batch size**: Larger batches (100-200) for faster training
2. **Time steps**: Use minimum necessary (5-10 typical)
3. **Data preprocessing**: Cache processed sequences
4. **GPU**: Enable CUDA for 10-50x speedup

## Citation

If you use this temporal modeling approach, please cite:

```bibtex
@software{mixehr_sage_temporal,
  title={MixEHR-SAGE: Temporal Markov Chain Topic Model for EHR},
  author={...},
  year={2024},
  url={https://github.com/...}
}
```
