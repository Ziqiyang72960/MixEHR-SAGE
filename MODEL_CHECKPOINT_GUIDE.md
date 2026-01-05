# Model Checkpoint Documentation

This guide explains all the files saved during temporal topic model training, what they contain, and how to use them.

## Overview

After training completes, the model saves **6 files** in the output directory (default: `./results/`):

1. **`model.pt`** - Full model checkpoint (PyTorch format)
2. **`temporal_theta.pt`** - Temporal theta distributions (PyTorch tensor)
3. **`vocabularies.pkl`** - Medical code vocabularies and mappings
4. **`patient_id_mapping.pkl`** - Patient ID to array index mappings
5. **`patient_sequences.pkl`** - Patient visit histories
6. **`theta_results.pkl`** - Theta distributions with metadata (NumPy arrays)

---

## 1. `model.pt` - Full Model Checkpoint

### What It Contains

Complete model state for resuming training or performing inference on new patients.

```python
{
    'model_state': OrderedDict,      # Complete MixEHR_SAGE state_dict
    'vocab_sizes': List[int],         # [V_icd, V_medication, V_procedures, ...]
    'modality_list': List[str],       # ['ICD', 'medication', 'OPCS', ...]
    'num_topics': int,                # Number of disease topics (K)
    'patient_metadata': Dict          # Metadata for training patients
}
```

### What Each Component Means

- **`model_state`**: Contains ALL learned parameters:
  - **Topic-word distributions** (phi matrices): Which medical codes belong to which disease topics
  - **LSTM weights**: For encoding patient visit sequences  
  - **VAE encoder/decoder weights**: For inferring temporal topic distributions
  - **Markov chain parameters**: For modeling disease progression
  - **Temporal theta storage**: Topic distributions for each patient at each time step

- **`vocab_sizes`**: Number of unique codes per modality
  - Example: `[3577, 877, 3836]` = 3577 ICD codes, 877 medications, 3836 procedures
  - **Critical**: These dimensions MUST match when loading the model

- **`modality_list`**: Names of data types
  - Example: `['ICD', 'medication', 'OPCS']`
  - Order matches `vocab_sizes`

- **`num_topics`**: Number of disease topics (K)
  - Example: `50` = model learned 50 disease patterns
  - Each topic is a mixture of medical codes

- **`patient_metadata`**: Training patient information
  - Number of visits per patient
  - First/last visit dates
  - Used for validation and consistency checks

### How to Load

```python
import torch

# Load checkpoint
checkpoint = torch.load('results/model.pt', map_location='cpu')

# Access components
model_state = checkpoint['model_state']
vocab_sizes = checkpoint['vocab_sizes']
num_topics = checkpoint['num_topics']
modality_list = checkpoint['modality_list']

print(f"Model has {num_topics} topics")
print(f"Vocabulary sizes: {vocab_sizes}")
print(f"Modalities: {modality_list}")
```

### When to Use

- **Resume Training**: Continue training from checkpoint
- **Inference on New Patients**: Apply trained model to new data
- **Model Inspection**: Examine learned parameters
- **Transfer Learning**: Fine-tune on different datasets

---

## 2. `temporal_theta.pt` - Temporal Theta Tensor

### What It Contains

Raw PyTorch tensor of topic distributions over time.

```python
{
    'theta_temporal': Tensor,        # Shape: (D, T, K)
    'patient_time_mask': Tensor      # Shape: (D, T)
}
```

### What Each Component Means

- **`theta_temporal`**: Topic probability distributions
  - **Shape**: `(D, T, K)` where:
    - `D` = number of patients
    - `T` = maximum time steps (visits)
    - `K` = number of topics
  - **Values**: `theta[d, t, k]` = probability that patient `d` has disease topic `k` at time `t`
  - **Range**: [0, 1] per topic, sums to 1 across topics: `sum(theta[d, t, :]) = 1`

- **`patient_time_mask`**: Data validity indicator
  - **Shape**: `(D, T)`
  - **Values**: 
    - `1` = patient has visit data at this time step
    - `0` = no visit data (patient had fewer than T visits)
  - **Use**: Filter out invalid time steps when analyzing

### How to Load

```python
import torch

# Load tensor
data = torch.load('results/temporal_theta.pt', map_location='cpu')
theta = data['theta_temporal']  # Shape: (D, T, K)
mask = data['patient_time_mask']  # Shape: (D, T)

print(f"Theta shape: {theta.shape}")
print(f"Number of patients: {theta.shape[0]}")
print(f"Max time steps: {theta.shape[1]}")
print(f"Number of topics: {theta.shape[2]}")

# Get valid data for patient 0
patient_0_mask = mask[0]  # Which time steps have data
valid_times = patient_0_mask.nonzero().squeeze()
patient_0_theta = theta[0, valid_times, :]  # Only valid visits
```

### When to Use

- **PyTorch Operations**: Need gradient-enabled tensors
- **GPU Acceleration**: Transfer to GPU for fast computation
- **Large-Scale Analysis**: Batch processing of patients
- **Deep Learning Pipelines**: Feed into downstream models

---

## 3. `vocabularies.pkl` - Medical Code Vocabularies

### What It Contains

Bidirectional mappings between human-readable medical codes and integer IDs used by the model.

```python
{
    'code_to_id': Dict[int, Dict[str, int]],  # Encode: code → integer
    'id_to_code': Dict[int, Dict[int, str]],  # Decode: integer → code
    'vocab_sizes': List[int],                  # Size per modality
    'modality_names': List[str]                # Modality names
}
```

### What Each Component Means

- **`code_to_id`**: Encoding dictionary
  - **Structure**: `{modality_id: {code_string: integer_id}}`
  - **Example**: `{0: {"J45": 123, "E11.9": 456}, 1: {"N02BE01": 789}}`
  - **Use**: Convert CSV codes to model's integer IDs

- **`id_to_code`**: Decoding dictionary  
  - **Structure**: `{modality_id: {integer_id: code_string}}`
  - **Example**: `{0: {123: "J45", 456: "E11.9"}, 1: {789: "N02BE01"}}`
  - **Use**: Convert model outputs back to medical codes

- **`vocab_sizes`**: Vocabulary dimensions
  - **Example**: `[3577, 877, 3836]`
  - **Must match**: Model's vocab_sizes parameter

- **`modality_names`**: Modality labels
  - **Example**: `['ICD', 'medication', 'OPCS']`
  - **Index matches**: modality_id in dictionaries

### How to Load and Use

```python
import pickle

# Load vocabularies
with open('results/vocabularies.pkl', 'rb') as f:
    vocab_data = pickle.load(f)

code_to_id = vocab_data['code_to_id']
id_to_code = vocab_data['id_to_code']
modality_names = vocab_data['modality_names']

# Encode a medical code
icd_modality = 0  # ICD is first modality
code = "J45"  # Asthma
word_id = code_to_id[icd_modality].get(code)
print(f"Code '{code}' → ID {word_id}")

# Decode a word ID
decoded_code = id_to_code[icd_modality][word_id]
print(f"ID {word_id} → Code '{decoded_code}'")

# List all codes in a modality
medication_codes = list(code_to_id[1].keys())
print(f"Medication codes: {medication_codes[:5]}")
```

### When to Use

- **Interpreting Results**: Convert integer IDs to readable codes
- **Data Preprocessing**: Encode new patient data
- **Topic Interpretation**: Decode top words in each topic
- **Validation**: Check code coverage and mappings

---

## 4. `patient_id_mapping.pkl` - Patient ID Mappings

### What It Contains

Mappings between patient identifiers (from CSV files) and array indices (used in tensors).

```python
{
    'patient_id_to_idx': Dict[Any, int],  # patient_id → array index
    'idx_to_patient_id': Dict[int, Any],  # array index → patient_id
    'description': str                     # Usage instructions
}
```

### What Each Component Means

- **`patient_id_to_idx`**: Forward mapping
  - **Structure**: `{patient_id: array_index}`
  - **Example**: `{1001: 0, 1002: 1, 1003: 2, 1004: 3, 1005: 4}`
  - **Patient IDs**: Can be any type (int, string, UUID) from CSV files
  - **Array indices**: Sequential integers 0, 1, 2, ... used in theta arrays

- **`idx_to_patient_id`**: Reverse mapping
  - **Structure**: `{array_index: patient_id}`
  - **Example**: `{0: 1001, 1: 1002, 2: 1003, 3: 1004, 4: 1005}`
  - **Use**: Convert array row numbers back to patient IDs

### Why This Exists

**Problem**: Patient IDs in CSV files (e.g., 1001, 1002) cannot be used directly as array indices because:
- They may not start at 0
- They may not be consecutive
- They may be strings or UUIDs

**Solution**: Create explicit mapping:
- Patient data stored in dictionaries using `patient_id` as key
- Tensor arrays indexed 0, 1, 2, ... (array_index)
- Mapping bridges the two representations

### How to Load and Use

```python
import pickle
import numpy as np

# Load mapping
with open('results/patient_id_mapping.pkl', 'rb') as f:
    mapping = pickle.load(f)

patient_id_to_idx = mapping['patient_id_to_idx']
idx_to_patient_id = mapping['idx_to_patient_id']

# Load theta for specific patient
with open('results/theta_results.pkl', 'rb') as f:
    theta_data = pickle.load(f)
theta = theta_data['theta_temporal']  # Shape: (D, T, K)

# Access data for patient 1003
patient_id = 1003
array_idx = patient_id_to_idx[patient_id]  # Get array index
patient_theta = theta[array_idx, :, :]  # Get theta for this patient

print(f"Patient {patient_id} is at array index {array_idx}")
print(f"Theta shape for patient: {patient_theta.shape}")

# Reverse: given array index, find patient ID
idx = 2
patient_id = idx_to_patient_id[idx]
print(f"Array index {idx} corresponds to patient {patient_id}")
```

### When to Use

- **Accessing Theta Arrays**: Convert patient_id to array index
- **Batch Processing**: Iterate through patients systematically
- **Result Attribution**: Map array rows back to patient IDs
- **Validation**: Ensure correct patient-data association

---

## 5. `patient_sequences.pkl` - Patient Visit Histories

### What It Contains

Chronological records of each patient's visits with medical codes.

```python
{
    'patient_sequences': Dict[Any, List[Dict]],  # Visit histories
    'patient_metadata': Dict[Any, Dict],         # Patient summaries
    'description': str                            # Usage instructions
}
```

### What Each Component Means

- **`patient_sequences`**: Visit-by-visit data
  - **Structure**: `{patient_id: [visit1, visit2, ...]}`
  - **Each visit**:
    ```python
    {
        'time': numpy.datetime64,           # Visit date
        'words': {                          # Medical codes per modality
            modality_id: {
                word_id: frequency           # How many times code appeared
            }
        }
    }
    ```
  - **Example**:
    ```python
    {
        1001: [
            {
                'time': numpy.datetime64('2020-01-15'),
                'words': {
                    0: {123: 2, 456: 1},  # ICD codes: J45 (x2), E11.9 (x1)
                    1: {789: 1},           # Medications: N02BE01 (x1)
                    2: {45: 1}             # Procedures: E85.1 (x1)
                }
            },
            {
                'time': numpy.datetime64('2020-03-20'),
                'words': {0: {123: 1, 789: 1}, 1: {789: 2}, 2: {}}
            }
        ]
    }
    ```

- **`patient_metadata`**: Summary statistics
  - **Structure**: `{patient_id: metadata_dict}`
  - **Each metadata**:
    ```python
    {
        'num_visits': int,              # Total number of visits
        'first_visit': numpy.datetime64, # Date of first visit
        'last_visit': numpy.datetime64   # Date of last visit
    }
    ```
  - **Example**: `{1001: {'num_visits': 3, 'first_visit': ..., 'last_visit': ...}}`

### How to Load and Use

```python
import pickle
import numpy as np

# Load sequences
with open('results/patient_sequences.pkl', 'rb') as f:
    data = pickle.load(f)

patient_sequences = data['patient_sequences']
patient_metadata = data['patient_metadata']

# Load vocabularies for decoding
with open('results/vocabularies.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
id_to_code = vocab_data['id_to_code']

# Examine patient 1001
patient_id = 1001
visits = patient_sequences[patient_id]
metadata = patient_metadata[patient_id]

print(f"Patient {patient_id} had {metadata['num_visits']} visits")
print(f"First visit: {metadata['first_visit']}")
print(f"Last visit: {metadata['last_visit']}")

# Decode first visit
first_visit = visits[0]
print(f"\nVisit on {first_visit['time']}:")

for modality_id, word_dict in first_visit['words'].items():
    modality_name = vocab_data['modality_names'][modality_id]
    print(f"  {modality_name}:")
    for word_id, freq in word_dict.items():
        code = id_to_code[modality_id][word_id]
        print(f"    {code} (x{freq})")
```

### When to Use

- **Understanding Inputs**: See what data the model saw
- **Temporal Analysis**: Track code evolution over time
- **Clinical Review**: Validate data quality and completeness
- **Feature Engineering**: Create time-based features from visits

---

## 6. `theta_results.pkl` - Theta with Metadata (NumPy)

### What It Contains

Topic distributions as NumPy arrays with complete metadata and usage instructions.

```python
{
    'theta_temporal': numpy.ndarray,          # Shape: (D, T, K)
    'patient_time_mask': numpy.ndarray,       # Shape: (D, T)
    'patient_id_to_idx': Dict[Any, int],      # ID → index mapping
    'idx_to_patient_id': Dict[int, Any],      # index → ID mapping
    'num_topics': int,                         # K
    'num_time_steps': int,                     # T
    'description': Dict[str, str]              # Usage guide
}
```

### What Each Component Means

- **`theta_temporal`**: Topic distributions (NumPy)
  - **Type**: `numpy.ndarray`, dtype=float64
  - **Shape**: `(D, T, K)`
  - **Values**: Same as `temporal_theta.pt` but as NumPy array
  - **Advantage**: Easier to use, no PyTorch required

- **`patient_time_mask`**: Validity mask (NumPy)
  - **Type**: `numpy.ndarray`, dtype=int/bool
  - **Shape**: `(D, T)`
  - **Values**: 1=valid, 0=invalid

- **Mappings**: Included for convenience (same as `patient_id_mapping.pkl`)

- **`description`**: Embedded documentation
  - Explains what each field means
  - Provides usage examples
  - Self-documenting file

### How to Load and Use

```python
import pickle
import numpy as np

# Load results
with open('results/theta_results.pkl', 'rb') as f:
    results = pickle.load(f)

theta = results['theta_temporal']  # (D, T, K)
mask = results['patient_time_mask']  # (D, T)
patient_id_to_idx = results['patient_id_to_idx']
num_topics = results['num_topics']

print(f"Theta shape: {theta.shape}")
print(f"Number of topics: {num_topics}")
print(results['description'])  # Read embedded instructions

# Analyze patient 1002
patient_id = 1002
idx = patient_id_to_idx[patient_id]

# Get valid visits
valid_mask = mask[idx] == 1
patient_theta = theta[idx, valid_mask, :]  # Only valid time steps

# Find dominant topics at each visit
dominant_topics = np.argmax(patient_theta, axis=1)
print(f"Patient {patient_id} dominant topics over time: {dominant_topics}")

# Track specific topic (e.g., cardiovascular disease = topic 5)
cv_topic_id = 5
cv_probs = patient_theta[:, cv_topic_id]
print(f"Cardiovascular disease probability over time: {cv_probs}")
```

### When to Use

- **Standard Analysis**: NumPy-based processing
- **Visualization**: Plotting with matplotlib/seaborn
- **Statistics**: Scipy/statsmodels analyses
- **Quick Exploration**: No deep learning framework needed
- **Sharing Results**: Easy to load without dependencies

---

## File Selection Criteria

### Which File Should I Use?

| Task | Recommended File | Reason |
|------|-----------------|--------|
| Resume training | `model.pt` | Contains full model state |
| Infer new patients | `model.pt` | Need complete model |
| Interpret topics | `vocabularies.pkl` + `model.pt` | Decode learned topics |
| Analyze single patient | `theta_results.pkl` + `patient_id_mapping.pkl` | Easy NumPy access |
| Track disease progression | `patient_sequences.pkl` + `theta_results.pkl` | Full temporal view |
| Visualize trends | `theta_results.pkl` | NumPy arrays for plotting |
| GPU-accelerated analysis | `temporal_theta.pt` | PyTorch tensors |
| Share with clinicians | `theta_results.pkl` + `vocabularies.pkl` + this guide | Self-contained, interpretable |

---

## Complete Analysis Example

```python
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load all necessary files
with open('results/theta_results.pkl', 'rb') as f:
    theta_data = pickle.load(f)
with open('results/vocabularies.pkl', 'rb') as f:
    vocab_data = pickle.load(f)
with open('results/patient_sequences.pkl', 'rb') as f:
    seq_data = pickle.load(f)

theta = theta_data['theta_temporal']
mask = theta_data['patient_time_mask']
patient_id_to_idx = theta_data['patient_id_to_idx']
id_to_code = vocab_data['id_to_code']
patient_sequences = seq_data['patient_sequences']

# Analyze patient 1003
patient_id = 1003
idx = patient_id_to_idx[patient_id]

# Get temporal theta
valid_mask = mask[idx] == 1
patient_theta = theta[idx, valid_mask, :]
num_visits = np.sum(valid_mask)

# Get visit dates
visits = patient_sequences[patient_id]
visit_dates = [v['time'] for v in visits]

# Plot topic evolution
fig, ax = plt.subplots(figsize=(12, 6))
for topic_id in range(min(5, theta.shape[2])):  # Plot top 5 topics
    ax.plot(range(num_visits), patient_theta[:, topic_id], 
            label=f'Topic {topic_id}', marker='o')

ax.set_xlabel('Visit Number')
ax.set_ylabel('Topic Probability')
ax.set_title(f'Disease Topic Evolution - Patient {patient_id}')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('patient_1003_topic_evolution.png')
print("Saved visualization to patient_1003_topic_evolution.png")

# Print clinical codes at each visit
print(f"\n{'='*60}")
print(f"Clinical Summary - Patient {patient_id}")
print(f"{'='*60}")
for i, visit in enumerate(visits):
    print(f"\nVisit {i+1} ({visit['time']}):")
    print(f"Dominant topic: {np.argmax(patient_theta[i])}")
    
    # Decode medical codes
    for mod_id, word_dict in visit['words'].items():
        if word_dict:  # Only if has codes
            mod_name = vocab_data['modality_names'][mod_id]
            print(f"  {mod_name}:")
            for word_id in list(word_dict.keys())[:3]:  # Show first 3
                code = id_to_code[mod_id][word_id]
                print(f"    - {code}")
```

---

## File Size Considerations

| File | Typical Size | Scales With |
|------|--------------|-------------|
| `model.pt` | 10-100 MB | num_topics × vocab_sizes |
| `temporal_theta.pt` | 1-50 MB | num_patients × num_time_steps × num_topics |
| `vocabularies.pkl` | <1 MB | Total unique codes |
| `patient_id_mapping.pkl` | <1 MB | num_patients |
| `patient_sequences.pkl` | 1-10 MB | num_patients × avg_codes_per_visit |
| `theta_results.pkl` | 1-50 MB | Same as temporal_theta.pt (NumPy overhead) |

For large datasets (>10K patients, >100 topics), files can be GB-scale. Consider:
- Loading only needed portions
- Using memory-mapped arrays (`np.load(..., mmap_mode='r')`)
- Compressing with `gzip` or `lzma`

---

## Version Compatibility

All files are created with:
- **PyTorch**: Saved with `torch.save()`, load with `torch.load()`
- **Pickle**: Python's pickle protocol (default version)
- **NumPy**: Standard ndarray serialization

**Compatibility Notes**:
- PyTorch files require compatible PyTorch version (usually backward compatible)
- Pickle files require Python 3.x
- NumPy arrays are cross-version compatible
- Datetime64 objects require NumPy

**Best Practice**: Save training environment info:
```python
import torch
import sys

with open('results/environment_info.txt', 'w') as f:
    f.write(f"Python: {sys.version}\n")
    f.write(f"PyTorch: {torch.__version__}\n")
    f.write(f"NumPy: {np.__version__}\n")
```

---

## Troubleshooting

### "Can't load model.pt"
- **Cause**: PyTorch version mismatch
- **Solution**: Load with `torch.load(..., map_location='cpu')`

### "Dimension mismatch when loading"
- **Cause**: vocab_sizes changed between training and loading
- **Solution**: Check `checkpoint['vocab_sizes']` matches your data

### "Patient ID not found in mapping"
- **Cause**: Trying to access patient not in training set
- **Solution**: Use `infer_new_patients()` function for new patients

### "All theta values are zero"
- **Cause**: Model not trained or training failed
- **Solution**: Check training logs, ensure epochs > 0

---

## Summary

After training, you have complete model information:

✅ **Full Model State** (`model.pt`) - Resume training, perform inference  
✅ **Temporal Theta** (`temporal_theta.pt`, `theta_results.pkl`) - Topic distributions over time  
✅ **Vocabularies** (`vocabularies.pkl`) - Decode model outputs to medical codes  
✅ **Patient Mappings** (`patient_id_mapping.pkl`) - Link IDs to array indices  
✅ **Visit Histories** (`patient_sequences.pkl`) - Input data for validation  

All files are **self-documented**, **cross-referenced**, and **ready for analysis**.

For clinical interpretation and analysis examples, see `TEMPORAL_MODEL_OUTPUT_GUIDE.md`.
