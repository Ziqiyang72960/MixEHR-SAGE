# Integration Guide: Temporal Inference with run_MixEHR.py

## Overview

Temporal Markov chain inference has been **integrated into the existing run_MixEHR.py pipeline**. The temporal functionality now works with the full preprocessing pipeline including PheCode mapping, GMM priors, and token counting.

## What Changed

### Integration Complete ✅

**run_MixEHR.py** now supports temporal inference:
- Added `--enable-temporal` flag
- Added `--num-time-steps N` parameter
- Passes temporal parameters through to main.py
- Uses full preprocessing pipeline

**main.py** updated:
- Added `-enable_temporal` argument
- Added `-num_time_steps` argument  
- Passes temporal parameters to MixEHR_SAGE initialization
- Saves temporal_theta.pt when temporal inference is enabled

**MixEHR_SAGE.py**: Already supports temporal inference (no changes needed)

### Pipeline Flow

```
run_MixEHR.py (with --enable-temporal)
    ↓
1. corpus.py process → Build corpus from raw data
    ↓
2. get_doc_phecode.py → Extract PheCode mappings
    ↓
3. get_prior_GMM.py → Compute GMM priors
    ↓
4. get_token_counts.py → Count tokens
    ↓
5. main.py (with -enable_temporal) → Train model with temporal inference
    ↓
   Output: temporal_theta.pt (D×T×K tensor)
```

## Usage

### Standard Training (No Temporal)

```bash
python run_MixEHR.py ./data/ --epochs 5 --batch-size 1000
```

### Training with Temporal Inference

```bash
python run_MixEHR.py ./data/ \
    --enable-temporal \
    --num-time-steps 10 \
    --epochs 5 \
    --batch-size 1000
```

### Quick Test with Example Data

```bash
# Use provided example data for quick testing
python run_MixEHR.py ./example_data/ \
    --enable-temporal \
    --num-time-steps 4 \
    --epochs 3 \
    --output ./test_results/
```

### All Options

```bash
python run_MixEHR.py <data_path> [OPTIONS]

Required:
  data_path                 Directory with ukbb_metadata.csv and data files

Optional:
  --output DIR, -o         Output directory (default: ./results/)
  --store DIR, -s          Corpus storage directory (default: ./store/)
  --epochs N, -e           Training epochs (default: 5)
  --batch-size N, -b       Batch size (default: 1000)
  --max-docs N, -n         Max documents to process (default: all)
  --skip-corpus            Skip corpus processing (use existing)
  --skip-prior             Skip prior computation (use existing)
  --enable-temporal        Enable temporal Markov chain inference
  --num-time-steps N       Max time steps per patient (default: 10)
```

## Data Format

### Directory Structure

```
data/
├── ukbb_metadata.csv          # Modality configuration
├── icd_codes.csv             # ICD diagnosis codes
├── medications.csv           # Medication records
└── procedures.csv            # Procedure codes
```

### ukbb_metadata.csv Format

```csv
index,path,word_column
icd,icd_codes.csv,code
medication,medications.csv,code
opcs,procedures.csv,code
```

**Required columns**:
- `index`: Modality name (first row is guided modality)
- `path`: Relative path to data file
- `word_column`: Column name containing codes

### Data Files Format

Standard format (used by corpus processing):
```csv
patient_id,code
1001,J45.0
1001,E11.9
1002,I25.1
```

Or with temporal information (for future temporal data processing):
```csv
patient_id,date,code
1001,2020-01-15,J45.0
1001,2020-03-20,I10
```

## Output Files

### Standard Output

When training completes, you'll find in the output directory:
- `model_checkpoint_*.pt` - Model checkpoints
- `elbo1.txt`, `elbo2.txt` - ELBO values
- `pz.txt`, `qz.txt`, `pw.txt` - Loss components

### Temporal Output

When `--enable-temporal` is used, additional file:
- **temporal_theta.pt** - PyTorch file containing:
  ```python
  {
      'theta_temporal': D×T×K tensor,  # Theta for each patient-time-topic
      'patient_time_mask': D×T boolean, # Valid data indicators
      'num_time_steps': T,
      'num_patients': D,
      'num_topics': K
  }
  ```

### Loading Temporal Results

```python
import torch

# Load temporal theta
data = torch.load('./results/temporal_theta.pt')
theta_temporal = data['theta_temporal']  # D×T×K
mask = data['patient_time_mask']  # D×T

# Get theta for patient 5, time step 2
patient_5_time_2_theta = theta_temporal[5, 2, :]  # K-dimensional vector

# Check if data exists
if mask[5, 2]:
    print("Patient 5 at time 2:", patient_5_time_2_theta)
```

## Benefits of Integration

✅ **Full preprocessing**: Uses PheCode mapping, GMM priors, token initialization  
✅ **Backward compatible**: Temporal is optional (disabled by default)  
✅ **Production ready**: Integrated into existing production pipeline  
✅ **Easy to use**: Just add `--enable-temporal` flag  
✅ **Proper initialization**: Uses get_doc_phecode.py, get_prior_GMM.py outputs  

## Differences from train_temporal.py

| Aspect | run_MixEHR.py (Integrated) | train_temporal.py (Standalone) |
|--------|---------------------------|-------------------------------|
| **Preprocessing** | ✅ Full pipeline | ❌ Skips preprocessing |
| **PheCode mapping** | ✅ Yes | ❌ No |
| **GMM priors** | ✅ Yes | ❌ Random |
| **Corpus** | ✅ Processed corpus | ❌ Mock corpus |
| **Seeds** | ✅ Seed topic matrix | ❌ Random seeds |
| **Use case** | Production | Demo/testing |

## Example Workflow

### 1. Prepare Data

Ensure your data directory has the correct structure:
```bash
ls data/
# Should show: ukbb_metadata.csv, icd_codes.csv, medications.csv, etc.
```

### 2. Run Training

```bash
# Standard training
python run_MixEHR.py ./data/ --epochs 10

# With temporal inference
python run_MixEHR.py ./data/ --enable-temporal --num-time-steps 10 --epochs 10
```

### 3. Analyze Results

```python
import torch
import matplotlib.pyplot as plt

# Load results
theta_data = torch.load('./results/temporal_theta.pt')
theta = theta_data['theta_temporal']  # D×T×K

# Plot topic evolution for patient 0
patient_0_theta = theta[0, :, :]  # T×K
num_visits = theta_data['patient_time_mask'][0].sum().item()

plt.figure(figsize=(10, 6))
for k in range(5):  # Plot top 5 topics
    plt.plot(range(num_visits), patient_0_theta[:num_visits, k], 
             label=f'Topic {k}', marker='o')

plt.xlabel('Visit Number')
plt.ylabel('Topic Probability')
plt.title('Patient 0: Topic Evolution Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('topic_evolution.png')
```

## Troubleshooting

### Issue: "No metadata file found"
**Solution**: Ensure `ukbb_metadata.csv` exists in data directory

### Issue: "PheCode mapping failed"
**Solution**: Check that ICD codes are valid and phecode_mapping directory exists

### Issue: "Temporal theta not saved"
**Solution**: Make sure you used `--enable-temporal` flag

### Issue: "Out of memory with temporal"
**Solution**: 
- Reduce `--num-time-steps`
- Reduce `--batch-size`
- Use smaller dataset initially

## Advanced Usage

### Resume Training

```bash
# First run (save to store/)
python run_MixEHR.py ./data/ --enable-temporal --epochs 5

# Resume (skip preprocessing)
python run_MixEHR.py ./data/ \
    --enable-temporal \
    --epochs 10 \
    --skip-corpus \
    --skip-prior
```

### Process Subset of Data

```bash
# Process only first 1000 patients
python run_MixEHR.py ./data/ \
    --enable-temporal \
    --max-docs 1000 \
    --epochs 5
```

### Custom Paths

```bash
python run_MixEHR.py ./my_data/ \
    --enable-temporal \
    --store ./my_corpus/ \
    --output ./my_results/ \
    --epochs 10
```

## Migration Guide

If you were using `train_temporal.py` standalone:

**Before** (standalone):
```bash
python train_temporal.py ./data/ --enable-temporal --epochs 5
```

**After** (integrated):
```bash
python run_MixEHR.py ./data/ --enable-temporal --epochs 5
```

**Key difference**: The integrated version includes full preprocessing, so results will be based on proper PheCode mappings and GMM priors rather than mock data.

## Future Enhancements

Planned improvements:
1. Direct temporal data loading (dates/time bins)
2. Integration with infer_patient.py for new patient inference
3. Temporal visualization tools
4. Time interval encoding
5. Attention mechanism for long sequences

## References

- **MARKOV_CHAIN_GUIDE.md**: Detailed architecture documentation
- **TRAINING_GUIDE.md**: train_temporal.py standalone guide
- **example_temporal.py**: Standalone demo examples
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details
