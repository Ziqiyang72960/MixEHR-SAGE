# Temporal Disease Prediction for MixEHR-SAGE

This document describes the temporal disease prediction functionality that enables time-based analysis and future disease risk prediction.

## Table of Contents

1. [Quick Start Workflow](#quick-start-workflow)
2. [Overview](#overview)
3. [Complete Pipeline: From Training to Temporal Inference](#complete-pipeline-from-training-to-temporal-inference)
4. [Data Format](#data-format)
5. [Usage](#usage)
6. [Implementation Details](#implementation-details)
7. [References](#references)

## Quick Start Workflow

**Can I use my existing trained model?**
- **Yes!** If you already have trained MixEHR-SAGE model results (e.g., in `./results/` folder), you can use them directly for temporal inference without retraining.
- The temporal inference script uses the **same phi and theta distributions** learned from your standard MixEHR-SAGE training.

**Do I need to retrain the model for temporal prediction?**
- **No!** Temporal prediction uses your existing trained model's phi distributions.
- You only need temporal patient data with timestamps (see [Data Format](#data-format)).

**Basic workflow:**
```bash
# If you already have results in ./results/, skip to step 3
# 1. Train MixEHR-SAGE (if needed)
python run_MixEHR.py ./data/

# 2. (Results are now in ./results/)

# 3. Run temporal inference with your temporal data
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method lstm \
    --output temporal_predictions.csv
```

## Overview

The temporal prediction framework implements three approaches for predicting future disease occurrences based on historical patient data with timestamps:

1. **LSTM-based Temporal VAE**: Learns temporal dynamics using variational autoencoders
2. **Simple Regression Models**: Predicts time to next visit or classification of visits within time windows
3. **Autoregressive Models**: Generates future topic mixtures (theta) autoregressively

## Complete Pipeline: From Training to Temporal Inference

### Step 1: Standard MixEHR-SAGE Training (One-Time Setup)

**If you don't have a trained model yet:**

1. **Prepare your training data** in the data directory (e.g., `./data/`):
   - `ukbb_metadata.csv`: Metadata file specifying modalities
   - Modality data files (e.g., `ukb_synthetic_icd.csv`, `ukb_synthetic_medication.txt`, etc.)

2. **Run the standard training pipeline:**
   ```bash
   python run_MixEHR.py ./data/ --output ./results/ --epochs 5
   ```

   This command:
   - Processes the corpus
   - Builds priors (GMM, token counts, PheCode mappings)
   - Trains the MixEHR-SAGE model
   - Saves learned parameters (phi, theta, pi) to `./results/`

3. **Training outputs** (saved in `./results/`):
   - `toy_exp_n_icd_*.pt`: Phi distributions for ICD codes
   - `toy_exp_n_med_*.pt`: Phi distributions for medications
   - `toy_exp_n_opcs_*.pt`: Phi distributions for OPCS codes
   - `toy_exp_m_*.pt`: Theta distributions for patients
   - `toy_exp_s_*.pt`: Seed word distributions
   - `toy_pi_*.pt`: Pi parameters (mixing weights)

**If you already have a trained model:**
- You can use your existing `./results/` folder directly
- No need to retrain unless you want to update the model with new data
- The temporal inference uses the **phi distributions** from these files

### Step 2: Prepare Temporal Patient Data

Create a CSV file with time-stamped patient records:

**Format: `SUBJECT_ID, code, timestamp, modality`**

Example (`data/temporal_patient_data.csv`):
```csv
SUBJECT_ID,code,timestamp,modality
patient_001,E11.9,2015-03-15,icd
patient_001,C03AB01,1,med
patient_001,K30100,2015-04-20,opcs
patient_001,I10,2015-09-10,icd
patient_002,E78.5,2016-01-05,icd
patient_002,C09AA01,2,med
```

**Timestamp specifications:**
- **ICD/OPCS codes**: Use `YYYY-MM-DD` format (e.g., `2015-03-15`)
- **Medications**: Use categorical values `0`, `1`, `2`, `3` representing time ranges:
  - `0` = 2000-2005
  - `1` = 2006-2010
  - `2` = 2010-2015
  - `3` = 2016-2020

### Step 3: Run Temporal Inference

Use the trained model (from `./results/`) with your temporal data:

```bash
# LSTM-based temporal prediction
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method lstm \
    --window-months 6 \
    --epochs 50 \
    --output lstm_predictions.csv

# Simple regression
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method regression \
    --task classification \
    --predict-window 6 \
    --output regression_predictions.csv

# Autoregressive prediction
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method autoregressive \
    --future-steps 3 \
    --output autoregressive_predictions.csv
```

### Step 4: Interpret Results

The temporal inference script:
1. **Loads phi distributions** from your trained model (`./results/`)
2. **Processes temporal data** into time windows
3. **Computes theta_t** for each patient at each time point
4. **Trains temporal model** (LSTM/regression/autoregressive) on historical sequences
5. **Predicts future** disease risk or time-to-event

**Output files contain:**
- Patient IDs
- Time points (or predicted time gaps)
- Predicted theta distributions (topic mixtures)
- Disease probabilities
- Healthy status (if threshold is specified)

### Important Notes

**Model Compatibility:**
- ✅ You can use your existing trained model from `./results/`
- ✅ No need to modify `run_MixEHR.py` for temporal inference
- ✅ The same phi distributions work for both static and temporal inference
- ❌ Do NOT retrain the base model just for temporal prediction

**Data Requirements:**
- **Training data** (for `run_MixEHR.py`): Standard EHR data without timestamps
- **Temporal data** (for `infer_patient_temporal.py`): Same codes but WITH timestamps

**When to Retrain the Base Model:**
- You have new patient cohorts to train on
- You want to include additional modalities
- You want to update the phi distributions with more data

**When NOT to Retrain:**
- You just want to do temporal predictions on new patients
- You want to try different temporal prediction methods (LSTM/regression/autoregressive)
- You want to adjust time windows or prediction horizons

## Data Format

### Input Data Structure

Temporal data should be provided as a CSV file with the following columns:

```csv
SUBJECT_ID,code,timestamp,modality
patient_001,E11.9,2015-03-15,icd
patient_001,C03AB01,1,med
patient_001,K30100,2015-04-20,opcs
```

### Column Specifications

- **SUBJECT_ID**: Unique patient identifier
- **code**: Medical code (ICD-10, ATC, OPCS)
- **timestamp**: 
  - For ICD and OPCS: `YYYY-MM-DD` format (e.g., `2015-03-15`)
  - For medications: Categorical value `0`, `1`, `2`, or `3` representing time ranges:
    - `0`: 2000-2005
    - `1`: 2006-2010  
    - `2`: 2010-2015
    - `3`: 2016-2020
- **modality**: One of `icd`, `med`, or `opcs`

## Usage

### 1. LSTM-based Temporal VAE

Learns q(eta_t | X_1...t-1) with temporal priors p(theta_t | theta_t-1).

```bash
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method lstm \
    --window-months 6 \
    --epochs 50 \
    --output lstm_predictions.csv
```

**Key Parameters:**
- `--window-months`: Size of time windows for aggregation (default: 6)
- `--epochs`: Number of training epochs (default: 50)

**Output:**
- Predictions of future disease risk with temporal dynamics
- KL divergence and reconstruction loss metrics

### 2. Simple Regression Models

Predicts either:
- **Regression**: Time gap (in years) to next visit
- **Classification**: Binary prediction of visit within specified time window

```bash
# Regression: Predict time to next visit
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method regression \
    --task regression \
    --window-months 6 \
    --output regression_predictions.csv

# Classification: Predict visit within 6 months
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method regression \
    --task classification \
    --predict-window 6 \
    --output classification_predictions.csv
```

**Key Parameters:**
- `--task`: Choose `regression` or `classification`
- `--predict-window`: Time window in months for classification (default: 6)

**Output:**
- Regression: Predicted time gap to next visit (in years)
- Classification: Probability of visit within time window

### 3. Autoregressive Models

Generates theta_t per patient per time window, then autoregressively predicts theta_t+1 using theta_1...t.

```bash
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method autoregressive \
    --future-steps 3 \
    --window-months 6 \
    --epochs 50 \
    --output autoregressive_predictions.csv
```

**Key Parameters:**
- `--future-steps`: Number of future time steps to predict (default: 3)

**Output:**
- Predicted topic mixtures (theta) for future time points
- Can compute actual disease codes (x_t+1) using phi distributions

## Healthy Prediction

The system can predict if a patient is "healthy" by checking if all disease probabilities fall below a threshold.

```python
# Default threshold is 0.1 (can be optimized via F-1 score)
is_healthy = predict_healthy_threshold(theta, threshold=0.1)
```

**Threshold Optimization:**
- Can be optimized using F-1 score on labeled data
- Adjustable via `--healthy-threshold` parameter

## Implementation Details

### LSTM-based Temporal VAE

The LSTM VAE implements the following architecture:

1. **Encoder**: LSTM layers process temporal sequence of BOW representations
2. **Variational Layer**: Generates mean and log-variance for q(eta_t | X_1...t-1)
3. **Temporal Prior**: Markov chain prior p(theta_t | theta_t-1)
4. **Loss Function**: KL divergence + reconstruction loss

**Mathematical Formulation:**
```
q(eta_t | X_1...t-1) = N(mu_t, sigma_t^2)
p(eta_t | eta_t-1) = N(eta_t-1, delta*I)
KL = KL(q(eta_t | X_1...t-1) || p(eta_t | eta_t-1))
```

### Simple Regression

Uses a feedforward neural network:
- Input: Topic mixture theta_t (K-dimensional)
- Hidden layers: 128 -> 64 neurons with ReLU
- Output: 
  - Regression: Single value (time gap)
  - Classification: Single probability (sigmoid activation)

### Autoregressive Prediction

Implements transformer-based architecture inspired by [TimelyGPT](https://github.com/li-lab-mcgill/TimelyGPT):

1. **Input Projection**: Maps theta to hidden dimension
2. **Positional Encoding**: Adds temporal position information
3. **Transformer Decoder**: Multi-head self-attention with causal masking
4. **Output Projection**: Maps back to theta space with softmax

**Features:**
- Causal masking ensures autoregressive property
- Multi-head attention captures temporal dependencies
- Generates multiple future time steps sequentially

## Example Workflow

### Complete Pipeline

```bash
# 1. Prepare temporal data
# Ensure data is in correct format with timestamps

# 2. Train LSTM model
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method lstm \
    --window-months 6 \
    --epochs 100 \
    --output lstm_results.csv

# 3. Train regression model  
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method regression \
    --task regression \
    --epochs 50 \
    --output regression_results.csv

# 4. Train autoregressive model
python infer_patient_temporal.py ./results/ \
    --temporal-data data/temporal_patient_data.csv \
    --method autoregressive \
    --future-steps 5 \
    --epochs 100 \
    --output autoregressive_results.csv
```

### Performance Comparison

You can compare the three methods by:
1. Training each model on the same temporal dataset
2. Evaluating on held-out test set
3. Comparing metrics:
   - LSTM: KL divergence, reconstruction loss, future prediction accuracy
   - Regression: MSE (regression task), AUC/F1 (classification task)
   - Autoregressive: MSE for theta predictions, accuracy for disease codes

## Technical Notes

### Time Window Aggregation

- Patient data is aggregated into fixed time windows (e.g., 6 months)
- Each window contains all codes occurring within that period
- Windows are used to compute time-specific topic mixtures (theta_t)

### Medication Time Ranges

Medications use categorical timestamps because prescriptions often span periods:
- Value `0`: Approximately 2000-2005
- Value `1`: Approximately 2006-2010
- Value `2`: Approximately 2010-2015
- Value `3`: Approximately 2016-2020

During window aggregation, medications are included if their categorical range overlaps with the window period.

### Model Persistence

Trained temporal models can be saved and loaded:

```python
# Save model
torch.save(model.state_dict(), 'temporal_model.pt')

# Load model
model = TemporalLSTMVAE(K, V)
model.load_state_dict(torch.load('temporal_model.pt'))
```

## Future Extensions

Potential improvements and extensions:

1. **Attention Mechanisms**: Add attention to LSTM to focus on important time points
2. **Multi-task Learning**: Jointly predict multiple outcomes
3. **Survival Analysis**: Incorporate survival modeling for time-to-event prediction
4. **Irregular Time Intervals**: Handle non-uniform time gaps more explicitly
5. **External Features**: Incorporate demographics, lab values, etc.
6. **Uncertainty Quantification**: Provide confidence intervals for predictions

## References

- **MixEHR-SAGE Paper**: Original MixEHR-SAGE methodology
- **MixEHR Nature Communications**: Longitudinal disease prediction approach
- **TimelyGPT**: https://github.com/li-lab-mcgill/TimelyGPT
- **TrajGPT**: Trajectory modeling with transformers
- **Dynamic Topic Models**: Blei & Lafferty (2006)

## Citation

If you use the temporal prediction functionality, please cite:

```bibtex
@article{mixehr_sage,
  title={MixEHR-SAGE: Seed-guided Adaptive Genetic Evidence for Multi-modal Electronic Health Records},
  author={...},
  journal={...},
  year={2024}
}
```

## Support

For questions or issues:
1. Check the main README.md for general MixEHR-SAGE usage
2. Review example temporal data format in `data/example_temporal_data.csv`
3. Open an issue on GitHub with the `temporal-prediction` label
