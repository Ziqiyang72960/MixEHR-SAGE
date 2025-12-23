# Example Temporal Data

This directory contains example data demonstrating mixed temporal granularity for training the Markov chain model.

## Data Description

### Patients
- **5 synthetic patients** (IDs: 1001-1005)
- **3-4 visits each** spanning several months
- **3 modalities**: ICD diagnoses, Medications, OPCS procedures

### Patient Conditions
- **Patient 1001**: Asthma (J45.0) + Diabetes (E11.9) - Chronic management
- **Patient 1002**: Ischemic heart disease (I25.1) + Heart failure (I50.0) - Cardiovascular progression
- **Patient 1003**: Lung cancer (C34.9) - Oncology treatment progression
- **Patient 1004**: Depression (F32.9) - Mental health treatment
- **Patient 1005**: Chronic kidney disease (N18.3→N18.5) - Renal disease progression

## File Formats

### icd_temporal.csv (Dated)
```
patient_id,date,code
1001,2020-01-15,J45.0  # Asthma diagnosis on Jan 15
1001,2020-03-20,I10    # Hypertension on Mar 20
```
- **30 records** across 5 patients
- **Specific dates** define temporal grid
- **ICD-10 codes** for diagnoses

### medication_temporal.csv (Time bins)
```
patient_id,time_bin,code
1001,0,N02BE01  # Ibuprofen at baseline
1001,1,A10BA02  # Metformin at 1st visit
```
- **50 records** across 5 patients
- **4 time bins**: 0=baseline, 1=1st visit, 2=2nd, 3=3rd
- **ATC codes** for medications

### opcs_temporal.csv (Dated)
```
patient_id,date,code
1001,2020-01-15,E85.1  # Respiratory function test
1002,2019-11-03,K40.2  # Cardiac catheterization
```
- **19 records** across 5 patients
- **Specific dates** match ICD visit dates
- **OPCS-4 codes** for procedures

## Temporal Alignment

### Patient 1001 Example

**ICD dates**: 2020-01-15, 2020-03-20, 2020-07-10

**Medication bins**: 0, 1, 2, 3

**Aligned visits**:
1. **Visit 0 (2020-01-15)**:
   - ICD: J45.0, E11.9
   - Med (bin 0): N02BE01, A10BA02
   - OPCS: E85.1

2. **Visit 1 (2020-03-20)**:
   - ICD: I10, E78.5
   - Med (bin 1): N02BE01, A10BA02, C03CA01
   - OPCS: E85.1

3. **Visit 2 (2020-07-10)**:
   - ICD: J45.0, Z23
   - Med (bin 2): N02BE01, A10BA02, C03CA01
   - OPCS: E85.1

4. **Visit 3 (implied)**:
   - Med (bin 3): A10BA02, C03CA01

## Usage

### Training
```bash
# From repository root
python train_temporal.py ./example_data/ --enable-temporal --epochs 5
```

### Expected Output
```
Processing 5 patients...
Processed 5 patients with temporal data
  Patients with sequences: 5
  Average visits per patient: 3.4

Training for 5 epochs...
Epoch 1/5
  Batch 1: Avg KL = 15.234
Epoch 1 complete. Avg KL: 14.987

Saved temporal theta to ./results/temporal_theta.pt
Saved model to ./results/model.pt
Saved results to ./results/patient_theta_results.pkl
```

### Analyzing Results
```python
import pickle
import torch

# Load results
with open('./results/patient_theta_results.pkl', 'rb') as f:
    results = pickle.load(f)

theta = results['theta_temporal']  # (5, 10, 50) - 5 patients, 10 max visits, 50 topics

# Patient 1001 trajectory
print(f"Patient 1001 visits: {results['patient_metadata'][1001]['num_visits']}")
print(f"Dominant topics: {theta[0, :3, :].argmax(axis=1)}")  # First 3 visits
```

## Disease Progression Patterns

### Patient 1002 (Cardiovascular)
```
Visit 1: I25.1 (Ischemic heart) + A10BA02, C07AB02 → Topic ~5 (Cardiac)
Visit 2: I25.1, I50.0 (Heart failure added) → Topic ~5 (Cardiac)
Visit 3: I50.0 (Heart failure dominates) + more meds → Topic ~12 (Heart failure)
```
**Pattern**: Progression from stable CAD to heart failure

### Patient 1005 (Renal)
```
Visit 1: N18.3 (CKD Stage 3) → Topic ~8 (Early CKD)
Visit 2: N18.4 (CKD Stage 4) → Topic ~15 (Advanced CKD)
Visit 3: N18.5 (CKD Stage 5) + dialysis (Z49.1) → Topic ~22 (ESRD)
```
**Pattern**: Clear CKD progression through stages

## Data Generation

To create your own example data:

```python
# See temporal_markov_utils.py
from temporal_markov_utils import PatientSequenceGenerator

gen = PatientSequenceGenerator(
    vocab_sizes=[1000, 500, 300],  # ICD, Med, OPCS
    num_modalities=3
)

patient_sequences, metadata = gen.generate_synthetic_patient_sequences(
    num_patients=100,
    min_visits=2,
    max_visits=10,
    progression_strength=0.4
)
```

## Notes

1. **Codes are synthetic**: Real ICD/ATC/OPCS codes used for realism
2. **Dates are consistent**: All within 2019-2021 timeframe
3. **Progression modeled**: Severity increases (e.g., CKD Stage 3→5)
4. **Medications align**: Drugs match conditions (insulin for diabetes, etc.)

## References

- **ICD-10**: International Classification of Diseases
- **ATC**: Anatomical Therapeutic Chemical classification
- **OPCS-4**: Office of Population Censuses and Surveys Classification
