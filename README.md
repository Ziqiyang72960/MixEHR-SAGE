# MixEHR-SAGE

MixEHR-SAGE is a seed-guided Bayesian topic model that fits large-scale, longitundinal, multi-modal EHR data with thousands of phenotypic topics. 

In the seed-guidance, each phenotypic topic is represented as two distributions: (1)a seed-topic distribution over only its set of seed words;
(2) a regular-topic distribution over the entire vocabulary.

Moreover, by associating the topic mixture of each patient (with a certain age) with an age-dependent topic hyperparameters, we can model temporal topic progression in the population. We compute initial topic probabilities using a 2-componet Gaussian mixture model (GMM) on each PheCode count and then used the GMM-inferred posteriors as the initialized topic hyperparameters to guide topic inference.

We generalize our model to multi-modality by the incorporation of diverse types of EHR data. We only guide topic inference in the ICD modality using the expert knowledge from the PheWAS catalog (https://phewascatalog.org/phecodes). It defines each PheCode as a set of ICD codes, which we treat as the seeds for the corresponding phenotypic topic.

To learn our model, we devise a hybrid Bayesian inference methodology in a stochastic manner. We infer the seed-guidance topic assignments by collapsed variational mean-field inference, while infer the age-dependent topic hyperparameters by an amortized inference using a LSTM network. We can compute the collapsed variables
topic mixture memberships $\theta$, regular topic distributions 
$\phi^{r}$, seed topic distributions $\phi^{s}$, with the respective variational expectations.

The proababilistic graphical model of MixEHR-S is shown:

<img src="https://github.com/li-lab-mcgill/MixEHR-Seed/blob/main/figures/PGM.jpg" width="920" height="350">


# Quick Start (Command Line)

The simplest way to run MixEHR-SAGE is using the command-line interface:

```bash
# Run with your EHR data directory
python run_MixEHR.py ./data/

# Run with custom settings
python run_MixEHR.py ./data/ --output ./results/ --epochs 10 --batch-size 500
```

## Command Line Options

```
usage: run_MixEHR.py [-h] [--output OUTPUT] [--store STORE] [--epochs EPOCHS]
                     [--batch-size BATCH_SIZE] [--max-docs MAX_DOCS]
                     [--skip-corpus] [--skip-prior]
                     data_path

positional arguments:
  data_path             Path to the directory containing EHR data and ukbb_metadata.csv

optional arguments:
  -h, --help            show help message and exit
  --output, -o          Directory to store model outputs (default: ./results/)
  --store, -s           Directory to store processed corpus (default: ./store/)
  --epochs, -e          Maximum number of training epochs (default: 5)
  --batch-size, -b      Batch size for training (default: 1000)
  --max-docs, -n        Maximum number of documents to process (default: all)
  --skip-corpus         Skip corpus processing step (use existing corpus)
  --skip-prior          Skip prior computation step (use existing priors)
```

# Online Patient Inference

After training, you can quickly infer topic mixtures (theta/patient risk) for new patients using the `infer_patient.py` script. The inference uses **pre-cached phi distributions** for fast real-time inference.

## Command Line Usage

```bash
# Single file with all modalities (auto-detect which codes belong to which modality)
python infer_patient.py ./results/ --data ./new_patients.csv -o patient_theta.csv

# Separate files for each modality
python infer_patient.py ./results/ --icd ./icd_data.csv --med ./med_data.csv -o theta.csv

# ICD codes only
python infer_patient.py ./results/ --icd ./patient_icd.csv -o theta.csv

# All three modalities from separate files
python infer_patient.py ./results/ \
  --icd ./patient_icd.csv \
  --med ./patient_med.csv \
  --opcs ./patient_opcs.csv \
  --output theta.csv \
  --iterations 10

# Output only top-5 topics per patient
python infer_patient.py ./results/ --icd ./icd.csv -o theta.csv --top-k 5
```

## Inference Options

```
usage: infer_patient.py [-h] [--data DATA] [--icd ICD] [--med MED] [--opcs OPCS]
                        [--output OUTPUT] [--corpus CORPUS] [--iterations ITERATIONS]
                        [--top-k TOP_K] [--explain] [--explain-output EXPLAIN_OUTPUT]
                        [--explain-top-topics N] [--explain-top-codes N]
                        model_path

positional arguments:
  model_path            Path to trained model directory (results folder)

optional arguments:
  --data, -d            Path to single patient data file with all modalities
  --icd                 Path to ICD codes file (CSV, TSV, JSON, TXT)
  --med                 Path to medication/ATC codes file (CSV, TSV, JSON, TXT)
  --opcs                Path to OPCS procedure codes file (CSV, TSV, JSON, TXT)
  --output, -o          Output file for theta values (default: patient_theta.csv)
  --corpus, -c          Path to corpus directory (default: ./store/)
  --iterations, -i      Number of VI iterations (default: 10)
  --top-k, -k           Output only top-k topics per patient
  --explain             Generate ChatGPT explanation prompts
  --explain-output      Output file for explanations (default: patient_explanations.txt)
  --explain-top-topics  Number of top topics in explanations (default: 5)
  --explain-top-codes   Number of top codes per topic (default: 10)
  --phi-csv-icd         External phi CSV for ICD modality (e.g., UKB_phi_icd.csv)
  --phi-csv-med         External phi CSV for medication modality  
  --phi-csv-opcs        External phi CSV for OPCS modality
```

### Using External Phi Distributions

You can use pre-computed phi (word-topic probability) matrices from CSV files instead of the phi learned during training. This is useful when you have phi distributions from external sources or different trained models:

```bash
# Infer with external phi from CSV files
python infer_patient.py ./results/ --icd ./patient_icd.csv \
  --phi-csv-icd UKB_phi_icd.csv \
  --output theta.csv

# Multiple modalities with external phi
python infer_patient.py ./results/ \
  --icd ./icd.csv \
  --med ./med.csv \
  --phi-csv-icd UKB_phi_icd.csv \
  --phi-csv-med UKB_phi_med.csv \
  --output theta.csv
```

**External Phi Format:**
- CSV files with no header
- Each row is a word/code (in vocabulary order)
- Each column is a topic  
- Values are probabilities (summing to 1 for each topic)
- Example: `UKB_phi_icd.csv` has shape `(V_icd, K)` where V_icd is ICD vocabulary size and K is number of topics

### ChatGPT Explanations

Generate detailed explanation prompts for ChatGPT to interpret patient phenotype probabilities. **Each patient gets a separate entry** for easy use:

```bash
# Generate explanations with inference (TXT format - one section per patient)
python infer_patient.py ./results/ --icd ./patient_icd.csv \
  --explain --explain-output explanations.txt

# Save as CSV (one row per patient)
python infer_patient.py ./results/ --icd ./patient_icd.csv \
  --explain --explain-output explanations.csv

# Save as JSON (one object per patient)
python infer_patient.py ./results/ --icd ./patient_icd.csv \
  --explain --explain-output explanations.json

# Customize number of topics and codes in explanations
python infer_patient.py ./results/ --icd ./icd.csv \
  --explain --explain-top-topics 3 --explain-top-codes 15
```

**Output formats:**
- **TXT**: One section per patient, separated by dividers (easy to copy individual prompts)
- **CSV**: One row per patient with `patient_id` and `prompt` columns
- **JSON**: Array of objects with `patient_id` and `prompt` fields

The generated prompts include:
1. **Top K inferred topic mixtures (θ)** - Patient's probability distribution over **PheCodes with phenotype names**
2. **Patient medical records** - Actual codes observed for the patient
3. **Top codes for each topic (φ)** - Most probable codes defining each **PheCode phenotype**

Copy the generated prompts to ChatGPT for human-readable explanations of the patient's risk profile.

## Input Data Format

Each input file should have at minimum:
- `SUBJECT_ID`: patient identifier
- `code` (or specified word column): medical codes

Example CSV for ICD codes:
```csv
SUBJECT_ID,code
new_patient_1,E11.9
new_patient_1,I10
new_patient_2,J44.1
```

Example CSV for medications:
```csv
SUBJECT_ID,code
new_patient_1,A02BC01
new_patient_1,C07AA05
new_patient_2,N02BE01
```

## Programmatic Inference (Fast Online API)

You can also use the fast inference API directly in Python:

```python
from MixEHR_SAGE import MixEHR_SAGE
from corpus import Corpus
import torch

# Load trained model (automatically caches phi for fast inference)
corpus = Corpus.read_corpus_from_directory('./store/')
seeds_matrix = torch.load('./phecode_mapping/seed_topic_matrix.pt')
model = MixEHR_SAGE.load_trained_model(
    './results/', corpus, seeds_matrix, corpus.modalities
)

# --- INFERENCE WITH ANY SUBSET OF MODALITIES ---

# Patient with ONLY ICD codes (no medication or procedure data)
theta = model.infer_theta_by_modality({'icd': {0: 1, 5: 2}})

# Patient with ICD + medication codes
theta = model.infer_theta_by_modality({
    'icd': {0: 1, 5: 2},
    'med': {10: 1, 15: 3}
})

# Patient with all three modalities (ICD, medication, procedures)
theta = model.infer_theta_by_modality({
    'icd': {0: 1},
    'med': {10: 1},
    'opcs': {5: 1}
})

# Get top risk topics
top_k = torch.topk(theta, k=5)
print(f"Top 5 topics: {top_k.indices.tolist()}")
print(f"Top 5 values: {top_k.values.tolist()}")

# --- BATCH INFERENCE ---
patients = [
    {'icd': {0: 1}},                          # Patient 1: only ICD
    {'icd': {5: 2}, 'med': {10: 1}},          # Patient 2: ICD + med
    {'icd': {0: 1}, 'med': {15: 1}, 'opcs': {3: 1}}  # Patient 3: all modalities
]
thetas = model.infer_theta_batch_by_modality(patients, num_iterations=5)

# --- INFERENCE WITH EXTERNAL PHI DISTRIBUTIONS ---

# Load phi distributions from external CSV files (e.g., from another trained model)
phi_dists = MixEHR_SAGE.load_phi_from_csv({
    'icd': 'UKB_phi_icd.csv',
    'med': 'UKB_phi_med.csv',
    'opcs': 'UKB_phi_opcs.csv'
}, ['icd', 'med', 'opcs'])

# Infer theta using external phi instead of model's learned phi
theta = model.infer_theta_with_external_phi(
    {'icd': {0: 1, 5: 2}},
    phi_dists,
    num_iterations=10
)

# This allows you to use pre-computed phi matrices from different sources
# without retraining the model
```

# Dynamic Modality Support

MixEHR-SAGE now supports **any number of modalities** (1 or more). The modalities are dynamically read from your `ukbb_metadata.csv` file. The first modality listed is treated as the **guided modality** (typically ICD codes with PheCode mappings).

## Adding or Removing Modalities

To use a different number of modalities, simply modify your `ukbb_metadata.csv` file:

**Example with 2 modalities:**
```csv
index,path,word_column
icd,./data/icd_data.csv,code
med,./data/med_data.csv,code
```

**Example with 4 modalities:**
```csv
index,path,word_column
icd,./data/icd_data.csv,code
med,./data/med_data.csv,code
opcs,./data/opcs_data.csv,code
lab,./data/lab_data.csv,code
```

# Dataset Preparation

We evaluated MixEHR-SAGE on the extracted clinical dataset from UKB, and MIMIC-III database. 

For these datasets, we organize each data type of EHRs into one single input file (such as ICD codes). Moreover, the temporal information (such as a patient's age group) is listed in a separate file. 
 
In the path "MixEHR_SAGE/data/", we have extracted a toy data from UKB database including ICD, ATC medication code, OPCS-4 procedure code three modalities.

## Supported File Formats

MixEHR-SAGE supports multiple input file formats:
- **CSV** (.csv) - Comma-separated values
- **TSV** (.tsv) - Tab-separated values
- **JSON** (.json) - JSON format (records orientation)
- **TXT** (.txt) - Text files (auto-detects tab or comma separator)

Both the metadata file and data files can use any of these formats.

## Required Files

Your data directory must contain:

1. **ukbb_metadata** - Metadata file defining your modalities (supports .csv, .tsv, .json, .txt):
   
   **CSV format (ukbb_metadata.csv):**
   ```csv
   index,path,word_column
   icd,synthetic_icd.csv,code
   med,synthetic_med.csv,code
   opcs,synthetic_opcs.csv,code
   ```

   **TSV format (ukbb_metadata.tsv):**
   ```
   index	path	word_column
   icd	synthetic_icd.tsv	code
   med	synthetic_med.tsv	code
   ```

   **JSON format (ukbb_metadata.json):**
   ```json
   [
     {"index": "icd", "path": "synthetic_icd.json", "word_column": "code"},
     {"index": "med", "path": "synthetic_med.json", "word_column": "code"}
   ]
   ```

2. **Data files** for each modality referenced in the metadata (supports .csv, .tsv, .json, .txt):

   - **Guided modality (first row)** - Must have columns: SUBJECT_ID, code column, PheCode
     ```
     Headers: SUBJECT_ID,code,PheCode
     ```

   - **Other modalities** - Must have columns: SUBJECT_ID, code column
     ```
     Headers: SUBJECT_ID,code
     ```

## Example Data Format

- icd_toy_data.csv has 3 columns rows: patient ID, ICD code, PheCode

                            Headers:SUBJECT_ID,ICD10,PheCode

- atc_toy_data.csv has 2 columns rows: patient ID, drug code

                            Headers:SUBJECT_ID,ATC_CODE
			    
- opcs_toy_data.csv has 2 columns rows: patient ID, OPCS-4 procedure code

                            Headers:SUBJECT_ID,OPCS4_CODE
  
              
# Code Description (Step-by-Step)

If you prefer to run the pipeline step by step instead of using the CLI:

## STEP 1: Process Dataset and extract seeds

The input data files include the multi-modal EHR data in `./data`. We have to transform the inputs into the built-in, readable data structure "Corpus" class. Moreover, we need extract the seed ICD codes of PheCodes from the dataset.

You can use `corpus.py` to transform the raw inputs into a runnable Corpus data structure and generate the seeds of phenotypic topics. 

Place dataset to the specific path `./data/` and then run the following code:

```bash
python corpus.py process ./data/ ./store/
# Or with max documents limit:
python corpus.py process -n 150 ./data/ ./store/
```
    
you also need to split the dataset into train/validation/test subset. The data path and detailed split ratio could be edited:

```bash
python corpus.py split store/test/ store/
```
	
The extracted PheCode-ICD mapping is located at path `./phecode_mapping/all_seed_topic_matrix.pt`, where each row and column represents a word and a topic, respectively.


## STEP 2: Compute initial topic prior using GMM

For each phenotypic topic, count the frequency of its seed ICD codes under the PheCode by runing the file `./guide_prior/get_doc_phecode.py`

For each phenotypic topic, train a 2-component GMM for the PheCode counts over all patients to have predictive probabilities (initial topic prior) by running the file `./guide_prior/get_prior_GMM.py`

Compute the initial sufficient statistics by running the file: `./guide_prior/get_token_counts.py`

## STEP 3: Topic Modelling

We can run `./main.py` to perform seed-guided topic modelling on the extracted train data:

```bash
python main.py ./store/ ./result/ -epoch 5 -batch_size 1000
```
    
The topic hyperparameters of regular topics and seed topics need to fine-tune by minimizing the held-out negative log-likelihood on the validation set. We then apply MixEHR-Seed with the estiated hyperparameters on the train set. 
 

## STEP 4: Evalutions

The learned parameters are saved at `./parameters/`, we then evaluate the topic interpretability, phenotype prediction, and temporal disease progression analysis. 
    






