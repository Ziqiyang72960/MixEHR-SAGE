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
    






