#!/usr/bin/env python3
"""
Fast online inference for new patient risk (theta) using trained MixEHR-SAGE model.

Usage:
    # Single file with all modalities
    python infer_patient.py ./results/ --data ./new_patients.csv --output patient_theta.csv

    # Separate files for each modality
    python infer_patient.py ./results/ --icd icd_data.csv --med med_data.csv --output theta.csv

Examples:
    # Single file inference
    python infer_patient.py ./results/ --data ./new_patients.csv -o patient_theta.csv

    # ICD codes only
    python infer_patient.py ./results/ --icd ./patient_icd.csv -o theta.csv

    # ICD + medications
    python infer_patient.py ./results/ --icd ./icd.csv --med ./med.csv -o theta.csv

    # All modalities from separate files
    python infer_patient.py ./results/ --icd ./icd.csv --med ./med.csv --opcs ./opcs.csv -o theta.csv
"""
import argparse
import os
import sys
import pickle
import torch
import pandas as pd
import numpy as np
import json

from corpus import Corpus, read_data_file
from MixEHR_SAGE import MixEHR_SAGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab_mappings(mapping_path='./mapping/'):
    """Load vocabulary mappings for each modality."""
    vocab_mappings = {}
    for f in os.listdir(mapping_path):
        if f.endswith('_vocab_ids.pkl'):
            modality = f.replace('_vocab_ids.pkl', '')
            with open(os.path.join(mapping_path, f), 'rb') as handle:
                vocab_mappings[modality] = pickle.load(handle)
    return vocab_mappings


def convert_patient_data_to_bow(patient_df, vocab_mappings, modality_list, word_column='code'):
    """
    Convert patient data DataFrame to bag-of-words format.
    
    Args:
        patient_df: DataFrame with columns SUBJECT_ID and word_column
        vocab_mappings: dict of {modality: {word: id}}
        modality_list: list of modality names in order
        word_column: column name containing the codes/words
    
    Returns:
        dict: {patient_id: [{word_id: freq}, ...]} for each modality
    """
    patients_bow = {}
    
    for _, row in patient_df.iterrows():
        patient_id = str(row['SUBJECT_ID'])  # Convert to string
        code = str(row[word_column]).strip()  # Convert to string and strip whitespace
        
        # Skip NaN or empty codes
        if code == 'nan' or code == '' or pd.isna(row[word_column]):
            continue
        
        if patient_id not in patients_bow:
            patients_bow[patient_id] = [{} for _ in modality_list]
        
        # Try to find which modality this code belongs to
        for m, modality in enumerate(modality_list):
            if modality in vocab_mappings and code in vocab_mappings[modality]:
                word_id = vocab_mappings[modality][code]
                if word_id not in patients_bow[patient_id][m]:
                    patients_bow[patient_id][m][word_id] = 0
                patients_bow[patient_id][m][word_id] += 1
                break
    
    return patients_bow


def convert_modality_files_to_bow(modality_files, vocab_mappings, modality_list, word_column='code'):
    """
    Convert separate modality files to bag-of-words format.
    
    Args:
        modality_files: dict of {modality_name: file_path}
        vocab_mappings: dict of {modality: {word: id}}
        modality_list: list of modality names in order
        word_column: column name containing the codes/words
    
    Returns:
        dict: {patient_id: [{word_id: freq}, ...]} for each modality
    """
    patients_bow = {}
    
    for modality_name, file_path in modality_files.items():
        if file_path is None or not os.path.exists(file_path):
            continue
            
        # Get modality index
        if modality_name not in modality_list:
            print(f"ERROR: Modality '{modality_name}' not found in trained model.")
            print(f"  Available modalities in model: {modality_list}")
            print(f"  The model was only trained with these modalities.")
            print(f"  To use '{modality_name}', you need to:")
            print(f"    1. Add '{modality_name}' to data/ukbb_metadata.csv")
            print(f"    2. Retrain the model with: python run_MixEHR.py ./data/")
            continue
        m = modality_list.index(modality_name)
        
        # Read modality data
        df = read_data_file(file_path)
        print(f"Loaded {len(df)} records from {file_path} for modality '{modality_name}'")
        
        # Show sample codes from input file
        sample_codes = df[word_column].dropna().astype(str).str.strip().unique()[:5]
        print(f"  Sample codes in input file: {list(sample_codes)}")
        
        # Check if vocabulary exists
        if modality_name not in vocab_mappings:
            print(f"ERROR: No vocabulary mapping found for modality '{modality_name}'")
            print(f"  Available vocabularies: {list(vocab_mappings.keys())}")
            print(f"  Make sure {modality_name}_vocab_ids.pkl exists in the mapping directory")
            continue
        
        # Show sample vocabulary codes for comparison
        vocab_sample = list(vocab_mappings[modality_name].keys())[:5]
        print(f"  Sample codes in '{modality_name}' vocabulary: {vocab_sample}")
        
        # Track statistics
        total_codes = 0
        matched_codes = 0
        unknown_codes = set()
        
        # Process each row
        for _, row in df.iterrows():
            patient_id = str(row['SUBJECT_ID'])  # Convert to string
            code = str(row[word_column]).strip()  # Convert to string and strip whitespace
            
            # Skip NaN or empty codes
            if code == 'nan' or code == '' or pd.isna(row[word_column]):
                continue
                
            total_codes += 1
            
            if patient_id not in patients_bow:
                patients_bow[patient_id] = [{} for _ in modality_list]
            
            # Look up word ID in vocabulary
            if code in vocab_mappings[modality_name]:
                word_id = vocab_mappings[modality_name][code]
                if word_id not in patients_bow[patient_id][m]:
                    patients_bow[patient_id][m][word_id] = 0
                patients_bow[patient_id][m][word_id] += 1
                matched_codes += 1
            else:
                unknown_codes.add(code)
        
        # Print statistics
        print(f"  Modality '{modality_name}': {matched_codes}/{total_codes} codes matched in vocabulary")
        if unknown_codes:
            print(f"  WARNING: {len(unknown_codes)} unique codes not found in vocabulary")
            if len(unknown_codes) <= 10:
                print(f"  Unknown codes: {sorted(list(unknown_codes))}")
            else:
                print(f"  First 10 unknown codes: {sorted(list(unknown_codes))[:10]}")
    
    return patients_bow


def infer_from_modality_files(model, modality_files, vocab_mappings, modality_list,
                               word_column='code', num_iterations=10, return_bow=False, external_phi=None):
    """
    Infer theta for patients from separate modality files.
    
    Args:
        model: trained MixEHR_SAGE model
        modality_files: dict of {modality_name: file_path}
        vocab_mappings: vocabulary mappings
        modality_list: list of modality names
        word_column: column name for codes
        num_iterations: VI iterations
        return_bow: if True, also return patients_bow dict
        external_phi: optional list of external phi distributions from CSV files
    
    Returns:
        DataFrame with patient_id and theta values
        (optionally) dict of patients_bow if return_bow=True
    """
    # Convert to BOW format
    patients_bow = convert_modality_files_to_bow(
        modality_files, vocab_mappings, modality_list, word_column
    )
    
    if not patients_bow:
        print("ERROR: No valid patient data found in provided files.")
        print("Possible issues:")
        print("  1. No codes matched the vocabulary")
        print("  2. Vocabulary mapping files are missing")
        print("  3. Code column name is incorrect (use --word-column if not 'code')")
        if return_bow:
            return pd.DataFrame(columns=['patient_id']), {}
        return pd.DataFrame(columns=['patient_id'])
    
    print(f"\nProcessing {len(patients_bow)} patients for inference...")
    
    # Infer theta for each patient
    results = []
    for patient_id, bow in patients_bow.items():
        # Check if patient has any data
        has_data = any(len(bow_m) > 0 for bow_m in bow)
        if not has_data:
            print(f"WARNING: Patient {patient_id} has no valid codes after vocabulary matching")
            continue
        
        # Convert bow list to dict format for external phi method
        patient_data = {modality: bow[m] for m, modality in enumerate(modality_list)}
        
        # Use external phi if provided, otherwise use model's trained phi
        if external_phi is not None:
            theta = model.infer_theta_with_external_phi(patient_data, external_phi, num_iterations=num_iterations)
        else:
            theta = model.infer_theta_fast(bow, num_iterations=num_iterations)
            
        theta_np = theta.cpu().numpy()
        result = {'patient_id': patient_id}
        for k in range(len(theta_np)):
            result[f'topic_{k}'] = theta_np[k]
        results.append(result)
    
    if return_bow:
        return pd.DataFrame(results), patients_bow
    return pd.DataFrame(results)


def infer_from_file(model, patient_file, vocab_mappings, modality_list, 
                    word_column='code', num_iterations=10, return_bow=False, external_phi=None):
    """
    Infer theta for patients from a data file.
    
    Args:
        model: trained MixEHR_SAGE model
        patient_file: path to patient data file (CSV, TSV, JSON, TXT)
        vocab_mappings: vocabulary mappings
        modality_list: list of modality names
        word_column: column name for codes
        num_iterations: VI iterations
        return_bow: if True, also return patients_bow dict
        external_phi: optional list of external phi distributions from CSV files
    
    Returns:
        DataFrame with patient_id and theta values
        (optionally) dict of patients_bow if return_bow=True
    """
    # Read patient data
    patient_df = read_data_file(patient_file)
    
    # Convert to BOW format
    patients_bow = convert_patient_data_to_bow(
        patient_df, vocab_mappings, modality_list, word_column
    )
    
    # Infer theta for each patient
    results = []
    for patient_id, bow in patients_bow.items():
        # Convert bow list to dict format for external phi method
        patient_data = {modality: bow[m] for m, modality in enumerate(modality_list)}
        
        # Use external phi if provided, otherwise use model's trained phi
        if external_phi is not None:
            theta = model.infer_theta_with_external_phi(patient_data, external_phi, num_iterations=num_iterations)
        else:
            theta = model.infer_theta_fast(bow, num_iterations=num_iterations)
            
        theta_np = theta.cpu().numpy()
        result = {'patient_id': patient_id}
        for k in range(len(theta_np)):
            result[f'topic_{k}'] = theta_np[k]
        results.append(result)
    
    if return_bow:
        return pd.DataFrame(results), patients_bow
    return pd.DataFrame(results)


def load_phecode_definitions(phecode_def_path='./mapping/phecode_definitions1.2.csv'):
    """Load PheCode definitions (PheCode -> Phenotype name mapping)."""
    try:
        phecode_df = pd.read_csv(phecode_def_path)
        # Create dictionary: phecode -> phenotype name
        phecode_dict = dict(zip(phecode_df['phecode'].astype(str), phecode_df['phenotype']))
        return phecode_dict
    except Exception as e:
        print(f"Warning: Could not load PheCode definitions from {phecode_def_path}: {e}")
        return {}


def load_phecode_ids_mapping(phecode_ids_path='./mapping/phecode_ids.pkl'):
    """Load PheCode IDs mapping (topic index -> PheCode)."""
    try:
        with open(phecode_ids_path, 'rb') as f:
            phecode_ids = pickle.load(f)
        # Create inverse mapping: topic_idx -> phecode
        if isinstance(phecode_ids, dict):
            # If it's already {phecode: idx}, invert it
            inv_phecode_ids = {idx: str(phecode) for phecode, idx in phecode_ids.items()}
        else:
            # If it's a list/array, use index as topic_idx
            inv_phecode_ids = {i: str(phecode) for i, phecode in enumerate(phecode_ids)}
        return inv_phecode_ids
    except Exception as e:
        print(f"Warning: Could not load PheCode IDs from {phecode_ids_path}: {e}")
        return {}


def generate_chatgpt_explanation_prompt(patient_id, patient_bow, theta, model, vocab_mappings, 
                                         modality_list, top_k_topics=5, top_n_codes=10):
    """
    Generate a ChatGPT prompt to explain inferred phenotype probabilities for a patient.
    
    Args:
        patient_id: patient identifier
        patient_bow: patient's bag-of-words data (list of dicts for each modality)
        theta: inferred topic mixture (K-dimensional tensor or array)
        model: trained MixEHR_SAGE model
        vocab_mappings: dict of {modality: {code: word_id}}
        modality_list: list of modality names
        top_k_topics: number of top topics to include (default: 5)
        top_n_codes: number of top codes per topic to include (default: 10)
    
    Returns:
        str: formatted ChatGPT prompt
    """
    # Load PheCode definitions and mappings
    phecode_dict = load_phecode_definitions()
    inv_phecode_ids = load_phecode_ids_mapping()
    # Convert theta to numpy if needed
    if torch.is_tensor(theta):
        theta_np = theta.cpu().numpy()
    else:
        theta_np = np.array(theta)
    
    # Get top K topics
    top_topic_indices = np.argsort(theta_np)[::-1][:top_k_topics]
    top_topic_probs = theta_np[top_topic_indices]
    
    # Create reverse vocabulary mappings (word_id -> code)
    reverse_vocabs = {}
    for modality, vocab in vocab_mappings.items():
        reverse_vocabs[modality] = {word_id: code for code, word_id in vocab.items()}
    
    # Get patient records by modality
    patient_records_by_modality = {}
    for m, modality_name in enumerate(modality_list):
        if modality_name not in vocab_mappings:
            continue
        codes = []
        for word_id, freq in patient_bow[m].items():
            if word_id in reverse_vocabs[modality_name]:
                code = reverse_vocabs[modality_name][word_id]
                codes.extend([code] * freq)
        if codes:
            patient_records_by_modality[modality_name] = codes
    
    # Get top codes for each of the top K topics based on phi
    topic_top_codes = {}
    for topic_idx in top_topic_indices:
        topic_codes_by_modality = {}
        
        for m, modality_name in enumerate(modality_list):
            if modality_name not in vocab_mappings:
                continue
                
            # Get phi for this modality
            phi = model.get_phi(modality=m)  # V x K matrix
            phi_np = phi.cpu().numpy()
            
            # Get top N codes for this topic in this modality
            topic_phi = phi_np[:, topic_idx]
            top_code_indices = np.argsort(topic_phi)[::-1][:top_n_codes]
            top_codes_info = []
            
            for code_idx in top_code_indices:
                if code_idx in reverse_vocabs[modality_name]:
                    code = reverse_vocabs[modality_name][code_idx]
                    prob = topic_phi[code_idx]
                    top_codes_info.append(f"{code} (φ={prob:.4f})")
            
            if top_codes_info:
                topic_codes_by_modality[modality_name] = top_codes_info
        
        topic_top_codes[topic_idx] = topic_codes_by_modality
    
    # Build the prompt
    prompt = f"""I have a patient (ID: {patient_id}) and I used a topic modeling approach (MixEHR-SAGE) to infer their phenotype risk profile. Please help me interpret the results.

**Input 1: Top {top_k_topics} Inferred Topic Mixtures (θ)**
These represent the patient's probability distribution over latent phenotypes/disease topics:
"""
    
    for i, (topic_idx, prob) in enumerate(zip(top_topic_indices, top_topic_probs), 1):
        # Get PheCode and phenotype name for this topic
        topic_label = f"Topic {topic_idx}"
        if topic_idx in inv_phecode_ids:
            phecode = inv_phecode_ids[topic_idx]
            if phecode in phecode_dict:
                phenotype_name = phecode_dict[phecode]
                topic_label = f"PheCode {phecode} ({phenotype_name})"
            else:
                topic_label = f"PheCode {phecode}"
        
        prompt += f"\n  {topic_label}: {prob:.4f} ({prob*100:.2f}%)"
    
    prompt += "\n\n**Input 2: Patient's Medical Records**\n"
    prompt += "The actual medical codes observed for this patient:\n"
    
    for modality_name, codes in patient_records_by_modality.items():
        unique_codes = list(set(codes))
        if len(unique_codes) > 20:
            prompt += f"\n  {modality_name.upper()}: {', '.join(unique_codes[:20])} ... ({len(unique_codes)} total unique codes)"
        else:
            prompt += f"\n  {modality_name.upper()}: {', '.join(unique_codes)}"
    
    prompt += "\n\n**Input 3: Top Codes for Each Topic (φ)**\n"
    prompt += f"These are the top {top_n_codes} most probable codes for each of the patient's dominant topics:\n"
    
    for topic_idx in top_topic_indices:
        # Get PheCode label for this topic
        topic_label = f"Topic {topic_idx}"
        if topic_idx in inv_phecode_ids:
            phecode = inv_phecode_ids[topic_idx]
            if phecode in phecode_dict:
                phenotype_name = phecode_dict[phecode]
                topic_label = f"PheCode {phecode} ({phenotype_name})"
            else:
                topic_label = f"PheCode {phecode}"
        
        prompt += f"\n  {topic_label} (probability {theta_np[topic_idx]:.4f}):"
        if topic_idx in topic_top_codes:
            for modality_name, codes_info in topic_top_codes[topic_idx].items():
                prompt += f"\n    {modality_name.upper()}: {', '.join(codes_info)}"
        else:
            prompt += "\n    (No significant codes)"
    
    prompt += "\n\n**Question:**"
    prompt += "\nBased on these three pieces of information, please explain:"
    prompt += "\n1. What phenotypes or disease patterns do the top topics likely represent?"
    prompt += "\n2. How well does the patient's actual medical history align with their inferred topic mixtures?"
    prompt += "\n3. What insights can we draw about this patient's health condition and risk profile?"
    prompt += "\n4. Are there any surprising findings or inconsistencies that warrant further investigation?"
    
    return prompt


def main():
    parser = argparse.ArgumentParser(
        description="Fast online inference for new patient risk using trained MixEHR-SAGE model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Single file with all modalities (auto-detect which codes belong to which modality)
    python infer_patient.py ./results/ --data ./new_patients.csv -o patient_theta.csv

    # Separate files for each modality
    python infer_patient.py ./results/ --icd ./icd_data.csv --med ./med_data.csv -o theta.csv

    # ICD codes only
    python infer_patient.py ./results/ --icd ./patient_icd.csv -o theta.csv

    # All three modalities from separate files
    python infer_patient.py ./results/ --icd ./icd.csv --med ./med.csv --opcs ./opcs.csv -o theta.csv --iterations 10

Input Data Format:
    Each input file should have at minimum:
    - SUBJECT_ID: patient identifier
    - code (or specified word_column): medical codes

    Example CSV:
        SUBJECT_ID,code
        patient1,E11.9
        patient1,I10
        patient2,E11.0
        """
    )
    parser.add_argument(
        'model_path',
        help='Path to directory containing trained model (results folder)'
    )
    parser.add_argument(
        '--data', '-d',
        default=None,
        help='Path to single patient data file with all modalities (CSV, TSV, JSON, TXT)'
    )
    parser.add_argument(
        '--icd',
        default=None,
        help='Path to ICD codes file (CSV, TSV, JSON, TXT)'
    )
    parser.add_argument(
        '--med',
        default=None,
        help='Path to medication/ATC codes file (CSV, TSV, JSON, TXT)'
    )
    parser.add_argument(
        '--opcs',
        default=None,
        help='Path to OPCS procedure codes file (CSV, TSV, JSON, TXT)'
    )
    parser.add_argument(
        '--output', '-o',
        default='patient_theta.csv',
        help='Output file path for theta values (default: patient_theta.csv)'
    )
    parser.add_argument(
        '--corpus', '-c',
        default='./store/',
        help='Path to corpus directory (default: ./store/)'
    )
    parser.add_argument(
        '--seed-matrix', '-s',
        default='./phecode_mapping/seed_topic_matrix.pt',
        help='Path to seed topic matrix (default: ./phecode_mapping/seed_topic_matrix.pt)'
    )
    parser.add_argument(
        '--mapping', '-m',
        default='./mapping/',
        help='Path to vocabulary mapping directory (default: ./mapping/)'
    )
    parser.add_argument(
        '--guide-prior', '-g',
        default='./guide_prior/',
        help='Path to guide prior directory (default: ./guide_prior/)'
    )
    parser.add_argument(
        '--word-column', '-w',
        default='code',
        help='Column name for codes in patient data (default: code)'
    )
    parser.add_argument(
        '--iterations', '-i',
        type=int,
        default=10,
        help='Number of VI iterations for inference (default: 10)'
    )
    parser.add_argument(
        '--top-k', '-k',
        type=int,
        default=None,
        help='Only output top-k topics per patient (default: all)'
    )
    parser.add_argument(
        '--explain',
        action='store_true',
        help='Generate ChatGPT explanation prompts for each patient'
    )
    parser.add_argument(
        '--explain-output',
        default='patient_explanations.txt',
        help='Output file for ChatGPT explanation prompts. Supports .txt, .csv, .json formats. Each patient gets a separate entry. (default: patient_explanations.txt)'
    )
    parser.add_argument(
        '--explain-top-topics',
        type=int,
        default=5,
        help='Number of top topics to include in explanations (default: 5)'
    )
    parser.add_argument(
        '--explain-top-codes',
        type=int,
        default=10,
        help='Number of top codes per topic to include in explanations (default: 10)'
    )
    parser.add_argument(
        '--phi-csv-icd',
        default=None,
        help='External phi CSV file for ICD modality (e.g., UKB_phi_icd.csv). Use this to infer with pre-computed phi instead of trained model phi.'
    )
    parser.add_argument(
        '--phi-csv-med',
        default=None,
        help='External phi CSV file for medication modality (e.g., UKB_phi_med.csv)'
    )
    parser.add_argument(
        '--phi-csv-opcs',
        default=None,
        help='External phi CSV file for OPCS modality (e.g., UKB_phi_opcs.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        sys.exit(1)
    
    # Validate that at least one data source is provided
    has_modality_files = args.icd or args.med or args.opcs
    if not has_modality_files and not args.data:
        print("Error: You must provide either --data for a single file or --icd/--med/--opcs for separate modality files.")
        sys.exit(1)
    
    print(f"Loading model from {args.model_path}...")
    
    # Load corpus for vocabulary info
    corpus = Corpus.read_corpus_from_directory(args.corpus)
    modality_list = corpus.modalities
    print(f"Modalities: {modality_list}")
    
    # Load seed topic matrix
    seeds_topic_matrix = torch.load(args.seed_matrix, map_location=device, weights_only=False)
    print(f"Loaded seed matrix with shape: {seeds_topic_matrix.shape}")
    
    # Load trained model
    model = MixEHR_SAGE.load_trained_model(
        args.model_path,
        corpus,
        seeds_topic_matrix,
        modality_list,
        guide_prior_path=args.guide_prior
    )
    model = model.to(device)
    
    # Load vocabulary mappings
    vocab_mappings = load_vocab_mappings(args.mapping)
    print(f"Loaded vocabulary mappings for: {list(vocab_mappings.keys())}")
    
    # Check if external phi CSVs are provided
    external_phi = None
    phi_csv_files = {}
    if args.phi_csv_icd:
        phi_csv_files['icd'] = args.phi_csv_icd
    if args.phi_csv_med:
        phi_csv_files['med'] = args.phi_csv_med
    if args.phi_csv_opcs:
        phi_csv_files['opcs'] = args.phi_csv_opcs
    
    if phi_csv_files:
        print(f"\nLoading external phi distributions from CSV files...")
        external_phi = MixEHR_SAGE.load_phi_from_csv(phi_csv_files, modality_list)
        print(f"External phi loaded for: {[m for m, phi in zip(modality_list, external_phi) if phi is not None]}")
    
    # Determine which mode to use: separate modality files or single file
    modality_files = {}
    if args.icd:
        modality_files['icd'] = args.icd
    if args.med:
        modality_files['med'] = args.med
    if args.opcs:
        modality_files['opcs'] = args.opcs
    
    # Check if we have modality-specific files or a single data file
    patients_bow = None
    if modality_files:
        # Use separate modality files
        print(f"Using separate modality files: {modality_files}")
        if args.explain:
            results_df, patients_bow = infer_from_modality_files(
                model, 
                modality_files, 
                vocab_mappings, 
                modality_list,
                word_column=args.word_column,
                num_iterations=args.iterations,
                return_bow=True,
                external_phi=external_phi
            )
        else:
            results_df = infer_from_modality_files(
                model, 
                modality_files, 
                vocab_mappings, 
                modality_list,
                word_column=args.word_column,
                num_iterations=args.iterations,
                external_phi=external_phi
            )
    elif args.data:
        # Use single data file
        if not os.path.exists(args.data):
            print(f"Error: Patient data file '{args.data}' does not exist.")
            sys.exit(1)
        print(f"Inferring theta for patients in {args.data}...")
        if args.explain:
            results_df, patients_bow = infer_from_file(
                model, 
                args.data, 
                vocab_mappings, 
                modality_list,
                word_column=args.word_column,
                num_iterations=args.iterations,
                return_bow=True,
                external_phi=external_phi
            )
        else:
            results_df = infer_from_file(
                model, 
                args.data, 
                vocab_mappings, 
                modality_list,
                word_column=args.word_column,
                num_iterations=args.iterations,
                external_phi=external_phi
            )
    else:
        print("Error: You must provide either --data for a single file or --icd/--med/--opcs for separate modality files.")
        sys.exit(1)
    
    # Filter to top-k if specified
    if args.top_k is not None:
        topic_cols = [c for c in results_df.columns if c.startswith('topic_')]
        
        def get_top_k(row):
            topic_values = {c: row[c] for c in topic_cols}
            sorted_topics = sorted(topic_values.items(), key=lambda x: x[1], reverse=True)
            return {t[0]: t[1] for t in sorted_topics[:args.top_k]}
        
        top_k_results = []
        for _, row in results_df.iterrows():
            result = {'patient_id': row['patient_id']}
            top_topics = get_top_k(row)
            result.update(top_topics)
            top_k_results.append(result)
        results_df = pd.DataFrame(top_k_results)
    
    # Save results
    output_ext = os.path.splitext(args.output.lower())[1]
    if output_ext == '.json':
        results_df.to_json(args.output, orient='records', indent=2)
    elif output_ext == '.tsv':
        results_df.to_csv(args.output, sep='\t', index=False)
    else:
        results_df.to_csv(args.output, index=False)
    
    print(f"Results saved to {args.output}")
    print(f"Inferred theta for {len(results_df)} patients")
    
    # Print summary
    topic_cols = [c for c in results_df.columns if c.startswith('topic_')]
    if topic_cols:
        print(f"\nTheta statistics (across {len(topic_cols)} topics):")
        for col in topic_cols[:5]:  # Show first 5 topics
            print(f"  {col}: mean={results_df[col].mean():.4f}, std={results_df[col].std():.4f}")
        if len(topic_cols) > 5:
            print(f"  ... and {len(topic_cols) - 5} more topics")
    
    # Generate ChatGPT explanation prompts if requested
    if args.explain and patients_bow is not None:
        print(f"\nGenerating ChatGPT explanation prompts...")
        explanations = []
        
        for _, row in results_df.iterrows():
            patient_id = row['patient_id']
            if patient_id not in patients_bow:
                continue
            
            # Extract theta from row
            theta_values = [row[col] for col in topic_cols]
            theta = np.array(theta_values)
            
            # Generate prompt
            prompt = generate_chatgpt_explanation_prompt(
                patient_id=patient_id,
                patient_bow=patients_bow[patient_id],
                theta=theta,
                model=model,
                vocab_mappings=vocab_mappings,
                modality_list=modality_list,
                top_k_topics=args.explain_top_topics,
                top_n_codes=args.explain_top_codes
            )
            
            explanations.append({
                'patient_id': patient_id,
                'prompt': prompt
            })
        
        # Save explanations to file based on format
        explain_ext = os.path.splitext(args.explain_output.lower())[1]
        
        if explain_ext == '.json':
            # Save as JSON (one object per patient)
            with open(args.explain_output, 'w') as f:
                json.dump(explanations, f, indent=2)
            print(f"Generated {len(explanations)} explanation prompts")
            print(f"Explanations saved to {args.explain_output} (JSON format)")
            
        elif explain_ext == '.csv':
            # Save as CSV (one row per patient)
            explanations_df = pd.DataFrame(explanations)
            explanations_df.to_csv(args.explain_output, index=False)
            print(f"Generated {len(explanations)} explanation prompts")
            print(f"Explanations saved to {args.explain_output} (CSV format)")
            
        else:
            # Save as TXT (one section per patient, easy to separate)
            with open(args.explain_output, 'w') as f:
                for i, expl in enumerate(explanations):
                    if i > 0:
                        f.write("\n\n" + "="*80 + "\n\n")
                    f.write(f"Patient ID: {expl['patient_id']}\n")
                    f.write("="*80 + "\n\n")
                    f.write(expl['prompt'])
            print(f"Generated {len(explanations)} explanation prompts")
            print(f"Explanations saved to {args.explain_output} (TXT format)")
        
        print(f"\nYou can copy these prompts and paste them into ChatGPT for detailed explanations.")


if __name__ == "__main__":
    main()
