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


def compute_phi_full(model, topic_idx):
    """
    Compute the full/combined phi distribution for a topic for display purposes.
    
    Note: The model's infer_theta_fast already uses combined phi internally.
    This function is for extracting phi values for explanation/display.
    
    For the guided modality: φ_full = π × φ^s + (1 - π) × φ^r (for seed words)
    For other modalities: φ_full = φ^r
    
    Args:
        model: trained MixEHR_SAGE model
        topic_idx: topic index
        
    Returns:
        dict mapping modality index to phi_full array for that topic
    """
    phi_full = {}
    
    for m in range(model.modaltiy_num):
        phi_r = model.get_phi(modality=m)  # V x K
        phi_r_np = phi_r.cpu().numpy()[:, topic_idx]
        
        if m == model.guided_modality:
            phi_s = model.get_phi_seed()  # V x K
            phi_s_np = phi_s.cpu().numpy()[:, topic_idx]
            seeds_matrix = model.seeds_topic_matrix.cpu().numpy()[:, topic_idx]
            
            pi = model.pi.cpu().numpy() if torch.is_tensor(model.pi) else model.pi
            pi_k = pi[topic_idx] if hasattr(pi, '__len__') else pi
            
            # φ_full = (1 - π) × φ^r + π × φ^s for seed words
            phi_full_topic = phi_r_np.copy()
            is_seed = seeds_matrix > 0
            phi_full_topic[is_seed] = (1 - pi_k) * phi_r_np[is_seed] + pi_k * phi_s_np[is_seed]
        else:
            phi_full_topic = phi_r_np
        
        phi_full[m] = phi_full_topic
    
    return phi_full


def get_top_codes_for_topic(model, topic_idx, vocab_mappings, modality_list, top_n=10):
    """
    Get top contributing codes for a topic using φ_full.
    
    Args:
        model: trained MixEHR_SAGE model
        topic_idx: topic index
        vocab_mappings: vocabulary mappings
        modality_list: list of modality names
        top_n: number of top codes to return
        
    Returns:
        dict mapping modality name to list of (code, probability) tuples
    """
    phi_full = compute_phi_full(model, topic_idx)
    
    # Create reverse vocabulary mappings
    reverse_vocabs = {}
    for modality, vocab in vocab_mappings.items():
        reverse_vocabs[modality] = {word_id: code for code, word_id in vocab.items()}
    
    top_codes = {}
    for m, modality_name in enumerate(modality_list):
        if modality_name not in vocab_mappings:
            continue
        
        phi_m = phi_full.get(m)
        if phi_m is None:
            continue
        
        top_indices = np.argsort(phi_m)[::-1][:top_n]
        codes_info = []
        
        for idx in top_indices:
            if idx in reverse_vocabs[modality_name]:
                code = reverse_vocabs[modality_name][idx]
                prob = phi_m[idx]
                codes_info.append((code, prob))
        
        if codes_info:
            top_codes[modality_name] = codes_info
    
    return top_codes


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
                               word_column='code', num_iterations=10, return_bow=False, external_phi=None, method='gibbs'):
    """
    Infer theta for patients from separate modality files.
    
    Args:
        model: trained MixEHR_SAGE model
        modality_files: dict of {modality_name: file_path}
        vocab_mappings: vocabulary mappings
        modality_list: list of modality names
        word_column: column name for codes
        num_iterations: inference iterations
        return_bow: if True, also return patients_bow dict
        external_phi: optional list of external phi distributions from CSV files
        method: inference method - 'gibbs' (default) or 'variational'
    
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
            theta = model.infer_theta_fast(bow, num_iterations=num_iterations, method=method)
            
        theta_np = theta.cpu().numpy()
        result = {'patient_id': patient_id}
        for k in range(len(theta_np)):
            result[f'topic_{k}'] = theta_np[k]
        results.append(result)
    
    if return_bow:
        return pd.DataFrame(results), patients_bow
    return pd.DataFrame(results)


def infer_from_file(model, patient_file, vocab_mappings, modality_list, 
                    word_column='code', num_iterations=10, return_bow=False, external_phi=None, method='gibbs'):
    """
    Infer theta for patients from a data file.
    
    Args:
        model: trained MixEHR_SAGE model
        patient_file: path to patient data file (CSV, TSV, JSON, TXT)
        vocab_mappings: vocabulary mappings
        modality_list: list of modality names
        word_column: column name for codes
        num_iterations: inference iterations
        return_bow: if True, also return patients_bow dict
        external_phi: optional list of external phi distributions from CSV files
        method: inference method - 'gibbs' (default) or 'variational'
    
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
            theta = model.infer_theta_fast(bow, num_iterations=num_iterations, method=method)
            
        theta_np = theta.cpu().numpy()
        result = {'patient_id': patient_id}
        for k in range(len(theta_np)):
            result[f'topic_{k}'] = theta_np[k]
        results.append(result)
    
    if return_bow:
        return pd.DataFrame(results), patients_bow
    return pd.DataFrame(results)


def infer_temporal_patient(model, temporal_data, vocab_mappings, modality_list,
                           cutoff_date=None, bucket_type='yearly', num_iterations=10,
                           method='variational', forecast_horizon=0, forecast_smoothing=0.9):
    """
    Temporal inference for new patients using trained model parameters.
    
    This function supports longitudinal patient data and computes per-time theta sequences
    while preserving all trained parameters (phi, pi, etc.) unchanged.
    
    Uses φ_full = (1 - π) × φ^r + π × φ^s for the guided modality during inference.
    
    Args:
        model: trained MixEHR_SAGE model (parameters NOT modified)
        temporal_data: Either:
            - DataFrame with columns: SUBJECT_ID, code, timestamp, modality
            - Dict mapping modality to DataFrame with: SUBJECT_ID, code, timestamp
        vocab_mappings: vocabulary mappings
        modality_list: list of modality names
        cutoff_date: Only include records up to this date (optional, string 'YYYY-MM-DD')
        bucket_type: Bucketing strategy ('yearly', 'monthly', 'quarterly', 'visit')
        num_iterations: inference iterations
        method: inference method ('variational' or 'gibbs')
        forecast_horizon: number of future time steps to forecast (0 = no forecast)
        forecast_smoothing: smoothing factor for theta extrapolation (0-1, higher=more persistence)
        
    Returns:
        dict: {
            'patient_id': patient_id,
            'theta_sequence': list of theta arrays (T x K),
            'time_labels': list of time labels,
            'theta_current': current (latest) theta,
            'top_topics': list of (topic_idx, probability, topic_name) for top topics,
            'top_codes_per_topic': dict mapping topic_idx to top contributing codes,
            'forecast': list of forecasted theta arrays (if forecast_horizon > 0),
            'forecast_labels': list of forecast time labels
        }
    """
    from temporal_corpus import TemporalCorpus
    
    # Create temporal corpus from data
    if isinstance(temporal_data, dict):
        # Separate files per modality
        corpus = TemporalCorpus(bucket_type=bucket_type)
        
        for modality_name, df in temporal_data.items():
            if df is None or len(df) == 0:
                continue
            
            # Parse timestamps
            if bucket_type == 'visit':
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            for _, row in df.iterrows():
                patient_id = str(row['SUBJECT_ID'])
                code = str(row['code'])
                timestamp = row['timestamp']
                
                if pd.isna(timestamp):
                    continue
                    
                # Apply cutoff date if specified
                if cutoff_date and bucket_type != 'visit':
                    cutoff_dt = pd.to_datetime(cutoff_date)
                    if timestamp > cutoff_dt:
                        continue
                
                # Get or create patient
                if patient_id not in corpus.patients:
                    from temporal_corpus import TemporalPatient
                    corpus.patients[patient_id] = TemporalPatient(patient_id)
                patient = corpus.patients[patient_id]
                
                # Compute bucket info
                bucket_info = corpus._compute_bucket_info(timestamp)
                time_index = bucket_info['index']
                
                # Get or create bucket
                bucket = patient.get_bucket(time_index)
                if bucket is None:
                    from temporal_corpus import TemporalBucket
                    bucket = TemporalBucket(
                        patient_id=patient_id,
                        time_index=time_index,
                        start_time=bucket_info['start'],
                        end_time=bucket_info['end'],
                        bucket_type=bucket_type
                    )
                    patient.add_bucket(bucket)
                
                bucket.add_record(code, modality_name, timestamp)
    else:
        # Single DataFrame with modality column
        corpus = TemporalCorpus.from_dataframe(temporal_data, bucket_type=bucket_type)
    
    corpus.set_vocab_mappings(vocab_mappings, modality_list)
    
    # Process each patient
    results = []
    for patient_id, patient in corpus.patients.items():
        # Compute theta for each time bucket (cumulative to prevent data leakage)
        theta_sequence = []
        time_labels = []
        
        for bucket in patient.buckets:
            bow = patient.get_cumulative_bow(bucket.time_index, vocab_mappings, modality_list)
            has_records = any(len(bow_m) > 0 for bow_m in bow)
            
            if has_records:
                theta_t = model.infer_theta_fast(bow, num_iterations=num_iterations, method=method)
            else:
                if len(theta_sequence) > 0:
                    theta_t = theta_sequence[-1].clone()
                else:
                    theta_t = torch.ones(model.K, dtype=torch.double, device=device) / model.K
            
            theta_sequence.append(theta_t)
            time_labels.append(bucket.time_index)
        
        # Get current (latest) theta
        theta_current = theta_sequence[-1] if theta_sequence else torch.ones(model.K, dtype=torch.double, device=device) / model.K
        theta_current_np = theta_current.cpu().numpy()
        
        # Get top topics
        top_k = 10
        top_topic_indices = np.argsort(theta_current_np)[::-1][:top_k]
        
        # Load phenotype names
        phecode_dict = load_phecode_definitions()
        inv_phecode_ids = load_phecode_ids_mapping()
        
        top_topics = []
        for topic_idx in top_topic_indices:
            prob = theta_current_np[topic_idx]
            topic_name = f"Topic {topic_idx}"
            if topic_idx in inv_phecode_ids:
                phecode = inv_phecode_ids[topic_idx]
                if phecode in phecode_dict:
                    topic_name = f"{phecode} ({phecode_dict[phecode]})"
                else:
                    topic_name = f"PheCode {phecode}"
            top_topics.append((topic_idx, prob, topic_name))
        
        # Get top codes for each top topic
        top_codes_per_topic = {}
        for topic_idx, _, _ in top_topics[:5]:  # Top 5 topics
            top_codes_per_topic[topic_idx] = get_top_codes_for_topic(
                model, topic_idx, vocab_mappings, modality_list, top_n=10
            )
        
        # Forecast future theta if requested
        forecast = []
        forecast_labels = []
        if forecast_horizon > 0 and len(theta_sequence) >= 2:
            # Simple smoothing-based forecast: theta_{t+1} = α * theta_t + (1-α) * theta_{t-1}
            # Or Markov-style if model available
            current_theta = theta_current_np.copy()
            
            # Compute trend from recent history
            if len(theta_sequence) >= 2:
                prev_theta = theta_sequence[-2].cpu().numpy()
                trend = current_theta - prev_theta
            else:
                trend = np.zeros_like(current_theta)
            
            last_time = time_labels[-1] if time_labels else 0
            
            for h in range(1, forecast_horizon + 1):
                # Forecast: smooth transition with dampened trend
                dampening = forecast_smoothing ** h
                forecast_theta = current_theta + dampening * trend
                
                # Ensure valid probability distribution
                forecast_theta = np.maximum(forecast_theta, 1e-10)
                forecast_theta = forecast_theta / forecast_theta.sum()
                
                forecast.append(forecast_theta)
                
                # Generate forecast time label
                if bucket_type == 'yearly':
                    forecast_labels.append(last_time + h)
                elif bucket_type == 'monthly':
                    forecast_labels.append(last_time + h)
                elif bucket_type == 'quarterly':
                    forecast_labels.append(last_time + h)
                else:
                    forecast_labels.append(last_time + h)
        
        # Compile results for this patient
        result = {
            'patient_id': patient_id,
            'theta_sequence': [t.cpu().numpy() for t in theta_sequence],
            'time_labels': time_labels,
            'theta_current': theta_current_np,
            'top_topics': top_topics,
            'top_codes_per_topic': top_codes_per_topic,
            'forecast': forecast,
            'forecast_labels': forecast_labels,
            'bucket_type': bucket_type
        }
        results.append(result)
    
    return results[0] if len(results) == 1 else results


def generate_temporal_explanation(patient_result, model=None, include_forecast=True):
    """
    Generate a structured explanation prompt with temporal information.
    
    Args:
        patient_result: dict from infer_temporal_patient
        model: optional MixEHR_SAGE model for additional context
        include_forecast: whether to include forecast section
        
    Returns:
        str: formatted explanation prompt
    """
    patient_id = patient_result['patient_id']
    theta_sequence = patient_result['theta_sequence']
    time_labels = patient_result['time_labels']
    theta_current = patient_result['theta_current']
    top_topics = patient_result['top_topics']
    top_codes = patient_result.get('top_codes_per_topic', {})
    forecast = patient_result.get('forecast', [])
    forecast_labels = patient_result.get('forecast_labels', [])
    bucket_type = patient_result.get('bucket_type', 'yearly')
    
    prompt = f"""# Temporal Disease Profile Analysis: Patient {patient_id}

## Current Health Status (Latest θ)

Based on MixEHR-SAGE analysis using φ_full = (1-π)×φ^r + π×φ^s:

**Top Disease Phenotypes:**
"""
    
    for i, (topic_idx, prob, topic_name) in enumerate(top_topics[:5], 1):
        prompt += f"  {i}. {topic_name}: θ={prob:.4f} ({prob*100:.2f}%)\n"
    
    # Timeline section
    if len(theta_sequence) > 1:
        prompt += f"\n## Disease Trajectory Over Time ({bucket_type} granularity)\n"
        prompt += f"Time points: {len(time_labels)} ({min(time_labels)} to {max(time_labels)})\n\n"
        
        # Show evolution of top 3 topics
        prompt += "**Phenotype Evolution (Top 3):**\n"
        for topic_idx, _, topic_name in top_topics[:3]:
            prompt += f"\n{topic_name}:\n  "
            trajectory = []
            for t_idx, (theta_t, label) in enumerate(zip(theta_sequence, time_labels)):
                theta_val = theta_t[topic_idx] if topic_idx < len(theta_t) else 0
                trajectory.append(f"{label}:{theta_val:.3f}")
            prompt += " → ".join(trajectory[-6:])  # Last 6 time points
            
            # Add trend analysis
            if len(theta_sequence) >= 2:
                first_val = theta_sequence[0][topic_idx] if topic_idx < len(theta_sequence[0]) else 0
                last_val = theta_sequence[-1][topic_idx] if topic_idx < len(theta_sequence[-1]) else 0
                change = last_val - first_val
                if change > 0.05:
                    prompt += " ↑ INCREASING"
                elif change < -0.05:
                    prompt += " ↓ DECREASING"
                else:
                    prompt += " → STABLE"
            prompt += "\n"
    
    # Top codes section
    if top_codes:
        prompt += "\n## Contributing Medical Codes (using φ_full)\n"
        for topic_idx in list(top_codes.keys())[:3]:
            topic_name = next((name for idx, _, name in top_topics if idx == topic_idx), f"Topic {topic_idx}")
            prompt += f"\n**{topic_name}:**\n"
            for modality, codes in top_codes[topic_idx].items():
                prompt += f"  {modality.upper()}: "
                code_strs = [f"{code} (φ={prob:.4f})" for code, prob in codes[:5]]
                prompt += ", ".join(code_strs) + "\n"
    
    # Forecast section
    if include_forecast and forecast:
        prompt += f"\n## Future Risk Forecast (Next {len(forecast)} {bucket_type} periods)\n"
        prompt += "Based on exponential smoothing extrapolation:\n\n"
        
        for topic_idx, _, topic_name in top_topics[:3]:
            current_val = theta_current[topic_idx]
            prompt += f"**{topic_name}:**\n"
            prompt += f"  Current: {current_val:.4f}\n"
            prompt += "  Forecast: "
            forecast_vals = []
            for f_theta, f_label in zip(forecast, forecast_labels):
                f_val = f_theta[topic_idx] if topic_idx < len(f_theta) else 0
                forecast_vals.append(f"{f_label}:{f_val:.4f}")
            prompt += " → ".join(forecast_vals) + "\n"
    
    # Analysis questions
    prompt += """
## Analysis Questions

Based on the temporal disease profile above:

1. **Current Risk Assessment**: What are the patient's primary health conditions based on the current phenotype distribution?

2. **Temporal Patterns**: Are there concerning trends in disease progression over time? Which conditions are worsening?

3. **Future Disease Risk**: Based on the trajectory and current profile, what new conditions might develop in the next 6-12 months?

4. **Comorbidity Analysis**: What comorbidity patterns are evident, and how do they affect prognosis?

5. **Recommended Monitoring**: Which phenotypes require closest monitoring based on the trends?
"""
    
    return prompt


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
                                         modality_list, top_k_topics=5, top_n_codes=10,
                                         temporal_data=None, markov_risk=None,
                                         prediction_horizon=None):
    """
    Generate a ChatGPT prompt to explain inferred phenotype probabilities and predict future risks.
    
    Uses combined phi: φ_combined = π * φ^s + (1-π) * φ^r for the guided modality.
    
    Args:
        patient_id: patient identifier
        patient_bow: patient's bag-of-words data (list of dicts for each modality)
        theta: inferred topic mixture (K-dimensional tensor or array)
        model: trained MixEHR_SAGE model
        vocab_mappings: dict of {modality: {code: word_id}}
        modality_list: list of modality names
        top_k_topics: number of top topics to include (default: 5)
        top_n_codes: number of top codes per topic to include (default: 10)
        temporal_data: optional dict with temporal information:
            - 'theta_sequence': list of theta arrays over time
            - 'time_labels': list of time labels (e.g., ['2015', '2016', '2017'])
            - 'bucket_type': type of bucketing ('yearly', 'monthly', etc.)
        markov_risk: optional dict with Markov model predictions:
            - 'next_state_distribution': predicted next state probabilities
            - 'risk': disease risk predictions at horizon
        prediction_horizon: time horizon for risk prediction (e.g., 3 for 3 years)
    
    Returns:
        str: formatted ChatGPT prompt for disease risk interpretation and prediction
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
    
    # Get combined phi for each topic: φ_combined = π * φ^s + (1-π) * φ^r
    # This properly combines seed and regular word-topic distributions
    topic_top_codes = {}
    pi = model.pi.cpu().numpy() if torch.is_tensor(model.pi) else model.pi
    
    for topic_idx in top_topic_indices:
        topic_codes_by_modality = {}
        
        for m, modality_name in enumerate(modality_list):
            if modality_name not in vocab_mappings:
                continue
            
            # Get regular phi for this modality
            phi_regular = model.get_phi(modality=m)  # V x K matrix
            phi_regular_np = phi_regular.cpu().numpy()
            
            if m == model.guided_modality:
                # For guided modality, combine seed and regular phi
                # φ_combined = π * φ^s + (1-π) * φ^r
                phi_seed = model.get_phi_seed()  # V x K matrix
                phi_seed_np = phi_seed.cpu().numpy()
                seeds_matrix = model.seeds_topic_matrix.cpu().numpy()
                
                # Compute combined phi for this topic
                # Use boolean mask directly for better performance
                pi_k = pi[topic_idx] if hasattr(pi, '__len__') else pi
                is_seed_word = seeds_matrix[:, topic_idx] > 0
                phi_combined = phi_regular_np[:, topic_idx].copy()
                phi_combined[is_seed_word] = (
                    pi_k * phi_seed_np[is_seed_word, topic_idx] + 
                    (1 - pi_k) * phi_regular_np[is_seed_word, topic_idx]
                )
            else:
                # For non-guided modality, just use regular phi
                phi_combined = phi_regular_np[:, topic_idx]
            
            # Get top N codes for this topic
            top_code_indices = np.argsort(phi_combined)[::-1][:top_n_codes]
            top_codes_info = []
            
            for code_idx in top_code_indices:
                if code_idx in reverse_vocabs[modality_name]:
                    code = reverse_vocabs[modality_name][code_idx]
                    prob = phi_combined[code_idx]
                    top_codes_info.append(f"{code} (φ={prob:.4f})")
            
            if top_codes_info:
                topic_codes_by_modality[modality_name] = top_codes_info
        
        topic_top_codes[topic_idx] = topic_codes_by_modality
    
    # Build the prompt
    prompt = f"""I have a patient (ID: {patient_id}) and I used a seed-guided topic modeling approach (MixEHR-SAGE) to infer their phenotype risk profile. Please help me interpret the results and **predict future disease risks**.

**Input 1: Current Inferred Topic Mixtures (θ)**
These represent the patient's probability distribution over latent disease phenotypes.
The combined word-topic distribution φ_combined = π × φ^seed + (1-π) × φ^regular is used for interpretable phenotype mapping:
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
        
        # Add pi value for guided modality
        pi_k = pi[topic_idx] if hasattr(pi, '__len__') else pi
        prompt += f"\n  {i}. {topic_label}: θ={prob:.4f} ({prob*100:.1f}%), π={pi_k:.3f}"
    
    # Add temporal trajectory if available
    if temporal_data is not None:
        prompt += "\n\n**Input 2: Disease Trajectory Over Time**\n"
        bucket_type = temporal_data.get('bucket_type', 'yearly')
        time_labels = temporal_data.get('time_labels', [])
        theta_sequence = temporal_data.get('theta_sequence', [])
        
        prompt += f"The patient's phenotype trajectory ({bucket_type} granularity):\n"
        
        # Show temporal evolution for top topics
        for topic_idx in top_topic_indices[:3]:  # Show top 3 topics over time
            topic_label = f"Topic {topic_idx}"
            if topic_idx in inv_phecode_ids:
                phecode = inv_phecode_ids[topic_idx]
                if phecode in phecode_dict:
                    topic_label = f"{phecode} ({phecode_dict[phecode][:30]}...)" if len(phecode_dict.get(phecode, '')) > 30 else f"{phecode} ({phecode_dict.get(phecode, '')})"
            
            prompt += f"\n  {topic_label}:"
            trajectory_str = []
            for t, (label, theta_t) in enumerate(zip(time_labels, theta_sequence)):
                if torch.is_tensor(theta_t):
                    theta_t = theta_t.cpu().numpy()
                prob_t = theta_t[topic_idx] if len(theta_t) > topic_idx else 0
                trajectory_str.append(f"{label}:{prob_t:.3f}")
            prompt += " → ".join(trajectory_str)
        
        # Identify trends
        prompt += "\n\nTrend Analysis:"
        for topic_idx in top_topic_indices[:3]:
            if len(theta_sequence) >= 2:
                first_theta = theta_sequence[0]
                last_theta = theta_sequence[-1]
                if torch.is_tensor(first_theta):
                    first_theta = first_theta.cpu().numpy()
                if torch.is_tensor(last_theta):
                    last_theta = last_theta.cpu().numpy()
                
                change = last_theta[topic_idx] - first_theta[topic_idx]
                if change > 0.05:
                    trend = "↑ INCREASING"
                elif change < -0.05:
                    trend = "↓ DECREASING"
                else:
                    trend = "→ STABLE"
                
                topic_name = inv_phecode_ids.get(topic_idx, f"Topic {topic_idx}")
                prompt += f"\n  {topic_name}: {trend} (Δ={change:+.3f})"
    
    prompt += "\n\n**Input 3: Patient's Observed Medical Records**\n"
    
    for modality_name, codes in patient_records_by_modality.items():
        unique_codes = list(set(codes))
        if len(unique_codes) > 15:
            prompt += f"\n  {modality_name.upper()}: {', '.join(unique_codes[:15])} ... ({len(unique_codes)} total)"
        else:
            prompt += f"\n  {modality_name.upper()}: {', '.join(unique_codes)}"
    
    prompt += "\n\n**Input 4: Top Codes for Dominant Phenotypes (Combined φ)**\n"
    prompt += "Using φ_combined = π × φ^seed + (1-π) × φ^regular:\n"
    
    for topic_idx in top_topic_indices:
        topic_label = f"Topic {topic_idx}"
        if topic_idx in inv_phecode_ids:
            phecode = inv_phecode_ids[topic_idx]
            if phecode in phecode_dict:
                phenotype_name = phecode_dict[phecode]
                topic_label = f"PheCode {phecode} ({phenotype_name})"
            else:
                topic_label = f"PheCode {phecode}"
        
        prompt += f"\n  {topic_label} (θ={theta_np[topic_idx]:.4f}):"
        if topic_idx in topic_top_codes:
            for modality_name, codes_info in topic_top_codes[topic_idx].items():
                prompt += f"\n    {modality_name.upper()}: {', '.join(codes_info[:7])}"
        else:
            prompt += "\n    (No significant codes)"
    
    # Add Markov risk predictions if available
    if markov_risk is not None:
        horizon = prediction_horizon or 1
        prompt += f"\n\n**Input 5: Predicted Disease Risk ({horizon}-step ahead)**\n"
        prompt += "Based on learned disease progression patterns (Markov model):\n"
        
        next_dist = markov_risk.get('next_state_distribution', None)
        risk_data = markov_risk.get('risk', {})
        
        if next_dist is not None:
            top_future = np.argsort(next_dist)[::-1][:5]
            prompt += "\n  Most likely future phenotypes:"
            for rank, topic_idx in enumerate(top_future, 1):
                topic_label = inv_phecode_ids.get(topic_idx, f"Topic {topic_idx}")
                if topic_label in phecode_dict:
                    topic_label = f"{topic_label} ({phecode_dict[topic_label]})"
                current_prob = theta_np[topic_idx] if topic_idx < len(theta_np) else 0
                future_prob = next_dist[topic_idx]
                change = future_prob - current_prob
                prompt += f"\n    {rank}. {topic_label}: {future_prob:.3f} (Δ={change:+.3f} from current)"
        
        if risk_data:
            prompt += f"\n\n  Risk Summary:"
            prompt += f"\n    - Highest risk phenotype: Topic {risk_data.get('max_risk_topic', 'N/A')}"
            prompt += f"\n    - Maximum risk probability: {risk_data.get('max_risk', 0):.4f}"
            prompt += f"\n    - Prediction uncertainty (entropy): {risk_data.get('entropy', 0):.3f}"
    
    # Questions focused on future disease risk prediction
    prompt += "\n\n**Questions for Analysis:**"
    prompt += "\nBased on the above information, please provide:"
    prompt += "\n"
    prompt += "\n1. **Current Health Status**: What are the patient's dominant disease phenotypes and their clinical interpretation?"
    prompt += "\n"
    prompt += "\n2. **Future Disease Risk Prediction**: Based on the current phenotype profile"
    if temporal_data is not None:
        prompt += " and observed trajectory over time"
    prompt += ", what diseases or complications is this patient most at risk of developing in the next 6-12 months? Consider:"
    prompt += "\n   - Natural disease progression patterns"
    prompt += "\n   - Comorbidity relationships"
    prompt += "\n   - Medication-related risks"
    prompt += "\n"
    prompt += "\n3. **Risk Factors**: What specific risk factors in the patient's profile should clinicians monitor closely?"
    prompt += "\n"
    prompt += "\n4. **Preventive Recommendations**: What preventive measures or interventions could reduce the patient's future disease risk?"
    
    if temporal_data is not None:
        prompt += "\n"
        prompt += "\n5. **Temporal Pattern Insights**: How does the patient's disease trajectory inform the risk prediction? Are there concerning trends that suggest accelerated disease progression?"
    
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
        help='Number of inference iterations (default: 10)'
    )
    parser.add_argument(
        '--method',
        choices=['gibbs', 'variational'],
        default='gibbs',
        help='Inference method: gibbs (default) or variational (default: gibbs)'
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
    parser.add_argument(
        '--temporal-data',
        default=None,
        help='Path to temporal patient data file (CSV with SUBJECT_ID, code, timestamp, modality columns)'
    )
    parser.add_argument(
        '--temporal-icd',
        default=None,
        help='Path to temporal ICD codes file (alternative to --temporal-data)'
    )
    parser.add_argument(
        '--temporal-med',
        default=None,
        help='Path to temporal medication codes file (alternative to --temporal-data)'
    )
    parser.add_argument(
        '--temporal-opcs',
        default=None,
        help='Path to temporal OPCS procedure codes file (alternative to --temporal-data)'
    )
    parser.add_argument(
        '--bucket-type',
        choices=['yearly', 'monthly', 'quarterly', 'visit'],
        default='yearly',
        help='Time bucketing strategy for temporal data (default: yearly)'
    )
    parser.add_argument(
        '--markov-model',
        default=None,
        help='Path to trained Markov transition model (.pkl) for risk prediction'
    )
    parser.add_argument(
        '--prediction-horizon',
        type=int,
        default=1,
        help='Prediction horizon for future disease risk (default: 1 time step)'
    )
    parser.add_argument(
        '--temporal-inference',
        action='store_true',
        help='Enable temporal inference mode for time-series patient data'
    )
    parser.add_argument(
        '--cutoff-date',
        default=None,
        help='Only include records up to this date (YYYY-MM-DD format)'
    )
    parser.add_argument(
        '--forecast-horizon',
        type=int,
        default=0,
        help='Number of future time steps to forecast (default: 0, no forecast)'
    )
    parser.add_argument(
        '--forecast-smoothing',
        type=float,
        default=0.9,
        help='Smoothing factor for theta extrapolation (0-1, higher=more persistence, default: 0.9)'
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
        external_phi, phi_code_mappings = MixEHR_SAGE.load_phi_from_csv(phi_csv_files, modality_list)
        print(f"External phi loaded for: {[m for m, phi in zip(modality_list, external_phi) if phi is not None]}")
        
        # Update vocab mappings with code mappings from phi CSV
        for modality, code_map in phi_code_mappings.items():
            if code_map and modality in vocab_mappings:
                # Create reverse mapping: full description -> vocab_id
                # The vocab_ids in vocab_mappings map original codes to indices
                # We need to update it so both code and full description map to same index
                print(f"  Updating {modality} vocabulary with {len(code_map)} code mappings from phi CSV")
                
                # Build new vocab mapping that includes both formats
                existing_vocab = vocab_mappings[modality]
                for code, full_desc in code_map.items():
                    # If code is in existing vocab, add full_desc mapping to same index
                    if code in existing_vocab:
                        vocab_id = existing_vocab[code]
                        existing_vocab[full_desc] = vocab_id
    
    # Determine which mode to use: separate modality files or single file
    modality_files = {}
    if args.icd:
        modality_files['icd'] = args.icd
    if args.med:
        modality_files['med'] = args.med
    if args.opcs:
        modality_files['opcs'] = args.opcs
    
    # Check for temporal inference mode
    has_temporal_data = args.temporal_data or args.temporal_icd or args.temporal_med or args.temporal_opcs
    
    if args.temporal_inference or has_temporal_data:
        # ============================================
        # TEMPORAL INFERENCE MODE
        # ============================================
        print("\n" + "="*60)
        print("TEMPORAL INFERENCE MODE")
        print("="*60)
        print("Using trained model parameters (phi, pi) unchanged.")
        print("Inference uses φ_full = π × φ^s + (1-π) × φ^r internally.")
        
        # Collect temporal data files
        temporal_modality_data = {}
        
        if args.temporal_data:
            # Single file with modality column
            print(f"\nLoading temporal data from {args.temporal_data}...")
            df = read_data_file(args.temporal_data)
            # This will be handled by infer_temporal_patient
            temporal_df = df
        else:
            # Separate files per modality
            temporal_df = None
            if args.temporal_icd:
                print(f"Loading temporal ICD data from {args.temporal_icd}...")
                temporal_modality_data['icd'] = read_data_file(args.temporal_icd)
            if args.temporal_med:
                print(f"Loading temporal medication data from {args.temporal_med}...")
                temporal_modality_data['med'] = read_data_file(args.temporal_med)
            if args.temporal_opcs:
                print(f"Loading temporal OPCS data from {args.temporal_opcs}...")
                temporal_modality_data['opcs'] = read_data_file(args.temporal_opcs)
        
        # Run temporal inference
        if temporal_df is not None:
            temporal_results = infer_temporal_patient(
                model=model,
                temporal_data=temporal_df,
                vocab_mappings=vocab_mappings,
                modality_list=modality_list,
                cutoff_date=args.cutoff_date,
                bucket_type=args.bucket_type,
                num_iterations=args.iterations,
                method=args.method,
                forecast_horizon=args.forecast_horizon,
                forecast_smoothing=args.forecast_smoothing
            )
        else:
            temporal_results = infer_temporal_patient(
                model=model,
                temporal_data=temporal_modality_data,
                vocab_mappings=vocab_mappings,
                modality_list=modality_list,
                cutoff_date=args.cutoff_date,
                bucket_type=args.bucket_type,
                num_iterations=args.iterations,
                method=args.method,
                forecast_horizon=args.forecast_horizon,
                forecast_smoothing=args.forecast_smoothing
            )
        
        # Handle single patient or multiple patients
        if not isinstance(temporal_results, list):
            temporal_results = [temporal_results]
        
        print(f"\nProcessed {len(temporal_results)} patients with temporal inference")
        
        # Convert temporal results to DataFrame for theta output
        theta_rows = []
        for result in temporal_results:
            patient_id = result['patient_id']
            theta_current = result['theta_current']
            
            row = {'patient_id': patient_id}
            for k, val in enumerate(theta_current):
                row[f'topic_{k}'] = val
            theta_rows.append(row)
        
        results_df = pd.DataFrame(theta_rows)
        
        # Save theta sequences if forecast was requested
        if args.forecast_horizon > 0:
            sequence_rows = []
            for result in temporal_results:
                patient_id = result['patient_id']
                
                # Historical theta
                for t, (theta_t, label) in enumerate(zip(result['theta_sequence'], result['time_labels'])):
                    row = {
                        'patient_id': patient_id,
                        'time_index': label,
                        'time_type': 'historical'
                    }
                    for k, val in enumerate(theta_t):
                        row[f'theta_{k}'] = val
                    sequence_rows.append(row)
                
                # Forecasted theta
                for f_theta, f_label in zip(result['forecast'], result['forecast_labels']):
                    row = {
                        'patient_id': patient_id,
                        'time_index': f_label,
                        'time_type': 'forecast'
                    }
                    for k, val in enumerate(f_theta):
                        row[f'theta_{k}'] = val
                    sequence_rows.append(row)
            
            seq_df = pd.DataFrame(sequence_rows)
            seq_output = args.output.replace('.csv', '_sequences.csv')
            seq_df.to_csv(seq_output, index=False)
            print(f"Theta sequences (with forecast) saved to {seq_output}")
        
        # Generate explanations if requested
        if args.explain:
            print(f"\nGenerating temporal explanation prompts...")
            explanations = []
            
            for result in temporal_results:
                prompt = generate_temporal_explanation(
                    result,
                    model=model,
                    include_forecast=(args.forecast_horizon > 0)
                )
                explanations.append({
                    'patient_id': result['patient_id'],
                    'prompt': prompt
                })
            
            # Save explanations
            explain_ext = os.path.splitext(args.explain_output.lower())[1]
            
            if explain_ext == '.json':
                with open(args.explain_output, 'w') as f:
                    json.dump(explanations, f, indent=2)
            elif explain_ext == '.csv':
                pd.DataFrame(explanations).to_csv(args.explain_output, index=False)
            else:
                with open(args.explain_output, 'w') as f:
                    for i, expl in enumerate(explanations):
                        if i > 0:
                            f.write("\n\n" + "="*80 + "\n\n")
                        f.write(f"Patient ID: {expl['patient_id']}\n")
                        f.write("="*80 + "\n\n")
                        f.write(expl['prompt'])
            
            print(f"Explanations saved to {args.explain_output}")
        
        # Print summary for each patient
        print("\n" + "="*60)
        print("TEMPORAL INFERENCE RESULTS SUMMARY")
        print("="*60)
        
        for result in temporal_results[:5]:  # Show first 5 patients
            print(f"\nPatient: {result['patient_id']}")
            print(f"  Time points: {len(result['time_labels'])} ({result['bucket_type']} buckets)")
            print(f"  Top 5 topics (current θ):")
            for topic_idx, prob, topic_name in result['top_topics'][:5]:
                print(f"    - {topic_name}: {prob:.4f}")
            
            if result['forecast']:
                print(f"  Forecast: {len(result['forecast'])} steps ahead")
        
        if len(temporal_results) > 5:
            print(f"\n  ... and {len(temporal_results) - 5} more patients")
        
        # Save results
        output_ext = os.path.splitext(args.output.lower())[1]
        if output_ext == '.json':
            results_df.to_json(args.output, orient='records', indent=2)
        elif output_ext == '.tsv':
            results_df.to_csv(args.output, sep='\t', index=False)
        else:
            results_df.to_csv(args.output, index=False)
        
        print(f"\nCurrent theta saved to {args.output}")
        print(f"Inferred temporal theta for {len(results_df)} patients")
        
        return  # Exit after temporal inference
    
    # ============================================
    # STANDARD (NON-TEMPORAL) INFERENCE MODE
    # ============================================
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
                external_phi=external_phi,
                method=args.method
            )
        else:
            results_df = infer_from_modality_files(
                model, 
                modality_files, 
                vocab_mappings, 
                modality_list,
                word_column=args.word_column,
                num_iterations=args.iterations,
                external_phi=external_phi,
                method=args.method
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
                external_phi=external_phi,
                method=args.method
            )
        else:
            results_df = infer_from_file(
                model, 
                args.data, 
                vocab_mappings, 
                modality_list,
                word_column=args.word_column,
                num_iterations=args.iterations,
                external_phi=external_phi,
                method=args.method
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
        
        # Load temporal data if provided (single file or separate modality files)
        temporal_corpus = None
        theta_sequences = None
        
        # Check for separate modality temporal files
        temporal_modality_files = {}
        if args.temporal_icd:
            temporal_modality_files['icd'] = args.temporal_icd
        if args.temporal_med:
            temporal_modality_files['med'] = args.temporal_med
        if args.temporal_opcs:
            temporal_modality_files['opcs'] = args.temporal_opcs
        
        if temporal_modality_files or args.temporal_data:
            try:
                from temporal_corpus import TemporalCorpus
                
                if temporal_modality_files:
                    print(f"Loading temporal data from modality files: {temporal_modality_files}")
                    temporal_corpus = TemporalCorpus.from_modality_files(
                        temporal_modality_files,
                        bucket_type=args.bucket_type
                    )
                else:
                    print(f"Loading temporal data from {args.temporal_data}...")
                    temporal_corpus = TemporalCorpus.from_file(
                        args.temporal_data,
                        bucket_type=args.bucket_type
                    )
                
                temporal_corpus.set_vocab_mappings(vocab_mappings, modality_list)
                
                # Compute theta sequences
                theta_sequences = temporal_corpus.compute_theta_sequences(
                    model,
                    num_iterations=args.iterations,
                    use_cumulative=True,
                    method=args.method
                )
                print(f"Computed temporal theta for {len(theta_sequences)} patients")
            except ImportError:
                print("Warning: temporal_corpus module not found. Skipping temporal analysis.")
            except Exception as e:
                print(f"Warning: Could not load temporal data: {e}")
        
        # Load Markov model if provided
        markov_model = None
        if args.markov_model:
            try:
                from temporal_models import MarkovTransitionModel
                print(f"Loading Markov model from {args.markov_model}...")
                markov_model = MarkovTransitionModel.load(args.markov_model)
                print(f"Loaded Markov model with {markov_model.K} topics")
            except ImportError:
                print("Warning: temporal_models module not found. Skipping Markov predictions.")
            except Exception as e:
                print(f"Warning: Could not load Markov model: {e}")
        
        for _, row in results_df.iterrows():
            patient_id = row['patient_id']
            if patient_id not in patients_bow:
                continue
            
            # Extract theta from row
            theta_values = [row[col] for col in topic_cols]
            theta = np.array(theta_values)
            
            # Prepare temporal data for this patient if available
            temporal_data = None
            if temporal_corpus is not None and patient_id in temporal_corpus.patients:
                patient = temporal_corpus.patients[patient_id]
                if patient_id in theta_sequences:
                    theta_seq = theta_sequences[patient_id]
                    time_labels = [str(bucket.time_index) for bucket in patient.buckets]
                    
                    # Convert to list of arrays
                    if torch.is_tensor(theta_seq):
                        theta_seq_list = [theta_seq[t] for t in range(theta_seq.shape[0])]
                    else:
                        theta_seq_list = [theta_seq[t] for t in range(len(theta_seq))]
                    
                    temporal_data = {
                        'theta_sequence': theta_seq_list,
                        'time_labels': time_labels,
                        'bucket_type': args.bucket_type
                    }
            
            # Prepare Markov risk predictions if available
            markov_risk = None
            if markov_model is not None:
                try:
                    next_dist = markov_model.predict_next_state(theta)
                    risk = markov_model.predict_disease_risk(
                        theta, 
                        horizon=args.prediction_horizon
                    )
                    markov_risk = {
                        'next_state_distribution': next_dist,
                        'risk': risk
                    }
                except Exception as e:
                    print(f"Warning: Could not compute Markov predictions for {patient_id}: {e}")
            
            # Generate prompt with temporal and prediction data
            prompt = generate_chatgpt_explanation_prompt(
                patient_id=patient_id,
                patient_bow=patients_bow[patient_id],
                theta=theta,
                model=model,
                vocab_mappings=vocab_mappings,
                modality_list=modality_list,
                top_k_topics=args.explain_top_topics,
                top_n_codes=args.explain_top_codes,
                temporal_data=temporal_data,
                markov_risk=markov_risk,
                prediction_horizon=args.prediction_horizon
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
