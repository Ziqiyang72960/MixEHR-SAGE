#!/usr/bin/env python3
"""
Fast online inference for new patient risk (theta) using trained MixEHR-SAGE model.

Usage:
    python infer_patient.py <model_path> <patient_data> [options]

Examples:
    # Infer theta for a single patient from CSV
    python infer_patient.py ./results/ ./new_patient.csv --output patient_theta.csv

    # Infer theta for multiple patients
    python infer_patient.py ./results/ ./new_patients.csv --output patients_theta.csv

    # Use JSON input format
    python infer_patient.py ./results/ ./new_patient.json --output patient_theta.json
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
        patient_id = row['SUBJECT_ID']
        code = row[word_column]
        
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


def infer_from_file(model, patient_file, vocab_mappings, modality_list, 
                    word_column='code', num_iterations=10):
    """
    Infer theta for patients from a data file.
    
    Args:
        model: trained MixEHR_SAGE model
        patient_file: path to patient data file (CSV, TSV, JSON, TXT)
        vocab_mappings: vocabulary mappings
        modality_list: list of modality names
        word_column: column name for codes
        num_iterations: VI iterations
    
    Returns:
        DataFrame with patient_id and theta values
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
        theta = model.infer_theta(bow, num_iterations=num_iterations)
        theta_np = theta.cpu().numpy()
        result = {'patient_id': patient_id}
        for k in range(len(theta_np)):
            result[f'topic_{k}'] = theta_np[k]
        results.append(result)
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(
        description="Fast online inference for new patient risk using trained MixEHR-SAGE model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Infer theta for patients from CSV
    python infer_patient.py ./results/ ./new_patients.csv -o patient_theta.csv

    # Use more VI iterations for better accuracy
    python infer_patient.py ./results/ ./new_patients.csv -o theta.csv --iterations 20

Input Data Format:
    The input file should have at minimum:
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
        'patient_data',
        help='Path to patient data file (CSV, TSV, JSON, TXT)'
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
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.patient_data):
        print(f"Error: Patient data file '{args.patient_data}' does not exist.")
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
    
    # Infer theta
    print(f"Inferring theta for patients in {args.patient_data}...")
    results_df = infer_from_file(
        model, 
        args.patient_data, 
        vocab_mappings, 
        modality_list,
        word_column=args.word_column,
        num_iterations=args.iterations
    )
    
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


if __name__ == "__main__":
    main()
