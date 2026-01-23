#!/usr/bin/env python3
"""
Temporal Analysis Pipeline for MixEHR-SAGE

This script provides a complete pipeline for temporal EHR analysis:
1. Load temporal patient data
2. Compute per-time topic mixtures (θ_t)
3. Train temporal progression models (Markov/LSTM)
4. Predict future disease risk

Usage:
    # Basic temporal theta computation
    python run_temporal.py compute_theta --data ./data/temporal_data.csv --output ./results/

    # Train Markov progression model
    python run_temporal.py train_markov --theta ./results/theta_sequences.csv --output ./results/

    # Train LSTM temporal model
    python run_temporal.py train_lstm --data ./data/temporal_data.csv --output ./results/

    # Predict disease risk
    python run_temporal.py predict_risk --model ./results/markov_model.pkl --patient ./patient.csv

Examples:
    # Full pipeline with sample data
    python run_temporal.py demo

    # Compute theta sequences from temporal data
    python run_temporal.py compute_theta \\
        --data ./data/example_temporal_data.csv \\
        --model-path ./results/ \\
        --output ./results/temporal/ \\
        --bucket-type yearly

    # Train Markov model on computed theta sequences
    python run_temporal.py train_markov \\
        --theta ./results/temporal/theta_sequences.csv \\
        --output ./results/temporal/markov_model.pkl
"""

import argparse
import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
import torch

from corpus import Corpus
from MixEHR_SAGE import MixEHR_SAGE
from temporal_corpus import TemporalCorpus, generate_sample_temporal_data
from temporal_models import (
    MarkovTransitionModel, TemporalLSTMModel, TemporalMixEHR,
    TemporalMixEHRTrainer, analyze_disease_progression
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_vocab_mappings(mapping_path: str = './mapping/'):
    """Load vocabulary mappings for each modality."""
    vocab_mappings = {}
    for f in os.listdir(mapping_path):
        if f.endswith('_vocab_ids.pkl'):
            modality = f.replace('_vocab_ids.pkl', '')
            with open(os.path.join(mapping_path, f), 'rb') as handle:
                vocab_mappings[modality] = pickle.load(handle)
    return vocab_mappings


def load_trained_model(model_path: str, corpus_path: str = './store/',
                      seed_matrix_path: str = './phecode_mapping/seed_topic_matrix.pt',
                      guide_prior_path: str = './guide_prior/'):
    """Load a trained MixEHR-SAGE model."""
    # Load corpus for model parameters
    corpus = Corpus.read_corpus_from_directory(corpus_path)
    
    # Load seed topic matrix
    # Note: Using weights_only=True for security when loading untrusted files
    try:
        seeds_topic_matrix = torch.load(seed_matrix_path, map_location=device, weights_only=True)
    except Exception:
        # Fall back to weights_only=False for backward compatibility with older saved models
        seeds_topic_matrix = torch.load(seed_matrix_path, map_location=device, weights_only=False)
    
    # Load trained model
    model = MixEHR_SAGE.load_trained_model(
        model_path, corpus, seeds_topic_matrix, corpus.modalities,
        guided_modality=0, guide_prior_path=guide_prior_path
    )
    model = model.to(device)
    
    return model, corpus


def compute_theta_sequences(args):
    """Compute temporal theta sequences from longitudinal data."""
    logger.info("Computing temporal theta sequences...")
    
    # Load model
    model, corpus = load_trained_model(
        args.model_path, args.corpus_path,
        args.seed_matrix, args.guide_prior_path
    )
    
    # Load vocabulary mappings
    vocab_mappings = load_vocab_mappings(args.mapping_path)
    modality_list = corpus.modalities
    
    # Check if using separate modality files or single file
    modality_files = {}
    if args.icd:
        modality_files['icd'] = args.icd
    if args.med:
        modality_files['med'] = args.med
    if args.opcs:
        modality_files['opcs'] = args.opcs
    
    if modality_files:
        # Load from separate modality files
        logger.info(f"Loading temporal data from modality files: {modality_files}")
        temporal_corpus = TemporalCorpus.from_modality_files(
            modality_files,
            bucket_type=args.bucket_type,
            subject_col=args.subject_col,
            code_col=args.code_col,
            time_col=args.time_col
        )
    elif args.data:
        # Load from single file with modality column
        logger.info(f"Loading temporal data from {args.data}")
        temporal_corpus = TemporalCorpus.from_file(
            args.data,
            bucket_type=args.bucket_type,
            subject_col=args.subject_col,
            code_col=args.code_col,
            time_col=args.time_col,
            modality_col=args.modality_col
        )
    else:
        raise ValueError("Must provide either --data or modality-specific files (--icd, --med, --opcs)")
    
    temporal_corpus.set_vocab_mappings(vocab_mappings, modality_list)
    
    logger.info(f"Created temporal corpus: {temporal_corpus}")
    
    # Limit patients if --max-patients specified (for testing)
    if args.max_patients is not None and args.max_patients > 0:
        patient_ids = list(temporal_corpus.patients.keys())[:args.max_patients]
        limited_patients = {pid: temporal_corpus.patients[pid] for pid in patient_ids}
        temporal_corpus.patients = limited_patients
        logger.info(f"Limited to {len(limited_patients)} patients (--max-patients={args.max_patients})")
    
    # Create output directory first (for checkpoints)
    os.makedirs(args.output, exist_ok=True)
    theta_csv_path = os.path.join(args.output, 'theta_sequences.csv')
    
    # Compute theta sequences with progress tracking and checkpoints
    theta_sequences = temporal_corpus.compute_theta_sequences(
        model,
        num_iterations=args.num_iterations,
        use_cumulative=args.use_cumulative,
        method=args.inference_method,
        save_interval=1000,
        output_path=theta_csv_path
    )
    
    # Export final theta sequences
    temporal_corpus.export_theta_sequences(theta_csv_path, format='csv')
    
    # Also save as pickle for convenience
    theta_pkl_path = os.path.join(args.output, 'theta_sequences.pkl')
    with open(theta_pkl_path, 'wb') as f:
        theta_data = {pid: seq.cpu().numpy() for pid, seq in theta_sequences.items()}
        pickle.dump(theta_data, f)
    
    logger.info(f"Exported theta sequences to {theta_csv_path}")
    logger.info(f"Computed theta sequences for {len(theta_sequences)} patients")
    
    return theta_sequences


def train_markov_model(args):
    """Train Markov transition model from theta sequences."""
    logger.info("Training Markov transition model...")
    
    # Load theta sequences
    if args.theta.endswith('.csv'):
        # Load from CSV
        df = pd.read_csv(args.theta)
        
        # Reconstruct theta sequences from CSV
        theta_sequences = {}
        theta_cols = [c for c in df.columns if c.startswith('theta_')]
        K = len(theta_cols)
        
        for patient_id in df['patient_id'].unique():
            patient_df = df[df['patient_id'] == patient_id].sort_values('time_index')
            T = len(patient_df)
            theta_seq = np.zeros((T, K))
            for t, (_, row) in enumerate(patient_df.iterrows()):
                for k in range(K):
                    theta_seq[t, k] = row[f'theta_{k}']
            theta_sequences[patient_id] = torch.tensor(theta_seq, dtype=torch.double)
    else:
        # Load from pickle
        with open(args.theta, 'rb') as f:
            theta_data = pickle.load(f)
        theta_sequences = {pid: torch.tensor(seq, dtype=torch.double) 
                         for pid, seq in theta_data.items()}
    
    # Determine K from data
    K = next(iter(theta_sequences.values())).shape[1]
    
    # Create and fit Markov model
    markov = MarkovTransitionModel(
        num_topics=K,
        discretization=args.discretization
    )
    markov.fit(theta_sequences, smoothing=args.smoothing)
    
    # Save model
    markov.save(args.output)
    
    # Print summary
    print("\n" + "=" * 50)
    print("MARKOV TRANSITION MODEL SUMMARY")
    print("=" * 50)
    print(f"Number of topics/states: {K}")
    print(f"Discretization method: {args.discretization}")
    print(f"Number of patient sequences: {len(theta_sequences)}")
    
    stationary = markov.get_stationary_distribution()
    top_stationary = np.argsort(stationary)[::-1][:5]
    print(f"\nTop 5 stationary states: {top_stationary}")
    print(f"Stationary probabilities: {stationary[top_stationary]}")
    
    logger.info(f"Saved Markov model to {args.output}")
    
    return markov


def train_lstm_model(args):
    """Train LSTM temporal model."""
    logger.info("Training LSTM temporal model...")
    
    # Load base model
    model, corpus = load_trained_model(
        args.model_path, args.corpus_path,
        args.seed_matrix, args.guide_prior_path
    )
    
    # Load vocabulary mappings
    vocab_mappings = load_vocab_mappings(args.mapping_path)
    modality_list = corpus.modalities
    
    # Calculate total vocabulary size
    V = sum(len(v) for v in vocab_mappings.values())
    K = model.K
    
    # Load temporal data - support both single file and separate modality files
    modality_files = {}
    if hasattr(args, 'icd') and args.icd:
        modality_files['icd'] = args.icd
    if hasattr(args, 'med') and args.med:
        modality_files['med'] = args.med
    if hasattr(args, 'opcs') and args.opcs:
        modality_files['opcs'] = args.opcs
    
    if modality_files:
        logger.info(f"Loading temporal data from modality files: {modality_files}")
        temporal_corpus = TemporalCorpus.from_modality_files(
            modality_files,
            bucket_type=args.bucket_type,
            subject_col=args.subject_col,
            code_col=args.code_col,
            time_col=args.time_col
        )
    elif args.data:
        temporal_corpus = TemporalCorpus.from_file(
            args.data,
            bucket_type=args.bucket_type,
            subject_col=args.subject_col,
            code_col=args.code_col,
            time_col=args.time_col,
            modality_col=args.modality_col
        )
    else:
        raise ValueError("Must provide either --data or modality-specific files (--icd, --med, --opcs)")
    
    temporal_corpus.set_vocab_mappings(vocab_mappings, modality_list)
    
    logger.info(f"Created temporal corpus: {temporal_corpus}")
    
    # Limit patients if --max-patients specified
    if args.max_patients is not None and args.max_patients > 0:
        patient_ids = list(temporal_corpus.patients.keys())[:args.max_patients]
        limited_patients = {pid: temporal_corpus.patients[pid] for pid in patient_ids}
        temporal_corpus.patients = limited_patients
        logger.info(f"Limited to {len(limited_patients)} patients (--max-patients={args.max_patients})")
    
    # Create LSTM model
    lstm_model = TemporalLSTMModel(
        vocab_size=V,
        num_topics=K,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout
    ).to(device)
    
    # Create temporal MixEHR
    temporal_mixehr = TemporalMixEHR(model, lstm_model)
    
    # Train
    loss_history = temporal_mixehr.train_temporal(
        temporal_corpus, vocab_mappings, modality_list,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    temporal_mixehr.save(args.output)
    
    # Compute and export theta sequences using trained model
    theta_sequences = temporal_mixehr.infer_temporal_theta(
        temporal_corpus, vocab_mappings, modality_list,
        num_iterations=args.num_iterations
    )
    
    # Export
    theta_output = args.output.replace('.pt', '_theta_sequences.csv')
    temporal_corpus.export_theta_sequences(theta_output, format='csv')
    
    print("\n" + "=" * 50)
    print("LSTM TEMPORAL MODEL TRAINING COMPLETE")
    print("=" * 50)
    print(f"Final loss: {loss_history[-1]:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"Theta sequences saved to: {theta_output}")
    
    return temporal_mixehr, theta_sequences


def train_from_scratch(args):
    """
    Train temporal model from scratch without pre-trained phi.
    
    This trains exp_m (document-topic) and exp_n (word-topic) simultaneously
    along with the LSTM temporal component, similar to the original MixEHR_SAGE
    training procedure (SCVB0).
    """
    logger.info("Training temporal model from scratch...")
    
    # Load seed topic matrix
    try:
        seeds_topic_matrix = torch.load(args.seed_matrix, map_location=device, weights_only=True)
    except Exception:
        seeds_topic_matrix = torch.load(args.seed_matrix, map_location=device, weights_only=False)
    
    # Determine which modalities are being used based on provided files
    modality_files = {}
    if hasattr(args, 'icd') and args.icd:
        modality_files['icd'] = args.icd
    if hasattr(args, 'med') and args.med:
        modality_files['med'] = args.med
    if hasattr(args, 'opcs') and args.opcs:
        modality_files['opcs'] = args.opcs
    
    if not modality_files and not args.data:
        raise ValueError("Must provide either --data or modality-specific files (--icd, --med, --opcs)")
    
    # Load vocabulary mappings
    vocab_mappings = load_vocab_mappings(args.mapping_path)
    
    # Only use modalities that have data files provided
    if modality_files:
        # Filter to only modalities with provided data files
        modality_list = [m for m in modality_files.keys() if m in vocab_mappings]
        if not modality_list:
            raise ValueError(f"No matching modalities found. Provided: {list(modality_files.keys())}, "
                           f"Available: {list(vocab_mappings.keys())}")
        # Filter vocab_mappings to only include used modalities
        vocab_mappings = {m: vocab_mappings[m] for m in modality_list}
    else:
        modality_list = list(vocab_mappings.keys())
    
    vocab_sizes = [len(vocab_mappings[m]) for m in modality_list]
    
    logger.info(f"Modalities: {modality_list}")
    logger.info(f"Vocabulary sizes: {vocab_sizes}")
    
    # Load temporal data
    if modality_files:
        logger.info(f"Loading temporal data from modality files: {modality_files}")
        temporal_corpus = TemporalCorpus.from_modality_files(
            modality_files,
            bucket_type=args.bucket_type,
            subject_col=args.subject_col,
            code_col=args.code_col,
            time_col=args.time_col
        )
    elif args.data:
        temporal_corpus = TemporalCorpus.from_file(
            args.data,
            bucket_type=args.bucket_type,
            subject_col=args.subject_col,
            code_col=args.code_col,
            time_col=args.time_col,
            modality_col=args.modality_col
        )
    
    temporal_corpus.set_vocab_mappings(vocab_mappings, modality_list)
    logger.info(f"Created temporal corpus: {temporal_corpus}")
    
    # Limit patients if --max-patients specified
    if args.max_patients is not None and args.max_patients > 0:
        patient_ids = list(temporal_corpus.patients.keys())[:args.max_patients]
        limited_patients = {pid: temporal_corpus.patients[pid] for pid in patient_ids}
        temporal_corpus.patients = limited_patients
        logger.info(f"Limited to {len(limited_patients)} patients (--max-patients={args.max_patients})")
    
    # Create trainer model
    K = seeds_topic_matrix.shape[1]  # Number of topics from seed matrix
    logger.info(f"Number of topics (K): {K}")
    
    # Find the guided modality index (ICD should be guided)
    guided_modality_idx = 0
    for i, m in enumerate(modality_list):
        if m == 'icd':
            guided_modality_idx = i
            break
    logger.info(f"Guided modality: {modality_list[guided_modality_idx]} (index {guided_modality_idx})")
    
    trainer = TemporalMixEHRTrainer(
        vocab_sizes=vocab_sizes,
        num_topics=K,
        seeds_topic_matrix=seeds_topic_matrix,
        modalities=modality_list,
        guided_modality=guided_modality_idx,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_layers,
        dropout=args.dropout
    )
    
    # Train
    elbo_history = trainer.train_temporal(
        temporal_corpus, vocab_mappings, modality_list,
        num_epochs=args.num_epochs,
        stochastic=True
    )
    
    # Save model
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    trainer.save(args.output)
    
    # Compute and export theta sequences
    theta_sequences = trainer.infer_temporal_theta(
        temporal_corpus, vocab_mappings, modality_list,
        num_iterations=args.num_iterations
    )
    
    # Export theta sequences
    theta_output = args.output.replace('.pt', '_theta_sequences.csv')
    rows = []
    for patient_id, theta_seq in theta_sequences.items():
        theta_np = theta_seq.cpu().numpy()
        patient = temporal_corpus.patients[patient_id]
        for t, bucket in enumerate(patient.buckets):
            row = {
                'patient_id': patient_id,
                'time_index': bucket.time_index,
                'bucket_type': bucket.bucket_type,
            }
            for k in range(theta_np.shape[1]):
                row[f'theta_{k}'] = theta_np[t, k]
            rows.append(row)
    pd.DataFrame(rows).to_csv(theta_output, index=False)
    
    # Export learned phi
    phi_output_dir = os.path.dirname(args.output) or '.'
    for m, modality in enumerate(modality_list):
        phi = trainer.get_combined_phi(m) if m == 0 else trainer.get_phi(m)
        phi_path = os.path.join(phi_output_dir, f'learned_phi_{modality}.pt')
        torch.save(phi, phi_path)
        logger.info(f"Saved phi for {modality} to {phi_path}")
    
    print("\n" + "=" * 50)
    print("TEMPORAL MODEL TRAINING FROM SCRATCH COMPLETE")
    print("=" * 50)
    print(f"Final ELBO: {elbo_history[-1]:.4f}")
    print(f"Model saved to: {args.output}")
    print(f"Theta sequences saved to: {theta_output}")
    print(f"Learned phi saved to: {phi_output_dir}/learned_phi_*.pt")
    
    return trainer, theta_sequences


def predict_disease_risk(args):
    """Predict disease risk using trained model."""
    logger.info("Predicting disease risk...")
    
    # Load model
    if args.model.endswith('.pkl'):
        # Markov model
        markov = MarkovTransitionModel.load(args.model)
        model_type = 'markov'
    else:
        raise ValueError("Currently only Markov models (.pkl) are supported for prediction")
    
    # Load patient data
    if args.patient:
        # Single patient from file
        df = pd.read_csv(args.patient)
        theta_cols = [c for c in df.columns if c.startswith('theta_')]
        current_theta = df[theta_cols].values[-1]  # Use last time point
    elif args.theta:
        # Direct theta values
        current_theta = np.array([float(x) for x in args.theta.split(',')])
    else:
        raise ValueError("Must provide --patient or --theta")
    
    # Predict risk
    risk = markov.predict_disease_risk(current_theta, horizon=args.horizon)
    next_state = markov.predict_next_state(current_theta)
    
    # Print results
    print("\n" + "=" * 50)
    print("DISEASE RISK PREDICTION")
    print("=" * 50)
    print(f"Prediction horizon: {args.horizon} time steps")
    print(f"\nCurrent top 3 topics: {np.argsort(current_theta)[::-1][:3]}")
    print(f"Current top 3 probabilities: {np.sort(current_theta)[::-1][:3]}")
    print(f"\nPredicted next state distribution:")
    print(f"  Top 3 likely states: {np.argsort(next_state)[::-1][:3]}")
    print(f"  Top 3 probabilities: {np.sort(next_state)[::-1][:3]}")
    print(f"\n{args.horizon}-step risk:")
    print(f"  Max risk topic: {risk['max_risk_topic']}")
    print(f"  Max risk probability: {risk['max_risk']:.4f}")
    print(f"  Entropy: {risk['entropy']:.4f}")
    
    # Save to file if specified
    if args.output:
        results = {
            'horizon': args.horizon,
            'current_theta': current_theta.tolist(),
            'next_state_distribution': next_state.tolist(),
            'risk': risk
        }
        with open(args.output, 'w') as f:
            import json
            json.dump(results, f, indent=2)
        logger.info(f"Saved predictions to {args.output}")
    
    return risk


def run_demo(args):
    """Run a complete demo of the temporal analysis pipeline."""
    logger.info("Running temporal analysis demo...")
    
    # Create output directory
    output_dir = args.output or './temporal_demo_results/'
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Generate sample temporal data
    print("\n" + "=" * 60)
    print("STEP 1: Generating Sample Temporal Data")
    print("=" * 60)
    
    sample_data_path = os.path.join(output_dir, 'sample_temporal_data.csv')
    df = generate_sample_temporal_data(
        sample_data_path,
        num_patients=args.num_patients,
        time_range=(2015, 2020)
    )
    print(f"Generated {len(df)} records for {args.num_patients} patients")
    print(f"Sample data:\n{df.head(10)}")
    
    # Step 2: Create temporal corpus
    print("\n" + "=" * 60)
    print("STEP 2: Creating Temporal Corpus")
    print("=" * 60)
    
    temporal_corpus = TemporalCorpus.from_file(
        sample_data_path,
        bucket_type='yearly'
    )
    print(f"Created corpus: {temporal_corpus}")
    
    # Show sample patient
    sample_patient = list(temporal_corpus.patients.values())[0]
    print(f"\nSample patient: {sample_patient}")
    for bucket in sample_patient.buckets:
        print(f"  {bucket}")
    
    # Step 3: Generate synthetic theta sequences (since we may not have trained model)
    print("\n" + "=" * 60)
    print("STEP 3: Generating Synthetic Theta Sequences")
    print("=" * 60)
    
    np.random.seed(42)
    K = 10  # Number of topics
    theta_sequences = {}
    
    for patient_id, patient in temporal_corpus.patients.items():
        T = patient.num_time_steps
        # Generate somewhat smooth theta sequences
        theta_seq = np.random.dirichlet(np.ones(K) * 2, size=T)
        # Add temporal smoothing
        for t in range(1, T):
            theta_seq[t] = 0.7 * theta_seq[t] + 0.3 * theta_seq[t-1]
            theta_seq[t] /= theta_seq[t].sum()
        theta_sequences[patient_id] = torch.tensor(theta_seq, dtype=torch.double)
    
    print(f"Generated theta sequences for {len(theta_sequences)} patients")
    
    # Export theta sequences
    theta_csv_path = os.path.join(output_dir, 'theta_sequences.csv')
    rows = []
    for patient_id, theta_seq in theta_sequences.items():
        theta_np = theta_seq.cpu().numpy()
        patient = temporal_corpus.patients[patient_id]
        for t, bucket in enumerate(patient.buckets):
            row = {
                'patient_id': patient_id,
                'time_index': bucket.time_index,
                'bucket_type': bucket.bucket_type,
            }
            for k in range(theta_np.shape[1]):
                row[f'theta_{k}'] = theta_np[t, k]
            rows.append(row)
    pd.DataFrame(rows).to_csv(theta_csv_path, index=False)
    print(f"Exported to {theta_csv_path}")
    
    # Step 4: Train Markov model
    print("\n" + "=" * 60)
    print("STEP 4: Training Markov Transition Model")
    print("=" * 60)
    
    markov = MarkovTransitionModel(num_topics=K, discretization='soft')
    markov.fit(theta_sequences, smoothing=1.0)
    
    markov_path = os.path.join(output_dir, 'markov_model.pkl')
    markov.save(markov_path)
    
    print(f"Transition matrix shape: {markov.transition_matrix.shape}")
    stationary = markov.get_stationary_distribution()
    print(f"Stationary distribution (top 3): {np.sort(stationary)[::-1][:3]}")
    
    # Step 5: Disease risk prediction
    print("\n" + "=" * 60)
    print("STEP 5: Predicting Disease Risk")
    print("=" * 60)
    
    # Get a sample patient's current state
    sample_pid = list(theta_sequences.keys())[0]
    current_theta = theta_sequences[sample_pid][-1].numpy()
    
    print(f"Patient: {sample_pid}")
    print(f"Current top 3 topics: {np.argsort(current_theta)[::-1][:3]}")
    
    # Predict next state
    next_dist = markov.predict_next_state(current_theta)
    print(f"Predicted next state (top 3): {np.argsort(next_dist)[::-1][:3]}")
    
    # Predict 3-step risk
    risk = markov.predict_disease_risk(current_theta, horizon=3)
    print(f"3-step risk: max_risk={risk['max_risk']:.4f} for topic {risk['max_risk_topic']}")
    
    # Step 6: Analyze progression
    print("\n" + "=" * 60)
    print("STEP 6: Analyzing Disease Progression")
    print("=" * 60)
    
    progression_df = analyze_disease_progression(
        theta_sequences,
        target_topics=[0, 1, 2]
    )
    
    progression_path = os.path.join(output_dir, 'progression_analysis.csv')
    progression_df.to_csv(progression_path, index=False)
    
    print(f"Progression analysis (sample):\n{progression_df.head(10)}")
    print(f"\nSaved to {progression_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print("Generated files:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
    
    return {
        'temporal_corpus': temporal_corpus,
        'theta_sequences': theta_sequences,
        'markov_model': markov,
        'progression_df': progression_df
    }


def main():
    parser = argparse.ArgumentParser(
        description="Temporal Analysis Pipeline for MixEHR-SAGE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--mapping-path', default='./mapping/',
                              help='Path to vocabulary mapping files')
    common_parser.add_argument('--subject-col', default='SUBJECT_ID',
                              help='Column name for patient ID')
    common_parser.add_argument('--code-col', default='code',
                              help='Column name for medical codes')
    common_parser.add_argument('--time-col', default='timestamp',
                              help='Column name for timestamp')
    common_parser.add_argument('--modality-col', default='modality',
                              help='Column name for modality type')
    common_parser.add_argument('--max-patients', type=int, default=None,
                              help='Maximum number of patients to process (for testing, default: all)')
    
    # compute_theta command
    parser_theta = subparsers.add_parser('compute_theta', parents=[common_parser],
                                         help='Compute temporal theta sequences')
    parser_theta.add_argument('--data', default=None, 
                             help='Path to temporal data file (single file with all modalities)')
    parser_theta.add_argument('--icd', default=None,
                             help='Path to temporal ICD codes file (alternative to --data)')
    parser_theta.add_argument('--med', default=None,
                             help='Path to temporal medication codes file (alternative to --data)')
    parser_theta.add_argument('--opcs', default=None,
                             help='Path to temporal OPCS procedure codes file (alternative to --data)')
    parser_theta.add_argument('--model-path', default='./results/',
                             help='Path to trained model directory')
    parser_theta.add_argument('--corpus-path', default='./store/',
                             help='Path to corpus directory')
    parser_theta.add_argument('--seed-matrix', default='./phecode_mapping/seed_topic_matrix.pt',
                             help='Path to seed topic matrix')
    parser_theta.add_argument('--guide-prior-path', default='./guide_prior/',
                             help='Path to guide prior directory')
    parser_theta.add_argument('--output', default='./results/temporal/',
                             help='Output directory')
    parser_theta.add_argument('--bucket-type', default='yearly',
                             choices=['yearly', 'monthly', 'quarterly', 'visit'],
                             help='Time bucketing strategy')
    parser_theta.add_argument('--num-iterations', type=int, default=10,
                             help='Number of inference iterations')
    parser_theta.add_argument('--use-cumulative', action='store_true', default=True,
                             help='Use cumulative records (prevents data leakage)')
    parser_theta.add_argument('--inference-method', default='variational',
                             choices=['variational', 'gibbs'],
                             help='Inference method')
    
    # train_markov command
    parser_markov = subparsers.add_parser('train_markov',
                                          help='Train Markov transition model')
    parser_markov.add_argument('--theta', required=True,
                              help='Path to theta sequences (CSV or pickle)')
    parser_markov.add_argument('--output', default='./results/markov_model.pkl',
                              help='Output path for model')
    parser_markov.add_argument('--discretization', default='soft',
                              choices=['dominant', 'soft', 'threshold'],
                              help='State discretization method')
    parser_markov.add_argument('--smoothing', type=float, default=1.0,
                              help='Laplace smoothing parameter')
    
    # train_lstm command
    parser_lstm = subparsers.add_parser('train_lstm', parents=[common_parser],
                                        help='Train LSTM temporal model')
    parser_lstm.add_argument('--data', default=None,
                            help='Path to temporal data file (single file with all modalities)')
    parser_lstm.add_argument('--icd', default=None,
                            help='Path to temporal ICD codes file (alternative to --data)')
    parser_lstm.add_argument('--med', default=None,
                            help='Path to temporal medication codes file (alternative to --data)')
    parser_lstm.add_argument('--opcs', default=None,
                            help='Path to temporal OPCS procedure codes file (alternative to --data)')
    parser_lstm.add_argument('--model-path', default='./results/',
                            help='Path to trained MixEHR model')
    parser_lstm.add_argument('--corpus-path', default='./store/',
                            help='Path to corpus directory')
    parser_lstm.add_argument('--seed-matrix', default='./phecode_mapping/seed_topic_matrix.pt',
                            help='Path to seed topic matrix')
    parser_lstm.add_argument('--guide-prior-path', default='./guide_prior/',
                            help='Path to guide prior directory')
    parser_lstm.add_argument('--output', default='./results/temporal_lstm.pt',
                            help='Output path for model')
    parser_lstm.add_argument('--bucket-type', default='yearly',
                            choices=['yearly', 'monthly', 'quarterly', 'visit'],
                            help='Time bucketing strategy')
    parser_lstm.add_argument('--hidden-size', type=int, default=200,
                            help='LSTM hidden size')
    parser_lstm.add_argument('--num-layers', type=int, default=3,
                            help='Number of LSTM layers')
    parser_lstm.add_argument('--dropout', type=float, default=0.0,
                            help='Dropout rate')
    parser_lstm.add_argument('--num-epochs', type=int, default=10,
                            help='Number of training epochs')
    parser_lstm.add_argument('--batch-size', type=int, default=32,
                            help='Batch size')
    parser_lstm.add_argument('--num-iterations', type=int, default=10,
                            help='Number of inference iterations')
    
    # train_from_scratch command - trains both phi and theta together
    parser_scratch = subparsers.add_parser('train_from_scratch', parents=[common_parser],
                                           help='Train temporal model from scratch (no pre-trained phi)')
    parser_scratch.add_argument('--data', default=None,
                               help='Path to temporal data file (single file with all modalities)')
    parser_scratch.add_argument('--icd', default=None,
                               help='Path to temporal ICD codes file (alternative to --data)')
    parser_scratch.add_argument('--med', default=None,
                               help='Path to temporal medication codes file (alternative to --data)')
    parser_scratch.add_argument('--opcs', default=None,
                               help='Path to temporal OPCS procedure codes file (alternative to --data)')
    parser_scratch.add_argument('--seed-matrix', default='./phecode_mapping/seed_topic_matrix.pt',
                               help='Path to seed topic matrix')
    parser_scratch.add_argument('--output', default='./results/temporal_scratch.pt',
                               help='Output path for model')
    parser_scratch.add_argument('--bucket-type', default='yearly',
                               choices=['yearly', 'monthly', 'quarterly', 'visit'],
                               help='Time bucketing strategy')
    parser_scratch.add_argument('--hidden-size', type=int, default=200,
                               help='LSTM hidden size')
    parser_scratch.add_argument('--num-layers', type=int, default=3,
                               help='Number of LSTM layers')
    parser_scratch.add_argument('--dropout', type=float, default=0.0,
                               help='Dropout rate')
    parser_scratch.add_argument('--num-epochs', type=int, default=20,
                               help='Number of training epochs')
    parser_scratch.add_argument('--num-iterations', type=int, default=10,
                               help='Number of inference iterations')
    
    # predict_risk command
    parser_risk = subparsers.add_parser('predict_risk',
                                        help='Predict disease risk')
    parser_risk.add_argument('--model', required=True,
                            help='Path to trained model')
    parser_risk.add_argument('--patient', help='Path to patient theta CSV')
    parser_risk.add_argument('--theta', help='Comma-separated theta values')
    parser_risk.add_argument('--horizon', type=int, default=1,
                            help='Prediction horizon (time steps)')
    parser_risk.add_argument('--output', help='Output path for predictions')
    
    # demo command
    parser_demo = subparsers.add_parser('demo',
                                        help='Run complete demo')
    parser_demo.add_argument('--output', default='./temporal_demo_results/',
                            help='Output directory')
    parser_demo.add_argument('--num-patients', type=int, default=30,
                            help='Number of patients to generate')
    
    args = parser.parse_args()
    
    if args.command == 'compute_theta':
        compute_theta_sequences(args)
    elif args.command == 'train_markov':
        train_markov_model(args)
    elif args.command == 'train_lstm':
        train_lstm_model(args)
    elif args.command == 'train_from_scratch':
        train_from_scratch(args)
    elif args.command == 'predict_risk':
        predict_disease_risk(args)
    elif args.command == 'demo':
        run_demo(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
