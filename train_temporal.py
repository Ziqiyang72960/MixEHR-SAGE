#!/usr/bin/env python3
"""
Training script for temporal Markov chain model with MixEHR-SAGE

This script handles:
1. Mixed temporal granularity (medications in time bins, ICD/OPCS with specific dates)
2. Patient sequential data processing
3. Command-line training interface
4. Theta inference for new patients

Usage:
    # Train on temporal data
    python train_temporal.py ./data/ --output ./results/ --enable-temporal --num-time-steps 10

    # Infer theta for new patients
    python train_temporal.py ./data/ --infer-only --model-path ./results/model.pt
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import pickle
from typing import Dict, List, Tuple, Optional
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from MixEHR_SAGE import MixEHR_SAGE
from corpus import Corpus
from temporal_markov_utils import PatientSequenceGenerator, SequenceDataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MixedTemporalDataProcessor:
    """
    Handles mixed temporal granularity:
    - Medications: Time bins (baseline, 1st visit, 2nd visit, 3rd visit)
    - ICD/OPCS: Specific dates
    
    Aligns all data to common temporal grid based on visit dates
    """
    
    def __init__(self, data_dir: str, metadata_path: str):
        """
        Args:
            data_dir: Directory containing data files
            metadata_path: Path to metadata CSV
        """
        self.data_dir = Path(data_dir)
        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path, index_col='index')
        
    def load_dated_modality(self, modality_path: str, date_column: str = 'date') -> pd.DataFrame:
        """
        Load modality with specific dates (ICD, OPCS)
        
        Expected format:
            patient_id, date, code
        """
        df = pd.read_csv(self.data_dir / modality_path)
        
        # Ensure patient_id is integer for consistent comparisons
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].astype(int)
        
        # Convert date to datetime if present
        if date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        return df
    
    def load_binned_modality(self, modality_path: str, time_col: str = 'time_bin') -> pd.DataFrame:
        """
        Load modality with time bins (Medications)
        
        Expected format:
            patient_id, time_bin, code
        
        Where time_bin = {0: baseline, 1: first_visit, 2: second_visit, 3: third_visit}
        """
        df = pd.read_csv(self.data_dir / modality_path)
        
        # Ensure patient_id is integer for consistent comparisons
        if 'patient_id' in df.columns:
            df['patient_id'] = df['patient_id'].astype(int)
        
        # Ensure time_bin is integer for consistent comparisons
        if time_col in df.columns:
            df[time_col] = df[time_col].astype(int)
        
        return df
    
    def align_temporal_data(self, patient_id: int, 
                           dated_data: Dict[str, pd.DataFrame],
                           binned_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Align mixed temporal data for a single patient
        
        Creates visit sequence where:
        - Visit dates from ICD/OPCS determine temporal grid
        - Medication time bins mapped to nearest visits
        
        Args:
            patient_id: Patient identifier
            dated_data: Dict of {modality_name: DataFrame with dates}
            binned_data: Dict of {modality_name: DataFrame with time bins}
        
        Returns:
            List of visits with format:
            [
                {
                    'time': datetime or visit_index,
                    'words': {
                        modality_id: {word_id: frequency}
                    }
                },
                ...
            ]
        """
        visits = []
        
        # Step 1: Get visit dates from dated modalities (ICD, OPCS)
        visit_dates = []
        dated_records = {}
        
        for mod_name, df in dated_data.items():
            patient_df = df[df['patient_id'] == patient_id]
            if 'date' in patient_df.columns:
                dates = patient_df['date'].dropna().unique()
                visit_dates.extend(dates)
                dated_records[mod_name] = patient_df
        
        # Sort unique visit dates, filtering out NaT (Not a Time) values
        if visit_dates:
            # Remove NaT values before sorting
            valid_dates = [d for d in visit_dates if pd.notna(d)]
            if valid_dates:
                visit_dates = sorted(set(valid_dates))
            else:
                # If all dates were invalid, fallback to time bins
                visit_dates = list(range(4))
        else:
            # If no dated data, use time bins directly
            # Assume 4 time bins map to 4 visits
            visit_dates = list(range(4))
        
        # Step 2: For each visit date, collect all modality data
        for visit_idx, visit_time in enumerate(visit_dates):
            visit_words = {}
            
            # Dated modalities: exact date match
            for mod_id, (mod_name, df) in enumerate(dated_records.items()):
                if isinstance(visit_time, int):
                    # No actual dates, just use all data
                    patient_data = df
                else:
                    # Match by date (allow small window)
                    patient_data = df[df['date'] == visit_time]
                
                if len(patient_data) > 0:
                    words = {}
                    for _, row in patient_data.iterrows():
                        word_id = row['code'] if 'code' in row else row.get('word_id', 0)
                        words[word_id] = words.get(word_id, 0) + 1
                    visit_words[mod_id] = words
            
            # Binned modalities: map time bin to visit
            for mod_id_offset, (mod_name, df) in enumerate(binned_data.items()):
                mod_id = len(dated_records) + mod_id_offset
                patient_df = df[df['patient_id'] == patient_id]
                
                # Map visit_idx to time_bin
                # Simple mapping: visit_0 -> bin_0, visit_1 -> bin_1, etc.
                time_bin = min(visit_idx, 3)  # Medications have max 4 bins
                
                bin_data = patient_df[patient_df['time_bin'] == time_bin]
                
                if len(bin_data) > 0:
                    words = {}
                    for _, row in bin_data.iterrows():
                        word_id = row['code'] if 'code' in row else row.get('word_id', 0)
                        words[word_id] = words.get(word_id, 0) + 1
                    visit_words[mod_id] = words
            
            # Only add visit if has data
            if visit_words:
                visits.append({
                    'time': visit_time if not isinstance(visit_time, int) else visit_idx,
                    'words': visit_words
                })
        
        return visits
    
    def process_all_patients(self, vocab_sizes: List[int]) -> Tuple[Dict, Dict]:
        """
        Process all patients and create sequential data
        
        Args:
            vocab_sizes: List of vocabulary sizes per modality
        
        Returns:
            patient_sequences: Dict mapping patient_id to visit list
            patient_metadata: Dict with patient information
        """
        # Categorize modalities by temporal type
        dated_mods = {}  # ICD, OPCS with dates
        binned_mods = {}  # Medications with time bins
        
        for mod_name, row in self.metadata.iterrows():
            path = row['path']
            # Explicitly specify dtype for patient_id and time_bin columns during CSV reading
            # This prevents pandas from reading them as strings
            dtype_spec = {'patient_id': int}
            # Check if file has time_bin column by reading first line
            try:
                sample_df = pd.read_csv(self.data_dir / path, nrows=0)
                if 'time_bin' in sample_df.columns:
                    dtype_spec['time_bin'] = int
            except:
                pass
            df = pd.read_csv(self.data_dir / path, dtype=dtype_spec)
            
            if 'date' in df.columns:
                # Convert date to datetime
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                dated_mods[mod_name] = df
            elif 'time_bin' in df.columns:
                # time_bin already int from dtype spec
                binned_mods[mod_name] = df
            else:
                # Default: treat as single time point
                binned_mods[mod_name] = df
        
        # Get all patient IDs (already integers from dtype specification)
        all_patients = set()
        for df in list(dated_mods.values()) + list(binned_mods.values()):
            all_patients.update(df['patient_id'].unique())
        
        patient_sequences = {}
        patient_metadata = {}
        
        print(f"Processing {len(all_patients)} patients...")
        for patient_id in sorted(all_patients):
            visits = self.align_temporal_data(patient_id, dated_mods, binned_mods)
            
            if visits:
                patient_sequences[patient_id] = visits
                patient_metadata[patient_id] = {
                    'num_visits': len(visits),
                    'first_visit': visits[0]['time'],
                    'last_visit': visits[-1]['time']
                }
        
        print(f"Processed {len(patient_sequences)} patients with temporal data")
        return patient_sequences, patient_metadata


def load_corpus_and_seeds(data_dir: str, metadata_path: str):
    """
    Load corpus and seed topic matrix
    """
    print(f"Loading corpus from {data_dir}...")
    
    # Use existing corpus processing
    # For now, create mock corpus
    metadata = pd.read_csv(metadata_path, index_col='index')
    
    # Load first modality to get patient count
    first_mod_path = metadata.iloc[0]['path']
    first_df = pd.read_csv(Path(data_dir) / first_mod_path)
    num_patients = first_df['patient_id'].nunique()
    
    # Mock vocabulary sizes (should be computed from data)
    vocab_sizes = [1000, 500, 300]  # ICD, medication, OPCS
    
    # Create mock corpus with PyTorch DataLoader compatibility
    class MockCorpus:
        def __init__(self):
            self.D = num_patients
            self.V = vocab_sizes
            # C is total word count per modality across all documents
            # Estimate: ~20 words per patient per modality on average
            self.C = [num_patients * 20 for _ in vocab_sizes]
            
            # Create mock documents for DataLoader compatibility
            from corpus import Corpus
            self.documents = []
            for doc_id in range(self.D):
                doc = Corpus.Document(doc_id, modality_num=len(vocab_sizes))
                # Add some mock words to each document
                for m, vocab_size in enumerate(vocab_sizes):
                    num_words = np.random.randint(10, 30)
                    for _ in range(num_words):
                        word_id = np.random.randint(0, vocab_size)
                        freq = np.random.randint(1, 5)
                        doc.append_record(word_id, freq, m)
                self.documents.append(doc)
        
        def __len__(self):
            return self.D
        
        def __getitem__(self, idx):
            return self.documents[idx]
    
    corpus = MockCorpus()
    
    # Create mock seed matrix (first modality is guided)
    num_topics = 50
    seeds = torch.zeros(vocab_sizes[0], num_topics, dtype=torch.double)
    for k in range(num_topics):
        num_seeds = np.random.randint(5, 15)
        seed_words = np.random.choice(vocab_sizes[0], size=num_seeds, replace=False)
        seeds[seed_words, k] = 1.0
    
    return corpus, seeds, vocab_sizes


def train_temporal_model(data_dir: str, metadata_path: str, output_dir: str,
                         enable_temporal: bool = True, num_time_steps: int = 10,
                         epochs: int = 5, batch_size: int = 50):
    """
    Train temporal Markov chain model
    """
    print("="*70)
    print("Training Temporal Markov Chain Model")
    print("="*70)
    
    # Load corpus and seeds
    corpus, seeds, vocab_sizes = load_corpus_and_seeds(data_dir, metadata_path)
    
    print(f"\nCorpus statistics:")
    print(f"  Patients: {corpus.D}")
    print(f"  Modalities: {len(vocab_sizes)}")
    print(f"  Vocabulary sizes: {vocab_sizes}")
    print(f"  Topics: {seeds.shape[1]}")
    
    # Process temporal data
    processor = MixedTemporalDataProcessor(data_dir, metadata_path)
    patient_sequences, patient_metadata = processor.process_all_patients(vocab_sizes)
    
    print(f"\nTemporal data:")
    print(f"  Patients with sequences: {len(patient_sequences)}")
    avg_visits = np.mean([m['num_visits'] for m in patient_metadata.values()])
    print(f"  Average visits per patient: {avg_visits:.1f}")
    
    # Initialize model
    print(f"\nInitializing model...")
    modality_list = ['icd', 'medication', 'opcs']
    
    model = MixEHR_SAGE(
        corpus=corpus,
        seeds_topic_matrix=seeds,
        modality_list=modality_list,
        guided_modality=0,  # ICD is guided
        batch_size=batch_size,
        enable_temporal=enable_temporal,
        num_time_steps=num_time_steps,
        out=output_dir
    )
    
    # Create data loader
    loader = SequenceDataLoader(patient_sequences, vocab_sizes)
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Process patients in batches
        patient_ids = list(patient_sequences.keys())
        np.random.shuffle(patient_ids)
        
        total_kl = 0.0
        num_processed = 0
        
        for i in range(0, len(patient_ids), batch_size):
            batch_ids = patient_ids[i:i+batch_size]
            
            for patient_id in batch_ids:
                try:
                    # Get patient sequence
                    seq_data = loader.get_patient_sequence_data(patient_id)
                    
                    # Perform VAE inference
                    theta_samples, mu, logvar = model.infer_theta_variational(
                        seq_data, patient_id
                    )
                    
                    # Compute KL divergence
                    kl = model.compute_markov_chain_kl(theta_samples, mu, logvar)
                    total_kl += kl.item()
                    num_processed += 1
                    
                except Exception as e:
                    print(f"  Error processing patient {patient_id}: {e}")
            
            if (i // batch_size + 1) % 5 == 0:
                avg_kl = total_kl / max(num_processed, 1)
                print(f"  Batch {i//batch_size + 1}: Avg KL = {avg_kl:.3f}")
        
        avg_kl = total_kl / max(num_processed, 1)
        print(f"Epoch {epoch+1} complete. Avg KL: {avg_kl:.3f}")
    
    # Save model and results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save temporal theta
    theta_path = os.path.join(output_dir, 'temporal_theta.pt')
    model.save_temporal_theta(theta_path)
    print(f"\nSaved temporal theta to {theta_path}")
    
    # Save model state
    model_path = os.path.join(output_dir, 'model.pt')
    torch.save({
        'model_state': model.state_dict(),
        'vocab_sizes': vocab_sizes,
        'modality_list': modality_list,
        'num_topics': seeds.shape[1],
        'patient_metadata': patient_metadata
    }, model_path)
    print(f"Saved model to {model_path}")
    
    # Save patient-topic mapping
    results_path = os.path.join(output_dir, 'patient_theta_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump({
            'patient_sequences': patient_sequences,
            'patient_metadata': patient_metadata,
            'theta_temporal': model.theta_temporal.cpu().numpy(),
            'patient_time_mask': model.patient_time_mask.cpu().numpy()
        }, f)
    print(f"Saved results to {results_path}")
    
    return model, patient_sequences, patient_metadata


def infer_new_patients(model_path: str, data_dir: str, metadata_path: str,
                      output_path: str):
    """
    Infer theta for new patients given a trained model
    """
    print("="*70)
    print("Inferring Theta for New Patients")
    print("="*70)
    
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    vocab_sizes = checkpoint['vocab_sizes']
    modality_list = checkpoint['modality_list']
    num_topics = checkpoint['num_topics']
    
    # Load corpus and seeds
    corpus, seeds, _ = load_corpus_and_seeds(data_dir, metadata_path)
    
    # Initialize model
    model = MixEHR_SAGE(
        corpus=corpus,
        seeds_topic_matrix=seeds,
        modality_list=modality_list,
        guided_modality=0,
        enable_temporal=True,
        num_time_steps=10
    )
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    
    print("Model loaded successfully")
    
    # Process new patient data
    processor = MixedTemporalDataProcessor(data_dir, metadata_path)
    patient_sequences, patient_metadata = processor.process_all_patients(vocab_sizes)
    
    # Create data loader
    loader = SequenceDataLoader(patient_sequences, vocab_sizes)
    
    # Inference
    print(f"\nInferring theta for {len(patient_sequences)} patients...")
    results = {}
    
    for patient_id in patient_sequences.keys():
        try:
            seq_data = loader.get_patient_sequence_data(patient_id)
            theta_samples, mu, logvar = model.infer_theta_variational(seq_data, patient_id)
            
            results[patient_id] = {
                'theta_sequence': theta_samples.cpu().numpy(),
                'num_visits': len(seq_data),
                'dominant_topics': [torch.argmax(theta_samples[t]).item() 
                                   for t in range(len(seq_data))],
                'topic_evolution': theta_samples.cpu().numpy().tolist()
            }
            
            # Print sample results
            if patient_id in list(patient_sequences.keys())[:3]:
                print(f"\nPatient {patient_id}:")
                print(f"  Visits: {len(seq_data)}")
                print(f"  Dominant topics: {results[patient_id]['dominant_topics']}")
                
        except Exception as e:
            print(f"Error processing patient {patient_id}: {e}")
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved inference results to {output_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Train temporal Markov chain model with mixed granularity data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Train model with temporal inference
  python train_temporal.py ./data/ --enable-temporal --num-time-steps 10 --epochs 5

  # Train without temporal (baseline)
  python train_temporal.py ./data/ --epochs 5

  # Infer theta for new patients
  python train_temporal.py ./data/ --infer-only --model-path ./results/model.pt

  # Custom settings
  python train_temporal.py ./data/ --output ./my_results/ --batch-size 100 --epochs 10

Data Format:
  
  Metadata file (ukbb_metadata.csv):
    index,path,word_column
    icd,icd_data.csv,code
    medication,med_data.csv,code
    opcs,opcs_data.csv,code
  
  Dated modality (ICD, OPCS):
    patient_id,date,code
    1001,2020-01-15,J45.0
    1001,2020-03-20,E11.9
  
  Binned modality (Medications):
    patient_id,time_bin,code
    1001,0,N02BE01  # 0=baseline
    1001,1,N02BE01  # 1=first visit
    1001,2,A10BA02 # 2=second visit

Output:
  - temporal_theta.pt: D×T×K tensor with theta for each patient-visit
  - model.pt: Model checkpoint
  - patient_theta_results.pkl: Detailed results
  - patient_theta_inference.json: Theta sequences for new patients
        """
    )
    
    parser.add_argument('data_dir', help='Directory containing data files')
    parser.add_argument('--metadata', default='ukbb_metadata.csv',
                       help='Metadata file name (default: ukbb_metadata.csv)')
    parser.add_argument('--output', '-o', default='./results/',
                       help='Output directory (default: ./results/)')
    parser.add_argument('--enable-temporal', action='store_true',
                       help='Enable temporal Markov chain inference')
    parser.add_argument('--num-time-steps', type=int, default=10,
                       help='Maximum time steps per patient (default: 10)')
    parser.add_argument('--epochs', '-e', type=int, default=5,
                       help='Number of training epochs (default: 5)')
    parser.add_argument('--batch-size', '-b', type=int, default=50,
                       help='Batch size (default: 50)')
    parser.add_argument('--infer-only', action='store_true',
                       help='Only perform inference on new patients (requires --model-path)')
    parser.add_argument('--model-path', help='Path to trained model for inference')
    parser.add_argument('--inference-output', default='./patient_theta_inference.json',
                       help='Output file for inference results (default: ./patient_theta_inference.json)')
    
    args = parser.parse_args()
    
    # Validate arguments
    metadata_path = os.path.join(args.data_dir, args.metadata)
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found: {metadata_path}")
        sys.exit(1)
    
    if args.infer_only:
        if not args.model_path:
            print("Error: --model-path required for inference mode")
            sys.exit(1)
        if not os.path.exists(args.model_path):
            print(f"Error: Model file not found: {args.model_path}")
            sys.exit(1)
        
        # Inference mode
        infer_new_patients(args.model_path, args.data_dir, metadata_path,
                          args.inference_output)
    else:
        # Training mode
        train_temporal_model(
            args.data_dir,
            metadata_path,
            args.output,
            enable_temporal=args.enable_temporal,
            num_time_steps=args.num_time_steps,
            epochs=args.epochs,
            batch_size=args.batch_size
        )


if __name__ == '__main__':
    main()
