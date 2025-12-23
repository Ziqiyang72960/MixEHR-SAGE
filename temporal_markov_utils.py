"""
Temporal Data Generation and Preprocessing for Markov Chain Dynamic Topic Models

This module provides utilities for:
1. Generating patient sequential data (multiple visits per patient)
2. Creating observation sequences for VAE/LSTM processing
3. Handling longitudinal EHR data
4. Time-ordered visit management
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PatientSequenceGenerator:
    """
    Generates and manages patient sequential data for Markov chain temporal modeling
    Each patient can have multiple visits over time, creating a sequence
    """
    
    def __init__(self, vocab_sizes: List[int], num_modalities: int = 2):
        """
        Args:
            vocab_sizes: List of vocabulary sizes for each modality [V_0, V_1, ...]
            num_modalities: Number of modalities (e.g., ICD, medication, procedures)
        """
        self.vocab_sizes = vocab_sizes
        self.num_modalities = num_modalities
        
    def create_patient_sequence(self, patient_visits: List[Dict], patient_id: int):
        """
        Create a temporal sequence for a single patient from their visits
        
        Args:
            patient_visits: List of visit dictionaries, each containing:
                           {
                               'time': timestamp or visit number,
                               'words': {modality_id: {word_id: frequency}}
                           }
            patient_id: Patient identifier
        
        Returns:
            sequence_data: List of T dictionaries with BOW tensors for each modality
                          Format: [{modality_0: BOW_tensor, modality_1: BOW_tensor}, ...]
        """
        # Sort visits by time
        sorted_visits = sorted(patient_visits, key=lambda x: x['time'])
        
        sequence_data = []
        for visit in sorted_visits:
            visit_obs = {}
            
            for m in range(self.num_modalities):
                # Create BOW tensor for this modality
                bow = torch.zeros(self.vocab_sizes[m], dtype=torch.double, device=device)
                
                if m in visit['words']:
                    for word_id, freq in visit['words'][m].items():
                        bow[word_id] = freq
                
                visit_obs[m] = bow
            
            sequence_data.append(visit_obs)
        
        return sequence_data
    
    def generate_synthetic_patient_sequences(self, num_patients: int = 100,
                                             min_visits: int = 2,
                                             max_visits: int = 10,
                                             words_per_visit: int = 30,
                                             progression_strength: float = 0.3):
        """
        Generate synthetic longitudinal patient data with disease progression
        
        Args:
            num_patients: Number of patients to generate
            min_visits: Minimum number of visits per patient
            max_visits: Maximum number of visits per patient
            words_per_visit: Average number of words per visit
            progression_strength: How much topics shift over time (0-1)
        
        Returns:
            patient_sequences: Dictionary mapping patient_id to list of visits
            patient_metadata: Dictionary with patient information
        """
        patient_sequences = {}
        patient_metadata = {}
        
        for patient_id in range(num_patients):
            # Random number of visits
            num_visits = np.random.randint(min_visits, max_visits + 1)
            
            # Generate visit times (e.g., days from first visit)
            visit_times = sorted(np.random.choice(365 * 5, size=num_visits, replace=False))
            
            # Initial topic distribution for this patient
            initial_topics = np.random.dirichlet(np.ones(50) * 0.5)
            
            visits = []
            current_topics = initial_topics.copy()
            
            for t, visit_time in enumerate(visit_times):
                visit = {
                    'time': int(visit_time),
                    'words': {}
                }
                
                # Evolve topics slightly (Markov transition)
                if t > 0:
                    # Add Gaussian noise for topic evolution
                    noise = np.random.normal(0, 0.1 * progression_strength, size=len(current_topics))
                    current_topics = current_topics + noise
                    current_topics = np.maximum(current_topics, 0)  # Keep non-negative
                    current_topics = current_topics / current_topics.sum()  # Renormalize
                
                # Generate words for each modality based on current topics
                for m in range(self.num_modalities):
                    modality_words = {}
                    
                    # Sample number of words for this visit/modality
                    num_words = max(5, int(np.random.poisson(words_per_visit)))
                    
                    # Sample topics for words based on current topic distribution
                    word_topics = np.random.choice(len(current_topics), size=num_words, p=current_topics)
                    
                    # For each topic, sample words from vocabulary
                    for topic in word_topics:
                        # Simple: each topic prefers certain words (topic % vocab_size)
                        word_id = (topic * 10 + np.random.randint(0, 10)) % self.vocab_sizes[m]
                        modality_words[word_id] = modality_words.get(word_id, 0) + 1
                    
                    visit['words'][m] = modality_words
                
                visits.append(visit)
            
            patient_sequences[patient_id] = visits
            patient_metadata[patient_id] = {
                'num_visits': num_visits,
                'first_visit_time': visit_times[0],
                'last_visit_time': visit_times[-1]
            }
        
        return patient_sequences, patient_metadata
    
    def load_patient_sequences_from_csv(self, visits_csv: str, 
                                       patient_col: str = 'patient_id',
                                       time_col: str = 'visit_time'):
        """
        Load patient sequential data from CSV file
        
        Args:
            visits_csv: Path to CSV with columns: patient_id, visit_time, ...
            patient_col: Name of patient ID column
            time_col: Name of time/visit order column
        
        Returns:
            patient_sequences: Dictionary mapping patient_id to list of visits
        """
        df = pd.read_csv(visits_csv)
        
        patient_sequences = defaultdict(list)
        
        # Group by patient
        for patient_id, patient_df in df.groupby(patient_col):
            # Sort by time
            patient_df = patient_df.sort_values(time_col)
            
            for _, row in patient_df.iterrows():
                visit = {
                    'time': row[time_col],
                    'words': {}  # Would need to parse from additional columns
                }
                patient_sequences[patient_id].append(visit)
        
        return dict(patient_sequences)


class SequenceDataLoader:
    """
    DataLoader for sequential patient data compatible with MixEHR-SAGE
    """
    
    def __init__(self, patient_sequences: Dict, vocab_sizes: List[int]):
        """
        Args:
            patient_sequences: Dictionary mapping patient_id to list of visits
            vocab_sizes: List of vocabulary sizes per modality
        """
        self.patient_sequences = patient_sequences
        self.vocab_sizes = vocab_sizes
        self.patient_ids = list(patient_sequences.keys())
    
    def get_patient_sequence_data(self, patient_id: int):
        """
        Get formatted sequence data for a patient
        
        Args:
            patient_id: Patient identifier
        
        Returns:
            sequence_data: List of observation dictionaries
        """
        if patient_id not in self.patient_sequences:
            raise ValueError(f"Patient {patient_id} not found")
        
        visits = self.patient_sequences[patient_id]
        sequence_data = []
        
        for visit in visits:
            visit_obs = {}
            for m, V_m in enumerate(self.vocab_sizes):
                bow = torch.zeros(V_m, dtype=torch.double, device=device)
                
                if m in visit['words']:
                    for word_id, freq in visit['words'][m].items():
                        if word_id < V_m:  # Ensure within vocabulary
                            bow[word_id] = freq
                
                visit_obs[m] = bow
            
            sequence_data.append(visit_obs)
        
        return sequence_data
    
    def get_batch(self, patient_ids: List[int]):
        """
        Get batch of patient sequences
        
        Args:
            patient_ids: List of patient IDs
        
        Returns:
            batch: List of (patient_id, sequence_data) tuples
        """
        batch = []
        for pid in patient_ids:
            seq_data = self.get_patient_sequence_data(pid)
            batch.append((pid, seq_data))
        return batch


def example_usage():
    """
    Example of generating and using patient sequences
    """
    print("=" * 70)
    print("Patient Sequential Data Generation for Markov Chain Models")
    print("=" * 70)
    
    # Initialize generator
    vocab_sizes = [500, 300]  # ICD: 500, Medication: 300
    gen = PatientSequenceGenerator(vocab_sizes, num_modalities=2)
    
    # Generate synthetic data
    print("\nGenerating synthetic patient sequences...")
    patient_sequences, metadata = gen.generate_synthetic_patient_sequences(
        num_patients=50,
        min_visits=3,
        max_visits=8,
        progression_strength=0.4
    )
    
    print(f"Generated {len(patient_sequences)} patients")
    
    # Show example patient
    patient_id = 0
    visits = patient_sequences[patient_id]
    print(f"\nPatient {patient_id} has {len(visits)} visits:")
    for i, visit in enumerate(visits):
        total_words = sum(sum(words.values()) for words in visit['words'].values())
        print(f"  Visit {i+1} at time {visit['time']}: {total_words} total words")
        for m in visit['words']:
            print(f"    Modality {m}: {len(visit['words'][m])} unique words")
    
    # Create data loader
    print("\nCreating data loader...")
    loader = SequenceDataLoader(patient_sequences, vocab_sizes)
    
    # Get sequence data for patient
    seq_data = loader.get_patient_sequence_data(patient_id)
    print(f"\nSequence data for patient {patient_id}:")
    print(f"  Number of time steps: {len(seq_data)}")
    print(f"  Each time step has {len(seq_data[0])} modalities")
    print(f"  Modality 0 BOW shape: {seq_data[0][0].shape}")
    
    print("\n" + "=" * 70)
    print("Ready to use with MixEHR-SAGE temporal inference!")
    print("=" * 70)


if __name__ == "__main__":
    example_usage()
