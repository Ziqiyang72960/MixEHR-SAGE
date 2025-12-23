"""
Temporal Data Generation and Preprocessing Utilities for MixEHR-SAGE

This module provides utilities for:
1. Generating synthetic time series EHR data
2. Binning patient data by age/time
3. Creating temporal sequences for LSTM input
4. Handling missing temporal metadata
"""

import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalDataGenerator:
    """
    Generates synthetic or processes real temporal EHR data for MixEHR-SAGE
    """
    
    def __init__(self, num_time_steps: int = 10, min_age: int = 0, max_age: int = 100):
        """
        Args:
            num_time_steps: Number of time bins to create
            min_age: Minimum age for binning
            max_age: Maximum age for binning
        """
        self.T = num_time_steps
        self.min_age = min_age
        self.max_age = max_age
        self.age_bins = np.linspace(min_age, max_age, num_time_steps + 1)
        
    def get_time_bin(self, age: float) -> int:
        """
        Get the time bin index for a given age
        
        Args:
            age: Patient age
            
        Returns:
            Time bin index (0 to T-1)
        """
        if age < self.min_age:
            return 0
        if age >= self.max_age:
            return self.T - 1
        
        bin_idx = np.digitize(age, self.age_bins) - 1
        return min(max(bin_idx, 0), self.T - 1)
    
    def create_temporal_corpus_from_ages(self, corpus, patient_ages: Dict[int, float]) -> Dict[int, List]:
        """
        Create temporal bins for corpus based on patient ages
        
        Args:
            corpus: MixEHR corpus object
            patient_ages: Dictionary mapping document_id to patient age
            
        Returns:
            time_bins: Dictionary mapping time_bin_idx to list of document_ids
        """
        time_bins = defaultdict(list)
        
        for doc_id in range(corpus.D):
            if doc_id in patient_ages:
                age = patient_ages[doc_id]
                time_bin = self.get_time_bin(age)
                time_bins[time_bin].append(doc_id)
            else:
                # If age is missing, assign to middle bin
                time_bins[self.T // 2].append(doc_id)
        
        return dict(time_bins)
    
    def aggregate_word_distributions_by_time(self, corpus, time_bins: Dict[int, List], 
                                            modality: int = 0) -> torch.Tensor:
        """
        Aggregate word distributions for each time bin
        
        Args:
            corpus: MixEHR corpus object
            time_bins: Dictionary mapping time_bin_idx to list of document_ids
            modality: Which modality to aggregate (default: 0, typically ICD codes)
            
        Returns:
            time_word_dist: T x V tensor, normalized word distributions per time bin
        """
        V = corpus.V[modality]
        time_word_dist = torch.zeros(self.T, V, dtype=torch.double, device=device)
        
        for t in range(self.T):
            if t in time_bins and len(time_bins[t]) > 0:
                # Aggregate word counts for all documents in this time bin
                word_counts = torch.zeros(V, dtype=torch.double, device=device)
                
                for doc_id in time_bins[t]:
                    doc = corpus.docs[doc_id]
                    if modality in doc.words_dict:
                        for word_id, freq in doc.words_dict[modality].items():
                            word_counts[word_id] += freq
                
                # Normalize to get distribution
                total = word_counts.sum()
                if total > 0:
                    time_word_dist[t] = word_counts / total
                else:
                    # Uniform distribution if no data
                    time_word_dist[t] = torch.ones(V, dtype=torch.double, device=device) / V
            else:
                # No documents in this bin - use uniform distribution
                time_word_dist[t] = torch.ones(V, dtype=torch.double, device=device) / V
        
        return time_word_dist
    
    def generate_synthetic_temporal_data(self, num_patients: int, vocab_size: int,
                                        words_per_patient: int = 50,
                                        trend_strength: float = 0.3) -> Tuple[List, Dict]:
        """
        Generate synthetic temporal EHR data for testing
        
        Args:
            num_patients: Number of synthetic patients
            vocab_size: Size of vocabulary
            words_per_patient: Average number of words per patient
            trend_strength: How much word distributions change over time (0-1)
            
        Returns:
            documents: List of (doc_id, {word_id: frequency}) tuples
            patient_ages: Dictionary mapping doc_id to age
        """
        documents = []
        patient_ages = {}
        
        # Create time-varying word probabilities
        base_probs = np.random.dirichlet(np.ones(vocab_size) * 0.5)
        
        for doc_id in range(num_patients):
            # Random age
            age = np.random.uniform(self.min_age, self.max_age)
            patient_ages[doc_id] = age
            
            # Time-varying probability
            time_bin = self.get_time_bin(age)
            time_factor = (time_bin / self.T) * trend_strength
            
            # Shift probabilities based on time
            word_probs = base_probs * (1 + time_factor * np.sin(np.arange(vocab_size) * 2 * np.pi / vocab_size))
            word_probs = word_probs / word_probs.sum()
            
            # Sample words
            num_words = max(10, int(np.random.poisson(words_per_patient)))
            words = np.random.choice(vocab_size, size=num_words, p=word_probs)
            
            # Count frequencies
            word_freq = {}
            unique, counts = np.unique(words, return_counts=True)
            for word_id, count in zip(unique, counts):
                word_freq[int(word_id)] = int(count)
            
            documents.append((doc_id, word_freq))
        
        return documents, patient_ages
    
    def load_patient_ages_from_metadata(self, metadata_path: str, 
                                       id_column: str = 'patient_id',
                                       age_column: str = 'age') -> Dict[int, float]:
        """
        Load patient ages from metadata CSV file
        
        Args:
            metadata_path: Path to CSV file with patient metadata
            id_column: Name of column containing patient/document IDs
            age_column: Name of column containing ages
            
        Returns:
            patient_ages: Dictionary mapping document_id to age
        """
        try:
            df = pd.read_csv(metadata_path)
            patient_ages = {}
            
            for idx, row in df.iterrows():
                doc_id = int(row[id_column])
                age = float(row[age_column])
                patient_ages[doc_id] = age
            
            return patient_ages
        except Exception as e:
            print(f"Error loading patient ages: {e}")
            return {}


class TemporalSequencePreprocessor:
    """
    Preprocesses temporal sequences for LSTM input
    """
    
    @staticmethod
    def create_sliding_windows(time_series_data: torch.Tensor, 
                               window_size: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create sliding windows for temporal prediction
        
        Args:
            time_series_data: T x V tensor
            window_size: Size of sliding window
            
        Returns:
            X: (T-window_size) x window_size x V input sequences
            Y: (T-window_size) x V target sequences
        """
        T, V = time_series_data.shape
        if T <= window_size:
            raise ValueError(f"Time series too short ({T}) for window size {window_size}")
        
        num_windows = T - window_size
        X = torch.zeros(num_windows, window_size, V, dtype=time_series_data.dtype, 
                       device=time_series_data.device)
        Y = torch.zeros(num_windows, V, dtype=time_series_data.dtype,
                       device=time_series_data.device)
        
        for i in range(num_windows):
            X[i] = time_series_data[i:i+window_size]
            Y[i] = time_series_data[i+window_size]
        
        return X, Y
    
    @staticmethod
    def smooth_temporal_sequence(time_series_data: torch.Tensor,
                                window_size: int = 3) -> torch.Tensor:
        """
        Apply moving average smoothing to temporal sequence
        
        Args:
            time_series_data: T x V tensor
            window_size: Size of smoothing window
            
        Returns:
            smoothed: T x V tensor
        """
        T, V = time_series_data.shape
        smoothed = torch.zeros_like(time_series_data)
        
        for t in range(T):
            start = max(0, t - window_size // 2)
            end = min(T, t + window_size // 2 + 1)
            smoothed[t] = time_series_data[start:end].mean(dim=0)
        
        return smoothed
    
    @staticmethod
    def interpolate_missing_time_bins(time_series_data: torch.Tensor,
                                     missing_mask: torch.Tensor) -> torch.Tensor:
        """
        Interpolate missing time bins using linear interpolation
        
        Args:
            time_series_data: T x V tensor
            missing_mask: T tensor, 1 if time bin has data, 0 if missing
            
        Returns:
            interpolated: T x V tensor
        """
        T, V = time_series_data.shape
        interpolated = time_series_data.clone()
        
        # Find gaps
        has_data = missing_mask.bool()
        
        for v in range(V):
            # Get values and time indices where data exists
            values = time_series_data[has_data, v]
            time_indices = torch.where(has_data)[0].float()
            
            if len(time_indices) >= 2:
                # Interpolate missing values
                all_times = torch.arange(T, dtype=torch.float, device=device)
                interpolated[:, v] = torch.from_numpy(
                    np.interp(all_times.cpu().numpy(),
                             time_indices.cpu().numpy(),
                             values.cpu().numpy())
                ).to(device)
        
        return interpolated


def example_usage():
    """
    Example of how to use temporal utilities
    """
    print("Example: Generating synthetic temporal data")
    
    # Initialize generator
    gen = TemporalDataGenerator(num_time_steps=10, min_age=0, max_age=100)
    
    # Generate synthetic data
    documents, patient_ages = gen.generate_synthetic_temporal_data(
        num_patients=1000,
        vocab_size=500,
        words_per_patient=50
    )
    
    print(f"Generated {len(documents)} documents")
    print(f"Sample ages: {list(patient_ages.values())[:5]}")
    print(f"Sample document: {documents[0]}")
    
    # Example of binning
    time_bins = {}
    for doc_id, age in patient_ages.items():
        time_bin = gen.get_time_bin(age)
        if time_bin not in time_bins:
            time_bins[time_bin] = []
        time_bins[time_bin].append(doc_id)
    
    print(f"\nTime bins distribution:")
    for t in range(gen.T):
        count = len(time_bins.get(t, []))
        print(f"  Time bin {t} ({gen.age_bins[t]:.1f}-{gen.age_bins[t+1]:.1f} years): {count} patients")


if __name__ == "__main__":
    example_usage()
