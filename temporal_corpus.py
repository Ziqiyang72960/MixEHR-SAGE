"""
Temporal Corpus Module for MixEHR-SAGE

This module provides functionality to process longitudinal EHR data into
discrete time steps for temporal topic modeling. It supports:
- Multiple bucketing strategies (yearly, monthly, visit-index)
- Per-patient per-time topic mixture computation
- Data leakage prevention (θ_t depends only on records up to time t)
"""

import numpy as np
import pandas as pd
import pickle
import os
import logging
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union
import torch

from corpus import Corpus, read_data_file

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalBucket:
    """Represents a single time bucket for a patient."""
    
    def __init__(self, patient_id: str, time_index: int, start_time, end_time, 
                 bucket_type: str = 'yearly'):
        self.patient_id = patient_id
        self.time_index = time_index
        self.start_time = start_time
        self.end_time = end_time
        self.bucket_type = bucket_type
        self.records = []  # List of (code, modality) tuples
        
    def add_record(self, code: str, modality: str, timestamp=None):
        """Add a medical record to this bucket."""
        self.records.append({
            'code': code,
            'modality': modality,
            'timestamp': timestamp
        })
    
    def get_bow_by_modality(self, vocab_mappings: Dict, modality_list: List[str]) -> List[Dict]:
        """
        Convert records to bag-of-words format for each modality.
        
        Args:
            vocab_mappings: Dict of {modality: {code: word_id}}
            modality_list: List of modality names in order
            
        Returns:
            List of dicts, one per modality: [{word_id: freq}, ...]
        """
        bow = [{} for _ in modality_list]
        
        for record in self.records:
            modality = record['modality']
            code = str(record['code']).strip()
            
            if modality in vocab_mappings and modality in modality_list:
                m = modality_list.index(modality)
                if code in vocab_mappings[modality]:
                    word_id = vocab_mappings[modality][code]
                    bow[m][word_id] = bow[m].get(word_id, 0) + 1
        
        return bow
    
    def __len__(self):
        return len(self.records)
    
    def __repr__(self):
        return f"TemporalBucket(patient={self.patient_id}, t={self.time_index}, records={len(self.records)})"


class TemporalPatient:
    """Represents a patient's longitudinal medical history."""
    
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.buckets: List[TemporalBucket] = []
        self.theta_sequence: Optional[torch.Tensor] = None  # T x K
        
    def add_bucket(self, bucket: TemporalBucket):
        """Add a time bucket to this patient's history."""
        self.buckets.append(bucket)
        # Keep buckets sorted by time index
        self.buckets.sort(key=lambda b: b.time_index)
    
    def get_bucket(self, time_index: int) -> Optional[TemporalBucket]:
        """Get bucket at specific time index."""
        for bucket in self.buckets:
            if bucket.time_index == time_index:
                return bucket
        return None
    
    def get_cumulative_bow(self, time_index: int, vocab_mappings: Dict, 
                          modality_list: List[str]) -> List[Dict]:
        """
        Get cumulative bag-of-words up to and including time_index.
        This ensures no data leakage: θ_t only depends on records up to time t.
        
        Args:
            time_index: Time index (inclusive)
            vocab_mappings: Dict of {modality: {code: word_id}}
            modality_list: List of modality names
            
        Returns:
            List of dicts for each modality with cumulative word counts
        """
        cumulative_bow = [{} for _ in modality_list]
        
        for bucket in self.buckets:
            if bucket.time_index <= time_index:
                bucket_bow = bucket.get_bow_by_modality(vocab_mappings, modality_list)
                for m, bow_m in enumerate(bucket_bow):
                    for word_id, freq in bow_m.items():
                        cumulative_bow[m][word_id] = cumulative_bow[m].get(word_id, 0) + freq
        
        return cumulative_bow
    
    @property
    def num_time_steps(self) -> int:
        return len(self.buckets)
    
    def __len__(self):
        return len(self.buckets)
    
    def __repr__(self):
        return f"TemporalPatient(id={self.patient_id}, T={len(self.buckets)})"


class TemporalCorpus:
    """
    Temporal corpus that organizes EHR data into discrete time buckets.
    
    Supports bucketing strategies:
    - 'yearly': Records grouped by calendar year
    - 'monthly': Records grouped by month
    - 'quarterly': Records grouped by quarter
    - 'visit': Records grouped by visit index (for pre-bucketed data)
    """
    
    BUCKET_TYPES = ['yearly', 'monthly', 'quarterly', 'visit']
    
    def __init__(self, bucket_type: str = 'yearly'):
        if bucket_type not in self.BUCKET_TYPES:
            raise ValueError(f"bucket_type must be one of {self.BUCKET_TYPES}")
        
        self.bucket_type = bucket_type
        self.patients: Dict[str, TemporalPatient] = {}
        self.vocab_mappings: Optional[Dict] = None
        self.modality_list: Optional[List[str]] = None
        self.min_time = None
        self.max_time = None
        
    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, bucket_type: str = 'yearly',
                      subject_col: str = 'SUBJECT_ID',
                      code_col: str = 'code',
                      time_col: str = 'timestamp',
                      modality_col: str = 'modality') -> 'TemporalCorpus':
        """
        Create TemporalCorpus from a pandas DataFrame.
        
        Args:
            df: DataFrame with columns for subject, code, timestamp, modality
            bucket_type: Bucketing strategy ('yearly', 'monthly', 'quarterly', 'visit')
            subject_col: Column name for patient ID
            code_col: Column name for medical code
            time_col: Column name for timestamp (or visit index for 'visit' bucket_type)
            modality_col: Column name for modality type
            
        Returns:
            TemporalCorpus object
        """
        corpus = cls(bucket_type=bucket_type)
        
        # Validate columns
        required_cols = [subject_col, code_col, time_col, modality_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        # Parse timestamps based on bucket type
        if bucket_type == 'visit':
            # For visit-based, time_col contains visit index (integer)
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        else:
            # For time-based, parse as datetime
            df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        
        # Process each row
        for _, row in df.iterrows():
            patient_id = str(row[subject_col])
            code = str(row[code_col])
            modality = str(row[modality_col])
            timestamp = row[time_col]
            
            # Skip invalid timestamps
            if pd.isna(timestamp):
                continue
            
            # Get or create patient
            if patient_id not in corpus.patients:
                corpus.patients[patient_id] = TemporalPatient(patient_id)
            patient = corpus.patients[patient_id]
            
            # Compute bucket index and boundaries
            bucket_info = corpus._compute_bucket_info(timestamp)
            time_index = bucket_info['index']
            start_time = bucket_info['start']
            end_time = bucket_info['end']
            
            # Get or create bucket
            bucket = patient.get_bucket(time_index)
            if bucket is None:
                bucket = TemporalBucket(
                    patient_id=patient_id,
                    time_index=time_index,
                    start_time=start_time,
                    end_time=end_time,
                    bucket_type=bucket_type
                )
                patient.add_bucket(bucket)
            
            # Add record to bucket
            bucket.add_record(code, modality, timestamp)
            
            # Update global time bounds
            if corpus.min_time is None or time_index < corpus.min_time:
                corpus.min_time = time_index
            if corpus.max_time is None or time_index > corpus.max_time:
                corpus.max_time = time_index
        
        return corpus
    
    @classmethod
    def from_file(cls, file_path: str, bucket_type: str = 'yearly', **kwargs) -> 'TemporalCorpus':
        """Load temporal corpus from file (CSV, TSV, JSON, TXT)."""
        df = read_data_file(file_path)
        return cls.from_dataframe(df, bucket_type=bucket_type, **kwargs)
    
    def _compute_bucket_info(self, timestamp) -> Dict:
        """
        Compute bucket index and boundaries for a timestamp.
        
        Args:
            timestamp: For 'visit' bucket_type, must be numeric (int/float).
                      For 'yearly', 'monthly', 'quarterly' bucket_types, 
                      must be a datetime object or datetime-compatible.
        
        Returns:
            Dict with 'index', 'start', 'end' keys
            
        Raises:
            TypeError: If timestamp type is incompatible with bucket_type
        """
        if self.bucket_type == 'visit':
            # Visit-based: timestamp is already the index (numeric)
            if not isinstance(timestamp, (int, float)):
                try:
                    timestamp = int(timestamp)
                except (ValueError, TypeError):
                    raise TypeError(
                        f"For 'visit' bucket_type, timestamp must be numeric, got {type(timestamp)}"
                    )
            return {
                'index': int(timestamp),
                'start': int(timestamp),
                'end': int(timestamp)
            }
        
        # Time-based bucketing - timestamp should be datetime
        if not hasattr(timestamp, 'year'):
            raise TypeError(
                f"For '{self.bucket_type}' bucket_type, timestamp must be datetime-compatible, "
                f"got {type(timestamp)}. Use bucket_type='visit' for numeric indices."
            )
        
        if self.bucket_type == 'yearly':
            year = timestamp.year
            return {
                'index': year,
                'start': datetime(year, 1, 1),
                'end': datetime(year, 12, 31)
            }
        
        elif self.bucket_type == 'monthly':
            # Index = year * 12 + month
            index = timestamp.year * 12 + timestamp.month
            year = timestamp.year
            month = timestamp.month
            
            # Compute end of month
            if month == 12:
                end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year, month + 1, 1) - timedelta(days=1)
            
            return {
                'index': index,
                'start': datetime(year, month, 1),
                'end': end
            }
        
        elif self.bucket_type == 'quarterly':
            # Index = year * 4 + quarter (0-3)
            quarter = (timestamp.month - 1) // 3
            index = timestamp.year * 4 + quarter
            year = timestamp.year
            
            quarter_starts = [1, 4, 7, 10]
            quarter_ends = [3, 6, 9, 12]
            
            start_month = quarter_starts[quarter]
            end_month = quarter_ends[quarter]
            
            if end_month == 12:
                end = datetime(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = datetime(year, end_month + 1, 1) - timedelta(days=1)
            
            return {
                'index': index,
                'start': datetime(year, start_month, 1),
                'end': end
            }
    
    def set_vocab_mappings(self, vocab_mappings: Dict, modality_list: List[str]):
        """Set vocabulary mappings for bag-of-words conversion."""
        self.vocab_mappings = vocab_mappings
        self.modality_list = modality_list
    
    def compute_theta_sequences(self, model, num_iterations: int = 10, 
                                use_cumulative: bool = True,
                                method: str = 'variational') -> Dict[str, torch.Tensor]:
        """
        Compute per-patient per-time topic mixture sequences θ_t.
        
        Args:
            model: Trained MixEHR-SAGE model
            num_iterations: Number of inference iterations
            use_cumulative: If True, use cumulative records up to time t (prevents data leakage)
                           If False, use only records in time bucket t
            method: Inference method ('variational' or 'gibbs')
            
        Returns:
            Dict mapping patient_id to theta sequence tensor (T x K)
        """
        if self.vocab_mappings is None:
            raise ValueError("vocab_mappings not set. Call set_vocab_mappings() first.")
        
        theta_sequences = {}
        
        for patient_id, patient in self.patients.items():
            T = patient.num_time_steps
            K = model.K
            theta_seq = torch.zeros(T, K, dtype=torch.double, device=device)
            
            for t, bucket in enumerate(patient.buckets):
                if use_cumulative:
                    # Cumulative BOW up to time t (prevents data leakage)
                    bow = patient.get_cumulative_bow(
                        bucket.time_index, 
                        self.vocab_mappings, 
                        self.modality_list
                    )
                else:
                    # Only records in time bucket t
                    bow = bucket.get_bow_by_modality(self.vocab_mappings, self.modality_list)
                
                # Check if bucket has any records
                has_records = any(len(bow_m) > 0 for bow_m in bow)
                
                if has_records:
                    theta_t = model.infer_theta_fast(bow, num_iterations=num_iterations, method=method)
                else:
                    # If no records, use uniform distribution or previous theta
                    if t > 0:
                        theta_t = theta_seq[t-1].clone()
                    else:
                        theta_t = torch.ones(K, dtype=torch.double, device=device) / K
                
                theta_seq[t] = theta_t
            
            patient.theta_sequence = theta_seq
            theta_sequences[patient_id] = theta_seq
        
        return theta_sequences
    
    def export_theta_sequences(self, output_path: str, format: str = 'csv'):
        """
        Export theta sequences to file for downstream prediction.
        
        Args:
            output_path: Path to output file
            format: Output format ('csv', 'pickle', 'pt')
        """
        if format == 'csv':
            rows = []
            for patient_id, patient in self.patients.items():
                if patient.theta_sequence is not None:
                    theta_seq = patient.theta_sequence.cpu().numpy()
                    for t, bucket in enumerate(patient.buckets):
                        row = {
                            'patient_id': patient_id,
                            'time_index': bucket.time_index,
                            'bucket_type': bucket.bucket_type,
                        }
                        # Add theta values
                        for k in range(theta_seq.shape[1]):
                            row[f'theta_{k}'] = theta_seq[t, k]
                        rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported theta sequences to {output_path}")
            
        elif format == 'pickle':
            data = {
                patient_id: patient.theta_sequence.cpu().numpy() 
                for patient_id, patient in self.patients.items()
                if patient.theta_sequence is not None
            }
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Exported theta sequences to {output_path}")
            
        elif format == 'pt':
            data = {
                patient_id: patient.theta_sequence 
                for patient_id, patient in self.patients.items()
                if patient.theta_sequence is not None
            }
            torch.save(data, output_path)
            logger.info(f"Exported theta sequences to {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @property
    def num_patients(self) -> int:
        return len(self.patients)
    
    @property
    def total_time_steps(self) -> int:
        return sum(patient.num_time_steps for patient in self.patients.values())
    
    def __len__(self):
        return len(self.patients)
    
    def __repr__(self):
        return (f"TemporalCorpus(bucket_type={self.bucket_type}, "
                f"patients={self.num_patients}, time_steps={self.total_time_steps})")


def generate_sample_temporal_data(output_path: str, num_patients: int = 50,
                                  time_range: Tuple[int, int] = (2015, 2020),
                                  codes_per_visit: Tuple[int, int] = (2, 6)) -> pd.DataFrame:
    """
    Generate sample temporal EHR data for testing.
    
    Args:
        output_path: Path to save generated data
        num_patients: Number of patients to generate
        time_range: (start_year, end_year) for data generation
        codes_per_visit: (min, max) codes per visit
        
    Returns:
        Generated DataFrame
    """
    np.random.seed(42)
    
    # Sample ICD codes (common chronic conditions)
    icd_codes = [
        'E11.9', 'I10', 'E78.5', 'J44.1', 'N18.3', 'I25.1', 'I50.0',
        'M54.5', 'K21.0', 'F32.9', 'G43.9', 'J45.9', 'K58.9'
    ]
    
    # Sample medication codes
    med_codes = ['C03AB01', 'C09AA02', 'A03FA01', 'A10BA02', 'C10AA05', 'N02BE01']
    
    # Sample procedure codes
    opcs_codes = ['K30100', 'K30200', 'K30300', 'K45100', 'K63100']
    
    records = []
    
    for i in range(num_patients):
        patient_id = f'patient_{i+1:03d}'
        
        # Random number of visits (2-8)
        num_visits = np.random.randint(2, 9)
        
        # Generate visit dates across time range
        start_date = datetime(time_range[0], 1, 1)
        end_date = datetime(time_range[1], 12, 31)
        date_range = (end_date - start_date).days
        
        visit_days = sorted(np.random.randint(0, date_range, num_visits))
        visit_dates = [start_date + timedelta(days=int(d)) for d in visit_days]
        
        for visit_idx, visit_date in enumerate(visit_dates):
            # Generate ICD codes for this visit
            num_icd = np.random.randint(codes_per_visit[0], codes_per_visit[1] + 1)
            selected_icd = np.random.choice(icd_codes, size=min(num_icd, len(icd_codes)), replace=False)
            
            for code in selected_icd:
                records.append({
                    'SUBJECT_ID': patient_id,
                    'code': code,
                    'timestamp': visit_date.strftime('%Y-%m-%d'),
                    'modality': 'icd'
                })
            
            # Maybe add medication (70% chance)
            if np.random.random() < 0.7:
                med_code = np.random.choice(med_codes)
                records.append({
                    'SUBJECT_ID': patient_id,
                    'code': med_code,
                    'timestamp': visit_idx,  # Use visit index for meds
                    'modality': 'med'
                })
            
            # Maybe add procedure (30% chance)
            if np.random.random() < 0.3:
                opcs_code = np.random.choice(opcs_codes)
                records.append({
                    'SUBJECT_ID': patient_id,
                    'code': opcs_code,
                    'timestamp': visit_date.strftime('%Y-%m-%d'),
                    'modality': 'opcs'
                })
    
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info(f"Generated {len(df)} temporal records for {num_patients} patients")
    
    return df


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    sample_data_path = './data/sample_temporal_data.csv'
    df = generate_sample_temporal_data(sample_data_path, num_patients=20)
    print(f"Generated sample data:\n{df.head(10)}")
    
    # Create temporal corpus
    corpus = TemporalCorpus.from_file(sample_data_path, bucket_type='yearly')
    print(f"\nTemporal corpus: {corpus}")
    
    # Show a sample patient
    patient = list(corpus.patients.values())[0]
    print(f"\nSample patient: {patient}")
    for bucket in patient.buckets:
        print(f"  {bucket}")
