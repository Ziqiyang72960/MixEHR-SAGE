#!/usr/bin/env python3
"""
Temporal disease prediction for MixEHR-SAGE with time-based patient records.

This script implements three approaches for temporal disease prediction:
1. LSTM-based Temporal VAE: Learn q(eta_t | X_1...t-1) with temporal priors p(theta_t | theta_t-1)
2. Simple Regression: Predict visit within time window or time gap to next visit
3. Autoregressive Models: Generate theta_t per time, predict theta_t+1 using theta_1...t

Usage:
    # LSTM-based temporal prediction
    python infer_patient_temporal.py ./results/ --temporal-data temporal_patient_data.csv \\
        --method lstm --output predictions.csv

    # Simple regression (time to next visit)
    python infer_patient_temporal.py ./results/ --temporal-data temporal_patient_data.csv \\
        --method regression --predict-window 6 --output predictions.csv
    
    # Autoregressive prediction
    python infer_patient_temporal.py ./results/ --temporal-data temporal_patient_data.csv \\
        --method autoregressive --future-steps 3 --output predictions.csv

Data Format:
    CSV with columns: SUBJECT_ID, code, timestamp, modality
    - timestamp: YYYY-MM-DD for ICD/OPCS, categorical (0,1,2,3) for medications
    - modality: 'icd', 'med', or 'opcs'
"""

import argparse
import os
import sys
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from corpus import Corpus, read_data_file
from MixEHR_SAGE import MixEHR_SAGE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TemporalLSTMVAE(nn.Module):
    """
    LSTM-based Variational Autoencoder for temporal disease prediction.
    Learns q(eta_t | X_1...t-1) with temporal priors p(theta_t | theta_t-1).
    """
    def __init__(self, K, V, eta_hidden_size=200, eta_nlayers=3, eta_dropout=0.0, delta=0.01):
        super(TemporalLSTMVAE, self).__init__()
        self.K = K  # Number of topics
        self.V = V  # Vocabulary size
        self.eta_hidden_size = eta_hidden_size
        self.eta_nlayers = eta_nlayers
        self.eta_dropout = eta_dropout
        self.delta = delta  # Prior variance
        
        # LSTM for encoding temporal sequence
        self.q_eta_map = nn.Linear(self.V, self.eta_hidden_size)
        self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, 
                             self.eta_nlayers, dropout=self.eta_dropout, batch_first=True)
        
        # Output layers for mean and log-variance
        self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K, bias=True)
        self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K, bias=True)
        
        # Constraints on log-sigma
        self.max_logsigma_t = 5.0
        self.min_logsigma_t = -5.0
        
    def forward(self, bow_sequence, theta_prev=None):
        """
        Forward pass through temporal VAE.
        
        Args:
            bow_sequence: Tensor of shape (batch_size, seq_len, V) - bag-of-words at each timestep
            theta_prev: Tensor of shape (batch_size, K) - previous theta (optional)
        
        Returns:
            mu: Mean of q(eta_t | X_1...t-1)
            logsigma: Log-variance of q(eta_t | X_1...t-1)
            eta: Sampled eta_t
        """
        batch_size, seq_len, _ = bow_sequence.shape
        
        # Map BOW to hidden space
        h = self.q_eta_map(bow_sequence)  # (batch_size, seq_len, eta_hidden_size)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.q_eta(h)  # lstm_out: (batch_size, seq_len, eta_hidden_size)
        
        # Use last hidden state
        h_last = lstm_out[:, -1, :]  # (batch_size, eta_hidden_size)
        
        # Concatenate with previous theta if available
        if theta_prev is not None:
            h_combined = torch.cat([h_last, theta_prev], dim=1)
        else:
            h_combined = torch.cat([h_last, torch.zeros(batch_size, self.K, device=device)], dim=1)
        
        # Compute mean and log-variance
        mu = self.mu_q_eta(h_combined)
        logsigma = torch.clamp(self.logsigma_q_eta(h_combined), 
                               min=self.min_logsigma_t, max=self.max_logsigma_t)
        
        # Reparameterization trick
        std = torch.exp(0.5 * logsigma)
        eps = torch.randn_like(std)
        eta = mu + eps * std
        
        return mu, logsigma, eta
    
    def compute_kl_divergence(self, mu, logsigma, eta_prev=None):
        """
        Compute KL divergence KL(q(eta_t | X_1...t-1) || p(eta_t | eta_t-1)).
        
        Args:
            mu: Mean from variational distribution
            logsigma: Log-variance from variational distribution
            eta_prev: Previous eta (if None, use standard normal prior)
        
        Returns:
            kl: KL divergence
        """
        if eta_prev is None:
            # KL(q(eta_t) || N(0, I))
            kl = -0.5 * torch.sum(1 + logsigma - mu.pow(2) - logsigma.exp(), dim=1)
        else:
            # KL(q(eta_t) || N(eta_t-1, delta*I))
            var = logsigma.exp()
            kl = -0.5 * torch.sum(
                1 + logsigma - torch.log(torch.tensor(self.delta)) 
                - (mu - eta_prev).pow(2) / self.delta - var / self.delta,
                dim=1
            )
        return kl.mean()


class SimpleRegressionPredictor(nn.Module):
    """
    Simple regression model to predict time to next visit or binary classification 
    for visit within time window.
    """
    def __init__(self, K, hidden_dim=128, task='regression'):
        super(SimpleRegressionPredictor, self).__init__()
        self.K = K
        self.task = task  # 'regression' or 'classification'
        
        # Simple feedforward network
        self.fc1 = nn.Linear(K, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        if task == 'regression':
            self.output = nn.Linear(hidden_dim // 2, 1)  # Predict time gap
        else:
            self.output = nn.Linear(hidden_dim // 2, 1)  # Binary: visit within window
            
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, theta):
        """
        Forward pass.
        
        Args:
            theta: Tensor of shape (batch_size, K) - topic mixtures
        
        Returns:
            prediction: Time gap (regression) or probability (classification)
        """
        h = self.relu(self.fc1(theta))
        h = self.relu(self.fc2(h))
        
        if self.task == 'regression':
            return self.output(h)
        else:
            return self.sigmoid(self.output(h))


class AutoregressivePredictor(nn.Module):
    """
    Autoregressive model to predict theta_t+1 from theta_1...t.
    Inspired by TimelyGPT architecture.
    """
    def __init__(self, K, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.1):
        super(AutoregressivePredictor, self).__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_proj = nn.Linear(K, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, hidden_dim))  # Max 1000 time steps
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, K)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, theta_sequence, predict_steps=1):
        """
        Forward pass for autoregressive prediction.
        
        Args:
            theta_sequence: Tensor of shape (batch_size, seq_len, K)
            predict_steps: Number of future steps to predict
        
        Returns:
            predictions: Tensor of shape (batch_size, predict_steps, K)
        """
        batch_size, seq_len, _ = theta_sequence.shape
        
        # Project to hidden dimension
        h = self.input_proj(theta_sequence)
        
        # Add positional encoding
        h = h + self.pos_encoding[:, :seq_len, :]
        
        # Create causal mask
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)
        
        # Decode
        decoded = self.transformer_decoder(h, h, tgt_mask=tgt_mask)
        
        # Predict next steps autoregressively
        predictions = []
        current_seq = decoded
        
        for _ in range(predict_steps):
            # Get last hidden state
            last_hidden = current_seq[:, -1:, :]
            
            # Project to theta space
            next_theta = self.softmax(self.output_proj(last_hidden))
            predictions.append(next_theta)
            
            # Update sequence for next prediction
            next_hidden = self.input_proj(next_theta) + self.pos_encoding[:, seq_len:seq_len+1, :]
            current_seq = torch.cat([current_seq, next_hidden], dim=1)
            seq_len += 1
        
        return torch.cat(predictions, dim=1)


def load_vocab_mappings(mapping_path='./mapping/'):
    """Load vocabulary mappings for each modality."""
    vocab_mappings = {}
    for f in os.listdir(mapping_path):
        if f.endswith('_vocab_ids.pkl'):
            modality = f.replace('_vocab_ids.pkl', '')
            with open(os.path.join(mapping_path, f), 'rb') as handle:
                vocab_mappings[modality] = pickle.load(handle)
    return vocab_mappings


def parse_timestamp(timestamp_str, modality):
    """
    Parse timestamp based on modality.
    
    Args:
        timestamp_str: String timestamp
        modality: 'icd', 'med', or 'opcs'
    
    Returns:
        datetime object or categorical value
    """
    if modality == 'med':
        # Categorical: 0, 1, 2, 3 representing time ranges
        try:
            return int(timestamp_str)
        except:
            return None
    else:
        # ICD/OPCS: year-month-day
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%d')
        except:
            return None


def load_temporal_data(filepath, vocab_mappings, modalities):
    """
    Load temporal patient data with timestamps.
    
    Expected format: CSV with columns [SUBJECT_ID, code, timestamp, modality]
    
    Returns:
        patient_sequences: Dict mapping patient_id to list of (timestamp, bow_dict, modality) tuples
    """
    print(f"Loading temporal data from {filepath}...")
    df = pd.read_data_file(filepath) if hasattr(pd, 'read_data_file') else read_data_file(filepath)
    
    if not isinstance(df, pd.DataFrame):
        # Convert to DataFrame if needed
        df = pd.DataFrame(df)
    
    # Ensure required columns exist
    required_cols = ['SUBJECT_ID', 'code', 'timestamp', 'modality']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    patient_sequences = defaultdict(list)
    
    for _, row in df.iterrows():
        patient_id = str(row['SUBJECT_ID'])
        code = str(row['code']).strip()
        timestamp_str = str(row['timestamp'])
        modality = str(row['modality']).strip().lower()
        
        # Parse timestamp
        timestamp = parse_timestamp(timestamp_str, modality)
        if timestamp is None:
            continue
        
        # Map code to vocabulary index
        if modality in vocab_mappings and code in vocab_mappings[modality]:
            vocab_idx = vocab_mappings[modality][code]
            patient_sequences[patient_id].append((timestamp, vocab_idx, modality))
    
    # Sort sequences by timestamp for each patient
    for patient_id in patient_sequences:
        patient_sequences[patient_id].sort(key=lambda x: x[0])
    
    print(f"Loaded data for {len(patient_sequences)} patients")
    return patient_sequences


def create_time_windows(patient_sequences, window_size_months=6):
    """
    Aggregate patient data into time windows.
    
    Args:
        patient_sequences: Dict from load_temporal_data
        window_size_months: Size of each time window in months
    
    Returns:
        patient_windows: Dict mapping patient_id to list of (window_start, bow_dict) tuples
    """
    patient_windows = {}
    
    for patient_id, sequence in patient_sequences.items():
        if not sequence:
            continue
        
        # Find min/max timestamps (handle both datetime and categorical)
        datetime_entries = [(t, v, m) for t, v, m in sequence if isinstance(t, datetime)]
        
        if not datetime_entries:
            # Only categorical timestamps (medications)
            patient_windows[patient_id] = []
            continue
        
        min_time = min(t for t, _, _ in datetime_entries)
        max_time = max(t for t, _, _ in datetime_entries)
        
        # Create windows
        windows = []
        current_time = min_time
        window_delta = timedelta(days=window_size_months * 30)
        
        while current_time <= max_time:
            window_end = current_time + window_delta
            
            # Collect all codes in this window
            window_codes = defaultdict(int)
            for timestamp, vocab_idx, modality in sequence:
                if isinstance(timestamp, datetime):
                    if current_time <= timestamp < window_end:
                        modality_key = f"{modality}_{vocab_idx}"
                        window_codes[modality_key] += 1
                else:
                    # For categorical medication timestamps, include if they overlap
                    modality_key = f"{modality}_{vocab_idx}"
                    window_codes[modality_key] += 1
            
            if window_codes:
                windows.append((current_time, dict(window_codes)))
            
            current_time = window_end
        
        patient_windows[patient_id] = windows
    
    return patient_windows


def compute_theta_sequence(model, patient_windows, vocab_mappings, modalities):
    """
    Compute theta for each time window.
    
    Args:
        model: Trained MixEHR_SAGE model
        patient_windows: Output from create_time_windows
        vocab_mappings: Vocabulary mappings
        modalities: List of modality names
    
    Returns:
        theta_sequences: Dict mapping patient_id to list of (timestamp, theta) tuples
    """
    theta_sequences = {}
    
    for patient_id, windows in patient_windows.items():
        if not windows:
            continue
        
        patient_thetas = []
        
        for timestamp, bow_dict in windows:
            # Convert aggregated codes to modality-specific BOWs
            bow_by_modality = {m: {} for m in modalities}
            
            for key, count in bow_dict.items():
                parts = key.split('_', 1)
                if len(parts) == 2:
                    modality, vocab_idx = parts[0], int(parts[1])
                    if modality in bow_by_modality:
                        bow_by_modality[modality][vocab_idx] = count
            
            # Infer theta for this time window
            try:
                theta = model.infer_theta_by_modality(bow_by_modality, num_iterations=10)
                if theta is not None:
                    patient_thetas.append((timestamp, theta))
            except Exception as e:
                print(f"Error inferring theta for patient {patient_id}: {e}")
                continue
        
        if patient_thetas:
            theta_sequences[patient_id] = patient_thetas
    
    return theta_sequences


def train_lstm_temporal_model(theta_sequences, K, V, epochs=50, lr=0.001):
    """
    Train LSTM-based temporal VAE model.
    
    Args:
        theta_sequences: Output from compute_theta_sequence
        K: Number of topics
        V: Vocabulary size
        epochs: Number of training epochs
        lr: Learning rate
    
    Returns:
        model: Trained TemporalLSTMVAE model
    """
    print("Training LSTM temporal model...")
    model = TemporalLSTMVAE(K, V).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.2e-6)
    
    # Prepare training data
    training_sequences = []
    for patient_id, sequence in theta_sequences.items():
        if len(sequence) >= 2:  # Need at least 2 time points
            training_sequences.append(sequence)
    
    if not training_sequences:
        print("No valid training sequences found!")
        return model
    
    print(f"Training on {len(training_sequences)} patient sequences")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for sequence in training_sequences:
            # Convert sequence to tensors
            # This is a simplified version - in practice, you'd batch this properly
            seq_len = len(sequence)
            
            # Create BOW sequence (placeholder - would need actual BOW construction)
            bow_sequence = torch.zeros(1, seq_len, V, device=device)
            
            # Extract theta values
            theta_values = torch.stack([theta for _, theta in sequence]).unsqueeze(0)
            
            # Forward pass
            optimizer.zero_grad()
            mu, logsigma, eta = model(bow_sequence, theta_prev=theta_values[:, :-1, :])
            
            # Compute KL divergence
            kl_loss = model.compute_kl_divergence(mu, logsigma, eta_prev=theta_values[:, :-1, :])
            
            # Reconstruction loss (simplified)
            recon_loss = torch.nn.functional.mse_loss(eta, theta_values[:, -1, :])
            
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(training_sequences)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def train_simple_regression_model(theta_sequences, task='regression', window_months=6, epochs=50):
    """
    Train simple regression model to predict time to next visit.
    
    Args:
        theta_sequences: Output from compute_theta_sequence
        task: 'regression' or 'classification'
        window_months: Time window for classification task (months)
        epochs: Number of training epochs
    
    Returns:
        model: Trained SimpleRegressionPredictor
    """
    print(f"Training simple {task} model...")
    
    # Prepare training data
    X_train = []
    y_train = []
    
    for patient_id, sequence in theta_sequences.items():
        for i in range(len(sequence) - 1):
            timestamp_current, theta_current = sequence[i]
            timestamp_next, _ = sequence[i + 1]
            
            # Calculate time gap
            if isinstance(timestamp_current, datetime) and isinstance(timestamp_next, datetime):
                time_gap_days = (timestamp_next - timestamp_current).days
                
                X_train.append(theta_current.cpu().numpy())
                
                if task == 'regression':
                    y_train.append([time_gap_days / 365.0])  # Normalize to years
                else:
                    # Binary: within window or not
                    within_window = 1 if time_gap_days <= (window_months * 30) else 0
                    y_train.append([within_window])
    
    if not X_train:
        print("No valid training data!")
        return None
    
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32, device=device)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32, device=device)
    
    print(f"Training on {len(X_train)} samples")
    
    K = X_train.shape[1]
    model = SimpleRegressionPredictor(K, task=task).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    if task == 'regression':
        criterion = nn.MSELoss()
    else:
        criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model


def train_autoregressive_model(theta_sequences, K, epochs=50):
    """
    Train autoregressive model to predict theta_t+1 from theta_1...t.
    
    Args:
        theta_sequences: Output from compute_theta_sequence
        K: Number of topics
        epochs: Number of training epochs
    
    Returns:
        model: Trained AutoregressivePredictor
    """
    print("Training autoregressive model...")
    
    model = AutoregressivePredictor(K).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.MSELoss()
    
    # Prepare training sequences
    training_data = []
    for patient_id, sequence in theta_sequences.items():
        if len(sequence) >= 3:  # Need at least 3 points
            theta_seq = torch.stack([theta for _, theta in sequence])
            training_data.append(theta_seq)
    
    if not training_data:
        print("No valid training sequences!")
        return model
    
    print(f"Training on {len(training_data)} patient sequences")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for theta_seq in training_data:
            # Use all but last as input, predict last
            input_seq = theta_seq[:-1].unsqueeze(0)
            target = theta_seq[-1:].unsqueeze(0)
            
            optimizer.zero_grad()
            prediction = model(input_seq, predict_steps=1)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    return model


def predict_healthy_threshold(theta, threshold=0.1):
    """
    Predict if patient is healthy based on disease probabilities.
    
    Args:
        theta: Topic mixture (disease probabilities)
        threshold: Threshold value (default 0.1, can be optimized via F1 score)
    
    Returns:
        is_healthy: Boolean indicating if patient is deemed healthy
    """
    # If all disease probabilities are below threshold, patient is healthy
    return torch.all(theta < threshold).item()


def main():
    parser = argparse.ArgumentParser(description='Temporal disease prediction for MixEHR-SAGE')
    parser.add_argument('model_dir', type=str, help='Path to trained model directory')
    parser.add_argument('--temporal-data', type=str, required=True,
                        help='Path to temporal patient data CSV')
    parser.add_argument('--method', type=str, choices=['lstm', 'regression', 'autoregressive'],
                        default='lstm', help='Prediction method to use')
    parser.add_argument('--output', '-o', type=str, default='temporal_predictions.csv',
                        help='Output file for predictions')
    parser.add_argument('--window-months', type=int, default=6,
                        help='Time window in months for aggregation')
    parser.add_argument('--predict-window', type=int, default=6,
                        help='Prediction window in months for classification')
    parser.add_argument('--future-steps', type=int, default=3,
                        help='Number of future steps to predict (autoregressive)')
    parser.add_argument('--healthy-threshold', type=float, default=0.1,
                        help='Threshold for healthy prediction')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--task', type=str, choices=['regression', 'classification'],
                        default='regression', help='Task type for simple regression method')
    parser.add_argument('--iterations', type=int, default=10,
                        help='Number of inference iterations for computing theta')
    
    args = parser.parse_args()
    
    print("="*80)
    print("MixEHR-SAGE Temporal Disease Prediction")
    print("="*80)
    print(f"Method: {args.method}")
    print(f"Model directory: {args.model_dir}")
    print(f"Temporal data: {args.temporal_data}")
    print(f"Output: {args.output}")
    print("="*80)
    
    # Load vocabulary mappings
    print("\nLoading vocabulary mappings...")
    vocab_mappings = load_vocab_mappings('./mapping/')
    modality_list = list(vocab_mappings.keys())
    print(f"Loaded vocabulary for modalities: {modality_list}")
    
    # Create a minimal corpus object for model loading
    print("\nCreating corpus structure...")
    # We need V (vocabulary sizes) and modalities
    V_sizes = [len(vocab_mappings[mod]) for mod in modality_list]
    
    # Create dummy corpus (just for structure, not actual training)
    class DummyCorpus:
        def __init__(self, V_sizes, modalities):
            self.V = V_sizes
            self.modalities = modalities
            self.D = 0  # No documents needed for inference
            self.C = 0
    
    corpus = DummyCorpus(V_sizes, modality_list)
    
    # Load or create seed matrix (K x V for guided modality)
    # This is simplified - you may need actual seed matrix from training
    print("\nLoading model configuration...")
    K = 5  # Default number of topics - will be determined from loaded model
    
    # Try to infer K from model files
    import glob
    exp_m_files = glob.glob(os.path.join(args.model_dir, "toy_exp_m_*.pt"))
    if exp_m_files:
        # Load one file to get K
        sample = torch.load(exp_m_files[0], map_location=device, weights_only=False)
        K = sample.shape[0]
        print(f"Detected K={K} topics from model files")
    
    # Create dummy seed matrix (will be overwritten by loaded model)
    seeds_topic_matrix = np.zeros((V_sizes[0], K))
    
    # Load trained model
    print(f"\nLoading trained MixEHR-SAGE model from {args.model_dir}...")
    try:
        model = MixEHR_SAGE.load_trained_model(
            args.model_dir, 
            corpus, 
            seeds_topic_matrix, 
            modality_list,
            guided_modality=0,
            guide_prior_path='./guide_prior/'
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model directory contains trained .pt files")
        return
    
    # Load temporal data
    print("\nLoading temporal patient data...")
    patient_sequences = load_temporal_data(args.temporal_data, vocab_mappings, modality_list)
    print(f"Loaded data for {len(patient_sequences)} patients")
    
    # Create time windows
    print(f"\nCreating time windows (window size: {args.window_months} months)...")
    patient_windows = create_time_windows(patient_sequences, window_size_months=args.window_months)
    print(f"Created time windows for {len(patient_windows)} patients")
    
    # Compute theta sequences
    print(f"\nComputing theta sequences (iterations: {args.iterations})...")
    theta_sequences = compute_theta_sequence(model, patient_windows, modality_list, 
                                            num_iterations=args.iterations)
    print(f"Computed theta sequences for {len(theta_sequences)} patients")
    
    # Run selected prediction method
    print("\n" + "="*80)
    print(f"Running {args.method.upper()} temporal prediction")
    print("="*80)
    
    if args.method == 'lstm':
        print("\n1. Training LSTM Temporal VAE...")
        temporal_model = train_lstm_temporal(theta_sequences, K, epochs=args.epochs)
        
        print("\n2. Making predictions...")
        predictions = {}
        for patient_id, sequence in theta_sequences.items():
            if len(sequence) >= 2:
                # Use sequence to predict next theta
                theta_seq = torch.stack([theta for _, theta in sequence])
                input_seq = theta_seq.unsqueeze(0)
                
                with torch.no_grad():
                    pred_theta = temporal_model(input_seq, predict_steps=args.future_steps)
                
                predictions[patient_id] = {
                    'method': 'lstm',
                    'predictions': pred_theta.cpu().numpy(),
                    'is_healthy': [predict_healthy_threshold(pred_theta[i], args.healthy_threshold) 
                                  for i in range(pred_theta.shape[0])]
                }
        
        print(f"Generated predictions for {len(predictions)} patients")
    
    elif args.method == 'regression':
        print(f"\n1. Training {args.task} model...")
        regression_model = train_regression_model(theta_sequences, K, task=args.task, 
                                                  predict_window_months=args.predict_window,
                                                  epochs=args.epochs)
        
        print("\n2. Making predictions...")
        predictions = {}
        for patient_id, sequence in theta_sequences.items():
            if len(sequence) >= 2:
                # Use last theta to predict
                last_theta = sequence[-1][1].unsqueeze(0)
                
                with torch.no_grad():
                    if args.task == 'classification':
                        pred = torch.sigmoid(regression_model(last_theta))
                        predictions[patient_id] = {
                            'method': 'regression_classification',
                            'visit_probability': pred.item(),
                            'visit_predicted': pred.item() > 0.5
                        }
                    else:  # regression
                        pred = regression_model(last_theta)
                        predictions[patient_id] = {
                            'method': 'regression_time',
                            'time_to_next_visit_years': pred.item()
                        }
        
        print(f"Generated predictions for {len(predictions)} patients")
    
    elif args.method == 'autoregressive':
        print("\n1. Training Autoregressive Transformer...")
        ar_model = train_autoregressive_model(theta_sequences, K, epochs=args.epochs)
        
        print("\n2. Making predictions...")
        predictions = {}
        for patient_id, sequence in theta_sequences.items():
            if len(sequence) >= 3:
                theta_seq = torch.stack([theta for _, theta in sequence])
                input_seq = theta_seq.unsqueeze(0)
                
                with torch.no_grad():
                    pred_theta = ar_model(input_seq, predict_steps=args.future_steps)
                
                predictions[patient_id] = {
                    'method': 'autoregressive',
                    'predictions': pred_theta.cpu().numpy(),
                    'is_healthy': [predict_healthy_threshold(pred_theta[i], args.healthy_threshold) 
                                  for i in range(pred_theta.shape[0])]
                }
        
        print(f"Generated predictions for {len(predictions)} patients")
    
    # Save predictions
    print(f"\nSaving predictions to {args.output}...")
    save_predictions(predictions, args.output)
    
    print("\n" + "="*80)
    print("Temporal prediction completed successfully!")
    print("="*80)


def save_predictions(predictions, output_path):
    """Save predictions to CSV file."""
    rows = []
    for patient_id, pred_data in predictions.items():
        if pred_data['method'] in ['lstm', 'autoregressive']:
            for step, theta_pred in enumerate(pred_data['predictions']):
                row = {
                    'patient_id': patient_id,
                    'method': pred_data['method'],
                    'future_step': step + 1,
                    'predicted_theta': str(theta_pred.tolist() if hasattr(theta_pred, 'tolist') else theta_pred),
                    'is_healthy': pred_data['is_healthy'][step]
                }
                rows.append(row)
        else:
            row = {'patient_id': patient_id, 'method': pred_data['method']}
            row.update(pred_data)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(rows)} prediction rows")


if __name__ == '__main__':
    main()
