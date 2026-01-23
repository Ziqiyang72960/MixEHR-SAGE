"""
Temporal Progression Models for MixEHR-SAGE

This module implements temporal disease progression models for analyzing
longitudinal patient trajectories. Includes:

1. First-order Markov Model
   - Transition matrix estimation P(θ_t | θ_{t-1})
   - Next-state and disease-risk estimation

2. LSTM-based Temporal Model
   - Neural network for temporal dynamics
   - Variational inference for dynamic topic prior η
"""

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
from collections import defaultdict
import pickle
import logging

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mini_val = 1e-6


class MarkovTransitionModel:
    """
    First-order Markov Model for disease progression.
    
    Models state transitions as P(state_t | state_{t-1}) where states
    are discretized topic clusters or dominant topics.
    
    Supports:
    - Soft assignment (uses full theta distribution)
    - Hard assignment (uses dominant topic)
    - Custom state clustering
    """
    
    def __init__(self, num_topics: int, num_states: Optional[int] = None,
                 discretization: str = 'dominant'):
        """
        Initialize Markov Transition Model.
        
        Args:
            num_topics: Number of topics (K)
            num_states: Number of discrete states (default: num_topics)
            discretization: State assignment method
                - 'dominant': Assign to dominant topic
                - 'soft': Use soft counts from theta
                - 'threshold': Assign to topics above threshold
        """
        self.K = num_topics
        self.num_states = num_states or num_topics
        self.discretization = discretization
        
        # Transition matrix: P(state_t | state_{t-1})
        # Shape: (num_states, num_states)
        self.transition_matrix = None
        self.initial_distribution = None  # P(state_0)
        
        # Statistics for estimation
        self.transition_counts = None
        self.state_counts = None
        
    def fit(self, theta_sequences: Dict[str, torch.Tensor],
            smoothing: float = 1.0) -> 'MarkovTransitionModel':
        """
        Estimate transition matrix from theta sequences.
        
        Args:
            theta_sequences: Dict mapping patient_id to theta sequence (T x K)
            smoothing: Laplace smoothing parameter (default: 1.0)
            
        Returns:
            self
        """
        # Initialize counts with smoothing
        self.transition_counts = np.zeros((self.num_states, self.num_states)) + smoothing
        self.state_counts = np.zeros(self.num_states) + smoothing * self.num_states
        initial_counts = np.zeros(self.num_states) + smoothing
        
        for patient_id, theta_seq in theta_sequences.items():
            # Convert to numpy
            if torch.is_tensor(theta_seq):
                theta_seq = theta_seq.cpu().numpy()
            
            T = theta_seq.shape[0]
            if T < 1:
                continue
            
            # Get state assignments for each time step
            states = self._discretize_theta_sequence(theta_seq)
            
            # Count initial state
            initial_counts[states[0]] += 1
            
            # Count transitions
            for t in range(T - 1):
                s_prev = states[t]
                s_curr = states[t + 1]
                
                if self.discretization == 'soft':
                    # Soft counts using outer product
                    self.transition_counts += np.outer(theta_seq[t], theta_seq[t + 1])
                    self.state_counts += theta_seq[t]
                else:
                    # Hard counts
                    self.transition_counts[s_prev, s_curr] += 1
                    self.state_counts[s_prev] += 1
        
        # Normalize to get probabilities
        self.transition_matrix = self.transition_counts / self.state_counts[:, np.newaxis]
        self.initial_distribution = initial_counts / initial_counts.sum()
        
        return self
    
    def _discretize_theta_sequence(self, theta_seq: np.ndarray) -> np.ndarray:
        """
        Convert continuous theta sequence to discrete states.
        
        Args:
            theta_seq: (T x K) array of topic mixtures
            
        Returns:
            (T,) array of state indices
        """
        T = theta_seq.shape[0]
        states = np.zeros(T, dtype=int)
        
        if self.discretization == 'dominant':
            # Assign to dominant (highest probability) topic
            states = theta_seq.argmax(axis=1)
            
        elif self.discretization == 'threshold':
            # This would require clustering logic
            # For now, use dominant
            states = theta_seq.argmax(axis=1)
        
        return states
    
    def predict_next_state(self, current_theta: Union[np.ndarray, torch.Tensor],
                          return_distribution: bool = True) -> Union[int, np.ndarray]:
        """
        Predict next state given current theta.
        
        Args:
            current_theta: Current topic mixture (K,)
            return_distribution: If True, return full distribution; else return most likely state
            
        Returns:
            Next state distribution (K,) or most likely state (int)
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if torch.is_tensor(current_theta):
            current_theta = current_theta.cpu().numpy()
        
        # Get current state
        if self.discretization == 'soft':
            # Weighted combination of transition rows
            next_dist = current_theta @ self.transition_matrix
        else:
            current_state = current_theta.argmax()
            next_dist = self.transition_matrix[current_state]
        
        if return_distribution:
            return next_dist
        return next_dist.argmax()
    
    def predict_disease_risk(self, current_theta: Union[np.ndarray, torch.Tensor],
                           horizon: int = 1,
                           target_topics: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Estimate disease risk over a future horizon.
        
        Args:
            current_theta: Current topic mixture (K,)
            horizon: Number of time steps to predict ahead
            target_topics: List of topic indices representing target diseases
                          If None, returns risk for all topics
            
        Returns:
            Dict with risk estimates for each topic
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if torch.is_tensor(current_theta):
            current_theta = current_theta.cpu().numpy()
        
        # Compute n-step transition
        transition_n = np.linalg.matrix_power(self.transition_matrix, horizon)
        
        if self.discretization == 'soft':
            future_dist = current_theta @ transition_n
        else:
            current_state = current_theta.argmax()
            future_dist = transition_n[current_state]
        
        # Compute risks
        if target_topics is None:
            target_topics = list(range(self.K))
        
        risks = {}
        for topic_idx in target_topics:
            risks[f'topic_{topic_idx}'] = float(future_dist[topic_idx])
        
        # Add summary statistics
        risks['max_risk_topic'] = int(future_dist.argmax())
        risks['max_risk'] = float(future_dist.max())
        risks['entropy'] = float(-np.sum(future_dist * np.log(future_dist + mini_val)))
        
        return risks
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Compute stationary distribution of the Markov chain.
        
        Returns:
            Stationary distribution (K,)
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Find eigenvector with eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find index of eigenvalue closest to 1
        idx = np.argmin(np.abs(eigenvalues - 1))
        stationary = np.real(eigenvectors[:, idx])
        
        # Normalize
        stationary = stationary / stationary.sum()
        
        return stationary
    
    def save(self, path: str):
        """Save model to file."""
        data = {
            'K': self.K,
            'num_states': self.num_states,
            'discretization': self.discretization,
            'transition_matrix': self.transition_matrix,
            'initial_distribution': self.initial_distribution,
            'transition_counts': self.transition_counts,
            'state_counts': self.state_counts
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved Markov model to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'MarkovTransitionModel':
        """Load model from file."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls(
            num_topics=data['K'],
            num_states=data['num_states'],
            discretization=data['discretization']
        )
        model.transition_matrix = data['transition_matrix']
        model.initial_distribution = data['initial_distribution']
        model.transition_counts = data['transition_counts']
        model.state_counts = data['state_counts']
        
        return model


class TemporalLSTMModel(nn.Module):
    """
    LSTM-based Temporal Model for MixEHR-SAGE.
    
    Implements variational inference for dynamic topic prior η using RNN.
    This is the re-implementation of the commented temporal component from
    the original MixEHR_SAGE class.
    
    The model learns:
    - Temporal dynamics of topic mixtures via LSTM
    - Variational parameters (μ, σ) for η at each time step
    - Transition patterns between topic states
    
    Architecture:
        Input: Aggregated BOW representation at time t
        LSTM: Hidden state evolution over time
        Output: μ_t and log(σ_t) for variational distribution of η_t
    """
    
    def __init__(self, vocab_size: int, num_topics: int,
                 hidden_size: int = 200, num_layers: int = 3,
                 dropout: float = 0.0, delta: float = 0.01):
        """
        Initialize LSTM temporal model.
        
        Args:
            vocab_size: Vocabulary size (V)
            num_topics: Number of topics (K)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            delta: Prior variance for η
        """
        super(TemporalLSTMModel, self).__init__()
        
        self.V = vocab_size
        self.K = num_topics
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.delta = delta
        
        # Bounds for numerical stability
        self.max_logsigma = 5.0
        self.min_logsigma = -5.0
        
        # Input projection: BOW -> hidden
        self.input_projection = nn.Linear(vocab_size, hidden_size)
        
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Output layers for variational parameters
        # Combines LSTM output with previous α (topic prior)
        self.mu_layer = nn.Linear(hidden_size + num_topics, num_topics)
        self.logsigma_layer = nn.Linear(hidden_size + num_topics, num_topics)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with small values."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, bow_sequence: torch.Tensor, 
                alpha_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal model.
        
        Args:
            bow_sequence: (batch, T, V) bag-of-words at each time step
            alpha_prev: (batch, K) previous topic prior (default: uniform)
            
        Returns:
            mu: (batch, T, K) mean of variational distribution for η
            logsigma: (batch, T, K) log std of variational distribution
        """
        batch_size, T, V = bow_sequence.shape
        
        # Project input
        x = self.input_projection(bow_sequence)  # (batch, T, hidden)
        
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, T, hidden)
        
        # Initialize alpha_prev if not provided
        if alpha_prev is None:
            alpha_prev = torch.ones(batch_size, self.K, device=bow_sequence.device) / self.K
        
        # Compute variational parameters at each time step
        mu_list = []
        logsigma_list = []
        
        alpha_t = alpha_prev
        for t in range(T):
            # Concatenate LSTM output with previous alpha
            combined = torch.cat([lstm_out[:, t, :], alpha_t], dim=1)  # (batch, hidden + K)
            
            # Compute mu and logsigma
            mu_t = self.mu_layer(combined)  # (batch, K)
            logsigma_t = self.logsigma_layer(combined)  # (batch, K)
            
            # Clamp logsigma for stability
            logsigma_t = torch.clamp(logsigma_t, self.min_logsigma, self.max_logsigma)
            
            mu_list.append(mu_t)
            logsigma_list.append(logsigma_t)
            
            # Update alpha for next step (use softplus of eta)
            alpha_t = F.softplus(mu_t)
        
        mu = torch.stack(mu_list, dim=1)  # (batch, T, K)
        logsigma = torch.stack(logsigma_list, dim=1)  # (batch, T, K)
        
        return mu, logsigma
    
    def sample_eta(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """
        Sample η from variational distribution using reparameterization trick.
        
        Args:
            mu: (batch, T, K) mean
            logsigma: (batch, T, K) log std
            
        Returns:
            eta: (batch, T, K) sampled values
        """
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def get_alpha(self, eta: torch.Tensor) -> torch.Tensor:
        """
        Convert η to α (topic prior) using softplus.
        
        Args:
            eta: (batch, T, K) η values
            
        Returns:
            alpha: (batch, T, K) topic prior
        """
        return F.softplus(eta)
    
    def kl_divergence(self, mu: torch.Tensor, logsigma: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between variational posterior q and prior p.
        
        Mathematical formulation:
        - q(η_t | η_{t-1}, h_t) = N(μ_t, σ_t²) [variational posterior]
        - p(η_t | η_{t-1}) = N(η_{t-1}, δ²I) [prior with variance δ²]
        
        For Gaussian distributions, KL divergence is:
        KL(q || p) = 0.5 * [tr(Σ_q/Σ_p) + (μ_p - μ_q)ᵀ Σ_p⁻¹ (μ_p - μ_q) - K + ln(|Σ_p|/|Σ_q|)]
        
        With diagonal covariances, this simplifies to:
        KL = 0.5 * [Σ σ_q²/δ² + Σ (η_{t-1} - μ)²/δ² - K + K·ln(δ²) - Σ ln(σ_q²)]
        
        Args:
            mu: (batch, T, K) variational mean μ at each timestep
            logsigma: (batch, T, K) log of variational std σ at each timestep
            
        Returns:
            kl: (batch,) total KL divergence summed over all timesteps for each sequence
        """
        batch_size, T, K = mu.shape
        
        var = torch.exp(2 * logsigma)  # σ² = exp(2 * log(σ))
        
        # For t=0, prior is N(0, δ²I)
        # For t>0, prior is N(η_{t-1}, δ²I)
        kl_total = torch.zeros(batch_size, device=mu.device)
        
        eta_prev = torch.zeros(batch_size, K, device=mu.device)
        
        for t in range(T):
            # KL(N(μ, σ²) || N(η_prev, δ²))
            # = 0.5 * [Σ(σ²/δ²) + Σ((η_prev - μ)²/δ²) - K + K·ln(δ²) - Σ ln(σ²)]
            
            kl_t = 0.5 * (
                torch.sum(var[:, t, :], dim=1) / (self.delta ** 2) +
                torch.sum((eta_prev - mu[:, t, :]) ** 2, dim=1) / (self.delta ** 2) -
                K +
                K * np.log(self.delta ** 2) -
                torch.sum(2 * logsigma[:, t, :], dim=1)
            )
            
            kl_total += kl_t
            
            # Update eta_prev for next step (use sampled or mean)
            eta_prev = mu[:, t, :]
        
        return kl_total


class TemporalMixEHR(nn.Module):
    """
    Temporal MixEHR-SAGE model combining LSTM dynamics with topic modeling.
    
    This integrates the LSTM temporal model with MixEHR-SAGE for
    time-aware topic modeling of longitudinal EHR data.
    """
    
    def __init__(self, base_model, lstm_model: TemporalLSTMModel):
        """
        Initialize Temporal MixEHR.
        
        Args:
            base_model: Trained MixEHR-SAGE model
            lstm_model: LSTM temporal model
        """
        super(TemporalMixEHR, self).__init__()
        
        self.base_model = base_model
        self.lstm_model = lstm_model
        
        # Training parameters
        self.lr = 0.0001
        self.weight_decay = 1.2e-6
        self.clip_grad = 5.0
        
        # Only optimize LSTM parameters
        self.optimizer = optim.Adam(
            self.lstm_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
    
    def train_temporal(self, temporal_corpus, vocab_mappings: Dict,
                       modality_list: List[str], num_epochs: int = 10,
                       batch_size: int = 32) -> List[float]:
        """
        Train the temporal LSTM component.
        
        Args:
            temporal_corpus: TemporalCorpus object
            vocab_mappings: Vocabulary mappings
            modality_list: List of modality names
            num_epochs: Number of training epochs
            batch_size: Batch size
            
        Returns:
            List of loss values per epoch
        """
        self.train()
        loss_history = []
        
        # Prepare data
        patient_ids = list(temporal_corpus.patients.keys())
        V = sum(len(v) for v in vocab_mappings.values())  # Total vocab size
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            # Shuffle patients
            np.random.shuffle(patient_ids)
            
            for i in range(0, len(patient_ids), batch_size):
                batch_ids = patient_ids[i:i + batch_size]
                
                # Build batch data
                batch_bow_sequences = []
                max_T = 0
                
                for pid in batch_ids:
                    patient = temporal_corpus.patients[pid]
                    T = patient.num_time_steps
                    max_T = max(max_T, T)
                    
                    # Get BOW for each time step
                    bow_seq = []
                    for bucket in patient.buckets:
                        bow = bucket.get_bow_by_modality(vocab_mappings, modality_list)
                        # Flatten to single vector
                        bow_flat = np.zeros(V)
                        offset = 0
                        for m, modality in enumerate(modality_list):
                            if modality in vocab_mappings:
                                for word_id, freq in bow[m].items():
                                    bow_flat[offset + word_id] = freq
                                offset += len(vocab_mappings[modality])
                        bow_seq.append(bow_flat)
                    
                    batch_bow_sequences.append(bow_seq)
                
                # Pad sequences to max length
                batch_tensor = torch.zeros(len(batch_ids), max_T, V, device=device)
                for b, bow_seq in enumerate(batch_bow_sequences):
                    for t, bow in enumerate(bow_seq):
                        batch_tensor[b, t, :] = torch.tensor(bow, dtype=torch.float, device=device)
                
                # Forward pass
                self.optimizer.zero_grad()
                mu, logsigma = self.lstm_model(batch_tensor)
                
                # Compute KL loss
                kl_loss = self.lstm_model.kl_divergence(mu, logsigma).mean()
                
                # Sample eta and compute alpha
                eta = self.lstm_model.sample_eta(mu, logsigma)
                alpha = self.lstm_model.get_alpha(eta)
                
                # Total loss (can add reconstruction term if needed)
                loss = kl_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.lstm_model.parameters(), self.clip_grad)
                
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / max(num_batches, 1)
            loss_history.append(avg_loss)
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
        
        return loss_history
    
    def infer_temporal_theta(self, temporal_corpus, vocab_mappings: Dict,
                            modality_list: List[str], num_iterations: int = 10) -> Dict[str, torch.Tensor]:
        """
        Infer temporal theta sequences using trained LSTM.
        
        Args:
            temporal_corpus: TemporalCorpus object
            vocab_mappings: Vocabulary mappings
            modality_list: List of modality names
            num_iterations: Base model inference iterations
            
        Returns:
            Dict mapping patient_id to theta sequence (T x K)
        """
        self.eval()
        theta_sequences = {}
        
        V = sum(len(v) for v in vocab_mappings.values())
        
        with torch.no_grad():
            for patient_id, patient in temporal_corpus.patients.items():
                T = patient.num_time_steps
                
                # Build BOW sequence
                bow_seq = torch.zeros(1, T, V, device=device)
                for t, bucket in enumerate(patient.buckets):
                    bow = bucket.get_bow_by_modality(vocab_mappings, modality_list)
                    offset = 0
                    for m, modality in enumerate(modality_list):
                        if modality in vocab_mappings:
                            for word_id, freq in bow[m].items():
                                bow_seq[0, t, offset + word_id] = freq
                            offset += len(vocab_mappings[modality])
                
                # Get temporal alpha from LSTM
                mu, _ = self.lstm_model(bow_seq)
                alpha = self.lstm_model.get_alpha(mu)  # (1, T, K)
                
                # Infer theta using temporal alpha as prior
                theta_seq = torch.zeros(T, self.base_model.K, device=device)
                for t, bucket in enumerate(patient.buckets):
                    bow = patient.get_cumulative_bow(
                        bucket.time_index, vocab_mappings, modality_list
                    )
                    
                    # Temporarily modify base model's eta
                    original_eta = self.base_model.eta
                    self.base_model.eta = alpha[0, t, :]
                    
                    has_records = any(len(bow_m) > 0 for bow_m in bow)
                    if has_records:
                        theta_t = self.base_model.infer_theta_fast(bow, num_iterations=num_iterations)
                    else:
                        theta_t = alpha[0, t, :] / alpha[0, t, :].sum()
                    
                    # Restore original eta
                    self.base_model.eta = original_eta
                    
                    theta_seq[t] = theta_t
                
                theta_sequences[patient_id] = theta_seq
        
        return theta_sequences
    
    def save(self, path: str):
        """Save temporal model."""
        torch.save({
            'lstm_state_dict': self.lstm_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Saved temporal model to {path}")
    
    def load(self, path: str):
        """Load temporal model."""
        checkpoint = torch.load(path, map_location=device)
        self.lstm_model.load_state_dict(checkpoint['lstm_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Loaded temporal model from {path}")


def analyze_disease_progression(theta_sequences: Dict[str, torch.Tensor],
                               topic_names: Optional[Dict[int, str]] = None,
                               target_topics: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Analyze disease progression patterns from theta sequences.
    
    Args:
        theta_sequences: Dict mapping patient_id to theta sequence (T x K)
        topic_names: Optional dict mapping topic index to name
        target_topics: Optional list of topic indices to focus on
        
    Returns:
        DataFrame with progression statistics
    """
    results = []
    
    for patient_id, theta_seq in theta_sequences.items():
        if torch.is_tensor(theta_seq):
            theta_seq = theta_seq.cpu().numpy()
        
        T, K = theta_seq.shape
        
        if target_topics is None:
            target_topics_iter = range(K)
        else:
            target_topics_iter = target_topics
        
        for topic_idx in target_topics_iter:
            topic_name = topic_names.get(topic_idx, f'topic_{topic_idx}') if topic_names else f'topic_{topic_idx}'
            
            topic_probs = theta_seq[:, topic_idx]
            
            # Compute statistics
            result = {
                'patient_id': patient_id,
                'topic_idx': topic_idx,
                'topic_name': topic_name,
                'num_time_steps': T,
                'initial_prob': topic_probs[0],
                'final_prob': topic_probs[-1],
                'max_prob': topic_probs.max(),
                'min_prob': topic_probs.min(),
                'mean_prob': topic_probs.mean(),
                'std_prob': topic_probs.std(),
                'trend': 'increasing' if topic_probs[-1] > topic_probs[0] + 0.05 else 
                        ('decreasing' if topic_probs[-1] < topic_probs[0] - 0.05 else 'stable'),
                'prob_change': topic_probs[-1] - topic_probs[0]
            }
            
            results.append(result)
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    # Demo usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample theta sequences
    np.random.seed(42)
    K = 10  # Number of topics
    
    theta_sequences = {}
    for i in range(20):
        T = np.random.randint(3, 8)  # Random sequence length
        # Generate random but somewhat smooth theta sequences
        theta_seq = np.random.dirichlet(np.ones(K), size=T)
        theta_sequences[f'patient_{i}'] = torch.tensor(theta_seq, dtype=torch.double)
    
    # Fit Markov model
    print("Fitting Markov transition model...")
    markov = MarkovTransitionModel(num_topics=K, discretization='soft')
    markov.fit(theta_sequences)
    
    print(f"\nTransition matrix shape: {markov.transition_matrix.shape}")
    print(f"Initial distribution: {markov.initial_distribution[:5]}...")
    
    # Predict next state
    current_theta = theta_sequences['patient_0'][0].numpy()
    next_dist = markov.predict_next_state(current_theta)
    print(f"\nCurrent theta (first 5): {current_theta[:5]}")
    print(f"Predicted next distribution (first 5): {next_dist[:5]}")
    
    # Disease risk
    risk = markov.predict_disease_risk(current_theta, horizon=3)
    print(f"\n3-step disease risk: max_risk={risk['max_risk']:.4f} for topic {risk['max_risk_topic']}")
    
    # Analyze progression
    print("\nAnalyzing disease progression...")
    df = analyze_disease_progression(theta_sequences, target_topics=[0, 1, 2])
    print(df.head(10))
