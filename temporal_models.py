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
    
    IMPORTANT: For large K (>100 topics), use discretization='dominant' and 
    small smoothing (e.g., 1e-3 to 0.1) to avoid near-uniform distributions.
    """
    
    def __init__(self, num_topics: int, num_states: Optional[int] = None,
                 discretization: str = 'dominant'):
        """
        Initialize Markov Transition Model.
        
        Args:
            num_topics: Number of topics (K)
            num_states: Number of discrete states (default: num_topics)
            discretization: State assignment method
                - 'dominant': Assign to dominant topic (RECOMMENDED for large K)
                - 'soft': Use soft counts from theta (can over-smooth)
                - 'top_k': Use top-k topics with normalized weights
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
        self.raw_transition_counts = None  # Before smoothing (for diagnostics)
        
        # Diagnostic statistics
        self.num_transitions = 0
        self.num_patients = 0
        
    def fit(self, theta_sequences: Dict[str, torch.Tensor],
            smoothing: float = None,
            top_k: int = 5) -> 'MarkovTransitionModel':
        """
        Estimate transition matrix from theta sequences.
        
        Args:
            theta_sequences: Dict mapping patient_id to theta sequence (T x K)
            smoothing: Laplace smoothing parameter. 
                       Default: adaptive based on K (1e-3 for K>100, 0.1 for K<100)
                       Set to 0 for no smoothing (may have zero rows)
            top_k: For 'top_k' discretization, use top k topics
            
        Returns:
            self
        """
        # Adaptive smoothing based on K to avoid over-smoothing
        if smoothing is None:
            if self.K > 500:
                smoothing = 1e-4  # Very small for very large K
            elif self.K > 100:
                smoothing = 1e-3  # Small for large K
            else:
                smoothing = 0.1   # Moderate for small K
            logger.info(f"Using adaptive smoothing={smoothing} for K={self.K}")
        
        self.smoothing = smoothing
        self.top_k = top_k
        
        # Initialize raw counts (no smoothing)
        self.raw_transition_counts = np.zeros((self.num_states, self.num_states))
        raw_state_counts = np.zeros(self.num_states)
        raw_initial_counts = np.zeros(self.num_states)
        
        self.num_transitions = 0
        self.num_patients = 0
        
        for patient_id, theta_seq in theta_sequences.items():
            # Convert to numpy
            if torch.is_tensor(theta_seq):
                theta_seq = theta_seq.cpu().numpy()
            
            T = theta_seq.shape[0]
            if T < 1:
                continue
            
            self.num_patients += 1
            
            # Get state assignments for each time step
            states = self._discretize_theta_sequence(theta_seq)
            
            # Count initial state
            if self.discretization == 'soft':
                raw_initial_counts += theta_seq[0]
            elif self.discretization == 'top_k':
                top_indices = np.argsort(theta_seq[0])[-top_k:]
                top_weights = theta_seq[0][top_indices]
                top_weights = top_weights / (top_weights.sum() + mini_val)
                raw_initial_counts[top_indices] += top_weights
            else:
                raw_initial_counts[states[0]] += 1
            
            # Count transitions
            for t in range(T - 1):
                s_prev = states[t]
                s_curr = states[t + 1]
                self.num_transitions += 1
                
                if self.discretization == 'soft':
                    # Soft counts using outer product
                    self.raw_transition_counts += np.outer(theta_seq[t], theta_seq[t + 1])
                    raw_state_counts += theta_seq[t]
                elif self.discretization == 'top_k':
                    # Use top-k topics with normalized weights
                    top_prev = np.argsort(theta_seq[t])[-top_k:]
                    top_curr = np.argsort(theta_seq[t + 1])[-top_k:]
                    w_prev = theta_seq[t][top_prev]
                    w_prev = w_prev / (w_prev.sum() + mini_val)
                    w_curr = theta_seq[t + 1][top_curr]
                    w_curr = w_curr / (w_curr.sum() + mini_val)
                    for i, sp in enumerate(top_prev):
                        for j, sc in enumerate(top_curr):
                            self.raw_transition_counts[sp, sc] += w_prev[i] * w_curr[j]
                        raw_state_counts[sp] += w_prev[i]
                else:
                    # Hard counts (dominant)
                    self.raw_transition_counts[s_prev, s_curr] += 1
                    raw_state_counts[s_prev] += 1
        
        # Add smoothing
        self.transition_counts = self.raw_transition_counts + smoothing
        self.state_counts = raw_state_counts + smoothing * self.num_states
        initial_counts = raw_initial_counts + smoothing
        
        # Normalize to get probabilities
        # Avoid division by zero
        denom = self.state_counts[:, np.newaxis]
        denom = np.where(denom > 0, denom, 1.0)
        self.transition_matrix = self.transition_counts / denom
        
        # Normalize rows to sum to 1
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        self.transition_matrix = self.transition_matrix / row_sums
        
        self.initial_distribution = initial_counts / (initial_counts.sum() + mini_val)
        
        # Log diagnostic info
        self._log_diagnostics()
        
        return self
    
    def _log_diagnostics(self):
        """Log diagnostic statistics about the fitted model."""
        logger.info(f"Markov model fitted: {self.num_patients} patients, {self.num_transitions} transitions")
        
        # Check row variance (should be high for informative model)
        row_variances = np.var(self.transition_matrix, axis=1)
        mean_row_var = np.mean(row_variances)
        max_row_var = np.max(row_variances)
        
        # Check sparsity of raw counts
        nonzero_raw = np.sum(self.raw_transition_counts > 0)
        sparsity = nonzero_raw / (self.num_states ** 2)
        
        logger.info(f"  Transition matrix: mean row variance={mean_row_var:.6f}, max={max_row_var:.6f}")
        logger.info(f"  Raw counts sparsity: {sparsity:.4f} ({nonzero_raw}/{self.num_states**2} non-zero)")
        
        if mean_row_var < 1e-5:
            logger.warning("  WARNING: Very low row variance - predictions may be near-uniform!")
            logger.warning("  Consider: 1) lower smoothing, 2) use 'dominant' discretization, 3) more data")
    
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
        
        if self.discretization in ['dominant', 'top_k']:
            # Assign to dominant (highest probability) topic
            states = theta_seq.argmax(axis=1)
            
        elif self.discretization == 'soft':
            # For soft, still need discrete states for some operations
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
        elif self.discretization == 'top_k':
            # Use top-k weighted combination
            top_k = getattr(self, 'top_k', 5)
            top_indices = np.argsort(current_theta)[-top_k:]
            weights = current_theta[top_indices]
            weights = weights / (weights.sum() + mini_val)
            next_dist = np.zeros(self.num_states)
            for i, idx in enumerate(top_indices):
                next_dist += weights[i] * self.transition_matrix[idx]
        else:
            # Dominant topic
            current_state = current_theta.argmax()
            next_dist = self.transition_matrix[current_state]
        
        if return_distribution:
            return next_dist
        return next_dist.argmax()
    
    def predict_disease_risk(self, current_theta: Union[np.ndarray, torch.Tensor],
                           horizon: int = 1,
                           target_topics: Optional[List[int]] = None,
                           cumulative: bool = False) -> Dict[str, float]:
        """
        Estimate disease risk over a future horizon.
        
        Args:
            current_theta: Current topic mixture (K,)
            horizon: Number of time steps to predict ahead
            target_topics: List of topic indices representing target diseases
                          If None, returns risk for all topics
            cumulative: If True, return cumulative risk over 1..horizon steps
                       (probability of reaching state at ANY time in horizon)
                       If False, return risk at exactly step h (default)
            
        Returns:
            Dict with risk estimates for each topic
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if torch.is_tensor(current_theta):
            current_theta = current_theta.cpu().numpy()
        
        if cumulative:
            # Cumulative risk: probability of visiting state in ANY of steps 1..horizon
            # Approximation: max probability over all steps (upper bound)
            cumulative_risk = np.zeros(self.num_states)
            
            for h in range(1, horizon + 1):
                step_dist = self._compute_h_step_distribution(current_theta, h)
                cumulative_risk = np.maximum(cumulative_risk, step_dist)
            
            future_dist = cumulative_risk
        else:
            # Risk at exactly step h
            future_dist = self._compute_h_step_distribution(current_theta, horizon)
        
        # Compute risks
        if target_topics is None:
            target_topics = list(range(min(10, self.K)))  # Top 10 by default
        
        risks = {}
        for topic_idx in target_topics:
            if topic_idx < self.num_states:
                risks[f'topic_{topic_idx}'] = float(future_dist[topic_idx])
        
        # Add summary statistics
        risks['max_risk_topic'] = int(future_dist.argmax())
        risks['max_risk'] = float(future_dist.max())
        risks['entropy'] = float(-np.sum(future_dist * np.log(future_dist + mini_val)))
        
        # Add comparison with stationary distribution
        stationary = self.get_stationary_distribution()
        kl_from_stationary = np.sum(future_dist * np.log((future_dist + mini_val) / (stationary + mini_val)))
        risks['kl_from_stationary'] = float(kl_from_stationary)
        
        return risks
    
    def _compute_h_step_distribution(self, current_theta: np.ndarray, h: int) -> np.ndarray:
        """Compute h-step ahead distribution."""
        if h == 0:
            return current_theta
        
        # Compute n-step transition
        transition_n = np.linalg.matrix_power(self.transition_matrix, h)
        
        if self.discretization == 'soft':
            future_dist = current_theta @ transition_n
        elif self.discretization == 'top_k':
            top_k = getattr(self, 'top_k', 5)
            top_indices = np.argsort(current_theta)[-top_k:]
            weights = current_theta[top_indices]
            weights = weights / (weights.sum() + mini_val)
            future_dist = np.zeros(self.num_states)
            for i, idx in enumerate(top_indices):
                future_dist += weights[i] * transition_n[idx]
        else:
            current_state = current_theta.argmax()
            future_dist = transition_n[current_state]
        
        return future_dist
    
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
        stationary = np.abs(stationary)  # Ensure positive
        stationary = stationary / (stationary.sum() + mini_val)
        
        return stationary
    
    def diagnose(self) -> Dict[str, any]:
        """
        Run comprehensive diagnostic checks on the fitted model.
        
        Returns:
            Dict with diagnostic results and recommendations
        """
        if self.transition_matrix is None:
            return {'error': 'Model not fitted'}
        
        results = {}
        
        # 1. Row variance analysis
        row_variances = np.var(self.transition_matrix, axis=1)
        results['mean_row_variance'] = float(np.mean(row_variances))
        results['max_row_variance'] = float(np.max(row_variances))
        results['min_row_variance'] = float(np.min(row_variances))
        
        # 2. L1 distance between rows (measure of differentiation)
        n_sample = min(100, self.num_states)
        sample_indices = np.random.choice(self.num_states, n_sample, replace=False)
        l1_distances = []
        for i in range(n_sample):
            for j in range(i + 1, n_sample):
                l1 = np.sum(np.abs(self.transition_matrix[sample_indices[i]] - 
                                   self.transition_matrix[sample_indices[j]]))
                l1_distances.append(l1)
        results['mean_pairwise_l1'] = float(np.mean(l1_distances))
        
        # 3. Stationary distribution analysis
        stationary = self.get_stationary_distribution()
        results['stationary_entropy'] = float(-np.sum(stationary * np.log(stationary + mini_val)))
        results['stationary_max'] = float(stationary.max())
        results['uniform_entropy'] = float(np.log(self.num_states))
        
        # 4. Mixing time estimate (how fast converges to stationary)
        # Using spectral gap: 1 - |second eigenvalue|
        eigenvalues = np.linalg.eigvals(self.transition_matrix)
        eigenvalues_sorted = np.sort(np.abs(eigenvalues))[::-1]
        if len(eigenvalues_sorted) > 1:
            spectral_gap = 1 - eigenvalues_sorted[1]
            results['spectral_gap'] = float(spectral_gap)
            if spectral_gap > 0:
                results['mixing_time_approx'] = float(1 / spectral_gap)
        
        # 5. Sparsity of raw counts
        nonzero = np.sum(self.raw_transition_counts > 0) if self.raw_transition_counts is not None else 0
        results['raw_counts_sparsity'] = float(nonzero / (self.num_states ** 2))
        results['num_transitions'] = self.num_transitions
        results['num_patients'] = self.num_patients
        
        # 6. DIAGONAL DOMINANCE / STICKINESS ANALYSIS
        diagonal = np.diag(self.transition_matrix)
        results['diagonal_mean'] = float(np.mean(diagonal))
        results['diagonal_median'] = float(np.median(diagonal))
        results['diagonal_max'] = float(np.max(diagonal))
        results['diagonal_min'] = float(np.min(diagonal))
        results['diagonal_std'] = float(np.std(diagonal))
        
        # States with P(i->i) > 0.9 (highly sticky)
        sticky_states = np.sum(diagonal > 0.9)
        results['num_sticky_states_90'] = int(sticky_states)
        results['pct_sticky_states_90'] = float(sticky_states / self.num_states)
        
        # States with P(i->i) > 0.5 (moderately sticky)
        results['num_sticky_states_50'] = int(np.sum(diagonal > 0.5))
        results['pct_sticky_states_50'] = float(np.sum(diagonal > 0.5) / self.num_states)
        
        # 7. Row entropy (effective number of successors)
        row_entropies = -np.sum(self.transition_matrix * np.log(self.transition_matrix + mini_val), axis=1)
        results['mean_row_entropy'] = float(np.mean(row_entropies))
        results['min_row_entropy'] = float(np.min(row_entropies))
        results['max_row_entropy'] = float(np.max(row_entropies))
        
        # Effective number of successors (exp of entropy)
        effective_successors = np.exp(row_entropies)
        results['mean_effective_successors'] = float(np.mean(effective_successors))
        results['min_effective_successors'] = float(np.min(effective_successors))
        
        # 8. Top-5 successor mass per row
        top5_mass = []
        for i in range(self.num_states):
            row_sorted = np.sort(self.transition_matrix[i])[::-1]
            top5_mass.append(np.sum(row_sorted[:5]))
        results['mean_top5_successor_mass'] = float(np.mean(top5_mass))
        results['max_top5_successor_mass'] = float(np.max(top5_mass))
        
        # 9. Raw transition count statistics
        if self.raw_transition_counts is not None:
            raw_diagonal = np.diag(self.raw_transition_counts)
            raw_row_sums = np.sum(self.raw_transition_counts, axis=1)
            # Avoid division by zero warning
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_diagonal_ratio = np.where(raw_row_sums > 0, raw_diagonal / raw_row_sums, 0)
            results['raw_diagonal_ratio_mean'] = float(np.mean(raw_diagonal_ratio[raw_row_sums > 0]) if np.any(raw_row_sums > 0) else 0)
            results['raw_max_count'] = float(np.max(self.raw_transition_counts))
            results['raw_total_count'] = float(np.sum(self.raw_transition_counts))
        
        # 10. Recommendations
        recommendations = []
        
        if results['mean_row_variance'] < 1e-5:
            recommendations.append("❌ Very low row variance - transition matrix is near-uniform")
            recommendations.append("  -> Try: smoothing=1e-4 or lower")
            recommendations.append("  -> Try: discretization='dominant' instead of 'soft'")
        
        if results.get('mixing_time_approx', 0) < 2:
            recommendations.append(f"⚠️ Fast mixing (time ~{results.get('mixing_time_approx', 0):.1f}) - horizon>1 may give near-stationary")
            recommendations.append("  -> Use horizon=1 for meaningful short-term predictions")
        
        if results['raw_counts_sparsity'] < 0.01:
            recommendations.append(f"⚠️ Very sparse transitions ({results['raw_counts_sparsity']:.4f}) - many states never visited")
            recommendations.append("  -> Consider clustering topics or using top_k discretization")
        
        # Diagonal dominance warnings
        if results['diagonal_mean'] > 0.8:
            recommendations.append(f"❌ Highly diagonal-dominant (mean P(i→i)={results['diagonal_mean']:.3f})")
            recommendations.append("  -> This causes over-confident, sticky predictions")
            recommendations.append("  -> Try: discretization='soft' or 'top_k' to spread transitions")
            recommendations.append("  -> Try: predict_next_state_soft() to preserve theta uncertainty")
            recommendations.append("  -> Try: temperature scaling with predict_risk(..., temperature=2.0)")
        elif results['diagonal_mean'] > 0.5:
            recommendations.append(f"⚠️ Moderately diagonal-dominant (mean P(i→i)={results['diagonal_mean']:.3f})")
            recommendations.append("  -> Consider using soft predictions or temperature scaling")
        
        if results['pct_sticky_states_90'] > 0.1:
            recommendations.append(f"⚠️ {results['pct_sticky_states_90']*100:.1f}% of states are 'absorbing-ish' (P(i→i) > 0.9)")
            recommendations.append("  -> Consider clustering similar topics to reduce state space")
        
        if results['mean_effective_successors'] < 3:
            recommendations.append(f"⚠️ Low effective successors ({results['mean_effective_successors']:.1f}) - predictions lack diversity")
            recommendations.append("  -> Try: soft/top_k discretization during training")
        
        results['recommendations'] = recommendations
        
        return results
    
    def predict_next_state_soft(self, current_theta: Union[np.ndarray, torch.Tensor],
                                 return_distribution: bool = True) -> Union[int, np.ndarray]:
        """
        Predict next state using SOFT weighting from current_theta distribution.
        
        Unlike predict_next_state() which uses argmax(theta) for dominant discretization,
        this method ALWAYS uses the full theta distribution to weight transition rows,
        preserving uncertainty in the prediction.
        
        This is recommended when:
        - Predictions are over-confident (diagonal-dominant matrix)
        - You want to preserve uncertainty from the input theta
        
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
        
        # Always use weighted combination of all transition rows
        # This preserves uncertainty from the input theta
        next_dist = current_theta @ self.transition_matrix
        
        if return_distribution:
            return next_dist
        return next_dist.argmax()
    
    def predict_disease_risk_calibrated(self, current_theta: Union[np.ndarray, torch.Tensor],
                                        horizon: int = 1,
                                        temperature: float = 1.0,
                                        use_soft: bool = True,
                                        cumulative: bool = False,
                                        target_topics: Optional[List[int]] = None) -> Dict[str, float]:
        """
        Predict disease risk with calibration options.
        
        This method provides better-calibrated predictions when the transition matrix
        is diagonal-dominant (over-confident predictions).
        
        Args:
            current_theta: Current topic mixture (K,)
            horizon: Number of time steps to predict ahead
            temperature: Temperature for softmax scaling (>1 = more uncertain, <1 = more confident)
                        Default 1.0 = no scaling. Try 2.0-5.0 if predictions are over-confident.
            use_soft: If True, use soft weighting from theta (preserves uncertainty)
                     If False, use standard prediction method
            cumulative: If True, return max probability over 1..horizon steps
            target_topics: List of topic indices to report (default: top 10)
            
        Returns:
            Dict with calibrated risk estimates
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if torch.is_tensor(current_theta):
            current_theta = current_theta.cpu().numpy()
        
        # Apply temperature scaling to transition matrix
        if temperature != 1.0:
            # Scale log-probabilities by temperature
            log_trans = np.log(self.transition_matrix + mini_val)
            scaled_log_trans = log_trans / temperature
            # Renormalize rows
            trans_scaled = np.exp(scaled_log_trans)
            trans_scaled = trans_scaled / trans_scaled.sum(axis=1, keepdims=True)
        else:
            trans_scaled = self.transition_matrix
        
        if cumulative:
            # Cumulative risk over 1..horizon
            cumulative_risk = np.zeros(self.num_states)
            for h in range(1, horizon + 1):
                trans_h = np.linalg.matrix_power(trans_scaled, h)
                if use_soft:
                    step_dist = current_theta @ trans_h
                else:
                    current_state = current_theta.argmax()
                    step_dist = trans_h[current_state]
                cumulative_risk = np.maximum(cumulative_risk, step_dist)
            future_dist = cumulative_risk
        else:
            trans_h = np.linalg.matrix_power(trans_scaled, horizon)
            if use_soft:
                future_dist = current_theta @ trans_h
            else:
                current_state = current_theta.argmax()
                future_dist = trans_h[current_state]
        
        # Build results
        if target_topics is None:
            target_topics = list(range(min(10, self.K)))
        
        risks = {}
        for topic_idx in target_topics:
            if topic_idx < self.num_states:
                risks[f'topic_{topic_idx}'] = float(future_dist[topic_idx])
        
        risks['max_risk_topic'] = int(future_dist.argmax())
        risks['max_risk'] = float(future_dist.max())
        risks['entropy'] = float(-np.sum(future_dist * np.log(future_dist + mini_val)))
        
        # Comparison with stationary
        stationary = self.get_stationary_distribution()
        kl_from_stationary = np.sum(future_dist * np.log((future_dist + mini_val) / (stationary + mini_val)))
        risks['kl_from_stationary'] = float(kl_from_stationary)
        
        # Report calibration settings
        risks['temperature'] = temperature
        risks['use_soft'] = use_soft
        
        return risks
    
    def evaluate_calibration(self, theta_sequences: Dict[str, torch.Tensor],
                            temperature: float = 1.0) -> Dict[str, float]:
        """
        Evaluate calibration of predictions using held-out one-step transitions.
        
        Computes:
        - Brier score (lower is better)
        - Negative log-likelihood (lower is better)
        - Empirical vs predicted next-state accuracy
        
        Args:
            theta_sequences: Dict mapping patient_id to theta sequence (T x K)
            temperature: Temperature for calibrated predictions
            
        Returns:
            Dict with calibration metrics
        """
        if self.transition_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Apply temperature scaling
        if temperature != 1.0:
            log_trans = np.log(self.transition_matrix + mini_val)
            scaled_log_trans = log_trans / temperature
            trans_scaled = np.exp(scaled_log_trans)
            trans_scaled = trans_scaled / trans_scaled.sum(axis=1, keepdims=True)
        else:
            trans_scaled = self.transition_matrix
        
        brier_scores = []
        nll_scores = []
        accuracies_hard = []  # argmax prediction accuracy
        accuracies_soft = []  # soft prediction accuracy (prob mass on true state)
        
        for patient_id, theta_seq in theta_sequences.items():
            if torch.is_tensor(theta_seq):
                theta_seq = theta_seq.cpu().numpy()
            
            T = theta_seq.shape[0]
            if T < 2:
                continue
            
            for t in range(T - 1):
                current_theta = theta_seq[t]
                next_theta = theta_seq[t + 1]
                true_next_state = next_theta.argmax()
                
                # Soft prediction (weighted by current theta)
                pred_dist_soft = current_theta @ trans_scaled
                
                # Hard prediction (from dominant state)
                current_state = current_theta.argmax()
                pred_dist_hard = trans_scaled[current_state]
                
                # Brier score for soft prediction (vs one-hot true state)
                true_onehot = np.zeros(self.num_states)
                true_onehot[true_next_state] = 1.0
                brier = np.mean((pred_dist_soft - true_onehot) ** 2)
                brier_scores.append(brier)
                
                # NLL for soft prediction
                nll = -np.log(pred_dist_soft[true_next_state] + mini_val)
                nll_scores.append(nll)
                
                # Accuracy
                accuracies_hard.append(1.0 if pred_dist_hard.argmax() == true_next_state else 0.0)
                accuracies_soft.append(pred_dist_soft[true_next_state])  # Probability mass on true state
        
        results = {
            'brier_score': float(np.mean(brier_scores)),
            'nll': float(np.mean(nll_scores)),
            'accuracy_hard': float(np.mean(accuracies_hard)),
            'mean_prob_true_state': float(np.mean(accuracies_soft)),
            'num_transitions': len(brier_scores),
            'temperature': temperature
        }
        
        # Interpretation
        results['calibration_notes'] = []
        if results['brier_score'] > 0.1:
            results['calibration_notes'].append(f"High Brier score ({results['brier_score']:.4f}) - predictions may be poorly calibrated")
        if results['nll'] > 5:
            results['calibration_notes'].append(f"High NLL ({results['nll']:.2f}) - predictions assign low prob to true states")
        if results['mean_prob_true_state'] < 0.1:
            results['calibration_notes'].append(f"Low mean prob on true state ({results['mean_prob_true_state']:.4f}) - model under-confident or misspecified")
        
        return results
    
    def save(self, path: str):
        """Save model to file."""
        data = {
            'K': self.K,
            'num_states': self.num_states,
            'discretization': self.discretization,
            'transition_matrix': self.transition_matrix,
            'initial_distribution': self.initial_distribution,
            'transition_counts': self.transition_counts,
            'state_counts': self.state_counts,
            'raw_transition_counts': self.raw_transition_counts,
            'num_transitions': self.num_transitions,
            'num_patients': self.num_patients,
            'smoothing': getattr(self, 'smoothing', 1.0),
            'top_k': getattr(self, 'top_k', 5),
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
        model.raw_transition_counts = data.get('raw_transition_counts')
        model.num_transitions = data.get('num_transitions', 0)
        model.num_patients = data.get('num_patients', 0)
        model.smoothing = data.get('smoothing', 1.0)
        model.top_k = data.get('top_k', 5)
        
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


class TemporalMixEHRTrainer(nn.Module):
    """
    Temporal MixEHR-SAGE trainer that trains from scratch.
    
    This class allows training the model without pre-trained phi by jointly
    updating exp_m (document-topic) and exp_n (word-topic) distributions
    along with the LSTM temporal component, similar to the original 
    MixEHR_SAGE training procedure (SCVB0).
    
    Key features:
    - Train from scratch without existing phi
    - Intertwined updates of exp_m and exp_n during training
    - LSTM for temporal dynamics of topic prior η
    - Supports temporal bucketing (yearly, monthly, etc.)
    """
    
    def __init__(self, vocab_sizes: List[int], num_topics: int,
                 seeds_topic_matrix: torch.Tensor,
                 modalities: List[str],
                 guided_modality: int = 0,
                 hidden_size: int = 200,
                 num_lstm_layers: int = 3,
                 dropout: float = 0.0,
                 eta: float = 0.01,
                 beta: float = 0.01,
                 mu: float = 0.01,
                 delta: float = 0.01):
        """
        Initialize Temporal MixEHR Trainer.
        
        Args:
            vocab_sizes: List of vocabulary sizes for each modality
            num_topics: Number of topics (K)
            seeds_topic_matrix: Seed word-topic matrix (V x K)
            modalities: List of modality names
            guided_modality: Index of guided modality (default: 0 for ICD)
            hidden_size: LSTM hidden size
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
            eta: Dirichlet prior for document-topic distribution
            beta: Dirichlet prior for word-topic distribution (regular)
            mu: Dirichlet prior for word-topic distribution (seed)
            delta: Prior variance for η temporal dynamics
        """
        super(TemporalMixEHRTrainer, self).__init__()
        
        self.V = vocab_sizes
        self.K = num_topics
        self.modality_num = len(modalities)
        self.modalities = modalities
        self.guided_modality = guided_modality
        self.seeds_topic_matrix = seeds_topic_matrix.to(device)
        
        # Hyperparameters
        self.eta_prior = eta  # Document-topic prior
        self.beta = beta  # Regular word-topic prior
        self.mu = mu  # Seed word-topic prior
        self.delta = delta  # Temporal variance
        
        # Compute prior sums
        self.beta_sum = [beta * V for V in self.V]
        self.mu_sum = mu * self.V[guided_modality]
        
        # Initialize expected sufficient statistics
        # exp_n[m]: word-topic counts for modality m (V_m x K)
        # exp_s: seed word-topic counts (V_guided x K)
        # exp_m: document-topic counts (will be T x K per patient)
        self.exp_n = [torch.zeros(V, self.K, dtype=torch.double, device=device) 
                      for V in self.V]
        self.exp_s = torch.zeros(self.V[guided_modality], self.K, dtype=torch.double, device=device)
        
        # Sum statistics
        self.exp_n_sum = [torch.zeros(self.K, dtype=torch.double, device=device) 
                          for _ in self.V]
        self.exp_s_sum = torch.zeros(self.K, dtype=torch.double, device=device)
        
        # Pi: mixing proportion for seed vs regular topic
        self.pi = torch.ones(self.K, dtype=torch.double, device=device) * 0.5
        
        # LSTM for temporal dynamics
        self.lstm_model = TemporalLSTMModel(
            vocab_size=sum(self.V),
            num_topics=self.K,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            delta=delta
        ).to(device)
        
        # Optimizer for LSTM
        self.lr = 0.0001
        self.weight_decay = 1.2e-6
        self.clip_grad = 5.0
        self.optimizer = optim.Adam(
            self.lstm_model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Training history
        self.elbo_history = []
        
    def _initialize_from_data(self, temporal_corpus, vocab_mappings: Dict,
                              modality_list: List[str]):
        """
        Initialize exp_n and exp_s from the data using uniform priors.
        
        Args:
            temporal_corpus: TemporalCorpus with patient data
            vocab_mappings: Vocabulary mappings
            modality_list: List of modality names
        """
        logger.info("Initializing sufficient statistics from data...")
        
        # Get the size of seed topic matrix for bounds checking
        seed_matrix_size = self.seeds_topic_matrix.shape[0]
        
        # Count word occurrences per topic (uniform initialization)
        for patient_id, patient in temporal_corpus.patients.items():
            for bucket in patient.buckets:
                bow = bucket.get_bow_by_modality(vocab_mappings, modality_list)
                
                for m, modality in enumerate(modality_list):
                    if m >= self.modality_num:
                        continue
                    
                    for word_id, freq in bow[m].items():
                        if word_id >= self.V[m]:
                            continue
                        
                        if m == self.guided_modality:
                            # For guided modality, split between seed and regular
                            # Check bounds for seed matrix
                            if word_id < seed_matrix_size:
                                is_seed = self.seeds_topic_matrix[word_id].sum() > 0
                                if is_seed:
                                    # Initialize seed words to their seed topics
                                    self.exp_s[word_id] += self.seeds_topic_matrix[word_id] * freq * 0.5
                                    self.exp_n[m][word_id] += self.seeds_topic_matrix[word_id] * freq * 0.5
                                else:
                                    # Initialize regular words uniformly
                                    self.exp_n[m][word_id] += freq / self.K
                            else:
                                # Word outside seed matrix range - treat as regular
                                self.exp_n[m][word_id] += freq / self.K
                        else:
                            # Unguided modality: use exp_m from guided to distribute
                            self.exp_n[m][word_id] += freq / self.K
        
        # Update sums
        for m in range(self.modality_num):
            self.exp_n_sum[m] = torch.sum(self.exp_n[m], dim=0)
        self.exp_s_sum = torch.sum(self.exp_s, dim=0)
        
        logger.info("Initialization complete")
    
    def _compute_gamma(self, bow_m: Dict[int, int], exp_m_t: torch.Tensor,
                       modality: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute gamma (topic assignment probabilities) for a document at time t.
        
        This follows the SCVB0 algorithm from the original MixEHR_SAGE.
        
        Args:
            bow_m: Bag-of-words for modality m {word_id: freq}
            exp_m_t: Expected topic counts for this time step (K,)
            modality: Modality index
            
        Returns:
            gamma: Topic assignment probabilities (V x K sparse)
            gamma_ss: Seed word, seed topic assignments
            gamma_sr: Seed word, regular topic assignments
        """
        V_m = self.V[modality]
        
        # Convert BOW to tensor
        word_ids = list(bow_m.keys())
        if not word_ids:
            return None, None, None
        
        # Filter out word_ids that are out of bounds
        word_ids = [w for w in word_ids if w < V_m]
        if not word_ids:
            return None, None, None
        
        word_ids_t = torch.tensor(word_ids, device=device)
        seed_matrix_size = self.seeds_topic_matrix.shape[0]
        
        if modality == self.guided_modality:
            # Guided modality: handle seed and regular words separately
            gamma_ss = torch.zeros(len(word_ids), self.K, dtype=torch.double, device=device)
            gamma_sr = torch.zeros(len(word_ids), self.K, dtype=torch.double, device=device)
            gamma_rr = torch.zeros(len(word_ids), self.K, dtype=torch.double, device=device)
            
            for i, word_id in enumerate(word_ids):
                # Check bounds for seed matrix
                if word_id < seed_matrix_size:
                    is_seed = self.seeds_topic_matrix[word_id]
                else:
                    is_seed = torch.zeros(self.K, dtype=torch.double, device=device)
                
                # Seed word, seed topic
                gamma_ss[i] = is_seed * (exp_m_t + self.eta_prior) * \
                             (self.mu + self.exp_s[word_id]) / (self.mu_sum + self.exp_s_sum + mini_val) * self.pi
                
                # Seed word, regular topic
                gamma_sr[i] = is_seed * (exp_m_t + self.eta_prior) * \
                             (self.beta + self.exp_n[modality][word_id]) / \
                             (self.beta_sum[modality] + self.exp_n_sum[modality] + mini_val) * (1 - self.pi)
                
                # Regular word, regular topic
                gamma_rr[i] = (1 - is_seed) * (exp_m_t + self.eta_prior) * \
                             (self.beta + self.exp_n[modality][word_id]) / \
                             (self.beta_sum[modality] + self.exp_n_sum[modality] + mini_val)
            
            # Normalize
            gamma_s_sum = gamma_ss.sum(dim=1, keepdim=True) + gamma_sr.sum(dim=1, keepdim=True)
            gamma_r_sum = gamma_rr.sum(dim=1, keepdim=True)
            
            gamma_ss = gamma_ss / (gamma_s_sum + mini_val)
            gamma_sr = gamma_sr / (gamma_s_sum + mini_val)
            gamma_rr = gamma_rr / (gamma_r_sum + mini_val)
            
            # Combined gamma for exp_m update
            gamma = self.pi.unsqueeze(0) * gamma_ss + (1 - self.pi).unsqueeze(0) * (gamma_sr + gamma_rr)
            
            return gamma, gamma_ss, gamma_sr + gamma_rr
        else:
            # Unguided modality: standard LDA
            gamma = torch.zeros(len(word_ids), self.K, dtype=torch.double, device=device)
            
            for i, word_id in enumerate(word_ids):
                gamma[i] = (exp_m_t + self.eta_prior) * \
                          (self.beta + self.exp_n[modality][word_id]) / \
                          (self.beta_sum[modality] + self.exp_n_sum[modality] + mini_val)
            
            # Normalize
            gamma = gamma / (gamma.sum(dim=1, keepdim=True) + mini_val)
            
            return gamma, None, None
    
    def train_temporal(self, temporal_corpus, vocab_mappings: Dict,
                       modality_list: List[str], num_epochs: int = 10,
                       stochastic: bool = True) -> List[float]:
        """
        Train temporal MixEHR from scratch with intertwined exp_m and exp_n updates.
        
        This implements the full training procedure where phi (via exp_n) and 
        theta (via exp_m) are updated together at each iteration, along with
        the LSTM temporal component.
        
        Args:
            temporal_corpus: TemporalCorpus with patient data
            vocab_mappings: Vocabulary mappings
            modality_list: List of modality names
            num_epochs: Number of training epochs
            stochastic: Use stochastic VI (default: True)
            
        Returns:
            List of ELBO values per epoch
        """
        self.train()
        
        # Initialize from data if not already done
        if self.exp_n_sum[0].sum() == 0:
            self._initialize_from_data(temporal_corpus, vocab_mappings, modality_list)
        
        patient_ids = list(temporal_corpus.patients.keys())
        num_patients = len(patient_ids)
        total_V = sum(self.V)  # Use the vocab sizes from model, not vocab_mappings
        
        elbo_history = []
        
        for epoch in range(num_epochs):
            epoch_elbo = 0.0
            np.random.shuffle(patient_ids)
            
            # Process each patient
            for p_idx, patient_id in enumerate(patient_ids):
                patient = temporal_corpus.patients[patient_id]
                T = patient.num_time_steps
                
                # Build BOW sequence for LSTM
                bow_seq = torch.zeros(1, T, total_V, device=device)
                for t, bucket in enumerate(patient.buckets):
                    bow = bucket.get_bow_by_modality(vocab_mappings, modality_list)
                    offset = 0
                    for m, modality in enumerate(modality_list):
                        for word_id, freq in bow[m].items():
                            if word_id < self.V[m]:
                                bow_seq[0, t, offset + word_id] = freq
                        offset += self.V[m]
                
                # Get temporal alpha from LSTM
                self.optimizer.zero_grad()
                mu, logsigma = self.lstm_model(bow_seq)
                eta = self.lstm_model.sample_eta(mu, logsigma)
                alpha_t = self.lstm_model.get_alpha(eta)  # (1, T, K)
                
                # KL divergence for LSTM
                kl_loss = self.lstm_model.kl_divergence(mu, logsigma).mean()
                
                # Initialize exp_m for this patient (T x K) - no grad needed
                exp_m_patient = torch.zeros(T, self.K, dtype=torch.double, device=device)
                
                # SCVB0-style updates for each time step
                # Initialize as tensor for proper dtype/device handling
                reconstruction_loss = torch.tensor(0.0, dtype=torch.double, device=device)
                rho = 1 / (epoch * num_patients + p_idx + 1) ** 0.9  # Learning rate
                
                for t, bucket in enumerate(patient.buckets):
                    bow = bucket.get_bow_by_modality(vocab_mappings, modality_list)
                    
                    # Use LSTM alpha as dynamic prior
                    # dynamic_eta keeps grad for reconstruction loss
                    # dynamic_eta_ng (no grad) for SCVB0 count updates
                    dynamic_eta = alpha_t[0, t, :]
                    dynamic_eta_ng = dynamic_eta.detach()
                    
                    # Temporary accumulators for this time step (no grad needed)
                    temp_exp_n = [torch.zeros_like(self.exp_n[m]) for m in range(self.modality_num)]
                    temp_exp_s = torch.zeros_like(self.exp_s)
                    
                    for m, modality in enumerate(modality_list):
                        if m >= self.modality_num:
                            continue
                        
                        bow_m = bow[m]
                        if not bow_m:
                            continue
                        
                        # Filter word_ids to valid range
                        V_m = self.V[m]
                        word_ids = [w for w in bow_m.keys() if w < V_m]
                        if not word_ids:
                            continue
                        
                        freqs = torch.tensor([bow_m[w] for w in word_ids], 
                                            dtype=torch.double, device=device)
                        
                        # SCVB0 gamma/count updates under no_grad (use detached eta)
                        with torch.no_grad():
                            # Compute gamma with dynamic prior (uses detached eta)
                            filtered_bow_m = {w: bow_m[w] for w in word_ids}
                            gamma, gamma_ss, gamma_rr = self._compute_gamma(
                                filtered_bow_m, exp_m_patient[t] + dynamic_eta_ng, m
                            )
                            
                            if gamma is not None:
                                # Update exp_m for this time step
                                exp_m_patient[t] += (gamma * freqs.unsqueeze(1)).sum(dim=0)
                                
                                # Accumulate exp_n updates
                                for i, word_id in enumerate(word_ids):
                                    if m == self.guided_modality:
                                        temp_exp_s[word_id] += gamma_ss[i] * freqs[i]
                                        temp_exp_n[m][word_id] += gamma_rr[i] * freqs[i]
                                    else:
                                        temp_exp_n[m][word_id] += gamma[i] * freqs[i]
                        
                        # Compute reconstruction loss OUTSIDE no_grad
                        # Detach global stats so they don't carry old graphs
                        word_ids_t = torch.tensor(word_ids, device=device)
                        
                        # Detach exp_n/exp_s to prevent gradient flow through them
                        exp_n_words = self.exp_n[m].detach()[word_ids_t]
                        exp_n_sum_m = self.exp_n_sum[m].detach()
                        
                        if m == self.guided_modality:
                            exp_s_words = self.exp_s.detach()[word_ids_t]
                            exp_s_sum = self.exp_s_sum.detach()
                            phi_combined = self.pi.unsqueeze(0) * \
                                          (self.mu + exp_s_words) / (self.mu_sum + exp_s_sum + mini_val) + \
                                          (1 - self.pi).unsqueeze(0) * \
                                          (self.beta + exp_n_words) / (self.beta_sum[m] + exp_n_sum_m + mini_val)
                        else:
                            phi_combined = (self.beta + exp_n_words) / \
                                          (self.beta_sum[m] + exp_n_sum_m + mini_val)
                        
                        # exp_m_patient is detached (from no_grad), dynamic_eta keeps grad
                        exp_m_t_detached = exp_m_patient[t].detach()
                        theta_t = (exp_m_t_detached + dynamic_eta) / \
                                 (exp_m_t_detached.sum() + dynamic_eta.sum() + mini_val)
                        
                        log_prob = torch.log(torch.matmul(phi_combined, theta_t) + mini_val)
                        reconstruction_loss = reconstruction_loss - (freqs * log_prob).sum()
                    
                    # Stochastic update of exp_n and exp_s under no_grad
                    with torch.no_grad():
                        if stochastic:
                            for m in range(self.modality_num):
                                self.exp_n[m].mul_(1 - rho).add_(rho * temp_exp_n[m] * num_patients)
                                self.exp_n_sum[m] = torch.sum(self.exp_n[m], dim=0)
                            
                            self.exp_s.mul_(1 - rho).add_(rho * temp_exp_s * num_patients)
                            self.exp_s_sum = torch.sum(self.exp_s, dim=0)
                
                # Total loss
                total_loss = kl_loss + reconstruction_loss / (T * 1000)  # Scale reconstruction
                
                # Backward pass for LSTM
                total_loss.backward()
                
                if self.clip_grad > 0:
                    nn.utils.clip_grad_norm_(self.lstm_model.parameters(), self.clip_grad)
                
                self.optimizer.step()
                
                epoch_elbo -= total_loss.item()
            
            avg_elbo = epoch_elbo / num_patients
            elbo_history.append(avg_elbo)
            self.elbo_history.append(avg_elbo)
            
            logger.info(f"Epoch {epoch + 1}/{num_epochs}, ELBO: {avg_elbo:.4f}")
            
            # Save checkpoint periodically
            if (epoch + 1) % 5 == 0:
                self._update_pi()
        
        return elbo_history
    
    def _update_pi(self):
        """Update pi based on learned exp_s and exp_n."""
        # Pi is the proportion of seed topic vs regular topic for seed words
        seed_total = self.exp_s_sum + mini_val
        regular_total = self.exp_n_sum[self.guided_modality] + mini_val
        self.pi = seed_total / (seed_total + regular_total)
        self.pi = torch.clamp(self.pi, 0.1, 0.9)  # Keep pi bounded
    
    def get_phi(self, modality: int = 0) -> torch.Tensor:
        """
        Get learned word-topic distribution (phi) for a modality.
        
        Args:
            modality: Modality index
            
        Returns:
            phi: V x K tensor of word-topic probabilities
        """
        phi = (self.beta + self.exp_n[modality]) / \
              (self.beta_sum[modality] + self.exp_n_sum[modality].unsqueeze(0) + mini_val)
        return phi
    
    def get_phi_seed(self) -> torch.Tensor:
        """Get learned seed word-topic distribution."""
        phi_s = (self.mu + self.exp_s) / \
                (self.mu_sum + self.exp_s_sum.unsqueeze(0) + mini_val)
        return phi_s
    
    def get_combined_phi(self, modality: int = 0) -> torch.Tensor:
        """
        Get combined phi using π * φ^s + (1-π) * φ^r.
        
        Only applies to guided modality; returns regular phi for others.
        """
        if modality == self.guided_modality:
            phi_s = self.get_phi_seed()
            phi_r = self.get_phi(modality)
            # Apply seed mask
            is_seed = (self.seeds_topic_matrix.sum(dim=1, keepdim=True) > 0).float()
            phi_combined = is_seed * (self.pi.unsqueeze(0) * phi_s + 
                                      (1 - self.pi).unsqueeze(0) * phi_r) + \
                          (1 - is_seed) * phi_r
            return phi_combined
        else:
            return self.get_phi(modality)
    
    def infer_theta(self, bow: List[Dict[int, int]], num_iterations: int = 10) -> torch.Tensor:
        """
        Infer theta for new patient data using learned phi.
        
        Args:
            bow: List of BOW dicts, one per modality
            num_iterations: Number of inference iterations
            
        Returns:
            theta: K-dimensional topic mixture
        """
        theta = torch.ones(self.K, dtype=torch.double, device=device) / self.K
        
        for _ in range(num_iterations):
            exp_topic_counts = torch.zeros(self.K, dtype=torch.double, device=device)
            
            for m in range(min(len(bow), self.modality_num)):
                bow_m = bow[m]
                if not bow_m:
                    continue
                
                word_ids = list(bow_m.keys())
                freqs = torch.tensor([bow_m[w] for w in word_ids], 
                                    dtype=torch.double, device=device)
                
                # Get phi for this modality
                if m == self.guided_modality:
                    phi = self.get_combined_phi(m)
                else:
                    phi = self.get_phi(m)
                
                phi_words = phi[word_ids]  # (num_words, K)
                gamma = theta.unsqueeze(0) * phi_words
                gamma = gamma / (gamma.sum(dim=1, keepdim=True) + mini_val)
                
                exp_topic_counts += (gamma * freqs.unsqueeze(1)).sum(dim=0)
            
            theta = (exp_topic_counts + self.eta_prior) / \
                   (exp_topic_counts.sum() + self.K * self.eta_prior)
        
        return theta
    
    def infer_temporal_theta(self, temporal_corpus, vocab_mappings: Dict,
                            modality_list: List[str], 
                            num_iterations: int = 10) -> Dict[str, torch.Tensor]:
        """
        Infer temporal theta sequences using learned model.
        
        Args:
            temporal_corpus: TemporalCorpus
            vocab_mappings: Vocabulary mappings
            modality_list: List of modality names
            num_iterations: Inference iterations per time step
            
        Returns:
            Dict mapping patient_id to theta sequence (T x K)
        """
        self.eval()
        theta_sequences = {}
        total_V = sum(len(v) for v in vocab_mappings.values())
        
        with torch.no_grad():
            for patient_id, patient in temporal_corpus.patients.items():
                T = patient.num_time_steps
                
                # Build BOW sequence for LSTM
                bow_seq = torch.zeros(1, T, total_V, device=device)
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
                
                # Infer theta at each time step
                theta_seq = torch.zeros(T, self.K, dtype=torch.double, device=device)
                
                for t, bucket in enumerate(patient.buckets):
                    bow = patient.get_cumulative_bow(
                        bucket.time_index, vocab_mappings, modality_list
                    )
                    
                    # Use LSTM alpha as prior for this time step
                    dynamic_eta = alpha[0, t, :]
                    
                    # Initialize theta with dynamic prior
                    theta_t = dynamic_eta / (dynamic_eta.sum() + mini_val)
                    
                    for _ in range(num_iterations):
                        exp_counts = torch.zeros(self.K, dtype=torch.double, device=device)
                        
                        for m, modality in enumerate(modality_list):
                            if m >= self.modality_num:
                                continue
                            bow_m = bow[m]
                            if not bow_m:
                                continue
                            
                            word_ids = list(bow_m.keys())
                            freqs = torch.tensor([bow_m[w] for w in word_ids],
                                                dtype=torch.double, device=device)
                            
                            phi = self.get_combined_phi(m) if m == self.guided_modality else self.get_phi(m)
                            phi_words = phi[word_ids]
                            
                            gamma = theta_t.unsqueeze(0) * phi_words
                            gamma = gamma / (gamma.sum(dim=1, keepdim=True) + mini_val)
                            
                            exp_counts += (gamma * freqs.unsqueeze(1)).sum(dim=0)
                        
                        # Update with dynamic prior
                        theta_t = (exp_counts + dynamic_eta) / \
                                 (exp_counts.sum() + dynamic_eta.sum() + mini_val)
                    
                    theta_seq[t] = theta_t
                
                theta_sequences[patient_id] = theta_seq
        
        return theta_sequences
    
    def save(self, path: str):
        """Save the complete model state."""
        torch.save({
            'lstm_state_dict': self.lstm_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'exp_n': self.exp_n,
            'exp_s': self.exp_s,
            'exp_n_sum': self.exp_n_sum,
            'exp_s_sum': self.exp_s_sum,
            'pi': self.pi,
            'V': self.V,
            'K': self.K,
            'modalities': self.modalities,
            'elbo_history': self.elbo_history,
        }, path)
        logger.info(f"Saved temporal trainer model to {path}")
    
    @classmethod
    def load(cls, path: str, seeds_topic_matrix: torch.Tensor) -> 'TemporalMixEHRTrainer':
        """Load a saved model."""
        checkpoint = torch.load(path, map_location=device)
        
        model = cls(
            vocab_sizes=checkpoint['V'],
            num_topics=checkpoint['K'],
            seeds_topic_matrix=seeds_topic_matrix,
            modalities=checkpoint['modalities']
        )
        
        model.lstm_model.load_state_dict(checkpoint['lstm_state_dict'])
        model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.exp_n = checkpoint['exp_n']
        model.exp_s = checkpoint['exp_s']
        model.exp_n_sum = checkpoint['exp_n_sum']
        model.exp_s_sum = checkpoint['exp_s_sum']
        model.pi = checkpoint['pi']
        model.elbo_history = checkpoint.get('elbo_history', [])
        
        logger.info(f"Loaded temporal trainer model from {path}")
        return model


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
