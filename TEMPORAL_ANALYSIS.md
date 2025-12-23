# Temporal LSTM Implementation Analysis

## Issues with Current Commented Code

### 1. Missing Parameter: `self.T` (Time Steps)
**Problem**: Line 69 references `self.T` which is never defined
```python
# self.eta = torch.rand(self.T, self.K, requires_grad=False, device=device)
```
**Solution**: Need to add `self.T` parameter representing number of time steps/age groups

### 2. Missing Method: `alpha_softplus_act()`
**Problem**: Line 70 calls undefined method
```python
# self.alpha = self.alpha_softplus_act().to(device) # T x K
```
**Solution**: Need to implement method that applies softplus activation to eta

### 3. Vocabulary Size Conflict
**Problem**: Line 76 uses `self.V` which is a list, not a scalar
```python
# self.q_eta_map = nn.Linear(self.V, self.eta_hidden_size)
```
**Current**: `self.V` is a list of vocabulary sizes per modality
**Solution**: Use `self.V[self.guided_modality]` for the guided modality or create aggregated vocabulary representation

### 4. Variable Name Conflict
**Problem**: Line 59 defines `self.eta = 0.1` as a scalar hyperparameter
**Conflict**: Lines 69-79 want to use `self.eta` as a T x K tensor for temporal inference
**Solution**: Rename the hyperparameter (e.g., to `self.alpha_prior`) to avoid conflict

### 5. Missing Forward Pass
**Problem**: No implementation of LSTM forward pass to generate eta
**Solution**: Need to implement:
- Method to prepare input sequences for LSTM
- LSTM forward pass
- Variational inference for eta (mean and log-variance)
- Reparameterization trick for sampling

### 6. Missing Integration with Inference Loop
**Problem**: No integration of temporal component with existing SCVB0_guided
**Solution**: Need to:
- Update SCVB0_guided to use time-varying alpha (from eta)
- Modify ELBO calculation to include KL divergence for eta
- Update inference loop to call LSTM at appropriate times

### 7. Missing Time Series Data
**Problem**: No temporal/time-indexed data structure
**Solution**: Need to:
- Add temporal metadata (e.g., patient age, visit time)
- Create time-indexed document representations
- Generate time-varying word distributions for input to LSTM

### 8. Missing Optimizer for LSTM Parameters
**Problem**: Lines 81-84 define optimizer but it's commented
**Solution**: Need to properly initialize optimizer and integrate with training loop

## Required Implementation Steps

1. **Add Temporal Parameters to Constructor**
   - Add `num_time_steps` or `time_bins` parameter
   - Add temporal metadata handling

2. **Implement LSTM Architecture**
   - Implement `alpha_softplus_act()` method
   - Create proper LSTM input preparation
   - Implement variational inference for eta

3. **Fix Variable Naming**
   - Rename scalar `self.eta` to `self.alpha_prior`
   - Use `self.eta` as T x K tensor

4. **Implement Temporal Data Generation**
   - Add methods to create time-binned corpora
   - Generate temporal word distribution sequences
   - Handle missing time information gracefully

5. **Update Inference Methods**
   - Modify `get_elbo()` to include temporal KL divergence
   - Update `SCVB0_guided()` to use time-varying alpha
   - Integrate LSTM training in inference loop

6. **Add Helper Methods**
   - `encode_temporal_data()`: Prepare LSTM input
   - `reparameterize()`: Sampling with reparameterization trick
   - `get_alpha()`: Convert eta to alpha via softplus

## Recommended Architecture

```python
class MixEHR_SAGE(nn.Module):
    def __init__(self, ..., num_time_steps=10, enable_temporal=False):
        # ...existing code...
        
        # Rename scalar hyperparameter
        self.alpha_prior = 0.1  # Renamed from self.eta
        
        if enable_temporal:
            self.T = num_time_steps
            self.eta_hidden_size = 200
            self.eta_dropout = 0.0
            self.eta_nlayers = 3
            self.delta = 0.01
            
            # LSTM for temporal inference
            self.q_eta_map = nn.Linear(self.V[self.guided_modality], self.eta_hidden_size)
            self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, 
                                 self.eta_nlayers, dropout=self.eta_dropout)
            self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K, bias=True)
            self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K, bias=True)
            
            # Initialize eta
            self.eta = torch.rand(self.T, self.K, dtype=torch.double, 
                                 requires_grad=False, device=device)
            
            # Optimizer for LSTM parameters
            self.clip = 0
            self.lr = 0.0001
            self.wdecay = 1.2e-6
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wdecay)
            self.max_logsigma_t = 5.0
            self.min_logsigma_t = -5.0
```

## Next Steps

1. Implement temporal parameter addition
2. Create temporal data generation utilities
3. Implement LSTM forward pass
4. Integrate with existing inference
5. Add comprehensive tests
6. Update documentation
