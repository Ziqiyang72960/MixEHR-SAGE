import time
import math
import numpy as np
from corpus import Corpus
import torch
from torch import nn, optim
import torch.nn.functional as F
#from torch.special import gammaln
from torch import lgamma
from utils import logsumexp

import os

mini_val = 1e-6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MixEHR_SAGE(nn.Module):
    def __init__(self, corpus, seeds_topic_matrix, modality_list, guided_modality=0, stochastic_VI=True, elbo_modality=0, batch_size=1000, out='./store/', guide_prior_path='./guide_prior/', enable_temporal=False, num_time_steps=10):
        """
        Arguments:
            corpus: document class.
            seeds_topic_matrix: V x K matrix, each element represents the existence of seed word w for topic k.
            batch_size: batch size for a minibatch
            out: output path
            guide_prior_path: path to guide_prior directory containing initialized tokens
            enable_temporal: whether to enable temporal inference with LSTM (default: False)
            num_time_steps: number of time steps/age groups for temporal modeling (default: 10)
        """
        super(MixEHR_SAGE, self).__init__()
        self.modalities = modality_list # name of modalites
        self.modaltiy_num = len(modality_list) # number of modaltiy M
        self.guided_modality = guided_modality # the modality defined as the guided modality
        # self.elbo_modality = elbo_modality
        self.out = out  # folder to save experiments
        self.guide_prior_path = guide_prior_path  # path to guide_prior directory

        self.stochastic_VI = stochastic_VI
        self.full_batch_generator = Corpus.generator_full_batch(corpus)
        self.batch_size = batch_size # document number in a mini batch
        self.mini_batch_generator = Corpus.generator_mini_batch(corpus, self.batch_size) # default batch size 1000

        self.C = corpus.C  # C is number of words in the corpus, use for updating gamma for SCVB0
        self.D = corpus.D # document number in full batch
        self.K = seeds_topic_matrix.shape[1]
        self.V = corpus.V  # vocabulary size of regular words
        self.seeds_topic_matrix = seeds_topic_matrix # V x K matrix, if exists value which indicates seed word w (row w) for topic k (column k)
        self.S = self.seeds_topic_matrix.sum(axis=0)
        self.beta = 0.1 # hyperparameter for prior of regular topic mixture phi_r
        self.beta_sum = [self.beta * V_m for V_m in self.V]
        self.mu = 0.05 # hyperparameter for prior of seed topic mixture phi_s
        self.mu_sum = self.mu * self.S # mu_sum is a K-length vector, mu_sum[k] is the sum of mu over all seed word (S[k]
        self.pi_init = 0.7
        self.pi = torch.full([self.K], self.pi_init, dtype=torch.double, requires_grad=False, device=device) # hyperparameter weight for indicator x
        # expected tokens
        self.exp_m = torch.zeros(self.D, self.K, dtype=torch.double, requires_grad=False, device=device) # suppose a general m_dk across different modalities
        # self.exp_m = [torch.zeros(self.D, self.K, requires_grad=False, device=device) for V_m in self.V] # suppose a modality-specified m_dk
        self.exp_n = [torch.zeros(self.V[m], self.K, dtype=torch.double, requires_grad=False, device=device)
                      for m in range(self.modaltiy_num)] # exp_n for differnt modality
        self.exp_s = torch.zeros(self.V[guided_modality], self.K, dtype=torch.double, requires_grad=False, device=device) # use V to represent, regular word for a topic is 0, only for guided modality
        self.exp_q_z = 0
        self.alpha_prior = 0.1  # Renamed from eta to avoid conflict with temporal eta
        self.init_priors = "./guide_prior/init_tokens/"
        self.initialize_tokens()
        self.elbo = []
        self.term1 = []
        self.term2 = []
        self.term3 = []
        self.term4 = []
        
        # Store temporal configuration
        self.enable_temporal = enable_temporal
        self.num_time_steps = num_time_steps

        # temporal inference component
        if self.enable_temporal:
            self.T = num_time_steps  # number of time steps
            # Initialize eta as T x K matrix for temporal hyperparameters
            self.eta = torch.rand(self.T, self.K, dtype=torch.double, requires_grad=True, device=device)
            
            # variational distribution for eta via amortization, eta is T x K matrix
            self.eta_hidden_size = 200  # number of hidden units for rnn
            self.eta_dropout = 0.0  # dropout rate on rnn for eta
            self.eta_nlayers = 3  # number of layers for eta
            self.delta = 0.01  # prior variance
            
            # LSTM network for temporal inference
            # Input: vocabulary distribution at each time step
            self.q_eta_map = nn.Linear(self.V[self.guided_modality], self.eta_hidden_size)
            self.q_eta = nn.LSTM(self.eta_hidden_size, self.eta_hidden_size, self.eta_nlayers, 
                                dropout=self.eta_dropout, batch_first=True)
            self.mu_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K, bias=True)
            self.logsigma_q_eta = nn.Linear(self.eta_hidden_size + self.K, self.K, bias=True)
            
            # optimizer for LSTM parameters
            self.clip = 0
            self.lr = 0.0001
            self.wdecay = 1.2e-6
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wdecay)
            self.max_logsigma_t = 5.0  # avoid the value to be too big
            self.min_logsigma_t = -5.0
            
            print(f"Temporal inference enabled with {self.T} time steps")
        else:
            # Use scalar alpha_prior for non-temporal inference
            self.T = 1
            print("Temporal inference disabled, using static alpha_prior")

    def initialize_tokens(self):
        '''
        obtain initialized tokens E[n_wk], E[s_wk], E[m_dk]
        dynamically loads tokens for all modalities based on self.modalities list
        '''
        print("Obtain initialized tokens")
        guided_modality_name = self.modalities[self.guided_modality]
        
        # Load exp_n for each modality dynamically
        for m, modality_name in enumerate(self.modalities):
            exp_n_path = os.path.join(self.guide_prior_path, f"init_exp_n_{modality_name}.pt")
            if os.path.exists(exp_n_path):
                self.exp_n[m] = torch.load(exp_n_path, map_location=device, weights_only=False)
            else:
                print(f"Warning: {exp_n_path} not found, using zeros for modality {modality_name}")
        
        # Load exp_s for guided modality
        exp_s_path = os.path.join(self.guide_prior_path, f"init_exp_s_{guided_modality_name}.pt")
        if os.path.exists(exp_s_path):
            self.exp_s = torch.load(exp_s_path, map_location=device, weights_only=False)
        else:
            print(f"Warning: {exp_s_path} not found, using zeros for exp_s")
        
        # Load exp_m (document-topic matrix, not modality-specific)
        exp_m_path = os.path.join(self.guide_prior_path, "init_exp_m.pt")
        if os.path.exists(exp_m_path):
            self.exp_m = torch.load(exp_m_path, map_location=device, weights_only=False)
        else:
            print(f"Warning: {exp_m_path} not found, using zeros for exp_m")
        
        self.exp_n_sum = [torch.sum(exp_n, dim=0) for exp_n in self.exp_n] # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector for each modality
        self.exp_s_sum = torch.sum(self.exp_s, dim=0) # sum over w, exp_p is [V K] dimensionality, exp_s_sum is K-len vector
        self.exp_m_sum = torch.sum(self.exp_m, dim=1) # sum over k, exp_m is [D K] dimensionality, exp_m_sum is D-len vector

        # print("Initialize tokens")
        # for i, d in enumerate(self.mini_batch_generator):  # For each epoach, we sample a series of mini_batch data once
        #     print("Running for %d minibatch", i)
        #     batch_docs, batch_indices, batch_C = d  # batch_C is total number of words within a minibatch for SCVB0
        #     batch_BOW = torch.zeros(len(batch_docs), self.V[0], dtype=torch.int, requires_grad=False, device=device)  # M x V
        #     for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
        #         for word_id, freq in doc.words_dict[0].items():
        #             batch_BOW[d_i, word_id] = freq
        #     for d_i, doc_id in enumerate(batch_indices):
        #         BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
        #         self.exp_s[BOW_nonzero] += self.seeds_topic_matrix[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1) * (self.pi_init)
        #         self.exp_n[0][BOW_nonzero] += self.seeds_topic_matrix[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1) * (1-self.pi_init)
        #         self.exp_n[0][BOW_nonzero] += (1-self.seeds_topic_matrix)[BOW_nonzero] * batch_BOW[d_i, BOW_nonzero].unsqueeze(1) / (self.K - 1)
        #         self.exp_m[doc_id] = 1 / self.K * batch_BOW[d_i].sum()
        # self.exp_m_sum = torch.sum(self.exp_m, dim=1,) # sum over k, exp_m is [D K] dimensionality, exp_m_sum is D-len vector
        # self.exp_n_sum = [None, None]
        # self.exp_n_sum[0] = torch.sum(self.exp_n[0], dim=0) # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector
        # self.exp_n_sum[1] = torch.sum(self.exp_n[0], dim=0) # sum over w, exp_n is [V K] dimensionality, exp_n_sum is K-len vector
        # self.exp_s_sum = torch.sum(self.exp_s, dim=0) # sum over w, exp_p is [V K] dimensionality, exp_s_sum is K-len vector


    def get_elbo(self, batch_indices, batch_C, minibatch, epoch, start_time):
        '''
        compute the elbo excluding eta, kl with respect to eta is computed seperately after estimation by neural network
        '''
        # compute kl(q_z || p_z)
        # E_q[log q(z | gamma)]
        # E_q[ log p(z | alpha), alpha is softplus(eta) if the temporal component is opened
        alpha = self.alpha_prior  # Use alpha_prior for non-temporal or as default
        constant_terms = self.D * torch.lgamma(torch.tensor(self.K * alpha)) - self.D * self.K * torch.lgamma(torch.tensor(alpha))
        p_z = (torch.sum(torch.lgamma(alpha + self.exp_m[batch_indices]), dim=1) -
               torch.lgamma(self.K * alpha + self.exp_m_sum[batch_indices]))
        kl_z = torch.sum(p_z) + constant_terms - self.exp_q_z
        # kl_z = torch.sum(p_z) - self.exp_q_z
        # E_q[log p(w | z, beta, mu, pi)]
        # kl_z = torch.sum(p_z) - self.exp_q_z
        log_sum_n_terms = 0
        log_sum_s_terms = 0
        for k in range(self.K):
            log_sum_n_terms += lgamma(torch.tensor(self.beta_sum[0])) \
                               - self.V[0] * lgamma(torch.tensor(self.beta)) \
                               + torch.sum(lgamma(self.exp_n[0][:, k] + self.beta) + self.exp_n[0][:, k] * torch.log(1 - self.pi[k])) \
                               - lgamma(self.exp_n_sum[0][k] + self.beta_sum[0])
            seed_exp_s_k = self.exp_s[torch.nonzero(self.exp_s[:, k]).squeeze(dim=1), k]  # a S[k]-len vector
            log_sum_s_terms += lgamma(torch.tensor(self.mu_sum[k])) \
                               - self.S[k] * lgamma(torch.tensor(self.mu)) \
                               + torch.sum(lgamma(seed_exp_s_k + self.mu) + seed_exp_s_k * torch.log(self.pi[k])) \
                               - lgamma(self.exp_s_sum[k] + self.mu_sum[k])
        loss = kl_z + torch.logsumexp(torch.stack([log_sum_n_terms, log_sum_s_terms]), dim=0)
        #print("elbo: ", loss.detach().cpu().numpy().item(),
        #      "p_z: ", (torch.sum(p_z) + constant_terms).detach().cpu().numpy().item(),
        #      "q_z: ", self.exp_q_z.detach().cpu().numpy().item(),
        #      "E_q[log p(w | z, beta, mu, pi)]: ", logsumexp([log_sum_n_terms, log_sum_s_terms]).detach().cpu().numpy().item(),
        #      )
        #print("constant: ", constant_terms.detach().cpu().numpy().item(),
        #      "kl of z:", kl_z,
        #      "p_z: ", torch.sum(p_z).detach().cpu().numpy().item(),
        #      "q_z: ", self.exp_q_z.detach().cpu().numpy().item(),
        #      )
        # print("took %s seconds for minibatch %s" % (time.time() - start_time, minibatch))
        self.elbo.append(loss.detach().cpu().numpy().item())
        self.term1.append((torch.sum(p_z) +constant_terms).detach().cpu().numpy().item())
        self.term2.append(self.exp_q_z.detach().cpu().numpy().item())
        self.term3.append(torch.logsumexp(torch.stack([log_sum_n_terms, log_sum_s_terms]), dim=0).detach().cpu().numpy().item())
        self.term4.append([log_sum_s_terms.detach().cpu().numpy().item(),
                           self.exp_s_sum.sum().detach().cpu().numpy().item(),])
        return loss.detach().cpu().numpy().item()

    def SCVB0_guided(self, batch_BOW, batch_indices, batch_C, iter_n, guided_m=0):
        # temp_exp_m = torch.zeros(self.D, self.K, device=device)
        temp_exp_m_batch = torch.zeros(batch_BOW.shape[0], self.K, dtype=torch.double, device=device)
        temp_exp_n = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device)
        temp_exp_s = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device)
        gamma_ss_sum = torch.zeros(self.K, device=device)
        gamma_sr_sum = torch.zeros(self.K, device=device)
        topic_occurrences = torch.matmul(batch_BOW.sum(0).float(), self.seeds_topic_matrix.float())
        topic_presence = (topic_occurrences > 0.0).int()  # Convert to 0 or 1, and use int type
        # M step
        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma_ss = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device) #  V x K  # non-seed regular word will be zero
            temp_gamma_sr = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device)
            temp_gamma_rr = torch.zeros(self.V[guided_m], self.K, dtype=torch.double, device=device)
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            # seed word and seed topic
            temp_gamma_ss[BOW_nonzero] = self.seeds_topic_matrix[BOW_nonzero] * (self.exp_m[doc_id] + self.alpha_prior) \
                                         * (self.mu + self.exp_s[BOW_nonzero]) / (self.mu_sum + self.exp_s_sum) * self.pi
            # seed word but regular topic
            temp_gamma_sr[BOW_nonzero] = self.seeds_topic_matrix[BOW_nonzero] * (self.exp_m[doc_id] + self.alpha_prior) \
                                         * (self.beta + self.exp_n[guided_m][BOW_nonzero]) / (self.beta_sum[guided_m] + self.exp_n_sum[guided_m]) * (1-self.pi)
            # regular word must be regular topic
            temp_gamma_rr[BOW_nonzero] = (1-self.seeds_topic_matrix[BOW_nonzero]) * (self.exp_m[doc_id] + self.alpha_prior) \
                                         * (self.beta + self.exp_n[guided_m][BOW_nonzero]) / (self.beta_sum[guided_m] + self.exp_n_sum[guided_m])
            # normalization
            temp_gamma_s_sum = temp_gamma_ss.sum(dim=1).unsqueeze(1) + temp_gamma_sr.sum(dim=1).unsqueeze(1)
            temp_gamma_r_sum = temp_gamma_rr.sum(dim=1).unsqueeze(1)
            temp_gamma_ss /= temp_gamma_s_sum + mini_val
            temp_gamma_sr /= temp_gamma_s_sum + mini_val
            temp_gamma_rr /= temp_gamma_r_sum + mini_val
            # calculate frequency * gamma for each word in a document
            seed_word_list = torch.nonzero(self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)).squeeze(1)
            non_seed_index = torch.where((1-self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)) >= 0, (1-self.seeds_topic_matrix[BOW_nonzero].sum(axis=1)), 0)
            non_seed_word_list = torch.nonzero(non_seed_index).squeeze(1)
            temp_gamma_s = (self.pi*temp_gamma_ss + (1-self.pi)*(temp_gamma_sr+temp_gamma_rr))[BOW_nonzero][seed_word_list]
            temp_gamma_r = temp_gamma_rr[BOW_nonzero][non_seed_word_list]
            temp_gamma = torch.cat((temp_gamma_s, temp_gamma_r), 0)

            # calculate sufficient statistics
            # temp_exp_m[doc_id] += torch.sum(torch.cat((temp_gamma_s, temp_gamma_r), 0) * batch_BOW[d_i, BOW_nonzero].unsqueeze(1), dim=0)
            temp_exp_m_batch[d_i] += torch.sum(temp_gamma * batch_BOW[d_i, BOW_nonzero].unsqueeze(1), dim=0)
            temp_exp_n += (temp_gamma_sr + temp_gamma_rr) * batch_BOW[d_i].unsqueeze(1)
            temp_exp_s += temp_gamma_ss * batch_BOW[d_i].unsqueeze(1)
            gamma_ss_sum += temp_gamma_ss.sum(0)
            gamma_sr_sum += temp_gamma_sr.sum(0)
            self.exp_q_z += torch.sum(temp_gamma * torch.log(temp_gamma+mini_val)) # used for update ELBO
        # E step
        # update expected terms
        rho = 1 / math.pow((iter_n + 1), 0.9)
        if self.stochastic_VI:
            self.exp_m[batch_indices] = (1 - rho) * self.exp_m[batch_indices] + rho * temp_exp_m_batch
            self.exp_m_sum = torch.sum(self.exp_m, dim=1)  # sum over k, exp_m is [D K] dimensionality
            self.exp_s = (1 - rho) * self.exp_s + rho * temp_exp_s * self.C[guided_m] / batch_C
            self.exp_s_sum = torch.sum(self.exp_s, dim=0)  # sum over w, exp_p is [V K] dimensionality
            self.exp_n[guided_m] = (1 - rho) * self.exp_n[guided_m] + rho * temp_exp_n * self.C[guided_m] / batch_C
            self.exp_n_sum[guided_m] = torch.sum(self.exp_n[guided_m], dim=0)  # sum over w, exp_n is [V K] dimensionality
            self.update_hyperparams(gamma_sr_sum, gamma_sr_sum, topic_presence)  # update hyperparameters
        else:
            self.exp_m[batch_indices] = temp_exp_m_batch
            self.exp_m_sum = torch.sum(self.exp_m, dim=1)  # sum over k, exp_m is [D K] dimensionality
            self.exp_s = temp_exp_s
            self.exp_s_sum = torch.sum(self.exp_s, dim=0)  # sum over w, exp_p is [V K] dimensionality
            self.exp_n[guided_m] = temp_exp_n
            self.exp_n_sum[guided_m] = torch.sum(self.exp_n[guided_m], dim=0)  # sum over w, exp_n is [V K] dimensionality
            self.update_hyperparams(gamma_ss_sum, gamma_sr_sum, topic_presence)  # update hyperparameters


    def SCVB0_unguided(self, batch_BOW, batch_indices, batch_C, iter_n, unguided_m):
        temp_exp_n = torch.zeros(self.V[unguided_m], self.K, dtype=torch.double, device=device)
        gamma_sum = torch.zeros(self.K, dtype=torch.double, device=device)
        # M step
        for d_i, doc_id in enumerate(batch_indices):
            temp_gamma = torch.zeros(self.V[unguided_m], self.K, dtype=torch.double, device=device) #  V x K
            BOW_nonzero = torch.nonzero(batch_BOW[d_i]).squeeze(dim=1)
            # regular word must be regular topic
            temp_gamma[BOW_nonzero] = (self.exp_m[doc_id] + self.alpha_prior) * (self.beta + self.exp_n[unguided_m][BOW_nonzero]) \
                                      / (self.beta_sum[unguided_m] + self.exp_n_sum[unguided_m])
            # normalization
            temp_gamma_sum = temp_gamma.sum(dim=1).unsqueeze(1)
            temp_gamma /= temp_gamma_sum + mini_val
            # calculate sufficient statistics
            temp_exp_n += temp_gamma * batch_BOW[d_i].unsqueeze(1)
        # E step
        # update expected terms
        rho = 1 / math.pow((iter_n + 5), 0.99)
        if self.stochastic_VI:
            self.exp_n[unguided_m] = (1-rho)*self.exp_n[unguided_m] + rho*temp_exp_n*self.C[unguided_m]/batch_C
            self.exp_n_sum[unguided_m] = torch.sum(self.exp_n[unguided_m], dim=0) # sum over w, exp_n is [V K] dimensionality
        else:
            self.exp_n[unguided_m] = temp_exp_n
            self.exp_n_sum[unguided_m] = torch.sum(self.exp_n[unguided_m], dim=0) # sum over w, exp_n is [V K] dimensionality


    def update_hyperparams(self, gamma_ss_sum, gamma_sr_sum, topic_presence):
        '''
        update hyperparameters pi using Bernoulli trial
        '''
        # update all pi
        # self.pi = self.exp_s_sum / (self.exp_s_sum + gamma_sr_sum + mini_val)

        # only update pi with topic presence
        potential_update = gamma_ss_sum / (gamma_ss_sum + gamma_sr_sum + mini_val)
        potential_update = potential_update.double()
        self.pi[topic_presence.bool()] = potential_update[topic_presence.bool()] - mini_val

        # self.pi = gamma_ss_sum / (gamma_ss_sum + gamma_sr_sum + mini_val)
        # self.pi = torch.where(self.pi > 0.7, self.pi, torch.ones(self.K, dtype=torch.double, device=device)*self.pi_init)
        # self.pi = torch.where(self.pi < 0.95, self.pi, torch.ones(self.K, dtype=torch.double, device=device)*self.pi_init*1.33)
        # print(self.pi.mean())

    def alpha_softplus_act(self):
        '''
        Apply softplus activation to eta to get alpha (Dirichlet hyperparameters)
        Softplus ensures alpha > 0: softplus(x) = log(1 + exp(x))
        '''
        return F.softplus(self.eta)
    
    def reparameterize(self, mu, logvar):
        '''
        Reparameterization trick for sampling from N(mu, var)
        z = mu + std * epsilon, where epsilon ~ N(0, 1)
        '''
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode_temporal_sequence(self, time_step_data):
        '''
        Encode temporal sequence data for LSTM input
        
        Args:
            time_step_data: T x V tensor, vocabulary distribution at each time step
        
        Returns:
            encoded: T x hidden_size tensor
        '''
        if not self.enable_temporal:
            raise ValueError("Temporal inference is not enabled")
        
        # Map vocabulary distributions to hidden space
        # time_step_data shape: (T, V[guided_modality])
        encoded = self.q_eta_map(time_step_data.float())  # (T, eta_hidden_size)
        return encoded
    
    def infer_eta_variational(self, time_step_data):
        '''
        Variational inference for eta (temporal hyperparameters) using LSTM
        
        Args:
            time_step_data: T x V tensor, word distributions at each time step
        
        Returns:
            eta_samples: T x K tensor, sampled eta values
            mu_eta: T x K tensor, means of variational distribution
            logvar_eta: T x K tensor, log-variances of variational distribution
        '''
        if not self.enable_temporal:
            raise ValueError("Temporal inference is not enabled")
        
        # Encode input sequence
        # time_step_data: (T, V) -> encoded: (T, hidden_size)
        encoded = self.encode_temporal_sequence(time_step_data)
        
        # Add batch dimension for LSTM: (1, T, hidden_size)
        encoded = encoded.unsqueeze(0)
        
        # LSTM forward pass
        lstm_out, _ = self.q_eta(encoded)  # (1, T, hidden_size)
        lstm_out = lstm_out.squeeze(0)  # (T, hidden_size)
        
        # Concatenate with previous eta for autoregressive modeling
        # For the first time step, use zeros
        prev_eta = torch.zeros(self.T, self.K, dtype=torch.float, device=device)
        prev_eta[1:] = self.eta[:-1].clone().detach().float()  # Shift eta by one time step
        
        # Concatenate LSTM output with previous eta
        lstm_eta_concat = torch.cat([lstm_out, prev_eta], dim=1)  # (T, hidden_size + K)
        
        # Compute variational parameters
        mu_eta = self.mu_q_eta(lstm_eta_concat)  # (T, K)
        logvar_eta = self.logsigma_q_eta(lstm_eta_concat)  # (T, K)
        
        # Clip log variance to avoid numerical issues
        logvar_eta = torch.clamp(logvar_eta, self.min_logsigma_t, self.max_logsigma_t)
        
        # Sample eta using reparameterization trick
        eta_samples = self.reparameterize(mu_eta, logvar_eta)
        
        return eta_samples, mu_eta, logvar_eta
    
    def compute_temporal_kl(self, mu_eta, logvar_eta):
        '''
        Compute KL divergence between variational distribution q(eta) and prior p(eta)
        KL(q(eta|mu,sigma) || p(eta|0,delta)) for each time step
        
        Args:
            mu_eta: T x K tensor, means of variational distribution
            logvar_eta: T x K tensor, log-variances of variational distribution
        
        Returns:
            kl_loss: scalar, total KL divergence
        '''
        # Prior: N(0, delta)
        # KL for Gaussian: 0.5 * sum(sigma^2/delta + mu^2/delta - 1 - log(sigma^2/delta))
        var_eta = torch.exp(logvar_eta)
        kl = 0.5 * torch.sum(
            var_eta / self.delta + 
            mu_eta.pow(2) / self.delta - 
            1 - 
            torch.log(var_eta / self.delta)
        )
        return kl
    
    def generate_temporal_word_distributions(self, corpus=None):
        '''
        Generate time-varying word distributions for temporal modeling
        This creates a T x V matrix where each row represents the vocabulary
        distribution at a specific time step.
        
        Args:
            corpus: Corpus object with temporal metadata (optional)
        
        Returns:
            time_step_data: T x V tensor
        '''
        if not self.enable_temporal:
            raise ValueError("Temporal inference is not enabled")
        
        # Initialize time step data
        V_guided = self.V[self.guided_modality]
        time_step_data = torch.zeros(self.T, V_guided, dtype=torch.double, device=device)
        
        # For now, create simple aggregated distributions per time bin
        # In a real implementation, this would use actual temporal metadata from corpus
        # Each time bin gets an equal share of the vocabulary
        for t in range(self.T):
            # Uniform distribution as placeholder
            # In practice, this should aggregate word counts from documents in time bin t
            time_step_data[t] = torch.ones(V_guided, dtype=torch.double, device=device) / V_guided
        
        return time_step_data

    def inference(self, max_epoch=5, save_every=1):
        '''
        inference algorithm for dynamic seed-guided topic model, apply stochastic collaposed variational inference for latent variable z,
        and apply stochastic gradient descent for dynamic variables \eta (\alpha)
        '''
        total_elbo_hist = []
        for epoch in range(0, max_epoch):
            start_time = time.time()
            print("Training for %d epoch" % epoch)
            elbo_hist = np.zeros((len(self.mini_batch_generator)))
            batch_n = (self.D // self.batch_size) + 1
            for minibatch, d in enumerate(self.mini_batch_generator):  # For each epoach, we sample a series of mini_batch data once
                print("Running for %d minibatch" % minibatch)
                start_time = time.time()
                batch_docs, batch_indices, batch_C = d  # batch_C is total number of ICD codes (only) in a minibatch for SCVB0
                for m in range(self.modaltiy_num):
                    # modaltiy specific BOW matrix, shape is D X V[m]
                    batch_BOW_m = torch.zeros(len(batch_docs), self.V[m], dtype=torch.int, requires_grad=False,
                                              device=device)  # document number (not M) x V
                    batch_C_m = sum([doc_C[m] for doc_C in batch_C])
                    for d_i, (doc_id, doc) in enumerate(zip(batch_indices, batch_docs)):
                        for word_id, freq in doc.words_dict[m].items():
                            batch_BOW_m[d_i, word_id] = freq
                    if m == self.guided_modality:
                        self.exp_q_z = 0  # update to zero for next minibatch
                        self.SCVB0_guided(batch_BOW_m, batch_indices, batch_C_m, epoch * batch_n + minibatch, guided_m=0)

                        # get per-iteration elbo
                        elbo = self.get_elbo(batch_indices, batch_C_m, minibatch, epoch, start_time)
                        # elbo_hist[minibatch] = elbo
                        # if minibatch == batch_n-1: # take per-epoch elbo
                        #     total_elbo_hist.append(np.mean(elbo_hist))
                        #     self.elbo.pop()

                        # only compute per-epoch elbo
                        if minibatch == batch_n-1:
                            elbo = self.get_elbo(batch_indices, batch_C_m, minibatch, epoch, start_time)
                            total_elbo_hist.append(elbo)
                            #print('elbo: ', self.elbo)
                            #print(total_elbo_hist)
                            # print('pz: ', self.term1)
                            # print('qz: ', self.term2)
                            # print('pw: ', self.term3)
                    else:
                        self.SCVB0_unguided(batch_BOW_m, batch_indices, batch_C_m, epoch * batch_n + minibatch, unguided_m=m)
            # for batch, d in enumerate(self.full_batch_generator):  # For each epoch, use full batch of data to compute elbo
            #     print("Compute ELBO for %d epoch" % epoch)
            #     batch_docs, batch_indices, batch_C = d  # batch_C is total number of ICD codes (only) in a minibatch for SCVB0
            #     for m in range(self.modaltiy_num):
            #         if m == self.guided_modality:
            #             batch_C_m = sum([doc_C[m] for doc_C in batch_C])
            #             elbo = self.get_elbo(batch_indices, batch_C_m, batch, epoch, start_time)
            #             total_elbo_hist.append(elbo)
            #         else:
            #             pass
            if epoch % 1 == 0:
                self.save_parameters(epoch)
        return total_elbo_hist

    def save_parameters(self, epoch):
        torch.save(self.exp_m, "results/toy_exp_m_%s.pt" % (epoch))
        for i, modality in enumerate(self.modalities):
            torch.save(self.exp_n[i], "results/toy_exp_n_%s_%s.pt" % (modality, epoch))
        torch.save(self.exp_s, "results/toy_exp_s_%s.pt" % (epoch))
        torch.save(self.pi, "results/toy_pi_%s.pt" % (epoch))

    def infer_theta(self, patient_bow, num_iterations=10, method='gibbs'):
        """
        Fast online inference for new patient's topic mixture (theta/risk).
        
        This implements LDA-style "folding-in" inference using either Gibbs sampling 
        or Variational Bayes. Uses the LEARNED phi (word-topic distributions) from 
        training to infer the topic mixture (theta) for new patients.
        
        Gibbs sampling:
        1. Initialize topic assignments for each word token randomly
        2. For each iteration, resample topic for each word token based on:
           P(z_i = k | z_{-i}, w, phi) ∝ n_{k,−i} * phi_{w,k}
        3. Compute theta from final topic counts
        
        Variational Bayes:
        1. For each word w, compute topic assignment probability: gamma_wk ∝ theta_k * phi_wk
        2. Update theta based on expected topic counts
        3. Repeat for num_iterations
        
        For the guided modality (ICD codes), we use both:
        - phi_regular: learned from exp_n (regular word-topic distribution)
        - phi_seed: learned from exp_s (seed word-topic distribution)
        
        Args:
            patient_bow: dict of {modality_index: {word_id: frequency}} for each modality
                        or list of dicts, one per modality
            num_iterations: number of inference iterations (default: 10)
            method: inference method - 'gibbs' (default) or 'variational'
        
        Returns:
            theta: K-dimensional tensor representing patient's topic mixture (risk profile)
        """
        # Convert patient_bow to list format if needed
        if isinstance(patient_bow, dict) and 0 not in patient_bow:
            # Single modality dict format
            patient_bow = [patient_bow]
        elif isinstance(patient_bow, dict):
            # Convert {modality_idx: bow_dict} to list
            patient_bow = [patient_bow.get(m, {}) for m in range(self.modaltiy_num)]
        
        # Pre-compute the LEARNED phi (word-topic distributions) for each modality
        # phi_m[w, k] = P(word w | topic k) for modality m
        phi = []
        for m in range(self.modaltiy_num):
            # Regular phi: (exp_n + beta) / (exp_n_sum + beta_sum)
            phi_m = (self.exp_n[m] + self.beta) / (self.exp_n_sum[m] + self.beta_sum[m] + mini_val)
            phi.append(phi_m)
        
        # For guided modality, also compute seed phi
        # phi_seed[w, k] = P(word w | topic k) for seed words
        phi_seed = (self.exp_s + self.mu) / (self.exp_s_sum + self.mu_sum + mini_val)
        
        if method.lower() == 'variational':
            return self._infer_theta_variational(patient_bow, phi, phi_seed, num_iterations)
        else:  # Default to Gibbs sampling
            return self._infer_theta_gibbs(patient_bow, phi, phi_seed, num_iterations)
    
    def _infer_theta_gibbs(self, patient_bow, phi, phi_seed, num_iterations):
        """
        Gibbs sampling inference for theta.
        
        Args:
            patient_bow: list of dicts, one per modality
            phi: list of phi matrices for each modality
            phi_seed: seed phi matrix for guided modality
            num_iterations: number of Gibbs sampling iterations
        
        Returns:
            theta: K-dimensional tensor
        """
        # Create a list of word tokens (modality, word_id) and initialize topic assignments
        word_tokens = []  # List of (modality_idx, word_id)
        topic_assignments = []  # Topic assignment for each token
        
        for m in range(min(len(patient_bow), self.modaltiy_num)):
            bow_m = patient_bow[m]
            if not bow_m:
                continue
                
            for word_id, freq in bow_m.items():
                if word_id >= self.V[m]:
                    continue  # Skip unknown words
                # Add freq tokens for this word
                for _ in range(int(freq)):
                    word_tokens.append((m, word_id))
                    # Initialize with random topic
                    topic_assignments.append(torch.randint(0, self.K, (1,), device=device).item())
        
        if len(word_tokens) == 0:
            # No valid words, return uniform distribution
            return torch.ones(self.K, dtype=torch.double, device=device) / self.K
        
        # Count initial topic assignments
        topic_counts = torch.zeros(self.K, dtype=torch.double, device=device)
        for topic in topic_assignments:
            topic_counts[topic] += 1
        
        # Gibbs sampling iterations
        for iteration in range(num_iterations):
            for token_idx, (m, word_id) in enumerate(word_tokens):
                # Remove current token's topic assignment
                old_topic = topic_assignments[token_idx]
                topic_counts[old_topic] -= 1
                
                # Compute sampling probability for each topic
                # P(z_i = k) ∝ (n_{k,−i} + alpha) * phi_{w,k}
                if m == self.guided_modality:
                    # For guided modality (ICD), use seed-guided inference
                    is_seed = self.seeds_topic_matrix[word_id]  # K-dim: 1 if seed for topic k, 0 otherwise
                    
                    # Combine seed and regular phi
                    phi_wk_seed = is_seed * phi_seed[word_id] * self.pi
                    phi_wk_regular = phi[m][word_id] * (1 - self.pi * is_seed)
                    phi_wk = phi_wk_seed + phi_wk_regular
                else:
                    # For unguided modality, standard LDA
                    phi_wk = phi[m][word_id]
                
                # Sampling probability
                prob = (topic_counts + self.eta) * phi_wk
                prob = prob / (prob.sum() + mini_val)
                
                # Sample new topic assignment
                new_topic = torch.multinomial(prob, 1).item()
                topic_assignments[token_idx] = new_topic
                topic_counts[new_topic] += 1
        
        # Compute final theta from topic counts
        # theta_k = (n_k + alpha) / (N + K * alpha)
        theta = (topic_counts + self.eta) / (topic_counts.sum() + self.K * self.eta)
        
        return theta
    
    def _infer_theta_variational(self, patient_bow, phi, phi_seed, num_iterations):
        """
        Variational Bayes inference for theta.
        
        Args:
            patient_bow: list of dicts, one per modality
            phi: list of phi matrices for each modality
            phi_seed: seed phi matrix for guided modality
            num_iterations: number of variational inference iterations
        
        Returns:
            theta: K-dimensional tensor
        """
        # Initialize theta with Dirichlet prior (uniform + alpha)
        theta = torch.ones(self.K, dtype=torch.double, device=device) / self.K
        
        # Variational inference iterations (folding-in)
        for iteration in range(num_iterations):
            # Accumulate expected topic counts for this document
            exp_topic_counts = torch.zeros(self.K, dtype=torch.double, device=device)
            
            for m in range(min(len(patient_bow), self.modaltiy_num)):
                bow_m = patient_bow[m]
                if not bow_m:
                    continue
                    
                for word_id, freq in bow_m.items():
                    if word_id >= self.V[m]:
                        continue  # Skip unknown words
                    
                    if m == self.guided_modality:
                        # For guided modality (ICD), use seed-guided inference
                        # Check if this word is a seed word for any topic
                        is_seed = self.seeds_topic_matrix[word_id]  # K-dim: 1 if seed for topic k, 0 otherwise
                        
                        # gamma_k ∝ theta_k * [pi_k * phi_seed_wk + (1-pi_k) * phi_regular_wk] for seed words
                        # gamma_k ∝ theta_k * phi_regular_wk for non-seed words
                        gamma_seed = is_seed * theta * phi_seed[word_id] * self.pi
                        gamma_regular = theta * phi[m][word_id] * (1 - self.pi * is_seed)
                        gamma = gamma_seed + gamma_regular
                    else:
                        # For unguided modality, standard LDA inference
                        # gamma_k ∝ theta_k * phi_wk
                        gamma = theta * phi[m][word_id]
                    
                    # Normalize to get topic assignment probabilities
                    gamma = gamma / (gamma.sum() + mini_val)
                    
                    # Accumulate expected counts (weighted by word frequency)
                    exp_topic_counts += gamma * freq
            
            # Update theta using variational update
            # theta_k ∝ alpha + sum_w (gamma_wk * n_w)
            theta = (exp_topic_counts + self.eta) / (exp_topic_counts.sum() + self.K * self.eta)
        
        return theta

    def infer_theta_batch(self, patients_bow_list, num_iterations=10, method='gibbs'):
        """
        Batch inference for multiple new patients' topic mixtures (theta/risk).
        
        Args:
            patients_bow_list: list of patient_bow dicts (see infer_theta for format)
            num_iterations: number of inference iterations (default: 10)
            method: inference method - 'gibbs' (default) or 'variational'
        
        Returns:
            thetas: (num_patients, K) tensor of topic mixtures
        """
        thetas = []
        for patient_bow in patients_bow_list:
            theta = self.infer_theta(patient_bow, num_iterations, method=method)
            thetas.append(theta)
        return torch.stack(thetas)

    def get_theta(self):
        """
        Get the learned theta (topic mixture) for all training documents.
        Theta is computed as normalized exp_m.
        
        Returns:
            theta: (D, K) tensor where each row is a document's topic mixture
        """
        # Normalize exp_m to get theta
        theta = (self.exp_m + self.eta) / (self.exp_m_sum.unsqueeze(1) + self.K * self.eta)
        return theta

    def get_phi(self, modality=0):
        """
        Get the learned phi (word-topic distribution) for a specific modality.
        This is the regular word-topic distribution learned from exp_n.
        
        In LDA terms: phi_wk = P(word w | topic k)
        
        Args:
            modality: modality index (default: 0, the guided modality)
        
        Returns:
            phi: (V, K) tensor where phi[w,k] = P(word w | topic k)
        """
        phi = (self.exp_n[modality] + self.beta) / (self.exp_n_sum[modality] + self.beta_sum[modality] + mini_val)
        return phi

    def get_phi_seed(self):
        """
        Get the learned seed phi (seed word-topic distribution) for the guided modality.
        This is specific to MixEHR-SAGE's seed-guided topic model.
        
        Returns:
            phi_seed: (V, K) tensor for seed word-topic distribution
        """
        phi_seed = (self.exp_s + self.mu) / (self.exp_s_sum + self.mu_sum + mini_val)
        return phi_seed

    @staticmethod
    def load_trained_model(model_path, corpus, seeds_topic_matrix, modality_list, 
                           guided_modality=0, guide_prior_path='./guide_prior/'):
        """
        Load a trained model for inference.
        
        Args:
            model_path: directory containing saved model parameters (exp_m, exp_n, exp_s, pi)
            corpus: Corpus object (can be empty, just needs V and modalities info)
            seeds_topic_matrix: seed topic matrix
            modality_list: list of modality names
            guided_modality: index of guided modality
            guide_prior_path: path to guide prior directory
        
        Returns:
            model: MixEHR_SAGE model loaded with trained parameters
        """
        # Create model instance
        model = MixEHR_SAGE(corpus, seeds_topic_matrix, modality_list, 
                           guided_modality=guided_modality, 
                           guide_prior_path=guide_prior_path)
        
        # Load trained parameters
        # Find the latest epoch
        import glob
        exp_m_files = glob.glob(os.path.join(model_path, "toy_exp_m_*.pt"))
        if not exp_m_files:
            raise FileNotFoundError(f"No trained model found in {model_path}")
        
        # Extract epoch numbers and find max
        epochs = [int(f.split('_')[-1].replace('.pt', '')) for f in exp_m_files]
        latest_epoch = max(epochs)
        
        # Load parameters
        model.exp_m = torch.load(os.path.join(model_path, f"toy_exp_m_{latest_epoch}.pt"), 
                                  map_location=device, weights_only=False)
        model.exp_s = torch.load(os.path.join(model_path, f"toy_exp_s_{latest_epoch}.pt"), 
                                  map_location=device, weights_only=False)
        model.pi = torch.load(os.path.join(model_path, f"toy_pi_{latest_epoch}.pt"), 
                               map_location=device, weights_only=False)
        
        for i, modality in enumerate(modality_list):
            exp_n_path = os.path.join(model_path, f"toy_exp_n_{modality}_{latest_epoch}.pt")
            if os.path.exists(exp_n_path):
                model.exp_n[i] = torch.load(exp_n_path, map_location=device, weights_only=False)
        
        # Recompute sums
        model.exp_n_sum = [torch.sum(exp_n, dim=0) for exp_n in model.exp_n]
        model.exp_s_sum = torch.sum(model.exp_s, dim=0)
        model.exp_m_sum = torch.sum(model.exp_m, dim=1)
        
        # Pre-compute and cache phi distributions for fast online inference
        model._cache_phi_distributions()
        
        print(f"Loaded model from epoch {latest_epoch}")
        return model

    def _cache_phi_distributions(self):
        """
        Pre-compute and cache phi (word-topic) distributions for fast online inference.
        Called automatically after loading a trained model.
        """
        # Cache regular phi for each modality: phi[w,k] = P(word w | topic k)
        self._phi_cached = []
        for m in range(self.modaltiy_num):
            phi_m = (self.exp_n[m] + self.beta) / (self.exp_n_sum[m] + self.beta_sum[m] + mini_val)
            self._phi_cached.append(phi_m)
        
        # Cache seed phi for guided modality - check shape compatibility first
        try:
            # Check if exp_s and mu/mu_sum have compatible shapes
            if self.exp_s.shape[0] == self.V[self.guided_modality]:
                self._phi_seed_cached = (self.exp_s + self.mu) / (self.exp_s_sum + self.mu_sum + mini_val)
            else:
                print(f"Warning: exp_s shape {self.exp_s.shape} doesn't match expected vocabulary size {self.V[self.guided_modality]}")
                print("Skipping seed phi caching - will compute on-the-fly if needed")
                self._phi_seed_cached = None
        except (ValueError, RuntimeError) as e:
            print(f"Warning: Could not cache seed phi distribution due to shape mismatch: {e}")
            print("Skipping seed phi caching - will compute on-the-fly if needed")
            self._phi_seed_cached = None
        
        print("Cached phi distributions for fast inference")

    def infer_theta_fast(self, patient_bow, num_iterations=5, method='gibbs'):
        """
        Ultra-fast online inference for new patient's topic mixture (theta/risk).
        
        Supports any subset of modalities - patients can have data for 1, 2, or all 
        modalities. Empty modalities are automatically skipped during inference.
        
        This is optimized for real-time/online inference scenarios:
        - Uses pre-cached phi distributions (no recomputation)
        - Vectorized operations where possible
        - Fewer default iterations (5 vs 10)
        - Handles partial modality data gracefully
        
        Args:
            patient_bow: list of dicts, one per modality. Each dict maps word_id to frequency.
                        Empty dicts {} for modalities without data.
                        Example with only ICD (modality 0): [{0: 1, 5: 2}, {}, {}]
                        Example with ICD + med: [{0: 1}, {10: 1, 15: 3}, {}]
            num_iterations: number of inference iterations (default: 5)
            method: inference method - 'gibbs' (default) or 'variational'
        
        Returns:
            theta: K-dimensional tensor representing patient's topic mixture (risk profile)
        
        Note:
            For easier usage with modality names, use infer_theta_by_modality() instead.
        """
        # Ensure phi is cached
        if not hasattr(self, '_phi_cached') or self._phi_cached is None:
            self._cache_phi_distributions()
        
        # Convert patient_bow to list format if needed
        if isinstance(patient_bow, dict) and 0 not in patient_bow:
            patient_bow = [patient_bow]
        elif isinstance(patient_bow, dict):
            patient_bow = [patient_bow.get(m, {}) for m in range(self.modaltiy_num)]
        
        # Use the method-specific implementation with cached phi
        if method.lower() == 'variational':
            return self._infer_theta_fast_variational(patient_bow, num_iterations)
        else:  # Default to Gibbs sampling
            return self._infer_theta_fast_gibbs(patient_bow, num_iterations)
    
    def _infer_theta_fast_variational(self, patient_bow, num_iterations):
        """Variational Bayes inference using cached phi."""
        # Initialize theta uniformly
        theta = torch.ones(self.K, dtype=torch.double, device=device) / self.K
        
        # Variational inference iterations using cached phi
        for _ in range(num_iterations):
            exp_topic_counts = torch.zeros(self.K, dtype=torch.double, device=device)
            
            for m in range(min(len(patient_bow), self.modaltiy_num)):
                bow_m = patient_bow[m]
                if not bow_m:
                    continue
                
                # Collect word ids and frequencies for vectorized computation
                word_ids = []
                freqs = []
                for word_id, freq in bow_m.items():
                    if word_id < self.V[m]:
                        word_ids.append(word_id)
                        freqs.append(freq)
                
                if not word_ids:
                    continue
                
                word_ids_t = torch.tensor(word_ids, device=device)
                freqs_t = torch.tensor(freqs, dtype=torch.double, device=device)
                
                if m == self.guided_modality:
                    # Vectorized seed-guided inference
                    is_seed = self.seeds_topic_matrix[word_ids_t]  # (num_words, K)
                    phi_regular = self._phi_cached[m][word_ids_t]   # (num_words, K)
                    phi_seed = self._phi_seed_cached[word_ids_t]    # (num_words, K)
                    
                    # gamma = theta * [is_seed * pi * phi_seed + (1 - is_seed * pi) * phi_regular]
                    gamma = theta.unsqueeze(0) * (is_seed * self.pi * phi_seed + 
                                                   (1 - is_seed * self.pi) * phi_regular)
                else:
                    # Vectorized standard LDA inference
                    phi_words = self._phi_cached[m][word_ids_t]  # (num_words, K)
                    gamma = theta.unsqueeze(0) * phi_words
                
                # Normalize each word's gamma
                gamma = gamma / (gamma.sum(dim=1, keepdim=True) + mini_val)
                
                # Accumulate weighted by frequency
                exp_topic_counts += (gamma * freqs_t.unsqueeze(1)).sum(dim=0)
            
            # Update theta
            theta = (exp_topic_counts + self.eta) / (exp_topic_counts.sum() + self.K * self.eta)
        
        return theta
    
    def _infer_theta_fast_gibbs(self, patient_bow, num_iterations):
        """Gibbs sampling inference using cached phi."""
        # Create word tokens and initialize topic assignments
        word_tokens = []
        topic_assignments = []
        
        for m in range(min(len(patient_bow), self.modaltiy_num)):
            bow_m = patient_bow[m]
            if not bow_m:
                continue
                
            for word_id, freq in bow_m.items():
                if word_id >= self.V[m]:
                    continue
                for _ in range(int(freq)):
                    word_tokens.append((m, word_id))
                    topic_assignments.append(torch.randint(0, self.K, (1,), device=device).item())
        
        if len(word_tokens) == 0:
            return torch.ones(self.K, dtype=torch.double, device=device) / self.K
        
        # Count initial topic assignments
        topic_counts = torch.zeros(self.K, dtype=torch.double, device=device)
        for topic in topic_assignments:
            topic_counts[topic] += 1
        
        # Gibbs sampling iterations
        for _ in range(num_iterations):
            for token_idx, (m, word_id) in enumerate(word_tokens):
                old_topic = topic_assignments[token_idx]
                topic_counts[old_topic] -= 1
                
                # Get phi from cache
                if m == self.guided_modality:
                    is_seed = self.seeds_topic_matrix[word_id]
                    phi_wk = (is_seed * self.pi * self._phi_seed_cached[word_id] + 
                             (1 - is_seed * self.pi) * self._phi_cached[m][word_id])
                else:
                    phi_wk = self._phi_cached[m][word_id]
                
                # Sample new topic
                prob = (topic_counts + self.eta) * phi_wk
                prob = prob / (prob.sum() + mini_val)
                new_topic = torch.multinomial(prob, 1).item()
                topic_assignments[token_idx] = new_topic
                topic_counts[new_topic] += 1
        
        # Compute final theta
        theta = (topic_counts + self.eta) / (topic_counts.sum() + self.K * self.eta)
        return theta

    def infer_theta_by_modality(self, patient_data, num_iterations=5):
        """
        Infer theta for a new patient using any subset of modalities.
        
        This method allows you to pass data for any combination of modalities
        (e.g., only ICD codes, or ICD + medications, or all three).
        
        Args:
            patient_data: dict mapping modality name to {word_id: frequency}
                         e.g., {'icd': {0: 1, 5: 2}, 'med': {10: 1}}
                         Only include modalities for which you have data.
            num_iterations: number of variational inference iterations (default: 5)
        
        Returns:
            theta: K-dimensional tensor representing patient's topic mixture (risk profile)
        
        Example:
            # Patient with only ICD codes
            theta = model.infer_theta_by_modality({'icd': {0: 1, 5: 2}})
            
            # Patient with ICD and medication codes
            theta = model.infer_theta_by_modality({
                'icd': {0: 1, 5: 2},
                'med': {10: 1, 15: 3}
            })
            
            # Patient with all modalities
            theta = model.infer_theta_by_modality({
                'icd': {0: 1},
                'med': {10: 1},
                'opcs': {5: 1}
            })
        """
        # Convert modality names to indices
        patient_bow = [{} for _ in range(self.modaltiy_num)]
        
        for modality_name, bow in patient_data.items():
            if modality_name in self.modalities:
                m = self.modalities.index(modality_name)
                patient_bow[m] = bow
            else:
                print(f"Warning: Unknown modality '{modality_name}', skipping. "
                      f"Available modalities: {self.modalities}")
        
        return self.infer_theta_fast(patient_bow, num_iterations)

    def infer_theta_batch_by_modality(self, patients_data_list, num_iterations=5):
        """
        Batch inference for multiple patients using modality names.
        
        Args:
            patients_data_list: list of dicts, each mapping modality name to {word_id: freq}
            num_iterations: number of VI iterations (default: 5)
        
        Returns:
            thetas: (num_patients, K) tensor of topic mixtures
        """
        thetas = []
        for patient_data in patients_data_list:
            theta = self.infer_theta_by_modality(patient_data, num_iterations)
            thetas.append(theta)
        return torch.stack(thetas)

    def infer_theta_batch_fast(self, patients_bow_list, num_iterations=5):
        """
        Fast batch inference for multiple new patients.
        
        Supports any subset of modalities - patients can have data for 1, 2, or all modalities.
        Empty modalities are automatically skipped.
        
        Args:
            patients_bow_list: list of patient_bow dicts (list format with one dict per modality)
            num_iterations: number of VI iterations (default: 5)
        
        Returns:
            thetas: (num_patients, K) tensor of topic mixtures
        """
        thetas = []
        for patient_bow in patients_bow_list:
            theta = self.infer_theta_fast(patient_bow, num_iterations)
            thetas.append(theta)
        return torch.stack(thetas)
    
    @staticmethod
    def _is_header_row(row_data):
        """
        Detect if a row is a header by checking if it contains non-numeric data.
        
        Args:
            row_data: pandas Series containing the row data (excluding first column)
        
        Returns:
            bool: True if row appears to be a header
        """
        # Check first few values to see if they're numeric
        for val in row_data.head(min(5, len(row_data))):
            try:
                float(val)
            except (ValueError, TypeError):
                # If we can't convert to float, it's likely a header
                return True
        return False
    
    @staticmethod
    def load_phi_from_csv(phi_csv_paths, modalities):
        """
        Load phi distributions from external CSV files.
        
        This method allows you to load pre-computed phi (word-topic probability) matrices
        from CSV files instead of training a model. Useful when you have pre-trained
        phi distributions from external sources (e.g., UKB_phi_icd.csv).
        
        The first column of the CSV should contain ICD codes with descriptions (e.g., "A00.0 Cholera").
        These will be parsed to extract codes and create a vocabulary mapping.
        
        Args:
            phi_csv_paths: dict mapping modality name to CSV file path
                          e.g., {'icd': 'UKB_phi_icd.csv', 'med': 'UKB_phi_med.csv'}
                          OR a single string path if only one modality
            modalities: list of modality names in order (e.g., ['icd', 'med', 'opcs'])
        
        Returns:
            tuple: (phi_distributions, code_mappings)
                phi_distributions: list of tensors, one per modality [V_m x K]
                code_mappings: dict mapping modality to {code: full_description}
        
        Example:
            phi_dists, code_maps = MixEHR_SAGE.load_phi_from_csv({
                'icd': 'UKB_phi_icd.csv',
                'med': 'UKB_phi_med.csv',
                'opcs': 'UKB_phi_opcs.csv'
            }, ['icd', 'med', 'opcs'])
        """
        import pandas as pd
        
        # Handle single path string
        if isinstance(phi_csv_paths, str):
            phi_csv_paths = {modalities[0]: phi_csv_paths}
        
        phi_distributions = []
        code_mappings = {}
        
        for modality in modalities:
            if modality in phi_csv_paths:
                csv_path = phi_csv_paths[modality]
                print(f"Loading phi for {modality} from {csv_path}")
                
                # Load CSV - first check if there's a header row
                # Read first two rows to detect header
                df_test = pd.read_csv(csv_path, nrows=2, header=None, low_memory=False)
                
                # Check if first row is a header (contains topic names or similar non-numeric data)
                # Header typically has strings like "Topic_0", "Topic_1", etc.
                header_row = 0 if MixEHR_SAGE._is_header_row(df_test.iloc[0, 1:]) else None
                
                # Load CSV with first column as index (ICD codes + descriptions)
                df = pd.read_csv(csv_path, header=header_row, index_col=0, dtype=float, low_memory=False)
                
                # Extract codes and create mapping
                code_to_desc = {}
                for full_desc in df.index:
                    full_desc_str = str(full_desc)
                    # Extract code (first part before space)
                    parts = full_desc_str.split(maxsplit=1)
                    code = parts[0]
                    code_to_desc[code] = full_desc_str
                    # Also map full description to itself
                    code_to_desc[full_desc_str] = full_desc_str
                
                code_mappings[modality] = code_to_desc
                
                # Convert to numpy array (all columns except index)
                phi_np = df.values.astype(float)
                phi_tensor = torch.tensor(phi_np, dtype=torch.double, device=device)
                phi_distributions.append(phi_tensor)
                print(f"  Loaded phi shape: {phi_tensor.shape} (V={phi_tensor.shape[0]}, K={phi_tensor.shape[1]})")
                print(f"  Created code mapping with {len(code_to_desc)} entries")
            else:
                print(f"Warning: No phi CSV provided for modality '{modality}', skipping")
                phi_distributions.append(None)
                code_mappings[modality] = {}
        
        return phi_distributions, code_mappings
    
    def infer_theta_with_external_phi(self, patient_data, phi_distributions, phi_seed=None, num_iterations=10):
        """
        Infer theta for a new patient using externally provided phi distributions.
        
        This method allows you to use pre-computed phi matrices from CSV files
        instead of the phi learned during training. Useful for inference with
        different trained models or external phi sources.
        
        Args:
            patient_data: dict mapping modality name to {word_id: frequency}
                         e.g., {'icd': {0: 1, 5: 2}, 'med': {10: 1}}
            phi_distributions: list of tensors from load_phi_from_csv(), one per modality
            phi_seed: optional seed phi tensor for guided modality (if None, uses phi_distributions[guided_modality])
            num_iterations: number of variational inference iterations (default: 10)
        
        Returns:
            theta: K-dimensional tensor representing patient's topic mixture (risk profile)
        
        Example:
            # Load external phi from CSV
            phi_dists = MixEHR_SAGE.load_phi_from_csv({
                'icd': 'UKB_phi_icd.csv',
                'med': 'UKB_phi_med.csv'
            }, ['icd', 'med', 'opcs'])
            
            # Infer theta using external phi
            theta = model.infer_theta_with_external_phi(
                {'icd': {0: 1, 5: 2}},
                phi_dists
            )
        """
        # Get K from phi distributions
        K = None
        for phi in phi_distributions:
            if phi is not None:
                K = phi.shape[1]
                break
        
        if K is None:
            raise ValueError("No valid phi distributions provided")
        
        # Initialize theta uniformly
        theta = torch.ones(K, dtype=torch.double, device=device) / K
        
        # Build patient bow list
        patient_bow_list = []
        for m, modality in enumerate(self.modalities):
            if modality in patient_data:
                patient_bow_list.append(patient_data[modality])
            else:
                patient_bow_list.append({})
        
        # Variational inference iterations
        for _ in range(num_iterations):
            exp_topic_counts = torch.zeros(K, dtype=torch.double, device=device)
            
            for m in range(len(patient_bow_list)):
                bow_m = patient_bow_list[m]
                if not bow_m or phi_distributions[m] is None:
                    continue
                
                # Collect word ids and frequencies
                word_ids = []
                freqs = []
                for word_id, freq in bow_m.items():
                    if word_id < phi_distributions[m].shape[0]:
                        word_ids.append(word_id)
                        freqs.append(freq)
                
                if not word_ids:
                    continue
                
                word_ids_t = torch.tensor(word_ids, device=device)
                freqs_t = torch.tensor(freqs, dtype=torch.double, device=device)
                
                # Get phi for these words
                if m == self.guided_modality and phi_seed is not None:
                    # Use seed-guided inference if phi_seed provided
                    is_seed = self.seeds_topic_matrix[word_ids_t]  # (num_words, K)
                    phi_regular = phi_distributions[m][word_ids_t]  # (num_words, K)
                    phi_s = phi_seed[word_ids_t]  # (num_words, K)
                    
                    # gamma = theta * [is_seed * pi * phi_seed + (1 - is_seed * pi) * phi_regular]
                    gamma = theta.unsqueeze(0) * (is_seed * self.pi * phi_s + 
                                                   (1 - is_seed * self.pi) * phi_regular)
                else:
                    # Standard LDA inference
                    phi_words = phi_distributions[m][word_ids_t]  # (num_words, K)
                    gamma = theta.unsqueeze(0) * phi_words
                
                # Normalize each word's gamma
                gamma = gamma / (gamma.sum(dim=1, keepdim=True) + mini_val)
                
                # Accumulate weighted by frequency
                exp_topic_counts += (gamma * freqs_t.unsqueeze(1)).sum(dim=0)
            
            # Update theta
            if exp_topic_counts.sum() > 0:
                theta = (exp_topic_counts + self.eta) / (exp_topic_counts.sum() + K * self.eta)
            else:
                # If no valid words, keep uniform distribution
                theta = torch.ones(K, dtype=torch.double, device=device) / K
        
        return theta

