# compute the initialized expected sufficient statistics for each modalities based on topic prior alpha
# Dynamically handles any number of modalities based on corpus modalities list
import torch
import os
import sys
from pathlib import Path

# Get absolute path to repository root
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from corpus import Corpus

# Define absolute paths
phecode_mapping_dir = root_dir / "phecode_mapping"
guide_prior_dir = root_dir / "guide_prior"
store_dir = root_dir / "store"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds_topic_matrix = torch.load(phecode_mapping_dir / "seed_topic_matrix.pt", map_location=device, weights_only=False) # get seed word-topic mapping, V x K matrix
topic_prior_alpha = torch.load(guide_prior_dir / "topic_prior_alpha.pt", map_location=device, weights_only=False)  # get topic prior alpha, D X K matrix
c = Corpus.read_corpus_from_directory(str(store_dir), 'corpus.pkl') # read corpus file
print(f"Vocabulary sizes for {len(c.modalities)} modalities: {c.V}")
print(f"Modalities: {c.modalities}")

# Get guided modality (first modality by default)
guided_modality = 0
guided_modality_name = c.modalities[guided_modality]

# Initialize exp_n for guided modality
exp_n_guided = torch.zeros(c.V[guided_modality], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_s_guided = torch.zeros(c.V[guided_modality], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)

# Process guided modality
for d_i, doc in enumerate(c.dataset):
    doc_id = doc.doc_id
    for word_id, freq in doc.words_dict[guided_modality].items():
        # update seed words
        exp_s_guided[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1
        exp_n_guided[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1
        # update regular words
        exp_n_guided[word_id] += (1-seeds_topic_matrix)[word_id] * freq * topic_prior_alpha[d_i]

# Save guided modality files
torch.save(exp_n_guided, guide_prior_dir / f"init_exp_n_{guided_modality_name}.pt")
torch.save(exp_s_guided, guide_prior_dir / f"init_exp_s_{guided_modality_name}.pt")
torch.save(topic_prior_alpha, guide_prior_dir / "init_exp_m.pt")
print(f"Saved init_exp_n_{guided_modality_name}.pt and init_exp_s_{guided_modality_name}.pt")

# Process unguided modalities dynamically
for m in range(len(c.modalities)):
    if m == guided_modality:
        continue  # Skip guided modality, already processed
    
    modality_name = c.modalities[m]
    exp_n_m = torch.zeros(c.V[m], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
    
    for d_i, doc in enumerate(c.dataset):
        for word_id, freq in doc.words_dict[m].items():
            exp_n_m[word_id] += topic_prior_alpha[d_i] * freq
    
    torch.save(exp_n_m, guide_prior_dir / f"init_exp_n_{modality_name}.pt")
    print(f"Saved init_exp_n_{modality_name}.pt")

print("Token counts computation completed for all modalities.")
