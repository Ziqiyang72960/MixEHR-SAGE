# compute the initialized expected sufficient statistics for each modalities based on topic prior alpha
# todo: read from the metedata instead of hard writing
import torch

import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent.absolute()
sys.path.append(str(root_dir))

from corpus import Corpus

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds_topic_matrix = torch.load("../phecode_mapping/seed_topic_matrix.pt", map_location=device) # get seed word-topic mapping, V x K matrix
topic_prior_alpha = torch.load("../guide_prior/topic_prior_alpha.pt", map_location=device)  # get topic prior alpha, D X K matrix
c = Corpus.read_corpus_from_directory('../store/', 'corpus.pkl') # read corpus file
print(c.V)

exp_n_icd = torch.zeros(c.V[0], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_s_icd = torch.zeros(c.V[0], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device) # default guided modality is 0
for d_i, doc in enumerate(c.dataset):
    #print(d_i)
    doc_id = doc.doc_id
    for word_id, freq in doc.words_dict[0].items(): # word_index v and freq
        # update seed words
        exp_s_icd[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1 # 0.7 /1
        exp_n_icd[word_id] += seeds_topic_matrix[word_id] * freq * topic_prior_alpha[d_i] * 1 # 0.3 /1
        # update regular words
        exp_n_icd[word_id] += (1-seeds_topic_matrix)[word_id] * freq * topic_prior_alpha[d_i]

exp_n_med = torch.zeros(c.V[1], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
exp_n_opcs = torch.zeros(c.V[2], topic_prior_alpha.shape[1], dtype=torch.double, requires_grad=False, device=device)
for d_i, doc in enumerate(c.dataset):
    #print(d_i)
    doc_id = doc.doc_id
    for word_id, freq in doc.words_dict[1].items(): # word_index v and freq
        exp_n_med[word_id] += topic_prior_alpha[d_i] * freq # if freq is 1, just add alpha prior
        # exp_n_med[word_id] += freq * topic_prior_alpha[d_i] # if freq is not 1, add freq * alpha prior
    for word_id, freq in doc.words_dict[2].items():
        exp_n_opcs[word_id] += topic_prior_alpha[d_i] * freq

torch.save(exp_n_icd, "init_exp_n_icd.pt")
torch.save(exp_s_icd, "init_exp_s_icd.pt")
torch.save(topic_prior_alpha, "init_exp_m.pt")
torch.save(exp_n_med, "./init_exp_n_med.pt")
torch.save(exp_n_opcs, "./init_exp_n_opcs.pt")
