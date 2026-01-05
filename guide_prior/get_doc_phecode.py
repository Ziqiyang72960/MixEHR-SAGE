# For each document, compute the count of ICD code under each PheCodes
import time
import torch
import pickle
import pandas as pd
#from corpus import Corpus
import sys
import os
from pathlib import Path

# Get absolute path to repository root
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
from corpus import Corpus

# Use absolute paths based on root_dir
mapping_dir = root_dir / "mapping"
phecode_mapping_dir = root_dir / "phecode_mapping"
store_dir = root_dir / "store"
guide_prior_dir = root_dir / "guide_prior"

vocab_ids = pickle.load(open(mapping_dir / "icd_vocab_ids.pkl", "rb"))
inv_vocab_ids = {v: k for k, v in vocab_ids.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds_topic_matrix = torch.load(phecode_mapping_dir / "seed_topic_matrix.pt", map_location=device, weights_only=False)  # get seed word-topic mapping, V x K matrix
V, K = seeds_topic_matrix.shape
c = Corpus.read_corpus_from_directory(str(store_dir), 'corpus.pkl') # read corpus file

pat_d = []
for d_i, doc in enumerate(c.dataset):
    pat_d.append(doc.doc_id)
df = pd.DataFrame(pat_d, columns=['SUBJECT_ID'])
df.to_csv(guide_prior_dir / 'pat_df.csv')
print('done')
time.sleep(200)

print('obtain D x K document-PheCode count matrix')
document_phecode_matrix = torch.zeros((c.D, K), device=device) # document-PheCode counts, D x K matrix
pat_d = []
for d_i, doc in enumerate(c.dataset):
    pat_d.append(doc.doc_id)
    for v, freq in doc.words_dict[0].items():
        document_phecode_matrix[d_i] += seeds_topic_matrix[v] * freq
    #print(doc, torch.sum(document_phecode_matrix[d_i]))
torch.save(document_phecode_matrix, guide_prior_dir / "document_phecode_matrix.pt")
