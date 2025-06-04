import pandas as pd
import pickle as pkl
import numpy as np
import csv
import seaborn as sns
import math
#import plotly.figure_factory as ff
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import leaves_list
#from adjustText import adjust_text
import torch
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_n = torch.load("results/toy_exp_n_icd_4.pt", map_location=device)
exp_n = exp_n.cpu().detach().numpy()
V, K = exp_n.shape
beta = 0.1
phi_r = (beta + exp_n) / (beta * V + exp_n.sum(axis=1, keepdims=1))
phi_r = phi_r / phi_r.sum(axis=0, keepdims=1)  # normalization over V for each topic

phecode_ids = pkl.load(open("mapping/phecode_ids.pkl", "rb"))
pheno_meaning = pd.read_csv("mapping/phecode_definitions1.2.csv")
icd_name = pkl.load(open("mapping/icd_vocab_ids.pkl", "rb"))

# First mapping (from CSV file)
icd_mapping = pd.read_csv("mapping/allvalid2020 (detailed_titles_headings).csv")
icd_mapping = icd_mapping.iloc[:, [1, 2]].dropna()
# Shift rows down by 1 (make the first row the header)
icd_mapping.columns = icd_mapping.iloc[0]  # Set the first row as the header
icd_mapping = icd_mapping.iloc[1:]  # Drop the first row since it’s now the header
icd_mapping.reset_index(drop=True, inplace=True)
icd_mapping.columns.name = None
icd_mapping['Code_nodot'] = icd_mapping['Code'].str.replace('.', '')
# Second mapping (from TSV file)
second_icd_mapping = pd.read_csv("mapping/icd10_definition_coding19.tsv", delimiter='\t')
#second_icd_mapping['coding'] = second_icd_mapping['coding'].str.replace(".", "")  # Remove dots in codes

# Create dictionaries for both mappings
icd_meaning_dict_first = dict(zip(icd_mapping['Code'], icd_mapping['ICD Title']))
icd_meaning_dict_second = dict(zip(second_icd_mapping['coding'], second_icd_mapping['meaning']))

# Inverse mapping for phecode and ICD vocab
inv_phecode_ids = {v: k for k, v in phecode_ids.items()}
inv_icd_ids = {v: k for k, v in icd_name.items()}
token = pkl.load(open("mapping/tokenized_phecode_icd.pkl", "rb"))


# Map ICD codes using first and second mapping with fallback
def map_icd_code(icd_code):
    # Try mapping using the first mapping
    if icd_code in icd_meaning_dict_first:
        return icd_meaning_dict_first[icd_code], True  # True indicates first mapping
    # If not found, try the second mapping
    elif icd_code.replace('.', '') in icd_meaning_dict_second:
        return icd_meaning_dict_second[icd_code.replace('.', '')], False  # False indicates second mapping
    else:
        return icd_code, False  # Return a default value if not found in both mappings

# Apply the mapping to each ICD code, avoiding duplication in the row names
pheno_meaning_dict = dict(zip(pheno_meaning['phecode'], pheno_meaning['phenotype']))
icd_meaning_list = []
for i in range(V):
    icd_code = str(inv_icd_ids[i]) 
    mapped_meaning, is_first_mapping = map_icd_code(icd_code)  # Map the ICD code using the function
    
    # Clean up extra spaces in the ICD code in the meaning column for the second mapping
    mapped_meaning_clean = mapped_meaning.strip()
    
    if is_first_mapping:
        # For the first mapping, always add the ICD code with the meaning
        icd_code_r = icd_mapping[icd_mapping['Code'] == icd_code]['Code'].iloc[0]
        icd_meaning_list.append(f"{icd_code_r} {mapped_meaning}")
    else:
        icd_meaning_list.append(f"{mapped_meaning}")

# Apply to the full DataFrame
df_full = pd.DataFrame(phi_r)
pheno_meaning_list = []
for i in range(K):
    pheno_meaning_list.append(f"{str(inv_phecode_ids[i])} {pheno_meaning_dict[inv_phecode_ids[i]]}")

df_full.columns = pheno_meaning_list
df_full.index = icd_meaning_list

disease_order = [250.2, 411.4, 562.1, 290.1, 272.11]
phecode_index = []
for i in disease_order:
    phecode_index.append(phecode_ids[i])


# 1) Your ICD-10 chapter map by first letter
icd10_chapter_map = {
    'A':'Infectious & parasitic',
    'B':'Infectious & parasitic',
    'C':'Neoplasms',
    'D':'Blood & blood-forming organs',
    'E':'Endocrine, nutritional & metabolic',
    'F':'Mental & behavioural',
    'G':'Nervous system',
    'H':'Eye, ear & mastoid',
    'I':'Circulatory system',
    'J':'Respiratory system',
    'K':'Digestive system',
    'L':'Skin & subcut. tissue',
    'M':'Musculoskel. & connective',
    'N':'Genitourinary system',
    'O':'Pregnancy & childbirth',
    'P':'Perinatal conditions',
    'Q':'Congenital malformations',
    'R':'Symptoms & signs',
    'S':'Injury & poisoning',
    'T':'Injury & poisoning',
    'V':'External causes',
    'W':'External causes',
    'X':'External causes',
    'Y':'External causes',
    'Z':'Health status & services',
    'U':'Special purposes'
}

# ----------------------------------------------------------------------------
# 3) EXTRACT TOP‑3 ICD PER AGGREGATED PHECODE
# ----------------------------------------------------------------------------
phenotypes_df = df_full.iloc[:, phecode_index]

    
top_genes_indices = []
for col in phenotypes_df.columns:
    top3 = phenotypes_df[col].nlargest(3).index.tolist()
    top_genes_indices.extend(top3)
# Create a DataFrame for the top genes, sorted within each phenotype.
sorted_top_genes_dfs = []
for col in phenotypes_df.columns:
    top_genes = phenotypes_df[col].nlargest(3)
    sorted_top_genes_dfs.append(top_genes)

heatmap_df = phenotypes_df.loc[top_genes_indices]

# ----------------------------------------------------------------------------
# 2) EXTRACT TOP‑3 ICD ROWS PER AGGREGATED PHECODE (preserving duplicates)
# ----------------------------------------------------------------------------
top_rows = []
tuples = []
for col in heatmap_df.columns:
    top3 = heatmap_df[col].nlargest(3).index.tolist()
    top_rows.extend(top3)
    for icd in top3:
        tuples.append((col.split()[0], icd.split()[0]))
#heatmap_df = phenotypes_df.loc[heatmap_indices,]
# 2) Pick a color palette for all chapters
chapters = sorted(set(icd10_chapter_map.values()))
palette  = sns.color_palette("Set3", len(chapters))
chapter_to_color = dict(zip(chapters, palette))

def first_letter(label):
    return label.strip().split()[0][0].upper()

used_chapters = {
    icd10_chapter_map[first_letter(lbl)]
    for lbl in heatmap_df.index
}

# 2) Generate a palette sized to exactly len(used_chapters)
palette = sns.color_palette("Set3", n_colors=len(used_chapters))
used_chapters = sorted(used_chapters)           # so the order is consistent
chapter_to_color = dict(zip(used_chapters, palette))

# 3) Build your row_colors Series
row_colors = pd.Series(
    [ chapter_to_color[ icd10_chapter_map[first_letter(lbl)] ]
      for lbl in heatmap_df.index ],
    index=heatmap_df.index
)
sns.set_context("notebook", font_scale=1) 
# 5) Draw your clustermap with that side-bar
cg = sns.clustermap(
    heatmap_df,
    cmap="BuPu",
    figsize=(4, 9), 
    linewidths=1, linecolor='white',
    row_cluster=False, col_cluster=False,
    row_colors=row_colors,                # ← this adds the left color bar
    colors_ratio=(0.08, 0.03), 
    cbar_kws={
        'orientation':'horizontal',
        'shrink':0.25,
        'pad':0.01
    },
    cbar_pos=(1.1, 0.01, 0.25, 0.02)
)

rc_ax = cg.ax_row_colors
hm_ax = cg.ax_heatmap

# grab their current widths & heights
rc_pos = rc_ax.get_position()  # Bbox(x0, y0, x1, y1)
hm_pos = hm_ax.get_position()

rc_width  = rc_pos.width
rc_height = rc_pos.height

# compute a new x0 so that rc_ax.x1 == hm_ax.x0
new_x0 = hm_pos.x0 - rc_width - 0.001  # subtract a tiny gap if you like

# set rc_ax to that position
rc_ax.set_position([new_x0, rc_pos.y0, rc_width, rc_height])
rc_ax.xaxis.set_ticks_position('none')
from matplotlib.patches import Patch

# 1) Compute which chapters actually appear in your rows
used_chapters = {
    icd10_chapter_map[first_letter(lbl)]
    for lbl in heatmap_df.index
}

# 2) Build legend handles just for those
handles = [
    Patch(facecolor=chapter_to_color[ch], label=ch)
    for ch in sorted(used_chapters)
]

# 3) Add the legend to your clustermap
cg.ax_col_dendrogram.legend(
    handles=handles,
    title="ICD-10 Chapter",
    bbox_to_anchor=(1.02, 1),
    loc='right',
    frameon=False
)
cax = cg.cax
max_val = heatmap_df.values.max()
cax.set_xticks([0, max_val])
cax.set_xticklabels(['0', f'{max_val:.4f}'])
cax.xaxis.set_ticks_position('none')
handles = [Patch(facecolor=chapter_to_color[ch], label=ch)
           for ch in used_chapters]
leg = cg.ax_col_dendrogram.legend(
    handles=handles,
    title = 'ICD10 Categories',
    bbox_to_anchor=(1.02, 1),
    loc='lower left',
    frameon=True             
)
ax = cg.ax_heatmap

# 2) Move ticks & labels to the left spine
#ax.yaxis.tick_left()
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
# 3) Add padding so labels clear the color‐bar
ax.tick_params(axis='y', left=False, pad = 15)
#desc_to_code = {v:k for k,v in icd_meaning_dict.items()}

# 4) Figure out which phecode_ids are in your heatmap columns
#    Assumes col labels look like "123.4 Some phecode desc"
phecode_codes = [col.split()[0] for col in heatmap_df.columns]
phecode_idxs  = [phecode_ids[float(c)] for c in phecode_codes]
from decimal import Decimal

def has_more_than_one_decimal(phecode):
    d = Decimal(str(phecode)).normalize()
    # as_tuple().exponent is negative of number of decimal places
    return -d.as_tuple().exponent > 1

label_match = []
inv_token = {}
for k, vals in token.items():
    for v in vals:
        inv_token.setdefault(v, k)
for phecode_str, icd_str in tuples:
    pid = phecode_ids[float(phecode_str)] # phecode ids index
    # convert icd_str back to your integer index:
    icd_idx = icd_name[icd_str]
    label_match.append(icd_idx in token[pid])
# Now iterate your y‑ticklabels and add “*” when label_match is False
print(label_match)
labels = ax.get_yticklabels()

# build a list of new label strings
new_texts = []
for lbl, match in zip(labels, label_match):
    txt = lbl.get_text()
    new_texts.append(("*" + txt) if not match else txt)

def wrap_label(label, max_words=8):
    words = label.split()
    if len(words) > max_words:
        # first line: first max_words words
        # second line: the rest
        return " ".join(words[:max_words]) + "\n" + " ".join(words[max_words:])
    return label
    

# set them all in one go
#ax.set_yticklabels(new_texts, rotation=0)
wrapped_texts = [wrap_label(txt, 7) for txt in new_texts]
ax.set_yticklabels(wrapped_texts, rotation=0)
# 6) Style the box (edge‐color, line‐width, face‐color)
frame = leg.get_frame()
frame.set_edgecolor('grey')
frame.set_linewidth(1.0)
frame.set_facecolor('white')
plt.show()


import torch
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
exp_n_med = torch.load("results/toy_exp_n_med_4.pt", map_location=device)
exp_n_med = exp_n_med.cpu().detach().numpy()
V, K = exp_n_med.shape
beta = 0.1
phi_r_med = (beta + exp_n_med) / (beta*V + exp_n_med.sum(axis=1, keepdims=1))
phi_r_med = phi_r_med / phi_r_med.sum(axis=0, keepdims=1) # normalization over V for each topic
#phi_r_exp = np.exp(phi_r)
#phi_r_softmax = phi_r_exp / phi_r_exp.sum(axis=0, keepdims=1)
inv_phecode_ids = {v: k for k, v in phecode_ids.items()}
med_ids = pkl.load(open("mapping/med_vocab_ids.pkl", "rb"))
inv_med_ids = {v: k for k, v in med_ids.items()}
#filtered_phecode_list = [phecode for phecode in pheno_meaning_dict.keys() if phecode in phecode_ids.keys()]
#filtered_phecode_list_phecode_index_list = [phecode_ids[phecode] for phecode in filtered_phecode_list]
df_full_med = pd.DataFrame(phi_r_med)
pheno_meaning_list = []
med_meaning_list = []
for i in range(K):
    pheno_meaning_list.append(str(inv_phecode_ids[i]) + ' ' + pheno_meaning_dict[inv_phecode_ids[i]])
for i in range(V):
    med_meaning_list.append(str(inv_med_ids[i]) )
    # pheno_meaning_list.append(pheno_meaning_dict[inv_phecode_ids[i]])
df_full_med.columns = pheno_meaning_list
df_full_med.index = med_meaning_list
#df_complete = df_full

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

phenotypes_df = df_full_med.iloc[:, phecode_index]

top_genes_indices = []
for col in phenotypes_df.columns:
    top3 = phenotypes_df[col].nlargest(3).index.tolist()
    top_genes_indices.extend(top3)

heatmap_df = phenotypes_df.loc[top_genes_indices]

# ----------------------------------------------------------------------------
# 2) TOP‑3 ATC Medication rows per phecode
# ----------------------------------------------------------------------------
top_rows = []
tuples = []
for col in heatmap_df.columns:
    top3 = heatmap_df[col].nlargest(3).index.tolist()
    top_rows.extend(top3)
    for icd in top3:
        tuples.append((col.split()[0], icd.split()[0]))

heatmap_df = heatmap_df.loc[top_rows].drop_duplicates()

# 1) Your ICD-10 chapter map by first letter
atc_first_level_map = {
    'A': 'Alimentary tract and metabolism',
    'B': 'Blood and blood forming organs',
    'C': 'Cardiovascular system',
    'D': 'Dermatologicals',
    'G': 'Genito-urinary system and sex hormones',
    'H': 'Systemic hormonal preparations, excluding sex hormones and insulins',
    'J': 'Antiinfectives for systemic use',
    'L': 'Antineoplastic and immunomodulating agents',
    'M': 'Musculo-skeletal system',
    'N': 'Nervous system',
    'P': 'Antiparasitic products, insecticides and repellents',
    'R': 'Respiratory system',
    'S': 'Sensory organs',
    'V': 'Various'
}


# 2) Pick a color palette for all chapters
chapters = sorted(set(atc_first_level_map.values()))
palette  = sns.color_palette("Set3", len(chapters))
chapter_to_color = dict(zip(chapters, palette))

def first_letter(label):
    return label.strip().split()[0][0].upper()

used_chapters = {
    atc_first_level_map[first_letter(lbl)]
    for lbl in heatmap_df.index
}

# 2) Generate a palette sized to exactly len(used_chapters)
palette = sns.color_palette("tab10", n_colors=len(used_chapters))
used_chapters = sorted(used_chapters)           # so the order is consistent
chapter_to_color = dict(zip(used_chapters, palette))

# 3) Build your row_colors Series
row_colors = pd.Series(
    [ chapter_to_color[ atc_first_level_map[first_letter(lbl)] ]
      for lbl in heatmap_df.index ],
    index=heatmap_df.index
)
sns.set_context("notebook", font_scale=1) 
# 5) Draw your clustermap with that side-bar
cg = sns.clustermap(
    heatmap_df,
    cmap="BuPu",
    figsize=(6, 9), 
    linewidths=1, linecolor='white',
    row_cluster=False, col_cluster=False,
    row_colors=row_colors,                # ← this adds the left color bar
    colors_ratio=(0.08, 0.03), 
    cbar_kws={
        'orientation':'horizontal',
        'shrink':0.25,
        'pad':0.01
    },
    cbar_pos=(1.2, 0.01, 0.25, 0.02)
)

rc_ax = cg.ax_row_colors
hm_ax = cg.ax_heatmap

# grab their current widths & heights
rc_pos = rc_ax.get_position()  # Bbox(x0, y0, x1, y1)
hm_pos = hm_ax.get_position()

rc_width  = rc_pos.width
rc_height = rc_pos.height

# compute a new x0 so that rc_ax.x1 == hm_ax.x0
new_x0 = hm_pos.x0 - rc_width - 0.001  # subtract a tiny gap if you like

# set rc_ax to that position
rc_ax.set_position([new_x0, rc_pos.y0, rc_width, rc_height])
rc_ax.xaxis.set_ticks_position('none')
# 6) (Optional) Legend for the chapter colors
from matplotlib.patches import Patch

# 1) Compute which chapters actually appear in your rows
used_chapters = {
    atc_first_level_map[first_letter(lbl)]
    for lbl in heatmap_df.index
}

# 2) Build legend handles just for those
handles = [
    Patch(facecolor=chapter_to_color[ch], label=ch)
    for ch in sorted(used_chapters)
]
cax = cg.cax
max_val = heatmap_df.values.max()

cax.set_xticks([0, max_val])
cax.set_xticklabels(['0', f'{max_val:.2f}'])
handles = [Patch(facecolor=chapter_to_color[ch], label=ch)
           for ch in used_chapters]
leg = cg.ax_col_dendrogram.legend(
    handles=handles,
    title = 'ATC Categories',
    bbox_to_anchor=(1.02, 1),
    loc='lower left',
    frameon=True              # turn the box on
)
ax = cg.ax_heatmap
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
labels = ax.get_yticklabels()
def wrap_label(label, max_words=8):
    words = label.split()
    if len(words) > max_words:
        # first line: first max_words words
        # second line: the rest
        return " ".join(words[:max_words]) + "\n" + " ".join(words[max_words:])
    return label
#print(labels)
# wrap any long labels
wrapped_texts = [wrap_label(txt.get_text(),4) for txt in labels]

# then set them
ax.set_yticklabels(wrapped_texts, rotation=0)

# 6) Style the box (edge‐color, line‐width, face‐color)
frame = leg.get_frame()
frame.set_edgecolor('grey')
frame.set_linewidth(1.0)
frame.set_facecolor('white')
plt.show()

exp_n_opcs = torch.load("results/toy_exp_n_opcs_4.pt", map_location=device)
exp_n_opcs = exp_n_opcs.cpu().detach().numpy()
V, K = exp_n_opcs.shape
beta = 0.1
phi_r_opcs = (beta + exp_n_opcs) / (beta*V + exp_n_opcs.sum(axis=1, keepdims=1))
phi_r_opcs = phi_r_opcs / phi_r_opcs.sum(axis=0, keepdims=1) # normalization over V for each topic
#phi_r_exp = np.exp(phi_r)
#phi_r_softmax = phi_r_exp / phi_r_exp.sum(axis=0, keepdims=1)
inv_phecode_ids = {v: k for k, v in phecode_ids.items()}
opcs_ids = pkl.load(open("mapping/opcs_vocab_ids.pkl", "rb"))
inv_opcs_ids = {v: k for k, v in opcs_ids.items()}
opcs_meaning = pd.read_csv("opcs_definition.tsv", delimiter = '\t')

pheno_meaning_dict = dict(zip(pheno_meaning['phecode'], pheno_meaning['phenotype']))
df_full_opcs = pd.DataFrame(phi_r_opcs)
pheno_meaning_list = []
opcs_meaning_list = []
for i in range(K):
    pheno_meaning_list.append(str(inv_phecode_ids[i]) + ' ' + pheno_meaning_dict[inv_phecode_ids[i]])
for i in range(V):
    opcs_meaning_list.append(inv_opcs_ids[i])
    # pheno_meaning_list.append(pheno_meaning_dict[inv_phecode_ids[i]])
df_full_opcs.columns = pheno_meaning_list
df_full_opcs.index = opcs_meaning_list

phenotypes_df = df_full_opcs.iloc[:, phecode_index]

# Find the top 5 genes for each phenotype and collect their indices.
top_genes_indices = []
for col in phenotypes_df.columns:
    top_genes = phenotypes_df[col].nlargest(3).index
    top_genes_indices.extend(top_genes)

heatmap_df = phenotypes_df.loc[top_genes_indices,].drop_duplicates()

# 1) Your ICD-10 chapter map by first letter
opcs4_chapter_map = {
    'A': 'Nervous System',
    'B': 'Endocrine System and Breast',
    'C': 'Eye',
    'D': 'Ear',
    'E': 'Respiratory Tract',
    'F': 'Mouth',
    'G': 'Upper Digestive System',
    'H': 'Lower Digestive System',
    'J': 'Other Abdominal Organs, Principally Digestive',
    'K': 'Heart',
    'L': 'Arteries and Veins',
    'M': 'Urinary',
    'N': 'Male Genital Organs',
    'P': 'Lower Female Genital Tract',
    'Q': 'Upper Female Genital Tract',
    'R': 'Female Genital Tract Associated with Pregnancy, Childbirth and the Puerperium',
    'S': 'Skin',
    'T': 'Soft Tissue',
    'U': 'Diagnostic Imaging, Testing and Rehabilitation',
    'V': 'Bones and Joints of Skull and Spine',
    'W': 'Other Bones and Joints',
    'X': 'Miscellaneous Operations',
    'Y': 'Subsidiary Classification of Methods of Operation',
    'Z': 'Subsidiary Classification of Sites of Operation',
    'O': 'Overflow codes'
    # Overflow codes (“O‐codes”) are treated as part of the chapter denoted in parentheses :contentReference[oaicite:3]{index=3}
}



# 2) Pick a color palette for all chapters
chapters = sorted(set(opcs4_chapter_map.values()))
palette  = sns.color_palette("Set3", len(chapters))
chapter_to_color = dict(zip(chapters, palette))

def first_letter(label):
    return label.strip().split()[0][0].upper()

used_chapters = {
    opcs4_chapter_map[first_letter(lbl)]
    for lbl in heatmap_df.index
}

# 2) Generate a palette sized to exactly len(used_chapters)
palette = sns.color_palette("Paired", n_colors=len(used_chapters))
used_chapters = sorted(used_chapters)           # so the order is consistent
chapter_to_color = dict(zip(used_chapters, palette))

# 3) Build your row_colors Series
row_colors = pd.Series(
    [ chapter_to_color[ opcs4_chapter_map[first_letter(lbl)] ]
      for lbl in heatmap_df.index ],
    index=heatmap_df.index
)
sns.set_context("notebook", font_scale=1) 
# 5) Draw your clustermap with that side-bar
cg = sns.clustermap(
    heatmap_df,
    cmap="BuPu",
    figsize=(4, 9), 
    linewidths=1, linecolor='white',
    row_cluster=False, col_cluster=False,
    row_colors=row_colors,                # ← this adds the left color bar
    colors_ratio=(0.08, 0.03), 
    cbar_kws={
        'orientation':'horizontal',
        'shrink':0.25,
        'pad':0.01
    },
    cbar_pos=(1, 0.01, 0.25, 0.02)
)

rc_ax = cg.ax_row_colors
hm_ax = cg.ax_heatmap

# grab their current widths & heights
rc_pos = rc_ax.get_position()  # Bbox(x0, y0, x1, y1)
hm_pos = hm_ax.get_position()

rc_width  = rc_pos.width
rc_height = rc_pos.height

# compute a new x0 so that rc_ax.x1 == hm_ax.x0
new_x0 = hm_pos.x0 - rc_width - 0.001  # subtract a tiny gap if you like

# set rc_ax to that position
rc_ax.set_position([new_x0, rc_pos.y0, rc_width, rc_height])
rc_ax.xaxis.set_ticks_position('none')
# 6) (Optional) Legend for the chapter colors
from matplotlib.patches import Patch

# 1) Compute which chapters actually appear in your rows
used_chapters = {
    opcs4_chapter_map[first_letter(lbl)]
    for lbl in heatmap_df.index
}

# 2) Build legend handles just for those
handles = [
    Patch(facecolor=chapter_to_color[ch], label=ch)
    for ch in sorted(used_chapters)
]

# 3) Add the legend to your clustermap
cg.ax_col_dendrogram.legend(
    handles=handles,
    title="ICD-10 Chapter",
    bbox_to_anchor=(1.02, 1),
    loc='upper right',
    frameon=False
)
cax = cg.cax
max_val = heatmap_df.values.max()
cax.set_xticks([0, max_val])
cax.set_xticklabels(['0', f'{max_val:.2f}'])
cax.xaxis.set_ticks_position('none')

handles = [Patch(facecolor=chapter_to_color[ch], label=ch)
           for ch in used_chapters]
leg = cg.ax_col_dendrogram.legend(
    handles=handles,
    title = 'OPCS4 Categories',
    bbox_to_anchor=(1.02, 1),
    loc='lower left',
    frameon=True              # turn the box on
)
ax = cg.ax_heatmap
#ax.yaxis.tick_left()
# 3) Add padding so labels clear the color‐bar
ax.tick_params(axis='y', left=False, pad = 15)
ax.yaxis.set_ticks_position('none')
ax.xaxis.set_ticks_position('none')
labels = ax.get_yticklabels()
def wrap_label(label, max_words=8):
    words = label.split()
    if len(words) > max_words:
        # first line: first max_words words
        # second line: the rest
        return " ".join(words[:max_words]) + "\n" + " ".join(words[max_words:])
    return label
wrapped_texts = [wrap_label(txt.get_text(),8) for txt in labels]

# then set them
ax.set_yticklabels(wrapped_texts, rotation=0)

# 6) Style the box (edge‐color, line‐width, face‐color)
frame = leg.get_frame()
frame.set_edgecolor('grey')
frame.set_linewidth(1.0)
frame.set_facecolor('white')
plt.show()

