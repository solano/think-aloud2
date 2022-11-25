# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 13:39:14 2022

Uses preprocessed phrases to produce trajectories in embedding space.
Phrases are embedded at the sub-row, row and probe level.

@author: Solano Felicio
"""

# %% Import modules

from laserembeddings import Laser
import pandas as pd
import numpy as np

# %% Create LASER object

laser = Laser()

# %% Read data

df_rows = pd.read_csv("text_rows.csv", sep="\t")
df_probes = pd.read_csv("text_probes.csv", sep="\t")
df_subrows = pd.read_csv("text_subrows.csv", sep="\t")

# %% Embedding at row level

row_embeddings = laser.embed_sentences(df_rows.SPEECH, lang="fr")
# Runtime: 1 min 20 s

np.save("row_embeddings.npy", row_embeddings)

# %% Embedding at probe level

probe_embeddings = laser.embed_sentences(df_probes.SPEECH, lang="fr")
# Runtime: 1 min 30 s

np.save("probe_embeddings.npy", probe_embeddings)

# %% Embedding at subrow level

subrow_embeddings = laser.embed_sentences(df_subrows.SPEECH, lang="fr")
# Runtime: 1 min

np.save("subrow_embeddings.npy", subrow_embeddings)