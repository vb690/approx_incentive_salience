import pickle

import numpy as np
import pandas as pd

from umap import aligned_umap

from modules.utils.general_utils.embedding_handlers import create_relationships
from modules.utils.general_utils.utilities import group_wise_binning

###############################################################################

# define the targets we are interested in and remappers for better naming
TARGETS = [
    'tar_sessions',
    'tar_delta_sessions',
    'tar_active_time',
    'tar_session_time',
    'tar_activity'
]
TARGETS_RMP = {
    'tar_delta_sessions': 'Future Absence',
    'tar_active_time': 'Future Active Time',
    'tar_session_time': 'Future Session Time',
    'tar_activity': 'Future Session Activity',
    'tar_sessions': 'Future N° Sessions'
}

T_STEPS = 5

# we get the data container
with (open('results\\saved_data_containers\\rnn.pkl', 'rb')) as container:
    DATA_CONTAINER = pickle.load(container)

# this ensure we allign the embedding of each user across time
relationships = create_relationships(
    users=DATA_CONTAINER['user_id'],
    t_steps=T_STEPS - 1
)

# load the embeddings and drop the nans
embeddings = [
    np.load(f'results\\saved_emb\\rnn_emb_{t}.npy') for t in range(T_STEPS)
]
embeddings = [
    embedding[~np.isnan(embedding).any(axis=1)] for embedding in embeddings
]

###############################################################################

print('Starting AlignedUMAP transformation')

transformer = aligned_umap.AlignedUMAP(
    metric='cosine',
    n_neighbors=100,
    alignment_regularisation=0.1,
    alignment_window_size=2,
    n_epochs=1000,
    min_dist=0.8,
    random_state=42,
    verbose=True
)
transformer.fit(
    embeddings,
    relations=relationships
)

print('Finished AlignedUMAP transformation')

###############################################################################

# we build a df with the otained embedding for each user alligned across time
df = pd.DataFrame(
    np.vstack(transformer.embeddings_),
    columns=('UMAP_1', 'UMAP_2')
)
df['session'] = np.concatenate(
    [[session] * len(embeddings[session]) for session in range(T_STEPS)]
)
df['user_id'] = np.hstack(
    [DATA_CONTAINER['user_id'][session] for session in range(T_STEPS)]
)
df['context'] = np.hstack(
    [DATA_CONTAINER['context'][session] for session in range(T_STEPS)]
)

# we add for each embedding the associated target
for target in TARGETS:

    colors = []

    for session in range(5):

        binned_array = group_wise_binning(
            array=DATA_CONTAINER['prediction_ds'][target][session],
            grouper=DATA_CONTAINER['context'][session],
            n_bins=100,
            method='discret'
        )
        colors.append(binned_array)

    colors = np.hstack(colors)
    df[TARGETS_RMP[target]] = colors

df.to_csv('results\\saved_dim_reduction\\rnn_emb_temporal.csv')

###############################################################################
