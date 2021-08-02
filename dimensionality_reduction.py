from umap import UMAP

from modules.utils.general_utils.embedding_handlers import reduce_dimensions

##############################################################################

# we specify the number of temporal steps we want to embed
SNAPSHOTS = [snapshot for snapshot in range(10)]
PATH = 'results\\saved_emb\\'
NAME = 'rnn_emb'

for componets in [2, 10, 3]:

    reduce_dimensions(
        reducer={'name': 'umap', 'algo': UMAP},
        path=PATH,
        name=NAME,
        snapshots=SNAPSHOTS,
        n_components=componets,
        verbose=True,
        n_neighbors=100,
        n_epochs=1000,
        min_dist=0.80,
        metric='cosine'
    )
