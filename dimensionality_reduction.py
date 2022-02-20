from sklearn.decomposition import PCA

from umap import UMAP

from modules.utils.general_utils.embedding_handlers import reduce_dimensions

##############################################################################

# we specify the number of temporal steps we want to embed
SNAPSHOTS = [snapshot for snapshot in range(10)]
COMPONENTS = [2, 3, 10]
PATH = 'results\\saved_emb\\'
NAME = 'td_mlp_eng_emb'

reduce_dimensions(
    reducer={'name': 'pca', 'algo': PCA},
    path=PATH,
    name=NAME,
    snapshots=SNAPSHOTS,
    n_components=2,
    context_aware=True
)

reduce_dimensions(
    reducer={'name': 'umap', 'algo': UMAP},
    path=PATH,
    name=NAME,
    snapshots=SNAPSHOTS,
    n_components=10,
    verbose=True,
    n_neighbors=30,
    n_epochs=1000,
    min_dist=0,
    metric='cosine',
    context_aware=True
)

for componets in COMPONENTS:

    reduce_dimensions(
        reducer={'name': 'pca', 'algo': PCA},
        path=PATH,
        name=NAME,
        snapshots=SNAPSHOTS,
        n_components=componets,
    )

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
#
