
from tqdm import tqdm

import numpy as np
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as sch

from sklearn.preprocessing import StandardScaler as ss
from sklearn.decomposition import PCA

import pandas as pd

from minepy import cstats
from pygam import LinearGAM, s

import matplotlib.pyplot as plt
import seaborn as sns

from .utilities import group_wise_binning, generate_dir


def visualize_auto_elbow(n_clusters, inertias, grad_line, optimal_k,
                         save_name):
    """
    """
    plot_path = 'results\\figures\\clusterer\\auto_elbow'
    generate_dir(plot_path)

    optimal_index = np.where(np.array(n_clusters) == optimal_k)
    optimal_index = optimal_index[0][0]

    plt.figure(figsize=(5, 5))
    # draw the traditional elbow plot
    plt.plot(
        n_clusters,
        inertias,
        marker='x',
        color='k'
    )
    # draw the overall gradient of the inertia with respect to the
    # number of clusters
    plt.plot(
        [n_clusters[0], n_clusters[-1]],
        [inertias[0], inertias[-1]],
        color='r',
        linestyle='--',
        alpha=0.5
    )
    # highlight the point of maximum curvature
    plt.vlines(
        optimal_k,
        inertias[optimal_index],
        grad_line[optimal_index],
        color='r',
        linestyle='--'
    )

    # plt.title(save_name)
    plt.ylabel('Inertia')
    plt.xlabel('Number of Partitions')
    plt.savefig(f'{plot_path}\\{save_name}.svg')
    plt.close()
    return None


def visualize_full_panel(reduction, model_name, contexts, context_remap,
                         colors_dict, colors_remap, colors_name,
                         save_name, snapshot, cmapper, figsize=(15, 6),
                         save_path='results\\figures\\embeddings\\full_panel',
                         subsample_ratio=1., visual_verbose=False, **kwargs):
    """
    """
    if subsample_ratio < 1:
        subsample_index = np.random.choice(
            [i for i in range(reduction.shape[0])],
            int(reduction.shape[0] * subsample_ratio),
            replace=False
        )
        reduction = reduction[subsample_index, :]
        contexts = contexts[subsample_index]
        colors_dict = {
            color_name: color_values[subsample_index] for color_name,
            color_values in colors_dict.items()
        }

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=10, nrows=2)

    ax_context = fig.add_subplot(spec[:, :4])
    axs_metrics = [
        fig.add_subplot(spec[0, 4:6]),
        fig.add_subplot(spec[0, 6:8]),
        fig.add_subplot(spec[0, 8:]),
        fig.add_subplot(spec[1, 5:7]),
        fig.add_subplot(spec[1, 7:9])
    ]

    for unique_context in np.unique(contexts):

        label = context_remap[int(unique_context)]
        context_index = np.argwhere(contexts == unique_context).flatten()
        c = cmapper(int(unique_context))
        ax_context.scatter(
            reduction[:, 0][context_index],
            reduction[:, 1][context_index],
            marker='o',
            edgecolor='',
            color=c,
            label='Object {}'.format(label),
            **kwargs
        )
    ax_context.set_ylabel('')
    ax_context.set_xlabel('')
    ax_context.set_title(f'Game Context - $t$ {snapshot}')

    index = 0
    for color_name, ax_metric in zip(colors_name, axs_metrics):

        img = ax_metric.scatter(
            reduction[:, 0],
            reduction[:, 1],
            c=colors_dict[color_name],
            marker='o',
            edgecolor='',
            cmap='coolwarm',
            # vmin=0,
            # vmax=100,
            **kwargs
        )
        ax_metric.set_title(
            f'{colors_remap[color_name]} - $t$ {snapshot}'
        )
        if index > 2:
            ax_metric.set_yticks([])
            ax_metric.set_xlabel('')
        else:
            ax_metric.set_yticks([])
            ax_metric.set_xticks([])
        index += 1

    fig.text(0.5, -0.01, 'Dimension 1', ha='center')
    fig.text(-0.01, 0.5, 'Dimension 2', va='center', rotation='vertical')
    handles, labels = ax_context.get_legend_handles_labels()
    leg = ax_context.legend(
        handles,
        labels,
        markerscale=8,
        ncol=1
    )
    leg.get_frame().set_edgecolor('k')

    plt.suptitle(f"Representation Model {model_name}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    cbaxes = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(
        img,
        cax=cbaxes,
        cmap='coolwarm',
        boundaries=np.linspace(0, 100, 100),
        ticks=[0, 25, 50, 75, 100]
    )
    cbar.set_label('Discretized Metric Value')

    plt.savefig(
        f'{save_path}\\{model_name}_{save_name}.png',
        dpi=100,
        bbox_inches='tight'
    )
    if visual_verbose:
        plt.show()

    return None


def visualize_temporal_panel(
        temporal_colors, temporal_contexts, context_remap, color_name, cmapper,
        model_name, model_remap, snapshots=[0, 1, 2, 3], reduction_type='umap',
        subsample_ratio=1., figsize=(12, 6), binning_method=None,
        visual_verbose=False,
        save_path='results\\figures\\embeddings\\temporal_panel', **kwargs):
    """
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=len(snapshots), nrows=2)

    for column, snapshot in enumerate(snapshots):

        reduction = np.load(
            f'results\\saved_dim_reduction\\2D\\{reduction_type}_{model_name}_eng_emb_{snapshot}.npy'
        )
        reduction = reduction[~np.isnan(reduction).any(axis=1)]
        if subsample_ratio < 1.:
            subsample_index = np.random.choice(
                [i for i in range(reduction.shape[0])],
                int(reduction.shape[0] * subsample_ratio),
                replace=False
            )
            reduction = reduction[subsample_index, :]
            temporal_context = temporal_contexts[snapshot][subsample_index]
            temporal_color = temporal_colors[snapshot][subsample_index]
        else:
            temporal_context = temporal_contexts[snapshot]
            temporal_color = temporal_colors[snapshot]

        # Plot context values
        ax_context = fig.add_subplot(spec[0, column])
        for unique_context in np.unique(temporal_context):

            label = context_remap[int(unique_context)]
            context_index = np.argwhere(
                temporal_context == unique_context
            ).flatten()
            c = cmapper(int(unique_context))
            ax_context.scatter(
                reduction[:, 0][context_index],
                reduction[:, 1][context_index],
                marker='o',
                edgecolor='',
                color=c,
                label=f'Object {label}',
                **kwargs
            )
        if column == 0:
            ax_context.set_xlabel('')
            ax_context.set_ylabel('')
        else:
            ax_context.set_xlabel('')
            ax_context.set_yticks([])
        ax_context.set_title(f'Game Context - $t$ {snapshot+1}')

        # Plot the metric values

        colors = group_wise_binning(
            array=temporal_color,
            grouper=temporal_context,
            n_bins=100,
            method=binning_method
        )

        ax_snapshot = fig.add_subplot(spec[1, column])
        img = ax_snapshot.scatter(
            reduction[:, 0],
            reduction[:, 1],
            c=colors,
            marker='o',
            edgecolor='',
            # vmin=0,
            # vmax=100,
            cmap='coolwarm',
            **kwargs
        )
        ax_snapshot.set_title(f'{color_name} - $t$ {snapshot+1}')
        if column == 0:
            ax_snapshot.set_xlabel('')
            ax_snapshot.set_ylabel('')
        else:
            ax_snapshot.set_xlabel('')
            ax_snapshot.set_yticks([])

    fig.text(0.5, -0.01, 'Dimension 1', ha='center')
    fig.text(-0.01, 0.5, 'Dimension 2', va='center', rotation='vertical')
    plt.suptitle(f"Representation Model {model_remap[model_name]}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    handles, labels = ax_context.get_legend_handles_labels()
    leg = ax_context.legend(
        handles,
        labels,
        markerscale=15,
        ncol=1,
        loc='center left',
        bbox_to_anchor=(1, 0.5)
    )
    leg.get_frame().set_edgecolor('k')

    cbaxes = fig.add_axes([1.0, 0.06, 0.02, 0.40])
    cbar = fig.colorbar(
        img,
        cax=cbaxes,
        cmap='coolwarm',
        boundaries=np.linspace(0, 100, 100),
        ticks=[0, 25, 50, 75, 100]
    )
    cbar.set_label('Discretized Metric Value')

    plt.savefig(
        f'{save_path}\\{model_name}_{color_name}.png',
        dpi=300,
        bbox_inches='tight'
    )
    if visual_verbose:
        plt.show()

    return None


def visualize_neurons_function(data, metric, snapshot, metric_rmp,
                               neurons, figsize=(3, 3), visually_verbose=False):
    """
    """
    for neuron in neurons:

        plt.figure(figsize=figsize)
        mean = data[
            data['Artificial Neurons'] == neuron
        ].groupby('Signal')[metric].mean().reset_index()
        mean = mean.dropna()

        sem = data[
            data['Artificial Neurons'] == neuron
        ].groupby('Signal')[metric].sem().reset_index()

        mic, tic = cstats(
            mean['Signal'].values.reshape(1, -1),
            mean[metric].values.reshape(1, -1)

        )
        mic = round(mic.flatten()[0], 3)

        rho, p = spearmanr(
            mean['Signal'].values,
            mean[metric].values
        )
        rho = round(rho, 3)

        gam = LinearGAM(
            s(0),
            fit_intercept=False
        )
        gam.fit(
            mean['Signal'].values,
            mean[metric].values
        )
        grid = gam.generate_X_grid(term=0)
        pdep, confi = gam.partial_dependence(term=0, X=grid, width=0.95)

        plt.scatter(
            mean['Signal'].values,
            mean[metric].values,
            s=5,
            c='k'
        )
        plt.errorbar(
            mean['Signal'].values,
            mean[metric].values,
            yerr=sem[metric],
            ls='none',
            c='k'
        )
        plt.plot(
            grid[:, 0],
            pdep,
            label=f'MIC {mic} \n $ \\rho\\: {rho}$',
            c='r'
        )
        plt.xlabel('Discretized Activation')
        plt.ylabel(f'{metric_rmp[metric]}')
        plt.title(f'Artificial Neuron {neuron} - $t$ {snapshot}')
        plt.legend()

        plt.savefig(
            f'results\\figures\\embeddings\\neurons_functions\\{metric}_{neuron}_{snapshot}.png',
            dpi=100,
            bbox_inches='tight'
        )
        if visually_verbose:
            plt.show()
        plt.close('all')

    return None


def visualize_temporal_corr(data_container, context_mapper, context=None,
                            snapshots=[0, 1, 2, 3], thresh=0.5, mask=0.1, visually_verbose=False):
    """
    """
    fig, axs = plt.subplots(
        1, len(snapshots),
        figsize=(16, 4), sharex=True, sharey=True)
    for snapshot in tqdm(snapshots):

        emb = np.load(
            f'results\\saved_emb\\melchior_eng_emb_{snapshot}.npy'
        )
        emb = emb[~np.isnan(emb).any(axis=1)]
        if context is not None:
            cont = data_container["context"][snapshot]
            idx_cont = np.argwhere(cont == context).flatten()
            emb = emb[idx_cont, :]
        df = pd.DataFrame(emb)
        d = df.corr(method="pearson").fillna(0).values

        d = sch.distance.pdist(d)
        link = sch.linkage(d, method='single')
        clust_ind = sch.fcluster(link, thresh*d.max(), 'distance')
        columns = [
            df.columns.tolist()[i] for i in list((np.argsort(clust_ind)))
        ]
        df = df.reindex(columns, axis=1)

        corr = df.corr()
        mask = abs(corr) < 0.1
        corr[mask] = np.nan

        img = axs[snapshot].matshow(
            corr,
            cmap='coolwarm',
            vmin=-1,
            vmax=1
        )

        axs[snapshot].set_title(
            f"Cross-Correlation - $t {snapshot+1}$"
        )
        if snapshot == 0:
            axs[snapshot].set_ylabel("Artificial Neurons")
        axs[snapshot].set_xlabel("Artificial Neurons")
        axs[snapshot].set_yticklabels([])
        axs[snapshot].set_xticklabels([])

    cbaxes = fig.add_axes([1.0, 0.06, 0.02, .8])
    cbar = fig.colorbar(
        img,
        cax=cbaxes,
        cmap='coolwarm',
        boundaries=np.linspace(-1, 1, 100),
        ticks=[-1, -0.5, 0, 0.5, 1]
    )
    cbar.set_label("Spearman's Rho")
    plt.suptitle(f"Context {context_mapper[context]}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(
        f'results\\figures\\embeddings\\neurons_corr\\cont_{context_mapper[context]}.png',
        dpi=300,
        bbox_inches='tight'
    )
    if visually_verbose:
        plt.show()
    return None


def visualize_game_specific_panel(
    embed,
    context,
    context_remap,
    colors_dict,
    colors_remap,
    colors_name,
    save_name,
    snapshot,
    cmapper,
    figsize=(15, 6),
    scatter_cmapper="coolwarm",
    scatter_cmap_name="Discretized Value",
    vmin_sc=0,
    vmax_sc=100,
    save_path="results\\figures\\embeddings\\full_panel",
    visual_verbose=False,
    **kwargs,
):
    """Short summary.

    Args:
        embed (type): Description of parameter `embed`.
        context (type): Description of parameter `context`.
        context_remap (type): Description of parameter `context_remap`.
        colors_dict (type): Description of parameter `colors_dict`.
        colors_remap (type): Description of parameter `colors_remap`.
        colors_name (type): Description of parameter `colors_name`.
        save_name (type): Description of parameter `save_name`.
        snapshot (type): Description of parameter `snapshot`.
        cmapper (type): Description of parameter `cmapper`.
        figsize (type): Description of parameter `figsize`.
        scatter_cmapper (type): Description of parameter `scatter_cmapper`.
        scatter_cmap_name (type): Description of parameter `scatter_cmap_name`.
        vmin_sc (type): Description of parameter `vmin_sc`.
        vmax_sc (type): Description of parameter `vmax_sc`.
        save_path (type): Description of parameter `save_path`.
        visual_verbose (type): Description of parameter `visual_verbose`.
        **kwargs (type): Description of parameter `**kwargs`.

    Returns:
        type: Description of returned object.

    """
    embed = ss().fit_transform(embed)

    reducer = PCA(n_components=2, whiten=True).fit(embed)
    variance_exp = reducer.explained_variance_ratio_
    variance_exp = np.around(variance_exp, 2) * 100
    variance_exp = variance_exp.astype(int)

    reduction = reducer.transform(embed)

    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=10, nrows=2)

    ax_context = fig.add_subplot(spec[:, :4])
    axs_metrics = [
        fig.add_subplot(spec[0, 4:6]),
        fig.add_subplot(spec[0, 6:8]),
        fig.add_subplot(spec[0, 8:]),
        fig.add_subplot(spec[1, 5:7]),
        fig.add_subplot(spec[1, 7:9]),
    ]

    c = cmapper(int(context))
    ax_context.scatter(
        reduction[:, 0],
        reduction[:, 1],
        marker="o",
        edgecolor="",
        color=c,
        label="Object {}".format(context_remap[context]),
        **kwargs,
    )
    ax_context.set_ylabel("")
    ax_context.set_xlabel("")
    ax_context.set_title(f"Game Context - $t$ {snapshot}")

    index = 0
    for color_name, ax_metric in zip(colors_name, axs_metrics):

        img = ax_metric.scatter(
            reduction[:, 0],
            reduction[:, 1],
            c=colors_dict[color_name],
            marker="o",
            edgecolor="",
            cmap=scatter_cmapper,
            vmin=vmin_sc,
            vmax=vmax_sc,
            **kwargs,
        )
        ax_metric.set_title(f"{colors_remap[color_name]} - $t$ {snapshot}")
        if index > 2:
            ax_metric.set_yticks([])
            ax_metric.set_xlabel("")
        else:
            ax_metric.set_yticks([])
            ax_metric.set_xticks([])
        index += 1

    fig.text(
        0.5,
        -0.01,
        f"Component 1 - Explained Variance {variance_exp[0]}%",
        ha="center",
    )
    fig.text(
        -0.01,
        0.5,
        f"Component 2 - Explained Variance {variance_exp[1]}%",
        va="center",
        rotation="vertical",
    )
    handles, labels = ax_context.get_legend_handles_labels()
    leg = ax_context.legend(handles, labels, markerscale=8, ncol=1)
    leg.get_frame().set_edgecolor("k")

    plt.tight_layout()

    cbaxes = fig.add_axes([1.0, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(
        img,
        cax=cbaxes,
        cmap="coolwarm",
        boundaries=np.linspace(vmin_sc, vmax_sc, 100),
        ticks=np.linspace(vmin_sc, vmax_sc, 5),
    )
    cbar.set_label(scatter_cmap_name)

    plt.savefig(
        f"{save_path}\\game_specific_{save_name}.png",
        dpi=100,
        bbox_inches="tight",
    )
    if visual_verbose:
        plt.show()

    return None
