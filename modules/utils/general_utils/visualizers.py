import numpy as np

import matplotlib.pyplot as plt

from minepy import cstats
from scipy.stats import spearmanr
from pygam import LinearGAM, s

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


def visualize_full_panel(reduction, contexts, context_remap,
                         colors_dict, colors_remap, colors_name,
                         save_name, snapshot, cmapper, figsize=(15, 6),
                         save_path='results\\figures\\embeddings\\full_panel',
                         visual_verbose=True, **kwargs):
    """
    """
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
            vmin=0,
            vmax=100,
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

    plt.tight_layout()

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
        f'{save_path}\\{save_name}.png',
        dpi=500,
        bbox_inches='tight'
    )
    if visual_verbose:
        plt.show()

    return None


def visualize_temporal_panel(
        temporal_color, temporal_contexts, context_remap, color_name, cmapper,
        snapshots=[1, 2, 3], reduction_type='umap',
        figsize=(9, 6), binning_method=None, visual_verbose=True,
        save_path='results\\figures\\embeddings\\temporal_panel', **kwargs):
    """
    """
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    spec = fig.add_gridspec(ncols=len(snapshots), nrows=2)

    for column, snapshot in enumerate(snapshots):

        reduction = np.load(
            f'results\\saved_dim_reduction\\2D\\{reduction_type}_melchior_eng_emb_{snapshot}.npy'
        )
        reduction = reduction[~np.isnan(reduction).any(axis=1)]

        # Plot context values
        ax_context = fig.add_subplot(spec[0, column])
        for unique_context in np.unique(temporal_contexts[snapshot]):

            label = context_remap[int(unique_context)]
            context_index = np.argwhere(
                temporal_contexts[snapshot] == unique_context
            ).flatten()
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
        if column == 0:
            ax_context.set_xlabel('')
            ax_context.set_ylabel('')
        else:
            ax_context.set_xlabel('')
            ax_context.set_yticks([])
        ax_context.set_title(f'Game Context - $t$ {snapshot+1}')

        # Plot the metric values

        colors = group_wise_binning(
            array=temporal_color[snapshot],
            grouper=temporal_contexts[snapshot],
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
            vmin=0,
            vmax=100,
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
    plt.tight_layout()

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
        f'{save_path}\\{color_name}.png',
        dpi=500,
        bbox_inches='tight'
    )
    if visual_verbose:
        plt.show()

    return None


def visualize_neurons_function(data, metric, snapshot, metric_rmp,
                               neurons, figsize=(3, 3)):
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
            dpi=500,
            bbox_inches='tight'
        )
        plt.close('all')

    return None
