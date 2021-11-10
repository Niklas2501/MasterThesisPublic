import argparse
import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from configuration.Configuration import Configuration
from configuration.Enums import Representation
from stgcn.Dataset import Dataset
import matplotlib as mlp


def get_color_list():
    color_list = [
        "#909090",  # light gray
        "#890000",  # dark red
        "#ff0000",  # red
        "#781897",  # dark purple
        "#ff00fb",  # pink
        "#ffcfff",  # light purple
        "#ffa200",  # orange
        "#fff300",  # yellow
        "#55ff00",  # light green
        "#00830e",  # dark green
        "#8dcc93",  # matt green
        "#e4e59e",  # matt yellow
        "#00fffb",  # cyan
        "#001fff",  # dark blue
        "#5284b3",  # matt blue
        "#1b7276",  # matt cyan
        "#373737",  # dark gray
        "#ffffff",  # whit
    ]
    return color_list


# Based on: https://stackoverflow.com/a/42516941/14648532
def discrete_cmap(N, base_cmap=None):
    # base = plt.cm.get_cmap(base_cmap)
    # color_list = base(np.linspace(0, 1, N))
    # # Set first color (will be no_failure) to grey
    # color_list[0] = np.array([0.5, 0.5, 0.5, 1])
    # cmap_name = base.name + str(N)
    # base.from_list(cmap_name, color_list, N)

    color_list = get_color_list()

    # Reduce the list to the number of colors needed.
    if N < len(color_list):
        color_list = color_list[0:N]

    # Convert to a hex color object required by the ListedColormap.
    color_list = [mlp.colors.hex2color(c) for c in color_list]

    return ListedColormap(color_list, name='OrangeBlue')


def visualise_tsne(config: Configuration, for_rocket, high_res):
    np.random.seed(config.random_seed)
    dataset = Dataset(config)
    dataset.load()
    vis_out_emb = np.empty(1)
    labels = dataset.get_train_val().get_y_strings()

    if for_rocket and not config.representation in [Representation.ROCKET, Representation.MINI_ROCKET]:
        raise ValueError('Rocket representation must be selected in the configuration.')

    try:
        if for_rocket:
            vis_out_emb = np.load(config.get_training_data_path() + 'vis_out_emb.npy')
        else:
            vis_out_emb = np.load(config.directory_model_to_use + 'vis_out_emb.npy')
    except FileNotFoundError:
        print('Required files not saved for the configured model.')
        exit(0)

    unique_labels = dataset.unique_labels_overall
    nbr_classes = dataset.num_classes

    # Split examples into failures and normal state for different plotting.
    nf_mask, f_mask = labels == 'no_failure', labels != 'no_failure'
    nf_emb, nf_labels = vis_out_emb[nf_mask], labels[nf_mask]
    f_emb, f_labels = vis_out_emb[f_mask], labels[f_mask]

    fig = plt.figure(figsize=(15, 10)) if high_res else plt.figure(figsize=(9, 7))

    ax = fig.add_subplot()
    le = LabelEncoder()
    le.fit(unique_labels)
    f_labels_int = le.transform(f_labels)
    c_bar_labels = le.inverse_transform(np.arange(nbr_classes))

    if high_res:
        c_bar_labels = [label[0:22] + '...' if len(label) > 25 else label for label in c_bar_labels]
    else:
        c_bar_labels = [label[0:12] + '...' if len(label) > 15 else label for label in c_bar_labels]

    # Ensure no failure is the first entry such that the plotting matches the color map
    c_bar_labels.remove('no_failure')
    c_bar_labels = ['no_failure'] + c_bar_labels

    # Create scatter plots. First for no_failure and then overlay with failure examples.
    nf_scat = ax.scatter(nf_emb[:, 0], nf_emb[:, 1], c=get_color_list()[0], s=25, edgecolors='#525252')
    f_scat = ax.scatter(f_emb[:, 0], f_emb[:, 1], c=f_labels_int, s=45, edgecolors='black',
                        cmap=discrete_cmap(nbr_classes, 'jet'))

    if high_res:
        cbar = plt.colorbar(f_scat, ticks=np.arange(nbr_classes), pad=0.05, orientation="horizontal")
        cbar.ax.tick_params(rotation=75, labelsize=8)
    else:
        cbar = plt.colorbar(f_scat, ticks=np.arange(nbr_classes), pad=0.03, orientation="horizontal")
        cbar.ax.tick_params(rotation=65, labelsize=6)

    cbar.set_ticklabels(c_bar_labels)
    f_scat.set_clim(vmin=-0.5, vmax=nbr_classes - 0.5)
    ax.axis('tight')

    # ax.xaxis.set_major_formatter(NullFormatter())
    # ax.yaxis.set_major_formatter(NullFormatter())
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                    labelleft=False, labelbottom=False)

    file_path = config.directory_model_to_use + 'tsne.pdf' if not for_rocket else '../logs/rocket_tsne.pdf'
    plt.savefig(file_path, dpi=200, bbox_inches='tight')

    print(f'Saved generated plot @ {file_path}')


def fit_tsne(config: Configuration, pca_components, for_rocket):
    dataset = Dataset(config)
    dataset.load()
    vis_out = np.empty(1)

    if for_rocket and not config.representation in [Representation.ROCKET, Representation.MINI_ROCKET]:
        raise ValueError('Rocket representation must be selected in the configuration.')

    try:
        if for_rocket:
            vis_out = dataset.get_train_val().get_x()
        else:
            vis_out = np.load(config.directory_model_to_use + 'vis_out.npy')
        print(f'Shape before embedding: {vis_out.shape}\n')
    except FileNotFoundError:
        print('Required file not saved for the configured model.')
        exit(0)

    # If enabled, reduce the data to the first pca_components principal components.
    if pca_components > 0:
        pca = PCA(n_components=pca_components, random_state=config.random_seed)
        vis_out = pca.fit_transform(vis_out)

        print(f'Cum. explained variation for {pca_components} components: {np.sum(pca.explained_variance_ratio_)}\n')
    else:
        print('Dimensionality reduction by PCA is not used.')

    lr = max(vis_out.shape[0] / 50.0 / 4.0, 50)  # Equal to auto setting in newer version
    print('Learning rate used:', lr)
    tsne = TSNE(n_components=2, perplexity=40, early_exaggeration=50.0, learning_rate=lr, n_iter=10_000,
                verbose=1, random_state=config.random_seed, n_jobs=config.multiprocessing_limit, init='pca')
    vis_out_emb = tsne.fit_transform(vis_out)

    if for_rocket:
        np.save(config.get_training_data_path() + 'vis_out_emb.npy', vis_out_emb)
    else:
        np.save(config.directory_model_to_use + 'vis_out_emb.npy', vis_out_emb)

    print('Finished embedding creation.')


def main():
    config = Configuration()

    parser = argparse.ArgumentParser()
    parser.add_argument("--fit", default=False, action="store_true")
    parser.add_argument("--for_rocket", default=False, action="store_true")
    parser.add_argument("--pca", type=int, default=-1)
    parser.add_argument("--vis", default=False, action="store_true")
    parser.add_argument("--high_res", default=False, action="store_true")

    args, _ = parser.parse_known_args()

    if args.fit:
        fit_tsne(config, args.pca, args.for_rocket)

    if args.vis:
        visualise_tsne(config, args.for_rocket, args.high_res)


if __name__ == '__main__':
    main()
