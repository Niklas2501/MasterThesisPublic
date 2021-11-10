import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# noinspection PyUnresolvedReferences
from sklearn.preprocessing import scale, minmax_scale

# suppress debugging messages of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

from configuration.Configuration import Configuration
from configuration.Enums import A_Variant
from analytic_tools.A_Analysis_Math import get_sim_matrix, mathematical_comparison


def plot_comparison(a_1, a_2, desc_1, desc_2, av_1, av_2, plot_labels, labels, name, vertical=False):
    vmin, vmax = (0, 1)

    if vertical:
        n_rows, n_cols = 2, 1
        size = 18 if plot_labels else 7.5
        dpi = 200 if plot_labels else 200

        fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(size * n_cols, size * n_rows), dpi=dpi)

        im1 = subplot(ax[0], a_1, desc_1, av_1, plot_labels, labels, vmin, vmax, x_labels=False, y_labels=True,
                      plot_cbar=True)
        im2 = subplot(ax[1], a_2, desc_2, av_2, plot_labels, labels, vmin, vmax, x_labels=True, y_labels=True,
                      plot_cbar=True)

        if plot_labels:
            plt.subplots_adjust(hspace=-0.25, bottom=0.05, top=1, left=0.125, right=.95)
        else:
            plt.subplots_adjust(hspace=-0.3, bottom=0.05, top=1, left=0.10, right=.90)

    else:
        n_rows, n_cols = 1, 2
        size = 18 if plot_labels else 7.5
        dpi = 200 if plot_labels else 200

        fig, ax = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(size * n_cols, size * n_rows), dpi=dpi)

        im1 = subplot(ax[0], a_1, desc_1, av_1, plot_labels, labels, vmin, vmax, x_labels=True, y_labels=True)
        im2 = subplot(ax[1], a_2, desc_2, av_2, plot_labels, labels, vmin, vmax, x_labels=True, y_labels=False)

        if plot_labels:
            fig.subplots_adjust(bottom=0.08, top=1, left=0.08, right=0.90, wspace=0.08, )  # hspace=0.05
            cb_ax = fig.add_axes([0.94, 0.145, 0.02, 0.795])
        else:
            fig.subplots_adjust(bottom=0.00, top=1, left=0.05, right=0.90, wspace=0.15, )  # hspace=0.05
            # pos from left, pos from bottom, width, height
            cb_ax = fig.add_axes([0.94, 0.105, 0.02, 0.794])

        cbar = fig.colorbar(im2, cax=cb_ax)

    fig.savefig(f"../logs/{name}.pdf", dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_diff(a_l, a_pre, desc_1, desc_2, av_1, av_2, plot_labels, labels, name):
    plt.set_cmap('bwr')

    a_diff = a_l - a_pre
    vmin, vmax = (-np.max(np.abs(a_diff)), np.max(np.abs(a_diff)))

    size = 22 if plot_labels else 7.5
    dpi = 200 if plot_labels else 200

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size, size), dpi=dpi)

    subplot(ax, a_diff, desc_1, av_1, plot_labels, labels, vmin, vmax, plot_cbar=True)

    fig.tight_layout()
    fig.savefig(f"../logs/{name}.pdf", dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_single(a_1, desc_1, av_1, plot_labels, labels, name):
    size = 22 if plot_labels else 7.5
    dpi = 200 if plot_labels else 200

    vmin, vmax = (0, 1)  # np.min(a_1), np.max(a_1)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size, size), dpi=dpi)

    subplot(ax, a_1, desc_1, av_1, plot_labels, labels, vmin, vmax, plot_cbar=True)

    fig.tight_layout()
    fig.savefig(f"../logs/{name}.pdf", dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close(fig)


def subplot(ax, a, desc, av, feature_name_labels, labels, vmin, vmax, plot_cbar=False, x_labels=True, y_labels=True):
    im = ax.imshow(a, vmin=vmin, vmax=vmax)

    if plot_cbar:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.2)
        plt.colorbar(im, cax=cax)

    # ax.set_ylabel('i (Zielknoten)')
    #
    # if plot_cbar and x_labels:
    #     ax.set_xlabel('j (Ausgangsknoten)')

    ax.set_ylabel('i (target)')

    if plot_cbar and x_labels:
        ax.set_xlabel('j (source)')

    draw_grid(ax, a.shape[0], 1, True, feature_name_labels)
    draw_grid(ax, a.shape[0], 10, False, feature_name_labels)

    ax.tick_params(which='major', width=1, color='black')
    ax.tick_params(which='minor', width=0, color='white')

    if feature_name_labels:
        major_ticks = np.arange(0, a.shape[0], 1)
        labels = [f[0:20] if len(f) > 20 else f for f in labels]
    else:
        major_ticks = np.arange(0, a.shape[0], 10)
        labels = (major_ticks + 0.5).astype('int')

    ax.set_xticks(major_ticks, minor=False)
    ax.set_yticks(major_ticks, minor=False)

    if x_labels:
        ax.set_xticklabels(labels, minor=False)

    if y_labels:
        ax.set_yticklabels(labels, minor=False)

        if feature_name_labels:
            plt.setp(ax.get_xticklabels(), rotation=75, ha="right", rotation_mode="anchor")

    return im


def draw_grid(ax, grid_size, step_size, minor, feature_name_labels):
    if feature_name_labels:
        if minor:
            mod_x, mod_y = 0, 0  # -0.1, -0.2
            color = '#a0a0a0'
            linewidth = 0.75
            alpha = 1
            gr_1 = np.arange(0, grid_size, step_size) + 0.5
        else:
            mod_x, mod_y = 0, 0
            color = 'black'
            linewidth = 1
            alpha = 1
            gr_1 = np.arange(10, grid_size, step_size) + 0.5
    else:
        if minor:
            mod_x, mod_y = 0, 0  # -0.05, -0.1
            color = '#a0a0a0'
            linewidth = 0.5
            alpha = 0.3
            gr_1 = np.arange(0, grid_size, step_size) + 0.5
        else:
            mod_x, mod_y = 0, 0
            color = 'black'
            linewidth = 0.75
            alpha = 1
            gr_1 = np.arange(10, grid_size, step_size) + 0.5

    gr_2, gr_3 = np.full(grid_size, 0) - 0.5, np.full(grid_size, grid_size) - 0.5
    gr_2, gr_3 = gr_2[::step_size], gr_3[::step_size]

    if not minor:
        gr_2, gr_3 = gr_2[1:], gr_3[1:]

    ax.hlines(y=gr_1 + mod_y, xmin=gr_2, xmax=gr_3, color=color, linewidth=linewidth, linestyle='-', alpha=alpha)
    ax.vlines(x=gr_1 + mod_x, ymin=gr_2, ymax=gr_3, color=color, linewidth=linewidth, linestyle='-', alpha=alpha)


def load_a_pre(config):
    path = config.get_additional_data_path(config.a_pre_file)
    a_df = pd.read_excel(path, engine='openpyxl')
    a_df = a_df.set_index('Features')
    a_pre = a_df.values.astype(dtype=np.float)

    return a_pre


def load_a_out(config: Configuration, models: [A_Variant, str]):
    a_outs_info, a_outs = [], []

    for av, model in models:

        if av == A_Variant.A_PRE:
            path_list = Path(config.models_folder + model + '/a_out').rglob('*.npy')

            a_outs_gat = {}

            for path in path_list:
                a_out = np.load(str(path))
                a_outs.append(a_out)

                substrings = path.name.split('.')[0].split('_')
                head, c = substrings[-1], '_'.join(substrings[2:-1])
                a_outs_info.append((model, av, c, head))

                if head in a_outs_gat.keys():
                    a_outs_gat[head].append(a_out)
                else:
                    a_outs_gat[head] = [a_out]

            for head, arrays in a_outs_gat.items():
                a_gat_head_avg = np.mean(arrays, axis=0)
                a_outs.append(a_gat_head_avg)
                a_outs_info.append((model, av, 'AVG', head))

        else:
            path = config.models_folder + model + '/a_out.npy'
            a_outs.append(np.load(path))
            a_outs_info.append((model, av, 'None', 'None'))

    return a_outs, a_outs_info


def normalise_a_outs(a_outs: [np.ndarray]):
    a_outs_normalised = []

    for i, a_out in enumerate(a_outs):
        x = minmax_scale(a_out, feature_range=(0, 1))
        # x = np.where(a_out > 0.0, 1, 0)
        # x = a_out
        a_outs_normalised.append(x)
    return a_outs_normalised


def count_column_values(a, feature_names):
    x = pd.DataFrame(data=a, columns=feature_names, index=feature_names)
    for i, c in enumerate(x.columns):
        print(c)
        values = x[c].values
        vs, cs = np.unique(-values, return_counts=True)
        print(*zip(-vs, cs), sep='\n')
        print()
        if i > 10:
            break


def main():
    plt.set_cmap('hot_r')
    config = Configuration()
    feature_names = np.load(config.training_data_folder + 'feature_names.npy')
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_settings", default=False, action="store_true")
    parser.add_argument("--plot_labels", default=False, action="store_true")
    parser.add_argument("--include_class_wise", default=False, action="store_true")
    parser.add_argument('--math_output', default=False, action='store_true')
    parser.add_argument('--output_sim_matrices', default=False, action='store_true')
    parser.add_argument('--plot', choices=['single', 'single_all', 'compare_two', 'diff', 'diff_all'],
                        required=False, default='none', help='')
    args, _ = parser.parse_known_args()
    func_requires_two = args.plot in ['compare_two', 'diff']

    models = [
        (A_Variant.A_PRE, 'vis_stgcn_gat_g_25011997'),
        (A_Variant.A_EMB_V1, 'vis_stgcn_gcn_g_a_emb_v1_25011997'),
        (A_Variant.A_PARAM, 'vis_stgcn_gcn_g_a_param_25011997')
    ]

    # Default settings
    a_1_index = 18
    a_2_index = 21  # len(a_outs) - 1  # A_Pre from file
    name = 'a_plot_temp'

    a_pre = load_a_pre(config)
    a_outs, a_outs_info = load_a_out(config, models)
    a_outs.append(a_pre)
    a_outs = normalise_a_outs(a_outs)

    a_outs_info.append(('m_default', A_Variant.A_PRE, 'None', 'None'))

    df = pd.DataFrame(data=a_outs_info, columns=['Model', 'A Variant', 'Desc.', 'Head (if GAT)'])
    print(df.to_string())
    print()

    if args.custom_settings:
        name = input('File name: ')
        name = name.strip() if name.strip() != '' else 'a_plot'

        a_1_index = int(input('Index of 1. ADJ: '))
        assert 0 <= a_1_index < len(df), '1. Index must be in range of imported ADJ.'

        if func_requires_two:
            a_2_index = int(input('Index of 2. ADJ: '))
            assert 0 <= a_2_index < len(df), '2. Index must be in range of imported ADJ.'

    a_1 = a_outs[a_1_index]
    av_1, desc_1 = df.loc[a_1_index, 'A Variant'], df.loc[a_1_index, 'Model'] + '_' + df.loc[a_1_index, 'Desc.']
    a_2 = a_outs[a_2_index]
    av_2, desc_2 = df.loc[a_2_index, 'A Variant'], df.loc[a_2_index, 'Model'] + '_' + df.loc[a_2_index, 'Desc.']

    if args.plot == 'none':
        pass
    elif args.plot == 'single':
        plot_single(a_1, desc_1, av_1, args.plot_labels, feature_names, name)
    elif args.plot == 'single_all':
        for row_i in range(len(df)):
            a = a_outs[row_i]
            model, av, desc, head = a_outs_info[row_i]

            # Skip the predefined matrix itself.
            if model == 'm_default':
                continue

            # Skip class wise gat outputs if parameter to include those is not set
            if av == A_Variant.A_PRE and desc not in ['AVG', 'None'] and not args.include_class_wise:
                continue

            file_name_parts = [model] if av != A_Variant.A_PRE else [model, head]
            file_name_parts = file_name_parts + ['labeled'] if args.plot_labels else file_name_parts
            name = '_'.join(file_name_parts)

            print(f'Plotting {row_i}/{len(df)}: {name}')
            plot_single(a, name, av, args.plot_labels, feature_names, name)

    elif args.plot == 'compare_two':
        plot_comparison(a_1, a_2, desc_1, desc_2, av_1, av_2, args.plot_labels, feature_names, name)
    elif args.plot == 'diff':
        plot_diff(a_1, a_2, desc_1, desc_2, av_1, av_2, args.plot_labels, feature_names, name)
    elif args.plot == 'diff_all':
        a_pre = a_outs[len(a_outs) - 1]
        a_pre_model, a_pre_av, a_pre_desc, a_pre_head = a_outs_info[len(a_outs) - 1]

        for row_i in range(len(df)):
            a = a_outs[row_i]
            model, av, desc, head = a_outs_info[row_i]

            # Skip the predefined matrix itself.
            if model == 'm_default':
                continue

            # Skip class wise gat outputs if parameter to include those is not set
            if av == A_Variant.A_PRE and desc not in ['AVG', 'None'] and not args.include_class_wise:
                continue

            file_name_parts = ['diff', model] if av != A_Variant.A_PRE else ['diff', model, head]
            file_name_parts = file_name_parts + ['labeled'] if args.plot_labels else file_name_parts
            name = '_'.join(file_name_parts)

            print(f'Plotting {row_i}: {name}')
            plot_diff(a, a_pre, desc, a_pre_desc, av, a_pre_av, args.plot_labels, feature_names, name)

    else:
        raise ValueError(args.func)

    if args.math_output:
        mathematical_comparison(a_1, a_2, desc_1, desc_2)

    if args.output_sim_matrices:
        all_metrics = ['mae', 'rmse', 'rse', 'sum_abs_errors', 'fro_norm', 'fro_norm_a_wise', 'mean_node_wise_cos_sim']

        for metric in all_metrics:
            print(f'Sim/Dis-Matrix for metric {metric}:')
            print()
            sim_matrix = get_sim_matrix(a_outs, a_outs_info, args.include_class_wise, metric)
            print(sim_matrix.to_string())
            print('\n-------------------------------------------------------------------\n')


if __name__ == '__main__':
    main()
