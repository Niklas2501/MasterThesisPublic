import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

np.set_printoptions(threshold=sys.maxsize)
np.random.seed(1997)

a = np.random.random(61 * 61)
zeros = np.random.choice(np.arange(a.size), replace=False,
                         size=int(a.size * 0.985))
a[zeros] = 0
a = np.reshape(a, newshape=(61, 61))
features = ['long_feature_' + str(i) for i in range(61)]

# https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
############################################################
plt.set_cmap('hot_r')

display_labels = False
size = 15 if display_labels else 8

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(size, size), dpi=300)
im = ax.imshow(a, vmin=0, vmax=1)

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.5)
plt.colorbar(im, cax=cax)

ax.set_xlabel('j (target)')
ax.set_ylabel('i (source)')

ax.tick_params(which='minor', width=0)
ax.set_xticks(np.arange(-.5, 61, 10), minor=True)
ax.set_yticks(np.arange(-.5, 61, 10), minor=True)

# Gridlines based on minor ticks
ax.grid(which='minor', color='black', linestyle='-', linewidth=0.75)

if display_labels:
    # Minor ticks with width = 0 so they are not really visible
    ax.set_xticks(np.arange(0, 61, 1), minor=False)
    ax.set_yticks(np.arange(0, 61, 1), minor=False)

    ax.set_xticklabels(features, minor=False)
    ax.set_yticklabels(features, minor=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")

fig.tight_layout()
# fig.savefig(self.selected_model_path + "a_plot.svg", dpi=300, bbox_inches='tight')

plt.show()

################################################################
# import networkx as nx
# import matplotlib.pyplot as plt
# import pandas as pd
#
# a = [
#     [0.0, 3.0, 0.0],
#     [0.5, 0.0, 1.0],
#     [0.0, 0.0, 0.0]
# ]
#
# a = np.array(a)
#
# print(a)
#
# features = np.array(['a', 'b', 'c'])
#
# adj_df_vis = pd.DataFrame(a, index=features, columns=features)
# adj_df = pd.DataFrame(a, index=features, columns=features)
#
# print(adj_df_vis)
#
# plt.figure(figsize=(25, 20))
# G: nx.classes.graph.Graph = nx.from_pandas_adjacency(adj_df, create_using=nx.DiGraph())
#
# edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
#
# pos = nx.spring_layout(G, )  # weight=weights, iterations=100, k=1
#
# nx.draw_networkx(G, pos, node_color='#919191', with_labels=True, edgelist=edges, edge_color=weights, width=2,
#                  edge_cmap=plt.cm.get_cmap('inferno'))
#
# # Draw invisible edges just to get a hook for the color bar
# G_edges = nx.draw_networkx_edges(G, pos, edge_color=weights, width=0, edge_cmap=plt.cm.get_cmap('inferno_r'))
# # plt.colorbar(G_edges)
#
# plt.tight_layout()
# plt.axis('off')
# plt.show()
#
# A = nx.adjacency_matrix(G)
# print(A)
# print('.......')
#
#
# # G = nx.Graph()
#
# for i in range(a.shape[0]):
#     G.add_node(features[i], node_color='#919191', with_labels=True)
#
#
# edges = [(features[i], features[j], a[i,j]) for i in np.arange(0, 61) for j in np.arange(0, 61) if a[i, j] != 0]
# weights = [a[i,j]for i in np.arange(0, 61) for j in np.arange(0, 61) if a[i, j] != 0]
#
# x = G.add_weighted_edges_from(edges, edge_color=weights, width=1, edge_cmap=plt.cm.get_cmap('inferno'))
# nodes = nx.draw_networkx_nodes(G, pos, node_color='#919191')
# labels = nx.draw_networkx_labels(G, pos, {f:f for f in features}) # Would break if not all features are added to the graph
