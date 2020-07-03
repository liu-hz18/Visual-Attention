# -*- coding: utf-8 -*-
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def draw_attn_heatmap(
        input_words,
        output_words,
        attentions,
        name='attn',
        show_label=True,
        input_label='src',
        output_label='tgt',
        cmap='bone'):
    assert cmap in ['hot', 'bone', 'cool', 'gray', 'spring', 'summer', 'autumn', 'winter'],\
        "param: \'cmap\' should in [hot, bone, cool, gray, spring, summer, autumn, winter]"
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().numpy(), cmap=cmap)
    fig.colorbar(cax)
    ax.set_xlabel(xlabel=input_label)
    ax.set_ylabel(ylabel=output_label)
    if show_label:
        # Set up axes
        ax.set_xticklabels([''] + output_words, rotation=45)
        ax.set_yticklabels([''] + input_words, rotation=45)
        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(name + "_heatmap.pdf", format="pdf")


def draw_attn_bipartite(
        input_words,
        output_words,
        attentions,
        name='attn',
        show_label=True):
    input_words = [word + '   ' for word in input_words]
    output_words = ['   ' + word for word in output_words]
    attn = attentions.cpu().numpy()
    left, right, bottom, top = .4, .6, .1, .9
    mid = (top + bottom)/2.
    layer_sizes = [len(input_words), len(output_words)]
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)
    src_layer_top = v_spacing*(layer_sizes[0]-1)/2. + mid
    tgt_layer_top = v_spacing*(layer_sizes[1]-1)/2. + mid
    src_layer_left = left
    tgt_layer_left = left + h_spacing
    # add nodes and edges
    G = nx.Graph()
    for i in range(layer_sizes[0]):
        G.add_node(input_words[i], pos=(src_layer_left, src_layer_top - i*v_spacing))
        for j in range(layer_sizes[1]):
            G.add_node(output_words[j], pos=(tgt_layer_left, tgt_layer_top - j*v_spacing))
            G.add_edge(input_words[i], output_words[j], weight=attn[i][j])

    pos = nx.get_node_attributes(G, 'pos')
    edge_colors = [edge[-1]['weight'] for edge in G.edges(data=True)]
    # draw graph
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_map = plt.cm.Purples
    nx.draw_networkx_nodes(
        G, pos, node_shape='s', alpha=0)
    edges = nx.draw_networkx_edges(
        G, pos, edge_color=edge_colors, width=0.8, edge_cmap=color_map)
    if show_label:
        nx.draw_networkx_labels(G, pos)

    edges.cmap = color_map
    plt.colorbar(edges, ax=ax)
    plt.savefig(name + "_bipartite.pdf", format="pdf")


if __name__ == '__main__':
    src_len = 10
    tgt_len = 15
    attn = torch.randn((src_len, tgt_len))
    src_lable = [str(i) for i in range(src_len)]
    tgt_lable = [str(10 + i) for i in range(tgt_len)]
    draw_attn_heatmap(src_lable, tgt_lable, attn, name='attn', show_label=True)
    draw_attn_bipartite(src_lable, tgt_lable, attn, name='attn', show_label=True)
