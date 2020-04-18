import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim


def accuracy(model, dataloader, batch_size, gpu=False):
    if gpu and torch.cuda.is_available():
        model.cuda()
    model.eval()

    # number of total (context_heroes, center_hero)
    length = len(dataloader)*batch_size
    count = 0
    for teams, targets in dataloader:
        if gpu and torch.cuda.is_available():
            teams = teams.cuda()
            targets = targets.cuda()
        inputs = autograd.Variable(teams)
        targets = autograd.Variable(targets.view(-1))
        out = model(inputs)

        # idx is the index of the maximum value
        val, idx = torch.max(out, dim=1)

        # count how many predictions are right and convert to python int
        count += idx.eq(targets).sum().cpu().item()
    return count/length


def make_plot_color(x, y, hero2ix):

    # divide heroes to their own categories/roles
    heros = []
    for name, idx in hero2ix.items():
        heros.append(idx)

    # plot tank, dps, and supporters respectively
    con_x, con_y = x[heros], y[heros]
    fig = plt.figure(figsize=(16, 12), dpi = 100)
    ax = plt.subplot(111)
    marker_size = 200
    ax.scatter(con_x, con_y, c = 'royalblue', s=marker_size)

    # annotate each hero's name
    for name, i in hero2ix.items():
        ax.annotate(name, (x[i], y[i]), fontsize=18)
    plt.show()
    fig.savefig('./output/lol_hero_embddings_2d.png')


def plot_embeddings(model, names):
    embeddings = model.embeddings.weight.cpu().data.numpy()

    #makes mean at 0
    embeddings -= np.mean(embeddings, axis=0)

    # run pca to reduce to 2 dimensions
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    x, y = embeddings_2d[:, 0], embeddings_2d[:, 1]
    make_plot_color(x, y, names)


def plot_loss(losses, directory):
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = plt.subplot(111)
    ax.plot(losses)
    ax.set_xlabel('Epochs', fontsize=24)
    ax.set_ylabel('Train_loss', fontsize=24)
    fig.savefig(directory)
    plt.close()
