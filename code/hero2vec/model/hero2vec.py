import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init


class CBOH(nn.Module):

    def __init__(self, heropool_size, embedding_dim):
        """
        Initialize an NN with one hidden layer. Weight of the hidden layer is
        the embedding.
        inputs:
            heropool_size: int
            embedding_dim: int
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(heropool_size, embedding_dim)
        self.affine = nn.Linear(embedding_dim, heropool_size)
        self.init_emb()

    def init_emb(self):
        """
        init embeddings and affine layer
        """
        initrange = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.affine.weight.data.uniform_(-0, 0)
        self.affine.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs:
            inputs: torch.autograd.Variable, size = (N, 4)
        returns:
            out: torch.autograd.Variable, size = (N, heropool_size)
        """
        embeds = self.embeddings(inputs).sum(dim=1) #contiuous
        out = self.affine(embeds)
        return out


