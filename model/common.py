from stringprep import b1_set
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import numpy as np
import clip
from collections import OrderedDict
# from clip_modules.model_loader import load
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce



class CustomTextEncoder(torch.nn.Module):
    def __init__(self, clip_model, dtype=torch.float16):
        super().__init__()
        self.dtype = dtype

        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.token_embedding = clip_model.token_embedding

    def tokenize(self, text):
        return torch.cat([clip.tokenize(tok) for tok in text])

    def encode_text(self, text, enable_pos_emb=True):
        token_ids = self.tokenize(text)
        text_features = self.forward(token_ids, None, enable_pos_emb)
        return text_features

    def forward(self, token_ids, token_tensors, enable_pos_emb):
        """The forward function to compute representations for the prompts.

        Args:
            token_ids (torch.tensor): the token ids, which
                contains the <eos> token.
            token_tensors (torch.Tensor, optional): the tensor
                embeddings for the token ids. Defaults to None.
            enable_pos_emb (bool, optional): adds the learned
                positional embeddigngs if true. Defaults to False.

        Returns:
            torch.Tensor: the vector representation of the prompt.
        """
        if token_tensors is not None:
            text_features = token_tensors
        else:
            text_features = self.token_embedding(token_ids)

        text_features = text_features.type(self.dtype)
        x = (
            text_features + self.positional_embedding.type(self.dtype)
            if enable_pos_emb
            else text_features
        )
        x = x.permute(1, 0, 2)
        text_feature = self.transformer(x)

        x = text_feature.permute(1, 0, 2)
        x = self.ln_final(x)
        tf = (
            x[
                torch.arange(x.shape[0]), token_ids.argmax(dim=-1)
            ]  # POS of <EOS>
            @ self.text_projection
        )
        return tf


class MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: Int, Input dimension
        out-dim: Int, Output dimension
        num_layer: Number of hidden layers
        relu: Bool, Use non linear function at output
        bias: Bool, Use bias
    '''
    def __init__(self, inp_dim, out_dim, num_layers = 1, relu = True, bias = True, dropout = False, norm = False, layers = []):
        super(MLP, self).__init__()
        mod = []
        incoming = inp_dim
        for layer in range(num_layers - 1):
            if len(layers) == 0:
                outgoing = incoming
            else:
                outgoing = layers.pop(0)
            mod.append(nn.Linear(incoming, outgoing, bias = bias))
            
            incoming = outgoing
            if norm:
                mod.append(nn.LayerNorm(outgoing))
                # mod.append(nn.BatchNorm1d(outgoing))
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
            if dropout:
                mod.append(nn.Dropout(p = 0.3))

        mod.append(nn.Linear(incoming, out_dim, bias = bias))

        if relu:
            mod.append(nn.ReLU(inplace = True))
            # mod.append(nn.LeakyReLU(inplace=True, negative_slope=0.2))
        self.mod = nn.Sequential(*mod)
    
    def forward(self, x):
        return self.mod(x)

