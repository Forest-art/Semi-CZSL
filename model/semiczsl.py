from itertools import product
from turtle import shape
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import argparse
import clip
from collections import OrderedDict
from clip_modules.model_loader import load
import numpy as np
from model.common import *
from diffusers import StableDiffusionPipeline
from dataset.dataset import CompositionDataset
from random import choice

class CSP(nn.Module):
    def __init__(self, config, attributes, classes, offset):
        super().__init__()
        clip_model, _ = load(config.clip_model, context_length=config.context_length)
        self.clip = clip_model
        self.config = config
        self.attributes = attributes
        self.classes = classes
        self.attr_dropout = nn.Dropout(config.attr_dropout)
        self.token_ids, self.soft_att_obj, ctx_vectors = self.construct_soft_prompt()
        self.offset = offset
        self.enable_pos_emb = True


        dtype = None
        if dtype is None:
            self.dtype = torch.float16
        else:
            self.dtype = dtype
        self.text_encoder = CustomTextEncoder(self.clip, self.dtype)
        for p in self.parameters():
            p.requires_grad=False

        self.softmax = nn.Softmax(dim = 1)
        self.soft_att_obj = nn.Parameter(self.soft_att_obj)
        self.soft_prompt = nn.Parameter(ctx_vectors).cuda()
        # self.fine_tune = MLP(config.width_txt, config.width_txt, num_layers=1, relu=True, bias=True, dropout=True, norm=True, layers=[1280, 768])
        # self.weight = config.res_w



    def construct_soft_prompt(self):
        token_ids = clip.tokenize("a photo of x x",
                              context_length=self.config.context_length).cuda()

        tokenized = torch.cat(
            [
                clip.tokenize(tok, context_length=self.config.context_length)
                for tok in self.attributes + self.classes
            ]
        )
        orig_token_embedding = self.clip.token_embedding(tokenized.cuda())

        # with torch.no_grad():
        soft_att_obj = torch.zeros(
            (len(self.attributes) + len(self.classes), orig_token_embedding.size(-1)),
        )
        for idx, rep in enumerate(orig_token_embedding):
            eos_idx = tokenized[idx].argmax()
            soft_att_obj[idx, :] = torch.mean(rep[1:eos_idx, :], axis=0)

        ctx_init = "a photo of "
        n_ctx = len(ctx_init.split())
        prompt = clip.tokenize(ctx_init,
                            context_length=self.config.context_length).cuda()
        with torch.no_grad():
            embedding = self.clip.token_embedding(prompt)
        ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        return token_ids, soft_att_obj, ctx_vectors


    def construct_token_tensors(self, pair_idx):
        attr_idx, obj_idx = pair_idx[:, 0], pair_idx[:, 1]
        class_token_ids = self.token_ids.repeat(len(pair_idx), 1)
        token_tensor = self.clip.token_embedding(
            class_token_ids.cuda()
        ).type(self.clip.dtype)
        soft_att_obj = self.attr_dropout(self.soft_att_obj)
        eos_idx = int(self.token_ids[0].argmax())
        token_tensor[:, eos_idx - 2, :] = soft_att_obj[
            attr_idx
        ].type(self.clip.dtype)
        token_tensor[:, eos_idx - 1, :] = soft_att_obj[
            obj_idx + self.offset
        ].type(self.clip.dtype)

        # adding the correct learnable context
        token_tensor[
            :, 1 : len(self.soft_prompt) + 1, :
        ] = self.soft_prompt.type(self.clip.dtype)
        return token_tensor



    def visual(self, x: torch.Tensor):
        x = self.clip.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.clip.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.clip.visual.positional_embedding.to(x.dtype)
        x = self.clip.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        img_feature = self.clip.visual.transformer(x)

        x = img_feature.permute(1, 0, 2)  # LND -> NLD

        x = self.clip.visual.ln_post(x[:, 0, :])
        if self.clip.visual.proj is not None:
            x = x @ self.clip.visual.proj
        return x




    def forward(self, batch_img, idx):
        b = batch_img.shape[0]
        l, _ = idx.shape
        batch_img = self.visual(batch_img.type(self.clip.dtype))   ## bs * 768
        # finetune_img = self.fine_tune(batch_img.type(torch.float)).type(self.clip.dtype)
        token_tensors = self.construct_token_tensors(idx)
        text_features = self.text_encoder(self.token_ids, token_tensors, enable_pos_emb=self.enable_pos_emb)  
        # batch_img = self.weight * batch_img + (1 - self.weight) * finetune_img
        normalized_img = batch_img / batch_img.norm(dim=-1, keepdim=True)
        idx_text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logits = (self.clip.logit_scale.exp() * normalized_img @ idx_text_features.t())
        scores = self.softmax(logits)

        return logits, scores



# class SemiCZSL(nn.Module):
#     def __init__(self, config, attributes, classes, offset):
#         self.model_o = CSP(config, attributes=attributes, classes=classes, offset=offset)
#         self.model_s = CSP(config, attributes=attributes, classes=classes, offset=offset)

#     def forward(self, batch_img, idx):
#         logits_o = self.model_o(batch_img, idx)
#         logits_s = self.model_s(batch_img, idx)