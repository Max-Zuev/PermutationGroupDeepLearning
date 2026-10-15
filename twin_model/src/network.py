# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The twin network: RE2 text matching (Yang et al. 2019), applied generator by generator.

Input: a pair of tokenised generating vectors (gamma_1, ..., gamma_r) and
(delta_1, ..., delta_r) of the same signature. Each generator is a block of
part_length tokens (order token, transposition word, -1 padding; see
data/data_tokenisation/tokenise_variable.py). A batch carries text1_k, text2_k
and the padding masks mask1_k, mask2_k for k = 1..r; interface.py groups a batch
by r, so every pair in it has the same number of generators.

Forward pass (the paper's Figure 1):

  1. Per generator k, both sides in parallel: embed the tokens, then `blocks`
     RE2 blocks in series. A block runs the CNN encoder over the token
     sequence, concatenates the encoding with the block's input, aligns the
     tokens of gamma_k against the tokens of delta_k (attention in both
     directions), and fuses each token's own features with its aligned
     features. Between blocks the augmented-residual connection feeds the next
     block the previous block's output together with the token embedding.
     After the last block a masked max over tokens gives one vector per
     generator and side: [B, H].
  2. Across generators: the r vectors are stacked to [B, r, H] and pooled.
     'convmax' (the paper's choice) adds GeLU(Conv1d, kernel 3) over the
     generator axis as a residual, so each generator's vector sees its
     neighbours, which is where braid moves act, then takes the max over
     generators. 'max' skips the convolution and is invariant to permuting the
     generators (which the paper argues against, Example 2).
  3. The prediction head maps the two side vectors to two logits; the
     'symmetric' head uses [a, b, |a-b|, a*b], so the decision does not depend
     on which vector is given first.

Weights are shared across generators (one block stack processes every k) and
across the two sides (a twin). The per-generator pass is a cross-encoder: side
a's features for generator k depend on side b's generator k through alignment,
so a generating vector has no standalone embedding.
"""

import torch
import torch.nn as nn
from .modules import Module, ModuleList, ModuleDict
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import Pooling
from .modules.prediction import registry as prediction


class Network(Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout
        self.embedding = Embedding(args)
        # with the 'aug' connection every block after the first receives the
        # embedding alongside the previous block's hidden output
        input_emb_size = args.embedding_dim if args.connection == 'aug' else 0
        self.blocks = ModuleList([ModuleDict({
            'encoder': Encoder(args, args.embedding_dim if i == 0 else input_emb_size + args.hidden_size),
            'alignment': alignment[args.alignment](
                args, args.embedding_dim + args.hidden_size if i == 0 else input_emb_size + args.hidden_size * 2),
            'fusion': fusion[args.fusion](
                args, args.embedding_dim + args.hidden_size if i == 0 else input_emb_size + args.hidden_size * 2),
        }) for i in range(args.blocks)])
        self.connection = connection[args.connection](args)
        self.pooling = Pooling()

        # pooling over generators, step 2 above
        self.part_pooling = args.part_pooling
        if self.part_pooling == 'convmax':
            # zero-initialised: the network starts as plain max pooling and learns
            # the order-aware residual from there
            self.part_conv = nn.Conv1d(args.hidden_size, args.hidden_size, 3, padding=1)
            nn.init.zeros_(self.part_conv.weight)
            nn.init.zeros_(self.part_conv.bias)
        elif self.part_pooling != 'max':
            raise ValueError(f"part_pooling must be 'convmax' or 'max', not {self.part_pooling!r}")

        self.prediction = prediction[args.prediction](args)

    def _encode_pair(self, a, b, mask_a, mask_b):
        """One generator of each side through the block stack (step 1).

        a, b: [B, L] token ids; mask_a, mask_b: [B, L, 1] True on real tokens.
        Returns the two pooled vectors, each [B, H].
        """
        a = self.embedding(a)
        b = self.embedding(b)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            if i > 0:
                a = self.connection(a, res_a, i)
                b = self.connection(b, res_b, i)
                res_a, res_b = a, b

            a_enc = block['encoder'](a, mask_a)
            b_enc = block['encoder'](b, mask_b)

            a = torch.cat([a, a_enc], dim=-1)
            b = torch.cat([b, b_enc], dim=-1)
            align_a, align_b = block['alignment'](a, b, mask_a, mask_b)
            a = block['fusion'](a, align_a)
            b = block['fusion'](b, align_b)

        return self.pooling(a, mask_a), self.pooling(b, mask_b)

    def forward(self, inputs):
        # the number of generators r is read off the batch keys text1_1 .. text1_r
        parts_a, parts_b = [], []
        k = 1
        while f'text1_{k}' in inputs:
            pa, pb = self._encode_pair(
                inputs[f'text1_{k}'], inputs[f'text2_{k}'],
                inputs[f'mask1_{k}'], inputs[f'mask2_{k}'],
            )
            parts_a.append(pa)  # [B, H]
            parts_b.append(pb)  # [B, H]
            k += 1

        pa_final = self._pool_parts(torch.stack(parts_a, dim=1))   # [B, r, H] -> [B, H]
        pb_final = self._pool_parts(torch.stack(parts_b, dim=1))
        return self.prediction(pa_final, pb_final)                  # [B, 2] logits

    def _pool_parts(self, z):
        """Pool the per-generator vectors [B, r, H] to one vector per side (step 2)."""
        if self.part_pooling == 'convmax':
            c = self.part_conv(z.transpose(1, 2)).transpose(1, 2)    # conv over the r axis
            z = z + torch.nn.functional.gelu(c)
        return z.max(dim=1).values
