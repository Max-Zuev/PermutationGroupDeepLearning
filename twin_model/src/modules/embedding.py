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


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


class Embedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.dropout = args.dropout

        embedding_new = getattr(args, 'embedding_new', False)
        embedding_path = getattr(args, 'embedding_path', None)
        embedding_requires_grad = getattr(args, 'embedding_requires_grad', True)

        if not embedding_new and embedding_path:
            # Load pre-computed embedding matrix from .npy file
            matrix = np.load(embedding_path)  # expected shape: [vocab_size, embedding_dim]
            loaded_vocab, loaded_dim = matrix.shape

            if loaded_dim != args.embedding_dim:
                raise ValueError(
                    f"Loaded embedding dim ({loaded_dim}) does not match "
                    f"config embedding_dim ({args.embedding_dim}). "
                    f"Update embedding_dim in your config to {loaded_dim}."
                )
            if loaded_vocab != args.num_vocab:
                raise ValueError(
                    f"Loaded embedding vocab size ({loaded_vocab}) does not match "
                    f"config num_vocab ({args.num_vocab})."
                )

            print(f"Embedding: loading from '{embedding_path}' "
                  f"[{loaded_vocab} × {loaded_dim}], requires_grad={embedding_requires_grad}")
            self.embedding = nn.Embedding(loaded_vocab, loaded_dim)
            self.embedding.weight = nn.Parameter(
                torch.tensor(matrix, dtype=torch.float32),
                requires_grad=embedding_requires_grad,
            )
        else:
            # Train a fresh random embedding
            if embedding_path and embedding_new:
                print(f"Embedding: embedding_new=True, ignoring embedding_path='{embedding_path}'")
            print(f"Embedding: random init [{args.num_vocab} × {args.embedding_dim}], "
                  f"requires_grad={embedding_requires_grad}")
            self.embedding = nn.Embedding(args.num_vocab, args.embedding_dim)
            self.embedding.weight.requires_grad = embedding_requires_grad

    def forward(self, x):
        # Validate indices are within vocabulary range
        if (x < 0).any() or (x >= self.embedding.num_embeddings).any():
            print(f"ERROR: Invalid embedding indices detected!")
            print(f"  Vocabulary size: {self.embedding.num_embeddings}")
            print(f"  Min index: {x.min().item()}, Max index: {x.max().item()}")
            print(f"  Invalid indices: {x[x < 0].unique().tolist() + x[x >= self.embedding.num_embeddings].unique().tolist()}")
            raise RuntimeError(f"Embedding indices out of bounds [0, {self.embedding.num_embeddings - 1}]")
        x = self.embedding(x)
        x = f.dropout(x, self.dropout, self.training)
        return x
