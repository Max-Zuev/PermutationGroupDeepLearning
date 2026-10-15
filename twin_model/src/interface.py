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
"""Batching: from loaded samples (loader.py) to the batches the model consumes.

A batch is a dict of lists: sig, num_parts, part_length, text1_k and text2_k
for k = 1..r, and (when labelled) target, id1, id2. Every sample in a batch has
the same number of generators r, because the network stacks the per-generator
vectors to [B, r, H]; so samples are bucketed by r before batching.

Token ids are passed through unchanged: the tokeniser's integers are the
embedding table's indices (num_vocab = num_embeddings from the run config),
and -1 marks padding (model.process_data turns it into the attention mask).
"""

import random
from .utils.loader import stream_data


class Interface:
    def __init__(self, args, log=None):
        self.args = args
        args.num_classes = 2                    # same class / different class
        args.num_vocab = args.num_embeddings

    # ---- evaluation sets: all in memory, fixed batches ----------------------
    def pre_process(self, data, training=True):
        """Samples -> list of batches, bucketed by r. Order within a bucket is kept."""
        result = [self.process_sample(sample) for sample in data]
        batch_size = self.args.batch_size
        batches = []
        for bucket in self._buckets_by_parts(result):
            batches.extend(self.make_batch(bucket[i:i + batch_size])
                           for i in range(0, len(bucket), batch_size))
        return batches

    @staticmethod
    def _buckets_by_parts(samples):
        buckets = {}
        for s in samples:
            buckets.setdefault(s['num_parts'], []).append(s)
        return [buckets[k] for k in sorted(buckets)]

    def process_sample(self, sample, with_target=True):
        num_parts = sample['num_parts']
        processed = {
            'sig': sample['sig'],
            'num_parts': num_parts,
            'part_length': sample['part_length'],
        }
        for k in range(1, num_parts + 1):
            processed[f'text1_{k}'] = sample[f'text1_{k}']
            processed[f'text2_{k}'] = sample[f'text2_{k}']
        if 'target' in sample and with_target:
            processed['target'] = sample['target']
            processed['id1'] = sample['id1']
            processed['id2'] = sample['id2']
        return processed

    # ---- training sets: streamed, shuffled within a window -------------------
    def stream_batches(self, data_dir, split, shuffle_buffer=10000):
        """Yield training batches from a CSV too large to hold in memory.

        Samples are read in file order into per-r buffers. Once shuffle_buffer
        samples are buffered, every buffer is shuffled and emitted in full
        batches; remainders stay buffered for the next round, and whatever is
        left at the end of the file is emitted as partial batches. The shuffle
        therefore only mixes samples within a window of the file: the pair
        builders write their rows pre-shuffled, and a class-blocked file would
        train badly.
        """
        batch_size = self.args.batch_size
        buffers = {}
        total = 0
        for sample in stream_data(data_dir, split, self.args.part_length):
            s = self.process_sample(sample)
            buffers.setdefault(s['num_parts'], []).append(s)
            total += 1
            if total >= shuffle_buffer:
                for r in list(buffers):
                    bucket = buffers[r]
                    random.shuffle(bucket)
                    while len(bucket) >= batch_size:
                        yield self.make_batch(bucket[:batch_size])
                        bucket = bucket[batch_size:]
                    buffers[r] = bucket
                total = sum(len(b) for b in buffers.values())
        for r in list(buffers):
            bucket = buffers[r]
            random.shuffle(bucket)
            for i in range(0, len(bucket), batch_size):
                if bucket[i:i + batch_size]:
                    yield self.make_batch(bucket[i:i + batch_size])

    @staticmethod
    def make_batch(batch, with_target=True):
        """List of samples -> dict of lists (one list per key)."""
        batch = {key: [sample[key] for sample in batch] for key in batch[0].keys()}
        if 'target' in batch and not with_target:
            del batch['target']
        return batch
