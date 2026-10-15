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
"""Read pair CSVs (the pair builders' output) into per-generator token lists.

A row has columns sig, text1, text2, target, id1, id2. text1 and text2 are the
two tokenised generating vectors: r blocks of part_length tokens each, one
block per generator (order token, word, -1 padding; see
data/data_tokenisation/tokenise_variable.py). The loader splits each text into
its blocks, text1_1 .. text1_r and text2_1 .. text2_r, which the model
processes generator by generator. part_length must be the value the data was
tokenised with (stages.tokenise.params.part_length in data/pipeline.json).
"""

import os
import csv


def _parse_row(row, part_length):
    """One CSV row -> sample dict with the two texts split into generator blocks."""
    text1 = list(map(int, row['text1'].split(',')))
    text2 = list(map(int, row['text2'].split(',')))

    if len(text1) % part_length != 0:
        raise ValueError(
            f"text1 length {len(text1)} for sig={row['sig']!r} is not a multiple of "
            f"part_length {part_length}: was the data tokenised with another block length?"
        )
    num_parts = len(text1) // part_length
    if len(text2) != len(text1):
        raise ValueError(
            f"text2 length {len(text2)} != text1 length {len(text1)} for sig={row['sig']!r}: "
            f"the two generating vectors of a pair must have the same signature length"
        )

    sample = {
        'sig': row['sig'],
        'target': int(row['target']),
        'id1': row['id1'],
        'id2': row['id2'],
        'num_parts': num_parts,
        'part_length': part_length,
    }
    for k in range(num_parts):
        start, end = k * part_length, (k + 1) * part_length
        sample[f'text1_{k + 1}'] = text1[start:end]
        sample[f'text2_{k + 1}'] = text2[start:end]

    return sample


def _rows(data_dir, split):
    if not (split.endswith('.tsv') or split.endswith('.csv')):
        split += '.csv'
    file = os.path.join(data_dir, split)
    delimiter = ',' if file.endswith('.csv') else '\t'
    with open(file, encoding='UTF8') as f:
        yield from csv.DictReader(f, delimiter=delimiter)


def load_data(data_dir, split, part_length):
    """All samples of data_dir/split (evaluation sets: small enough for memory)."""
    return [_parse_row(row, part_length) for row in _rows(data_dir, split)]


def stream_data(data_dir, split, part_length):
    """Yield samples one at a time (training sets: millions of rows)."""
    for row in _rows(data_dir, split):
        yield _parse_row(row, part_length)
