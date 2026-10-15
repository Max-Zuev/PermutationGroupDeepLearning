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
"""Run configuration: JSON5 files -> an args object.

A run config (configs/*.json5) is a list of dicts; each dict is one run. It
names its parents in __parents__ (configs/default.json5 holds every setting
with the paper's values); the dict's own keys override them. Nested dicts are
only grouping: every key ends up as a flat attribute of args. __repeat__ runs
the same config several times (name-0, name-1, ...), __iter__ sweeps one key.

After merging, _post_process resolves paths and logging settings against
twin_model/training.json (the model side's configuration) and the data
pipeline's data/pipeline.json it points to (storage root, token block length),
and converts the schedule from samples to optimiser updates.
"""

import os
import math
import shutil
from datetime import datetime
import json
import torch
import json5

# the model side's configuration (twin_model/training.json); it points to the data
# pipeline's, which the model must follow to read the data the way it was written
TRAINING_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'training.json')


class Object:
    """
    @DynamicAttrs
    """
    pass


def parse(config_file):
    """Config file -> list of (args, config dict), one per run."""
    root = os.path.dirname(config_file)  # __parents__ are relative to the config file
    config_group = _load_param('', config_file)
    if type(config_group) is dict:
        config_group = [config_group]
    configs = []
    for config in config_group:
        try:
            choice = config.pop('__iter__')
            assert len(choice) == 1, 'only support iterating over 1 variable'
            key, values = next(iter(choice.items()))
        except KeyError:
            key, value = config.popitem()
            values = [value]
        for value in values:
            config[key] = value
            repeat = config.get('__repeat__', 1)
            for index in range(repeat):
                config_ = config.copy()
                config_['__index__'] = index
                if repeat > 1:
                    config_['name'] += '-' + str(index)
                args = _parse_args(root, config_)
                configs.append((args, config_))
    return configs


def _parse_args(root, config):
    args = Object()
    assert type(config) is dict
    for parent in config.get('__parents__', []):
        parent = _load_param(root, parent)
        assert type(parent) is dict, 'only top-level configs can be a sequence'
        _add_param(args, parent)
    _add_param(args, config)
    _post_process(args)
    return args


def _add_param(args, x: dict):
    """Flatten x into args; a value may only overwrite one of the same type (or null)."""
    for k, v in x.items():
        if type(v) is dict:
            _add_param(args, v)
        else:
            k = _validate_param(k)
            if hasattr(args, k):
                previous_type = type(getattr(args, k))
                current_type = type(v)
                assert previous_type is current_type or \
                    isinstance(None, previous_type) or \
                    isinstance(None, current_type) or \
                    (previous_type is float and current_type is int), \
                    f'param "{k}" of type {previous_type} can not be overwritten by type {current_type}'
            setattr(args, k, v)


def _load_param(root, file: str):
    file = os.path.join(root, file)
    if not file.endswith('.json5'):
        file += '.json5'
    with open(file) as f:
        return json5.load(f)


def _post_process(args: Object):
    with open(TRAINING_JSON, encoding='utf-8') as f:
        training = json.load(f)
    with open(os.path.normpath(os.path.join(os.path.dirname(TRAINING_JSON), training['data_pipeline'])),
              encoding='utf-8') as f:
        pipeline = json.load(f)
    # tokens per generator block; fixed by the tokeniser, never by a run config
    args.part_length = pipeline['stages']['tokenise']['params']['part_length']
    # storage: a run config names data_dir / embedding_path / output_dir relative to
    # the data root (or the embeddings / outputs directories under it); an absolute
    # path is left alone
    root = pipeline['storage']['root']
    store = training['storage']
    if args.data_dir and not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(root, args.data_dir)
    if args.embedding_path and not os.path.isabs(args.embedding_path):
        args.embedding_path = os.path.join(root, store['embeddings'], args.embedding_path)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(root, store['outputs'], args.output_dir)
    args.wandb_mode = training['wandb']['mode']
    args.wandb_entity = training['wandb']['entity']
    args.wandb_project = training['wandb']['project']

    # one directory per run under output_dir; a rerun of the same name starts afresh
    os.makedirs(args.output_dir, exist_ok=True)
    if not args.name:
        args.name = str(datetime.now())
    args.summary_dir = os.path.join(args.output_dir, args.name)
    if os.path.exists(args.summary_dir):
        shutil.rmtree(args.summary_dir)
    os.makedirs(args.summary_dir)

    args.metric = args.metric.lower()
    args.watch_metrics = [m.lower() for m in args.watch_metrics]
    if args.metric not in args.watch_metrics:
        args.watch_metrics.append(args.metric)
    args.cuda = args.cuda and torch.cuda.is_available()

    # Schedule settings are given in samples and used in optimiser updates. The
    # conversion divides by batch_size only: with gradient accumulation an update
    # consumes batch_size * grad_accum_steps samples, so every *_samples value is
    # in effect grad_accum_steps times larger than written (x4 for the paper's
    # configs). Kept as is, because the paper's runs were trained this way.
    def samples2steps(n):
        return int(math.ceil(n / args.batch_size))

    args.log_per_updates = samples2steps(args.log_per_samples)
    args.eval_per_updates = samples2steps(args.eval_per_samples)
    args.eval_per_updates_warmup = samples2steps(args.eval_per_samples_warmup)
    args.eval_warmup_steps = samples2steps(args.eval_warmup_samples)
    args.min_steps = samples2steps(args.min_samples)
    args.early_stopping = samples2steps(args.tolerance_samples)
    args.lr_warmup_steps = samples2steps(args.lr_warmup_samples)
    args.lr_decay_steps = samples2steps(args.lr_decay_samples)
    args.summary_per_updates = args.summary_per_logs * args.log_per_updates
    assert args.lr >= args.min_lr, 'initial learning rate must be larger than min_lr'
    if args.time_limit:
        h, m, s = args.time_limit.split(':')
        args.time_limit_seconds = int(h) * 3600 + int(m) * 60 + int(s)
    else:
        args.time_limit_seconds = None


def validate_params(args):
    """Checks that need the interface's additions (num_classes, num_vocab)."""
    assert args.num_classes == 2, 'this task is binary'
    assert args.num_vocab > 0


def _validate_param(name):
    name = name.replace('-', '_')
    if not str.isidentifier(name):
        raise ValueError(f'Invalid param name: {name}')
    return name
