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
"""Evaluation metrics over a whole evaluation set.

Each takes outputs = {'target': [...], 'pred': [...], 'prob': [[p0, p1], ...]}
and returns {metric name: value}. The run config's `metric` selects the one
used for early stopping and checkpoint selection (the paper: accuracy);
`watch_metrics` lists the others to report.
"""

from functools import partial
import numpy as np
from sklearn import metrics

from .registry import register

registry = {}
register = partial(register, registry=registry)


@register('acc')
def acc(outputs):
    return {'acc': metrics.accuracy_score(outputs['target'], outputs['pred'])}


@register('f1')
def f1(outputs):
    return {'f1': metrics.f1_score(outputs['target'], outputs['pred']).item()}


@register('auc')
def auc(outputs):
    prob = np.array(outputs['prob'])
    return {'auc': metrics.roc_auc_score(outputs['target'], prob[:, 1])}
