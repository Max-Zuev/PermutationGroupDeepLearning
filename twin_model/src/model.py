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
"""Model = network + optimiser + learning-rate schedule + checkpointing.

Units. The run config gives the schedule in SAMPLES (lr_warmup_samples,
lr_decay_samples, eval_per_samples, tolerance_samples, ...). params.py converts
each to optimiser UPDATES by dividing by batch_size. With gradient accumulation
(grad_accum_steps = 4 in the paper's configs) one update consumes
batch_size * grad_accum_steps samples, so every "samples" setting is in effect
multiplied by grad_accum_steps: lr_warmup_samples = 20480 means warm-up over
20480 * 4 = 81920 samples. The paper's numbers must be read with that factor.

The learning rate: linear warm-up from min_lr to lr over lr_warmup_steps
updates, then step decay lr * lr_decay_rate^floor((t - warm-up) / lr_decay_steps),
floored at min_lr. Training stops when the rate decays below min_lr, when the
evaluation score has not improved for early_stopping updates, when the loss
exceeds max_loss, or at the wall-clock limit (trainer.py).
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn.functional as f
from tqdm import tqdm
from .network import Network
from .utils.metrics import registry as metrics


class Model:
    prefix = 'checkpoint'
    best_model_name = 'best.pt'

    def __init__(self, args, state_dict=None):
        self.args = args
        self.network = Network(args)
        self.device = torch.cuda.current_device() if args.cuda else torch.device('cpu')
        self.network.to(self.device)
        self.params = list(filter(lambda x: x.requires_grad, self.network.parameters()))
        self.opt = torch.optim.Adam(self.params, args.lr, betas=(args.beta1, args.beta2),
                                    weight_decay=args.weight_decay)
        self.opt.zero_grad()
        self._accum_count = 0                   # micro-batches since the last optimiser step
        self.updates = state_dict['updates'] if state_dict else 0

        if state_dict:
            new_state = set(self.network.state_dict().keys())
            for k in list(state_dict['model'].keys()):
                if k not in new_state:          # tolerate parameters of removed options
                    del state_dict['model'][k]
            self.network.load_state_dict(state_dict['model'])
            self.opt.load_state_dict(state_dict['opt'])

    def _update_schedule(self):
        """Set the learning rate for the current update count (see module docstring)."""
        if self.args.lr_decay_rate < 1.:
            args = self.args
            t = self.updates
            base_ratio = args.min_lr / args.lr
            if t < args.lr_warmup_steps:
                ratio = base_ratio + (1. - base_ratio) / max(1., args.lr_warmup_steps) * t
            else:
                ratio = max(base_ratio, args.lr_decay_rate ** math.floor((t - args.lr_warmup_steps) /
                                                                         args.lr_decay_steps))
            self.opt.param_groups[0]['lr'] = args.lr * ratio

    def update(self, batch):
        """One micro-batch of training. Returns stats after an optimiser step, else None.

        Gradients accumulate over grad_accum_steps micro-batches before a step,
        so the effective batch is batch_size * grad_accum_steps (8 * 4 = 32 in
        the paper). The loss is scaled by 1/grad_accum_steps so the accumulated
        gradient is the mean over the effective batch.
        """
        self.network.train()
        grad_accum_steps = self.args.grad_accum_steps

        inputs, target = self.process_data(batch)
        output = self.network(inputs)
        summary = self.network.get_summary()
        loss = self.get_loss(output, target) / grad_accum_steps
        loss.backward()

        self._accum_count += 1
        if self._accum_count % grad_accum_steps != 0:
            return None

        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_clipping)
        assert grad_norm >= 0, 'encounter nan in gradients.'
        if isinstance(grad_norm, torch.Tensor):
            grad_norm = grad_norm.item()
        self.opt.step()
        self.opt.zero_grad()
        self._update_schedule()
        self.updates += 1
        return {
            'updates': self.updates,
            'loss': loss.item() * grad_accum_steps,   # the unscaled loss of the last micro-batch
            'lr': self.opt.param_groups[0]['lr'],
            'gnorm': grad_norm,
            'summary': summary,
        }

    def evaluate(self, data):
        """Score an evaluation set (list of batches).

        Returns (score, stats, mistakes): score is the config's `metric`, stats
        holds every watched metric and the mean loss, mistakes lists the
        (id1, id2, FALSE_POSITIVE | FALSE_NEGATIVE) of each misclassified pair.
        """
        self.network.eval()
        targets, probabilities, predictions, losses, mistakes = [], [], [], [], []
        for batch in tqdm(data[:self.args.eval_subset], desc='evaluating', leave=False):
            inputs, target = self.process_data(batch)
            with torch.no_grad():
                output = self.network(inputs)
                loss = self.get_loss(output, target)
                pred = torch.argmax(output, dim=1)
                prob = torch.nn.functional.softmax(output, dim=1)
                for i in range(len(pred)):
                    if pred[i] != target[i]:
                        record = (inputs['id1'][i], inputs['id2'][i],
                                  "FALSE_POSITIVE" if pred[i].item() == 1 else "FALSE_NEGATIVE")
                        if record not in mistakes:
                            mistakes.append(record)
                losses.append(loss.item())
                targets.extend(target.tolist())
                probabilities.extend(prob.tolist())
                predictions.extend(pred.tolist())
        outputs = {'target': targets, 'prob': probabilities, 'pred': predictions, 'args': self.args}
        stats = {
            'updates': self.updates,
            # the last batch of a bucket can be partial; leave it out of the mean
            'loss': sum(losses[:-1]) / (len(losses) - 1) if len(losses) > 1 else sum(losses),
        }
        for metric in self.args.watch_metrics:
            if metric not in stats:
                stats.update(metrics[metric](outputs))
        assert 'score' not in stats, 'metric name collides with "score"'
        eval_score = stats[self.args.metric]
        stats['score'] = eval_score
        return eval_score, stats, mistakes

    def predict(self, batch):
        """Class probabilities [B, 2] for one batch (no labels needed)."""
        self.network.eval()
        inputs, _ = self.process_data(batch)
        with torch.no_grad():
            output = self.network(inputs)
            output = torch.nn.functional.softmax(output, dim=1)
        return output.tolist()

    def process_data(self, batch):
        """Batch (dict of lists) -> tensors on the device.

        For each generator k, text{1,2}_k becomes a LongTensor of token ids with
        the -1 padding clamped to 0 (id 0 is a real but masked token), and
        mask{1,2}_k [B, L, 1] marks the real tokens. The encoder zeroes masked
        positions, the alignment masks them out of the softmax, and the pooling
        ignores them, so the padding never influences the output.
        """
        inputs = {'id1': batch['id1'], 'id2': batch['id2']}
        k = 1
        while f'text1_{k}' in batch:
            t1 = torch.LongTensor(batch[f'text1_{k}']).to(self.device)
            t2 = torch.LongTensor(batch[f'text2_{k}']).to(self.device)
            inputs[f'mask1_{k}'] = torch.ne(t1, -1).unsqueeze(2)
            inputs[f'mask2_{k}'] = torch.ne(t2, -1).unsqueeze(2)
            inputs[f'text1_{k}'] = torch.clamp(t1, min=0)
            inputs[f'text2_{k}'] = torch.clamp(t2, min=0)
            k += 1
        if 'target' in batch:
            target = torch.LongTensor(batch['target']).to(self.device)
            return inputs, target
        return inputs, None

    @staticmethod
    def get_loss(logits, target):
        return f.cross_entropy(logits, target)

    def save(self, states, name=None):
        """Checkpoint: network and optimiser state, the run args, RNG states, and
        the trainer's `states`. The embedding table is also saved beside it as
        <name>.embeddings.npy for inspection without loading the model."""
        if name:
            filename = os.path.join(self.args.summary_dir, name)
        else:
            filename = os.path.join(self.args.summary_dir, f'{self.prefix}_{self.updates}.pt')
        params = {
            'state_dict': {
                'model': self.network.state_dict(),
                'opt': self.opt.state_dict(),
                'updates': self.updates,
            },
            'args': self.args,
            'random_state': random.getstate(),
            'torch_state': torch.random.get_rng_state()
        }
        params.update(states)
        if self.args.cuda:
            params['torch_cuda_state'] = torch.cuda.get_rng_state()
        torch.save(params, filename)
        emb_weights = self.network.embedding.embedding.weight.detach().cpu().numpy()
        np.save(filename.replace('.pt', '.embeddings.npy'), emb_weights)

    @classmethod
    def load(cls, file):
        """Rebuild a Model from a checkpoint; returns (model, checkpoint dict).

        The checkpoint stores the run args object, so weights_only=False is
        required (torch >= 2.6 defaults to True). Checkpoints are this
        project's own files.
        """
        checkpoint = torch.load(file, map_location='cpu', weights_only=False)
        prev_args = checkpoint['args']
        prev_args.output_dir = os.path.dirname(os.path.dirname(file))
        prev_args.summary_dir = os.path.join(prev_args.output_dir, prev_args.name)
        prev_args.cuda = prev_args.cuda and torch.cuda.is_available()
        return cls(prev_args, state_dict=checkpoint['state_dict']), checkpoint

    def num_parameters(self):
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)
