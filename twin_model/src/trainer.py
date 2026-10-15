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
"""The training loop.

For each epoch the training CSV is streamed in shuffled batches
(interface.stream_batches); every micro-batch goes through model.update, which
takes an optimiser step once grad_accum_steps micro-batches have accumulated.
Every eval_per_updates steps (eval_per_updates_warmup during the first
eval_warmup_steps) the evaluation set is scored; the checkpoint with the best
score so far is kept as best.pt. Training ends at the configured number of
epochs or earlier when: the score has not improved for early_stopping updates
(after min_steps), the loss exceeds max_loss, the learning rate has decayed
below min_lr, or the wall-clock limit is 5 minutes away. All of these counts
are in optimiser updates; params.py derives them from the config's sample
counts (see model.py on the grad_accum_steps factor).

Logging goes to <summary_dir>/log.txt (and tensorboard if enabled); run
tracking to wandb per wandb.mode in twin_model/training.json.
"""

import os
import random
import json5
import torch
from datetime import datetime
from pprint import pformat
from tqdm import tqdm
from .utils.loader import load_data
from .utils.logger import Logger
from .utils.params import validate_params
from .model import Model
from .interface import Interface


class Trainer:
    def __init__(self, args):
        self.args = args
        self.log = Logger(self.args)

    def train(self):
        start_time = datetime.now()
        model, interface, states = self.build_model()
        dev = load_data(self.args.data_dir, self.args.eval_file, self.args.part_length)
        self.log(f'Eval file: {self.args.eval_file} ({len(dev)})')
        dev_batches = interface.pre_process(dev, training=False)
        self.log('setup complete: {}s.'.format(str(datetime.now() - start_time).split(".")[0]))

        try:
            for epoch in range(states['start_epoch'], self.args.epochs + 1):
                states['epoch'] = epoch
                self.log.set_epoch(epoch)
                batches = interface.stream_batches(self.args.data_dir, self.args.train_file,
                                                   shuffle_buffer=self.args.shuffle_buffer)
                pbar = tqdm(batches, desc=f'Epoch {epoch}', unit='batch', dynamic_ncols=True)
                for batch in pbar:
                    stats = model.update(batch)
                    if stats is None:
                        continue                    # gradients accumulating, no step yet
                    self.log.update(stats)
                    pbar.set_postfix(loss=f"{stats['loss']:.4f}", lr=f"{stats['lr']:.2e}",
                                     gnorm=f"{stats['gnorm']:.2f}", step=model.updates)
                    train_log = {f'train/{k}': v for k, v in stats.items() if k != 'summary'}
                    train_log['epoch'] = epoch
                    self.wandb.log(train_log, step=model.updates)

                    eval_per_updates = self.args.eval_per_updates \
                        if model.updates > self.args.eval_warmup_steps else self.args.eval_per_updates_warmup
                    if model.updates % eval_per_updates == 0:
                        pbar.clear()
                        self.log.newline()
                        score, dev_stats, mistakes = model.evaluate(dev_batches)
                        if score > states['best_eval']:
                            states['best_eval'], states['best_epoch'], states['best_step'] = score, epoch, model.updates
                            if self.args.save:
                                model.save(states, name=model.best_model_name)
                        self.log.log_eval(dev_stats)
                        self.log_mistakes(mistakes)
                        fp = sum(m[2] == "FALSE_POSITIVE" for m in mistakes)
                        fn = len(mistakes) - fp
                        eval_log = {f'eval/{k}': v for k, v in dev_stats.items() if k not in ('updates', 'score')}
                        eval_log.update({
                            'eval/score': score, 'eval/best_score': states['best_eval'],
                            'eval/false_positives': fp, 'eval/false_negatives': fn,
                            'eval/total_mistakes': len(mistakes),
                            # how lopsided the errors are: +1 all false positives, -1 all false
                            # negatives, 0 balanced (the model's bias swings during training)
                            'eval/fp_fn_balance': (fp - fn) / (fp + fn) if fp + fn else 0.0,
                            'epoch': epoch,
                        })
                        self.wandb.log(eval_log, step=model.updates)
                        if model.updates - states['best_step'] > self.args.early_stopping \
                                and model.updates > self.args.min_steps:
                            raise EarlyStop('[Tolerance reached. Training is stopped early.]')
                    if stats['loss'] > self.args.max_loss:
                        raise EarlyStop('[Loss exceeds tolerance. Unstable training is stopped early.]')
                    if stats['lr'] < self.args.min_lr - 1e-6:
                        raise EarlyStop('[Learning rate has decayed below min_lr. Training is stopped early.]')
                    if self.args.time_limit_seconds is not None:
                        remaining = self.args.time_limit_seconds - (datetime.now() - start_time).total_seconds()
                        if remaining < 300:
                            raise EarlyStop(f'[Time limit reached. {remaining:.0f}s remaining, stopping early.]')
                pbar.close()
                self.log.newline()
            self.log('Training complete.')
        except KeyboardInterrupt:
            self.log.newline()
            self.log('Training interrupted. Stopped early.')
        except EarlyStop as e:
            self.log.newline()
            self.log(str(e))
        self.log(f'best dev score {states["best_eval"]} at step {states["best_step"]} '
                 f'(epoch {states["best_epoch"]}).')
        self.log(f'best eval stats [{self.log.best_eval_str}]')
        training_time = str(datetime.now() - start_time).split('.')[0]
        self.log(f'Training time: {training_time}.')
        states['start_time'] = str(start_time).split('.')[0]
        states['training_time'] = training_time
        for k in ('best_eval', 'best_epoch', 'best_step', 'training_time'):
            self.wandb.summary[k] = states[k]
        self.wandb.finish()
        return states

    def log_mistakes(self, mistakes):
        """List every misclassified evaluation pair in the log (ids only)."""
        self.log('============ [Mistakes] ============')
        for kind in ("FALSE_POSITIVE", "FALSE_NEGATIVE"):
            self.log(f'{kind.replace("_", " ").title()}s:')
            for m in mistakes:
                if m[2] == kind:
                    self.log(m)
        fp = sum(m[2] == "FALSE_POSITIVE" for m in mistakes)
        self.log(f"Total mistakes: {len(mistakes)}, FP: {fp}, FN: {len(mistakes) - fp}")
        self.log('===================================')
        self.log.newline()

    def build_model(self):
        interface = Interface(self.args, self.log)     # sets num_classes, num_vocab
        self.log(f'#classes: {self.args.num_classes}; #vocab: {self.args.num_vocab}')
        if self.args.seed:
            random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.cuda:
                torch.cuda.manual_seed(self.args.seed)
            if self.args.deterministic:
                torch.backends.cudnn.deterministic = True
        model = Model(self.args)

        states = {'start_epoch': 1, 'best_eval': 0., 'best_epoch': 0, 'best_step': 0}
        self.log(f'trainable params: {model.num_parameters():,d}')
        validate_params(self.args)
        with open(os.path.join(self.args.summary_dir, 'args.json5'), 'w') as f:
            json5.dump(self.args.__dict__, f, indent=2)
        self.log(pformat(vars(self.args), indent=2, width=120))

        # run tracking, per training.json wandb.mode: 'online' or 'offline'
        # (wandb; offline logs locally for a later `wandb sync`), or 'disabled' (no
        # logging, wandb not imported). The logged config carries every field of the
        # run config (train_orders, embedding_kind, ...) plus the cluster job and node.
        if self.args.wandb_mode != 'disabled':
            import socket
            import wandb
            wandb_config = dict(vars(self.args))
            wandb_config['slurm_job_id'] = os.environ.get('SLURM_JOB_ID')
            wandb_config['slurm_job_name'] = os.environ.get('SLURM_JOB_NAME')
            wandb_config['node'] = socket.gethostname()
            wandb.init(
                mode=self.args.wandb_mode,
                entity=self.args.wandb_entity,
                project=self.args.wandb_project,
                name=getattr(self.args, 'wandb_run_name', None),
                tags=getattr(self.args, 'wandb_tags', None),
                config=wandb_config,
                dir=self.args.summary_dir,
            )
            self.wandb = wandb
        else:
            self.wandb = NoWandb()
        return model, interface, states


class NoWandb:
    """Stand-in for the wandb module when training.wandb.mode is 'disabled'."""
    summary = {}

    def log(self, *args, **kwargs):
        pass

    def finish(self):
        pass


class EarlyStop(Exception):
    pass
