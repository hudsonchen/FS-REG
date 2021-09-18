import numpy as np
import os
import csv
import pickle
from typing import Tuple
from functools import partial
import jax
from jax import pmap
import flax
import jax.numpy as jnp
import utils

eps = 1e-6
state = {}


class Evaluate:
    def __init__(
            self,
            apply_fn,
            n_devices,
            num_classses,
            **kwargs,
    ):
        self.apply_fn = apply_fn
        self.n_devices = n_devices
        self.num_classes = num_classses
        self.kwargs = kwargs['kwargs']

    @partial(pmap, axis_name='num_devices', static_broadcasted_argnums=(0,))
    def llk_and_acc_fn(self, params, x, y):
        y_hat = jax.nn.softmax(self.apply_fn({'params': params, **state}, x, mutable=list(state.keys()))[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        log_likelihood = jax.lax.pmean(log_likelihood, axis_name='num_devices')

        acc = jnp.equal(jnp.argmax(y_hat, axis=1), jnp.argmax(y, axis=1)).sum()
        acc = jax.lax.psum(acc, axis_name='num_devices')
        return -log_likelihood, acc

    def evaluate(self, loader, params, state, rng_key, batch_size):
        acc_ = 0
        llk_ = 0
        metric = {'acc': {},
                  'llk': {}}

        for batch_idx, (image, label) in enumerate(loader):
            image, label = utils.tensor2array(image, label, self.num_classes)
            rng_key, _ = jax.random.split(rng_key)
            image = utils.split(image, self.n_devices)
            label = utils.split(label, self.n_devices)
            llk, acc = self.llk_and_acc_fn(params, image, label)
            acc_ += acc.mean()
            llk_ += llk.mean()

        acc_ /= len(loader) * batch_size / 100.
        llk_ /= len(loader)

        metric['acc'] = acc_
        metric['llk'] = llk_
        return metric

    def save_log(self, epoch, metric_train, metric_test):
        file_name = f'{self.kwargs["save_path"]}/metrics.csv'
        with open(file_name, 'a') as metrics_file:
            metrics_header = [
                'Epoch',
                'Train LLK',
                'Train Acc',
                'Test LLK',
                'Test Acc',
            ]
            writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
            if os.stat(file_name).st_size == 0:
                writer.writeheader()
            writer.writerow({
                'Epoch': epoch,
                'Train LLK': metric_train['llk'],
                'Train Acc': metric_train['acc'],
                'Test LLK': metric_test['llk'],
                'Test Acc': metric_test['acc'],
            })
            metrics_file.close()
        # Debuggin purpose on slurm
        with open(f'{self.kwargs["save_path"]}/metrics_{epoch}', "wb") as file:
            pickle.dump(metric_test, file)

    def save_params(self, epoch, params, state):
        with open(f'{self.kwargs["save_path"]}/params_{epoch}', "wb") as file:
            pickle.dump(params, file)
        with open(f'{self.kwargs["save_path"]}/state_{epoch}', "wb") as file:
            pickle.dump(state, file)

