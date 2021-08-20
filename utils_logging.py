import jax
import jax.numpy as jnp
import csv
import os
import pickle
import uncertainty_metrics.numpy as um
import utils

class Evaluate:
    def __init__(
            self,
            apply_fn,
            loss_fn,
            llk_fn,
            **kwargs,
    ):
        self.loss_fn = loss_fn
        self.apply_fn = apply_fn
        self.llk_fn = llk_fn
        self.kwargs = kwargs['kwargs']

    def evaluate(self, loader, params, state, rng_key, batch_size):
        loss_ = 0
        acc_ = 0
        llk_ = 0
        ece_ = 0
        metric = {'loss': {},
                  'acc': {},
                  'llk': {},
                  'ece': {}}

        for batch_idx, (image, label) in enumerate(loader):
            image, label = utils.tensor2array(image, label)
            rng_key, _ = jax.random.split(rng_key)
            loss_value = self.loss_fn(params, params, state, rng_key, image, label)[0]
            loss_ += loss_value

            preds = self.apply_fn(params, state, rng_key, image)[0]
            acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
            acc_ += acc

            llk = self.llk_fn(params, params, state, rng_key, image, label)[0]
            llk_ += llk

            ece = um.ece(label.argmax(-1), jax.nn.softmax(preds, axis=1))
            ece_ += ece

        loss_ /= len(loader)
        acc_ /= len(loader) * batch_size / 100.
        llk_ /= len(loader)
        ece_ /= len(loader)
        metric['loss'] = loss_
        metric['acc'] = acc_
        metric['llk'] = llk_
        metric['ece'] = ece_
        return metric

    def save_log(self, epoch, metric_train, metric_test):
        file_name = f'{self.kwargs["save_path"]}/metrics.csv'
        with open(file_name, 'a') as metrics_file:
            metrics_header = [
                'Epoch',
                'Train Loss',
                'Train LLK',
                'Train Acc',
                'Train ECE',
                'Test Loss',
                'Test LLK',
                'Test Acc',
                'Test ECE',
            ]
            writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
            if os.stat(file_name).st_size == 0:
                writer.writeheader()
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': metric_train['loss'],
                'Train LLK': metric_train['llk'],
                'Train Acc': metric_train['acc'],
                'Train ECE': metric_train['ece'],
                'Test Loss': metric_test['loss'],
                'Test LLK': metric_test['llk'],
                'Test Acc': metric_test['acc'],
                'Test ECE': metric_test['ece'],
            })
            metrics_file.close()

    def save_params(self, epoch, params, state):
        with open(f'{self.kwargs["save_path"]}/params_{epoch}', "wb") as file:
            pickle.dump(params, file)
        with open(f'{self.kwargs["save_path"]}/state_{epoch}', "wb") as file:
            pickle.dump(state, file)