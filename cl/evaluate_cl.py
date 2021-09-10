import jax
import jax.numpy as jnp
import csv
import os
import pickle
import uncertainty_metrics.numpy as um
import utils


class Evaluate_cl:
    def __init__(
            self,
            apply_fn,
            loss_fn,
            loss_fn_cl,
            **kwargs,
    ):
        self.apply_fn = apply_fn
        self.llk_fn = loss_fn
        self.loss_fn_cl = loss_fn_cl
        self.kwargs = kwargs['kwargs']

        self.acc_dict = {}
        self.average_acc_list = []
        self.llk_dict = {}
        self.loss_value_dict = {}

    def evaluate_per_epoch(self, x_train, y_train, params, params_last, params_list,
                           state, rng_key, ind_points, ind_id, fisher, task_id, batch_size):
        llk_ = 0
        loss_value_ = 0
        for batch_idx in range(int(x_train.shape[0] / batch_size)):
            image = x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
            label = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
            llk = self.llk_fn(params, params, state, rng_key, image, label, task_id)[0]
            llk_ += llk
            loss_value = self.loss_fn_cl(params, params_last, params_list, state, rng_key, image, label, task_id,
                                         ind_points, ind_id, fisher)[0]
            loss_value_ += loss_value
        llk_ /= (batch_idx + 1)
        loss_value_ /= (batch_idx + 1)
        self.loss_value_dict[f'{str(task_id)}'].append(loss_value_)
        self.llk_dict[f'{str(task_id)}'].append(llk_)
        return 0

    def evaluate_per_task(self, test_ids, x_testsets, y_testsets, params, state, rng_key, batch_size):
        acc = []
        for i in range(len(x_testsets)):
            x_test, y_test = x_testsets[i], y_testsets[i]
            task_id = test_ids[i] * jnp.ones(x_test.shape[0])
            pred = jax.nn.softmax(self.apply_fn(params, state, rng_key, x_test, task_id)[0], axis=1)
            pred_y = jnp.argmax(pred, axis=1)
            y = jnp.argmax(y_test, axis=1)
            cur_acc = len(jnp.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
            acc.append(cur_acc)

        self.acc_dict[f'{str(len(test_ids))}'] = acc
        self.average_acc_list.append(jnp.array(acc).mean())
        return acc, jnp.array(acc).mean()

    def save_log(self, task_id, acc):
        file_name = f'{self.kwargs["save_path"]}/metrics.csv'
        with open(file_name, 'a') as metrics_file:
            metrics_header = [
                'Task ID',
                'Test Acc',
            ]
            writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
            if os.stat(file_name).st_size == 0:
                writer.writeheader()
            writer.writerow({
                'Task ID': task_id,
                'Test Acc': acc
            })
            metrics_file.close()

    def save_params(self, task_id, params, state):
        with open(f'{self.kwargs["save_path"]}/params_{task_id}', "wb") as file:
            pickle.dump(params, file)
        with open(f'{self.kwargs["save_path"]}/state_{task_id}', "wb") as file:
            pickle.dump(state, file)