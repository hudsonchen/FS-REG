import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import numpy as np
import copy
import jax
from jax import jit
import matplotlib.pyplot as plt
import jax.numpy as jnp
import dataset
import network
from tqdm import tqdm
import haiku as hk
import loss_classification
import argparse
import optax
from typing import Tuple, Callable, List
import utils_cl
import loss_cl
import utils_logging

# from jax import config
# config.update('jax_disable_jit', True)
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pmnist')
parser.add_argument('--method', type=str, default='nothing')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--dummy_num', type=int, default=40)
parser.add_argument('--ind_method', type=str, default='core')
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--element_wise', action="store_true", default=False)
parser.add_argument('--data_augmentation', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--coreset_size', type=int, default=200)
parser.add_argument('--coreset_method', type=str, default='random')
parser.add_argument('--train_on_coreset', action="store_true", default=False)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_path', type=str, default="/home/xzhoubi/hudson/function_map/continual_learning/results")
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
kwargs = utils_cl.process_args(args)

# Load Data
class_num = 10
num_tasks = 10
data_gen = dataset.PermutedMnistGenerator(num_tasks)

train_size, in_dim, out_dim = data_gen.get_dims()

# Load MLP
model = network.MLP(output_dim=class_num,
                    architecture=[100, 100])
init_fn, apply_fn = hk.transform_with_state(model.forward_fn)
x_init = jnp.ones([1, in_dim])
rng_key = jax.random.PRNGKey(0)
params_init, state = init_fn(rng_key, x_init)
params = params_init

# Loss function
loss_classification_list = loss_classification.loss_classification_list(apply_fn=apply_fn,
                                                                        regularization=args.reg,
                                                                        dummy_input_dim=args.dummy_num,
                                                                        class_num=class_num,
                                                                        inverse=args.inverse,
                                                                        element_wise=args.element_wise)
loss_fun = loss_classification_list.llk_classification

# Continual Learning Loss function
loss_cl_list = loss_cl.loss_cl_list(apply_fn=apply_fn,
                                    regularization=args.reg,
                                    dummy_input_dim=args.dummy_num,
                                    class_num=class_num,
                                    inverse=args.inverse,
                                    element_wise=args.element_wise)
if args.method == 'nothing':
    loss_fun_cl = jax.jit(loss_cl_list.llk_classification)
elif args.method == 'weight_l2':
    loss_fun_cl = jax.jit(loss_cl_list.weight_l2_norm_loss)
elif args.method == 'function_l2':
    loss_fun_cl = jax.jit(loss_cl_list.f_l2_norm_loss)
elif args.method == 'ntk_norm':
    loss_fun_cl = jax.jit(loss_cl_list.ntk_norm_loss)
elif args.method == 'ntk_norm_all_prev':
    loss_fun_cl = jax.jit(loss_cl_list.ntk_norm_loss_all_prev)
else:
    raise NotImplementedError(args.method)

# Optimizer Initialization
opt = optax.adam(args.lr)
opt_state = opt.init(params_init)

# Hyperparameter
batch_size = 200
batch_num = int(train_size / batch_size)


@jit
def update(params: hk.Params,
           state: hk.State,
           opt_state: optax.OptState,
           rng_key: jnp.array,
           x,
           y,
           ):
    params_copy = params
    grads, new_state = jax.grad(loss_fun, argnums=0, has_aux=True)(params, params_copy, state, rng_key, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state


@jit
def update_cl(params: hk.Params,
              params_last: hk.Params,
              params_list,
              state: hk.State,
              opt_state: optax.OptState,
              rng_key: jnp.array,
              x,
              y,
              ind_points,
              fisher
              ):
    grads, new_state = jax.grad(loss_fun_cl, argnums=0, has_aux=True)(params, params_last, params_list,
                                                                      state, rng_key, x, y, ind_points, fisher)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state


Evaluate_cl = utils_logging.Evaluate_cl(apply_fn=apply_fn,
                                        loss_fn=loss_fun,
                                        loss_fn_cl=loss_fun_cl,
                                        kwargs=kwargs)

x_coresets, y_coresets = [], []
x_testsets, y_testsets = [], []
params_list = []


for task_id in range(data_gen.max_iter):
    x_train, y_train, x_test, y_test = data_gen.next_task()
    x_testsets.append(x_test)
    y_testsets.append(y_test)

    x_coresets, y_coresets, _, _ = utils_cl.coreset_selection(x_coresets, y_coresets, x_train, y_train,
                                                              args.coreset_method, args.coreset_size)
    x_coresets_train, y_coresets_train = utils_cl.merge_coresets(x_coresets, y_coresets)

    Evaluate_cl.llk_dict[str(task_id)] = []
    Evaluate_cl.loss_value_dict[str(task_id)] = []

    if task_id == 0:
        for _ in tqdm(range(args.epochs)):
            for batch_idx in range(batch_num):
                rng_key, _ = jax.random.split(rng_key)
                image = x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                label = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                params, state, opt_state = update(params, state, opt_state, rng_key, image, label)

    else:
        params_last = params
        for _ in tqdm(range(args.epochs)):
            for batch_idx in range(batch_num):
                rng_key, _ = jax.random.split(rng_key)
                image = x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                label = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                ind_points = utils_cl.ind_points_selection(x_coresets_train, image, args.dummy_num, args.ind_method)
                params, state, opt_state = update_cl(params, params_last, params_list, state,
                                                     opt_state, rng_key, image, label, ind_points, fisher)

            Evaluate_cl.evaluate_per_epoch(x_train, y_train, params, params_last, params_list,
                                           state, rng_key, ind_points, fisher,
                                           task_id=task_id,
                                           batch_size=1000)
    # Put params in list
    params_list.append(params)

    # Visual Training Process
    if task_id > 0:
        fig = plt.figure(figsize=(10, 6))
        ax_loss_value, ax_llk = fig.subplots(1, 2).flatten()
        ax_llk.plot(np.arange(args.epochs), np.array(Evaluate_cl.llk_dict[str(task_id)]), label='LLK')
        ax_loss_value.plot(np.arange(args.epochs), np.array(Evaluate_cl.loss_value_dict[str(task_id)]), label='Loss')
        ax_llk.set_title(f"LLK on Task #{task_id}")
        ax_loss_value.set_title(f"Loss on Task #{task_id}")
        plt.show()

    # Train on coreset
    params_eval = copy.deepcopy(params)
    state_eval = copy.deepcopy(state)
    opt_state_eval = copy.deepcopy(opt_state)

    if args.train_on_coreset and task_id > 0:
        core_set_batch_size = args.coreset_size
        for _ in range(args.epochs):
            for batch_idx in range(int(x_coresets_train.shape[0] / core_set_batch_size)):
                rng_key, _ = jax.random.split(rng_key)
                image = x_coresets_train[batch_idx * core_set_batch_size:(batch_idx + 1) * core_set_batch_size, :]
                label = y_coresets_train[batch_idx * core_set_batch_size:(batch_idx + 1) * core_set_batch_size, :]

                # Sample inducing points
                idx_batch = np.random.permutation(np.arange(image.shape[0]))[:args.dummy_num]
                ind_points = image[idx_batch, :]

                params_eval, state_eval, opt_state_eval = update_cl(params_eval, params, params_list, state_eval, opt_state_eval,
                                                                    rng_key, image, label, ind_points, fisher)

    acc_list, acc = Evaluate_cl.evaluate_per_task(task_id,
                                                  x_testsets,
                                                  y_testsets,
                                                  params_eval,
                                                  state_eval,
                                                  rng_key,
                                                  batch_size=1000)
    print(f"All Accuracy list: {acc_list}")
    print(f"Mean Accuracy: {acc}")

    if args.method == 'weight_l2':
        x_sample = x_train[:200, :]
        y_sample = y_train[:200, :]
        fisher = utils_cl.get_fisher(x_sample, y_sample, params, state, rng_key, llk_func=loss_classification_list.llk_classification)
    else:
        fisher = None

    if args.save:
        Evaluate_cl.save_log(task_id, acc)
        Evaluate_cl.save_params(task_id, params, state)
