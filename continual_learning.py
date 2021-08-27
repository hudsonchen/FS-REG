import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import numpy as np
import copy
import jax
import pickle
from jax import jit
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

# from jax import config

# config.update('jax_disable_jit', True)
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='nothing')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--dummy_num', type=int, default=30)
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--element_wise', action="store_true", default=False)
parser.add_argument('--data_augmentation', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--coreset_size', type=int, default=200)
parser.add_argument('--coreset_method', type=str, default='random')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

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
loss_fun = jax.jit(loss_classification_list.llk_classification)

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


# Optimizer Initialization
opt = optax.adam(args.lr)
opt_state = opt.init(params_init)

# Hyperparameter
batch_size = 100
batch_num = int(train_size / batch_size)


@jit
def update(params: hk.Params,
           params_copy: hk.Params,
           state: hk.State,
           opt_state: optax.OptState,
           rng_key: jnp.array,
           x,
           y,
           ):
    grads, new_state = jax.grad(loss_fun, argnums=0, has_aux=True)(params, params_copy, state, rng_key, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state


# @jit
def update_cl(params: hk.Params,
              params_last: hk.Params,
              state: hk.State,
              opt_state: optax.OptState,
              rng_key: jnp.array,
              x,
              y,
              ind_points
              ):
    grads, new_state = jax.grad(loss_fun_cl, argnums=0, has_aux=True)(params, params_last, state, rng_key, x, y, ind_points)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state


all_acc = np.array([])
x_coresets, y_coresets = [], []
x_testsets, y_testsets = [], []

for task_id in range(data_gen.max_iter):
    x_train, y_train, x_test, y_test = data_gen.next_task()
    x_testsets.append(x_test)
    y_testsets.append(y_test)

    x_coresets, y_coresets, _, _ = utils_cl.coreset_selection(x_coresets, y_coresets, x_train, y_train,
                                                              args.coreset_method, args.coreset_size)
    x_coresets_train, y_coresets_train = utils_cl.merge_coresets(x_coresets, y_coresets)

    if task_id == 0:
        for _ in tqdm(range(args.epochs)):
            for batch_idx in range(batch_num):
                rng_key, _ = jax.random.split(rng_key)
                image = x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                label = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                params, state, opt_state = update(params, params, state, opt_state, rng_key, image, label)
        with open(f'/home/xzhoubi/hudson/function_map/continual_learning/params', "wb") as file:
            params_ = params
            pickle.dump(params_, file)
    else:
        with open(f'/home/xzhoubi/hudson/function_map/continual_learning/params', "rb") as file:
            params_last = pickle.load(file)
        # params_last = params_init
        for _ in tqdm(range(args.epochs)):
            for batch_idx in range(batch_num):
                rng_key, _ = jax.random.split(rng_key)
                image = x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                label = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]

                # Sample inducing points
                idx_coreset = np.random.permutation(np.arange(x_coresets_train.shape[0]))[:int(args.dummy_num / 2)]
                ind_point_from_coreset = x_coresets_train[idx_coreset, :]
                idx_batch = np.random.permutation(np.arange(image.shape[0]))[:int(args.dummy_num / 2)]
                ind_point_from_batch = image[idx_batch, :]
                ind_points = np.concatenate((ind_point_from_batch, ind_point_from_coreset), axis=0)

                params, state, opt_state = update_cl(params, params_last, state, opt_state, rng_key, image, label,
                                                     ind_points)
                # debug
                if batch_idx % 20 == 0:
                    freg, _, _ = loss_cl_list.ntk_norm_loss_temp(params, params_last, state, rng_key, image, label,
                                                                 ind_points)
                    print('freg', freg)

                    # y_hat = apply_fn(params, state, rng_key, x_test)[0]
                    # acc = jnp.mean(y_hat.argmax(-1) == y_test.argmax(-1))
                    # print("Acc", acc)
                    #
                    # y_hat_last = apply_fn(params, state, rng_key, x_testsets[0])[0]
                    # acc_last = jnp.mean(y_hat_last.argmax(-1) == y_testsets[0].argmax(-1))
                    # print("Acc Last", acc_last)


    # Train with Core set
    # params_eval = copy.deepcopy(params)
    # state_eval = copy.deepcopy(state)
    # opt_state_eval = copy.deepcopy(opt_state)

    # core_set_batch_size = args.coreset_size
    # for _ in range(args.epochs):
    #     for batch_idx in range(int(x_coresets_train.shape[0] / core_set_batch_size)):
    #         rng_key, _ = jax.random.split(rng_key)
    #         image = x_coresets_train[batch_idx * core_set_batch_size:(batch_idx + 1) * core_set_batch_size, :]
    #         label = y_coresets_train[batch_idx * core_set_batch_size:(batch_idx + 1) * core_set_batch_size, :]
    #
    #         # Sample inducing points
    #         idx_batch = np.random.permutation(np.arange(image.shape[0]))[:args.dummy_num]
    #         ind_points = image[idx_batch, :]
    #
    #         params_eval, state_eval, opt_state_eval = update_cl(params_eval, params, state_eval, opt_state_eval,
    #                                                             rng_key, image, label, ind_points)

    # Evaluate
    # acc = []
    # for i in range(len(x_testsets)):
    #     x_test, y_test = x_testsets[i], y_testsets[i]
    #     pred = jax.nn.softmax(apply_fn(params_eval, state_eval, rng_key, x_test)[0], axis=1)
    #     pred_y = jnp.argmax(pred, axis=1)
    #     y = jnp.argmax(y_test, axis=1)
    #     cur_acc = len(jnp.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
    #     acc.append(cur_acc)
    # print(acc)

print(f'Final Average Accuracy:{np.array(acc).mean()}')
