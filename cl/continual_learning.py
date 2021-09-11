import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
path = '/import/home/xzhoubi/hudson/function_map/cl'
os.chdir(path)
print(os.getcwd())
import sys
sys.path.append('..')
import numpy as np
import copy
import jax
from jax import jit
import matplotlib.pyplot as plt
import jax.numpy as jnp
from tqdm import tqdm
import haiku as hk
import argparse
import optax
import cl.evaluate_cl as evaluate_cl
import cl.network_cl as network_cl
import cl.dataset_cl as dataset_cl
import cl.utils_cl as utils_cl
import cl.loss_cl as loss_cl

# from jax import config
# config.update('jax_disable_jit', True)
# config.update('jax_platform_name', 'cpu')
# config.update("jax_enable_x64", True)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='pmnist')
parser.add_argument('--method', type=str, default='nothing')
parser.add_argument('--reg_first', type=float, default='0.0')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--dummy_num', type=int, default=40)
parser.add_argument('--bs', type=int, default=200)
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--ind_method', type=str, default='core')
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--element_wise', action="store_true", default=False)
parser.add_argument('--data_augmentation', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--epochs_first', type=int, default=10)
parser.add_argument('--coreset_size', type=int, default=200)
parser.add_argument('--coreset_method', type=str, default='random')
parser.add_argument('--head_style', type=str, default='single')
parser.add_argument('--train_on_coreset', action="store_true", default=False)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_path', type=str, default="/home/xzhoubi/hudson/function_map/cl/results")
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()
kwargs = utils_cl.process_args(args)

# Load Data
if args.dataset == 'pmnist':
    class_num = 10
    num_tasks = 10
    data_gen = dataset_cl.PermutedMnistGenerator(num_tasks)
elif args.dataset == 'smnist':
    class_num = 5
    num_tasks = 5
    data_gen = dataset_cl.SplitMnistGenerator()
else:
    raise NotImplementedError(args.dataset)
train_size, in_dim, out_dim = data_gen.get_dims()

# Load MLP
model = network_cl.MLP(output_dim=class_num,
                       architecture=[args.hidden_dim, args.hidden_dim],
                       head_style=args.head_style)
init_fn, apply_fn = hk.transform_with_state(model.forward_fn)
x_init = jnp.ones([1, in_dim])
rng_key = jax.random.PRNGKey(0)
params_init, state = init_fn(rng_key, x_init, task_id=jnp.zeros(1))
params = params_init

# Continual Learning Loss function
loss_cl_list = loss_cl.loss_cl_list(apply_fn=apply_fn,
                                    regularization=args.reg,
                                    dummy_input_dim=args.dummy_num,
                                    head_style=args.head_style,
                                    class_num=class_num,
                                    inverse=args.inverse,
                                    element_wise=args.element_wise)
if args.method == 'weight_l2_with_fisher':
    loss_fun_cl = jax.jit(loss_cl_list.weight_l2_norm_loss_with_fisher)
elif args.method == 'weight_l2_without_fisher':
    loss_fun_cl = jax.jit(loss_cl_list.weight_l2_norm_loss_without_fisher)
elif args.method == 'function_l2':
    loss_fun_cl = jax.jit(loss_cl_list.f_l2_norm_loss)
elif args.method == 'ntk_norm':
    loss_fun_cl = jax.jit(loss_cl_list.ntk_norm_loss)
elif args.method == 'ntk_norm_all_prev':
    loss_fun_cl = jax.jit(loss_cl_list.ntk_norm_loss_all_prev)
else:
    raise NotImplementedError(args.method)

loss_fun = jax.jit(loss_cl_list.llk_classification)
# Optimizer Initialization
opt = optax.adam(args.lr)
opt_state = opt.init(params_init)

# Hyperparameter
batch_size = args.bs


@jit
def update(params: hk.Params,
           state: hk.State,
           opt_state: optax.OptState,
           rng_key: jnp.array,
           x,
           y,
           task_id
           ):
    params_copy = params
    grads, new_state = jax.grad(loss_fun, argnums=0, has_aux=True)(params, params_copy, state, rng_key, x, y, task_id)
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
              task_id,
              ind_points,
              ind_id,
              fisher
              ):
    grads, new_state = jax.grad(loss_fun_cl, argnums=0, has_aux=True)(params, params_last, params_list,
                                                                      state, rng_key, x, y, task_id, ind_points,
                                                                      ind_id, fisher)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state


Evaluate_cl = evaluate_cl.Evaluate_cl(apply_fn=apply_fn,
                                      loss_fn=loss_fun,
                                      loss_fn_cl=loss_fun_cl,
                                      kwargs=kwargs)

x_coresets, y_coresets = [], []
x_testsets, y_testsets = [], []
params_list = []
test_ids = []
coreset_ids = []

print(f"Start Training for {args.epochs} epochs on {args.dataset}")

for task_id in range(data_gen.max_iter):
    x_train, y_train, x_test, y_test = data_gen.next_task()
    x_testsets.append(x_test)
    y_testsets.append(y_test)
    test_ids.append(task_id)

    # Reset logging state for debugging
    Evaluate_cl.llk_dict[str(task_id)] = []
    Evaluate_cl.loss_value_dict[str(task_id)] = []
    Evaluate_cl.acc_dict_all = {'0': [],
                                '1': [],
                                '2': [],
                                '3': [],
                                '4': [],
                                '5': [],
                                '6': [],
                                '7': [],
                                '8': [],
                                '9': []}

    if task_id == 0:
        batch_num = int(x_train.shape[0] / batch_size)
        for _ in tqdm(range(args.epochs_first)):
            for batch_idx in range(batch_num):
                rng_key, _ = jax.random.split(rng_key)
                image = x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                label = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                params, state, opt_state = update(params, state, opt_state, rng_key, image, label, task_id)
    else:
        params_last = params
        batch_num = int(x_train.shape[0] / batch_size)
        for _ in tqdm(range(args.epochs)):
            for batch_idx in range(batch_num):
                rng_key, _ = jax.random.split(rng_key)
                image = x_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                label = y_train[batch_idx * batch_size:(batch_idx + 1) * batch_size, :]
                ind_points, ind_id = utils_cl.ind_points_selection(x_coresets_train, coreset_ids_train, task_id, image,
                                                                   args.dummy_num, args.ind_method)
                params, state, opt_state = update_cl(params, params_last, params_list, state,
                                                     opt_state, rng_key, image, label, task_id, ind_points, ind_id,
                                                     fisher)

            Evaluate_cl.evaluate_per_epoch(x_train, y_train, x_testsets, y_testsets, test_ids,
                                           params, params_last, params_list,
                                           state, rng_key, ind_points, ind_id, fisher,
                                           task_id=task_id,
                                           batch_size=1000)
    # Generate Coreset
    x_coresets, y_coresets, coreset_ids = utils_cl.coreset_selection(x_coresets, y_coresets, coreset_ids, x_train,
                                                                     y_train, task_id,
                                                                     args.coreset_method, args.coreset_size)
    x_coresets_train, y_coresets_train, coreset_ids_train = utils_cl.merge_coresets(x_coresets, y_coresets, coreset_ids)

    # Put params in list, put preds in list, put id in list
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

        fig = plt.figure(figsize=(15, 10))
        ax_all = fig.subplots(2, 5).flatten()
        for i in range(len(x_testsets)):
            ax_all[i].plot(np.arange(args.epochs), np.array(Evaluate_cl.acc_dict_all[f'{str(i)}']))
            ax_all[i].set_title(f"Acc on Task #{i}")
        plt.show()

    # Train on coreset
    params_eval = copy.deepcopy(params)
    state_eval = copy.deepcopy(state)
    opt_state_eval = copy.deepcopy(opt_state)

    if args.train_on_coreset and task_id > 0:
        coreset_batch_size = args.coreset_size
        for _ in range(args.epochs):
            for batch_idx in range(int(x_coresets_train.shape[0] / coreset_batch_size)):
                rng_key, _ = jax.random.split(rng_key)
                image = x_coresets_train[batch_idx * coreset_batch_size:(batch_idx + 1) * coreset_batch_size, :]
                label = y_coresets_train[batch_idx * coreset_batch_size:(batch_idx + 1) * coreset_batch_size, :]

                # Sample inducing points
                idx_batch = np.random.permutation(np.arange(image.shape[0]))[:args.dummy_num]
                ind_points = image[idx_batch, :]
                # ind_id = jnp.ones(ind_points.shape[0]) *

                params_eval, state_eval, opt_state_eval = update_cl(params_eval, params, params_list, state_eval,
                                                                    opt_state_eval,
                                                                    rng_key, image, label, task_id,
                                                                    ind_points, ind_id, fisher)

    acc_list, acc = Evaluate_cl.evaluate_per_task(test_ids,
                                                  x_testsets,
                                                  y_testsets,
                                                  params_eval,
                                                  state_eval,
                                                  rng_key,
                                                  batch_size=1000)
    print(f"All Accuracy list: {acc_list}")
    print(f"Mean Accuracy: {acc}")

    if args.method == 'weight_l2_with_fisher':
        x_sample = x_train[:200, :]
        y_sample = y_train[:200, :]
        fisher = utils_cl.get_fisher(x_sample, y_sample, params, state, rng_key,
                                     llk_func=loss_cl_list.llk_classification)
    else:
        fisher = None

    if args.save:
        Evaluate_cl.save_log(task_id, acc)
        Evaluate_cl.save_params(task_id, params, state)

if args.save:
    save_path = kwargs["save_path"]
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    os.rename(save_path, f"{save_path}__complete")