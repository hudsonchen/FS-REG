import numpy as np
import os
import jax
import jax.numpy as jnp
import argparse
from typing import Tuple, List, Dict
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import tree

def get_scores(params, state, apply_fn, rng_key, x_testsets, y_testsets):
    acc = []
    for i in range(len(x_testsets)):
        x_test, y_test = x_testsets[i], y_testsets[i]

        pred = jax.nn.softmax(apply_fn(params, state, rng_key, x_test)[0], axis=1)
        pred_y = jnp.argmax(pred, axis=1)
        y = jnp.argmax(y_test, axis=1)
        cur_acc = len(jnp.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)
    return acc


def coreset_selection(x_coreset, y_coreset, x_train, y_train, task_id, coreset_method, coreset_size):
    # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    coreset_id = jnp.ones(coreset_size) * task_id
    if coreset_method == 'k_means':
        dists = np.full(x_train.shape[0], np.inf)
        current_id = 0
        dists = update_distance(dists, x_train, current_id)
        idx = [current_id]

        for i in range(1, coreset_size):
            current_id = np.argmax(dists)
            dists = update_distance(dists, x_train, current_id)
            idx.append(current_id)

        x_coreset.append(x_train[idx, :])
        y_coreset.append(y_train[idx, :])
    elif coreset_method == 'random':
        idx = np.random.permutation(np.arange(x_train.shape[0]))[:coreset_size]
        x_coreset.append(x_train[idx, :])
        y_coreset.append(y_train[idx, :])
    else:
        raise NotImplementedError(coreset_method)
    return x_coreset, y_coreset, coreset_id


def update_distance(dists, x_train, current_id):
    for i in range(x_train.shape[0]):
        current_dist = np.linalg.norm(x_train[i, :] - x_train[current_id, :])
        dists[i] = np.minimum(current_dist, dists[i])
    return dists


def merge_coresets(x_coresets, y_coresets):
    merged_x, merged_y = x_coresets[0], y_coresets[0]
    for i in range(1, len(x_coresets)):
        merged_x = np.vstack((merged_x, x_coresets[i]))
        merged_y = np.vstack((merged_y, y_coresets[i]))
    # seq = np.random.permutation(np.arange(merged_x.shape[0]))
    # merged_x = merged_x[seq, :]
    # merged_y = merged_y[seq, :]
    return merged_x, merged_y


def ind_points_selection(coreset, coreset_id, batch, ind_size, ind_method):
    if ind_method == 'both':
        idx_coreset = np.random.permutation(np.arange(coreset.shape[0]))[:int(ind_size / 2)]
        ind_point_from_coreset = coreset[idx_coreset, :]
        idx_batch = np.random.permutation(np.arange(batch.shape[0]))[:int(ind_size / 2)]
        ind_point_from_batch = batch[idx_batch, :]
        ind_points = np.concatenate((ind_point_from_batch, ind_point_from_coreset), axis=0)
        ind_id = None
    elif ind_method == 'core':
        idx_coreset = np.random.permutation(np.arange(coreset.shape[0]))[:int(ind_size)]
        ind_point_from_coreset = coreset[idx_coreset, :]
        ind_points = ind_point_from_coreset
        ind_id = coreset_id[idx_coreset]
    elif ind_method == 'train':
        idx_batch = np.random.permutation(np.arange(batch.shape[0]))[:int(ind_size)]
        ind_point_from_batch = batch[idx_batch, :]
        ind_points = ind_point_from_batch
        ind_id = None
    else:
        raise NotImplementedError(ind_method)
    return ind_points, ind_id


def process_args(args: argparse.Namespace) -> Dict:
    kwargs = vars(args)
    save_path = args.save_path.rstrip()
    save_path = (
        f"{save_path}/{args.dataset}/dataset_{args.dataset}__method_{args.method}"
        f"__reg_{args.reg}__seed_{args.seed}__"
    )
    i = 1
    while os.path.exists(f"{save_path}{i}") or os.path.exists(
            f"{save_path}{i}__complete"
    ):
        i += 1
    save_path = f"{save_path}{i}"
    kwargs["save_path"] = save_path
    if args.save:
        os.mkdir(save_path)
    return kwargs


def get_fisher(x_sample, y_sample, params, state, rng_key, llk_func):
    params_copy = params
    grads_sum = tree_map(lambda p: p * 0, params)
    for i in range(x_sample.shape[0]):
        x = x_sample[i, :][None]
        y = y_sample[i, :][None]
        grads, new_state = jax.grad(llk_func, argnums=0, has_aux=True)(params, params_copy, state, rng_key, x, y)
        grads_square = tree_map(lambda p: p ** 2, grads)
        grads_sum = tree_map(lambda p, q: p + q, grads_square, grads_sum)
    return tree_map(lambda p: p / x_sample.shape[0], grads_sum)


def zero_params(params):
    return tree.map_structure(lambda p: p * 0, params)
