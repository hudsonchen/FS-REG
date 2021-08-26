import numpy as np
import jax
import jax.numpy as jnp


def get_scores(params, state, apply_fn, rng_key, x_testsets, y_testsets, x_coresets, y_coresets):
    acc = []
    for i in range(len(x_testsets)):
        x_test, y_test = x_testsets[i], y_testsets[i]

        pred = jax.nn.softmax(apply_fn(params, state, rng_key, x_test)[0], axis=1)
        pred_y = jnp.argmax(pred, axis=1)
        y = jnp.argmax(y_test, axis=1)
        cur_acc = len(jnp.where((pred_y - y) == 0)[0]) * 1.0 / y.shape[0]
        acc.append(cur_acc)
    return acc


""" Random coreset selection """


def rand_from_batch(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Randomly select from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
    idx = np.random.choice(x_train.shape[0], coreset_size, False)
    x_coreset.append(x_train[idx, :])
    y_coreset.append(y_train[idx, :])
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)
    return x_coreset, y_coreset, x_train, y_train


""" K-center coreset selection """


def k_center(x_coreset, y_coreset, x_train, y_train, coreset_size):
    # Select K centers from (x_train, y_train) and add to current coreset (x_coreset, y_coreset)
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
    x_train = np.delete(x_train, idx, axis=0)
    y_train = np.delete(y_train, idx, axis=0)

    return x_coreset, y_coreset, x_train, y_train


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
    seq = np.random.permutation(np.arange(merged_x.shape[0]))
    merged_x = merged_x[seq, :]
    merged_y = merged_y[seq, :]
    return merged_x, merged_y