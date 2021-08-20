import numpy as np
import jax
import jax.numpy as jnp


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

