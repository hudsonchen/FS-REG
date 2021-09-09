from typing import Tuple, List, Dict
import numpy as np
import tree
from jax import jit
import jax.numpy as jnp
import argparse
import os

dtype_default = np.float32

@jit
def sigma_transform(params_log_var):
    return tree.map_structure(lambda p: jnp.exp(p), params_log_var)


def one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def predicate_batchnorm(module_name, name, value):
    return name in {"w", "b"}


def piecewise_constant_schedule(init_value, boundaries, scale):
    """
    Return a function that takes in the update count and returns a step size.
    The step size is equal to init_value * (scale ** <number of boundaries points not greater than count>)
    """
    def schedule(count):
        v = init_value
        for threshold in boundaries:
            indicator = jnp.maximum(0.0, jnp.sign(threshold - count))
            v = v * indicator + (1 - indicator) * scale * v
        return v

    return schedule


def process_args(args: argparse.Namespace) -> Dict:
    kwargs = vars(args)
    save_path = args.save_path.rstrip()
    save_path = (
        f"{save_path}/{args.dataset}/dataset_{args.dataset}__method_{args.method}__train_size_{args.train_size}"
        f"__DA_{args.aug}__reg_{args.reg}__seed_{args.seed}__"
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


def tensor2array(image, label, num_classes):
    image = np.moveaxis(np.array(image, dtype=dtype_default), 1, 3)
    label = one_hot(np.array(label), num_classes)
    return image, label


def split(arr, n_devices):
    """Splits the first axis of `arr` evenly across the number of devices."""
    return arr.reshape(n_devices, arr.shape[0] // n_devices, *arr.shape[1:])

