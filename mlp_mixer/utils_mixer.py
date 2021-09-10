from typing import Tuple, List, Dict
import numpy as np
import tree
from jax import jit
import jax.numpy as jnp
import argparse
import os

dtype_default = np.float32


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