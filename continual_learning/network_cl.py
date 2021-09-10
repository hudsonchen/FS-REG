from functools import partial
from typing import Tuple, Callable, List
import jax
import jax.numpy as jnp
from jax import jit
import haiku as hk

ACTIVATION_DICT = {"tanh": jnp.tanh, "relu": jax.nn.relu}


class MLP:
    def __init__(
            self,
            output_dim: int,
            architecture: List[int],
            head_style: str,
            activation_fn: str = "relu",
            seed: int = 1,
    ):
        self.output_dim = output_dim
        self.activation_fn = ACTIVATION_DICT[activation_fn]
        self.architecture = architecture
        self.rng_key = jax.random.PRNGKey(seed)
        self.head_style = head_style

    def forward_fn(self, inputs: jnp.ndarray, task_id):
        out = inputs
        for unit in self.architecture:
            fully_connected = hk.Linear(unit)
            out = self.activation_fn(fully_connected(out))

        fully_connected = hk.Linear(self.output_dim)
        if self.head_style == 'single':
            out = fully_connected(out)
        elif self.head_style == 'multi':
            out = fully_connected(out)[:, task_id]
        return out
