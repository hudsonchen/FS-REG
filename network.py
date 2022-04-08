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
            activation_fn: str = "relu",
            seed: int = 1,
    ):
        self.output_dim = output_dim
        self.activation_fn = ACTIVATION_DICT[activation_fn]
        self.architecture = architecture
        self.rng_key = jax.random.PRNGKey(seed)

    def __call__(self, inputs: jnp.ndarray):
        out = inputs
        for unit in self.architecture:
            fully_connected = hk.Linear(unit)
            out = self.activation_fn(fully_connected(out))

        fully_connected = hk.Linear(self.output_dim)
        out = fully_connected(out)
        return out


class LeNet:
    def __init__(
            self,
            output_dim: int,
            use_dropout: bool = False,
            dropout_rate: float = 0.0,
            activation_fn: str = "relu",
            seed: int = 1,
    ):
        self.output_dim = output_dim
        self.activation_fn = ACTIVATION_DICT[activation_fn]
        self.rng_key = jax.random.PRNGKey(seed)
        self.use_dropout = use_dropout
        self.dropout_rate = dropout_rate

        self.conv1 = hk.Conv2D(
            output_channels=6,
            kernel_shape=5,
            padding="VALID",
        )
        self.conv2 = hk.Conv2D(
            output_channels=16,
            kernel_shape=5,
            padding="VALID",
        )
        self.conv3 = hk.Conv2D(
            output_channels=120,
            kernel_shape=5,
            padding="VALID",
        )
        self.fc1 = hk.Linear(output_size=84)
        self.fc2 = hk.Linear(output_size=self.output_dim)
        self.avg_pool = hk.AvgPool(
            window_shape=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding="VALID"
        )

    def __call__(self, inputs: jnp.ndarray):
        out = inputs
        out = self.activation_fn(self.conv1(out))
        out = self.avg_pool(out)
        out = self.activation_fn(self.conv2(out))
        out = self.avg_pool(out)
        out = self.activation_fn(self.conv3(out))
        out = out.reshape([inputs.shape[0], -1])
        if self.use_dropout:
            out = hk.dropout(self.rng_key, self.dropout_rate, out)
        out = self.activation_fn(self.fc1(out))
        if self.use_dropout:
            out = hk.dropout(self.rng_key, self.dropout_rate, out)
        out = self.fc2(out)
        return out
