import numpy as np
import tree
from typing import Tuple, Callable, List
from functools import partial
import jax
import jax.numpy as jnp
from jax.example_libraries import optimizers
from jax import jit, random
import matplotlib.pyplot as plt
import haiku as hk
import custom_ntk
import os
import utils

eps = 1e-6


class loss_classification_list:
    def __init__(self,
                 apply_fn,
                 regularization,
                 dummy_input_dim,
                 class_num,
                 element_wise,
                 inverse,
                 ):
        self.apply_fn = apply_fn
        self.regularization = regularization
        self.dummy_input_dim = dummy_input_dim
        self.class_num = class_num
        self.inverse = inverse
        self.element_wise = element_wise
        self.state = {}

    @partial(jit, static_argnums=(0,))
    def llk_loss(self, params, params_copy, rng_key, x, y):
        y_hat = jax.nn.softmax(self.apply_fn({'params': params, **self.state}, x, mutable=list(self.state.keys()))[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        return -log_likelihood

    @partial(jit, static_argnums=(0,))
    def map_loss(self, params, params_copy, rng_key, x, y):
        y_hat = jax.nn.softmax(self.apply_fn({'params': params, **self.state}, x, mutable=list(self.state.keys()))[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        reg = self.regularization * jnp.square(optimizers.l2_norm(params))
        return -log_likelihood + reg

    @partial(jit, static_argnums=(0,))
    def jac_norm_loss(self, params, params_copy, rng_key, x, y):
        y_hat = jax.nn.softmax(self.apply_fn({'params': params, **self.state}, x, mutable=list(self.state.keys()))[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        rng_key, _ = jax.random.split(rng_key)
        input_ = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]  # Specify input for NTK or jacobian

        # convert function to compute gradient wrt x
        def convert_apply(apply_fn, x, params):
            def apply_fn_(x):
                return apply_fn({'params': params, **self.state}, x, mutable=list(self.state.keys()))[0]
            return apply_fn_

        apply_fn_jacobian = convert_apply(self.apply_fn, input_, params)
        J = jax.jacrev(apply_fn_jacobian)(input_)
        # J is of shape (batch_size, class_num, batch_size, 32, 32, 1)
        jac_norm = jnp.sqrt(jnp.sum(J ** 2, axis=(1, 3, 4, 5)) + eps)
        jac_norm = jnp.diag(jac_norm).mean() ** 2
        return -log_likelihood + self.regularization * jac_norm

    @partial(jit, static_argnums=(0,))
    def f_norm_loss(self, params, params_copy, rng_key, x, y):
        y_hat = jax.nn.softmax(self.apply_fn({'params': params, **self.state}, x, mutable=list(self.state.keys()))[0],
                               axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        # Specify input for NTK or jacobian
        rng_key, _ = jax.random.split(rng_key)
        input_ = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]
        f = self.apply_fn({'params': params, **self.state}, input_, mutable=list(self.state.keys()))[0]
        # f is of shape (batch_size, class_num)
        f_norm = jnp.sqrt((f ** 2).sum(1) + eps).mean()
        f_norm = f_norm ** 2
        return -log_likelihood + self.regularization * f_norm

    @partial(jit, static_argnums=(0,))
    def ntk_norm_loss(self, params, params_copy, rng_key, x, y):
        y_hat = jax.nn.softmax(self.apply_fn({'params': params, **self.state}, x, mutable=list(self.state.keys()))[0],
                               axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        # Compute function norm f(x)^T @ J(x)^T @ J(x) @ f(x)
        # Specify input for NTK
        rng_key, _ = jax.random.split(rng_key)
        ntk_input_all = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]

        def convert_to_ntk(apply_fn, x):
            def apply_fn_ntk(params):
                return apply_fn({'params': params, **self.state}, x, mutable=list(self.state.keys()))[0]
            return apply_fn_ntk

        if self.element_wise:
            freg = 0
            for j in range(self.dummy_input_dim):
                ntk_input = ntk_input_all[j, :][None]
                apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input)
                # Use params_copy here to kill the gradient wrt params in NTK
                ntk = custom_ntk.get_ntk(apply_fn_ntk, params_copy)

                y_ntk = self.apply_fn({'params': params, **self.state}, ntk_input, mutable=list(self.state.keys()))[0]
                for i in range(self.class_num):
                    ntk_ = ntk[:, i, :, i]
                    y_ntk_ = y_ntk[:, i][:, None]
                    if self.inverse:
                        freg += jnp.squeeze(y_ntk_ / ntk_ * y_ntk_)
                    else:
                        freg += jnp.squeeze(y_ntk_ * ntk_ * y_ntk_)
        else:
            ntk_input = ntk_input_all
            apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input)
            ntk = custom_ntk.get_ntk(apply_fn_ntk, params_copy)

            y_ntk = self.apply_fn({'params': params, **self.state}, ntk_input, mutable=list(self.state.keys()))[0]
            freg = 0
            for i in range(self.class_num):
                ntk_ = ntk[:, i, :, i]
                y_ntk_ = y_ntk[:, i][:, None]
                if self.inverse:
                    freg += jnp.squeeze(y_ntk_.T @ jnp.linalg.inv(ntk_ + eps * jnp.eye(self.dummy_input_dim)) @ y_ntk_)
                else:
                    freg += jnp.squeeze(y_ntk_.T @ ntk_ @ y_ntk_)
        freg = (jnp.sqrt(freg + eps) / ntk_input_all.shape[0]) ** 2
        return -log_likelihood + self.regularization * freg
