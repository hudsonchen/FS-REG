import numpy as np
import tree
from typing import Tuple, Callable, List
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
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

    @partial(jit, static_argnums=(0, ))
    def llk_classification(self,
                           params: hk.Params,
                           params_copy: hk.Params,
                           state: hk.State,
                           rng_key: jnp.array,
                           x,
                           y,
                           ) -> jnp.ndarray:
        y_hat = jax.nn.softmax(self.apply_fn(params, state, rng_key, x)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        return -log_likelihood, state

    @partial(jit, static_argnums=(0,))
    def map_loss(self,
                 params: hk.Params,
                 params_copy: hk.Params,
                 state: hk.State,
                 rng_key: jnp.array,
                 x,
                 y,
                 ) -> jnp.ndarray:
        y_hat = jax.nn.softmax(self.apply_fn(params, state, rng_key, x)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        params_not_batchnorm = hk.data_structures.filter(utils.predicate_batchnorm, params)
        reg = self.regularization * jnp.square(optimizers.l2_norm(params_not_batchnorm))
        return -log_likelihood + reg, state

    @partial(jit, static_argnums=(0,))
    def jac_norm_loss(self,
                      params: hk.Params,
                      params_copy: hk.Params,
                      state: hk.State,
                      rng_key: jnp.array,
                      x,
                      y,
                      ) -> jnp.ndarray:
        y_hat = jax.nn.softmax(self.apply_fn(params, state, rng_key, x)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        input_ = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]  # Specify input for NTK or jacobian

        # convert function to compute gradient wrt x
        def convert_apply(apply_fn, inputs, params, state):
            def apply_fn_(inputs):
                return apply_fn(params, state, None, inputs)[0]
            return apply_fn_

        apply_fn_jacobian = convert_apply(self.apply_fn, input_, params, state)
        J = jax.jacrev(apply_fn_jacobian)(input_)
        # J is of shape (batch_size, class_num, batch_size, 32, 32, 1)
        jac_norm = jnp.sqrt(jnp.sum(J ** 2, axis=(1, 3, 4, 5)) + eps)
        jac_norm = jnp.diag(jac_norm).mean() ** 2
        return -log_likelihood + self.regularization * jac_norm, state

    @partial(jit, static_argnums=(0,))
    def f_norm_loss(self,
                    params: hk.Params,
                    params_copy: hk.Params,
                    state: hk.State,
                    rng_key: jnp.array,
                    x,
                    y,
                    ) -> jnp.ndarray:
        y_hat = jax.nn.softmax(self.apply_fn(params, state, rng_key, x)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        input_ = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]  # Specify input for NTK or jacobian
        f = self.apply_fn(params, state, rng_key, input_)[0]
        # f is of shape (batch_size, class_num)
        f_norm = jnp.sqrt((f ** 2).sum(1) + eps).mean()
        f_norm = f_norm ** 2
        return -log_likelihood + self.regularization * f_norm, state


    @partial(jit, static_argnums=(0, ))
    def laplacian_norm_loss(self,
                            params: hk.Params,
                            params_copy: hk.Params,
                            state: hk.State,
                            rng_key: jnp.array,
                            x,
                            y,
                            ) -> jnp.ndarray:
        y_hat = jax.nn.softmax(self.apply_fn(params, state, rng_key, x)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        input_ = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]  # Specify input for NTK or jacobian
        #
        # convert function to compute gradient wrt input
        def convert_apply(apply_fn, inputs, params, state):
            def apply_fn_(inputs):
                return apply_fn(params, state, None, inputs)[0]
            return apply_fn_

        apply_fn_jacobian = convert_apply(self.apply_fn, input_, params, state)
        J = jax.jacrev(apply_fn_jacobian)(input_)
        jac_norm = jnp.sqrt(jnp.sum(J ** 2, axis=(1, 3, 4, 5)) + eps)
        jac_norm = jnp.diag(jac_norm).mean() ** 2

        f = self.apply_fn(params, state, rng_key, input_)[0]
        f_norm = jnp.sqrt((f ** 2).sum(1) + eps).mean()
        f_norm = f_norm ** 2
        return -log_likelihood + self.regularization * (jac_norm + f_norm), state

    @partial(jit, static_argnums=(0, ))
    def ntk_norm_loss(self,
                      params: hk.Params,
                      params_copy: hk.Params,
                      state: hk.State,
                      rng_key: jnp.array,
                      x,
                      y,
                      ) -> jnp.ndarray:
        y_hat = jax.nn.softmax(self.apply_fn(params, state, rng_key, x)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        # Compute function norm f(x)^T @ J(x)^T @ J(x) @ f(x)
        ntk_input_all = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]  # Specify input for NTK

        def convert_to_ntk(apply_fn, inputs, state):
            def apply_fn_ntk(params):
                return apply_fn(params, state, None, inputs)[0]
            return apply_fn_ntk

        if self.element_wise:
            freg = 0
            for j in range(self.dummy_input_dim):
                ntk_input = ntk_input_all[j, :][None]
                apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input, state)
                # Use params_copy here to kill the gradient wrt params in NTK
                ntk = custom_ntk.get_ntk(apply_fn_ntk, params_copy)

                y_ntk = self.apply_fn(params, state, rng_key, ntk_input)[0]
                for i in range(self.class_num):
                    ntk_ = ntk[:, i, :, i]
                    y_ntk_ = y_ntk[:, i][:, None]
                    if self.inverse:
                        freg += jnp.squeeze(y_ntk_ / ntk_ * y_ntk_)
                    else:
                        freg += jnp.squeeze(y_ntk_ * ntk_ * y_ntk_)
        else:
            ntk_input = ntk_input_all
            apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input, state)
            ntk = custom_ntk.get_ntk(apply_fn_ntk, params_copy)

            y_ntk = self.apply_fn(params, state, rng_key, ntk_input)[0]
            freg = 0
            for i in range(self.class_num):
                ntk_ = ntk[:, i, :, i]
                y_ntk_ = y_ntk[:, i][:, None]
                if self.inverse:
                    freg += jnp.squeeze(y_ntk_.T @ jnp.linalg.inv(ntk_ + eps * jnp.eye(self.dummy_input_dim)) @ y_ntk_)
                else:
                    freg += jnp.squeeze(y_ntk_.T @ ntk_ @ y_ntk_)
        freg = (jnp.sqrt(freg + eps) / ntk_input_all.shape[0]) ** 2
        return -log_likelihood + self.regularization * freg, state


    # @partial(jit, static_argnums=(0, ))
    # def ntk_norm_loss(params: hk.Params,
    #                   state: hk.State,
    #                   rng_key: jnp.array,
    #                   x,
    #                   y,
    #                   ) -> jnp.ndarray:
    #     y_hat = jax.nn.softmax(apply_fn(params, state, rng_key, x)[0], axis=1)
    #     log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat) + eps) * y, axis=1), axis=0)
    #
    #     # Compute function norm f(x)^T @ J(x)^T @ J(x) @ f(x)
    #     ntk_input_all = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:dummy_input_dim], :]  # Specify input for NTK
    #
    #     def convert_to_ntk(apply_fn, inputs, params, state):
    #         def apply_fn_ntk(params):
    #             return apply_fn(params, state, None, inputs)[0]
    #
    #         return apply_fn_ntk
    #
    #     if element_wise:
    #         freg = 0
    #         for j in range(dummy_input_dim):
    #             ntk_input = ntk_input_all[j, :][None]
    #             apply_fn_ntk = convert_to_ntk(apply_fn, ntk_input, params, state)
    #             ntk = custom_ntk.get_ntk(apply_fn_ntk, params)
    #
    #             y_ntk = apply_fn(params, state, rng_key, ntk_input)[0]
    #             for i in range(class_num):
    #                 ntk_ = ntk[:, i, :, i]
    #                 y_ntk_ = y_ntk[:, i][:, None]
    #                 if inverse:
    #                     freg += jnp.squeeze(y_ntk_ / ntk_ * y_ntk_)
    #                 else:
    #                     freg += jnp.squeeze(y_ntk_ * ntk_ * y_ntk_)
    #     else:
    #         ntk_input = ntk_input_all
    #         apply_fn_ntk = convert_to_ntk(apply_fn, ntk_input, params, state)
    #         ntk = custom_ntk.get_ntk(apply_fn_ntk, params)
    #
    #         y_ntk = apply_fn(params, state, rng_key, ntk_input)[0]
    #         freg = 0
    #         for i in range(class_num):
    #             ntk_ = ntk[:, i, :, i]
    #             y_ntk_ = y_ntk[:, i][:, None]
    #             if inverse:
    #                 freg += jnp.squeeze(y_ntk_.T @ jnp.linalg.inv(ntk_ + 0.01 * jnp.eye(dummy_input_dim)) @ y_ntk_)
    #             else:
    #                 freg += jnp.squeeze(y_ntk_.T @ ntk_ @ y_ntk_)
    #     freg /= ntk_input.shape[0]
    #     return -log_likelihood + regularization * freg, state
