import numpy as np
import tree
from typing import Tuple, Callable, List
from functools import partial
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax import jit, random
from jax.tree_util import tree_flatten, tree_unflatten, tree_map
import matplotlib.pyplot as plt
import haiku as hk
import custom_ntk
import os
import utils

eps = 1e-6


class loss_cl_list:
    def __init__(self,
                 apply_fn,
                 regularization,
                 dummy_input_dim,
                 head_style,
                 class_num,
                 element_wise,
                 inverse,
                 ):
        self.apply_fn = apply_fn
        self.regularization = regularization
        self.dummy_input_dim = dummy_input_dim
        if head_style == 'single':
            self.class_num = class_num
        elif head_style == 'multi':
            self.class_num = 2
        self.inverse = inverse
        self.element_wise = element_wise

    @partial(jit, static_argnums=(0,))
    def llk_classification(self,
                           params: hk.Params,
                           params_last: hk.Params,
                           state: hk.State,
                           rng_key: jnp.array,
                           x,
                           y,
                           task_id,
                           ):
        task_id = jnp.ones(x.shape[0]) * task_id
        y_hat = jax.nn.softmax(self.apply_fn(params, state, rng_key, x, task_id)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        return -log_likelihood, state

    @partial(jit, static_argnums=(0,))
    def weight_l2_norm_loss_without_fisher(self,
                            params: hk.Params,
                            params_last: hk.Params,
                            params_list,
                            state: hk.State,
                            rng_key: jnp.array,
                            x,
                            y,
                            task_id,
                            ind_points,
                            ind_id,
                            fisher
                            ):
        task_id = jnp.ones(x.shape[0]) * task_id
        f_hat = self.apply_fn(params, state, rng_key, x, task_id)[0]
        y_hat = jax.nn.softmax(f_hat, axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        params_not_batchnorm = hk.data_structures.filter(utils.predicate_batchnorm, params)
        params_not_batchnorm_last = hk.data_structures.filter(utils.predicate_batchnorm, params_last)
        params_difference = tree_map(lambda p, q: p - q + eps, params_not_batchnorm, params_not_batchnorm_last)
        reg = self.regularization * jnp.square(optimizers.l2_norm(params_difference))
        return -log_likelihood + reg, state

    @partial(jit, static_argnums=(0,))
    def weight_l2_norm_loss_with_fisher(self,
                            params: hk.Params,
                            params_last: hk.Params,
                            params_list,
                            state: hk.State,
                            rng_key: jnp.array,
                            x,
                            y,
                            task_id,
                            ind_points,
                            ind_id,
                            fisher
                            ):
        task_id = jnp.ones(x.shape[0]) * task_id
        f_hat = self.apply_fn(params, state, rng_key, x, task_id)[0]
        y_hat = jax.nn.softmax(f_hat, axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        params_not_batchnorm = hk.data_structures.filter(utils.predicate_batchnorm, params)
        params_not_batchnorm_last = hk.data_structures.filter(utils.predicate_batchnorm, params_last)
        params_difference = tree_map(lambda p, q: p - q + eps, params_not_batchnorm, params_not_batchnorm_last)
        sqrt_fisher = tree_map(lambda p: jnp.sqrt(p), fisher)
        params_reg = tree_map(lambda p, q: p * q, params_difference, sqrt_fisher)
        reg = self.regularization * jnp.square(optimizers.l2_norm(params_reg))
        return -log_likelihood + reg, state

    @partial(jit, static_argnums=(0,))
    def f_l2_norm_loss(self,
                       params: hk.Params,
                       params_last: hk.Params,
                       params_list,
                       state: hk.State,
                       rng_key: jnp.array,
                       x,
                       y,
                       task_id,
                       ind_points,
                       ind_id,
                       fisher
                       ):
        task_id = jnp.ones(x.shape[0]) * task_id
        f_hat = self.apply_fn(params, state, rng_key, x, task_id)[0]
        y_hat = jax.nn.softmax(f_hat, axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        f_ind_hat_old = self.apply_fn(params_last, state, rng_key, ind_points)[0]
        f_ind_hat_new = self.apply_fn(params, state, rng_key, ind_points)[0]
        # f is of shape (batch_size, class_num)
        f_norm = jnp.sqrt(((f_ind_hat_new - f_ind_hat_old) ** 2).sum(1) + eps).mean() ** 2
        return -log_likelihood + self.regularization * f_norm, state

    @partial(jit, static_argnums=(0,))
    def ntk_norm_loss(self,
                      params: hk.Params,
                      params_last: hk.Params,
                      params_list,
                      state: hk.State,
                      rng_key: jnp.array,
                      x,
                      y,
                      task_id,
                      ind_points,
                      ind_id,
                      fisher
                      ):
        task_id = jnp.ones(x.shape[0]) * task_id
        f_hat = self.apply_fn(params, state, rng_key, x, task_id)[0]
        y_hat = jax.nn.softmax(f_hat, axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        # Compute function norm f(x)^T @ J(x)^T @ J(x) @ f(x)
        if ind_points is None:
            ntk_input_all = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]
        else:
            ntk_input_all = ind_points

        def convert_to_ntk(apply_fn, inputs, state, task_id):
            def apply_fn_ntk(params):
                return apply_fn(params, state, None, inputs, task_id)[0]
            return apply_fn_ntk

        if self.element_wise:
            freg = 0
            for j in range(self.dummy_input_dim):
                ntk_input = ntk_input_all[j, :][None]
                apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input, state, ind_id)
                # Use params_copy here to kill the gradient wrt params in NTK
                ntk = custom_ntk.get_ntk(apply_fn_ntk, params_last)

                y_ntk = self.apply_fn(params, state, rng_key, ntk_input, ind_id)[0] - \
                        self.apply_fn(params_last, state, rng_key, ntk_input, ind_id)[0]
                for i in range(self.class_num):
                    ntk_ = ntk[:, i, :, i]
                    y_ntk_ = y_ntk[:, i][:, None]
                    if self.inverse:
                        freg += jnp.squeeze(y_ntk_ / (ntk_ + eps) * y_ntk_).sum()
                    else:
                        freg += jnp.squeeze(y_ntk_ * ntk_ * y_ntk_).sum()
        else:
            # Note that in continual learning, inverse and full matrix will cause numerical instability
            ntk_input = ntk_input_all
            apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input, state)
            ntk = custom_ntk.get_ntk(apply_fn_ntk, params_last)

            y_ntk = self.apply_fn(params, state, rng_key, ntk_input, task_id - 1)[0] - \
                    self.apply_fn(params_last, state, rng_key, ntk_input, task_id - 1)[0]
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

    @partial(jit, static_argnums=(0,))
    def ntk_norm_loss_all_prev(self,
                               params: hk.Params,
                               params_last: hk.Params,
                               params_list,
                               state: hk.State,
                               rng_key: jnp.array,
                               x,
                               y,
                               task_id,
                               ind_points,
                               ind_id,
                               fisher
                               ):
        task_id = jnp.ones(x.shape[0]) * task_id
        f_hat = self.apply_fn(params, state, rng_key, x, task_id)[0]
        y_hat = jax.nn.softmax(f_hat, axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)

        # Compute function norm f(x)^T @ J(x)^T @ J(x) @ f(x)
        if ind_points is None:
            ntk_input_all = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]
        else:
            ntk_input_all = ind_points

        def convert_to_ntk(apply_fn, inputs, state):
            def apply_fn_ntk(params):
                return apply_fn(params, state, None, inputs, task_id)[0]
            return apply_fn_ntk

        for params_last in params_list:
            freg_ = 0
            if self.element_wise:
                freg = 0
                for j in range(self.dummy_input_dim):
                    ntk_input = ntk_input_all[j, :][None]
                    apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input, state)
                    # Use params_copy here to kill the gradient wrt params in NTK
                    ntk = custom_ntk.get_ntk(apply_fn_ntk, params_last)

                    y_ntk = self.apply_fn(params, state, rng_key, ntk_input, ind_id)[0] - \
                            self.apply_fn(params_last, state, rng_key, ntk_input, ind_id)[0]
                    for i in range(self.class_num):
                        ntk_ = ntk[:, i, :, i]
                        y_ntk_ = y_ntk[:, i][:, None]
                        if self.inverse:
                            freg += jnp.squeeze(y_ntk_ / (ntk_ + eps) * y_ntk_).sum()
                        else:
                            freg += jnp.squeeze(y_ntk_ * ntk_ * y_ntk_).sum()
            else:
                # Note that in continual learning, inverse full matrix will cause numerical instability
                ntk_input = ntk_input_all
                apply_fn_ntk = convert_to_ntk(self.apply_fn, ntk_input, state)
                ntk = custom_ntk.get_ntk(apply_fn_ntk, params_last)

                y_ntk = self.apply_fn(params, state, rng_key, ntk_input, ind_id)[0] - \
                        self.apply_fn(params_last, state, rng_key, ntk_input, ind_id)[0]
                freg = 0
                for i in range(self.class_num):
                    ntk_ = ntk[:, i, :, i]
                    y_ntk_ = y_ntk[:, i][:, None]
                    if self.inverse:
                        freg += jnp.squeeze(y_ntk_.T @ jnp.linalg.inv(ntk_ + eps * jnp.eye(self.dummy_input_dim)) @ y_ntk_)
                    else:
                        freg += jnp.squeeze(y_ntk_.T @ ntk_ @ y_ntk_)
            freg = (jnp.sqrt(freg + eps) / ntk_input_all.shape[0]) ** 2
            freg_ += freg
        return -log_likelihood + self.regularization * freg_, state

    # @partial(jit, static_argnums=(0,))
    def ntk_norm_loss_temp(self,
                           params: hk.Params,
                           params_copy: hk.Params,
                           state: hk.State,
                           rng_key: jnp.array,
                           x,
                           y,
                           ind_points=None
                           ):
        # Compute function norm f(x)^T @ J(x)^T @ J(x) @ f(x)
        if ind_points is None:
            ntk_input_all = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:self.dummy_input_dim], :]
        else:
            ntk_input_all = ind_points

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
                        freg += jnp.squeeze(y_ntk_ / ntk_ * y_ntk_).sum()
                    else:
                        freg += jnp.squeeze(y_ntk_ * ntk_ * y_ntk_).sum()
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
        return freg, ntk, jnp.linalg.inv(ntk_ + eps * jnp.eye(self.dummy_input_dim))
