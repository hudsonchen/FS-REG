import jax
from functools import partial
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node
from jax import vjp, jvp, eval_shape, numpy as jnp, jacobian, partial

def delta_vjp(predict_fn, params, delta):
    vjp_tp = jax.vjp(predict_fn, params)[1](delta)
    return vjp_tp

# @partial(jit, static_argnums=(0,1,))
def delta_vjp_jvp(predict_fn, delta_vjp, params, delta):
    delta_vjp_ = partial(delta_vjp, predict_fn, params)
    return jax.jvp(predict_fn, (params,), delta_vjp_(delta))[1]

# @partial(jit, static_argnums=(0,1,2))
def get_ntk(predict_fn, params):
    predict_struct = eval_shape(predict_fn, params)
    fx_dummy = jnp.ones(predict_struct.shape, predict_struct.dtype)
    delta_vjp_jvp_ = partial(delta_vjp_jvp, predict_fn, delta_vjp, params)
    gram_matrix = jacobian(delta_vjp_jvp_)(fx_dummy)
    return gram_matrix