import numpy as np
import tree
from typing import Tuple, Callable, List
import jax
import jax.numpy as jnp
from jax.experimental import optimizers
from jax import jit, random
import matplotlib.pyplot as plt
import haiku as hk
import os
import tensorflow_probability.substrates.jax.distributions as tfd
import optax
import argparse
from tqdm import tqdm
import dataset
import network
import custom_ntk


os.environ["CUDA_VISIBLE_DEVICES"] = "2"
abspath = os.path.abspath(__file__)
path = os.path.dirname(abspath)
os.chdir(path)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='map')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--element_wise', action="store_true", default=True)
args = parser.parse_args()

# Useful Global Variables
rng_key = jax.random.PRNGKey(0)
class_num = 1

# Load Snelson Dataset
x_train, y_train, x_test, noise_std = dataset.load_snelson()

# Model Initialization
model = network.MLP(output_dim=1,
                    architecture=[100, 100])

init_fn, apply_fn = hk.transform_with_state(model.forward_fn)
x_init = x_train[0, :]
params_init, state = init_fn(rng_key, x_init)
params = params_init


def convert_f_for_ntk(params, x, state, rng_key):
    return apply_fn(params, state, rng_key, x)[0]


apply_fn_for_ntk = convert_f_for_ntk

# Optimizer Initialization
opt = optax.adam(1e-2)
opt_state = opt.init(params_init)

# Hyperparameter
regularization = args.reg
element_wise = args.element_wise
ntk_input_dim = 10
inverse = args.inverse


@jit
def map_loss(params: hk.Params,
             state: hk.State,
             rng_key: jnp.array,
             x,
             y,
             noise_std) -> jnp.ndarray:
    y_hat = apply_fn(params, state, rng_key, x)[0]
    likelihood = tfd.Normal(y_hat, noise_std)
    log_likelihood = jnp.mean(jnp.mean(likelihood.log_prob(y), 0), 0)
    reg_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params) if 'batchnorm' not in mod_name]
    reg = regularization * jnp.square(optimizers.l2_norm(reg_params))
    return -log_likelihood + reg, state


@jit
def fmap_loss(params: hk.Params,
              state: hk.State,
              rng_key: jnp.array,
              x,
              y,
              noise_std) -> jnp.ndarray:
    y_hat = apply_fn(params, state, rng_key, x)[0]
    likelihood = tfd.Normal(y_hat, noise_std)
    log_likelihood = jnp.mean(jnp.mean(likelihood.log_prob(y), 0), 0)

    # Compute function norm f(x)^T @ J(x)^T @ J(x) @ f(x)
    ntk_input_all = x[random.shuffle(rng_key, np.arange(x.shape[0]))[:ntk_input_dim], :]  # Specify input for NTK

    def convert_to_ntk(apply_fn, inputs, params, state):
        def apply_fn_ntk(params):
            return apply_fn(params, state, None, inputs)[0]

        return apply_fn_ntk

    if element_wise:
        freg = 0
        for j in range(ntk_input_dim):
            ntk_input = ntk_input_all[j, :][None]
            apply_fn_ntk = convert_to_ntk(apply_fn, ntk_input, params, state)
            ntk = custom_ntk.get_ntk(apply_fn_ntk, params)

            y_ntk = apply_fn(params, state, rng_key, ntk_input)[0]
            for i in range(class_num):
                ntk_ = ntk[:, i, :, i]
                y_ntk_ = y_ntk[:, i][:, None]
                if inverse:
                    freg += jnp.squeeze(y_ntk_ / ntk_ * y_ntk_)
                else:
                    freg += jnp.squeeze(y_ntk_ * ntk_ * y_ntk_)
    else:
        ntk_input = ntk_input_all
        apply_fn_ntk = convert_to_ntk(apply_fn, ntk_input, params, state)
        ntk = custom_ntk.get_ntk(apply_fn_ntk, params)

        y_ntk = apply_fn(params, state, rng_key, ntk_input)[0]
        freg = 0
        for i in range(class_num):
            ntk_ = ntk[:, i, :, i]
            y_ntk_ = y_ntk[:, i][:, None]
            if inverse:
                freg += jnp.squeeze(y_ntk_.T @ jnp.linalg.inv(ntk_ + 0.01 * jnp.eye(ntk_input_dim)) @ y_ntk_)
            else:
                freg += jnp.squeeze(y_ntk_.T @ ntk_ @ y_ntk_)
    freg /= ntk_input.shape[0]

    return -log_likelihood + regularization * freg, state


@jit
def update(params: hk.Params,
           state: hk.State,
           opt_state: optax.OptState,
           rng_key: jnp.array,
           x,
           y,
           noise_std
           ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    grads, new_state = jax.grad(loss_fun, has_aux=True)(params, state, rng_key, x, y, noise_std)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state


if args.method == 'map':
    loss_fun = map_loss
elif args.method == 'fmap':
    loss_fun = fmap_loss

loss_list = []
print(f"Start Training with {args.method}\n")
for step in tqdm(range(50000)):
    rng_key, _ = random.split(rng_key)
    params, state, opt_state = update(params, state, opt_state, rng_key, x_train, y_train, noise_std)
    loss_value = loss_fun(params, state, rng_key, x_train, y_train, noise_std)[0]
    loss_list.append(loss_value)
    if (step + 1) % 5000 == 0:
        y_test_hat = apply_fn(params, state, rng_key, x_test)[0]

        plt.figure()
        plt.scatter(x_train, y_train, s=10, color="r")
        plt.plot(x_test, y_test_hat)
        plt.xlim([-10, 10])
        plt.ylim([-3.0, 3.0])
        plt.show()

plt.figure()
plt.plot(np.array(loss_list)[1000:])
plt.title("Loss Value")
plt.show()
