import jax
from jax import jit
import numpy as np
import jax.numpy as jnp
import haiku as hk
import argparse
import dataset
import utils
import os
from tqdm import tqdm
import custom_ntk
import optax
import matplotlib.pyplot as plt

relu = jax.nn.relu
eps = 1e-6


class MyLinear(hk.Module):
    def __init__(self, hidden_dim, c_sigma):
        super().__init__()
        self.output_size = 10
        self.hidden_dim = hidden_dim
        self.c_sigma = c_sigma

    def __call__(self, x):
        w_init = hk.initializers.RandomNormal()
        w1 = hk.get_parameter("w1", shape=[x.shape[-1], self.hidden_dim], dtype=x.dtype, init=w_init)
        # b1 = hk.get_parameter("b1", shape=[self.hidden_dim], dtype=x.dtype, init=jnp.zeros)
        w2 = hk.get_parameter("w2", shape=[self.hidden_dim, self.output_size], dtype=x.dtype, init=w_init)
        # b2 = hk.get_parameter("b2", shape=[1], dtype=x.dtype, init=jnp.zeros)

        x = jnp.dot(x, w1)
        x = relu(x) * jnp.sqrt(self.c_sigma / self.hidden_dim)
        x = jnp.dot(x, w2)
        return x


def get_empirical_ntk(inputs, params, state, rng_key):
    def convert_to_ntk(apply_fn, inputs, state):
        def apply_fn_ntk(params):
            return apply_fn(params, state, None, inputs)[0]

        return apply_fn_ntk

    apply_fn_ntk = convert_to_ntk(apply_fn, inputs, state)
    # Use params_copy here to kill the gradient wrt params in NTK
    ntk = custom_ntk.get_ntk(apply_fn_ntk, params)
    return ntk


def get_analytic_ntk(inputs, rng_key):
    normal_samples = jax.random.normal(rng_key, [100])
    c_sigma = 1. / (relu(normal_samples) ** 2).mean()

    def sigma_0(x1, x2):
        return x1.transpose() @ x2

    gamma_1 = sigma_to_gamma(sigma_0, rng_key)
    sigma_1 = gamma_to_sigma(gamma_1, c_sigma, rng_key)
    dif_sigma_1 = gamma_to_dif_sigma(gamma_1, c_sigma, rng_key)

    gamma_2 = sigma_to_gamma(sigma_1, rng_key)
    sigma_2 = gamma_to_sigma(gamma_2, c_sigma, rng_key)
    dif_sigma_2 = gamma_to_dif_sigma(gamma_2, c_sigma, rng_key)

    N = inputs.shape[0]
    K = jnp.zeros([N, N])
    for i in tqdm(range(N)):
        for j in range(N):
            x1 = inputs[i, :]
            x2 = inputs[j, :]
            K_ij_part_one = sigma_0(x1, x2) * dif_sigma_1(x1, x2) * dif_sigma_2(x1, x2)
            K_ij_part_two = sigma_1(x1, x2) * dif_sigma_2(x1, x2)
            K = K.at[i, j].set(K_ij_part_one + K_ij_part_two)
    return K


@jit
def dif_relu(x):
    return jnp.asarray((x > 0), dtype=jnp.float32)


def sigma_to_gamma(sigma, rng_key):
    @jit
    def gamma(x1, x2):
        return jnp.array([[sigma(x1, x1), sigma(x1, x2)],
                          [sigma(x2, x1), sigma(x2, x2)]])

    return gamma


def gamma_to_sigma(gamma, c_sigma, rng_key):
    rng_key, _ = jax.random.split(rng_key)

    @jit
    def sigma(x1, x2):
        mean = jnp.zeros([100, 2])
        cov = gamma(x1, x2) + jnp.eye(2)
        samples = jax.random.multivariate_normal(rng_key, mean, cov)
        return c_sigma * (relu(samples[:, 0]) * relu(samples[:, 1])).mean()

    return sigma


def gamma_to_dif_sigma(gamma, c_sigma, rng_key):
    rng_key, _ = jax.random.split(rng_key)

    @jit
    def dif_sigma(x1, x2):
        mean = jnp.zeros([100, 2])
        cov = gamma(x1, x2) + jnp.eye(2)
        samples = jax.random.multivariate_normal(rng_key, mean, cov)
        return c_sigma * (dif_relu(samples[:, 0]) * dif_relu(samples[:, 1])).mean()

    return dif_sigma


def train_mylinear(args, train_loader, test_loader, params, state, opt_state, rng_key):
    fkf_list = []
    for epoch in range(args.epochs):
        for batch_idx, (image, label) in enumerate(train_loader):
            rng_key, _ = jax.random.split(rng_key)
            image, label = utils.tensor2array(image, label, num_classes)
            image = image.reshape([-1, 32 * 32])

            params, state, opt_state = update(params, state, opt_state, rng_key, image, label)

            if batch_idx % 50 == 0:
                ntk = get_empirical_ntk(image, params, state, rng_key)[:, 0, :, 0]
                ntk_inv = np.linalg.inv(ntk)
                f = np.ones([1, args.batch_size])
                fkf = (f @ ntk_inv @ f.transpose())[0, 0]
                fkf_list.append(fkf)

                acc_ = 0
                for batch_idx, (image, label) in enumerate(test_loader):
                    rng_key, _ = jax.random.split(rng_key)
                    image, label = utils.tensor2array(image, label, num_classes)
                    image = image.reshape([-1, 32 * 32])

                    preds = apply_fn(params, state, rng_key, image)[0]
                    acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
                    acc_ += acc
                acc_ /= len(test_loader) * 1000 / 100.
                print(f"epoch is {epoch} batch idx is {batch_idx} and accuracy is {acc_}")
    return fkf_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden_dim', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=200)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    rng_key = jax.random.PRNGKey(args.seed)
    rng_key_1, rng_key_2 = jax.random.split(rng_key)

    image_size, num_classes, train_loader, test_loader = dataset.get_MNIST(batch_size=args.batch_size,
                                                                           train_size=50000)
    (image, label) = next(iter(train_loader))
    image, label = utils.tensor2array(image, label, num_classes)
    image = image.reshape([-1, 32 * 32])

    normal_samples = jax.random.normal(rng_key, [100])
    c_sigma = 1. / (relu(normal_samples) ** 2).mean()

    init_fn, apply_fn = hk.transform_with_state(lambda x: MyLinear(hidden_dim=args.hidden_dim, c_sigma=c_sigma)(x))
    x_init = jnp.ones([1, 32 * 32])
    params, state = init_fn(rng_key, x_init)

    # empirical_ntk = get_empirical_ntk(image, params, state, rng_key_2)[:, 0, :, 0]
    # analytic_ntk = get_analytic_ntk(image, rng_key_1)
    # jnp.save(f"/import/home/xzhoubi/hudson/function_map/rebuttal_icml/analytic_ntk_{args.batch_size}_{args.hidden_dim}.npy", analytic_ntk)
    # jnp.save(f"/import/home/xzhoubi/hudson/function_map/rebuttal_icml/empirical_ntk_{args.batch_size}_{args.hidden_dim}.npy", empirical_ntk)

    opt = optax.adam(args.lr)
    opt_state = opt.init(params)


    @jit
    def loss_fun(params: hk.Params,
                 state: hk.State,
                 rng_key: jnp.array,
                 x,
                 y,
                 ):
        y_hat = jax.nn.softmax(apply_fn(params, state, rng_key, x)[0], axis=1)
        log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
        return -log_likelihood, state


    @jit
    def update(params: hk.Params,
               state: hk.State,
               opt_state: optax.OptState,
               rng_key: jnp.array,
               x,
               y,
               ):
        """Learning rule (stochastic gradient descent)."""
        grads, new_state = jax.grad(loss_fun, argnums=0, has_aux=True)(params, state, rng_key, x, y)
        updates, opt_state = opt.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_state, opt_state


    fkf_list = train_mylinear(args, train_loader, test_loader, params, state, opt_state, rng_key)

    plt.figure()
    plt.plot(fkf_list)
    plt.show()
