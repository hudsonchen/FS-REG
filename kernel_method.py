from typing import Tuple, Callable, List
from jax import jit
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
import haiku as hk
import os
import optax
import argparse
import scipy
from tqdm import tqdm
import loss_classification
import dataset
import network
import utils
import utils_logging
import resnet_mod
import custom_ntk
# from jax import config
# config.update('jax_disable_jit', True)
from jax.tree_util import tree_flatten

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--method', type=str, default='map')
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--architecture', type=str, default='none')
parser.add_argument('--optimizer', type=str, default='none')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--lr_decay', type=float, default='0.5')
parser.add_argument('--dummy_input_dim', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--use_bn', action="store_true", default=False)
parser.add_argument('--element_wise', action="store_true", default=False)
parser.add_argument('--aug', action="store_true", default=False)
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--log_freq', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--save_path', type=str, default="/home/function_map/results")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_size', type=int, default=100)
parser.add_argument('--sigma', type=float, default=0.1)
args = parser.parse_args()
kwargs = utils.process_args(args)


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
abspath = os.path.abspath(__file__)
path = os.path.dirname(abspath)
os.chdir(path)

# Useful Global Variables
rng_key = jax.random.PRNGKey(args.seed)
eps = 1e-6
class_num = 10

# Load Dataset
if args.dataset == "mnist":
    image_size, num_classes, train_loader, test_loader = dataset.get_MNIST(
        batch_size=args.batch_size,
        train_size=args.train_size)
elif args.dataset == "cifar10":
    image_size, num_classes, train_loader, test_loader = dataset.get_CIFAR10(
        batch_size=args.batch_size,
        data_augmentation=args.aug,
        train_size=args.train_size)
else:
    raise NotImplementedError

# Model Initialization
if args.architecture == "lenet":
    init_fn, apply_fn = hk.transform_with_state(lambda x: network.LeNet(output_dim=10)(x))
    apply_fn_train = apply_fn
    apply_fn_eval = apply_fn
elif args.architecture == "resnet18":
    if args.train_size < 3000 and not args.aug:
        def forward(x, is_training):
            net = resnet_mod.ResNet18(10, use_bn=True, resnet_v1=True)
            return net(x, is_training)
    else:
        def forward(x, is_training):
            net = resnet_mod.ResNet18(10, use_bn=True)
            return net(x, is_training)

    forward = hk.transform_with_state(forward)
    init_fn = partial(forward.init, is_training=True)
    apply_fn_train = partial(forward.apply, is_training=True)
    apply_fn_eval = partial(forward.apply, is_training=False)
else:
    raise NotImplementedError

x_init = jnp.ones(image_size)
params_init, state = init_fn(rng_key, x_init)
params = params_init

X_train = []
y_train = []
X_test = []
y_test = []
for batch_idx, (image, label) in enumerate(train_loader):
    image, label = utils.tensor2array(image, label, num_classes)
    X_train.append(image)
    y_train.append(label)

for batch_idx, (image, label) in enumerate(test_loader):
    image, label = utils.tensor2array(image, label, num_classes)
    X_test.append(image)
    y_test.append(label)


def convert_to_ntk(apply_fn, inputs, state):
    def apply_fn_ntk(params):
        return apply_fn(params, state, None, inputs)[0]
    return apply_fn_ntk

if args.dataset == 'mnist':
    X_train = jnp.array(X_train).reshape(-1, 32, 32, 1)
    y_train = jnp.array(y_train).reshape(-1, 10) - 0.1
    X_test = jnp.array(X_test).reshape(-1, 32, 32, 1)
    y_test = jnp.array(y_test).reshape(-1, 10) - 0.1
elif args.dataset == 'cifar10':
    X_train = jnp.array(X_train).reshape(-1, 32, 32, 3)
    y_train = jnp.array(y_train).reshape(-1, 10) - 0.1
    X_test = jnp.array(X_test).reshape(-1, 32, 32, 3)
    y_test = jnp.array(y_test).reshape(-1, 10) - 0.1
else:
    raise NotImplementedError(args.dataset)

if args.dataset == 'mnist':
    ntk_dim = 50
elif args.dataset == 'cifar10':
    ntk_dim = 2
else:
    raise NotImplementedError(args.dataset)
train_num = args.train_size
test_num = 100
iter_train = int(train_num / ntk_dim)
iter_test = int(test_num / ntk_dim)
y_train_nystrom = []
y_test_nystrom = []
J_list_train = []
J_list_test = []

X_train = X_train[:train_num, :]
y_train = y_train[:train_num, :]
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print(f"Train Samples Size:{args.train_size}")
print(f"Compute Train NTK:")
K_train = jnp.zeros([train_num, train_num])


@jit
def J_J_T(J_1, J_2, ntk):
    shape_1, shape2 = ntk.shape
    for a in J_1:
        for b in J_1[a]:
            J_1_flatten = J_1[a][b].reshape(shape_1, -1)
            J_2_flatten = J_2[a][b].reshape(shape_1, -1)
            ntk += J_1_flatten @ J_2_flatten.T
    return ntk


if os.path.isfile(f'/home/function_map/kernel_saved_{args.dataset}/K_train_{train_num}.npy'):
    K_train = jnp.load(f'/home/function_map/kernel_saved_{args.dataset}/K_train_{train_num}.npy')
else:
    for i in tqdm(range(iter_train)):
        rng_key, _ = jax.random.split(rng_key)
        X_ntk_1 = X_train[i * ntk_dim:(i+1) * ntk_dim, :]
        def forward(params):
            return apply_fn_train(params, state, rng_key, X_ntk_1)[0]

        J_1 = jax.jacrev(forward)(params)

        for j in range(iter_train):
            rng_key, _ = jax.random.split(rng_key)
            X_ntk_2 = X_train[j * ntk_dim:(j+1) * ntk_dim, :]
            def forward(params):
                return apply_fn_train(params, state, rng_key, X_ntk_2)[0]
            J_2 = jax.jacrev(forward)(params)

            ntk_batch = J_J_T(J_1, J_2, jnp.zeros([ntk_dim, ntk_dim]))
            K_train = jax.ops.index_update(K_train, jax.ops.index[i * ntk_dim:(i+1) * ntk_dim, j * ntk_dim:(j+1) * ntk_dim], ntk_batch)
    np.save(f'/home/function_map/kernel_saved_{args.dataset}/K_train_{train_num}.npy', K_train)
print("Compute Train NTK Finished!\n\n")

# Scale down K_train for numerical reasons
K_mean = K_train.mean()
K_train = K_train / K_mean

if args.train_size > 500:
    print("Nystrom Approximation")
    nystrom_size = jnp.maximum(500, int(jnp.sqrt(train_num)))

    rng_key, _ = jax.random.split(rng_key)
    nystrom_idx = jax.random.permutation(rng_key, train_num)[:nystrom_size]

    K_nystrom = K_train[nystrom_idx, :][:, nystrom_idx]
    U, s, V = scipy.linalg.svd(K_nystrom)

    U_recon = jnp.sqrt(nystrom_size / train_num) * K_train[:, nystrom_idx] @ U @ jnp.diag(1. / s)
    S_recon = s * (train_num / nystrom_size)
    print("Nystrom Approximation Finished!\n\n")

    print("Compute K_train Inversion")
    sigma = args.sigma
    Sigma_inv = (1. / sigma) * jnp.eye(train_num)
    K_train_inv = Sigma_inv - Sigma_inv @ U_recon @ scipy.linalg.inv(
        jnp.diag(1. / S_recon) + U_recon.T @ Sigma_inv @ U_recon) @ U_recon.T @ Sigma_inv
    K_train_inv = K_train_inv / K_mean  # Don't forget the scaling!
    print("Compute K_train Inversion Finished!\n\n")

else:
    print("Compute K_train Inversion")
    sigma = args.sigma
    K_train_inv = scipy.linalg.inv(K_train + sigma * jnp.eye(train_num))
    K_train_inv = K_train_inv / K_mean  # Don't forget the scaling!
    print("Compute K_train Inversion Finished!\n\n")

accuracy_all = []
for itx in range(10):
    print(f"Compute Test NTK of iter {itx}")
    y_test_sample = []
    K_train_test = jnp.zeros([train_num + test_num, train_num + test_num])
    K_train_test = jax.ops.index_update(K_train_test, jax.ops.index[:train_num, :train_num], K_train)

    for j in range(iter_test):
        rng_key, _ = jax.random.split(rng_key)
        ntk_idx = jax.random.permutation(rng_key, X_test.shape[0])[:ntk_dim]
        X_ntk_2 = X_test[ntk_idx, :]
        def forward(params):
            return apply_fn_train(params, state, rng_key, X_ntk_2)[0]
        J_2 = jax.jacrev(forward)(params)
        y_test_sample.append(y_test[ntk_idx, :])

        for i in tqdm(range(iter_train)):
            rng_key, _ = jax.random.split(rng_key)
            X_ntk_1 = X_train[i * ntk_dim:(i + 1) * ntk_dim, :]
            def forward(params):
                return apply_fn_train(params, state, rng_key, X_ntk_1)[0]
            J_1 = jax.jacrev(forward)(params)

            ntk_batch = J_J_T(J_1, J_2, jnp.zeros([ntk_dim, ntk_dim]))
            K_train_test = jax.ops.index_update(K_train_test, jax.ops.index[i * ntk_dim:(i+1) * ntk_dim, train_num + j * ntk_dim:train_num + (j+1) * ntk_dim], ntk_batch)
            K_train_test = jax.ops.index_update(K_train_test, jax.ops.index[train_num + j * ntk_dim:train_num + (j+1) * ntk_dim, i * ntk_dim:(i+1) * ntk_dim], ntk_batch.T)

    y_test_sample = jnp.concatenate(y_test_sample, axis=0)
    y_test_hat = K_train_test[train_num:, :train_num] @ K_train_inv @ y_train
    acc = jnp.equal(y_test_sample.argmax(-1), y_test_hat.argmax(-1)).mean()
    print(f"Accuracy:{acc}")
    accuracy_all.append(acc)

print("Saving Results")
file_name = f'/home/function_map/results/kernel_reg_{args.dataset}/acc_{train_num}_{sigma}.npy'
np.save(file_name, jnp.array(accuracy_all))

