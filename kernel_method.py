from typing import Tuple, Callable, List
from jax import jit
import jax
import jax.numpy as jnp
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
parser.add_argument('--save_path', type=str, default="/home/xzhoubi/hudson/function_map/results")
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_size', type=int, default=100)
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

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

ntk_dim = int(50)
train_num = 500
test_num = 100
iter_train = int(train_num / ntk_dim)
iter_test = int(test_num / ntk_dim)
y_train_nystrom = []
y_test_nystrom = []
J_list_train = []
J_list_test = []

print("Compute Train NTK:")
for _ in tqdm(range(iter_train)):
    rng_key, _ = jax.random.split(rng_key)
    ntk_idx = jax.random.permutation(rng_key, X_train.shape[0])[:ntk_dim]
    X_ntk = X_train[ntk_idx, :]
    def forward(params):
        return apply_fn_train(params, state, rng_key, X_ntk)[0]
    J_ = jax.jacrev(forward)(params)
    J_list_train.append(J_)

    y_train_nystrom.append(y_train[ntk_idx, :])

y_train_nystrom = jnp.concatenate(y_train_nystrom, axis=0)

for _ in range(5):
    y_test_nystrom = []
    J_list_test = J_list_train[0:iter_train]
    print(len(J_list_test))

    print("Compute Test NTK:")
    for i in tqdm(range(iter_test)):
        rng_key, _ = jax.random.split(rng_key)
        ntk_idx = jax.random.permutation(rng_key, X_test.shape[0])[:ntk_dim]
        X_ntk = X_test[ntk_idx, :]
        def forward(params):
            return apply_fn_train(params, state, rng_key, X_ntk)[0]
        J_ = jax.jacrev(forward)(params)
        J_list_test.append(J_)

        y_test_nystrom.append(y_test[ntk_idx, :])

    y_test_nystrom = jnp.concatenate(y_test_nystrom, axis=0)
    ntk = jnp.zeros([test_num + train_num, test_num + train_num])

    for a in J_:
        for b in J_[a]:
            J_temp = []
            for J_batch in J_list_test:
                J_temp.append(J_batch[a][b])
            J_temp = jnp.concatenate(J_temp, axis=0)
            J_temp = J_temp.reshape(J_temp.shape[0], -1)
            ntk += J_temp @ J_temp.T

    print('Solve Matrix Inversion:')
    y_test_hat = ntk[train_num:, :train_num].dot(scipy.linalg.solve(ntk[:train_num, :train_num], y_train_nystrom))
    acc = jnp.equal(y_test_nystrom.argmax(-1), y_test_hat.argmax(-1)).mean()
    print(f"Accuracy:{acc}")

