import numpy as np
from typing import Tuple
from functools import partial
import jax
import jax.numpy as jnp
from jax import jit, random
import matplotlib.pyplot as plt
import haiku as hk
import os
import optax
import argparse
from tqdm import tqdm
import loss_classification
import dataset
import mlp_mixer.MLP_mixer_mod as MLP_mixer_mod
import mlp_mixer.checkpoint as checkpoint
import utils
# config.update('jax_disable_jit', True)

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
abspath = os.path.abspath(__file__)
path = os.path.dirname(abspath)
os.chdir(path)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='map')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--dummy_num', type=int, default=10)
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--element_wise', action="store_true", default=False)
parser.add_argument('--aug', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=200)
args = parser.parse_args()

# Useful Global Variables
rng_key = jax.random.PRNGKey(0)
eps = 1e-6
class_num = 10

# Hyperparameter
regularization = args.reg
element_wise = args.element_wise
dummy_input_dim = args.dummy_num
inverse = args.inverse
epochs = args.epochs

# Load Dataset
image_size, num_classes, train_loader, test_loader = dataset.get_CIFAR10(
    batch_size=args.batch_size,
    data_augmentation=args.aug,
    train_size=args.train_size,
    crop_size=224)

# Model Initialization
x_init = jnp.ones(image_size)

net = MLP_mixer_mod.MlpMixer(patches=[16, 16],
                             num_classes=10,
                             num_blocks=12,
                             hidden_dim=768,
                             tokens_mlp_dim=384,
                             channels_mlp_dim=3072)
init_state, init_params = net.init(rng_key, x_init).pop('params')

pretrained_path = '/home/xzhoubi/hudson/function_map/ckpts/imagenet1k_Mixer-B_16.npz'
params = checkpoint.load_pretrained(pretrained_path, init_params)
state = init_state

# Optimizer Initialization
opt = optax.adam(1e-3)
opt_state = opt.init(params)


def loss_fun(params, state, x, y):
    y_hat = jax.nn.softmax(net.apply({'params': params, **state}, x, mutable=list(state.keys()))[0], axis=1)
    log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
    return -log_likelihood, state


@jit
def update(params: hk.Params,
           state: hk.State,
           opt_state: optax.OptState,
           x,
           y,
           ):
    """Learning rule (stochastic gradient descent)."""
    grads, new_state = jax.grad(loss_fun, argnums=0, has_aux=True)(params, state, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, new_state


train_loss_list = []
train_acc_list = []
train_llk_list = []

test_loss_list = []
test_acc_list = []
test_llk_list = []

print(f"Partial Training Image Size:{len(train_loader) * args.batch_size}")

print(f"--- Start Training with {args.method}--- \n")
for epoch in tqdm(range(epochs)):
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = utils.tensor2array(image, label)
        params, opt_state, state = update(params, state, opt_state, image, label)

    if epoch % 5 == 0:
        train_loss = 0
        train_acc = 0
        train_llk = 0
        for idx, (image, label) in enumerate(train_loader):
            image, label = utils.tensor2array(image, label)
            rng_key, _ = random.split(rng_key)
            loss_value = loss_fun(params, state, image, label)
            train_loss += loss_value

            preds = net.apply({'params': params, **state}, image, mutable=list(state.keys()))[0]
            acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
            train_acc += acc

            llk = loss_fun(params, state, image, label)
            train_llk += llk

        test_loss = 0
        test_acc = 0
        test_llk = 0
        for batch_idx, (image, label) in enumerate(test_loader):
            image, label = utils.tensor2array(image, label)
            rng_key, _ = random.split(rng_key)
            loss_value = loss_fun(params, state, image, label)
            test_loss += loss_value

            preds = net.apply({'params': params, **state}, image, mutable=list(state.keys()))[0]
            acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
            test_acc += acc

            llk = loss_fun(params, state, image, label)
            test_llk += llk

        train_loss /= len(train_loader)
        train_acc /= (args.train_size / 100.)
        train_llk /= len(train_loader)
        train_loss /= len(train_loader)
        train_acc /= (len(train_loader) * args.batch_size / 100.)
        train_llk /= len(train_loader)

        test_loss /= len(test_loader)
        test_acc /= (len(test_loader) * args.batch_size / 100.)
        test_llk /= len(test_loader)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_llk_list.append(train_llk)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_llk_list.append(train_llk)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        test_llk_list.append(test_llk)

        print(
            f"Epoch:{epoch} Partial Train Acc:{train_acc:2f}% Train Acc:{train_acc:2f}% Test Acc:{test_acc:2f}%")
        print(f"Epoch:{epoch} Partial Train LLK:{train_llk:2f} Train LLK:{train_llk:2f} Test LLK:{test_llk:2f}")
        print(
            f"Epoch:{epoch} Partial Train Loss:{train_loss:3f} Train Loss:{train_loss:3f} Test Loss:{test_loss:3f}")

fig = plt.figure(figsize=(15, 15))
ax_all = fig.subplots(3, 3).flatten()
ax_all[0].plot(np.array(train_loss_list))
ax_all[0].set_title(f"Partial Train Loss")
ax_all[1].plot(np.array(train_acc_list))
ax_all[1].set_title(f"Partial Train Accuracy")
ax_all[2].plot(np.array(train_llk_list))
ax_all[2].set_title(f"Partial Train Log-likelihood")

ax_all[3].plot(np.array(train_loss_list))
ax_all[3].set_title(f"Complete Train Loss")
ax_all[4].plot(np.array(train_acc_list))
ax_all[4].set_title(f"Complete Train Accuracy")
ax_all[5].plot(np.array(train_llk_list))
ax_all[5].set_title(f"Complete Train Log-likelihood")

ax_all[6].plot(np.array(test_loss_list))
ax_all[6].set_title(f"Test Loss")
ax_all[7].plot(np.array(test_acc_list))
ax_all[7].set_title(f"Test Accuracy")
ax_all[8].plot(np.array(test_llk_list))
ax_all[8].set_title(f"Test Log-likelihood")

plt.show()
