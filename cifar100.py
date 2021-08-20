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
import os
import tensorflow_probability.substrates.jax.distributions as tfd
import optax
import argparse
from tqdm import tqdm
import loss_classification
import dataset
import MLP_mixer_mod

from jax.config import config

# config.update('jax_disable_jit', True)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
parser.add_argument('--data_augmentation', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_size', type=float, default=100)
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
data_augmentation = args.data_augmentation
batch_size = 100

# Load Dataset
image_size, num_classes, partial_trainloader, trainloader, testloader = dataset.get_CIFAR10(batch_size=batch_size,
                                                                                            data_augmentation=data_augmentation,
                                                                                            train_size=args.train_size)


# Model Initialization
def forward(x, is_training):
    net = MLP_mixer_mod.MlpMixer(patches=16,
                                 num_classes=10,
                                 num_blocks=8,
                                 hidden_dim=512,
                                 tokens_mlp_dim=256,
                                 channels_mlp_dim=2048)
    return net(x, is_training)


forward = hk.transform_with_state(forward)

init_fn = partial(forward.init, is_training=True)
apply_fn_train = partial(forward.apply, is_training=True)
apply_fn_eval = partial(forward.apply, is_training=False)

x_init = jnp.ones(image_size)
params_init, state = init_fn(rng_key, x_init)
params = params_init

# Optimizer Initialization
opt = optax.adam(1e-3)
opt_state = opt.init(params_init)


@jit
def update(params: hk.Params,
           state: hk.State,
           opt_state: optax.OptState,
           rng_key: jnp.array,
           x,
           y,
           ) -> Tuple[hk.Params, optax.OptState]:
    """Learning rule (stochastic gradient descent)."""
    params_copy = params
    grads, new_state = jax.grad(loss_fun, argnums=0, has_aux=True)(params, params_copy, state, rng_key, x, y)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_state, opt_state


loss_classification_list = loss_classification.loss_classification_list(apply_fn=apply_fn_train,
                                                                        regularization=args.reg,
                                                                        dummy_input_dim=dummy_input_dim,
                                                                        class_num=class_num,
                                                                        inverse=inverse,
                                                                        element_wise=element_wise)

if args.method == 'map':
    loss_fun = jax.jit(loss_classification_list.map_loss)
elif args.method == 'ntk_norm':
    loss_fun = loss_classification_list.ntk_norm_loss
elif args.method == 'jac_norm':
    loss_fun = loss_classification_list.jac_norm_loss
elif args.method == 'f_norm':
    loss_fun = loss_classification_list.f_norm_loss
elif args.method == 'laplacian_norm':
    loss_fun = loss_classification_list.laplacian_norm_loss
llk_classification = loss_classification_list.llk_classification

partial_train_loss_list = []
partial_train_acc_list = []
partial_train_llk_list = []
train_loss_list = []
train_acc_list = []
train_llk_list = []
test_loss_list = []
test_acc_list = []
test_llk_list = []

partial_train_size = len(partial_trainloader) * batch_size
print(f"Partial Training Image Size:{partial_train_size}")

print(f"--- Start Training with {args.method}--- \n")
for epoch in tqdm(range(epochs)):
    for batch_idx, (image, label) in enumerate(partial_trainloader):
        rng_key, _ = random.split(rng_key)
        params, state, opt_state = update(params, state, opt_state, rng_key, image, label)

    if epoch % 5 == 0:
        partial_train_loss = 0
        partial_train_acc = 0
        partial_train_llk = 0
        for idx, (image, label) in enumerate(partial_trainloader):
            rng_key, _ = random.split(rng_key)
            loss_value = loss_fun(params, params, state, rng_key, image, label)[0]
            partial_train_loss += loss_value

            preds = apply_fn_train(params, state, rng_key, image)[0]
            acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
            partial_train_acc += acc

            llk = llk_classification(params, params, state, rng_key, image, label)[0]
            partial_train_llk += llk

        train_loss = 0
        train_acc = 0
        train_llk = 0
        # for batch_idx, (image, label) in enumerate(trainloader):
        #     loss_value = loss_fun(params, state, rng_key, image, label)[0]
        #     train_loss += loss_value
        #
        #     preds = apply_fn_eval(params, state, rng_key, image)[0]
        #     acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
        #     train_acc += acc
        #
        #     llk = llk_classification(params, state, rng_key, image, label)
        #     train_llk += llk

        test_loss = 0
        test_acc = 0
        test_llk = 0
        for batch_idx, (image, label) in enumerate(testloader):
            rng_key, _ = random.split(rng_key)
            loss_value = loss_fun(params, params, state, rng_key, image, label)[0]
            test_loss += loss_value

            preds = apply_fn_train(params, state, rng_key, image)[0]
            acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
            test_acc += acc

            llk = llk_classification(params, params, state, rng_key, image, label)[0]
            test_llk += llk

        partial_train_loss /= len(partial_trainloader)
        partial_train_acc /= (partial_train_size / 100.)
        partial_train_llk /= len(partial_trainloader)
        train_loss /= len(trainloader)
        train_acc /= (len(trainloader) * batch_size / 100.)
        train_llk /= len(trainloader)

        test_loss /= len(testloader)
        test_acc /= (len(testloader) * batch_size / 100.)
        test_llk /= len(testloader)

        partial_train_loss_list.append(partial_train_loss)
        partial_train_acc_list.append(partial_train_acc)
        partial_train_llk_list.append(partial_train_llk)
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)
        train_llk_list.append(train_llk)
        test_loss_list.append(test_loss)
        test_acc_list.append(test_acc)
        test_llk_list.append(test_llk)

        print(
            f"Epoch:{epoch} Partial Train Acc:{partial_train_acc:2f}% Train Acc:{train_acc:2f}% Test Acc:{test_acc:2f}%")
        print(f"Epoch:{epoch} Partial Train LLK:{partial_train_llk:2f} Train LLK:{train_llk:2f} Test LLK:{test_llk:2f}")
        print(
            f"Epoch:{epoch} Partial Train Loss:{partial_train_loss:3f} Train Loss:{train_loss:3f} Test Loss:{test_loss:3f}")

fig = plt.figure(figsize=(15, 15))
ax_all = fig.subplots(3, 3).flatten()
ax_all[0].plot(np.array(partial_train_loss_list))
ax_all[0].set_title(f"Partial Train Loss")
ax_all[1].plot(np.array(partial_train_acc_list))
ax_all[1].set_title(f"Partial Train Accuracy")
ax_all[2].plot(np.array(partial_train_llk_list))
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
