import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 5, 6"
# path = '/import/home/xzhoubi/hudson/function_map'
# os.chdir(path)
print(os.getcwd())
import sys
sys.path.append('..')
import jax
import flax
import jax.numpy as jnp
from jax import jit, random
import optax
import argparse
from tqdm import tqdm
import dataset
import mlp_mixer.MLP_mixer_mod as MLP_mixer_mod
import mlp_mixer.checkpoint as checkpoint
from mlp_mixer.utils_logging_pmap import Evaluate
import utils

# config.update('jax_disable_jit', True)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='map')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--dummy_num', type=int, default=10)
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--optimizer', type=str, default='none')
parser.add_argument('--element_wise', action="store_true", default=False)
parser.add_argument('--aug', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=200)
args = parser.parse_args()
kwargs = {}

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
n_devices = jax.local_device_count()
print(n_devices)

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
def schedule_fn(learning_rate, n_batches):
    epoch_points = [int(args.epochs * 0.5), int(args.epochs * 0.8)]
    epoch_points = (jnp.array(epoch_points) * n_batches).tolist()
    return utils.piecewise_constant_schedule(learning_rate, epoch_points, args.lr_decay)


if args.optimizer == "adam":
    # opt = optax.adam(args.lr)
    schedule_fn_final = schedule_fn(args.lr, len(train_loader))
    opt = optax.chain(
        optax.scale_by_adam(eps=1e-4),
        optax.scale_by_schedule(schedule_fn_final),
        optax.scale(-1))
elif args.optimizer == "sgd":
    schedule_fn_final = schedule_fn(args.lr, len(train_loader))
    momentum = 0.9
    opt = optax.chain(
            optax.trace(decay=momentum, nesterov=False),
            optax.scale_by_schedule(schedule_fn_final),
            optax.scale(-1),
        )
else:
    raise NotImplementedError
opt_state = opt.init(params)

# Replicate to Multiple GPU
params = flax.jax_utils.replicate(params)
opt_state = flax.jax_utils.replicate(opt_state)


def loss_fun(params, x, y):
    y_hat = jax.nn.softmax(net.apply({'params': params, **state}, x, mutable=list(state.keys()))[0], axis=1)
    log_likelihood = jnp.mean(jnp.sum((jnp.log(y_hat + eps)) * y, axis=1), axis=0)
    return -log_likelihood


@jit
def update(params,
           opt_state: optax.OptState,
           x,
           y,
           ):
    """Learning rule (stochastic gradient descent)."""
    loss_value, grads = jax.value_and_grad(loss_fun, argnums=0)(params, x, y)
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss_value = jax.lax.pmean(loss_value, axis_name='num_devices')
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss_value, new_params, opt_state


update = jax.pmap(update, axis_name='num_devices')

Evaluate = Evaluate(apply_fn=net.apply,
                    n_devices=n_devices,
                    kwargs=kwargs)

print(f"Partial Training Image Size:{len(train_loader) * args.batch_size}")
print(f"--- Start Training with {args.method}--- \n")
for epoch in range(epochs):
    for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
        image, label = utils.tensor2array(image, label)
        image = utils.split(image, n_devices)
        label = utils.split(label, n_devices)
        loss_value, params, opt_state = update(params, opt_state, image, label)

        if batch_idx % 100 == 0:
            print('loss value', loss_value.mean())


    metric_train = Evaluate.evaluate(train_loader,
                                     params,
                                     state,
                                     rng_key,
                                     batch_size=args.batch_size)
    metric_test = Evaluate.evaluate(test_loader,
                                    params,
                                    state,
                                    rng_key,
                                    batch_size=1000)

    print(f"Epoch:{epoch} Train Acc:{metric_train['acc']:2f}% Test Acc:{metric_test['acc']:2f}%")
    print(f"Epoch:{epoch} Train LLK:{metric_train['llk']:2f} Test LLK:{metric_test['llk']:2f}")

# fig = plt.figure(figsize=(15, 15))
# ax_all = fig.subplots(3, 3).flatten()
# ax_all[0].plot(np.array(train_loss_list))
# ax_all[0].set_title(f"Partial Train Loss")
# ax_all[1].plot(np.array(train_acc_list))
# ax_all[1].set_title(f"Partial Train Accuracy")
# ax_all[2].plot(np.array(train_llk_list))
# ax_all[2].set_title(f"Partial Train Log-likelihood")
#
# ax_all[3].plot(np.array(train_loss_list))
# ax_all[3].set_title(f"Complete Train Loss")
# ax_all[4].plot(np.array(train_acc_list))
# ax_all[4].set_title(f"Complete Train Accuracy")
# ax_all[5].plot(np.array(train_llk_list))
# ax_all[5].set_title(f"Complete Train Log-likelihood")
#
# ax_all[6].plot(np.array(test_loss_list))
# ax_all[6].set_title(f"Test Loss")
# ax_all[7].plot(np.array(test_acc_list))
# ax_all[7].set_title(f"Test Accuracy")
# ax_all[8].plot(np.array(test_llk_list))
# ax_all[8].set_title(f"Test Log-likelihood")
#
# plt.show()
