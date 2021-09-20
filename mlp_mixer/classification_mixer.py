import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3, 4, 5, 6, 7"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION=.XX"] = "0.95"
# path = '/home/weizhong/hudson/function_map/mlp_mixer'
path = '/home/xzhoubi/hudson/function_map/mlp_mixer'
os.chdir(path)
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
import mlp_mixer.loss_mixer as loss_mixer
import mlp_mixer.checkpoint as checkpoint
from mlp_mixer.utils_logging_pmap import Evaluate
import utils
import mlp_mixer.utils_mixer as utils_mixer

# config.update('jax_disable_jit', True)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='map')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--dummy_input_dim', type=int, default=10)
parser.add_argument('--inverse', action="store_true", default=False)
parser.add_argument('--optimizer', type=str, default='none')
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--element_wise', action="store_true", default=False)
parser.add_argument('--aug', action="store_true", default=False)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay', type=float, default=0.3)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--train_size', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=200)
parser.add_argument('--save_path', type=str, default="/home/xzhoubi/hudson/function_map/results/mlp_mixer")
parser.add_argument('--save', action="store_true", default=False)
args = parser.parse_args()
kwargs = utils_mixer.process_args(args)

# Useful Global Variables
rng_key = jax.random.PRNGKey(args.seed)
eps = 1e-6

# Hyperparameter
regularization = args.reg
element_wise = args.element_wise
dummy_input_dim = args.dummy_input_dim
inverse = args.inverse
epochs = args.epochs
n_devices = jax.local_device_count()
print(n_devices)

# Load Dataset
if args.dataset == 'cifar10':
    image_size, num_classes, train_loader, test_loader = dataset.get_CIFAR10(
        batch_size=args.batch_size,
        data_augmentation=args.aug,
        train_size=args.train_size,
        crop_size=224)
elif args.dataset == 'cifar100':
    image_size, num_classes, train_loader, test_loader = dataset.get_CIFAR100(
        batch_size=args.batch_size,
        data_augmentation=args.aug,
        train_size=args.train_size,
        crop_size=224)
else:
    raise NotImplementedError(args.dataset)
# Model Initialization
x_init = jnp.ones(image_size)
print(image_size)

net = MLP_mixer_mod.MlpMixer(patches=[16, 16],
                             num_classes=num_classes,
                             num_blocks=12,
                             hidden_dim=768,
                             tokens_mlp_dim=384,
                             channels_mlp_dim=3072)
init_state, init_params = net.init(rng_key, x_init).pop('params')

pretrained_path = '/home/xzhoubi/hudson/function_map/ckpts/imagenet1k_Mixer-B_16.npz'
# pretrained_path = '/home/weizhong/hudson/function_map/ckpts/imagenet1k_Mixer-B_16.npz'
params = checkpoint.load_pretrained(pretrained_path, init_params)
del init_params
state = init_state


# Optimizer Initialization
def schedule_fn(learning_rate, n_batches):
    epoch_points = [int(args.epochs * 0.3), int(args.epochs * 0.5), int(args.epochs * 0.8)]
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
    raise NotImplementedError(args.optimizer)
opt_state = opt.init(params)

# Replicate to Multiple GPU
params = flax.jax_utils.replicate(params)
opt_state = flax.jax_utils.replicate(opt_state)
# rng_key = flax.jax_utils.replicate(rng_key)

loss_classification_list = loss_mixer.loss_classification_list(apply_fn=net.apply,
                                                               regularization=args.reg,
                                                               dummy_input_dim=args.dummy_input_dim,
                                                               class_num=num_classes,
                                                               inverse=args.inverse,
                                                               element_wise=args.element_wise)
if args.method == 'map' or args.method == 'map_no_wd':
    loss_fun = loss_classification_list.map_loss
elif args.method == 'ntk_norm':
    loss_fun = loss_classification_list.ntk_norm_loss
elif args.method == 'jac_norm':
    loss_fun = loss_classification_list.jac_norm_loss
elif args.method == 'f_norm':
    loss_fun = loss_classification_list.f_norm_loss
else:
    raise NotImplementedError(args.method)


@jit
def update(params,
           opt_state,
           rng_key,
           x,
           y,
           ):
    """Learning rule (stochastic gradient descent)."""
    loss_value, grads = jax.value_and_grad(loss_fun, argnums=0)(params, params, rng_key, x, y)
    grads = jax.lax.pmean(grads, axis_name='num_devices')
    loss_value = jax.lax.pmean(loss_value, axis_name='num_devices')
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return loss_value, new_params, opt_state


update = jax.pmap(update, axis_name='num_devices')

Evaluate = Evaluate(apply_fn=net.apply,
                    n_devices=n_devices,
                    kwargs=kwargs,
                    num_classses=num_classes)

print(f"Partial Training Image Size:{len(train_loader) * args.batch_size}")
print(f"--- Start Training with {args.method}--- \n")
for epoch in range(epochs):
    for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
        image, label = utils.tensor2array(image, label, num_classes)
        image = utils.split(image, n_devices)
        label = utils.split(label, n_devices)
        _, rng_key = jax.random.split(rng_key)
        rng_key_multi = jax.random.split(rng_key, num=8)
        loss_value, params, opt_state = update(params, opt_state, rng_key_multi, image, label)

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

    if args.save:
        Evaluate.save_log(epoch, metric_train, metric_test)
        # Evaluate.save_params(epoch, params, state)

if args.save:
    save_path = kwargs["save_path"]
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    os.rename(save_path, f"{save_path}__complete")
