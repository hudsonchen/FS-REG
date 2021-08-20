from typing import Tuple, Callable, List
from jax import jit
import jax
import jax.numpy as jnp
from functools import partial
import haiku as hk
import os
import optax
import argparse
from tqdm import tqdm
import loss_classification
import dataset
import network
import utils
import utils_logging
import resnet_mod

# from jax import config
# config.update('jax_disable_jit', True)


os.environ["CUDA_VISIBLE_DEVICES"] = "4,5"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
abspath = os.path.abspath(__file__)
path = os.path.dirname(abspath)
os.chdir(path)

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='map')
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--architecture', type=str, default='none')
parser.add_argument('--optimizer', type=str, default='none')
parser.add_argument('--reg', type=float, default='0.01')
parser.add_argument('--lr_decay', type=float, default='0.5')
parser.add_argument('--dummy_input_dim', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--inverse', action="store_true", default=False)
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
    def forward(x, is_training):
        net = resnet_mod.ResNet18(10, use_bn=True)
        return net(x, is_training)


    forward = hk.transform_with_state(forward)
    init_fn = partial(forward.init, is_training=True)
    apply_fn_train = partial(forward.apply, is_training=True)
    apply_fn_eval = partial(forward.apply, is_training=False)
elif args.architecture == "vgg11":
    pass
else:
    raise NotImplementedError

x_init = jnp.ones(image_size)
params_init, state = init_fn(rng_key, x_init)
params = params_init


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
                                                                        dummy_input_dim=args.dummy_input_dim,
                                                                        class_num=class_num,
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
elif args.method == 'laplacian_norm':
    loss_fun = loss_classification_list.laplacian_norm_loss
llk_classification = loss_classification_list.llk_classification

Evaluate = utils_logging.Evaluate(apply_fn=apply_fn_train,
                                  loss_fn=loss_fun,
                                  llk_fn=llk_classification,
                                  kwargs=kwargs)

print(f"Partial Training Image Size:{len(train_loader) * args.batch_size}")

print(f"--- Start Training with {args.method}--- \n")
for epoch in tqdm(range(args.epochs)):
    for batch_idx, (image, label) in enumerate(train_loader):
        image, label = utils.tensor2array(image, label)
        rng_key, _ = jax.random.split(rng_key)
        params, state, opt_state = update(params, state, opt_state, rng_key, image, label)

    if (epoch + 1) % args.log_freq == 0:
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

        print(f"Epoch:{epoch} Partial Train Acc:{metric_train['acc']:2f}% Test Acc:{metric_test['acc']:2f}%")
        print(f"Epoch:{epoch} Partial Train LLK:{metric_train['llk']:2f} Test LLK:{metric_test['llk']:2f}")
        print(f"Epoch:{epoch} Partial Train ECE:{metric_train['ece']:2f} Test ECE:{metric_test['ece']:2f}")
        print(f"Epoch:{epoch} Partial Train Loss:{metric_train['loss']:3f} Test Loss:{metric_test['loss']:3f}")

        if args.save:
            Evaluate.save_log(epoch, metric_train, metric_test)
            Evaluate.save_params(epoch, params, state)

if args.save:
    save_path = kwargs["save_path"]
    print(f"\nChanging save path from\n\n{save_path}\n\nto\n\n{save_path}__complete\n")
    os.rename(save_path, f"{save_path}__complete")
