import jax
import jax.numpy as jnp
import csv
import os
import pickle
import uncertainty_metrics.numpy as um
import utils


class Evaluate:
    def __init__(
            self,
            apply_fn,
            loss_fn,
            llk_fn,
            num_classes,
            **kwargs,
    ):
        self.loss_fn = loss_fn
        self.apply_fn = apply_fn
        self.llk_fn = llk_fn
        self.kwargs = kwargs['kwargs']
        self.num_classes = num_classes

    def evaluate(self, loader, params, state, rng_key, batch_size):
        loss_ = 0
        acc_ = 0
        llk_ = 0
        ece_ = 0
        metric = {'loss': {},
                  'acc': {},
                  'llk': {},
                  'ece': {}}

        for batch_idx, (image, label) in enumerate(loader):
            image, label = utils.tensor2array(image, label, self.num_classes)
            rng_key, _ = jax.random.split(rng_key)
            loss_value = self.loss_fn(params, params, state, rng_key, image, label)[0]
            loss_ += loss_value

            preds = self.apply_fn(params, state, rng_key, image)[0]
            acc = jnp.equal(jnp.argmax(preds, axis=1), jnp.argmax(label, axis=1)).sum()
            acc_ += acc

            llk = self.llk_fn(params, params, state, rng_key, image, label)[0]
            llk_ += llk

            # ece = um.ece(label.argmax(-1), jax.nn.softmax(preds, axis=1))
            ece = 0
            ece_ += ece

        loss_ /= len(loader)
        acc_ /= len(loader) * batch_size / 100.
        llk_ /= len(loader)
        ece_ /= len(loader)
        metric['loss'] = loss_
        metric['acc'] = acc_
        metric['llk'] = llk_
        metric['ece'] = ece_
        return metric

    def auroc(self, loader, params, state, rng_key, batch_size):
        auc_value = 0
        for batch_idx, (image, label) in enumerate(loader):
            image, label = utils.tensor2array(image, label, self.num_classes)
            preds = jax.nn.softmax(self.apply_fn(params, state, rng_key, image)[0], axis=1)
            auc_val = roc(jnp.argmax(label, axis=0).tolist(), preds[:, 0].tolist())[2]
            auc_value += auc_val
        auc_value /= len(loader)
        return auc_value

    def save_log(self, epoch, metric_train, metric_test):
        file_name = f'{self.kwargs["save_path"]}/metrics.csv'
        with open(file_name, 'a') as metrics_file:
            metrics_header = [
                'Epoch',
                'Train Loss',
                'Train LLK',
                'Train Acc',
                'Train ECE',
                'Test Loss',
                'Test LLK',
                'Test Acc',
                'Test ECE',
            ]
            writer = csv.DictWriter(metrics_file, fieldnames=metrics_header)
            if os.stat(file_name).st_size == 0:
                writer.writeheader()
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': metric_train['loss'],
                'Train LLK': metric_train['llk'],
                'Train Acc': metric_train['acc'],
                'Train ECE': metric_train['ece'],
                'Test Loss': metric_test['loss'],
                'Test LLK': metric_test['llk'],
                'Test Acc': metric_test['acc'],
                'Test ECE': metric_test['ece'],
            })
            metrics_file.close()

    def save_params(self, epoch, params, state):
        with open(f'{self.kwargs["save_path"]}/params_{epoch}', "wb") as file:
            pickle.dump(params, file)
        with open(f'{self.kwargs["save_path"]}/state_{epoch}', "wb") as file:
            pickle.dump(state, file)


def _trap_area(p1, p2):
    """
    Calculate the area of the trapezoid defined by points
    p1 and p2

    `p1` - left side of the trapezoid
    `p2` - right side of the trapezoid
    """
    base = abs(p2[0] - p1[0])
    avg_ht = (p1[1] + p2[1]) / 2.0

    return base * avg_ht


def roc(dvals, labels, rocN=50, normalize=True):
    """
    Compute ROC curve coordinates and area
    - `dvals`  - a list with the decision values of the classifier
    - `labels` - list with class labels, \in {0, 1}
    returns (FP coordinates, TP coordinates, AUC )
    """
    import numpy
    if rocN is not None and rocN < 1:
        rocN = int(rocN * numpy.sum(numpy.not_equal(labels, 1)))

    TP = 0.0  # current number of true positives
    FP = 0.0  # current number of false positives

    fpc = [0.0]  # fp coordinates
    tpc = [0.0]  # tp coordinates
    dv_prev = -numpy.inf  # previous decision value
    TP_prev = 0.0
    FP_prev = 0.0
    area = 0.0

    num_pos = labels.count(1)  # number of pos labels
    num_neg = labels.count(0)  # number of neg labels

    # sort decision values from highest to lowest
    indices = numpy.argsort(dvals)[::-1]

    idx_prev = -1
    for idx in indices:
        # increment associated TP/FP count
        if labels[idx] == 1:
            TP += 1.
        else:
            FP += 1.
            if rocN is not None and FP == rocN:
                break
        # Average points with common decision values
        # by not adding a coordinate until all
        # have been processed
        if dvals[idx] != dv_prev:
            if len(fpc) > 0 and FP == fpc[-1]:
                tpc[-1] = TP
            else:
                fpc.append(FP)
                tpc.append(TP)
            dv_prev = dvals[idx]
            area += _trap_area((FP_prev, TP_prev), (FP, TP))
            FP_prev = FP
            TP_prev = TP
            idx_prev = idx

    # Last few decision values were all the same,
    # so must append final points and area
    if idx_prev != indices[-1]:
        tpc.append(num_pos)
        fpc.append(num_neg)
        area += _trap_area((FP, TP), (FP_prev, TP_prev))

    # area += _trap_area( ( FP, TP ), ( FP_prev, TP_prev ) )
    # fpc.append( FP  )
    # tpc.append( TP )

    if normalize:
        fpc = [float(x) / FP for x in fpc]
        if TP > 0:
            tpc = [float(x) / TP for x in tpc]
        if area > 0:
            area /= (num_pos * FP)

    return fpc, tpc, area
