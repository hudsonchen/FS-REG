# A Neural Tangent Kernel Perspecitve for Function Space Regularization in Neural Networks

This repository contains the official implementation for

_**Continual Learning via Sequential Function-Space Variational Inference**_; Zonghao Chen, Xupeng Shi, Tim G. J. Rudner, 
Qixuan Feng, Weizhong Zhang, Tong Zhang. 

**Abstract:** Regularization can help reduce the gap between training and test error by systematically limiting model complexity. 
Popular regularization techniques such as $\ell_2$ weight regularization act directly on the network parameters, but 
do not explicitly take into account how the interplay between the parameters and the network architecture may affect the induced predictive functions.
To address this shortcoming, we propose a simple technique for effective function-space regularization. Drawing on the 
result that fully-trained wide multi-layer perceptrons are equivalent to kernel regression under the Neural Tangent Kernel (NTK), 
we propose to approximate the norm of neural network functions by the \textit{reproducing kernel Hilbert space} norm under the NTK  and use it as a function-space regularizer. We prove that neural networks trained using this regularizer are arbitrarily close to kernel ridge regression solutions under the NTK. Furthermore, we provide a generalization error bound under the proposed regularizer and empirically 
demonstrate improved generalization and state-of-the-art performance on downstream tasks where effective regularization 
on the induced space of functions is essential.

## Installation
Check out the requirements.txt file.

## Reproducing results
To reproduce the results on MNIST \
`python classification.py --dataset mnist --optimizer sgd --architecture lenet 
--lr 0.01 --method ntk_norm --inverse --reg 100.0 --epochs 200 --train_size 100 
--lr_decay 1.0 --seed 0`

To reproduce the results on CIFAR10 \
`python classification.py --dataset cifar10 --optimizer adam --architecture resnet18 
--lr 0.001 --batch_size 200 --method ntk_norm --inverse --dummy_input_dim 10 --reg 10.0 
--epochs 150 --train_size 10000 --lr_decay 1.0 --aug --seed 0`

To reproduce the results on CIFAR100 with MLP-Mixer \
`python mlp_mixer/classification_mixer.py --lr 0.01 --dataset cifar100 --method ntk_norm --optimizer sgd --reg 10 
--inverse --dummy_input_dim 2 --element_wise
--epochs 10 --train_size 50000 --batch_size 128 --aug --seed 0`

To reproduce the results on Permutated MNIST (continual learning) \
`python cl/continual_learning.py
--dataset pmnist --method ntk_norm --hidden_dim 100 --epochs 10 --reg 100.0 --bs 200 --head_style single --inverse 
--element_wise --lr 3e-4 --dummy_num 20 --ind_method core --coreset_size 200`

To reproduce the results on Split MNIST (continual learning) \
`python cl/continual_learning.py
--dataset pmnist --dataset smnist --method ntk_norm --epochs 30 --reg 10.0 --head_style multi --inverse --element_wise 
--lr 3e-4 --dummy_num 40 --hidden_dim 256 --ind_method core --coreset_size 40`

