# Gradient-based learning for the $F$-measure and other performance metrics

## System requirements

We only tested our implementation on Ubuntu 16.04 and Python 3.6.5.
Please follow website instructions to install pytorch 1.0.0 and torchvision 0.2.1 (GPU recommended).
Please install the following packages via `pip`:

```
pip install -U tensorflow
pip install tensorboardX
pip install -U protobuf
```

## Datasets

Please download the Adult, CIFAR10, CIFAR100, Covertype, KDDCup08, Letter and MNIST dataset:

```
bash data.sh
```

## Experiments

Binary and multi-class classification experiments are implemented respectively in `binary.py` and `multi.py`. To reproduce results in our paper:

### Binary classification, F1 score, and linear classifier

```
bash binary.sh f1 linear NUMBER-OF-PROCESSES
for x in f1-linear-*; do bash select.sh $x f1; done
```

### Binary classification, F1 score, and multi-layer perceptron

```
bash binary.sh f1 mlp NUMBER-OF-PROCESSES
for x in f1-mlp-*; do bash select.sh $x f1; done
```

### Multi-class classification, micro F1 score, and linear classifier

```
bash multi.sh f1_micro linear NUMBER-OF-PROCESSES
for x in f1_micro-linear-*; do bash select.sh $x f1; done
```

### Multi-class classification, micro F1 score, and multi-layer perceptron

```
bash multi.sh f1_micro mlp NUMBER-OF-PROCESSES
for x in f1_micro-mlp-*; do bash select.sh $x f1; done
```

### Binary classification, G-measure (a.k.a. Fowlkes–Mallows index), and linear classifier

```
bash binary.sh g1 linear NUMBER-OF-PROCESSES
for x in g1-linear-*; do bash select.sh $x g1; done
```

### Binary classification, G-measure (a.k.a. Fowlkes–Mallows index), and multi-layer perceptron

```
bash binary.sh g1 mlp NUMBER-OF-PROCESSES
for x in g1-mlp-*; do bash select.sh $x g1; done
```

### Multi-class classification, G-measure (a.k.a. Fowlkes–Mallows index), and linear classifier

```
bash multi.sh g1_micro linear NUMBER-OF-PROCESSES
for x in g1_micro-linear-*; do bash select.sh $x g1; done
```

### Multi-class classification, G-measure (a.k.a. Fowlkes–Mallows index), and multi-layer perceptron

```
bash multi.sh g1_micro mlp NUMBER-OF-PROCESSES
for x in g1_micro-mlp-*; do bash select.sh $x g1; done
```
