# CI-BCNN
This is the pytorch implementation for paper: *Learning Channel-wise Interactions for Binary Convolutional Neural Networks*.

# Quick Start
## Prerequisites
- python 3.5+
- pytorch 1.0.1
- keras 2.2.3
- other packages include numpy, tqdm


## Dataset & Backbone
Our demo code is for the experiment on CIFAR-10 with the backbone of Resnet-20.

## Training and Testing
With all required packages, you can start using the code in the following way.

To train from scratch, run:
```shell
python main.py
```

To train with pretrained backbone, run:
```shell
python main.py --pretrain 'path/to/weight'
```

To evaluate, put the .npy files (xx.npy, yy.npy, influence_state.npy) in one directory, run:
```shell
python main.py --evaluate True --pretrain 'path/to/weight' --CI 'dir/to/npys'
```


