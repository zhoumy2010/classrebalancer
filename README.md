## Usage


+ Here is an example to run DisCB on CIFAR-10 with non-IID data partition with imbalanced factor 50 , alpha=1 over 10 clients:

```
python main.py --dataset cifar10 \
--alpha_dirichlet 1 \
--imb_factor 0.02 \
--epochs 200 \
--num_users 10 \
--model resnet18 \
--gpu 0 \
--DisCB_epochs 200 \
--DisCB_lr 0.001 \
--h 0.001
```



