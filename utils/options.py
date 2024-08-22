import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=10, help="number of users")
    parser.add_argument('--frac', type=float, default=0.8, help="the fraction of clients")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs")
    parser.add_argument('--local_bs', type=int, default=128, help="local batch size")
    parser.add_argument('--bs', type=int, default=512, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.03, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.95, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.9, help="SGD momentum")
    parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
    parser.add_argument('--model', type=str, default='resnet18', help='model name')
    parser.add_argument('--DisCB_epochs', type=int, default=50, help="rounds of DisCB")
    parser.add_argument('--DisCB_lr', type=float, default=0.001, help="learning rate of DisCB")
    parser.add_argument('--h', type=float, default=0.001, help="h of DisCB")
    parser.add_argument('--dataset', type=str, default='cifar10', help="name of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--imb_factor', default=0.02, type=float, help='imbalance factor')
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--alpha_dirichlet', type=float, default=0.5, help="alpha_dirichlet")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--verbose', action='store_true', help='verbose print')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    args = parser.parse_args()
    return args
