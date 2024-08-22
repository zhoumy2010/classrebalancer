import random
import matplotlib

matplotlib.use('Agg')
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os

from collections import Counter
from utils.sampling import cifar_iid, non_iid_dirichlet_sampling
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Fed import FedWeightAvg
from models.test import test_img
from models.model_res import ResNet18
from utils.imbalance_cifar import IMBALANCECIFAR10
from utils.util import calculate_macro_metrics


if __name__ == '__main__':

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    print(f'{args.dataset}_{args.model}_{args.loss_type}loss_IF{args.imb_factor}_alpha{args.alpha_dirichlet}')

    if args.dataset == 'cifar10':
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])],
        )
        dataset_train = IMBALANCECIFAR10('./data', imb_factor=args.imb_factor, train=True, download=True, transform=trans_cifar_train)
        dataset_test = IMBALANCECIFAR10('./data', imb_factor=args.imb_factor, train=False, download=True, transform=trans_cifar_test)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            dict_users, user_class_counts = non_iid_dirichlet_sampling(dataset_train, args.num_classes, args.num_users, args.seed, args.alpha_dirichlet)
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    clients_sizes = [len(dict_users[i]) for i in range(args.num_users)]
    print("clients_sizes:{}".format(clients_sizes))

    for user_id, class_counts in user_class_counts.items():
        print(f"User {user_id} class sample counts: {class_counts}")

    if args.model == 'resnet18':
        block_expansion = 1
        net_glob = ResNet18(args.num_classes).to(args.device)
        classifier_glob = torch.nn.Linear(512 * block_expansion, args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')

    print(net_glob)

    g_linears = [torch.nn.Linear(512 * block_expansion, args.num_classes) for i in range(args.num_users)]
    for i in range(args.num_users):
        g_linears[i] = g_linears[i].to(args.device)

    best_F1 = 0

    acc_test = []
    f1_test = []
    clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
               for idx in range(args.num_users)]
    m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)
    for iter in range(args.epochs):
        backbone_w_locals, linear_w_locals, loss_locals, weight_locols = [], [], [], []
        idxs_users = np.random.choice(clients_index_array, m, replace=False)
        for idx in idxs_users:
            backbone_w_local, linear_w_local, loss = clients[idx].train(net=copy.deepcopy(net_glob).to(args.device), linear_layer=g_linears[idx])
            backbone_w_locals.append(copy.deepcopy(backbone_w_local))
            linear_w_locals.append(copy.deepcopy(linear_w_local))
            loss_locals.append(copy.deepcopy(loss))
            weight_locols.append(len(dict_users[idx]))

        backbone_w_avg, linear_w_avg = FedWeightAvg(backbone_w_locals, linear_w_locals, weight_locols)

        net_glob.load_state_dict(copy.deepcopy(backbone_w_avg))
        classifier_glob.load_state_dict(copy.deepcopy(linear_w_avg))

        g_linears = [copy.deepcopy(classifier_glob) for i in range(args.num_users)]

        net_glob.eval()
        classifier_glob.eval()
        acc_t, loss_t, precision_t, recall_t, f1_t, best_F1 = test_img(net_glob, classifier_glob, dataset_test, best_F1, iter, args)
        print(f"Round {iter:3d}, Testing loss: {loss_t:.3f}, accuracy: {acc_t:.4f}. precision: {precision_t:.4f}, recall: {recall_t:.4f}, f1-score: {f1_t:.4f}, Best-F1: {best_F1:.4f}")

        acc_test.append(acc_t.item())
        f1_test.append(f1_t.item())

    rootpath = f'./{args.dataset}_{args.model}_{args.loss_type}loss_IF{args.imb_factor}_alpha{args.alpha_dirichlet}_log'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)
    f1file = open(rootpath + f'/F1scorefile_fed_{args.dataset}_{args.model}_{args.epochs}_iid{args.iid}_lrdecay{args.lr_decay}.dat', "w")
    for f1 in f1_test:
        sf1 = str(f1)
        f1file.write(sf1)
        f1file.write('\n')
    f1file.close()

    print(f'DisCB_epochs={args.DisCB_epochs}_learningRate={args.DisCB_lr}_h={args.h}')
    # f1, precision, recall
    selected_metric = 'f1'

    labels = [label for _, label in dataset_train]

    label_counts = Counter(labels)

    num_classes = args.num_classes

    pi = np.zeros(num_classes)

    total_count = len(labels)
    for label, count in label_counts.items():
        pi[label] = count / total_count

    if args.model == 'resnet18':
        block_expansion = 1
        net_glob = ResNet18(args.num_classes).to(args.device)
        classifier_glob = torch.nn.Linear(512 * block_expansion, args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')


    checkpoint = torch.load(f'checkpoint_{args.dataset}_{args.model}_{args.loss_type}loss_IF{args.imb_factor}_alpha{args.alpha_dirichlet}/best_model.pth')
    net_glob.load_state_dict(checkpoint['net'])
    classifier_glob.load_state_dict(checkpoint['classifier'])
    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from epoch {epoch}")

    clients = [LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx]) for idx in range(args.num_users)]
    m, clients_index_array = max(int(args.frac * args.num_users), 1), range(args.num_users)

    L_rate = []
    pi_values = []

    for iter in range(args.DisCB_epochs):
        idxs_users = np.random.choice(clients_index_array, m, replace=False)

        L = np.zeros(num_classes)
        for temp_class in range(num_classes):
            confusion_matrix = np.zeros((num_classes, 4))

            for idx in idxs_users:
                temp_confusion_matrix = clients[idx].confusionMatrix(net=copy.deepcopy(net_glob).to(args.device), classifier=copy.deepcopy(classifier_glob).to(args.device), num_classes=num_classes, pi=pi, temp_class=temp_class, h=args.h)
                confusion_matrix += temp_confusion_matrix

            macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(confusion_matrix)

            if selected_metric == 'precision':
                L[temp_class] = macro_precision
            elif selected_metric == 'recall':
                L[temp_class] = macro_recall
            elif selected_metric == 'f1':
                L[temp_class] = macro_f1

        g = np.sum(L) / (num_classes * args.h)
        tmp = (L / args.h) - g
        tmp_pi = pi + args.DisCB_lr * tmp
        for j in range(num_classes):
            if tmp_pi[j] < 0:
                tmp_pi[j] = 0.000000001

        tmp_pi = tmp_pi / np.sum(tmp_pi)

        c_matrix = np.zeros((num_classes, 4))
        for idx in idxs_users:
            temp_confusion_matrix = clients[idx].updateX(num_classes=num_classes, tmp_pi=tmp_pi)
            c_matrix += temp_confusion_matrix

        macro_precision, macro_recall, macro_f1 = calculate_macro_metrics(confusion_matrix)
        print(f"Round {iter:3d},Train set: precision: {macro_precision:.4f}, recall: {macro_recall:.4f}, f1-score: {macro_f1:.4f}")

        pi = tmp_pi

        if selected_metric == 'precision':
            l = macro_precision
        elif selected_metric == 'recall':
            l = macro_recall
        elif selected_metric == 'f1':
            l = macro_f1

        L_rate.append(l)
        pi_values.append(np.copy(pi))

    max_l_index = np.argmax(L_rate)
    max_l_pi = pi_values[max_l_index]

    print('Max L_rate:', L_rate[max_l_index])
    print('Epoch', max_l_index)
    print('Corresponding pi:', max_l_pi.tolist())

    L_rate = np.array(L_rate)

    rootpath = f'./log/{args.dataset}_{args.model}_{args.loss_type}loss_IF{args.imb_factor}_alpha{args.alpha_dirichlet}'
    if not os.path.exists(rootpath):
        os.makedirs(rootpath)

    ratiofile = open(rootpath + f'/{selected_metric}_ratiofile_lr{args.DisCB_lr}_h{args.h}.dat', "w")
    ratiofile.write(str(max_l_pi.tolist()))
    ratiofile.write('\n')
    ratiofile.write(str(L_rate[max_l_index]))
    ratiofile.close()

