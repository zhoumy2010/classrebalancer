import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import random
from sklearn import metrics
from utils.losses import *


def softmax(x):
    e_x = torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
    return e_x / e_x.sum(dim=1, keepdim=True)


def calculate_confusion_matrix_elements(predictions, labels, num_classes):
    confusion_matrix = np.zeros((num_classes, 4))

    predicted_labels = np.argmax(predictions, axis=1)

    for i in range(num_classes):
        TP = np.sum((predicted_labels == i) & (labels == i))
        FP = np.sum((predicted_labels == i) & (labels != i))
        FN = np.sum((predicted_labels != i) & (labels == i))
        TN = np.sum((predicted_labels != i) & (labels != i))

        confusion_matrix[i, 0] = TP
        confusion_matrix[i, 1] = FP
        confusion_matrix[i, 2] = FN
        confusion_matrix[i, 3] = TN

    return confusion_matrix


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.ldr_train_houyan = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=False)
        self.lr = args.lr
        self.linear_lr = args.lr
        self.lr_decay = args.lr_decay
        self.X = np.array([])
        self.train_labels_list = np.array([])
        self.pi = []
        self.flag = False

    def get_loss(self):
        if self.args.loss_type == 'CE':
            return nn.CrossEntropyLoss()
        elif self.args.loss_type == 'focal':
            return FocalLoss(gamma=1, num_class=self.args.num_classes).cuda(self.args.gpu)

    def train(self, net, linear_layer):
        net.train()
        linear_layer.train()

        # train and update
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args.momentum, weight_decay=0.00001)
        linear_optimizer = torch.optim.SGD(linear_layer.parameters(), lr=self.linear_lr, momentum=self.args.momentum, weight_decay=0.00001)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)
        linear_layer_scheduler = torch.optim.lr_scheduler.StepLR(linear_optimizer, step_size=1, gamma=self.lr_decay)
        criterion = self.get_loss()

        epoch_loss = []
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                linear_layer.zero_grad()
                feat = net(images)
                logits = linear_layer(feat)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                linear_optimizer.step()
                if self.args.verbose and batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                        100. * batch_idx / len(self.ldr_train), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            scheduler.step()
            linear_layer_scheduler.step()
        self.lr = scheduler.get_last_lr()[0]
        self.linear_lr = linear_layer_scheduler.get_last_lr()[0]
        return net.state_dict(), linear_layer.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def confusionMatrix(self, net, classifier, num_classes, pi, temp_class, h):
        if not self.flag:
            self.flag = True
            self.pi = pi
            net.eval()
            classifier.eval()

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(self.ldr_train_houyan):
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    feat = net(images)
                    log_probs = classifier(feat)
                    outputs = softmax(log_probs)
                    if self.X.size == 0:
                        self.X = outputs.cpu().numpy()
                        self.train_labels_list = labels.cpu().numpy()
                    else:
                        self.X = np.concatenate((self.X, outputs.cpu().numpy()))
                        self.train_labels_list = np.concatenate((self.train_labels_list, labels.cpu().numpy()))

        tmp_X = np.copy(self.X)

        pi_ = pi[temp_class] + h
        factor = (pi_ * self.X[:, temp_class]) / (h * self.X[:, temp_class] + pi[temp_class])
        tmp_X[:, temp_class] = factor
        for col in range(num_classes):
            if col != temp_class:
                tmp_X[:, col] = (pi[temp_class] * self.X[:, col]) / (h * self.X[:, temp_class] + pi[temp_class])

        confusion_matrix = calculate_confusion_matrix_elements(tmp_X, self.train_labels_list, num_classes)

        return confusion_matrix

    def updateX(self, num_classes, tmp_pi):
        for i in range(num_classes):
            factor = (tmp_pi[i] * self.X[:, i]) / ((tmp_pi[i] - self.pi[i]) * self.X[:, i] + self.pi[i])
            self.X[:, i] = factor

        confusion_matrix = calculate_confusion_matrix_elements(self.X, self.train_labels_list, num_classes)

        self.pi = tmp_pi

        return confusion_matrix
