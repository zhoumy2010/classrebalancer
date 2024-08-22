import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support


def test_img(net_g, classifier_g, datatest, best_F1, iter, args):
    net_g.eval()
    classifier_g.eval()

    all_targets = []
    all_preds = []
    test_loss = 0
    correct = 0

    data_loader = DataLoader(datatest, batch_size=args.bs)
    l = len(data_loader)

    for idx, (data, target) in enumerate(data_loader):
        if torch.cuda.is_available() and args.gpu != -1:
            data, target = data.cuda(args.device), target.cuda(args.device)
        else:
            data, target = data.cpu(), target.cpu()

        feat = net_g(data)
        log_probs = classifier_g(feat)

        test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()

        y_pred = log_probs.data.max(1, keepdim=True)[1]
        correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

        all_targets.extend(target.cpu().numpy())
        all_preds.extend(y_pred.cpu().numpy().flatten())

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')

    if args.verbose:
        print(f'\nTest set: Average loss: {test_loss:.4f} \nAccuracy: {correct}/{len(data_loader.dataset)} ({accuracy:.2f}%)\n')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-score: {f1:.4f}')

    if f1 > best_F1:
        best_F1 = f1

        if iter > 100:
            print('Saving..')
            state = {
                'net': net_g.state_dict(),
                'classifier': classifier_g.state_dict(),
                'f1': f1,
                'epoch': iter,
            }
            rootpath = f'checkpoint_{args.dataset}_{args.model}_{args.loss_type}loss_IF{args.imb_factor}_alpha{args.alpha_dirichlet}'
            if not os.path.isdir(rootpath):
                os.mkdir(rootpath)
            torch.save(state, rootpath + f'/best_model.pth')

    return accuracy, test_loss, precision, recall, f1, best_F1
