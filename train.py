import argparse
import torch
import torch.optim as optim
from model import AMNet
from copy import deepcopy
from config import *
import pickle
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(data, model, criterion, optimizer, label, beta=.5):
    anomaly, normal = label
    idx_train = data.train_mask
    model.train()
    optimizer.zero_grad()
    output, bias_loss = model(data.x, data.edge_index, label=(data.train_mask & anomaly, data.train_mask & normal))
    loss_train = criterion(output[idx_train], data.y[idx_train]) + bias_loss * beta
    loss_train.backward()
    optimizer.step()
    return loss_train.item()


def main(args, exp_num=0):

     data = pickle.load(open('dataset/{}.dat'.format(args.dataset), 'rb'))
     data = data.to(device)

     net = AMNet(in_channels=data.x.shape[1], hid_channels=params_config['hidden_channels'], num_class=2,
                   K=params_config['M'], filter_num=params_config['K'])
     net.to(device)

     optimizer = optim.Adam([
          dict(params=net.filters.parameters(), lr=params_config['lr_f']),
          dict(params=net.lin, lr=params_config['lr'], weight_decay=params_config['weight_decay']),
          dict(params=net.attn, lr=params_config['lr'], weight_decay=params_config['weight_decay'])]

     )

     weights = torch.Tensor([1., 1.])
     criterion = torch.nn.CrossEntropyLoss(weight=weights.to(device))
     anomaly = (data.y == 1)
     normal = (data.y == 0)
     label = (anomaly, normal)

     c = 0
     auc_pr_best = 0
     auc_roc_best = 0
     auc_roc_test_epoch = 0
     auc_pr_test_epoch = 0
     best_net = None

     for epoch in range(params_config['epochs']):
          loss = train(data, net, criterion, optimizer, label, beta=params_config['beta'])
          auc_roc_val, auc_pr_val = net.evaluating(data.x, data.y, data.edge_index, data.val_mask)
          if (epoch + 1) % args.eval_interval == 0 or epoch == 0:
               print('Epoch:{:04d}\tloss:{:.4f}\tVal AUC-ROC:{:.4f}\tVal AUC-PR:{:.4f}'
                     '\tBest AUC-ROC:{:.4f}\tBest AUC-PR:{:.4f}'
                           .format(epoch + 1, loss, auc_roc_val, auc_pr_val, auc_roc_test_epoch, auc_pr_test_epoch))

          if auc_pr_val >= auc_pr_best:
               auc_pr_best = auc_pr_val
               auc_roc_best = auc_roc_val
               auc_roc_test_epoch, auc_pr_test_epoch = net.evaluating(data.x, data.y, data.edge_index, data.test_mask)
               best_net = deepcopy(net)
               c = 0
          else:
               c += 1
          if c == params_config['patience']:
               break

     auc_roc_test_exp, auc_pr_test_exp = best_net.evaluating(data.x, data.y, data.edge_index, data.test_mask)
     return auc_roc_test_exp, auc_pr_test_exp



if __name__ == '__main__':

     # The dataset-dependent arguments are hard-coded in config.py
     parser = argparse.ArgumentParser()

     parser.add_argument('--dataset', default='elliptic', help='Dataset [yelp, elliptic, FinV, Telecom]')
     parser.add_argument('--exp_num', type=int, default=10, help='Default Experiment Number')
     parser.add_argument('--eval_interval', type=int, default=100)

     args = parser.parse_args()

     params_config = dataset_config[args.dataset]
     auc_roc_list = []
     auc_pr_list = []

     for i in range(args.exp_num):
          auc_roc_test, auc_pr_test = main(args, exp_num=i)
          auc_roc_list.append(auc_roc_test)
          auc_pr_list.append(auc_pr_test)


     print("AUC ROC Mean:{:.5f}\tStd:{:.5f}\tAUC PR Mean:{:.5f}\tStd:{:.5f}".format(np.mean(auc_roc_list),
                                                                                    np.std(auc_roc_list),
                                                                                    np.mean(auc_pr_list),
                                                                                    np.std(auc_pr_list)))


